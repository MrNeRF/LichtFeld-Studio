I'll explore the densification/strategy layer read-only: strategies under `src/training/strategies/`, tensor ops they use, optimizer resize paths, and `TENSOR_LIB_FINDINGS.md`.Reading all strategy sources and the optimizer grow/prune paths next.Default is MRNF. Reading densification event paths in each strategy and the optimizer resize APIs.Reading densification cores and optimizer resize implementations.# Densification / Strategy Layer — Memory Spikes & Slow Patterns

**Scope:** `src/training/strategies/` (mcmc, mrnf, improved_gs_plus, strategy_utils), Adam optimizer grow/prune, interaction with capacity-based tensors.  
**Default strategy:** `mrnf` (`parameters.hpp:159`, factory at `strategy_factory.cpp:37-39`).  
**Method:** static read-only analysis only. Peak-VRAM multiples are **derived from code structure** (old+new coexistence), not measured.

**Relevant tensor-lib bug classes** (`TENSOR_LIB_FINDINGS.md:84-107`): strided `cat`/`stack`, `masked_select`/`masked_fill_`, `index_select`/`gather`/`scatter_`/`index_put_`, `multinomial`, plus Theme D stream races.

---

## 0. Architecture overview

| Strategy | Role | Growth model | Prune model |
|---|---|---|---|
| **MRNF** (default) | Soft free-slots + LAS split | Fill free, then `add_new_params` append | Soft: free mask + zero quat; hard compact only for max_cap overflow / `remove_gaussians` |
| **MCMC** | Relocate dead + ~5% grow | `add_new_params_gather` | Soft-delete (deleted mask + zero rot); no free-slot reuse |
| **IGS+** | Budget schedule + multinomial LAS | Fill free, then `append_zeros` + `index_put_` | Soft free-slot prune; opacity threshold |

Shared capacity philosophy (comments + code):

- Pre-allocate param + optimizer buffers at `max_cap` to avoid realloc double-buffering (`strategy_utils.cpp:47-54`, `77-82`; MRNF `mrnf.cpp:343-381`; MCMC `mcmc.cpp:830-880`).
- Adam growth factor `1.5` (`adam_optimizer.hpp:41`; `strategy_utils.cpp:81`; fallback `DEFAULT_GROWTH_MULTIPLIER` at `adam_optimizer.cpp:27,937-946`).
- `Tensor::reserve` **always** allocates a new buffer and copies while old storage lives (`tensor.cpp:3289-3435`) → peak ≥ 2× that tensor during reserve.

---

## 1. What happens at a densification event

### 1.1 MRNF (default) — `refine` / `grow_and_split`

**Trigger:** `is_refining` → `iter ∈ (start_refine, stop_refine)` and `iter % refine_every == 0` (`mrnf.cpp:586-590`). Defaults: every 200 steps, start 0, stop 28500, `max_cap=5e6` (`parameters.cpp:449-455`).

**Per-step (before refine, while `iter < stop_refine`):**

1. Ensure densification info shape `[2,N]` (`mrnf.cpp:441-449`, called from `post_backward:547`).
2. Elementwise max into `_refine_weight_max`, add into `_vis_count` from densification rows (`558-570`).
3. Zero densification info (`573`).
4. Optional noise injection if bounds valid (`575-577`).

**Refine event pipeline** (`refine` at `592-678`):

| Phase | Actions | Tensors built / rewritten |
|---|---|---|
| Bounds | Every 5 refine windows: `compute_bounds` | May `index_select` active means (`1305`) |
| Soft prune | Opacity/rot/scale/bounds masks → `nonzero` → `mark_as_free` + zero rotations + reset Adam scales | Masks O(N); no param size change (`615-660`) |
| Grow/split | Gumbel-topk replace + growth; LAS children; free-slot fill; optional append | Full child buffers of size K (`797-906`) |
| Cap | `enforce_max_cap` → possibly `compact_splats` | Full rebuild of all params + Adam state (`1099-1143`, `923-1055`) |
| Decay | In-place opacity/scale decay kernel | No size change (`1080-1097`) |
| Reset tracking | Zero/reset `_refine_weight_max`, `_vis_count`, densification info | Uses reserved capacity when possible (`46-103`, `672-677`) |

#### Soft prune (no size change)

- Criteria: raw opacity `< logit(1/255)`, near-zero rotation, log-scale min, optional bounds outliers (`615-635`).
- Restrict to active free-mask slice (`637-639`).
- `mark_as_free` + `deleted` mask + `rotation_raw.index_put_(zeros)` (`646-652`).
- Optimizer: **zero scales only** (quantized moments dequant to 0) via `reset_optimizer_state_at_indices` (`114-162`, called `655-660`).

#### Grow/split

1. Budget: `desired_total = round(count(refine_candidates) * grow_fraction)` (`698-700`); `grow_fraction` default `0.07` (`parameters.hpp:218`).
2. Cap budget: `max_cap - active_count` (`701-703`).
3. **Replace** up to `pruned_count` parents via Gumbel-topk on `opacity * (vis>0)` (`719-744`).
4. **Growth** (if `iter < grow_until_iter`): Gumbel-topk on refine candidates × refine weights, with replace mask zeroed (`747-772`).
5. Merge indices: `Tensor::cat({replace_inds, growth_inds}, 0)` (`775-777`) — only small index vectors.
6. Allocate **full child attribute tensors** of length K (`797-806`):
   - means `[K,3]`, log_scales `[K,3]`, raw_opacities `[K]`, rotations `[K,4]`, sh0 `[K,1,3]`, optional shN `[K, rest, 3]`.
7. LAS kernel writes children; parent means/scales/rot/opacity mutated in-place; shN parents unchanged (`810-835`).
8. Parent Adam state **zeroed** at split indices (`837-842`).
9. **Reuse free slots first** (`fill_free_slots_with_data`, `845-855`, `1210-1280`):
   - `index_put_` on means/rot/scale/sh0/opacity; swizzled `shN_swizzled_scatter_linear`; zero Adam again at filled indices.
10. **Append remainder** (`857-908`):
    - `add_new_params` for Means/Sh0/Scaling/Rotation/Opacity (`879-904`).
    - shN: `append_zeros` + `shN_swizzled_gather_from_linear` + `extend_state_for_new_params` (`882-900`).

#### Peak transient VRAM (MRNF refine, typical path with `max_cap` reserved)

Let \(S\) = steady-state param+optimizer footprint at live size \(N\).

| Scenario | Old+new coexist? | Peak multiple of steady (structural) |
|---|---|---|
| Soft prune only | No (in-place) | \(\approx 1.0S + O(N)\) masks + prune index temps |
| Grow via free-slot reuse only | Child buffers of size K only | \(\approx 1.0S + C(K)\); \(C(K) \ll S\) for modest K |
| Grow via append, capacity hits | In-place `append_zeros` / `append_gather` | \(\approx 1.0S + C(K) + o(N)\) |
| Grow via append, **slow path** `cat` | Yes, per-param then per-state | \(\approx 1.0S + \max_t(\text{size}(t)+\text{size}(t)+n_{\text{new}})\) sequentially → often **~1.5–2× largest buffer**, not full 2× all at once |
| `compact_splats` (max_cap overflow / hard remove) | Yes, **per tensor**: old + `index_select` result + possible `reserve(max_cap)` | **Worst case ~3× one large tensor** (old N + compact + reserve); sequential over ~6 params × ~4 state tensors; **shN is largest** |

Evidence for 3× path in compact:

```932:938:src/training/strategies/mrnf.cpp
        auto compact = [&](Tensor& t) {
            ...
            auto compacted = t.index_select(0, valid_indices).contiguous();
            if (cap > 0)
                compacted.reserve(cap);
            t = std::move(compacted);
```

`reserve` allocates a new larger buffer while the current compacted buffer still holds data (`tensor.cpp:3356-3435`), and original `t` is still alive until assignment.

MRNF does **not** call `trim_memory_pool` after refine (unlike MCMC `mcmc.cpp:730` and IGS+ `improved_gs_plus.cpp:700`) — fragmentation risk after any realloc path.

---

### 1.2 MCMC — `relocate_gs` + `add_new_gs`

**Trigger:** same refine schedule (`mcmc.cpp:906-910`). Default `max_cap` base `1e6` unless overridden (`parameters.hpp:147`).

#### `relocate_gs` (size unchanged)

1. Dead mask: opacity OR near-zero rotation (`203-209`).
2. Alive indices via `logical_not` + `nonzero` (`217-219`).
3. `index_select` sampling weights on alive (`233`) — **potential strided-op surface** if weights weren’t contiguous (they are newly built).
4. Fused multinomial+gather kernel (`248-259`).
5. Ratio count: `index_add_` + `index_select` (`267-269`).
6. Relocation kernel → update parent opacity/scale via CUDA kernel (capacity-preserving; avoids `index_put_`) (`316-324`).
7. Copy params dead←sampled via kernel; shN via gather-to-linear + scatter-linear staging buffer `[n_dead, rest, 3]` (`328-365`).
8. Optimizer: `relocate_params_at_indices_gpu` zeros moments/scales at sampled **and** dead indices (`121-136`, `1170-1229`).

Peak: \(\approx 1.0S + O(\text{n_dead})\) staging; no full param rebuild.

#### `add_new_gs` (grows ~5%)

```396:398:src/training/strategies/mcmc.cpp
        const int n_target = std::min(_params->max_cap, static_cast<int>(1.05f * current_n));
        const size_t n_new = std::max(0, n_target - current_n);
```

1. Fused sample-all kernel (`438-447`).
2. Ratio count + clamp (`450-463`).
3. Update original opacity/scale in-place (`501-510`).
4. **`add_new_params_gather` for all 6 param types** (`517-522`) — copies parent rows into reserved tail; optimizer new rows zeroed via `extend_state_for_new_params` (except shN moments which gather-parent on the shN-specific path).

Peak with reserved capacity: \(\approx 1.05S +\) small workspaces.  
Peak without capacity (slow path): sequential `cat`/`realloc` → per-tensor old+new (~2× that tensor).

Post-event: `trim_memory_pool` (`730`); densification_info reallocated zeros `[2,n]` (`750-752`); error score max may `cat` zeros for new rows (`736-740`).

---

### 1.3 Improved GS+ — `densify_with_score` / `LAS_densify`

**Trigger:** refine schedule (`is_refining` + precomputed edge scores in `pre_step`) (`620-671`). Defaults: every 500 iters, stop 15000, `max_cap=4e6` (`parameters.cpp:471-486`).

**Event:**

1. Candidate selection: sort active errors descending when candidate pool limited (`365-368`); `masked_fill` scores (`372`); multinomial sample (`417`).
2. Median-normalize edge/error via `masked_select` + **full sort of positive values** (`93-115`, `354-356`).
3. Allocate second-split child buffers of size `budget_for_alloc` (`433-443`).
4. LAS + shN gather; zero parent Adam (`475-519`).
5. Free-slot fill (`522-524`); append remainder via **`append_zeros` + `index_put_`** on each param (not `add_new_params`) + `extend_state_for_new_params` (`533-593`).
6. Then `opacity_prune` soft-free (`672`, `845-856`).
7. `trim_memory_pool` (`700`); rebuild densification info (`702-706`).

**Capacity caveat:** `append_zeros` **requires** `capacity > 0` (`tensor_masking_ops.cpp:2501-2505`). IGS+ relies on `initialize_gaussians(..., max_cap)` (`improved_gs_plus.cpp:239`). Without `max_cap`, append path is structurally broken (throws).

Peak: same shape as MRNF free-fill path; no hard compact in the normal densify loop. Soft prune only.

---

## 2. Optimizer state handling on grow / prune

### 2.1 Design (quantized Adam)

State per param (`adam_optimizer.hpp:45-54`):

- `exp_avg`, `exp_avg_sq`: **uint8** quantized moments  
- `exp_avg_scale`, `exp_avg_sq_scale`: **fp32** per-primitive scales  
- Comment (`adam_optimizer.hpp:194-195`): **zero scale ⇒ dequantized moment is zero** regardless of byte contents.

Initial capacity: `create_optimizer` sets `initial_capacity = max_cap`, `growth_factor = 1.5` (`strategy_utils.cpp:79-82`); `allocate_gradients(capacity)` (`adam_optimizer.cpp:167-214`) uses `zeros_direct` for reserved buffers (`305-327`).

### 2.2 Grow paths

| API | Used by | Param growth | Optimizer state |
|---|---|---|---|
| `add_new_params` | MRNF append | Fast: `append_zeros` + `copy_from`; Slow: **`Tensor::cat`** (`979-1010`) | `extend_state_for_new_params`: **zeros** new rows (fast append_zeros / slow full realloc) (`818-935`) |
| `add_new_params_gather` | MCMC | Fast: `append_gather` (in-place gather into reserved tail) (`1145-1147`); shN swizzle-aware (`1013-1126`) | Non-shN: **zeros** via `extend_state_for_new_params`. **shN**: **copies parent moment bytes + gathers parent scales** (`1085-1090`) |
| `extend_state_for_new_params` alone | IGS+ append; MRNF shN | N/A (params grown separately) | Same zero-init of new rows (`818-935`) |
| `extend_state_by_gather` | (available; gather parent moments) | — | Fast: `append_gather`; Slow: **`cat` + `index_select`** (`739-816`) |

**Slow path flags:** both extend functions **set `state.capacity = 0`** after realloc (`814-815`, `933-934`), so subsequent grows also miss reserved-capacity fast path until `reserve_capacity` is called again.

### 2.3 Prune / relocate / free-slot reuse

| Operation | Optimizer behavior |
|---|---|
| Soft prune (MRNF/IGS+/MCMC remove) | **Selective zero** of scales (+ grad rows if present) via `index_put_` / swizzled zero — **not** rebuilt (`mrnf.cpp:114-162`, `improved_gs_plus.cpp:900-945`, `mcmc.cpp:75-110`) |
| MCMC relocate | GPU zero of quantized rows at **source and dest** indices (`relocate_params_at_indices_gpu`, `1170-1229`) |
| Free-slot fill (MRNF/IGS+) | Zero state at target indices after writing children (`mrnf.cpp:1269-1274`, `improved_gs_plus.cpp:1027-1050`) |
| Split parents | Zero state at parent indices before children written (`mrnf.cpp:837-842`, `improved_gs_plus.cpp:514-519`) |
| Hard compact (`compact_splats`) | **Full rebuild**: `index_select` (or swizzled gather) of every moment/scale tensor; grad reallocated zeros (`mrnf.cpp:993-1024`) — **copies kept rows**, not zero-from-scratch |

**Not rebuilt from scratch** on soft prune/grow. **Copied** on compact (kept rows only). **Zero-initialized selectively** on free-fill, split parents, relocate, and new append rows (except MCMC shN moment gather).

---

## 3. Masking / gather / scatter / index call sites vs TENSOR_LIB_FINDINGS

Legend: **Risk** = could hit Theme A (strided/non-contiguous) or related bugs if operands are non-contiguous views. Contiguous newly allocated tensors are lower risk but still use the same kernels.

### 3.1 High-interest densification sites

| File:line | Op | Context | Risk note |
|---|---|---|---|
| `mrnf.cpp:777` | `cat` | Merge replace/growth index vectors | Small 1D Int64; low |
| `mrnf.cpp:879-904` | `add_new_params` → may `cat` | Param append slow path | **Critical if capacity miss**: `cat` ignores strides (`TENSOR_LIB_FINDINGS.md:89-91`); inputs should be contiguous slices of child buffers |
| `mrnf.cpp:862-863` | `reserve` + `append_zeros` | free_mask growth | reserve = alloc+copy |
| `mrnf.cpp:935` | `index_select(0, …).contiguous()` | compact params | **Theme A index_select**; inputs are primary contiguous row tensors — usually OK |
| `mrnf.cpp:1029` | `info.index_select(1, valid_indices)` | densification_info `[2,N]` along **dim 1** | **Elevated**: dim-1 select on `[2,N]`; if kernel assumes dim-0/contiguous layout, wrong results (`TENSOR_LIB_FINDINGS.md:95-97`) |
| `mrnf.cpp:1032-1051` | `index_select` | deleted, free, refine, vis, edge scores | Contiguous 1D sources typically |
| `mrnf.cpp:1041` | `cat({compacted_free, tail})` | free_mask restore to max_cap | 1D bool |
| `mrnf.cpp:652,743,1207,1242-1251,1277` | `index_put_` | prune rot, masks, free-slot writes | **Theme A index_put_** if target is a **view**; targets are full owning params — usually OK. **Do not** `index_put_` into slices of params |
| `mrnf.cpp:1259-1266` | `shN_swizzled_scatter_linear` | free fill shN | Custom kernel (not tensor scatter_); separate code path |
| `mrnf.cpp:828-835,891-897` | swizzled gather | child/append shN | Custom |
| `mrnf.cpp:311-318` | `masked_fill_`, `masked_select`, `sort` | median normalize (edge scores) | **Theme A masked_***; tensors are clones of densify scores |
| `mrnf.cpp:762` | `masked_fill` | zero growth weights under replace mask | bool mask + float weights |
| `mrnf.cpp:1131` | `masked_fill` | zero frozen opacities for topk | |
| `mrnf.cpp:737-739,769-771,1136-1138` | Gumbel topk (custom) | sampling | Not multinomial; avoids Theme A multinomial |
| `mrnf.cpp:1199-1200,1230-1231,1298-1305` | `index_select` chains | frozen filtering / bounds | Intermediate `nonzero().squeeze` indices |
| `mcmc.cpp:233,269,457` | `index_select` | weights/ratios | Contiguous sources intended |
| `mcmc.cpp:247-259,438-447` | fused multinomial kernels | sampling | Custom CUDA; bypasses `Tensor::multinomial` |
| `mcmc.cpp:116-118` | `Tensor::multinomial` | helper only | Theme A multinomial if used on strided weights |
| `mcmc.cpp:59,95-109,806,819` | `index_put_` | deleted mask, zero Adam, zero rot | Owning tensors |
| `mcmc.cpp:352-364` | swizzled gather/scatter | relocate shN | Custom + staging |
| `mcmc.cpp:738-740` | `cat` | error_score_max growth | 1D float |
| `improved_gs_plus.cpp:417` | **`Tensor::multinomial`** | LAS parent sample | **Theme A multinomial** (`TENSOR_LIB_FINDINGS.md:104-107`); scores from arithmetic — likely contiguous but **not** forced with `.contiguous()` |
| `improved_gs_plus.cpp:47,100,366,833` | `sort` | percentiles / candidates / opacity quantile | Full or filtered N-length sorts |
| `improved_gs_plus.cpp:94-95` | `masked_fill_`, `masked_select` | median normalize | Same as MRNF |
| `improved_gs_plus.cpp:343,867,978` | `masked_select` on indices | frozen filter | Bool mask on Int64 indices |
| `improved_gs_plus.cpp:372,377` | `masked_fill` | zero non-candidates | |
| `improved_gs_plus.cpp:487-510,555-569,912-936,990-998,1027-1050` | `index_put_` | Adam zero, append writes, free fill | Same owning-tensor caveat |
| `improved_gs_plus.cpp:466-472,578-584` | swizzled gather | shN | Custom |
| `strategy_utils.cpp:163-176,202-204` | `masked_fill` / `masked_fill_` / `where` | frozen + crop damping | Scores usually contiguous 1D; crop mask may be **sliced** (`199-200`) → **strided mask risk** for `masked_fill`/`where` |
| `adam_optimizer.cpp:809-812,1117-1120` | `cat` + `index_select` | slow-path state grow | **Critical capacity-miss path** |
| `adam_optimizer.cpp:1008` | `cat` | param slow path | Same |
| `adam_optimizer.cpp:789-792,1146` | `append_gather` | uses `launch_index_select` into tail (`tensor_masking_ops.cpp:2436-2442`) | Assumes contiguous source; params are row-major |

### 3.2 Views that appear in densification

| Site | View | Implication |
|---|---|---|
| `_free_mask.slice(0,0,n)` | Prefix of free_mask (`mrnf.cpp:638,689,1157,…`) | 1D prefix is contiguous; OK for logical ops |
| `child_*.slice(0, append_start, K)` | Row range for append (`mrnf.cpp:866-874`) | Contiguous if children are contiguous |
| `crop_mask.slice(0,0,scores.numel())` | `strategy_utils.cpp:199-200` | If scores shorter than mask, **sliced mask** into `masked_fill`/`where` → Theme A risk |
| `info.ptr + n` densification row | Raw pointer into `[2,N]` (`mrnf.cpp:558`, `mcmc.cpp:705`) | Kernel uses flat row pointer; bypasses tensor strides |
| `opacity.squeeze(-1)` | `get_opacity` → `sigmoid().squeeze` (`splat_data.cpp:624-625`) | New tensor from sigmoid; usually contiguous |

---

## 4. Allocation pattern: headroom vs exact-fit

### 4.1 With `max_cap` (production defaults)

| Layer | Policy | Evidence |
|---|---|---|
| Splat params | Pre-reserve `max_cap` via `zeros_direct` / `ensure_capacity_direct` only if `capacity < max_cap` | MRNF `343-381`, MCMC `830-874`, IGS+ via `initialize_gaussians` `strategy_utils.cpp:49-53` |
| Adam state | `initial_capacity = max_cap`, `zeros_direct` | `strategy_utils.cpp:79-82`, `adam_optimizer.cpp:167-214` |
| Free mask | Sized to `max_cap` at init | MRNF `394-396`, IGS+ `254-256` |
| Tracking (`_refine_weight_max`, `_vis_count`) | `zeros_direct` with `tracking_capacity=max_cap` | `reset_vector_buffer` `46-103`, `400-402` |
| Growth beyond capacity | `growth_factor=1.5` when `compute_new_capacity` used | `adam_optimizer.cpp:937-946` — **but slow path sets capacity=0**, so next grow may exact-fit again |
| Free-slot reuse | Amortizes densify without growing `N` | Soft free + fill |

**Amortized headroom is the design goal.** Fast path is in-place logical size growth inside reserved capacity (`append_zeros` / `append_gather`).

### 4.2 Exact-fit / churn sites

| Site | Behavior |
|---|---|
| `extend_state_*` slow path | Alloc exact `new_size`, `capacity=0` (`814-815`, `933-934`) |
| `add_new_params` slow path | `cat` exact size (`1008-1009`) |
| `compact_splats` | `index_select` to exact kept size, then `reserve(max_cap)` regrows headroom — **double allocation per tensor** (`935-938`) |
| `ensure_densification_info_shape` | Full reallocate zeros if shape mismatch (`mrnf.cpp:441-449`) — can churn `[2,N]` every size change if not careful; MRNF resets after refine via same helper |
| IGS+/MCMC densify end | **Replace** densification_info with new zeros (`improved_gs_plus.cpp:702-704`, `mcmc.cpp:750-752`) instead of in-place grow |
| `reset_vector_buffer` when size shrinks | `make_fresh()` full realloc (`102`) |

### 4.3 Soft prune vs compaction

- Soft free-slots: **no** param realloc; holes remain until filled. Total `size()` can exceed active count.  
- Hard compact only when `size > max_cap` (MRNF `1099-1108`) or external `remove_gaussians` (`1372-1386`).  
- MCMC never reuses slots; only grows (until max_cap) and soft-deletes.

---

## 5. Sort / permutation / scan work

### 5.1 Per densification event

| Strategy | Work | Complexity | Site |
|---|---|---|---|
| MRNF | Gumbel-topk for replace | O(N + k log …) kernel | `737-739` |
| MRNF | Gumbel-topk for growth | same | `769-771` |
| MRNF | Optional Gumbel-topk keep for max_cap | O(N) | `1136-1138` |
| MRNF | `nonzero` on prune mask, free mask, active mask | O(N) + compact indices | multiple |
| MRNF | `count_nonzero` on weight tensors | O(N) | `733,765` |
| MRNF | Bounds: percentile kernel on active means | O(active) every 5 refines | `1315-1319` |
| MRNF | Edge precompute: canny + normalize (**sort positives**) | O(P log P) positives | `310-318`, `pre_step` |
| MCMC | Fused multinomial | O(N) | `248,438` |
| MCMC | Dead/alive `nonzero` | O(N) | `208-219` |
| MCMC | Ratio `index_add_` | O(n_new) | `268,456` |
| IGS+ | `sort` active errors for candidate threshold | **O(A log A)** | `366-367` |
| IGS+ | `multinomial` without replacement | O(N) + sample | `417` |
| IGS+ | Dual median normalize = 2× (`masked_select` + **sort**) | O(P log P) each | `354-356` |
| IGS+ | `opacity_prune` threshold scan | O(N) | `849-856` |
| IGS+ | `prune_post_reset` (exists, unused in main path?) | full opacity **sort** | `829-842` |

### 5.2 Per training step (while refining window active)

| Strategy | Work | Site |
|---|---|---|
| All | Accumulate densification max/add from backward info | MRNF `557-573`, MCMC `698-713`, IGS+ `650-663` |
| MRNF | Noise injection every step if bounds valid | `575-577`, `1057-1078` |
| MCMC | Noise injection **every** `post_backward` | `755-756`, `641-678` |
| MRNF | Edge sample accumulation in `pre_step` (subset of iters) | `409-438` |
| IGS+ | Edge score **precompute multi-view Canny** only on refine iters in `pre_step` | `620-629`, `266-313` — **expensive** (load images, canny, rasterize per view) |

No per-step full opacity sort in the default MRNF path.

---

## 6. Ranked concrete improvements

### (a) Cap peak VRAM at densification

| Rank | Improvement | Why | Anchors |
|---|---|---|---|
| **1** | **Guarantee reserved capacity path always wins** (params + all Adam state including scales) before any densify append; re-`reserve_capacity` after any slow path that zeroes `capacity` | Eliminates sequential 2× `cat`/realloc; slow path is the documented double-buffer failure mode | `adam_optimizer.cpp:814-815,933-934,1003-1010,1080-1122`; init `strategy_utils.cpp:77-82` |
| **2** | **Avoid `index_select` + `reserve(max_cap)` compact pattern** — allocate once at `max_cap` (or kept+headroom) and gather into it; or compact into pre-reserved buffers | Current compact can peak ~3× per large tensor | `mrnf.cpp:932-938`; `tensor.cpp:3289+` |
| **3** | **Soft-cap by free-slot-only growth when near max_cap** (never grow `size()` above max_cap; only fill free) | Makes `enforce_max_cap` / compact dead code on the hot path | `grow_and_split` `857-908`, `enforce_max_cap` `1099-1143` |
| **4** | **Stream child buffers through a reusable densify workspace** (means/rot/scale/sh0/shN/opacity) sized to worst-case K | Cuts allocator pressure; K≪N but shN children still matter at high SH | `mrnf.cpp:797-806`; IGS+ `433-443` |
| **5** | **MRNF: call `trim_memory_pool` after refine** (parity with MCMC/IGS+) | Releases pool fragmentation after any incidental realloc | MCMC `730`, IGS+ `700`; MRNF missing |
| **6** | **In-place densification_info / error buffers** — grow with `append_zeros` into reserved capacity instead of full zeros reallocate each event | Avoids `[2,N]` realloc spikes | MCMC `750-752`, IGS+ `702-704`, MRNF `441-449` |
| **7** | **Compact optimizer state without rebuilding grad** (grad is transient; fused path often unused) | `compact_splats` reallocates grad for every param (`1013-1024`) | |
| **8** | **Defer shN staging** in MCMC relocate — write swizzle-to-swizzle without full linear `[n_dead,rest,3]` when possible | Staging is extra peak proportional to dead×SH | `mcmc.cpp:350-364` |

### (b) Speed up densification events

| Rank | Improvement | Why | Anchors |
|---|---|---|---|
| **1** | **Fuse free-slot write path**: one kernel writing means/rot/scale/sh0/opacity (+ optional shN scatter) instead of 5–6 `index_put_` launches | `index_put_` is heavy and Theme-A-sensitive | `mrnf.cpp:1242-1251`; IGS+ `990-998` |
| **2** | **Eliminate redundant Adam resets** (split zeros parents, fill zeros targets again; prune zeros then fill zeros) — single zero at final write sites | Double zero on overlapping semantics | `mrnf.cpp:837-842` then `1269-1274` |
| **3** | **Replace IGS+ `Tensor::multinomial` with fused sample kernel** (like MCMC) + force contiguous weights | Correctness (Theme A) + speed | `improved_gs_plus.cpp:417` vs `mcmc.cpp:438-447` |
| **4** | **Replace median-via-full-sort** with approx median / selection / histogram for edge/error normalize | `masked_select` + `sort` of all positives is O(P log P) per densify (×2 for IGS+) | `mrnf.cpp:310-318`; `improved_gs_plus.cpp:93-115,354-356` |
| **5** | **IGS+: avoid full `sort` of active errors** — partial select / bucket for top-`candidate_budget` threshold | `sort(0,true)` on all active (`366-367`) |
| **6** | **Batch `nonzero` / prune mask construction** into one custom kernel (opacity+rot+scale+bounds) | Many bool tensor ops + host `sum().item()` sync (`643`) | `mrnf.cpp:615-643` |
| **7** | **Remove device↔host syncs on refine** (`sum().item()`, `count_nonzero`, `item_as` for thresholds) | Pipeline stalls densify | e.g. `mrnf.cpp:643,699,733,765`; IGS+ `367` |
| **8** | **Reuse ratio/ones workspaces** (MCMC already has `_ones_int32`; keep contiguous) | Avoid clone of N Int32 per event | `mcmc.cpp:267-268,455-456` |
| **9** | **MRNF edge path**: cache canny workspace; skip edge when weight unused | pre_step + canny is off-refine-path cost | `409-438`, `257-308` |
| **10** | **Prefer `add_new_params_gather` or append-from-staging kernel for MRNF append** instead of slice + `add_new_params` + separate shN path | Unifies MCMC-fast path; fewer branches | MRNF `866-904` vs MCMC `517-522` |

---

## 7. Cross-cutting observations for optimization planning

1. **Default MRNF is already soft-delete oriented** — best VRAM path is “free slot fill only”; append + compact are the spike sources.  
2. **Capacity is correct when `max_cap` is set and nothing zeroes `state.capacity`** — the dangerous bug class is silent fallback to `cat`/`capacity=0` after one overflow.  
3. **Optimizer grow does not copy parent moments** except MCMC shN gather path; free-slot children get **fresh zero state** (good for correctness, cheap).  
4. **TENSOR_LIB_FINDINGS Theme A is most reachable** at: IGS+ `multinomial`, compact `index_select` (esp. densification_info dim 1), crop-damped `masked_fill` on **sliced** masks, and any slow-path `cat` of views. Contiguous param rows are the common case and partly explain “training looks fine.”  
5. **No measured VRAM numbers** in-repo for densify peaks; structural multiples above should be validated with `LFS_VRAM_SCOPE` / profiler around `refine`/`add_new_gs`/`LAS_densify`/`compact_splats`.

---

## 8. File index (primary)

| Path | Role |
|---|---|
| `src/training/strategies/mrnf.cpp` | Default densify/refine/compact |
| `src/training/strategies/mcmc.cpp` | Relocate + 5% gather-grow |
| `src/training/strategies/improved_gs_plus.cpp` | Budget LAS + multinomial |
| `src/training/strategies/strategy_utils.cpp` | Optimizer factory, frozen/crop masks, dead masks |
| `src/training/optimizer/adam_optimizer.cpp` | Capacity grow/prune state |
| `src/core/tensor/tensor_masking_ops.cpp` | `append_gather` / `append_zeros` |
| `src/core/tensor/tensor.cpp` | `reserve` / `zeros_direct` double-buffer |
| `TENSOR_LIB_FINDINGS.md` | Strided op bug classes |
| `src/core/include/core/parameters.hpp` + `parameters.cpp` | Defaults (`strategy=mrnf`, caps, schedules) |

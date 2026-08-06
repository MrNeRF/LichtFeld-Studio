I'll explore the tensor library and training consumers read-only, starting from known findings and the main tensor/training paths.Digging into dtype dispatch, in-place APIs, training workspaces, and copy materialization next.Checking a few remaining details on mutual workspace lifetime and GT storage.# Tensor Library Memory-Savings Levers (Beyond Allocator Work)

**Scope:** `src/core/tensor/` + training consumers (`src/training/`).  
**Method:** static read-only analysis. No runtime measurement.  
**Excluded:** already-planned allocator work (zero steady-state allocs, slab reclaim, pooled partials).  
**Size baseline:** 1080p = \(H{=}1080, W{=}1920\); \(P{=}2{,}073{,}600\) pixels; \(C{=}3\); \(N{=}1\).  
**Element counts:** full image \(C{=}3\): \(6{,}220{,}800\); map \(C{=}1\): \(2{,}073{,}600\).  
**Bytes:** fp32 \(C{=}3\) = **23.73 MiB**; fp16 \(C{=}3\) = **11.87 MiB**; fp32 \(C{=}1\) = **7.91 MiB**.  
**Splats baseline:** \(N_s{=}2{\times}10^6\).

---

## 1. Dtype Coverage: Float16 / BFloat16

### 1.1 Enum & dispatch surface

| Item | Status | Evidence |
|------|--------|----------|
| `DataType` members | **Float32, Float16, Int32, Int64, UInt8, Bool only** | `src/core/include/core/tensor_fwd.hpp:22–28` |
| **BFloat16** | **Does not exist** anywhere in the tensor dtype system | same enum; no `__nv_bfloat16` in tensor sources |
| Exhaustive `dispatch_dtype` | Includes `Float16` → `__half` | `src/core/tensor/internal/tensor_dtype_dispatch.hpp:21–41` |
| Element size | Float16 = 2 bytes | `tensor_fwd.hpp:31–39` |

### 1.2 What works for Float16 today

| Capability | Evidence | Notes |
|------------|----------|-------|
| Factory `full` / constants | `tensor_unified_ops.cpp:459–510, 547–549` | CUDA path builds CPU `__half` temp then H2D copy |
| Binary arith (same-shape + broadcast CUDA) | `tensor_exports.cu:66–67`; `tensor_broadcast_ops.cu:527–553`; `tensor_expr_impl.hpp:348–369` | Float16↔Float16 |
| Comparisons Float16→Bool | `tensor_broadcast_ops.cu:665–724`; `tensor_expr_impl.hpp:700+` | |
| Type convert ↔ f32/i32/i64/u8 | `tensor_ops.cu:2200–2209` | |
| Masked select/fill/scatter | `tensor_masking_ops.cpp:186–305, 2260–2293` | |
| Strided scatter/copy dispatch | `tensor_strided_ops.cu:135–167, 272+` via `dispatch_dtype` | **`TENSOR_LIB_FINDINGS.md` Theme B claim that strided ops omit Float16 is partially stale** — current code dispatches all enum dtypes. **[speculation]** residual silent-failure paths may remain in non-`dispatch_dtype` switches; re-verify with contiguous half transpose reproducer. |
| Promotion rules | `tensor_impl.hpp:106–130` | Float16 promoted with Float32 → Float32 |

### 1.3 Float16 holes (training-relevant)

| Hole | Behavior | Evidence |
|------|----------|----------|
| **Reductions** | `launch_reduce_op` only implements float / int32 / bool; other dtypes hit assert | `tensor_ops.cu:1366–1386` |
| Mean plan accepts Float16 | Plan allows it, then kernel rejects | `tensor_unified_ops.cpp:122–125` vs `tensor_ops.cu:1383–1386` |
| **Unary** (`abs`, `exp`, `relu`, …) | Exports only `float`/`int` for unaries | `tensor_exports.cu:47–51`; `tensor_ops.cu:2216+` — no `__half` unary instantiations |
| **Scalar ops** | Only Float32/Int32 inputs | `tensor_impl.hpp:675–682` |
| **In-place add_/mul_/…** | Destination must be Float32 | `tensor_impl.hpp:746–750, 793–798` |
| **fill_ / stream fill_** | Float32/Int32/Bool only | `tensor.cpp:2020–2021` |
| **clamp / clamp_** | Float32/Int32 only | `tensor_unified_ops.cpp:2481–2482`; `tensor.cpp:2424–2425` |
| **broadcast_to / expand** | CUDA: Float32 & Bool only; no Float16 | `tensor_broadcast.cpp:15–20` |
| **where** CUDA kernel | Float32 values only | `tensor_unified_ops.cpp:1890–1909` |
| **NN / matrix / advanced** | Float32 asserts | e.g. `tensor_utils.cpp:55,131`; `tensor_advanced_ops.cpp:17,116` |
| **Row proxy item** | Float16 unsupported | `tensor_impl.hpp:2775–2777` |

### 1.4 Training intermediates: fp16/bf16 candidacy

| Intermediate | Current dtype | Live size @1080p | fp16/bf16? | What it would take |
|--------------|---------------|------------------|------------|-------------------|
| **Fused `dm_dmu1/dsigma*`** | **Already Float16** | \(3{\times}11.87{=}35.6\) MiB | Done | `ssim.cuh:135–158`; kernels `ssim.cu:1899–1901, 1963–1969` |
| **Decoupled `app/raw_dm_*`** | Float32 | \(4{\times}23.73{=}94.9\) MiB | **Yes (high value)** | Mirror fused: allocate Float16 (`ssim.cuh:220–223`), template bwd already takes `PartialT` (`ssim.cu:2081` uses `float`) → use `__half` like fused path |
| **Masked fused/decoupled `dm_*`** | Float32 | \(3{\times}\) or \(4{\times}\) 23.73 MiB | **Yes** | Same as fused; still fp32 at `ssim.cuh:281–283, 339–342` |
| **SSIMWorkspace `dm_*` + maps** | Float32 (6 full buffers) | \(6{\times}23.73{=}142.4\) MiB | **Yes for dm_*; map maybe** | Pure SSIM path `photometric_loss.cpp:80–90`; still all-fp32 `ssim.cuh:37–42` |
| **`ssim_map` [N,1,H,W]** | Float32 | 7.91 MiB | **Possible** | Used for densification error (`trainer.cpp:4950–4980`); kernel `launch_ssim_to_error_map` would need half loads |
| **`grad_img` / `grad_corrected` / `grad_raw`** | Float32 | 23.73–47.5 MiB | **Risky** | Fed into rasterizer backward; half grads can change quality. Keep fp32 unless full mixed-precision training is designed. **Speed risk: low if only storage; quality risk: high.** |
| **L1 `grad_buffer_`** | Float32 | 23.73 MiB | Same risk | `photometric_loss.cpp:29` |
| **`densification_error_map_`** | Float32 [H,W] | 7.91 MiB | **Yes** | Scores are relative; normalize path uses float kernels (`trainer.cpp:5030–5034`) |
| **`zero_terms`** | Float32 zeros | 23.73 MiB | N/A — should **delete**, not downcast | See §5 |
| **GT images** | Often UInt8 already | ~6 MiB vs 24 MiB fp32 | Already supported in loss kernels | `loss_tensor_contract.hpp:38–41`; `ssim.cu` `pixel_value` for `uint8_t` |
| **Rendered image / corrected** | Float32 | 23.73 MiB | Hard — raster pipeline is fp32 | Out of pure tensor-lib scope |
| **Regularization temps** | Float32 small | ≤4 KiB | Irrelevant | `regularization.cpp:38–39` |
| **MCMC `_error_score_max`** | Float32 [N] | 7.63 MiB @2M | **Yes** | Relative scores; `mcmc.cpp:153` |
| **`_densification_info`** | Float32 [2,N] | 15.26 MiB @2M | **Maybe** | Kernel consumers must accept half; quality check needed |

**BFloat16:** full stack addition (enum, convert, kernels, promotion). No training consumer needs it until Float16 path is complete; bf16 mainly helps dynamic range for grads if half overflows — fused path already documents C1/C2 flooring keeps partials bounded (`ssim.cuh:131–134`).

---

## 2. In-Place Op Coverage

### 2.1 What exists

| API | Allocates? | Dtype | Evidence |
|-----|------------|-------|----------|
| `add_ / sub_ / mul_ / div_` | In-place (no new tensor) | **Float32 only** | `tensor_impl.hpp:2315–2346, 746–823` |
| `clamp_ / clamp_min_ / clamp_max_` | In-place | Float32/Int32 | `tensor.cpp:2421–2492` |
| `masked_fill_` | In-place | multi incl. Float16 | `tensor_masking_ops.cpp:238+`; `tensor_impl.hpp:2388` |
| `zero_ / fill_` | In-place | fill: f32/i32/bool | `tensor.cpp:1989–2021` |
| `index_add_ / scatter_ / index_put_` | In-place | various | `tensor_impl.hpp:2448` |
| Out-of-place `add/mul/clamp/where/masked_fill` | **Always new storage** | | Binary: `tensor_impl.hpp:880–886` (`Tensor::empty` inside expr eval); clamp: `tensor_unified_ops.cpp:2507–2508`; where: `1892`; masked_fill: `clone()` then fill `tensor_masking_ops.cpp:326–328` |

### 2.2 Gaps (always allocate where in-place exists or could)

| Op | Out-of-place | In-place / out= | Gap |
|----|--------------|-----------------|-----|
| `add/mul/sub/div` | Always new result | `add_` etc. exist | Call sites use out-of-place chains |
| `clamp` | Fused alloc+write | `clamp_` exists | Out-of-place only when result must be distinct |
| `where` | Always alloc + **materializes all 3 operands** | No `where_` / no out= | High peak |
| `masked_fill` | `clone()` then fill | `masked_fill_` exists | Trainer mask prep uses out-of-place pattern |
| `abs / neg / relu` | Always new | No generic `abs_` | Could add |
| Reductions | Always new | No out= | See §3 |
| `broadcast_to` | Always materializes full target | No zero-stride expand | See §6 |

### 2.3 Training call sites that would benefit

| Site | Pattern | Benefit |
|------|---------|---------|
| `trainer.cpp:1768–1769, 1837–1839` | Repeated `mask = mask.masked_fill(...)` | Use `masked_fill_` on owned buffer — drop 1 full mask clone per call |
| `trainer.cpp:5012, 5023` | `tile_error_map.mul_(mask)` | **Already in-place** — good |
| `trainer.cpp:1843` | `full(...) - mask` for bg_mask | `ones_like` prealloc + `sub_` or fused `1-x` kernel |
| `trainer.cpp:3138` | `rendered.clamp(0,1)` eval | `clamp_` if ownership allows |
| `trainer.cpp:4445` | `depth_loss_grad_alpha_.add_(...)` | **Already in-place** |
| `photometric_loss.cpp:87` | `full(1) - ssim_value` | Write into `loss_scalar_` with in-place or fused kernel |
| `regularization.cpp:38–39` | Per-call `empty` temps | Persistent member buffers (not in-place API, but same class of win) |
| `mcmc.cpp:267–268, 455` | `ones.slice.clone()` then `index_add_` | Prealloc ratios buffer; avoid clone |
| Error-map L1 path `trainer.cpp:4998–5006` | `(pred-gt).abs()` then `mean` then `contiguous` | Fused kernel into `densification_error_map_` (no intermediate full images) |

---

## 3. `out=` / Preallocated-Destination API

### 3.1 What the tensor lib supports today

| API | Purpose | Evidence |
|-----|---------|----------|
| `index_select_into(out, …)` | Write gather into caller buffer | `tensor_impl.hpp:2457`; `tensor_masking_ops.cpp:374–435` |
| NN `*_out` family | `relu_out`, `linear_out`, `conv1x1_bias_out`, pool outs | `tensor_impl.hpp:2378–2384`; `tensor_nn_ops.cpp:569+` |
| In-place mutators | `add_`, `fill_`, etc. | §2 |
| **Arithmetic / reduce / where / clamp / unary** | **No `out=`** | Binary always `empty`+write (`tensor_impl.hpp:884`; `tensor_expr_impl.hpp:341`) |

There is **no general PyTorch-style `torch.add(a,b,out=c)`** for core ops.

### 3.2 Where trainer / losses / strategies would use it

| Consumer | Per-step / per-call alloc | Destination API use |
|----------|---------------------------|---------------------|
| `fused_l1_ssim_forward` | `reduction_result.clone()` → loss scalar | `ssim.cu:1916, 2033` — write into trainer `loss_accumulator_` or workspace-owned loss cell |
| `PhotometricLoss` L1 path | Already uses `grad_buffer_`, `loss_scalar_` | Good pattern (`photometric_loss.cpp:63–78`) |
| Regularization | `empty({num_blocks})` + `empty({1})` every call | `regularization.cpp:36–39, 75–77, 121–122, 159–160` — persistent `out` members |
| Densification L1 error | `(corrected-gt).abs().mean(...)` temps | `trainer.cpp:4998–5006` — `error_map_out` kernel |
| `prepare_loss_images` | `contiguous()` may copy; `unsqueeze` is view | `loss_tensor_contract.hpp:59–62` — accept optional prealloc NCHW scratch |
| Strategy ratios | `clone` of ones | `mcmc.cpp:267,455,567` — `ratios_out` buffer |
| Masked fill chains | out-of-place returns | In-place / into prealloc mask workspace |

**Lib change for max leverage:** add `binary_op_out`, `unary_op_out`, `reduce_out`, `where_out`, `clamp_out` with shape/dtype/device checks and stream pin — mirrors existing `index_select_into` / NN `_out` pattern.

---

## 4. View vs Copy

### 4.1 Forced materializations

| Mechanism | When | Evidence | Peak extra mem |
|-----------|------|----------|----------------|
| **`TensorLeaf::eval_impl`** | Non-contiguous or nonzero offset leaf | `tensor.cpp:333–338` | Full tensor copy before every binary/unary through expr path |
| **Reduce: transpose+contiguous** | Single-axis reduce, not last dim, `inner_size ≥ 256` | `tensor_unified_ops.cpp:1321–1362` | **Full tensor** (explicit speed trade: comment says ~74µs→~15µs) |
| **Reduce: force contiguous** | Any non-contiguous input | `tensor_unified_ops.cpp:1389–1395` | Full tensor |
| **In-place on non-contig** | `mutate_logical_view` materializes | `tensor_impl.hpp:752–757, 800–804` | Full tensor |
| **`contiguous_read` in clamp/index_select** | Strided input | `tensor_unified_ops.cpp:2487–2489`; `tensor_masking_ops.cpp:347–351` | Full tensor |
| **`where`** | Clones operands that already match shape | `tensor_unified_ops.cpp:1866–1881` | Up to **3× full broadcast size** + result |
| **`_broadcasted`** | Always clone or broadcast_to | `tensor_unified_ops.cpp:2032–2033` | 2× broadcast volume |
| **`expand` → `broadcast_to`** | Always materializes | `tensor_shape_ops.cpp:171–172` + `tensor_broadcast.cpp:40–47` | Full expanded size (not a view) |
| **CPU broadcast of non-contig** | Contiguous then recurse | `tensor_broadcast.cpp:83–84` | Extra copy |
| **HWC→CHW densify path** | `permute.contiguous()` | `trainer.cpp:4988–4989` | 2× image (pred+gt) |

### 4.2 Stride-aware kernel coverage that would eliminate copies

| Kernel area | Needed capability | Eliminates |
|-------------|-------------------|------------|
| **Column / strided segmented reduce** | Already have fast dim0 2D path (`tensor_unified_ops.cpp:1300–1317`); extend for general non-last dims with coalesced or multi-pass strided reduce | Transpose+contig path at 1321–1375 |
| **Pointwise binary/unary** | Strided loads (shape+stride params) instead of `TensorLeaf` force-contig | Leaf copies in `tensor.cpp:335–336` |
| **where** | Broadcast indices without materializing a/b/c (kernel already takes shape arrays — `launch_where` — but host still clones first) | Host clones at 1866–1881; kernel at 1896–1909 already shape-aware for where — **host side is the bug** |
| **masked_fill / scatter** | Stride-aware (FINDINGS still claim linear scan) | Materialize path for non-contig masks |
| **broadcast binary** | Already stride-free shape-based indexing on CUDA | Good for same-dtype float/half |

**Speed note:** The reduce transpose path is **intentionally a memory/speed trade**. Removing it without a fast strided reduce **will regress speed**. Flag as **speed risk** unless replaced by equal-or-better kernel.

---

## 5. Workspace Sharing (Training Components)

### 5.1 Persistent workspaces & sizes @1080p

Assume NCHW `[1,3,H,W]` unless noted. Sizes from `ensure_size` field lists.

| Owner | Workspace | Fields (full-res) | ≈ MiB @1080p | Lifetime |
|-------|-----------|-------------------|--------------|----------|
| `PhotometricLoss` | `FusedL1SSIMWorkspace` | ssim_map C1 f32; **dm_×3 f16**; grad f32 | **~67.3** | After first fused step; member of trainer |
| `PhotometricLoss` | `SSIMWorkspace` | 6× C3 f32 + tiny | **~142.4** | Only if `lambda_dssim==1` pure SSIM |
| `PhotometricLoss` | `grad_buffer_` + L1 reduce | C3 f32 + ≤1024 | **~23.7** | Only pure L1 (`lambda==0`) |
| `Trainer` | `DecoupledFusedL1SSIMWorkspace` | ssim C1; **4× dm f32**; **zero_terms f32**; 2× grad | **~174.0** | Appearance path once used |
| `Trainer` | `MaskedFusedL1SSIMWorkspace` | ssim C1; **3× dm f32**; grad | **~103.0** | Mask path once used |
| `Trainer` | `MaskedDecoupledFusedL1SSIMWorkspace` | like decoupled | **~174.0** | Mask+appearance |
| `Trainer` | `SSIMMapWorkspace` densify | ssim_map C3 or C1 | **~7.9–23.7** | When error map needs separate SSIM |
| `Trainer` | `densification_error_map_` | [H,W] f32 | **~7.9** | Densify strategies |
| `Trainer` | depth/normal loss grads | image-sized when enabled | **~7.9–23.7 each** | Optional features |
| `Trainer` | `roi_weight_map_`, edge map | [H,W] | **~7.9 each** | Optional |
| `Trainer` | `bg_mix_buffer_` | [3] | negligible | Always tiny |
| `Trainer` | `bg_image_cache_` | up to 256 MiB budget | **≤256** | `trainer.hpp:482` |
| MCMC/IGS | `_densification_info` [2,N] | 2M×2×4 | **15.3** @2M | `mcmc.cpp:146–147` |
| MCMC | `_error_score_max` [N] | 2M×4 | **7.6** | `mcmc.cpp:153` |
| MCMC | `_ones_int32` | ≥N×4 | **≥7.6** | `mcmc.cpp:177–178` |
| Regularization | **no persistent workspace** | per-call temps | ~4 KiB churn | `regularization.cpp:38` |

Definitions: `ssim.cuh:15–48, 129–164, 200–231, 263–350`; members `trainer.hpp:509–548`; VRAM accounting `trainer.cpp:747–816`.

### 5.2 Overlap / never simultaneously live (same step)

| Pair | Simultaneous in one train step? | Share potential |
|------|---------------------------------|-----------------|
| Fused vs Decoupled | **No** — branch on `raw_rendered` (`trainer.cpp:1669–1692`) | **One arena region** for both (~174 MiB capacity covers fused 67) |
| Masked vs unmasked | **No** — different entry points (`1749+` vs `1664+`) | Share fused↔masked, decoupled↔masked_decoupled |
| Fused `grad_img` vs Decoupled `grad_corrected` | Exclusive | Same storage |
| Pure L1 `grad_buffer_` vs fused `grad_img` | Exclusive (`lambda` 0 vs mid) | Share |
| Pure SSIM workspace vs fused | Exclusive on `lambda` | Rarely both grown; if user sweeps λ, **both stick** as members |
| Densification error map vs loss `ssim_map` | Sequential same step: loss first, then error extract | Can alias reshape of ssim_map when C=1 (`trainer.cpp:4964–4968` already tries view path) |
| Depth/normal grads vs photo workspace | After photo forward/backward typically | **Could alias** photo grad buffer once photo bwd done — **[speculation]** needs phase ordering audit |
| Regularization temps vs loss reduce temps | After photo | Share 1024-float reduce scratch |
| All four trainer loss workspaces | Members retained forever once `ensure_size` ran | **Peak = sum if user ever hit each mode**; steady default = fused only |

**Default MRNF/MCMC 1080p no appearance/masks:** ~67 MiB fused + ~8 MiB error map + strategy N-buffers ≈ **90+ MiB** loss-related, not hundreds.

**Worst case** (appearance + mask modes ever used in process lifetime):  
\(67 + 174 + 103 + 174 + 142\) ≳ **650 MiB** of mutually exclusive workspaces held concurrently as separate allocations — **primary sharing win**.

### 5.3 `zero_terms` special case

Decoupled backward passes **zeros as unused SSIM partials** (`ssim.cu:2087–2088, 2335–2336`). That is a **full 23.73 MiB tensor of zeros** (`ssim.cuh:224, 343`).  
**Better:** specialized backward that omits those terms, or a **1-element zero** with a kernel flag — pure memory free with **no accuracy change**.

---

## 6. Broadcast / Expansion Materialization

**Yes — broadcasting always materializes expanded operands for explicit `broadcast_to` / `expand`.**

| Path | Behavior | Evidence |
|------|----------|----------|
| `Tensor::expand` | reshape pad then **`broadcast_to`** | `tensor_shape_ops.cpp:143–172` |
| `broadcast_to` | `Tensor::empty(target)` + copy kernel | `tensor_broadcast.cpp:40–77` |
| Same shape already | **Still `clone()`** | `tensor_broadcast.cpp:23–24` |
| Binary CUDA ops | Prefer **broadcast binary kernel** without expanding storage | `tensor_expr_impl.hpp:351–361` — good |
| Binary CPU | Materializes both sides | `tensor_impl.hpp:661–664` |
| `where` | Explicit expand/clone of all three | `tensor_unified_ops.cpp:1866–1881` |
| `_broadcasted` helper | clone or broadcast_to both | `2032–2033` |

**No zero-stride expand views** (PyTorch-style `as_strided` expand). Memory cost of expand is always \({\prod}\text{target dims} \times \text{elem size}\).

Training impact: moderate — most hot loss kernels take equal-shaped images. High impact if `where` or expand used on image-sized tensors; mask prep uses comparisons more than expand.

---

## 7. Top 10 Memory Levers (1080p / 2M splats)

Ranked by **realistic bytes saved**, with effort and speed risk. Excludes pure allocator work.

| # | Lever | ≈ Bytes saved | Where | Effort | Speed risk |
|---|-------|---------------|-------|--------|------------|
| **1** | **Union / arena-share exclusive loss workspaces** (fused ↔ decoupled ↔ masked ↔ pure SSIM); never retain 3–4 full variants | **100–500+ MiB** peak if multiple modes ever used; **0** if only default fused | `trainer.hpp:539–542`; `photometric_loss.hpp:49–52`; sizes §5 | **M** (lifetime redesign, single `LossWorkspaceArena`) | **None** if same kernels; **possible win** (better cache locality) |
| **2** | **Port Decoupled (+ Masked) `dm_*` to Float16** like fused | Decoupled: **~47.5 MiB**; Masked fused: **~35.6 MiB**; Masked decoupled: **~47.5 MiB** | `ssim.cuh:220–223, 281–283, 339–342`; kernels `ssim.cu:2015–2018, 2081+` | **M** (kernel + ensure_size + tests) | **Low** (fused already proves path; half bandwidth ↓ can be **faster**) |
| **3** | **Eliminate `zero_terms` full buffer** | **23.7 MiB** per decoupled workspace (×2 if both masked+unmasked) | `ssim.cuh:206–224`; `ssim.cu:2087–2088` | **S–M** | **None / slight win** |
| **4** | **Port pure `SSIMWorkspace` partials to f16** (and optional f16 ssim_map) | **~71 MiB** (3 dm half) + optional 11.9 if maps half | `ssim.cuh:17–42`; pure path `photometric_loss.cpp:80–90` | **M** | **Low** |
| **5** | **`where` host: stop cloning equal-shape operands; true broadcast where** | Peak **up to 3× image** when used on large tensors | `tensor_unified_ops.cpp:1866–1881` | **S–M** | **Win** (less mem traffic) |
| **6** | **Zero-stride `expand` / view broadcast** (or document + use broadcast kernels only) | Training-dependent; lib-wide can be large for expand-heavy code | `tensor_shape_ops.cpp:171–172`; `tensor_broadcast.cpp:40–47` | **L** (stride-aware everything) | **Win** if consumers stop contig; **regress** if kernels assume dense |
| **7** | **Fused densify error kernels** (replace abs-diff mean chain + HWC permute contig) | Peak temps **~24–48 MiB** + stable path avoids 2× permute copies | `trainer.cpp:4988–5006` | **S–M** | **Win** |
| **8** | **General `out=` for reduce / binary / clamp + use in loss/reg** | Steady: kills per-step small allocs; peak: error-map / ratios intermediates | `tensor_impl.hpp` API; `ssim.cu:1916`; `regularization.cpp:38`; `mcmc.cpp:267` | **M–L** | **Win** if avoids alloc; **none** for compute |
| **9** | **Strategy score buffers f16 / tighter types** (`_error_score_max`, maybe densify info) | **~7.6 MiB** + optional **7.6 MiB** @2M | `mcmc.cpp:146–154` | **S–M** | **Low**; validate densify sampling quality |
| **10** | **Strided reduce without transpose copy** | Peak = **full reduced tensor** (image or feature size) when dim≠last & inner≥256 | `tensor_unified_ops.cpp:1321–1362` | **L** | **HIGH** if new kernel slower than transpose+reduce; must beat ~15µs column path |

### Near-miss / smaller levers

- **In-place mask `masked_fill_`** in trainer (`1768+`): one mask-sized buffer (~2 MiB UInt8 / 8 MiB f32).  
- **Regularization persistent scratch**: tiny but free churn.  
- **`reduction_result.clone()`**: 4 bytes — noise vs pool traffic.  
- **Image grads in f16**: large bytes but **quality/speed risk** — not recommended as a pure memory play.  
- **bg_image_cache 256 MiB cap** (`trainer.hpp:482`): policy knob, not tensor-lib.

---

## 8. Cross-Cutting Constraints (for the optimization plan)

1. **Speed is top priority:** Prefer levers that **reduce bandwidth** (f16 partials, kill zero_terms, stop where-clones). Defer remove-transpose-reduce until strided kernel is proven faster.  
2. **Do not regress fused L1+SSIM** — it is already the optimized default with f16 partials (`ssim.cuh:131–158`).  
3. **Workspace union is the largest peak win** with zero numerical change.  
4. **BFloat16 is greenfield**; Float16 is the practical half type.  
5. **`TENSOR_LIB_FINDINGS.md` Float16 strided hole** may be fixed via `dispatch_dtype`; **reduce/fill/clamp/unary/broadcast/in-place** holes remain real for using half intermediates in the tensor API itself.  
6. **Allocator work is complementary:** sharing arenas + f16 shrinks *capacity demand*; allocator work shrinks *fragmentation/churn*. Both needed for VRAM headroom at 1080p / multi-million splats.

---

### Size cheat-sheet (one image buffer)

| Shape | fp32 | fp16 |
|-------|------|------|
| `[1,3,1080,1920]` | 23.73 MiB | 11.87 MiB |
| `[1,1,1080,1920]` | 7.91 MiB | 3.96 MiB |
| `[1080,1920]` | 7.91 MiB | 3.96 MiB |
| `[2, 2e6]` densify info | 15.26 MiB | 7.63 MiB |

*[speculation] marked where dynamic mode history or quality impact cannot be proven statically.*

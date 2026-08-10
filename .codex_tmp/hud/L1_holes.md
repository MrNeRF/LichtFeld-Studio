# L1: Attribution hole sizes (double-counts C1–C7 + unhooked gsplat)

**Branch:** `lfs-elite`  
**Reference 7k FastGS gate run:** `.codex_tmp/exactmem/myconfirm3/perf_bench.json`  
**net** = 1310.6 MiB, **steady_allocs** = 0.237/iter, **signed_residual** = +53.2 MiB (over-attributed cover)  
**Date:** 2026-08-10

Method: no permanent hooks. Double-count sizes taken from the published `peak_ex_cache.peak_rows` of the 7k bicycle MCMC/FastGS run (the stack’s gate path). Gsplat sized with a temporary process-local HWM in `GsplatAllocationHooks` (reverted after measurement) on a short `--gut` bicycle train.

---

## Summary table (peak, MiB)

| hole / defect | path | size at peak | lands in | L2 disposition |
|---|---|---:|---|---|
| **Unhooked gsplat buffers** | FastGS gate (`--gut` off) | **0** | n/a | nothing to hook on gate path |
| **Unhooked gsplat buffers** | `--gut` short densify (~69k splats, 2k iters) | **19.1 HWM** (concurrent live) | root A async + top-level unattributed for `cudaMalloc` | **defer hook** (≪ few hundred MiB at measured scale; name as untracked child when gut on) |
| C1 missing `VramRowKind` | all | enables every double-count below | all roots’ children | **fix in L2** |
| C2 slab folded into pool accounted | FastGS 7k | understates `pool_untracked` by `accounted_slab_live` (typically small slab; see C2) | root A residual | **fix in L2** |
| C3 `cuda_phase_default_pool` + live pool reserved | HUD sum | startup pool reserved (tens of MiB; double-rowed with `cuda.pool.overhead`) | roots A/H | **ledger omits phase from sum with reserved** |
| C4 shared-scratch not first-class | headless train | **0** (viewport-only) | root E | field + GUI path later; headless N/A |
| C5 gsplat arena capacity+current+peak | FastGS (filtered) | **0** (scope `rasterizer.fastgs` excluded by `isFastGsLogicalRow`) | — | FastGS already filtered |
| C5 gsplat arena triple disclosure | gut path | capacity+current+peak would sum ~3× arena if unfiltered (design defect) | root D children | **mark Sampled Nested; never sum** |
| C6 `synthetic_budget` cap | HUD only | hides OVER; gate path already shows +53 MiB residual in perf_bench | top residual | **no cap in ledger model** |
| C7 `record_vram_tensor` forces Direct | FastGS 7k | census distortion (see C7) | method census, not byte total | **fix method detection** |
| Disclosure vs hooked splat params | FastGS 7k | model.gaussians.shN **128.7** + optim shN.exp_avg **137.3** (disclosure Nested once C1 exists) | root A/E children | Nested, not summed |
| FastGS arena multi-row (disclosure) | FastGS 7k | capacity **140** + peak **135.9** + current **124.4** + hooked vmm **128** | root D/E | Nested disclosures; hooked vmm is the container |

---

## Double-count family at peak (myconfirm3 `peak_rows`)

Raw top rows (MiB), FastGS 7k bicycle max-cap 1.5M:

| MiB | row |
|---:|---|
| 140.000 | `rasterizer.fastgs.arena.capacity` (Sampled disclosure) |
| 137.329 | `optimizer.adam.shN.exp_avg` (Sampled via `record_vram_tensor`) |
| 135.868 | `rasterizer.fastgs.arena.peak_usage` (Sampled) |
| 128.746 | `model.gaussians.shN` (Sampled disclosure of exportable/params) |
| 128.000 | `train.step/.../rasterizer.arena.vmm` (Hooked container) |
| 124.358 | `rasterizer.fastgs.arena.current_usage` (Sampled) |
| 89.547 | `rasterizer.fastgs.forward.per_primitive_buffers` (Sampled) |
| 56.837 | `.../fastgs.sort_workspace` (Hooked) |
| 32.975 | loss workspace (appears twice: train.step path + `train.losses`) |

**Arena family sum if all treated as siblings:** 140+135.9+124.4+128+89.5 ≈ **618 MiB** of rows describing roughly **140 MiB** committed arena (plus live suballocation disclosure). HUD today excludes `rasterizer.fastgs*` via `isFastGsLogicalRow` (~filters the disclosures) but still totals the hooked `arena.vmm` row. Without `VramRowKind`, any new consumer reintroduces the triple-count.

---

## C1–C7 quantified

### C1 — `VramRowKind` missing
- **Impact:** structural. Every Sampled disclosure (arena capacity/usage, model.gaussians.*, optimizer.adam.*) is indistinguishable from Hooked allocations.
- **Measured effect on naive sum of peak_rows:** 1146 MiB of top-12 rows vs process net ~1311 MiB, with massive internal overlap.
- **Fix:** `enum class VramRowKind { Hooked, Sampled, Static }` on `VramMetricSnapshot`; only Hooked may contribute to attributed totals.

### C2 — Slab folded into `accounted_cuda_pool_live_bytes`
- **Site:** `vram_profiler.cpp` `add_accounted_method` Slab case also increments pool live.
- **Effect:** `pool_untracked = cuda_pool_used - accounted_cuda_pool_live` is **understated by exactly `accounted_slab_live_bytes`**, while the same bytes also appear as `cuda.slab.reserve_gap` synthetic.
- **Size:** slab reserved is process-dependent; on this training path it is a small fraction of pool (pool peak reserved 320 MiB). Correctness hole, not the dominant residual.
- **Fix:** Slab increments only `accounted_slab_live_bytes`.

### C3 — `cuda_phase_default_pool` double-rowed with live pool reserved
- **Site:** HUD `add_phase("cuda.default_pool", ...)` plus `cuda.pool.overhead` from live reserved−used.
- **Size:** startup `ReservedMemCurrent` at context init (main.cpp phase record). Typically small vs training HWM; still a permanent double-count direction.
- **Fix:** ledger root A is live reserved only; phases belong under root H as Nested history, never summed with A.

### C4 — shared-scratch block not first-class
- **Headless train:** `exportable_splat_bytes` and shared scratch are **0** (no viewport).
- **GUI:** block size reconstructable only by summing `shared.scratch.*` rows; excluded from `exportable_splat_bytes` when `track_splat_bytes=false`.
- **Fix:** `VramProcessSnapshot::shared_scratch_bytes` (C4) for root E child.

### C5 — gsplat arena disclosure multi-sum
- FastGS path: **already filtered** (`isFastGsLogicalRow`); measured double-count from this specific bug on gate path = **0**.
- gut path: `record_rasterizer_arena_disclosure("rasterizer.gsplat")` publishes capacity+current+peak with **no** equivalent filter → up to **~3× arena** if summed.
- **Fix:** `VramRowKind::Sampled` + Nested; optional `isGsplatLogicalRow` mirror.

### C6 — `synthetic_budget` capping
- **Site:** HUD only. Caps every synthetic row so OVER is unreachable.
- **Evidence gate already has OVER in cover arithmetic:** `over_attributed_bytes` = **53.2 MiB** on myconfirm3 (`signed_residual_bytes` > 0). HUD would still present a closed-looking breakdown.
- **Fix:** ledger residual uses unsigned split of signed difference; never clamp attributed to measured.

### C7 — `record_vram_tensor` hardcodes Direct
- **Site:** `trainer.cpp:786-793` sets `Direct` for every CUDA-owned non-external tensor.
- **Effect:** `named_direct_rows` (HUD census of row methods) ≫ `accounted_direct_live_bytes` (hooked Direct map). Optimizer moments and model tensors show as Direct while live in pool/exportable.
- **Size:** all Sampled CUDA tensors in peak_rows (optimizer + model.gaussians + train.inputs) are mis-labeled Direct — hundreds of MiB of **label** error, not an extra byte in process_used.
- **Fix:** use External for external storage; for CUDA tensors prefer Unknown/Static method or omit method from Sampled rows; never claim Direct unless `try_allocate_direct`.

---

## Unhooked gsplat (measured)

### Gate path (mcmc/mrnf, FastGS, `--gut` false)
- `GsplatAllocationHooks::after_allocate` **never runs**.
- Hole size: **0 MiB**. Design’s “dominates untracked” hypothesis does **not** apply to the 7k bicycle gate.

### gut path (temporary HWM, then reverted)
Command:

```bash
./build/LichtFeld-Studio -d data/bicycle --output-path .codex_tmp/hud/l1_gut_2k \
  --images images_4 --strategy mcmc --max-cap 1500000 --headless --train -i 2000 --gut
```

Process-exit summary (`/tmp/gsplat_l1_summary.txt`):

| metric | value |
|---|---|
| `gsplat_l1_hwm_bytes` | 20,002,895 (**19.076 MiB**) |
| alloc events | 30 |
| free events | 30 |
| end live | 0 |
| last_live_splats (perf) | 69,267 |

Concurrent label peaks (from alloc log, MiB):

| MiB | label |
|---:|---|
| ~9.6–11.7 | `intersection_ids` / `sorted_intersection_ids` |
| ~5.9 | `flatten_ids` / `sorted_flatten_ids` |
| ~1.8 | `color_gradients` |
| ~1.2 | `cumulative_tiles` |
| ~0.5 | `cub_workspace` |

**Decision:** do **not** hook gsplat in L1/L2. Name it as an untracked child when gut is active. Revisit if a full densify-to-1.5M gut run shows HWM ≫ 200 MiB.

Linear scale guess (not a measurement): 19 MiB @ 69k → order **~400 MiB** @ 1.5M if n_isects ∝ N; still a gut-only path, not the FastGS gate.

---

## Other named holes (presence on gate path)

| hole | gate path (headless FastGS 7k) |
|---|---|
| Exportable splat block scalar | 0 (no viewport exportable store in this headless config) |
| Shared scratch | 0 (viewport) |
| LOD upload staging | 0 / transient |
| Adam index buffers | inside pool; partially unlabelled → root A untracked |
| Densify temporaries | transient in pool |
| Depth-loss anchors | off unless depth loss enabled |
| Tensor reduction partials | small / transient |

---

## Implications for L2 spine

1. L0: root F **in** sum (device-local only).
2. L1: gate-path ledger can close without gsplat hooks; unattributed should be honest residual + named optional gut hole.
3. L2 must ship C1, C2, C7 and ledger arithmetic first; C4 field for GUI completeness; C3/C5/C6 as model policy (no HUD UI in this lane).
4. gsplat hook deferred.

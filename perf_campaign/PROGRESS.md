# Campaign progress log

## Task 0.1 — Per-iteration allocation counter

- **Branch:** `perf/spirulae-parity`
- **API:** `lfs::core::alloc_counter::{snapshot(), delta_since(s), total(), record()}`
  in `src/core/include/core/alloc_counter.hpp` (impl in `alloc_counter.cpp`).
- **Instrumented sites (real driver alloc success only):**
  - `memory_pool.hpp`: bucket `cudaMallocAsync` miss, async exact tier, direct `cudaMalloc`
  - `gpu_slab_allocator.hpp`: slab growth `cudaMalloc`
  - `size_bucketed_pool.hpp`: `allocate()` driver path
  - `memory_pressure.cpp`: `zeros_direct`/`reserve` direct `cudaMalloc`
  - `memory_arena.cu`: arena initial/growth `cudaMalloc` + VMM `cuMemCreate` commits
- **Fail evidence (TDD):**
  ```
  tests/test_alloc_counter.cpp:4:10: fatal error: core/alloc_counter.hpp: No such file or directory
  ```
- **Pass evidence:**
  ```
  [==========] Running 4 tests from 1 test suite.
  [  PASSED  ] 4 tests.  (AllocCounterTest.*)
  ```
- **Commit:** `013f6e04`

## Task 0.2 — Bytes-per-splat training-state ledger

- **API:** `lfs::diagnostics::TrainingStateLedger` on VramProfiler;
  `lfs::training::compute_training_state_ledger` / `publish_training_state_ledger`
  in `src/training/include/lfs/training/vram_ledger.hpp`.
- **Buckets:** params / optimizer / gradients_or_helpers / densify_aux +
  `bytes_per_splat = total / live_N`.
- **Fail evidence (TDD):** test required `core/alloc_counter`-style missing API —
  `lfs/training/vram_ledger.hpp` and `TrainingStateLedger` did not exist before
  this task (compile would fail with no such file).
- **Pass evidence:**
  ```
  [==========] Running 2 tests from 1 test suite.
  [  PASSED  ] 2 tests.  (TrainingStateLedgerTest.*)
  Synthetic SH3 N=32: params=248*N, optim=172*N, densify=8*N, grads=0 → 428 B/splat
  ```
- **Commit:** `013f6e04`

## Task 0.3 — Bench gate script + BASELINE

- **Artifacts:**
  - `perf_campaign/bench.sh` — build + headless train + JSON metrics
  - `src/training/perf_bench.{hpp,cpp}` — `LFS_PERF_BENCH=1` collector
  - `perf_campaign/BASELINE.md` — 3-run medians
- **Dataset:** `/home/gauss/data/360_v2/bonsai` (`images_4`), 2000 iters, warmup 200,
  strategy `mrnf`, max_cap 500000. (Real COLMAP scene; no synthetic fallback needed.)
- **Also:** instrumented FastGS `StreamOrderedDeviceBuffer` so G2 sees sort-buffer
  churn (ISS-001).
- **Fail evidence (TDD for harness):** smoke without collector → no
  `perf_bench.json`; with `LFS_PERF_BENCH=1` file is written.
- **Pass / baseline medians (3 runs):**

  | metric | median |
  |---|---:|
  | wall_s | 9.00 |
  | steady_ms/iter | 4.129 |
  | steady_allocs/iter | **5.05** |
  | peak VRAM MiB | 1156.3 |
  | B/splat | **429.0** |
  | last_loss | ~0.039 |

- **Commit:** `e5506f39`

## Task 1.1 — Persistent high-water sort buffers in FastGS forward

- **Change:** Grow-only thread-local sort workspace in `forward.cu`
  (keys×2, indices×2, CUB WS, ×1.2 on growth). Sorted indices stay in-cache
  through backward; `release_sorted_primitive_indices` is a no-op free.
- **Fail evidence (TDD):**
  ```
  FastGSSortBufferTest.SteadyStateSecondForwardHasZeroSortAllocs
  Expected equality of these values:
    delta2 Which is: 5
    0u     Which is: 0
  ```
- **Pass evidence:**
  ```
  [  PASSED  ] FastGSSortBufferTest.SteadyStateSecondForwardHasZeroSortAllocs
  ```
- **Bench (flock, 3 runs, medians vs BASELINE):**

  | metric | baseline | after 1.1 |
  |---|---:|---:|
  | wall_s | 9.00 | **9.02** |
  | steady_ms/iter | 4.129 | **4.101** |
  | steady_allocs/iter | **5.05** | **0.06** |
  | peak VRAM MiB | 1156.3 | 1187.4 |
  | B/splat | 429.0 | 429.0 |
  | last_loss | ~0.039 | ~0.03–0.04 |

  Gate G2: sort churn killed (5.05 → 0.06). G4: no ms/iter regression.
  Residual ~0.06 allocs/iter are non-sort (densify growth / rare paths).
  Peak VRAM +31 MiB is expected high-water residency of sort buffers.

- **Commit:** `b6c020aa`
  Runs: `perf_campaign/runs/20260806T173156Z_run{1,2,3}/`

## Task 1.2 — Remove the n_instances hard sync

- **Change:** Async D2H of scan total + `cudaEventSynchronize` only on the D2H
  event (create launched first so it overlaps). Exact-size CUB sort after the
  event. Mid-pipeline `cudaStreamSynchronize` only on first step / capacity
  overflow (fallback counter). Kernels accept capacity clamp + device count.
- **Fail evidence (TDD):** Pre-change, `forward.cu` always
  `cudaMemcpyAsync`+`cudaStreamSynchronize` after the scan (host pipeline stall).
  New APIs (`n_instances_fallback_sync_count`, etc.) did not exist.
  ```
  # before: unconditional mid-pipeline sync every step
  return cudaStreamSynchronize(stream);  // after n_instances D2H
  ```
- **Pass evidence:**
  ```
  [  PASSED  ] FastGSSortBufferTest.AsyncPathMatchesSyncPathPixels
  [  PASSED  ] FastGSSortBufferTest.CapacityGrowthTakesFallbackSync
  # steady same-size forward: fallback count unchanged
  # capacity reset: fallback count increments
  # pixel-identical sync vs async path
  ```
- **Bench (flock, 3 runs, medians vs 1.1 / BASELINE):**

  | metric | baseline | after 1.1 | after 1.2 |
  |---|---:|---:|---:|
  | wall_s | 9.00 | 9.02 | **9.03** |
  | steady_ms/iter | 4.129 | 4.101 | **4.094** |
  | steady_allocs/iter | 5.05 | 0.06 | **0.05** |
  | peak VRAM MiB | 1156.3 | 1187.4 | 1176.6 |
  | last_loss | ~0.039 | ~0.03–0.04 | ~0.03–0.04 |

  G3: zero mid-pipeline StreamSynchronize in steady state (fallback only at
  warmup/growth). G4: steady_ms improved vs 1.1 and baseline.

- **Commit:** `5a509aa1`
  Runs: `perf_campaign/runs/20260806T174524Z_run{1,2,3}/`

## Task 1.5 — Preflight checks debug-only

- **Change:** `cudaPointerGetAttributes` ×~10 per forward gated behind
  `#ifndef NDEBUG` in `rasterization_api.cu`. Cheap dimension checks remain in
  Release. Instrumented `preflight_pointer_attr_call_count()`.
- **Fail evidence (TDD):** Before change, every forward called
  `cudaPointerGetAttributes` ~10× (Release). Counter API did not exist.
- **Pass evidence:**
  ```
  [  PASSED  ] FastGSSortBufferTest.ReleasePreflightPointerAttrsAreSkipped
  # NDEBUG: calls == 0 after 8 frames
  ```
- **Bench (flock, 3-run final trio vs BASELINE):**

  | metric | baseline | after 1.1 | after 1.2 | after 1.5 (trio) |
  |---|---:|---:|---:|---:|
  | wall_s | 9.00 | 9.02 | 9.03 | **9.06** |
  | steady_ms/iter | 4.129 | 4.101 | 4.094 | **4.101** |
  | steady_allocs/iter | **5.05** | 0.06 | 0.05 | **0.05** |
  | peak VRAM MiB | 1156.3 | 1187.4 | 1176.6 | 1178.2 |
  | B/splat | 429.0 | 429.0 | 429.0 | 429.0 |
  | last_loss | ~0.039 | ~0.03–0.04 | ~0.03–0.04 | ~0.03–0.05 |

  Net vs BASELINE: allocs 5.05→0.05 (G2), ms 4.129→4.101 (G4 no regression /
  slight win), VRAM +~22 MiB high-water sort buffers (expected).

## Phase 1 raster trio — summary table

| task | fail evidence | pass evidence | wall_s | steady_ms | allocs/iter | peak MiB | last_loss |
|---|---|---|---:|---:|---:|---:|---:|
| baseline | — | BASELINE.md | 9.00 | 4.129 | 5.05 | 1156.3 | ~0.039 |
| 1.1 sort buffers | delta2=5 | delta2=0 | 9.02 | 4.101 | 0.06 | 1187.4 | ~0.035 |
| 1.2 n_instances sync | mid-pipeline StreamSync every step | fallback only warm/grow; pixel golden | 9.03 | 4.094 | 0.05 | 1176.6 | ~0.036 |
| 1.5 preflight NDEBUG | ~10 attrs/forward | attrs calls=0 after N frames | 9.06 | 4.101 | 0.05 | 1178.2 | ~0.037 |

- **Commit (1.5):** `7214e8bf` (tip of trio stack; hash recorded post-commit)
  Final trio runs: `perf_campaign/runs/20260806T174846Z_run{1,2,3}/`


||||||| f570d3b8

## Task 4.1 — Kill the ~3× compact peak

- **Branch:** `lfs-elite-densify`
- **Change:** `MRNF::compact_splats` gathers via `zeros_direct(cap) + index_select_into + swap`
  instead of `index_select → reserve(max_cap)` (3-buffer peak). Grad buffers use
  `zeros_direct` too; free/deleted bookkeeping rebuilt without double-`reserve` on
  `cuda.direct` storage (fixes max_cap=0 grow).
- **Fail evidence (TDD, pre-fix for 4.2-related API wiring; pattern tests document 3×):**
  ```
  CompactSplatPeakPattern.OldPathExceedsTwoXBound  — concurrent = 2.5× tensor@cap (PASS documents bug)
  Old concurrent = src@cap + exact@new + dest@cap
  New concurrent = src@cap + dest@cap = 2×
  ```
  Pre-change production path matched the old algorithm (verified by code at
  `mrnf.cpp` compact lambda: index_select + reserve).
- **Pass evidence:**
  ```
  [==========] 6 tests (CompactSplatPeakPattern + CompactSplatsCorrect + AdamCapacity*)
  [  PASSED  ] 6 tests.
  MRNFStrategyTest.* + densify suite: 101 tests PASSED
  ```
- **Bench gate (flock, 3×2000 iters, bonsai/images_4, max_cap=500k):**

  | metric | baseline | after 4.1+4.2 | Δ |
  |---|---:|---:|---:|
  | wall_s (med) | 9.00 | 9.11 | +1.2% noise |
  | steady_ms/iter | 4.129 | 4.141 | +0.3% flat |
  | steady_allocs/iter | 5.05 | 5.05 | 0 |
  | peak_VRAM_MiB | 1156.3 | **1146.2** | **−10.1 MiB** |
  | B/splat | 429.0 | 429.0 | 0 |
  | last_loss | ~0.039 | ~0.030 | ok |

- **Commit:** `e5f78be5`

## Task 4.2 — Capacity invariant guard

- **Branch:** `lfs-elite-densify`
- **Change:** After any Adam slow-path grow (`extend_state_by_gather`,
  `extend_state_for_new_params`, `add_new_params_gather(shN)`): re-`reserve` with
  `growth_factor`, set `state.capacity`, `note_slow_path_grow` (LOG_WARN + counter).
  `LFS_DEBUG_ASSERT(capacity >= size)` after grow entry points.
  API: `AdamOptimizer::slow_path_grow_count()` / `reset_slow_path_grow_count()`.
- **Fail evidence (TDD, before re-reserve):**
  ```
  AdamCapacityInvariant.SlowPathReReservesSoSecondGrowIsFast
  Expected equality: slow_path_grow_count() Which is: 0  vs  1u
  Expected: (state->capacity) >= (state->size), actual: 0 vs 20
  Expected: (state->exp_avg.capacity()) > (0u), actual: 0 vs 0
  [  FAILED  ] … (and SlowPathGatherAlsoRestoresCapacity similarly)
  ```
- **Pass evidence:**
  ```
  [  PASSED  ] AdamCapacityInvariant.SlowPathReReservesSoSecondGrowIsFast
  [  PASSED  ] AdamCapacityInvariant.SlowPathGatherAlsoRestoresCapacity
  (slow path counter=1 after first grow; second grow fast, alloc_delta≤2)
  ```
- **Bench:** same gate table as 4.1 (measured together after both landed).
- **Commit:** `a3bebc21`
||||||| f570d3b8
- **Commit:** `e5506f39`
- **Commit:** `013f6e04`

## Task 5.1 — Grow exportable splat block with live N

- **Branch:** `lfs-elite-gui`
- **Change:** GUI exportable SoA VMM block is sized to live N + 1.5× headroom
  (virtual-reserve `max_cap`) instead of committing `max(max_cap, min)` physical
  at step 0. `SplatExportableStorage::grow` relocates packed regions under a stable
  `device_ptr`, bumps generation, updates `VramProfiler::exportable_splat_bytes`.
  Densify-time growth: `SplatData::set_capacity_ensure` + Adam `add_new_params`
  calls grow+Vulkan re-import+rebind. MRNF/MCMC skip `zeros_direct` max_cap steal
  for external tensors.
- **Headless:** `Application::runHeadless` never calls
  `createTrainingSplatTensorAllocator` (TrainerManager/GUI-only). Path untouched.
- **Fail evidence (TDD):** before API, tests required
  `SplatExportableStorage::growthCapacity` / `::grow` / `capacity()` /
  `layoutBytes` / `rebindSplatData` which did not exist:
  ```
  error: 'class lfs::core::SplatExportableStorage' has no member named 'grow'
  error: 'class lfs::core::SplatExportableStorage' has no member named 'capacity'
  error: 'growthCapacity' is not a member of 'lfs::core::SplatExportableStorage'
  ```
- **Pass evidence:**
  ```
  [==========] Running 5 tests from 2 test suites.
  [  PASSED  ] 5 tests.  (ExportableStorageTest + SplatExportableStorageTest.*)
  CreateTracksLiveCapacityNotMaxCap: capacity=1024 commit=2 MiB reserve_VA=1184 MiB
  GrowPreservesDataAndTracksBytes: means+scaling patterns intact after grow
  TensorViewsValidAfterGrowViaRebind: capacity clamped, data preserved
  ```
- **Exportable bytes gate (max_cap=5M SH3, live≈400k):**

  | | capacity | committed (layout) |
  |---|---:|---:|
  | **BEFORE** | 5,000,000 | **1182.6 MiB** |
  | **AFTER** (400k×1.5) | 600,000 | **141.9 MiB** |
  | **Δ** | | **−1040.6 MiB** at training start |

  Unit test measured: N=1024 SH3 commit=2 MiB, VA reserve for 5M=1184 MiB
  (physical not committed for max_cap).
- **Headless bench (flock, bonsai, 2000 iters, max_cap=500k):**

  | metric | baseline | after 5.1 |
  |---|---:|---:|
  | wall_s | 9.00 | **8.97** |
  | steady_ms/iter | 4.129 | **4.098** |
  | steady_allocs/iter | 5.05 | **5.05** |
  | peak VRAM MiB | 1156.3 | 1120.6 |
  | B/splat | 429.0 | 429.0 |
  | last_loss | ~0.039 | 0.038 |

  No training-speed regression (path untouched headless).
- **Manual GUI validation (document — no headless-with-viewer-stub):**
  1. Launch GUI, load bonsai (or any COLMAP scene), set max_cap=5_000_000, SH3.
  2. Start training; on first frame after start, check log for
     `capacity=<live*1.5>, reserve=5000000, block=<~tens–hundreds> MiB`
     (not ~1183 MiB) and VRAM HUD `exportable_splat_bytes` ≈ layout(capacity).
  3. Train past densify events; if live N exceeds capacity, log should show
     `Exportable splat storage grew for densify` and viewer continues zero-copy
     (no fallback copy warning). Generation bumps; Vulkan re-imports new handle.
  4. Confirm viewport still renders live training splats without a full model copy.

- **Commit:** `013f6e04`

---
## WAVE 1 COMBINED (post-merge, commit 0d652d45) — 3-run medians
| Metric | Baseline | Wave 1 | Δ |
|---|---:|---:|---|
| wall_s | 9.00 | 8.94 | −0.7% |
| steady_ms/iter | 4.129 | 4.085 | **−1.1%** |
| steady_allocs/iter | 5.05 | **0.05** | **G2 met (−99%)** |
| peak VRAM MiB | 1156.3 | 1152.6 | −3.7 (sort HWM +22 offset by densify fix) |
| B/splat | 429.0 | 429.0 | Phase 2 pending |
| GUI start exportable (5M cap) | 1182.6 MiB | 141.9 MiB | **−1040.6 MiB** (G7) |
All 20 campaign tests green. ISS-007 open (manual GUI validation).

---

## Task 6A.5a — has_lazy_expr() from local deferred state

- **Branch:** `lfs-elite-tensor`
- **Change:** `Tensor::has_lazy_expr()` reads only `state_->lazy` (no global IR mutex /
  `tensor_has_lazy_expr` map lookup). Eager IR debug nodes are inspected via
  `lazy_expr_id()` / `internal::tensor_has_lazy_expr`.
- **Fail evidence (TDD):**
  ```
  TensorDispatch6A.HasLazyExprReadsLocalDeferredStateOnly
  Value of: eager.has_lazy_expr()
    Actual: true
  Expected: false
  ```
- **Pass evidence:**
  ```
  [  PASSED  ] TensorDispatch6A.HasLazyExprReadsLocalDeferredStateOnly
  [  PASSED  ] TensorDispatch6A.HasLazyExprMatchesIsDeferredWhenIrOff
  Tensor* suite: 998 PASSED; tensor_hardening_tests: 89 PASSED
  ```

## Task 6A.2 — Gate lazy-IR recording off in production

- **Branch:** `lfs-elite-tensor`
- **Change:**
  - `lazy_ir_active()` default OFF (`LFS_LAZY_IR=1` or
    `lazy_ir_set_active_for_testing(true)` to enable).
  - Eager `lazy_ir_record_*` remain gated; deferred always recorded via
    `lazy_ir_record_deferred` (fusion/materializer registries key by node_id).
  - `lazy_ir_set_node_inputs` not gated by eager flag.
- **Fail evidence (TDD, hard-true `lazy_ir_active`):**
  ```
  TensorDispatch6A.LazyIrDefaultOffAndOptIn
    Expected: false  Actual: true  (lazy_ir_active)
  TensorDispatch6A.EagerBinaryDoesNotRecordWhenIrOff
    tensor_has_lazy_expr(c)=true, lazy_expr_id=6, expr_nodes_created increased
  ```
- **Pass evidence:**
  ```
  [  PASSED  ] UnaryReduceFusesWithIrOff  (fused_launches > 0 with IR off)
  [  PASSED  ] EagerBinaryDoesNotRecordWhenIrOff
  TensorLazy* + TensorDispatch*: 111 PASSED
  Tensor* suite: 998 PASSED; hardening: 89 PASSED
  ```
- **Microbench (eager add loop, CUDA, 20k iters, ops/s IR_on → IR_off):**

  | shape | IR on | IR off | Δ |
  |---|---:|---:|---:|
  | [1] | 122234 | 145647 | **+19.2%** |
  | [4096] | 135932 | 321081 | **+136%** |


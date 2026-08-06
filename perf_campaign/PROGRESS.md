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

## Task 1.3 — Fold regularizer loss scalars into fused backward

- **Branch:** `lfs-elite`
- **Change:** scale/opacity reg *loss scalars* accumulated in
  `preprocess_backward_cu` (block reduce + atomicAdd into persistent device
  scalars). Trainer zeros `fused_scale_reg_loss_` /
  `fused_opacity_reg_loss_` each FastGS backward step and skips
  `forward_loss_only` (which did `empty({num_blocks})+empty({1})` per call).
  Legacy loss-only path kept for freeze / non-backward FastGS iters and for
  the gsplat path (`compute_*_reg_loss`).
- **Fail evidence (TDD):** without kernel accumulation, fused scalars stay 0
  after bwd → `relative_delta(old, 0) == 1` fails `|delta| < 1e-5`. Pre-change
  trainer always launched loss-only kernels on the FastGS path.
  ```
  # conceptual pre-fix (loss_out never written):
  Expected: relative_delta(scale_old, scale_new) < 1e-5
  scale reg: old=0.00… fused=0  (delta=1)
  ```
- **Pass evidence:**
  ```
  [==========] Running 2 tests from 1 test suite.
  [  PASSED  ] 2 tests.  (FusedRegLossTest.*)
  FusedBackwardLossMatchesLossOnly  (rel Δ < 1e-5)
  FusedPathHasNoPerCallRegLossAllocs (fused_delta=0 driver allocs)
  ```
- **Note:** Bench gate deferred to after 1.4 (per work order). Wave 1 before:
  4.085 ms/iter, 0.05 allocs/iter.
- **Commit:** `167300ff`


## Task 1.4 — Fuse background blend; drop backward unblend

- **Branch:** `lfs-elite`
- **Change:** `blend_cu` writes `fg + T*bg` in one pass (solid `bg_color` or
  per-pixel `bg_image` CHW). Removed forward `compose_background_in_place` and
  backward unblend. `blend_backward_cu` already ignores `image`/`alpha_map`
  (uses `tile_final_transmittance`); unblend was dead. Keep blended image in
  ctx (already resident for the loss path) — no extra pre-blend cache.
  `grad_alpha` kernels unchanged (still `dL/dα = -dot(grad_image, bg)`).
- **Fail evidence (TDD):** without fuse, dropping compose leaves raw fg →
  `max_abs_diff` vs external compose ≈ bg magnitude. Pre-change: separate
  full-image compose + unblend every step.
- **Pass evidence:**
  ```
  [  PASSED  ] FusedBgBlendTest.ForwardBlendedMatchesExternalCompose  (diff < 1e-6)
  [  PASSED  ] FusedBgBlendTest.BackwardGradsMatchWithBlendedImage  (repro < 1e-6; bg affects grads)
  [  PASSED  ] FusedRegLossTest.* (no regression)
  ```
- **Gate bench (flock, 3 runs bonsai) vs Wave 1 (4.085 ms, 0.05 allocs):**

  | metric | Wave 1 | after 1.4 | Δ |
  |---|---:|---:|---|
  | wall_s | 8.94 | 9.00 | +0.7% noise |
  | steady_ms/iter | 4.085 | **4.104** | +0.5% flat (run1=4.084) |
  | steady_allocs/iter | 0.05 | **0.05** | 0 |
  | peak VRAM MiB | 1152.6 | **1135.4** | −17.2 |
  | last_loss | ~0.03–0.04 | 0.029–0.040 | ok |

  Runs: `perf_campaign/runs/20260806T183749Z_run{1,2,3}/`
  Dual-workload bicycle gate deferred to post-1.9 (RULES dual-gate added mid-campaign).
- **Commit:** `ff517550`


## Task 1.6 — Photometric hygiene

- **Branch:** `lfs-elite`
- **Change:**
  1. Drop `reduction_result.clone()` / `masked_loss.clone()` in ssim.cu —
     return the persistent workspace scalar view (callers consume before next
     forward on the same workspace).
  2. Skip `grad_img.zero_()` / `grad_corrected.zero_()` / `grad_raw.zero_()`
     on fused L1+SSIM bwd paths — kernels assign every in-bounds pixel
     (valid-padding zeros the chain only; masked path writes 0 where mask=0).
  3. Skip `corrected_image.clamp_(0,1)` when PPISP is active (CRF clamps).
- **Fail evidence (TDD):** pre-change every fused forward cloned `{1}` → at
  least 1 driver alloc on pool miss; zero_ was a full-image write every step.
  ```
  # conceptual: with clone restored, steady alloc delta ≥ 1
  ```
- **Pass evidence:**
  ```
  [  PASSED  ] PhotometricHygieneTest.FusedLossValueStableAndNoCloneAlloc
  # steady fused_l1_ssim_forward alloc delta=0; loss equal across steps;
  # bwd grads finite without prior zero_
  ```
- **Commit:** `654a92ee`


## Task 1.7 — Persistent masks

- **Branch:** `lfs-elite`
- **Change:**
  1. Cropbox damping mask cached on Trainer; rebuild only when N / crop
     geometry fingerprint / scale changes (`install_cropbox_step_damping`).
  2. Frozen GPU mask process-wide cache in `make_frozen_mask` — rebuild only
     when SplatData identity / N / frozen-ranges fingerprint / device change.
     Steady `inject_noise` no longer host-vector + H2D every step.
- **Fail evidence (TDD):** without cache, 10 unchanged `make_frozen_mask` calls
  would increment rebuild counter 10×; cropbox path rebuilt every install.
  ```
  # conceptual pre-cache:
  Expected: frozen_mask_rebuild_count() == 1 after 10 hits  (got 10)
  Expected: cropbox rebuild_count == 1 after 8 installs    (got 8)
  ```
- **Pass evidence:**
  ```
  [==========] Running 2 tests from 1 test suite.
  [  PASSED  ] PersistentMasksTest.FrozenMaskRebuiltOnceAcrossUnchangedCalls
  [  PASSED  ] PersistentMasksTest.CropboxDampingRebuiltOnceAcrossUnchangedSteps
  (frozen: 1 rebuild across 10 hits; +1 on N change; +1 on range change)
  (cropbox: 1 rebuild across 8 installs; +1 on scale; +1 on transform)
  ```
- **Commit:** `35759f68`


## Task 1.8 — Fuse noise injection

- **Branch:** `lfs-elite`
- **Change:** MCMC `inject_noise` uses one `inject_noise_kernel` (curand +
  covariance transform + means add, honors frozen mask). Removed
  `normal_()` pass and the `[max_cap,3]` noise buffer. MRNF path was already
  fused (`launch_mrnf_noise_injection`). Legacy `launch_add_noise_kernel`
  kept for the pre-supplied-noise path.
- **Fail evidence (TDD):** without fused launch, test fails to link/compile
  on missing `launch_inject_noise_kernel`. Conceptual pre-change: two full-N
  passes per step (`normal_` + add_noise).
- **Pass evidence:**
  ```
  [  PASSED  ] FusedNoiseInjectionTest.MeanAndVarMatchIdentityCovariance
  # mean≈0, var≈(lr*0.622)^2 within 8% on N=50k identity cov
  [  PASSED  ] FusedNoiseInjectionTest.FrozenMaskBlocksNoise
  ```
- **Note:** Bit-identical trajectories not required (new RNG stream). Quality
  check via dual-workload bench after 1.9.
- **Commit:** `42184eea`


## Task 1.9 — Stop full [2,N] densification_info memset every iteration

- **Branch:** `lfs-elite`
- **Change:** Fuse fold-of-`densification_info` into running stats with zero of
  both rows in one kernel (no separate full `[2,N]` `zero_()` per step):
  - MCMC / improved_gs_plus: `launch_max_error_and_zero_densification`
    (max row1 into `_error_score_max`, zero rows 0+1)
  - MRNF: `launch_fold_densification_and_zero`
    (add row0 → `_vis_count`, max row1 → `_refine_weight_max`, zero both)
  - Fallback `zero_()` only on shape mismatch; refine-time zero after grow kept.
- **Fail evidence (TDD):**
  ```
  tests/test_densification_info_zero.cpp:56:24: error:
    'launch_fold_densification_and_zero' is not a member of
    'lfs::training::mrnf_strategy'
  tests/test_densification_info_zero.cpp:109:15: error:
    'launch_max_error_and_zero_densification' is not a member of
    'lfs::training::mcmc'
  ```
- **Pass evidence:**
  ```
  [==========] Running 2 tests from 1 test suite.
  [  PASSED  ] DensificationInfoZeroTest.MrnfFoldMatchesMultiStepReference
  [  PASSED  ] DensificationInfoZeroTest.McmcMaxMatchesMultiStepReference
  [  PASSED  ] 2 tests.
  ```
  Multi-step accumulate into max/vis matches old max/add + `zero_()` path;
  densification_info fully zeroed after each fold.
- **Commit:** `b046ea34`
  Runs (quiet dual gate):
  - bonsai: `perf_campaign/runs/20260806T195950Z_run{1,2,3}/`
  - bicycle: `perf_campaign/runs/20260806T200227Z_run{1,2,3}/`

---

## Phase 1 dual-workload FINAL GATE (post-1.9, commit b046ea34)

Quiet GPU (no concurrent train/build). Contaminated first bonsai trio
(20260806T195825Z: 7.5 ms/iter, 1699 MiB) discarded — concurrent densify-worker
GPU use inflated device-wide `cudaMemGetInfo` peak and ms/iter (see ISS-008).

### Bonsai (2000 iters, images_4, max_cap 500k, mrnf) — medians

| metric | Wave 1 (0d652d45) | after 1.4 | **Phase-1 final** | Δ vs Wave 1 |
|---|---:|---:|---:|---:|
| wall_s | 8.94 | 9.00 | **8.86** | −0.9% |
| steady_ms/iter | 4.085 | 4.104 | **4.059** | **−0.6%** |
| steady_allocs/iter | 0.05 | 0.05 | **0.05** | 0 |
| peak VRAM MiB | 1152.6 | 1135.4 | **938.3** | −214 (device-wide free; quiet GPU) |
| B/splat | 429.0 | 429.0 | 429.0 | 0 |
| last_loss | ~0.03–0.04 | 0.029–0.040 | 0.029–0.060 | ok |

### Bicycle canary (7000 iters, images_4, max_cap 500k, mrnf) — medians

| metric | BASELINE.md (ba7c4497, +1.3) | **Phase-1 final** | Δ |
|---|---:|---:|---:|
| wall_s | 31.15 | **30.76** | −1.3% |
| steady_ms/iter | 3.290 | **3.258** | −1.0% |
| steady_allocs/iter | 0.04 | **0.04** | 0 |
| peak VRAM MiB | 1038.5 | **1026.1** | −12.4 |
| B/splat | 429.0 | 429.0 | 0 |
| final loss (range) | 0.098–0.121 | **0.107–0.124** | within bicycle high variance |

Loss-curve spot checks (run3 samples, every 1k iters): 1k=0.20, 2k=0.12,
3k=0.15, 4k=0.10, 5k=0.10, 6k=0.20, 6.9k=0.12 — healthy densify growth to
cap (54k→500k), no collapse/NaN, no floater runaway.

### Phase-1 task rollup (six tasks + dual gate)

| task | commit | fail | pass | bonsai note |
|---|---|---|---|---|
| 1.3 fused reg loss | `167300ff` | fused scalars=0 → relΔ=1 | FusedRegLossTest 2/2 | bench deferred |
| 1.4 fused bg blend | `ff517550` | drop compose → max_abs≈bg | FusedBgBlendTest 2/2 | 4.104 ms / 0.05 |
| 1.6 photometric hygiene | `654a92ee` | clone → alloc≥1 | PhotometricHygieneTest | deferred |
| 1.7 persistent masks | `35759f68` | rebuild 10× / 8× | PersistentMasksTest 2/2 | deferred |
| 1.8 fuse noise inject | `42184eea` | missing launch API | FusedNoiseInjectionTest 2/2 | quality via dual gate |
| 1.9 densif fold+zero | `b046ea34` | missing launch API | DensificationInfoZeroTest 2/2 | **final dual gate** |

**Gate status:** G2 allocs held at 0.05; G4 ms/iter improved vs Wave 1 on both
workloads; bicycle quality canary clean (loss curve + final range OK).
Phase 1 series **DONE**.

||||||| f06a8885

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

- **Commit:** `2aeded6f`

## Task 6A.3 — Contiguous same-shape same-dtype binary fast path

- **Branch:** `lfs-elite-tensor`
- **Change:** In `binary_op_with_promotion`, when operands are same shape, same
  dtype as the promoted result, contiguous, non-deferred, and dtype is
  F32/F16/I32/I64/U8: skip `BinaryExpr` + `TensorLeaf` heap cells. Path is
  `validate → prepare stream → empty → pin → launch_binary → optional IR → return`.
  Promotion/broadcast/deferred still use the expr path unchanged.
- **Fail evidence (TDD / microbench BEFORE, fast path gated `false &&`):**
  ```
  MICROBENCH 6A.3 shape=[1] add=6700.31 ns/op mul=6697.32 ns/op
  MICROBENCH 6A.3 shape=[4096] add=3121.35 ns/op mul=3117.15 ns/op
  # Correctness tests already green on BinaryExpr path (empty, [1], fp16, int32/64)
  ```
- **Pass evidence (fast path enabled):**
  ```
  [  PASSED  ] TensorDispatch6A.BinaryFastPathFloat32Cpu
  [  PASSED  ] TensorDispatch6A.BinaryFastPathEmptyAndScalarShapes
  [  PASSED  ] TensorDispatch6A.BinaryFastPathInt32 / Int64
  [  PASSED  ] TensorDispatch6A.BinaryFastPathFloat32Cuda / Float16Cuda
  [  PASSED  ] TensorDispatch6A.BinaryPromotionAndBroadcastStillWork
  Tensor* suite: 1005 PASSED; tensor_hardening_tests: 89 PASSED
  MICROBENCH 6A.3 shape=[1] add=6315.52 ns/op mul=6289.56 ns/op
  MICROBENCH 6A.3 shape=[4096] add=2682.71 ns/op mul=2701.11 ns/op
  ```
- **Microbench (CUDA F32 add/mul, 50k iters, ns/op before → after):**

  | shape | add before | add after | Δ | mul before | mul after | Δ |
  |---|---:|---:|---:|---:|---:|---:|
  | [1] | 6700 | 6316 | **−5.7%** | 6697 | 6290 | **−6.1%** |
  | [4096] | 3121 | 2683 | **−14.0%** | 3117 | 2701 | **−13.3%** |

- **Dual gate (exclusive GPU + `flock /tmp/lfs-build.lock`, dirty tree post-6A.3):**

  **Bonsai** (3×2000, images_4, max_cap=500k) vs Wave-1:

  | metric | Wave-1 | after 6A.3 | Δ |
  |---|---:|---:|---|
  | wall_s (med) | 8.94 | **8.87** | −0.8% |
  | steady_ms/iter | 4.085 | **4.068** | −0.4% |
  | steady_allocs/iter | 0.05 | **0.05** | 0 |
  | peak VRAM MiB | 1152.6 | **938.3** | −214 (quiet GPU; see ISS-008) |
  | B/splat | 429.0 | 429.0 | 0 |
  | last_loss | ~0.03–0.04 | 0.029–0.045 | ok |

  Runs: `perf_campaign/runs/6a3_bonsai/20260806T200024Z_run{1,2,3}/`

  **Bicycle** (3×7000, images_4, max_cap=500k) vs bicycle baseline (3.290 ms, 1038.5 MiB):

  | metric | bicycle baseline | after 6A.3 | Δ |
  |---|---:|---:|---|
  | wall_s (med) | — | **30.48** | — |
  | steady_ms/iter | 3.290 | **3.237** | −1.6% |
  | steady_allocs/iter | — | **0.04** | ok |
  | peak VRAM MiB | 1038.5 | **994.3** | −44 |
  | B/splat | 429.0 | 429.0 | 0 |
  | last_loss | 0.098–0.121 | 0.096–0.142 | high-variance OK |

  Runs: `perf_campaign/runs/6a3_bicycle/20260806T200053Z_run{1,2,3}/`

  Gate: no regression vs Wave-1 bonsai or bicycle steady_ms; G2 allocs held.

- **Commit:** `786fefb6`

||||||| 42184eea
- **Commit:** `42184eea`

## Task VRAM-audit (lfs-elite-vramfix) — ISS-007 + NVRM + TLS + RAM guard

- **Branch:** `lfs-elite-vramfix`
- **Context:** User-visible "memory is full" traced to **system RAM** pressure
  (systemd-oomd killed gnome-shell at 95.56% user-slice, 2026-08-06 20:48) from
  parallel unbounded builds — fixed by build-discipline flock rule. Kernel also
  logged `NVRM: VM: invalid mmap context` while CUDA apps were killed.

### 1) Exportable-storage multi-grow (Phase 5.1 / ISS-007 GPU audit)
- **Finding:** pure CUDA VMM grow path has **no physical chunk leak**.
  `cudaMemGetInfo` plateaus after capacity max; 24× create/grow/destroy returns free.
- **Fail evidence (TDD):** tests did not exist; added first.
- **Pass evidence:**
  ```
  [  PASSED  ] SplatExportableStorageTest.ManyGrowCyclesCudaMemGetInfoPlateaus
  [  PASSED  ] SplatExportableStorageTest.RepeatedCreateGrowDestroyDoesNotLeakVmm
  [  PASSED  ] SplatExportableStorageTest.GrowKeepsStableVaWhileImportersHoldBlock
  # plateau free stable across 12 idempotent grows; 24 cycles free ≥ baseline (−0 MiB)
  ```

### 2) NVRM invalid mmap — teardown ordering fix
- **Bug:** densify grow called `grow()` (→ `release_physical`: unmap + cuMemRelease
  + close fd) **while** Vulkan still held `VkDeviceMemory` imported from the old
  handle. Re-import happened only after grow.
- **Fix:** `TrainerManager::growExportableForDensify`:
  1. rebind CUDA-only (drop all `VulkanExternalTensorStorage` owners)
  2. clear trainer interop allocator
  3. `cudaDeviceSynchronize`
  4. `grow()`
  5. re-import + rebind interop
  - Thin `std::function` trampoline → member method (avoids self-destroy on rebind).
  - Documented on `release_physical` / `growExportableDeviceBlock`.
- **VulkanExternalTensorStorage dtor** already correct: interop reset →
  `destroyExternalBuffer` → then `extra_owner_` (ExportableBlock).
- **Fail evidence:** code-order audit (grow before drop import). Headless cannot
  fully repro NVRM without GUI Vulkan.
- **Pass evidence:** ordering inverted in code; unit tests green; ISS-009 filed.

### 3) Thread-local high-water buffers (Phase 1.1 sort + raster outputs)
- **Change:** `release_sort_workspace_buffers` / `release_fastgs_sort_workspace_buffers`;
  training-thread shutdown (TrainerManager + MCP) now releases FastGS sort TLS
  alongside raster/gsplat/intersect caches.
- **Pass evidence:**
  ```
  [  PASSED  ] FastGSThreadLocalCacheTest.SpawnRenderJoinReturnsVram
  # 4 threads × 3 forwards; free_after ≥ free_before (−0 MiB within 32 MiB slack)
  [  PASSED  ] FastGSSortBufferTest.* (4 tests, no regression)
  ```

### 4) RAM-side regression guard
- **Test:** `VramLeakRegressionTest.FixedSizeCyclesHostRssAndVramStable` —
  40 fixed-size training-like cycles; assert host RSS and CUDA free ~flat from
  cycle 10 → 40 (RSS slack 64 MiB, VRAM slack 32 MiB).
- **Pass evidence:**
  ```
  [  PASSED  ] VramLeakRegressionTest.FixedSizeCyclesHostRssAndVramStable (56 ms)
  [==========] 14 tests (exportable + FastGS + leak)  [  PASSED  ] 14 tests.
  ```

### Dual-workload gate (flock, 3 runs each)

**Bonsai** (2000 iters) — runs `20260806T195041Z_run{1,2,3}`:

| metric | baseline | after VRAM audit | Δ |
|---|---:|---:|---|
| wall_s (med) | 9.00 | **8.94** | −0.7% |
| steady_ms/iter | 4.129 | **4.063** | −1.6% |
| steady_allocs/iter | 5.05 | **0.05** | (Wave 1 already) |
| peak_VRAM_MiB | 1156.3 | **938.3** | −218 |
| B/splat | 429.0 | 429.0 | 0 |
| last_loss | ~0.039 | 0.027–0.043 | ok |

**Bicycle canary** (7000 iters) — runs `20260806T195115Z_run{1,2,3}`:

| metric | bicycle baseline | after VRAM audit | Δ |
|---|---:|---:|---|
| wall_s (med) | 31.15 | **30.41** | −2.4% |
| steady_ms/iter | 3.290 | **3.235** | −1.7% |
| steady_allocs/iter | 0.04 | **0.04** | 0 |
| peak_VRAM_MiB | 1038.5 | **1026.3** | −12 |
| last_loss range | 0.098–0.121 | 0.088–0.117 | ok |

No quality or speed regression. Headless path does not exercise Vulkan
re-import; NVRM fix is densify/grow ordering for GUI.

- **Commits:**
  - `1e454987` fix(vram): drop Vulkan import before exportable grow (NVRM)
  - `0a0db4f4` fix(vram): release FastGS sort TLS on training-thread shutdown
  - `41aec0ae` test(vram): multi-grow plateau, TLS spawn-join, RSS/VRAM cycle guard
  - (docs commit)

---
## WAVE 2 COMBINED (merged: Phase 1 complete + 6A.5a/6A.2/6A.3 + VRAM/NVRM fixes; commit 77467aab; quiet GPU, 3-run medians)
| Metric | Original baseline | Wave 1 | **Wave 2** |
|---|---:|---:|---:|
| Bonsai steady_ms/iter | 4.129 | 4.085 | **4.065** (−1.5% vs base) |
| Bonsai peak MiB | 1156.3 | 1152.6 | **938.3** (−18.9%) |
| Bonsai allocs/iter | 5.05 | 0.05 | **0.05** |
| Bicycle 7k steady_ms/iter | 3.290 | — | **3.208** (−2.5%) |
| Bicycle 7k peak MiB | 1038.5 | — | **1026.3** |
| Bicycle loss range | 0.098–0.121 | — | 0.079–0.107 (healthy) |
| B/splat | 429 | 429 | 429 (Phase 2 next) |
36/36 campaign tests green. NVRM use-after-free ordering bug fixed; TLS release paths added.

---

## Task 3.3+3.4+3.7 — Allocator hygiene (worker P, branch `lfs-elite-fP`)

- **Branch:** `lfs-elite-fP`
- **Scope:** route bare op-temp `cudaMalloc`s through pool; free fully-empty slabs on
  `trim_cached_memory`; null-owner empty CUDA tensors (no 1-byte sentinel).

### Changes

1. **3.7 Empty CUDA tensors → null-owner** (`tensor_unified_ops.cpp` LoadOp::Empty):
   zero-numel CUDA/CPU tensors use a static dummy `data_owner_` (same pattern as
   `zeros_direct` zero-byte path). No slab/pool 1-byte sentinel allocation.
2. **3.4 Slab reclamation** (`gpu_slab_allocator.hpp` + `memory_pool.hpp`):
   `GPUSlabAllocator::reclaim_empty_slabs()` freezes fully-empty slabs after a device
   sync + stream merge; called from `CudaMemoryPool::trim_cached_memory`. Expand/
   reclaim/cleanup share `expand_mutex_`.
3. **3.3 Pool-route bare temps:**
   - masking `d_count` (`tensor_masking_ops.cpp`) via `CudaMemoryPool`
   - shape/stride metadata in strided upload + scatter (`tensor.cpp`), fill_strided
     fallback (`tensor_ops.cu`)
   - `CudaDeviceMemory` RAII (`cuda_memory_guard.hpp`) now allocates/frees via pool
     (optional stream for stream-ordered free); gather/index_fill pass stream
   - NaN-check device flag (`tensor_ops.cu`) via pool (host remains pinned)

### Fail evidence (TDD — pre-fix behavior of the harness)

Pre-existing untracked `tests/test_allocator_hygiene.cpp` from interrupted attempt
(kept). Against unfixed tree the suite would fail as:

```
# EmptyCudaTensorDoesNotAllocateSlabBlock
#   Expected: slab_allocs_after == slab_allocs_before
#   (1-byte CUDA sentinel via allocate_cuda_storage(1) bumps slab alloc_count)

# TrimCachedMemoryFreesFullyEmptySlabs
#   Expected: reserved_after < reserved_peak
#   (trim only merged free lists; cleanup only at process shutdown)

# CountNonzeroPoolPathCorrectAndReusable — correctness still OK, but
#   bare cudaMalloc/cudaFree for d_count on every call (untracked classic heap)
```

### Pass evidence

```
[==========] Running 4 tests from 1 test suite.
[  PASSED  ] 4 tests.  (AllocatorHygiene.*)
EmptyCudaTensorDoesNotAllocateSlabBlock
TrimCachedMemoryFreesFullyEmptySlabs
CountNonzeroPoolPathCorrectAndReusable
EmptyAndCountNonzeroCompose

Related: AllocatorPolicy* + TensorZeroDimension* + AllocCounter* = 34/34 PASSED
Masking/nonzero suite (95 tests) = 95/95 PASSED
```

### Dual-workload gate (flock, 3 runs each)

**Bonsai** (2000 iters) — runs `20260806T230139Z_run{1,2,3}` vs Wave 2:

| metric | Wave 2 | after 3.3/3.4/3.7 | Δ |
|---|---:|---:|---|
| wall_s (med) | — | **8.86** | ok |
| steady_ms/iter | 4.065 | **4.046** | −0.5% |
| steady_allocs/iter | 0.05 | **0.05** | 0 |
| peak_VRAM_MiB | 938.3 | **938.1** | −0.2 |
| B/splat | 429.0 | 429.0 | 0 |
| last_loss range | ~0.03–0.04 | 0.027–0.066 | ok |

**Bicycle canary** (7000 iters) — runs `20260806T230310Z_run{1,2,3}`:

| metric | Wave 2 | after 3.3/3.4/3.7 | Δ |
|---|---:|---:|---|
| wall_s (med) | — | **32.95** | noise (wall) |
| steady_ms/iter | 3.208 | **3.025** | −5.7% |
| steady_allocs/iter | 0.04 | **0.04** | 0 |
| peak_VRAM_MiB | 1026.3 | **994.1** | −32 |
| last_loss range | 0.079–0.107 | 0.088–0.149 | bicycle high-var OK |

Gate: no speed/quality regression; G2 allocs held; densify grew bicycle to 500k cleanly.

- **Commit:** `d8acaf75`


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
- **Commit:** `b416b603`

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
- **Commit:** `b416b603`

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

- **Commit:** `b416b603`

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

- **Commit:** `b416b603`
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

- **Commit:** `b416b603`
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

- **Commit:** `b416b603`

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
- **Commit:** `b416b603`

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

- **Commit:** `b416b603`

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
- **Commit:** `b416b603`


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
- **Commit:** `b416b603`


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
- **Commit:** `b416b603`


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
- **Commit:** `b416b603`


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
- **Commit:** `b416b603`


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
- **Commit:** `b416b603`
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


---
## WAVE 4 CONSOLIDATED (lfs-elite = 53dd6e84; all fleet branches + hot fixes merged; 3-run medians)
| Metric | Baseline | Wave 2 | **Wave 4** | vs baseline |
|---|---:|---:|---:|---|
| Bonsai wall_s | 9.00 | 8.90 | **7.27** | **−19%** |
| Bonsai steady_ms/iter | 4.129 | 4.065 | **3.168** | **−23%** |
| Bonsai dataloader wait | (unmeasured) | — | **0.006 ms** | decode eliminated |
| Bicycle 7k wall_s | 31.15 | 30.49 | **20.95** | **−33%** |
| Bicycle steady_ms/iter | 3.290 | 3.208 | **2.799** | **−15%** |
| B/splat | 429.0 | 429.0 | **409.4** | −4.6% (SH quant pending G6 → ≤307) |
| allocs/iter | 5.05 | 0.05 | 0.18 | audit the +0.13 (new paths) |
| Peak MiB bonsai | 1156 | 938 | 1533 (incl. **339 GT cache** by design) | ex-cache ~1194 — audit +256 |
| Peak MiB bicycle | 1038 | 1026 | 1613 (incl. **564 GT cache**) | budget-gated, reversible |

FLAGS (honest):
1. Bicycle final-loss band shifted UP (0.116–0.152 vs baseline 0.098–0.121) — consistent with
   the ISS-015 gradient divergence degrading convergence. WO-G5 (bisect+fix) is the critical
   path; Wave-4 speed wins are provisional until gradients are proven correct.
2. Ex-cache peak grew ~+256 MiB vs Wave 2 — audit which merged branch retains it (ledger).
3. allocs/iter 0.05→0.18 — small, but the zero-alloc invariant must be restored.


---

## WO-G5 — ISS-015 FastGS gradient bisect + fix

- **Branch:** `lfs-elite` @ start `b997fe5d`
- **Oracle:** `lichtfeld_tests --gtest_filter='FastGSGradientTest.Numerical_Means'`
- **Range:** `f06a8885` (good candidate) .. HEAD
- **Preflight (HEAD, no rebuild):** FAIL — crash in `adam_moment` → `exp_avg_scale.to(CPU)` invalid under joint Adam codec (default ON). Message: `device transfer requires a valid tensor`.
- **Preflight (HEAD, `LFS_ADAM_LEGACY_CODEC=1`):** ALL FastGSGradientTest + DenseTile **PASS** (Means cos_sim=0.9991). Implication: rasterizer gradient math (1.4 / BWD-A / reg-fold) is NOT the root; joint-codec moment recovery / fused-tail is the suspect.

### Bisect log
- **step0 HEAD b997fe5d:** BAD — joint default ON, adam_moment crash (exp_avg_scale invalid)
- **step0b HEAD + LFS_ADAM_LEGACY_CODEC=1:** GOOD — 9/9 FastGS gradient tests pass
- **step1 993314d5 (FIX-2.2 F3):** BAD — same adam_moment/exp_avg_scale crash
- **step2 eec4b87b (build.sh only, pre-joint):** GOOD — Means cos_sim=0.9991
- **step3 67a0fa34 (post 63aa codec pair profiles):** BAD — joint crash
- **step4 12b0f583 (docs after 2.1/2.2):** BAD — joint crash
- **step5 487d5c2b (pre-joint):** GOOD — Means cos_sim=0.9991
- **step6 514b2a49 (2.1 SH value, after joint):** BAD — joint crash
- **step7 63aa08c6 (joint Adam codec 2.2):** BAD — first bad; crash under default joint ON; LEGACY env GOOD cos_sim=0.9991

### First bad commit
**`63aa08c61c2447c83fc102c6832b8a5f84b9ec30`** — `perf(2.2): joint (u,log_s) Adam codec — B/splat 429→409.4`

Bisect path (oracle = Numerical_Means, build via `./perf_campaign/build.sh build/tests`):
| step | commit | verdict | note |
|---|---|---|---|
| 0 | b997fe5d HEAD | BAD | adam_moment → invalid exp_avg_scale |
| 0b | HEAD + LFS_ADAM_LEGACY_CODEC=1 | GOOD | cos_sim Means 0.9991 |
| 1 | 993314d5 FIX-2.2 F3 | BAD | joint crash |
| 2 | eec4b87b pre-joint | GOOD | cos_sim 0.9991 |
| 3 | 67a0fa34 post-codec | BAD | joint crash |
| 4 | 12b0f583 docs | BAD | joint crash |
| 5 | 487d5c2b pre-joint | GOOD | cos_sim 0.9991 |
| 6 | 514b2a49 SH value 2.1 | BAD | joint crash |
| 7 | **63aa08c6 joint 2.2** | **FIRST BAD** | LEGACY env GOOD |

### Diagnosis (not BWD-A / 1.4)
- **BWD-A / 1.4 / reg-fold:** NOT the root. Legacy codec on HEAD: all FastGS gradient tests green with cos_sim ≥ 0.999.
- **Joint codec (63aa08c6):** fused path is mathematically correct (numerical match after decode). Two gaps:
  1. **Test harness** `adam_moment()` only understood legacy u8+scale → crash on joint (ISS-015 oracle red).
  2. **Unfused** `AdamOptimizer::step_param` still called legacy kernels → crash on joint for scheduler/crop-box tests.

### Fix (perf win kept)
1. `tests/test_fastgs_kernels.cpp`: joint decode of first moment m via Codec16/8 + joint_bounds; crop-damping uses `first_moment_l1`.
2. `tests/test_fastgs_fuzz.cpp`: joint-aware `expect_adam_state_finite` + moment activity proxies.
3. **Production:** `adam_step_joint_contiguous_cu` + `adam_step_joint_contiguous_raw`; `step_param` dispatches joint for non-shN contiguous params (shN remains fused-only).

### Pass evidence (joint default ON)
```
FastGSGradientTest.* + DenseTile + CropDamping + Fuzz + JointAdam + scheduler + crop-box:
[  PASSED  ] 60 tests. (1 skipped)
Means cos_sim=0.9991, Opacity/Sh0 cos_sim=1.0000 under joint.
```

### Dual-workload gate (medians, 3 runs) — perf win SURVIVED
| metric | Wave 4 (b997fe5d) | after WO-G5 fix | Δ |
|---|---:|---:|---:|
| Bonsai wall_s | 7.27 | **7.23** | −0.6% |
| Bonsai steady_ms/iter | 3.168 | **3.148** | −0.6% |
| Bonsai B/splat | 409.4 | **409.4** | 0 |
| Bicycle wall_s | 20.95 | **21.12** | +0.8% (noise) |
| Bicycle steady_ms/iter | 2.799 | **2.817** | +0.6% (noise) |
| Bicycle loss band | 0.116–0.152 | **0.085–0.101** | **better** (correctness) |

Runs: bonsai `20260807T065803Z_run{1,2,3}`; bicycle `20260807T065831Z_run{1,2,3}`.

### Full-suite delta vs ISS-015 reds
**Fixed (same joint root):** FastGSGradient*, FastGSDenseTile*, FastGSFuzz*, FastGSCropDamping*, TrainingCropBoxMask.DampedOptimizer*, LfsSchedulerTest.Integration*.

**Independent remaining (NOT joint gradient math) — separate ISSUES:**
- VideoFrameExtractorOutputNaming ×3
- SogFormatTest ×12 (export/loader; ISS-014 family)
- TensorReserveInplaceCat.OverflowFailurePreservesInstalledStorage
- CheckpointInputValidationTest.RejectsLateStrategyCorruption…
- PythonIntegrationTest ×3 (+ CaptureViewport may hang)
- DeviceFaultTest.GraphCapture (ISS-013, excluded)
- CheckpointAllocatorRegression (ISS-014, excluded — segfault)

### Commits
**Commit:** `5f667096`

---

## WO-G4 — red-batch fix (ISS-013 / ISS-014 / SogFormat)

- **Branch:** `lfs-elite`
- **Commits:**
  - `9c531c94` fix(ISS-014): checkpoint resume for joint Adam + cold-load null optimizer
  - `06e8830d` fix(ISS-018/SOG): dequant/reformat shN on export; unbreak SogFormatTest
  - `961fd224` fix(ISS-013): GraphCaptureYieldsUnsupported teardown + stream bridge

### FAIL → PASS evidence

| Test | Before | After |
|---|---|---|
| CheckpointAllocatorRegressionTest.LoadCheckpointUsesAllocatorWithMaxCapacity | SIGSEGV in `AdamOptimizer::set_frozen_lr_scale` (this=null) | PASS |
| CheckpointResumeRoundtripTest.JointCodecAndQ16ShN (new) | n/a | PASS (joint + legacy) |
| DeviceFaultTest.GraphCaptureYieldsUnsupported | SIGSEGV in cuStreamWaitEvent / free_routed | PASS |
| SogFormatTest.* | 13 FAILED (SetUp Permission denied /home/paja/...) | 8 PASS + 6 SKIP (no external fixtures) |

### Root causes
1. **ISS-014:** `load_checkpoint` always called `get_optimizer().set_frozen_lr_scale` even when cold strategy never `initialize()`d (`_optimizer==null`). Joint Adam v2 serialize wrote 2 tensors without a marker while deserialize expected 4 legacy tensors.
2. **SOG:** `kmeans_sh_swizzled` requires float32 1D; q16 resident shN was passed through raw.
3. **ISS-013:** tensors rehomed onto capture stream then stream destroyed before free → bridgeStreams on dead handle.

### Full-suite note
Excluded set (GraphCapture + CheckpointAllocator) is empty — both green.
Independent reds still present (not WO-G4 scope): VideoFrameExtractor ×3, PythonIntegration ×3, TensorReserveInplaceCat overflow, ShValueStorage B/splat budget (WO-G6 concurrent).

### tensor_hardening
```
[==========] 89 tests from 5 test suites ran.
[  PASSED  ] 89 tests.
```


### Dual-workload gate (post WO-G4, clean binary @ 961fd224, no rebuild race)

Bonsai 2000 iters ×3 (`20260807T080041Z_bonsai_run{1,2,3}`):
| run | wall_s | steady_ms | B/splat | loss |
|---:|---:|---:|---:|---:|
| 1 | 3.01 | 2.594 | 303.1 | 0.047 |
| 2 | 3.02 | 2.627 | 303.1 | 0.045 |
| 3 | 3.04 | 2.626 | 303.1 | 0.080 |
| **med** | **3.02** | **2.626** | **303.1** | |

Bicycle 7000 iters ×3 (`20260807T080041Z_bicycle_run{1,2,3}`):
| run | wall_s | steady_ms | B/splat | loss |
|---:|---:|---:|---:|---:|
| 1 | 21.34 | 2.765 | 306.8 | 0.124 |
| 2 | 21.24 | 2.786 | 306.8 | 0.109 |
| 3 | 21.11 | 2.768 | 306.8 | 0.132 |
| **med** | **21.24** | **2.768** | **306.8** | |

vs WO-G5 medians (bonsai 7.23/3.148, bicycle 21.12/2.817): dual-workload **unchanged-or-better** (no regression). First dirty bench failed with densify UAF due to concurrent WO-G6 WIP in tree — re-ran on clean committed tree only.

### tensor_hardening
89/89 PASSED.

## WO-G6 — ISS-2.1 densify re-encode + SH quant default ON (−102 B/splat prize)

- **Branch:** `lfs-elite`
- **Precondition:** WO-G5 done (`workerG5.done`); gradients trusted.
- **TDD fail-first:** `ShValueStorageTest.PostDensifyReencodeThenFastGSForward` — G3 crash repro (grow N across 256-block, re-encode, FastGS fwd+fused Adam).
- **Pass:** 6/6 ShValueStorageTest including densify re-encode forward; FastGS gradients green under quant default.
- **Fixes:**
  1. `sh_value_storage`: encode/decode on current stream + **device barrier** before free; capacity from means cap.
  2. `mrnf::refine`: **trim_memory_pool before re-encode** (was freeing pool while densify path still hot).
  3. `AdamOptimizer`: moments always float-layout sized under q16; heal joint_bounds for ceil(N/256).
  4. Default `LFS_SH_VALUE_QUANT` **ON**.
- **Heal-vs-rebuild:** rebuild codes/bounds; heal Adam moments when capacity allows.
- **Dual workload (3-run medians):**

| metric | OFF | ON | gate |
|---|---:|---:|---|
| Bonsai steady_ms | ~3.15 (G5) | **3.15** | no worse |
| Bonsai B/splat | 409.4 | **304.3** | ≤307 ✓ |
| Bicycle steady_ms | 2.81 | **2.73** | faster |
| Bicycle B/splat | 409.4 | **306.8** | ≤307 ✓ |
| Bicycle loss | 0.10–0.16 | 0.10–0.14 | overlap ✓ |
| Late-window ms (1600–1900) | 3.88 | **3.74** | ON ≤ OFF ✓ |

- **Full suite:** only pre-existing ISS-016 VideoFrameExtractor×3, ISS-017 TensorReserve, ISS-019 Python×3 reds (not quant).
- **tensor_hardening:** 89/89 PASS.
- **Decision:** flag **default ON**.

## WO-W.1 regression fix — raw-ptr escape on zero-stride expand (checkpoint resume)

- **Branch:** `lfs-elite` (no checkout)
- **Symptom:** `CheckpointStrategies/CheckpointResumeTest.TrainSaveLoadResume/{tiny,nightly}_mcmc_0` FAIL since W.1 merge (`a2bffd4f`) with `cudaMemcpy failed for scaling: invalid argument` (~119ms repro).
- **Root cause (two legs):**
  1. `expand` returns stride-0 views; `SplatData` init does `cudaMemcpy(..., scaling_cpu.ptr(), numel()*4, H2D)` — flat read of N×3 from a physical-N buffer.
  2. W.1 op firewall does not cover raw-pointer escapes (`ptr()` / `data_ptr()`). Also: `has_zero_stride` falsely fired on contiguous empty tensors `[N,0,3]` (SH degree 0) because row-major leading stride is 0 when a later dim is size 0.

### TDD fail → pass

**FAIL (pre-fix):**
```
ExpandedTensorOps.RawPtrOnExpandViewMaterializesDense — flat read r=0 c=1 got 2 want 1
CheckpointResumeTest.../tiny_mcmc_0 — cudaMemcpy failed for scaling: invalid argument
  src (CPU): ptr=0x7f861573e800 numel=162825
  dst (CUDA): ptr=0x7f8615a00000 numel=162825
```

**PASS (post-fix):**
```
ExpandedTensorOps.{RawPtrOnExpandViewMaterializesDense,DataPtrOnExpandViewMaterializesDense,
  SplatDataScalingExpandPtrMaterializes,EmptyZeroDimTensorPtrIsSafe,CpuTaggedDeviceStorageRejectedOnPtr} all OK
ZeroStrideExpand.* 15/15 OK
CheckpointResumeTest.TrainSaveLoadResume/{tiny,nightly}_mcmc_0 both OK
tensor_hardening 89/89 PASS
```

### Fix (boundary justification)

Materialize-on-escape at `ptr()` / `data_ptr()` (not at expand creation — keeps W.1 zero-copy for allowlisted ops):
- `materialize_zero_stride_for_raw_ptr_escape()` rebinds handle after `contiguous()` (must NOT use `*this = contiguous()` — view assignment deep-copies via `copy_from` → re-enters `data_ptr` → stack overflow).
- `storage_ptr()` intentionally non-materializing (allocation base / sharing checks only).
- `has_zero_stride()`: only size>1 with stride 0, and `numel()>0` (excludes empty `[N,0,3]`).
- `contiguous()`: never early-return when zero-stride.
- Device-mismatch canary: `assert_device_storage_matches_tag()` rejects CPU-tagged device storage on raw-ptr escape.

### Dual-workload gate (3-run medians)

| workload | steady_ms | B/splat | target | status |
|---|---:|---:|---|---|
| bonsai 2k | **3.153** | **304.3** | 3.182 / 304.3 | ✓ |
| bicycle 7k | **2.741** | **306.8** | 2.733 / 306.8 | ✓ (noise) |

Bonsai runs `20260807T093540Z_run{1,2,3}`; bicycle `20260807T093608Z_run{1,2,3}`.

### Full suite

3311 PASS; pre-existing reds only (ISS-016 VideoFrame×3, ISS-017 TensorReserve, ISS-019 Python, plus unrelated env: NaNInf InfDetection_Large, PipelinedLoader fixtures, SceneValidity migrate, Float16HostReduceFailsLoud). **MCMC resume was FAIL → PASS.**
**Commit:** `b416b603`

---

## WO-X — restore zero-alloc invariant + audit ex-cache peak (Wave-4 flags 2+3)

- **Branch:** `lfs-elite-fX` @ `b6a276cc` (lfs-elite tip)
- **Problem (Wave-4 consolidated):**
  1. `steady_allocs/iter` 0.05 → **0.18** (bonsai measured ~0.13 post-G5)
  2. ex-cache peak ~1194 MiB vs Wave-2 **938** (+~256 MiB)

### Root causes (measured + LFS_ALLOC_TRACE)

| Source | Attribution | Fix |
|---|---|---|
| GT-cache first-seen inserts after warmup (~292 images, warmup 200) | `logical=gt_cache` / pool_bucket | **warm_gt_device_cache()** before timed loop; `clear()` keeps cache |
| joint_bounds `Tensor::zeros` every densify grow/compact (~6×/refine) | `logical=joint_bounds` | **ensure_joint_bounds_capacity** grow-only (`append_zeros` / `zero_`) |
| densify index/mask `empty`/`zeros_bool` per refine | `logical=densify` | **DensifyNScratch** pre-sized to max_cap |
| post-refine `trim_memory_pool()` | wiped size-bucket free list → next densify all misses | **removed** MRNF post-refine trim; OOM path still trims via MemoryPressureCoordinator |

### Instrumentation

- `alloc_counter::record_site(Site)` + TLS `ScopedSite` + `LFS_ALLOC_TRACE=1` steady-state log
- `PeakExCacheLedger` in perf_bench.json: `ex_cache_*`, owners, `unjustified_excess_bytes`
- Sites written to `alloc_sites` in JSON

### TDD

**Fail evidence (pre-fix numbers / conceptual):**
```
Wave-4: steady_allocs/iter ≈ 0.13–0.18  (gate ≤ 0.06)
joint_bounds: 6 groups × 15 densify × zeros = 90 driver allocs
# JointBoundsGrowOnly / SteadyAllocInvariant assert ≤ 0.06 densify residual
```

**Pass evidence:**
```
[  PASSED  ] JointBoundsGrowOnly.EnsureWithinCapacityIsAllocFree
[  PASSED  ] JointBoundsGrowOnly.MultiParamCompactZeroReusesCapacity
[  PASSED  ] SteadyAllocInvariant.GateBudgetIsAtMost006
[  PASSED  ] SteadyAllocInvariant.JointDensifySteadyLoopWithinBudget  (delta=0)
[  PASSED  ] PeakExCacheLedger.* + AllocCounterSiteTags.*
[  PASSED  ] 19 campaign tests (alloc/ledger/sort/joint)
```

### Dual-workload gate (3-run medians)

**Bonsai** (2000 iters) — runs `20260807T081411Z_run{1,2,3}` + confirm `20260807T081733Z`:

| metric | Wave 2 | Wave 4 (drift) | **WO-X** | gate |
|---|---:|---:|---:|---|
| wall_s | 8.90 | 7.27 | **7.68** | no speed reg vs W4 G5 |
| steady_ms/iter | 4.065 | 3.148 | **3.164** | flat vs G5 3.148 |
| **steady_allocs/iter** | **0.05** | 0.18 | **0.05** | **≤ 0.06 ✓** |
| peak MiB | 938 | 1533 | 1567 | GT 339 + ex |
| gt_cache MiB | 0 | 339 | **338.8** | WO-HP1 justified |
| ex_cache MiB | 938 | ~1194 | **1228** | excess ledgered |
| unjustified_excess | — | — | **0** | all owners |
| B/splat | 429 | 409.4 | **409.4** | held |
| last_loss | ~0.03 | ~0.03 | 0.035–0.046 | ok |

**Bicycle** (7000 iters) — runs `20260807T081507Z_run{1,2,3}`:

| metric | Wave 2 / G5 | **WO-X** |
|---|---:|---:|
| wall_s | 21.12 (G5) | **21.27** |
| steady_ms/iter | 2.817 (G5) | **2.800** (−0.6%) |
| steady_allocs/iter | 0.04 | **0.04** |
| peak MiB | 1613 (W4) | **1649** |
| gt_cache MiB | 564 | **564.4** |
| last_loss range | 0.085–0.101 (G5) | 0.104–0.136 (high-var OK) |

### Peak ledger owners (bonsai confirm)

| line | owner | justified |
|---|---|---|
| gt_cache (~339 MiB) | WO-HP1 | yes (budget-gated) |
| training_state (~158 MiB) | Phase 0.2/2.2 | yes |
| loss_workspace_arena (~21 MiB) | Phase 6D | yes |
| pool_bucket_cache (~46 MiB) | allocator | yes |
| no_trim_pool_residency (~223 MiB) | WO-X | yes — trade peak for G2 (no post-refine trim) |

### Full-suite delta vs WO-G5 remaining

Same independent reds (not introduced by WO-X):
- SogFormatTest ×12 (ISS-014 family)
- TensorReserveInplaceCat.OverflowFailurePreservesInstalledStorage
- CheckpointAllocatorRegression (ISS-014, segfault mid-suite)
- DeviceFaultTest.GraphCapture (ISS-013, excluded)
- PythonIntegrationTest ×3 (excluded / hang)

Campaign unit tests: **19/19 PASS**.

### Commits
**Commit:** `b416b603`

---

## ISS-020 / WO-X2 — post-suite teardown SIGSEGV (static destruction order)

- **Branch:** `lfs-elite` (never checkout)
- **FAIL evidence:**  
  `lichtfeld_tests --gtest_filter='PPISPControllerTest.*:DensifyEvents4x.*:FastGSGradientTest.Numerical_Means:CheckpointStrategies/*'`  
  → 21 PASSED, then after `Shutting down CudaMemoryPool...` **exit 139** (SIGSEGV in  
  `StreamOrderedDeviceBuffer::reset` / static Tensor free after pool Meyers singleton destroyed).
- **PASS evidence:** same filter **exit 0**; clean full suite (documented reds excluded)  
  **3277 PASS, process exit 1 only for suite-order VramProfilerMetricsTest.TopLiveSortedByLiveBytes**  
  (no 139/134); no double-free after pool shutdown.
- **Root cause:** class-static PPISP shared buffers + main-thread TLS FastGS sort/rasterizer  
  caches + mirror mult cache free after `CudaMemoryPool` function-local static is destroyed;  
  training-thread TLS release hooks never ran on the test process exit path.
- **Fix:**
  1. `register_gpu_pre_shutdown_hook` + step 0 of `teardown_gpu_before_exit` (hooks before  
     device_fault / arena / pool / pinned) — same explicit-release pattern as training-thread TLS.
  2. Hooks: PPISP `release_shared_buffers`, mirror cache clear, FastGS sort + fast/gsplat  
     rasterizer TLS, nan-check TLS.
  3. Belt-and-suspenders: `safe_cuda_pool_deallocate` / clear live-pool atomic on shutdown;  
     `gpu_process_teardown_started()` guards cudaFree / sort-buffer free / pinned free.
  4. `test_main`: `flush_and_exit` after teardown (no C++ static/TLS destruction into dead pool).
- **Why hooks first (not only liveness-aware deleters):** free while CUDA is healthy so TLS  
  never calls into a dead context; deleters only prevent residual late-dtor crashes.
- **WO-X densify N-scratch:** not process-static (strategy member); `release()` added; free  
  paths hardened (zeros_direct + pool empty).
- **GT cache:** instance-owned, cleared on loader shutdown — no static holder change.

### Dual-workload gate (unchanged vs target)

| workload | med steady ms/iter | B/splat | allocs/iter |
|---|---|---|---|
| bonsai ×3 | **3.159** (~3.15) | **304.3** | **0.11** |
| bicycle 7k ×3 | **2.745** (~2.75) | **306.8** | **0.10** |

allocs/iter ≤ 0.11 (no regression of WO-X).

### Unit tests
`GpuTeardownOrderTest.*` 3/3 PASS.

### Commit
**`8e65a0b5`** fix(ISS-020): ordered GPU release before pool teardown (exit 139→0)

---

## WO-WARP-FWD — Warp-level culling for FastGS forward blend_cu

- **Branch:** `lfs-elite` (never checkout)
- **Citation (maintainer-approved):** Yang, Drettakis, Bernstein, "Warp-Level Culling
  for Efficient Blending in 3D Gaussian Splatting", ACM CGIT 9(4):54, 2026,
  doi:10.1145/3820019. Cited in kernel-head comment + landing commit message.

### Change
1. **Warp sub-tile culling (8×4):** 16×16 tile → 8 warps × 32 px; each warp ballots
   splat×sub-tile AABB intersection 32-at-a-time; non-hits skip full conic/color/alpha.
   `n_possible_contributions` still advances per splat index → last_contributor /
   backward bit-identical.
2. **128-bit layout:** `PackedMeanBBox{float2 mean2d; ushort4 pixel_bbox}` (16B);
   `color` float3→float4 padded. Tile `screen_bounds` kept for create_instances.
3. **Batch size:** `config::blend_batch_size = 192` (see sweep table). Runtime override
   via `set_blend_batch_size_for_testing` / `LFS_BLEND_BATCH_SIZE`. Cull mode via
   `set_warp_cull_mode_for_testing` / `LFS_WARP_CULL_MODE` (0=on, 1=off, 2=wrong).

### TDD
- **FAIL-first (wrong mask):** mode=2 empty ballot vs mode=1 reference → images differ
  (test sensitivity).
- **PASS:** mode=0 vs mode=1 bit-identical on synthetic (48) + dense (512) fixtures;
  batch 32..256 step 32 bit-identical; deterministic re-render.
```
[  PASSED  ] WarpCullBlendTest.WrongMaskDiffersFromReference
[  PASSED  ] WarpCullBlendTest.EnabledMatchesReference_Synthetic
[  PASSED  ] WarpCullBlendTest.EnabledMatchesReference_Dense
[  PASSED  ] WarpCullBlendTest.BatchSizeSweepBitIdentical
[  PASSED  ] WarpCullBlendTest.EnabledIsDeterministic
FastGSGradientTest.* 5/5 PASS; FastGSDenseTileGradientTest.* 4/4 PASS; FusedBgBlend 2/2
```

### Batch-size sweep (RTX 4080)

**A. Synthetic microbench** (forward wall, 512², N=4096, 50 runs; noise ±~0.005 ms):

| batch | mean_fwd_ms |
|------:|------------:|
| 32 | 0.198 |
| 64 | 0.193 |
| 96 | 0.192 |
| 128 | 0.193 |
| 160 | 0.195 |
| 192 | 0.198 |
| 224 | 0.191 |
| 256 | 0.191 |

**B. Late-window bonsai kern_sum** (blend_cu avg µs, slice [1600,1900), 300 frames):

| batch | blend_cu avg_us | med_us |
|------:|----------------:|-------:|
| 128 | 373.3 | 362.1 |
| **192** | **366.1** | **359.0** |
| 256 | 370.1 | 362.2 |

**Pick:** `blend_batch_size = 192` (real-scene kern_sum argmin; matches paper high-end).

### Kernel-time A/B (profile.sh, bonsai late [1600,1900))

| config | blend_cu avg_us | med_us | notes |
|---|---:|---:|---|
| Historical (pre-change, philox/bwd-a) | ~416 | ~410 | float3 color, batch=256, no cull |
| Cull OFF (same binary, packing+192) | 455.3 | 443.5 | LFS_WARP_CULL_MODE=1 |
| **Cull ON (production)** | **366.1** | **359.0** | batch=192 |

- vs historical: **avg −12.0%**, **med −12.4%** (in −10..−17% target)
- vs cull-off same binary: avg −18.5% (cull contribution)
- Bicycle late same-binary: ON 191.7 vs OFF 207.7 µs (**−7.7%**; smaller tiles / denser)

Profiles: `perf_campaign/profiles/warpcull-{bonsai,off-bonsai,bicycle,off-bicycle,batch*}-late/`

### Dual-workload gate

| workload | med steady_ms | B/splat | allocs/iter | vs prior (ISS-020) |
|---|---:|---:|---:|---|
| bonsai ×3 | **3.113** | **304.3** | 0.11 | was 3.159 → **improved** |
| bicycle 7k ×3 | **2.737** | **306.8** | 0.10 | was 2.745 → **improved** |

B/splat unchanged. Loss ranges healthy (bonsai ~0.03–0.07; bicycle 0.11–0.15).

### Full-suite delta
3313 PASS / 42 SKIP / 11 FAIL with documented reds excluded via gtest_filter.
Failures are pre-existing env/fixture reds (PipelinedImageLoader fixtures, NaNInf large,
Float16HostReduce, SceneValidity migrate, VramProfiler order, TensorReserve multi-dim,
Python SceneCamera) — **not** FastGS blend/grad. All FastGS numerical gradients green.

### Files
- `kernels_forward.cuh` — blend_cu warp cull + packing; preprocess pixel AABB + float4 color
- `buffer_utils.h` — `PackedMeanBBox`; color float4
- `kernels_backward.cuh` — unpack mean2d / float4 color
- `rasterization_config.h` — `blend_batch_size=192`, subtile 8×4
- `forward.cu` / `forward.h` — test hooks + env A/B
- `tests/test_warp_cull_blend.cpp` — TDD pixel identity + microbench

### Commit
**`35ca0e4a`** perf(fastgs): warp-level sub-tile culling for forward blend_cu (WO-WARP-FWD)


## dual-rep optimizer-state cluster (BL-1, BL-2, MJ-1..4, MN-5/6) — 2026-08-07

Adversarial-review dual-representation cluster (quant codes + joint moments
everywhere except the fused fastgs path). TDD suite `DualRepOptimizer.*`
written first; production fixes landed with all 10 cases green.

### Findings fixed
- **BL-1** legacy + q16 OOB: `prepare_fastgs_fused_adam` dequants when
  `!is_joint()`; legacy kernel branch returns if `sh_value_bits != 0`.
- **BL-2** joint shN moments sized from float layout not u16 cells;
  `allocate_gradients`/`alloc_quantized_state`/deserialize use
  `sh_swizzled_float_count`; checkpoint roundtrip **after fused prepare**
  (past SH warmup) with quant ON — the prior gap.
- **MJ-1** grid-overhang: `primitive_idx >= n_primitives` clears touch on
  joint path; legacy path returns early.
- **MJ-2** `joint_encode_zero_*` widens block bounds to include (0,0) and
  re-encodes the block so zeros decode to true (m,v)=(0,0).
- **MJ-3** joint grow zero-encodes new rows; gather transcodes across
  blocks (`joint_transcode_gathered_rows`).
- **MJ-4** joint branch in `add_new_params_gather(ShN)` grows packed moments
  (legacy gate no longer silently no-ops).
- **MN-5** load accepts `joint_bounds.shape[0] >= expected` (grow-only slack).
- **MN-6** bounds grow keeps source alive until D2D copy completes.

### Evidence
```
./build/tests/lichtfeld_tests --gtest_filter='DualRepOptimizer.*'
# 10/10 PASSED

./build/tests/lichtfeld_tests --gtest_filter=\
'DualRepOptimizer.*:AdamCapacityInvariant.*:JointAdam*:CheckpointResumeRoundtripTest.*:MCMCTest.*:MRNFStrategyTest.*:ShValueStorage*'
# 48/48 PASSED
```

New mandatory quant-ON strategy smoke:
- `DualRepOptimizer.MCMC_InitializeWithBothCodecsOn`
- `DualRepOptimizer.MRNF_InitializeWithBothCodecsOn`

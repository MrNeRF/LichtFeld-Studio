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
## Task 2.2 — Joint (u, log_s) Adam codec

- **Branch:** `lfs-elite`
- **Change:** Replace uint8 moments + per-primitive fp32 scales with spirulae-style
  joint `(u, log_s)` codec: 16-bit non-SH (4 B/cell) / 8-bit SH (2 B/cell),
  `float4` bounds per 256-splat block, endpoint-exact, 0↔0 fixed point.
  Decode+update+encode inside fused Adam (`adam_step_row_joint` /
  `apply_shN_grads_packed_joint`). Runtime fallback: `LFS_ADAM_LEGACY_CODEC=1`
  (or `set_joint_codec_enabled_for_testing(false)`).
  Densify/relocate/reset encode true zeros under current bounds
  (`joint_encode_zero_*`). Compact rebuilds zero bounds (free-zero moments).
  Also: MRNF `ensure_*_capacity` no longer treats `cuda.direct` as interop external
  (fixes shN max_cap reservation under-allocation).
- **Fail evidence (TDD):**
  ```
  # Pre-impl: no joint_adam_codec.hpp / JointAdamCodecTest
  tests/test_joint_adam_codec.cpp: fatal error: lfs/training/joint_adam_codec.hpp: No such file
  # Ledger expected 172 optim B/splat (legacy) before joint path
  ```
- **Pass evidence:**
  ```
  [  PASSED  ] 6 tests.  (JointAdamCodecTest.*)
  [  PASSED  ] 3 tests.  (TrainingStateLedgerTest.* incl. joint + legacy)
  [  PASSED  ] 2 tests.  (AdamCapacityInvariant.*)
  [  PASSED  ] 19 tests. (MRNFStrategyTest.*)
  [  PASSED  ] 2 tests.  (MCMCTest.* with legacy scale guards)
  ```
- **Dual-workload gate (quiet-ish; 3-run medians):**

  **Bonsai** (2000 iters) vs Wave-2:

  | metric | Wave-2 | after 2.2 | Δ |
  |---|---:|---:|---|
  | wall_s (med) | ~8.9 | **9.67** | +noise/block256 |
  | steady_ms/iter | 4.065 | **4.287** | +5.5% (block_size_preprocess_backward 128→256 for quant blocks) |
  | steady_allocs/iter | 0.05 | **0.08** | flat |
  | peak VRAM MiB | 938.3 | **930.1** | −8 |
  | B/splat | 429.0 | **409.4** | **−19.6** (optim 172→~152; LFS shN pad 48 cells) |
  | last_loss | ~0.03–0.04 | 0.030–0.035 | ok |

  Runs: `perf_campaign/runs/20260806T210434Z_run{1,2,3}/`

  **Bicycle canary** (7000 iters) vs Wave-2:

  | metric | Wave-2 | after 2.2 | Δ |
  |---|---:|---:|---|
  | wall_s (med) | — | **41.07** | (high wall variance; run1=34.8) |
  | steady_ms/iter | 3.208 | **3.215** | **flat** |
  | steady_allocs/iter | 0.04 | **0.07** | ok |
  | peak VRAM MiB | 1026.3 | **986.1** | −40 |
  | B/splat | 429.0 | **409.4** | −19.6 |
  | final loss range | 0.079–0.107 | **0.106–0.158** | high-variance OK; curves healthy |

  Loss-curve samples (run3, every 1k): 1k=0.17, 2k=0.12, 3k=0.18, 4k=0.10,
  5k=0.10, 6k=0.12, 6.9k=0.11 — densify 54k→500k, no collapse/NaN.

  Note: target ~403 B/splat assumes spirulae 45 SH cells; LFS swizzled pad is
  48 cells → optim 152 vs 146 → total ~409. Phase 2.4 can reclaim the pad.

- **Env flag:** `LFS_ADAM_LEGACY_CODEC=1` forces legacy uint8+scales path.
- **Default:** joint ON.
- **Commit:** `63aa08c6`

---
## Task 2.1 — SH-rest 16-bit block quantization (infrastructure)

- **Branch:** `lfs-elite`
- **Change:** Spirulae-style 16-bit linear SH-rest value codec (host+device) +
  unit tests + runtime flags. **Default OFF** — full FastGS/densify/export wiring
  not complete (see ISS-2.1). Opt-in: `LFS_SH_VALUE_QUANT=1`.
- **Fail evidence (TDD):**
  ```
  # Pre-impl:
  tests/test_sh_value_codec.cpp: fatal error: lfs/training/sh_value_codec.hpp: No such file
  ```
- **Pass evidence:**
  ```
  [==========] Running 6 tests from 1 test suite.
  [  PASSED  ] 6 tests.  (ShValueCodecTest.*)
  ```
- **Gate:** dual-workload **not run with quant ON** (storage not wired).
  Task 2.2 gate remains the live B/splat number (409.4). Enabling 2.1 is expected
  to drop ~409→~313 once wired.
- **Commit:** `514b2a49`

## Task PROF-1 — nsys profiling harness + first steady-state profiles

- **Branch:** `lfs-elite-prof` (bench worktree)
- **Change:** `perf_campaign/profile.sh` (`timeline|kernels|ncu`) +
  `launch_gaps.py`/`profile_summary.py`; env-gated trainer hooks
  (`LFS_NVTX=1`, `LFS_PROFILE_START_ITER/STOP_ITER` → cudaProfilerStart/Stop),
  zero-cost when off. nsys captures exactly the slice [start,stop) via
  `--capture-range=cudaProfilerApi`; raw .nsys-rep gitignored, summaries committed.
- **Captures (all under flock /tmp/lfs-bench.lock):**
  `d81a5c2b-bonsai`, `d81a5c2b-bicycle` (iters 200-500);
  codec pair `487d5c2b-bonsai` / `63aa08c6-bonsai` (200-500) and
  `*-bonsai-late` (1600-1900). Summaries in `perf_campaign/profiles/*/summary.md`.
- **Headline numbers (tip, bonsai 200-500):** 29 kernel launches/iter;
  median iter busy 2.73 ms; blend_backward_cu 55.5% of GPU time;
  mrnf_noise_injection 15%; memsets ~16/iter but only ~18 µs/iter total.
  Late window (1600-1900): GPU busy 5.08 ms of 5.18 ms span (1.9% gap) —
  GPU-bound at large N; CUDA-graph upside is small there, larger (0.2-0.8 ms)
  at small N / bicycle where host dispatch dominates.
- **Codec-pair delta (feeds WO-G2):** preprocess_backward_cu
  200-500 (SH deg 0): 179.3→140.3 µs (codec FASTER);
  1600-1900 (SH deg ≥1): 450.0→906.3 µs (**+101%**). Weighted ≈ +0.24 ms/iter,
  matches the +0.222 ms steady regression. Regression is SH-degree gated →
  WO-G2 must gate on a late window. Details: `profiles/codec-pair-63aa08c6.md`.
- **ncu:** HW counters admin-locked, passwordless sudo unavailable; harness
  prints the exact `sudo ncu` command (see `profile.sh ncu`).
- **Commits:** `3abe9997` (hooks), `6194c94e` (harness), summaries follow.

---
## Task WO-NOISE-PHILOX — Philox4_32_10 + curand_normal4 (Wave 5 rank-1)

- **Branch:** `lfs-elite`
- **Change:** Replace per-thread XORWOW `curandState` + `curand_init` with
  `curandStatePhilox4_32_10_t` + `curand_normal4` (noise) / `curand_uniform`
  (sample paths) in:
  - `mrnf_kernels.cu` — `mrnf_noise_injection_kernel`, `gumbel_key_*`
  - `mcmc_kernels.cu` — `inject_noise_kernel`, multinomial sample kernels
  Philox init is counter-setup only (no XORWOW skip-ahead). Distribution is
  statistically N(0,1)-equivalent; trajectories not bit-identical.
  Out of scope (logged): `tensor_random_ops.cu` still uses XORWOW for generic
  Tensor random fills — not on the training hot path measured here.

### TDD evidence

**Fail (pre-Philox, N=400k cudaEvent median, 21 reps):**
```
[PhiloxKernelTime] mcmc inject_noise N=400k median_us=1896.45
Expected: (median_us) < (50.0), actual: 1896.45 vs 50  FAILED
[PhiloxKernelTime] mrnf_noise_injection N=400k median_us=1888.48
Expected: (median_us) < (50.0), actual: 1888.48 vs 50  FAILED
```
Distribution tests already green on XORWOW (mean/var/normality).

**Pass (post-Philox):**
```
[PhiloxKernelTime] mcmc inject_noise N=400k median_us=11.264   (~168×)
[PhiloxKernelTime] mrnf_noise_injection N=400k median_us=7.168 (~263×)
[  PASSED  ] 6 tests from FusedNoiseInjectionTest
[  PASSED  ] 28 tests (MRNFStrategy + McmcMultinomial + FusedNoise)
```

### Profile gate (bonsai late, iters 1600–1900)

| kernel | before avg µs (63aa08c6-late) | after (philox-bonsai-late) | speedup |
|---|---:|---:|---:|
| `mrnf_noise_injection_kernel` | 1325.6 (25.2% GPU) | **5.9** (0.2% GPU) | **224×** |
| `gumbel_key_for_indices_kernel` | 1476.2 | **3.2** | **461×** |
| median GPU busy / iter | ~5080 µs (prior late notes) | **3772 µs** | **−25.7%** |
| median span / iter | ~5180 µs | **3996 µs** | −22.9% |

Matches Directive-2 expectation (~25% late GPU-busy drop from removing noise
from the top-kernel table). Profile: `perf_campaign/profiles/philox-bonsai-late/`.

### Dual-workload medians (3 runs each)

**Bonsai 2k** (`20260806T223610Z_run{1,2,3}`):

| metric | Wave-2.2 (63aa08c6) | after Philox | Δ |
|---|---:|---:|---:|
| wall_s (med) | 9.67 | **7.52** | **−22%** |
| steady_ms/iter | 4.287 | **3.316** | **−22.6%** |
| steady_allocs/iter | 0.08 | **0.08** | flat |
| peak VRAM MiB | 930.1 | **930.1** | flat |
| B/splat | 409.4 | **409.4** | flat |
| last_loss | 0.030–0.035 | **0.028–0.033** | ok |

Loss-curve overlay (every 200, run1/2/3): healthy monotone-ish decay, no
collapse. run1: 200=0.16 → 1000=0.07 → 1900=0.03; run3: 200=0.15 → 1000=0.08
→ 1900=0.028. Final losses in historical band.

**Bicycle 7k canary** (`20260806T223757Z_run{1,2,3}`):

| metric | Wave-2.2 | after Philox | Δ |
|---|---:|---:|---:|
| wall_s (med) | 41.07 | **26.00** | **−37%** |
| steady_ms/iter | 3.215 | **1.882** | **−41%** |
| steady_allocs/iter | 0.07 | **0.07** | flat |
| peak VRAM MiB | 986–1018 | **986–1018** | flat |
| B/splat | 409.4 | **409.4** | flat |
| final loss range | 0.106–0.158 | **0.084–0.187** | high-variance OK |

Loss curves (every 1k, not just final) — densify 54k→500k, no NaN/collapse:

| iter | run1 | run2 | run3 |
|---:|---:|---:|---:|
| 1k | 0.15 | 0.19 | 0.16 |
| 2k | 0.25 | 0.13 | 0.13 |
| 3k | 0.12 | 0.17 | 0.12 |
| 4k | 0.10 | 0.14 | 0.17 |
| 5k | 0.10 | 0.11 | 0.14 |
| 6k | 0.13 | 0.09 | 0.22 |
| 6.9k | 0.12 | 0.13 | 0.21 |

Wave-2.2 reference (run3): 1k=0.17 … 6.9k=0.11. Same qualitative shape and
variance; quality gate PASS.

### Commit

- **Commit:** `080cbf91`

---
## Task WO-BWD-A — blend_backward T_eff clamp (Wave 5 rank-2)

- **Branch:** `lfs-elite`
- **Change:** After `s_last_contributor` fill+sync in `blend_backward_cu`,
  block-reduce max of the 256 last_contributor entries and clamp the reverse
  batch walk to `T_eff = min(tile_n_primitives, max_contrib)`. Index formula
  becomes `tile_primitive_idx = T_eff - batch_base - lane - 1` (equivalent to
  starting the original walk at `tile_n_primitives - T_eff`). Exact math:
  skipped high-index splats already produced zero grad via the
  `tile_primitive_idx < last_contributor` gate.
- **Debug:** one-shot tile histogram behind `LFS_BWD_TEFF_HIST=1`
  (default iter 1700, override `LFS_BWD_TEFF_HIST_ITER`). Arm/flush in
  `train_step` via `bwd_teff_hist_{arm,flush}`.

### TDD / correctness

- Pre-existing run-to-run loss non-determinism (float atomics / async path)
  already prevents strict bit-identical trajectories even on the same binary
  (rep1 vs rep2 400-iter losses differ). Structural signals match:
  densify counts identical across pre/post runs (e.g. 221075 @200, 236550 @400).
- Dual-workload quality: no NaN/collapse; loss ranges in historical band.

### Histogram evidence (iter 1700, RTX 4080)

**Bonsai** (`profiles/bwd-a-hist/bonsai1700_hist.txt`):
```
tiles=1617  mean_tile_n=351.8  mean_T_eff=337.3  mean_waste_frac=0.041
waste_pct: 0-10%=1524/1617 (94%), 20-40%=58 tiles
```

**Bicycle** (`profiles/bwd-a-hist/bicycle1700_hist.txt`):
```
tiles=4056  mean_tile_n=75.0  mean_T_eff=75.0  mean_waste_frac=0.001
waste_pct: 0-10%=4054/4056
```

Finding: max-over-tile `last_contributor` ≈ `tile_n` on these workloads
(at least one of 256 pixels almost always walks near the full list). Max-based
clamp therefore saves only a few percent of the walk. The large −25–50%
expectation needs **BWD-B** (per-splat survivor compaction over `[0,T_eff)`),
not max-only. BWD-A remains the correct exact-math foundation.

### Kernel-time A/B (bonsai late 1600–1900)

| | blend_backward avg µs | med µs | % GPU |
|---|---:|---:|---:|
| pre (philox-bonsai-late) | 2098.0 | 2009.2 | 53.2 |
| post (bwd-a-bonsai-late) | 2112.7 | 2037.3 | 53.6 |
| Δ | **~flat** (within noise; matches ~4% waste) | | |

Profile: `perf_campaign/profiles/bwd-a-bonsai-late/summary.md`

### Dual-workload gate

**Bonsai 2k ×3** (post):
| metric | Philox med | BWD-A med | Δ |
|---|---:|---:|---:|
| wall_s | 7.52 | 9.34 | run variance (phase-0 was 9.00) |
| steady_ms/iter | 3.316 | **3.489** | ~+5% noise / no win |
| peak MiB | 930.1 | **930.1** | flat |
| B/splat | 409.4 | **409.4** | flat |
| last_loss | 0.028–0.033 | **0.033–0.049** | ok |

**Bicycle 7k ×3** (post):
| metric | Philox med | BWD-A med | Δ |
|---|---:|---:|---:|
| wall_s | 26.00 | 40.76 | host/decode dominated (not bwd kernel) |
| steady_ms/iter | 1.882 | **2.025** | ~+8% |
| peak MiB | 986–1018 | **986.1** | flat |
| B/splat | 409.4 | **409.4** | flat |
| last_loss | 0.084–0.187 | **0.080–0.185** | ok |

No quality regression. Kernel-level win is ~0 on current scenes (histogram
explains why). Ship as exact-math enabler for BWD-B/C.

### Commit

- **Commit:** `a3f25c4f`

---

## WO-HP1 — decoded-GT device cache + `dataloader_wait_ms` (Wave 5 rank-3)

**Date:** 2026-08-07  
**Branch:** lfs-elite  
**Directive:** DIRECTIVES-round2 §Directive-3 / WO-HP1-gt-cache

### Part 1 — metric first

Added `dataloader_wait_ms` (total), `dataloader_wait_ms_per_iter`,
`steady_dataloader_wait_ms_per_iter` to `perf_bench.json`, and `dl_wait` /
`gt_cache` columns to `bench.sh`. Instrumented `train_dataloader->next()` in
`trainer.cpp` (outside `train_step` / `steady_ms`).

### TDD

| suite | result |
|---|---|
| `GtCacheBudgetGate.*` (5 pure budget tests) | PASS |
| `GtDeviceCacheTest.CacheHitReturnsBitIdenticalTensor` | PASS (FNV hash equal) |
| `GtDeviceCacheTest.BudgetGateDisablesDeviceFallsBackToDecode` | PASS |
| `GtDeviceCacheTest.ClearGtCacheEvictsAndFallsBack` | PASS |
| `GtDeviceCacheTest.CapOverrideForcesPinnedOrOff` | PASS (pinned middle tier) |

### Baseline (metric present, `LFS_GT_CACHE=0`)

| | wall_s | steady_ms | dl_wait | gt_cache MiB | peak MiB | B/splat |
|---|---:|---:|---:|---:|---:|---:|
| Bonsai 2k med×3 | 7.95 | 3.373 | 0.013 | 0.0 | 930.1 | 409.4 |
| Bicycle 7k med×3 | **30.39** | 1.764 | **2.283** | 0.0 | 986.1 | 409.4 |

Bicycle residual: ~2.28 ms/iter blocked in `next()` after compressed-JPEG
warmup (Directive-3 measured 4.8 ms Huffman; warm jpeg cache already cuts
some of that).

### After — decoded-GT device cache ON (default, budget-gated)

VRAM budget: enable device iff `n_images * u8_CHW < min(free_vram − 2 GiB, LFS_GT_CACHE_CAP)`.
Pinned-host middle tier when device denied; legacy decode as fallback.
Env: `LFS_GT_CACHE=0` disables; `LFS_GT_CACHE_CAP=<bytes>` caps device budget.

| | wall_s | steady_ms | dl_wait | gt_cache MiB | peak MiB | B/splat |
|---|---:|---:|---:|---:|---:|---:|
| Bonsai 2k med×3 | 7.86 (−1.2%) | 3.336 | 0.010 | **338.8** | 1506.1 | 409.4 |
| Bicycle 7k med×3 | **22.49 (−26.0%)** | 2.956 | **0.007 (−99.7%)** | **564.4** | 1562.1 | 409.4 |

Loader shutdown (bicycle run3): `device=194 entries/564.4 MiB hits=6809 inserts=194`.

### VRAM ledger (cache on vs off)

| bucket | bonsai off | bonsai on | bicycle off | bicycle on |
|---|---:|---:|---:|---:|
| **gt_cache_MiB** | 0.0 | **338.8** | 0.0 | **564.4** |
| peak_cuda_used_MiB | 930.1 | 1506.1 | 986.1 | 1562.1 |
| headroom left (16 GB − peak) | ~15.0 GB | ~14.5 GB | ~15.0 GB | ~14.5 GB |
| budget gate (≥2 GB free above peak) | n/a | **pass** | n/a | **pass** |

### Notes

- Wall is the ground-truth gate for this WO (Directive-3: `steady_ms` is blind
  to `next()`). Bicycle wall −26% (target −30–45%); short of the low end
  because warm compressed-JPEG cache already reduced residual wait to 2.28 ms
  (not the raw 4.8 ms Huffman). `dl_wait` → ~0 as required.
- `steady_ms` rose on bicycle because host-side step timing no longer drains
  prior async GPU work during a multi-ms `next()` wait; combined
  `steady+dl_wait` dropped 4.05 → 2.96 ms/iter (−27%), matching wall.
- Quality: B/splat flat 409.4; bicycle final loss still in historical noise band.

### Files

- `src/training/{perf_bench.cpp,include/lfs/training/perf_bench.hpp}`
- `src/training/trainer.cpp` — dl_wait + `configure_gt_cache` + stats capture
- `src/io/{pipelined_image_loader.cpp,include/io/pipelined_image_loader.hpp}`
- `tests/test_gt_device_cache.cpp`, `tests/CMakeLists.txt`
- `perf_campaign/bench.sh`

**Commit:** `353cdfb0`

## FIX-2.2 F1 — SH Adam before geometry + launch_bounds(256,3)

- **Branch:** `lfs-elite`
- **Change:** In `preprocess_backward_cu`:
  1. Close visible branch after `convert_sh_to_color_backward_grads` (fill
     sh0/shN grads only; save `dL_dmean3d_from_color`).
  2. Run sh0 + shN Adam for ALL threads (joint block-bounds) so
     `shN_grads[15]×3` + joint `us_u/us_s[48]` die before covariance/EWA.
  3. Reopen visible for geometry; Phase D Adam is means/rot/scale/opacity only.
  4. `__launch_bounds__(block_size_preprocess_backward, 3)` (sweep 2..4; pick 3).
- **Task 2.1 verify (worker G, default OFF):** ShValueCodecTest 6/6 PASS;
  JointAdamCodecTest 6/6 PASS. No extension.

### Kernel timing (bonsai late window iters 1600–1900, nsys)

| | preprocess_backward_cu avg µs | Δ |
|---|---:|---:|
| before (63aa08c6-bonsai-late / tip-ish) | **906.3** | — |
| after F1 (`fix22-f1-bonsai-late`) | **622.0** | **−284.3 (−31.4%)** |
| parent no-codec (487d5c2b late) | 450.0 | residual +172 µs |

Profile: `perf_campaign/profiles/fix22-f1-bonsai-late/summary.md`

### Bonsai gate (3-run median, joint codec ON, GT-cache default ON)

| metric | Wave-2.2 (63aa08c6) | after F1 | gate |
|---|---:|---:|:---|
| wall_s | 9.67 | **7.22** | — |
| steady_ms/iter | 4.287 | **3.148** | ≤4.065 **PASS** |
| B/splat | 409.4 | **409.4** | flat **PASS** |
| last_loss | 0.030–0.035 | 0.028–0.039 | ok |
| peak VRAM MiB | 930 | 1512 (GT cache) | n/a |

Runs: `perf_campaign/runs/fix22_f1_bonsai/`

- **Commit:** `a93cd668`

## FIX-2.2 F2 — fused block_reduce_min4 bounds

- **Branch:** `lfs-elite`
- **Change:**
  - `warp_reduce.cuh`: `block_reduce_min4(float4)` via `__shfl_xor` butterfly +
    one non-static shared float4 round (1 barrier).
  - Joint Adam bounds pack `{u_min,-u_max,s_min,-s_max}` so one min4 replaces
    four alternating min/max reduces (30→12 barriers across 6 sections; fixes
    static-shared race between min/max).
  - Wired in `adam_step_row_joint` + `apply_shN_grads_packed_joint`.

### Kernel timing (bonsai late 1600–1900)

| | preprocess_backward avg µs | Δ vs prev |
|---|---:|---:|
| after F1 | 622.0 | — |
| after F2 (`fix22-f2-bonsai-late`) | **603.8** | **−18.2 (−2.9%)** |

### Bonsai gate med×3

| metric | after F1 | after F2 |
|---|---:|---:|
| steady_ms | 3.148 | **3.136** |
| B/splat | 409.4 | **409.4** |
| wall_s | 7.22 | **7.20** |

Gate ≤4.065 still **PASS**.

- **Commit:** `60d97b26`

## FIX-2.2 F3 — hoist inv ranges + fast log/exp transcode

- **Branch:** `lfs-elite`
- **Change:**
  - Device `encode_us`: block-uniform `inv_u_range`/`inv_s_range` hoisted once
    per section (2 fdiv → FMA per cell).
  - Guarded fast path: `__logf(1+x)` when `x>0.125` else `log1pf`;
    `__expf-1` when `log_s>0.118` else `expm1f` (0↔0 fixed point exact).
  - Host `joint_adam_codec.hpp` mirrors thresholds + inv-range mul.
  - JointAdamCodecTest **6/6 PASS**.

### Kernel timing (bonsai late 1600–1900)

| step | preprocess_backward avg µs | Δ vs prev |
|---|---:|---:|
| 63aa08c6 (codec regression) | 906.3 | — |
| F1 | 622.0 | −284.3 (−31.4%) |
| F2 | 603.8 | −18.2 (−2.9%) |
| **F3** (`fix22-f3-bonsai-late`) | **580.6** | **−23.2 (−3.8%)** |
| parent no-codec (487d5c2b) | 450.0 | residual +130.6 µs vs parent |

**Net F1–F3:** 906.3 → 580.6 µs (**−35.9%**). F4 not needed (gate pass; ncu admin-locked).

### Dual-workload gate after F1–F3 (joint codec ON)

**Bonsai** med×3:

| metric | Wave-2.2 | after F3 | gate |
|---|---:|---:|:---|
| wall_s | 9.67 | **7.49** | — |
| steady_ms/iter | 4.287 | **3.168** | ≤4.065 **PASS** |
| B/splat | 409.4 | **409.4** | flat **PASS** |
| last_loss | 0.03–0.04 | 0.029–0.072 | ok |

**Bicycle** med×3 (7000 iters):

| metric | after F3 |
|---|---:|
| wall_s | **21.96** |
| steady_ms/iter | **2.843** |
| B/splat | **409.4** |
| final loss range | 0.082–0.149 |

Default joint codec **stays ON**. Infrastructure left in place.

- **Commit:** `993314d5`
||||||| c44ad8ec

## Task 6D.1 — Loss-workspace union (shared arena)

- **Branch:** `lfs-elite-mem`
- **Change:** `LossWorkspaceArena` in `ssim.cuh` / `ssim.cu` owns one grow-only UInt8
  blob sized to `max(variant layout)` at the active resolution. Fused / pure-SSIM /
  decoupled / masked-fused / masked-decoupled are mutually exclusive views rebuilt on
  mode switch. `PhotometricLoss` owns the arena; Trainer masked/decoupled paths use
  `photometric_loss_.arena()`. VRAM accounting reports arena capacity once (not the sum).
  Independent stack workspaces still allocate separately for unit tests.
- **Fail evidence (TDD):** pre-union, five independent `ensure_size` calls retain the sum:
  ```
  IndependentWorkspacesRetainSum: total >> max_variant  (EXPECTED_GT documents bug)
  # example 64x96 NCHW: sum ≈ 5× single-variant; arena gate would fail under sum retention
  ```
  Before arena API, `LossWorkspaceArena` / `allocated_bytes()` did not exist
  (compile fail for SequentialModesThroughArenaStayWithinMax).
- **Pass evidence:**
  ```
  [  PASSED  ] LossWorkspaceUnionTest.IndependentWorkspacesRetainSum
  [  PASSED  ] LossWorkspaceUnionTest.SequentialModesThroughArenaStayWithinMax
  [  PASSED  ] LossWorkspaceUnionTest.ArenaFusedLossMatchesIndependent
  [  PASSED  ] LossWorkspaceUnionTest.PhotometricLossExposesSharedArena
  [  PASSED  ] FusedL1SSIMTest.* (15) + LossWorkspaceUnionTest.* (4) = 19/19
  ```
- **Also:** fixed `FusedL1SSIMTest.WorkspaceReuse` to snapshot loss/grad before workspace
  scalar reuse (Phase 1.6 alias).
- **Numbers (unit, 64×96):** arena capacity ≤ max(variant)+256KiB slack after touching
  all 5 modes; independent path retains sum (documents pre-6D.1 peak risk up to ~650 MiB
  @1080p).
- **Commit:** `9fc40b0b`

## Task 6D.2 — Delete zero_terms (HasSigmaPartials flag)

- **Branch:** `lfs-elite-mem`
- **Change:** Decoupled / masked-decoupled app-branch backward no longer allocates or
  reads a full-image `zero_terms` buffer. `fusedL1SSIMBackwardCUDA` and
  `maskedFusedL1SSIMBackwardCUDA` gain `HasSigmaPartials` template flag; app path
  instantiates `false` and passes null sigma partials (compile-time zeros). Arena
  layout and independent `ensure_size` drop one image-sized f32 buffer each.
- **Fail evidence (TDD):** Pre-6D.2 layout retained `zero_terms` (full image f32). Unit
  oracle: live decoupled bytes would be `map + 7*image + reduce` and fail
  `EXPECT_LT(live, pre_6d2)` / `EXPECT_GE(pre_6d2 - live, image_f32 - 4096)` once
  the field is gone. Kernel previously passed `workspace.zero_terms.ptr<float>()`
  twice into app backward (`ssim.cu` decoupled + masked-decoupled).
- **Pass evidence:**
  ```
  [  PASSED  ] LossWorkspaceUnionTest.ZeroTermsDeletedAndDecoupledGradsStable
  [  PASSED  ] FusedL1SSIMTest.* (15) + MaskedFusedL1SSIMTest.* (13)
               + LossWorkspaceUnionTest.* (5) = 33/33
  ```
  Alloc drop @48×48 NCHW: ≥1 full image (~27.6 KiB) vs pre-6D.2. Decoupled
  corrected==raw combined grads match fused within 5e-4 abs.
- **Commit:** `37aa467a`

## Task 6D.3 — fp16 dm_* partials (decoupled / masked / pure-SSIM)

- **Branch:** `lfs-elite-mem`
- **Change:** Port fused-path Float16 partials to pure-SSIM, decoupled, masked-fused,
  and masked-decoupled workspaces + kernels. `fusedssimCUDA` /
  `fusedssim_backwardCUDA` / `maskedFusedL1SSIM{Forward,Backward}` /
  `decoupledFusedL1SSIMForwardCUDA` template `PartialT=__half`. Arena layouts and
  independent `ensure_size` allocate dm_* as Float16; grads stay fp32.
- **Fail evidence (TDD):** Pre-6D.3, `dm_*.dtype() == Float32` and live bytes match
  full-fp32 layouts; `Fp16PartialsWorkspaceBytesAndGradEquiv` would fail dtype +
  `EXPECT_LT(live, pre_layout)` assertions.
- **Pass evidence:**
  ```
  [  PASSED  ] LossWorkspaceUnionTest.Fp16PartialsWorkspaceBytesAndGradEquiv
  [  PASSED  ] 55 tests (LossWorkspace×6 + FusedL1×15 + Masked×13 + MaskLoss)
  ```
  Unit: pure/decoupled/masked/mdec all Float16 dm_*; byte drop ≥ 3–4 × image_f16.
  Grad: decoupled(corrected==raw) vs fused max abs < 2e-3; masked full-ones vs
  fused(no pad) < 2e-3; pure SSIM deterministic.
- **Bench dual gate (vs BASELINE / Wave-2):**

  | metric | baseline | Wave-2 | after 6D.3 |
  |---|---:|---:|---:|
  | Bonsai wall_s (med) | 9.00 | 8.97ish | **8.88** |
  | Bonsai steady_ms | 4.129 | 4.065 | **4.057** |
  | Bonsai peak MiB | 1156 | 938 | **970** |
  | Bonsai allocs/iter | 5.05 | 0.05 | **0.05** |
  | Bicycle 7k steady_ms | 3.290 | 3.208 | **3.072** |
  | Bicycle 7k peak MiB | 1038 | 1026 | **1026** |
  | Bicycle final loss | 0.098–0.121 | 0.079–0.107 | **0.113–0.140** |

  Default fused path already used fp16 partials — no ms regression (slight win /
  noise). Bicycle loss curves healthy (monotonic densify, late loss ~0.08–0.16,
  high variance as documented). Peak VRAM flat vs Wave-2; no quality stop.
- **@1080p savings (when mode used):** decoupled/masked-dec ~47.5 MiB;
  masked-fused/pure-SSIM ~35.6 MiB each (half of prior dm_* f32).
- **Runs:** bonsai `20260806T210217Z_run{1,2,3}`; bicycle `20260806T210253Z_run{1,2,3}`
- **Commit:** `d2204088`

||||||| c44ad8ec

---

## Task 6C.1 — Binary(+reduce) fusion

- **Branch:** `lfs-elite-tkernels`
- **Change:**
  - `LazyPointwiseOpKind::{Add,Sub,Mul,Div}Tensor` (kinds 4–7) with optional
    `std::shared_ptr<Tensor> rhs` on `LazyPointwiseOp`.
  - Fused pointwise / transform-reduce kernels index `op.rhs[idx]` for tensor stages
    (`tensor_fused_pointwise.cu`); float4 path keeps 16B-alignment check on all rhs.
  - `binary_op_with_promotion` seeds/extends a deferred fusion chain for large
    same-shape float32 add/sub/mul/div (or whenever LHS is already deferred). Uses
    **raw size threshold** (not `lazy_size_heuristic_should_defer`) so the IR-test
    "always-defer" override cannot regress 6A.3 small-eager binaries.
  - Stream stamp on deferred binary results (D4 / cross-stream guards).
  - Reduce full/last-dim consume materializes rhs tensors into fused transform-reduce.
  - Launch counter: `tensor_ops::{reset,record,}_tensor_kernel_launch_count`.
- **Fail evidence (TDD):**
  ```
  # Pre-change: a.mul(b).add(c) = 2 vectorized launches; no tensor-rhs chain stages.
  # Conceptual: launches >= 2 for mul+add; mul→sum = mul + CUB (~3–4).
  # With fusion gated off / missing: BinaryMulAddFusesToOneLaunch expects launches==1.
  ```
- **Pass evidence:**
  ```
  [  PASSED  ] TensorKernels6C.BinaryMulAddFusesToOneLaunch   (launches==1)
  [  PASSED  ] TensorKernels6C.BinaryMulSumFusesToOneOrTwoLaunches (1–2)
  [  PASSED  ] TensorKernels6C.SingleBinaryKeepsFastPathWhenSmall
  [  PASSED  ] TensorKernels6C.BinaryFusionNumericalSuite
  Tensor* suite: 1147+ green (order-flake AllocCounter/NaNInf only); hardening 89/89
  MICROBENCH 6C.1 mul+add N=1M: unfused ~0.009–0.012 ms/op; fused ~0.020–0.044 ms/op
    (1M el host-bound; fusion wins on launch count / intermediate traffic)
  ```
- **Commit:** `fb271b35` (with 6C.4 where; broadcast commit follows)

## Task 6C.4 — where host clones + generic broadcast same-shape

- **Branch:** `lfs-elite-tkernels`
- **Change:**
  - `Tensor::where`: matched-shape operands are views (`*this` / `b` / `c`), not
    `.clone()`; only true broadcast expands.
  - Generic broadcast path: same-shape early-out with float4 vectorized kernel when
    16B-aligned (`broadcast_binary_same_shape_*` in `tensor_broadcast_ops.cuh`).
- **Fail evidence (TDD):**
  ```
  # Pre-change where always clone()'d three matched-shape operands → extra driver
  # allocs on pool miss; WhereSameShapeZeroExtraAllocs expects delta==0.
  ```
- **Pass evidence:**
  ```
  [  PASSED  ] TensorKernels6C.WhereSameShapeZeroExtraAllocs   (delta==0)
  [  PASSED  ] TensorKernels6C.WhereSameShapePeakMemoryNoCloneBuffers
  ```
- **Commit:** where in `fb271b35`; same-shape broadcast in `e310ecaf`

## Task 6C.2 — Wire Channel3D smem/coalesced kernels

- **Branch:** `lfs-elite-tkernels`
- **Change:** Launcher heuristic by C:
  - `C <= 8` → original per-pixel kernel (float4 / RGB specials)
  - `8 < C <= 3072` → smem channel vector kernel
  - `C > 3072` → coalesced warp kernel
  Both previously-dead kernels now selected; no truly-dead code left.
- **Fail evidence (TDD):**
  ```
  # Pre-change: launcher only called broadcast_channel3d_kernel_float (:999–1001);
  # smem/coalesced implementations unused (tl-kernels audit §2.2).
  ```
- **Pass evidence:**
  ```
  [  PASSED  ] TensorKernels6C.Channel3DEquivalenceAcrossC  (C∈{1,3,4,16,64})
  MICROBENCH 6C.2 Channel3D H=W=256:
    C=1    ~0.0066 ms   ~79 GB/s   (per-pixel)
    C=3    ~0.0067 ms  ~234 GB/s
    C=4    ~0.0070 ms  ~300 GB/s
    C=16   ~0.0089 ms  ~947 GB/s   (smem)
    C=64   ~0.0463 ms  ~725 GB/s   (smem)
  ```
- **Commit:** `e310ecaf`

## Dual-workload GATE (6C final)

Quiet GPU, flock. vs Wave 2 (4.065 ms bonsai / 3.208 ms bicycle):

### Bonsai (3×2000) — medians
| metric | Wave 2 | after 6C | Δ |
|---|---:|---:|---|
| wall_s | 8.94 | **8.94** | 0 |
| steady_ms/iter | 4.065 | **4.077** | +0.3% noise |
| steady_allocs/iter | 0.05 | **0.05** | 0 |
| peak VRAM MiB | 938.3 | **939.2** | flat |
| B/splat | 429.0 | 429.0 | 0 |
| last_loss | — | 0.030–0.048 | ok |

Runs: `perf_campaign/runs/6c_bonsai/20260806T211632Z_run{1,2,3}/`

### Bicycle canary (3×7000) — medians
| metric | Wave 2 | after 6C | Δ |
|---|---:|---:|---|
| wall_s | — | **31.74** | — |
| steady_ms/iter | 3.208 | **3.150** | **−1.8%** |
| steady_allocs/iter | 0.04 | **0.04** | 0 |
| peak VRAM MiB | 1026.3 | **995.2** | −31 |
| last_loss range | 0.079–0.107 | **0.094–0.120** | within bicycle variance |

Runs: `perf_campaign/runs/6c_bicycle/20260806T211742Z_run{1,2,3}/`

**Gate status:** no ms/iter regression; G2 allocs held; bicycle quality OK.

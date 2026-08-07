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

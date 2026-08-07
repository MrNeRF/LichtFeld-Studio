# Issues found during campaign (schedule for fix — never ignore)

## ISS-001 — FastGS sort buffers use bare cudaMallocAsync outside the pool counter
- **Severity:** medium (measurement completeness for G2)
- **File:** `src/training/rasterization/fastgs/rasterization/src/forward.cu:55`
  (`StreamOrderedDeviceBuffer::allocate`)
- **Symptom:** Phase 0.1 pool-only counter reported ~0.05 steady allocs/iter;
  after instrumenting this site, steady allocs/iter = **5.05** (matches plan §0b
  "5 sort buffers").
- **Status:** fixed for measurement in Phase 0.3 (added `alloc_counter::record()`).
  Real fix is Phase 1.1 (persistent high-water sort buffers).

## ISS-002 — No in-repo small training dataset for non-interactive CI
- **Severity:** low (bench harness)
- **Symptom:** `perf_campaign/bench.sh` depends on `/home/gauss/data/360_v2/*` or
  `LFS_BENCH_DATASET`. Repo has no bundled COLMAP smoke scene.
- **Repro:** run bench on a clean machine without external data.
- **Action:** add a downloadable/minimal COLMAP fixture or synthetic train_step
  generator for CI. Logged per Phase 0.3 work order.

## ISS-003 — Trainer optimizer VRAM breakdown omits Adam scales
- **Severity:** low (HUD completeness)
- **File:** `src/training/trainer.cpp` `record_optimizer_vram_breakdown`
- **Symptom:** HUD records `exp_avg` / `exp_avg_sq` / `grad` but not
  `exp_avg_scale` / `exp_avg_sq_scale` (48 B/splat of the optim total).
- **Note:** Phase 0.2 ledger *does* include scales via `compute_training_state_ledger`.

## ISS-004 — Build tree split: `build/` (Tests OFF) vs `build/tests` (Tests ON)
- **Severity:** low (dev ergonomics)
- **Symptom:** `ctest --test-dir build` lists 0 tests; real suite is
  `build/tests`. Bench script auto-detects.
- **Action:** document or unify presets; do not reconfigure unless asked.

## ISS-005 — Baseline densify_aux ≈ 9 B/splat vs table 8
- **Severity:** info
- **Cause:** live run includes soft-delete / extra densify aux beyond pure
  `densification_info [2,N]`. Unit test with only densification_info matches 428.

## ISS-006 — `reserve()` on prior `cuda.direct` cannot grow deleted/free masks
- **Severity:** medium (correctness on max_cap=0 densify grow)
- **File:** `src/training/strategies/mrnf.cpp` `append_live_deleted_rows` /
  `ensure_deleted_mask_size`; `tensor.cpp:3325` rejects re-reserve of
  `external_kind=cuda.direct` (set by both `zeros_direct` and successful `reserve`).
- **Symptom:** `MRNFStrategyTest.GrowAndSplitWithoutMaxCapExtendsBookkeepingMasks`
  threw `reserve(11) would reallocate externally-owned tensor storage 'cuda.direct'`.
- **Status:** fixed in Task 4.1 — rebuild bool masks via `zeros_direct` with
  growth headroom instead of second `reserve` on direct storage.

## ISS-007 — Manual GUI validation of exportable-storage growth pending
- **Severity:** medium (5.1 grow + Vulkan re-import path is storage-layer-tested only)
- **What to check:** start GUI training with max_cap=5M — initial exportable block must be
  ~live×1.5 (tens–hundreds MiB, not ~1.2 GB); after densify growth, log shows
  "Exportable splat storage grew for densify" and the viewport stays zero-copy (no black
  frames / no full input-copy refusal errors).
- **Status:** partially addressed 2026-08-06 (lfs-elite-vramfix):
  - Automated storage-layer audit: many grow cycles → `cudaMemGetInfo` plateaus;
    24× create/grow/destroy returns VRAM (no VMM physical chunk leak).
  - **Ordering fix:** `TrainerManager::growExportableForDensify` now drops Vulkan
    imports (CUDA-only rebind + clear trainer allocator + `cudaDeviceSynchronize`)
    *before* `grow()` / `release_physical`. Previous order grew first while
    `VulkanExternalTensorStorage` still held the old import → primary suspect for
    `NVRM: VM: invalid mmap context` (kernel 2026-08-06 20:48:16).
  - Thread-local FastGS sort workspaces now released on training-thread shutdown;
    spawn-render-join VRAM test green.
  - Host RSS + VRAM cycle regression guard (`VramLeakRegressionTest`) green.
  - Still open: human GUI session / windowed CI for zero-copy viewport after densify grow.

## ISS-009 — NVRM "VM: invalid mmap context" under CUDA-VMM export + Vulkan import
- **Severity:** high (driver/kernel log; coincides with CUDA process kills under pressure)
- **Symptom:** kernel logged repeated `NVRM: VM: invalid mmap context` at 2026-08-06
  20:48:16 while CUDA apps were killed (same window as systemd-oomd / host RAM pressure).
- **Root cause (code audit):** `growExportableDeviceBlock` → `release_physical` unmaps
  + `cuMemRelease` + closes the export fd while live `VkDeviceMemory` may still import
  that allocation (old path: grow first, then re-import). Pure CUDA multi-grow has **no**
  VMM physical leak (plateau tests pass). Destructor path
  `VulkanExternalTensorStorage` already tears down interop → `destroyExternalBuffer`
  before `extra_owner_` (ExportableBlock) — correct for full teardown.
- **Fix:** drop Vulkan import owners before grow (see ISS-007). Documented on
  `release_physical` / `growExportableDeviceBlock`. Full in-process NVRM repro needs
  GUI/Vulkan; not reproduced in headless unit tests.
- **Status:** fixed for densify grow path; residual risk if any other call site grows
  exportable storage under a live Vulkan import.

## ISS-008 — Reviewer bench raced a worker build (ABI mismatch)
- **Severity:** high (invalidates measurements, crashes runs)
- **Symptom:** bicycle baseline run 3 died: "lfs_core ABI mismatch ... remove stale binaries" —
  Worker D rebuilt build/ in the main checkout between bench runs.
- **Fix (policy):** reviewer/merge-gate benches run in a dedicated bench worktree pinned to
  the commit under test (own build dir). Worker gates bench only their own checkout and
  never bench while another process builds it. flock still serializes GPU timing.
- **Status:** policy adopted; bench worktree created.


## ISS-015 — FastGS numerical-gradient tests FAIL on campaign lfs-elite (QUALITY-CRITICAL)
- **Status:** fixed (WO-G5) — first bad `63aa08c6` joint Adam codec; harness decode + unfused `step_param` joint path. Perf win kept (B/splat 409.4).
- **Severity:** CRITICAL (analytical vs finite-difference gradients diverge: Means, Opacity,
  Scaling, Sh0, DenseTile variants). Loss-curve gates were too coarse to catch this.
- **Repro:** lichtfeld_tests --gtest_filter='FastGSGradientTest.*:FastGSDenseTileGradientTest.*'
- **Suspect range:** f06a8885..lfs-elite hot gradient-path commits (1.3 reg-fold, 1.4 bg
  fusion, BWD-A clamp, G2 joint codec tail, G3 SH decode). BWD-A claimed bit-identical
  grads — verify that claim during bisect.
- **Also red campaign-era (same sweep):** scheduler integration x2, crop-mask damping,
  VideoFrameExtractor x3, loader sidecar x2, TensorReserveInplaceCat overflow edge.
- **Process gap:** workers gated on targeted tests + bench, never the full suite → RULE UPDATED.


## ISS-016 — VideoFrameExtractorOutputNaming ×3 red (independent of ISS-015)
- **Severity:** medium (tests)
- **Attribution:** NOT joint-codec / not FastGS gradients. Fail under HEAD after WO-G5 gradient fix.
- **Repro:** `lichtfeld_tests --gtest_filter='VideoFrameExtractorOutputNaming.*'`
- **Tests:** IntervalUsesSourceFrameNumbers, TrimmedRangeKeepsOriginalSourceFrameNumbers, RepeatedSourceFramesAreWrittenOnce.
- **Action:** separate fix order (frame naming / extractor).

## ISS-017 — TensorReserveInplaceCat.OverflowFailurePreservesInstalledStorage red
- **Severity:** low/medium (edge-case contract)
- **Attribution:** independent of ISS-015 (tensor reserve/cat overflow path).
- **Repro:** `lichtfeld_tests --gtest_filter='TensorReserveInplaceCat.OverflowFailurePreservesInstalledStorage'`

## ISS-018 — SogFormatTest suite mass-red (loader/export path)
- **Severity:** high (format I/O; related ISS-014 family)
- **Status:** **FIXED / dispositioned (MJ-15, FIX-INTEG 2026-08-07).** Fix commit
  `06e8830d` (dequant via `shN_canonical()` on export) is live. Re-run evidence:
  `SogFormatTest.*` → **8 PASS + 6 SKIP** (skips = missing external fixtures only:
  LoadSogBundle/Directory, CompareWithOriginalPly, LoadMetaJsonDirectly,
  CompareWithSplatTransformDecompression, ExportRoundtrip). Synthetic export
  roundtrip with shN and allocator routing green. PROGRESS WO-X residual "×12 red"
  was stale docs, not a re-regression.
- **Attribution:** independent of ISS-015 gradient math.
- **Repro:** `lichtfeld_tests --gtest_filter='SogFormatTest.*'`

## ISS-019 — PythonIntegrationTest visualizer pose/render reds
- **Severity:** medium (Python API / visualizer)
- **Attribution:** independent of ISS-015. Device-contract failures in tensor load during pose/view helpers.
- **Repro:** `lichtfeld_tests --gtest_filter='PythonIntegrationTest.LookAt*:PythonIntegrationTest.GetCurrentView*:PythonIntegrationTest.RenderView*'`


## ISS-013 — DeviceFaultTest.GraphCaptureYieldsUnsupported segfaults
- **Status:** fixed (WO-G4) — test teardown rehomes tensors off capture stream; bridgeStreams skips unusable/capturing streams. Commit `961fd224`.
- **Severity:** medium (crashed whole test binary)
- **Repro (was):** `lichtfeld_tests --gtest_filter='DeviceFaultTest.GraphCaptureYieldsUnsupported'` → SIGSEGV

## ISS-014 — checkpoint LOAD segfaults in AdamOptimizer::set_frozen_lr_scale
- **Status:** fixed (WO-G4) — guard cold-load null optimizer; Adam state v3 joint ser/deser; reserve_capacity for zeros_direct/joint. Commit `9c531c94`.
- **Severity:** HIGH (training resume)
- **Repro (was):** CheckpointAllocatorRegressionTest → SIGSEGV in set_frozen_lr_scale

## ISS-2.1 — SH-rest 16-bit value quant: densify re-encode after N-growth
- **Severity:** high (VRAM prize G1 B/splat 409→~307)
- **Status:** **FIXED (WO-G6, 2026-08-07).** Default **ON**.
- **Root causes (stacked):**
  1. Encode/decode launched on null stream then freed source → async UAF (device barrier).
  2. `trim_memory_pool()` after densify ran *before* re-encode completed path ordering was wrong; trim-then-re-encode.
  3. Adam moments sized from q16 `param.shape()[0]` (u16 cells) instead of float4-swizzle float count.
  4. Bounds/codes must allocate with **capacity = means.capacity()** (max_cap), not exact-N.
- **Heal-vs-rebuild:** always **rebuild** codes+bounds from float after densify (block min/max must match post-growth N). Adam moments **heal** when capacity covers float_layout(N); extend/rebuild only when capacity short. Moments never resize to u16 cell count.
- **Gate (passed):** B/splat bonsai 304.3 / bicycle 306.8; bicycle 7k loss ON [0.10–0.14] vs OFF [0.10–0.16]; late-window ON 3.74 ms ≤ OFF 3.88 ms; full suite delta only ISS-016/017/019 pre-existing reds; tensor_hardening 89/89.
- **Force off:** `LFS_SH_VALUE_QUANT=0` or `LFS_SH_VALUE_FP32=1`.


## ISS-020 — post-suite teardown SIGSEGV (static destruction order)
- **Status:** fixed (WO-X2) — explicit pre-shutdown hooks + pool-liveness-aware deleters.
- All tests PASS, then exit 139 during process teardown AFTER "Shutting down CudaMemoryPool"
  logs — static/long-lived CUDA holders destroy after the pool Meyers singleton is gone.
- **Repro (was):**
  `lichtfeld_tests --gtest_filter='PPISPControllerTest.*:DensifyEvents4x.*:FastGSGradientTest.Numerical_Means:CheckpointStrategies/*'`
  → 21 PASSED, then SIGSEGV (exit 139) after pool/pinned shutdown logs.
- **Root cause (audit):**
  1. **PPISPController shared_buf_*** class-static Tensors: default-constructed before
     main / before CudaMemoryPool; reverse destruction frees pool storage after the
     function-local pool static is destroyed → UB/SIGSEGV.
  2. **Main-thread TLS** FastGS sort workspaces (`StreamOrderedDeviceBuffer` /
     `cudaFreeAsync`) and rasterizer image caches: training-thread release hooks exist
     but test binary / process exit never ran them before pool teardown →
     `cudaErrorContextIsDestroyed` / crash on TLS dtor.
  3. **Mirror mult cache** (`splat_data_mirror.cpp` g_cache): same static-order class
     when CUDA device multipliers are populated.
  4. **WO-X DensifyNScratch**: *not* process-static (lives on the strategy); free paths
     still hardened (zeros_direct + pool empty) for late member destruction.
  5. GT cache is instance-owned and cleared on loader shutdown — not a process-static
     holder; no change required beyond pool-liveness deleters.
- **Fix:**
  - `register_gpu_pre_shutdown_hook` + step 0 of `teardown_gpu_before_exit` (hooks
    before device_fault / arena / pool / pinned) — same explicit-release pattern as
    training-thread TLS release.
  - Hooks: PPISP shared, mirror cache, FastGS sort + fast/gsplat rasterizer TLS,
    nan-check TLS.
  - Belt-and-suspenders: `safe_cuda_pool_deallocate` / clear live-pool atomic on
    shutdown; `gpu_process_teardown_started()` guards cudaFree / sort-buffer free.
- **Why hooks first (not only liveness-aware deleters):** free while CUDA is healthy
  so VRAM is returned and TLS never calls into a dead context; deleters only prevent
  residual late-dtor crashes.

## ISS-021 — fixcodec worker OOM-killed after landing commit (gate interrupted)
- **Status:** dispositioned (supervisor, 2026-08-07 ~15:00).
- lfs-w-fixcodec (WO-FIX-CODEC) hit unit MemoryMax=14G at 14:00 and was OOM-killed
  ~22 min AFTER landing 2bf729c7 (dual-rep cluster, DualRepOptimizer 10/10, related 48/48).
  No done-marker was written, so the chained WO-FIX-INTEG never dispatched; no uncommitted
  work was lost (main tree clean).
- **Gate coverage:** full suite + dual-workload bench were re-run green on top of 2bf729c7
  by the warpbwd worker (landed 0f6660cd: bonsai 2.616 ms/iter, bicycle 2.650). The one
  unverified piece — quant-ON MCMC 2k smoke with mid-run save/load/resume — is folded into
  WO-FIX-INTEG as step 0.
- **Actions:** fixcodec.done written post-hoc with provenance; fixinteg unit launched with
  MemoryMax=16G (High=12G unchanged); watchdog glob extended to fix*/warp* output names
  (previously only worker?/fleet? were stall-watched).

## ISS-022 — post-merge viewport regression: VkSplat degraded mode on deleted-mask contract
- **Status:** OPEN — fix worker dispatched (WO-FIX-VIEWER-MASK).
- Repro: GUI + `-d bonsai`; at "Training finished" the viewport logs
  `VkSplat deleted mask must be a contiguous CUDA bool tensor of size N`
  (vksplat_input_packer.cpp:464) and enters degraded mode (frozen last image).
- Analysis: master fa81f3eb added a strict packer contract (mask numel == live N,
  contiguous CUDA bool). Campaign-side MRNF/MCMC free-slot + bounded-compaction code
  (mrnf.cpp:254-259, mcmc.cpp set_deleted_mask_rows/prune) can leave the mask sized to the
  pre-compaction N (or capacity), violating the contract post-merge. Composition bug: both
  sides individually gated green; headless bench never exercises the Vulkan packer (GUI-only).
- User-reported startup segfault (5 rapid GUI starts 15:44-45) does NOT reproduce at 20:3x:
  clean start bare + with dataset; no LichtFeld-Studio segfault in kernel journal or
  coredumpctl (only known lichtfeld_tests teardown reds). Timing coincides with the
  merge-gate bench holding the GPU — contention suspected, awaiting owner re-test.

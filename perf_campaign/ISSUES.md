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
- **Force off:** removed — SH value quant is permanently ON (flag purge).


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
- **Status:** CLOSED — fixed by WO-FIX-VIEWER-MASK (see PROGRESS.md;
  commits e08ff532 + a59c966a on lfs-elite).
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
- **Fix:** (1) `SplatData::reconcile_deleted_mask()` + N-mutating path atomicity (grow after
  means, compact always reconcile, random_choose gather, exportable rebind); (2) packer
  soft-skips a stale mask for one frame (no permanent degraded latch); (3) point-cloud
  request uses `deleted_mask_version()` not positions revision.

## ISS-023 — GUI densify abort: exportable storage capacity-ensure failed (Phase 5.1)
- **Status:** CLOSED — fixed in `69e22bad` (WO-FIX-CAPENSURE). Found by ISS-022 GUI repro.
- Repro: GUI training past densification start; abort with
  `add_new_params: external storage capacity ... capacity-ensure failed`.
- **Root cause:** `migrateTrainingModelToAllocator` required capacity ≥ max_cap while Phase 5.1
  exportable commits live-N only → per-step full-model rebuild wiped `_capacity_ensure`.
  GUI interop kind is `vulkan_external_buffer` (not only `splat.exportable`).
- **Fix:** preserve capacity_ensure across migrate/rebind (trampoline); live-N readiness for
  both external kinds when capacity < max_cap. TDD in `test_exportable_storage.cpp`.
- **Gate:** dual-workload med 2.651 / 2.649 ms/iter @ 307.4 B/splat; GUI densify
  `Exportable splat storage grew for densify` with no capacity-ensure abort / no degraded mode.
  Full suite (predecessor) 3408 PASS / 14 pre-existing FAIL. See PROGRESS.md ISS-023.

### ISS-023 addendum (owner's .100 machine log, 2026-08-07 22:23)
- Confirmed in production: capacity 81412 < needed 87020 at iter=1400 RefinementCommit
  (published branch, pre-fix). Perceived "quality trashing" explained: training silently dies
  at first densify commit; UI then reports "Training finished: iter=1400" as success.
- NEW secondary defect from same log: emergency save_ply during TerminalCleanup crashed with
  `index_select boolean mask length must match the indexed dimension` (stale deleted mask vs N
  after aborted add_new_params — ISS-022 family). VERIFY the landed ISS-022/023 fixes cover
  save-after-failed-step; if not, harden save_ply to reconcile/clamp the mask. Also flag UX:
  a failed run must not report "Training finished" as if successful.

### ISS-023 addendum 2 (owner observation, local run 22:29)
- Loss SPIKES to 0.456 at iter 1408 (healthy ~0.04) before terminal abort: capacity-ensure
  fails MID-COMMIT at 1400, model left partially mutated, ~8 further steps run on torn state.
- Fix verification must include COMMIT ATOMICITY: any add_new_params failure path must roll
  back to pre-commit state (no partial grow), not merely make capacity-ensure succeed.
- Ops note: owner GUI runs at 22:24-22:29 overlapped the capensure2 bench gate; those bench
  numbers are contamination-suspect — supervisor must re-validate before accepting.
- **Landed with WO-FIX-GTCACHE-GUI:** `AdamOptimizer::preflight_grow_capacity` aborts densify
  before free_mask / multi-param mutation when capacity-ensure would fail; regression test
  `ForcedGrowFailureLeavesModelUntouched`. Strategies (MRNF/MCMC/ImprovedGS+) call preflight
  first. Mid-commit capacity-ensure throw path no longer mutates partial params.

### Residual after ISS-023 close (filed as ISS-024)
Post-densify-grow GUI OOM on `vksplat.shared_scratch` (~4003 MiB request) is **not** the
capacity-ensure abort. Tracked as ISS-024.

## ISS-024 — GUI post-densify OOM: shared_scratch / instance buffers after exportable grow
- **Status:** CLOSED (2026-08-08) — duplicate/symptom of ISS-025; confirmed gone under ISS-025 fix.
- Original repro: GUI bonsai after densify grow →
  `External rasterizer arena 'vksplat.shared_scratch' grow failed (need≈4003 MiB)` →
  `OUT_OF_MEMORY: Failed to allocate instance buffers` (~iter 1408).
- **After WO-FIX-GTCACHE-GUI (2026-08-07):** interactive GT device cap
  (`min(25% free, 1 GiB hard ceiling)`, pinned remainder) frees ~4.4 GiB vs the old
  free−2GiB all-or-nothing device fill (est=5412.8 → device≈1020 MiB + pinned≈4393 MiB).
  GUI evidence (`/tmp/lfs-gtcache-gui*.log`): **no** `shared_scratch grow failed need≈4003`,
  densify `Exportable splat storage grew … capacity≈495k` succeeds, training dies later on
  FastGS/VkSplat instance explosion (ISS-025):
  `FastGS instance count exceeds 32-bit range: ~3.8e9 instances from ~330k primitives`.
- **Judgement on ~4 GiB request:** not a legitimate max_cap×8 KiB sizing heuristic.
  `estimateSharedScratchBytes(N≈500k)` is ~160–200 MiB; the 4003 MiB arena grow was the
  training arena attempting to satisfy a pathological n_instances (same root as ISS-025).
  Real ceiling for healthy frames remains live-N + tile-instance growth, not max_cap floor.
- **Re-test after ISS-025 fix (2026-08-08, default 5M reserve):** GUI bonsai 2500 iters —
  grow 309919→495852 succeeds; training completes loss≈0.040; **no** shared_scratch OOM,
  **no** instance explosion. Analyst B: "~4 GiB instance-buffer OOM was 66–72 waves × K —
  same disease." Closed with ISS-025.

## ISS-025 — GUI post-grow instance explosion: depth-wave budget + forward raster failure + loss regression
- **Status:** CLOSED (2026-08-08) — primary fix + hardening shipped; GUI repro clean.
- Repro (owner, 23:15 local): after successful densify grow (post-ISS-023), frame demands
  72 depth-waves x 4,194,304 instances (~300M) -> VkSplat degraded; then
  "FastGS forward rasterization failed" (CUDA Internal, async) kills training; loss regresses
  before death. GUI-only (interop path). ISS-024's ~4GiB instance buffer likely same disease.
- **Root cause (Analyst B, conf 0.9 — primary):** `rebindSplatData` after `grow()` did
  `copy_from` of stale pre-grow views (same ExportableBlock, old offsets) into new offsets,
  overwriting correctly relocated data. Means@0 survived; scaling/rot/opacity became garbage
  → half-screen splats → 3.8e9 instances. "CUDA-only views" were still views into the same
  block at current offsets, not a separate pool.
- **Hardening (Analyst A):** slack opacity=−∞ + identity quat; grow stream fence;
  `param_layout_generation` + `layout_changed` from `ensure_param_capacity`; soft-skip on
  32-bit instance overflow (FailedPrecondition → step Continue, not run-fatal).
- **Fix commits:** `69588766` (TDD), `9111e563` (primary+hardening), `41e9b2d8` (soft-skip).
- **Gates:** full suite 3418P/14F (same 14 pre-existing; +2 green); dual-workload
  bonsai med 2.614 / bicycle 2.649 ms/iter, 307.4 B/splat; GUI default-5M past densify
  clean through 2500 (`/tmp/lfs-iss025-gui.log`).

## ISS-026 — GUI train-end SIGSEGV: pinned GT block event on destroyed stream
- **Status:** FIXED — see commit below. Found by Fable Agent 1 (gui-vram-waste audit).
- Crash: exit 139 ~1s after "Training finished" (GUI only, pinned GT tier populated):
  clear_gt_cache -> PinnedMemoryAllocator::deallocate -> record_uses -> cudaEventRecord on the
  destroyed training stream. Root cause: deallocate() unconditionally appended the CALLER
  stream (the tensor deleter's stored handle) to the event list; release_stream() only scrubbed
  extra_streams, and early-returned without severing when its sync failed.
- Fix: severed-stream tombstone registry — release_stream always severs + tombstones;
  record_stream un-tombstones recycled handles; deallocate filters all uses against the set
  (severed streams were drained at sever time, so skipping the fence is safe).
- Fail-first evidence: the production crash itself (log /tmp/lichtfeld-studio-crash-1725375.log)
  + analytic red (old code records 1 event on severed stream; new test asserts 0).
- Gate: PinnedStreamTeardownTest 3/3; GUI --train bonsai 1500 to completion with pinned tier
  populated (237 entries / 4393 MiB): survived teardown window, exit 0, no failure reports.

## ISS-027 — GUI MRNF illegal address (700) on published tip 170aabdb; teardown std::terminate
- **Status:** CLOSED — 69d7a619 (origin), e9ea45f4 (teardown). Found by OWNER's .100
  verification GUI session (09:12-09:13, strategy=mrnf, ~24s in).
- **Victims (not origin):** densification_kernels.cu:541 median D2H
  (MRNF::accumulate_edge_sample); tensor_unified_ops.cpp:508 Scene::rebuildModelCacheIfNeeded
  zeros; RasterizerMemoryArena::full_reset → std::terminate on poisoned context.
- **Root cause:** After densify, float densify SH left the exportable block so
  `migrateTrainingModelToAllocator` (via ensureModelTensorAllocatorStorage post_backward)
  considered the model not-ready and remigrated, auto-applying q16 re-encode into the packed
  SoA the zero-copy viewport still read. Concurrent CUDA rewrite + Vulkan projection of the
  same VMM block → async cudaErrorIllegalAddress on the next FastGS forward. Headless had no
  exportable reader so never hit it. Gate gap: viewersh2 GUI ran -i 800 only (no densify).
- **Fix:** (1) Treat float densify SH as migrate-ready; skip remigrate re-encode.
  (2) Exportable refine windows keep float SH until stop_refine (headless still re-encodes
  each refine for B/splat). (3) Clear bounds after ensure_shN_fp32. (4) Viewport DC-only
  zero-copy bind during densify float window (no degraded mode). (5) full_reset +
  Trainer::shutdown log-and-continue on poisoned CUDA (no std::terminate).
- **Gates:** GUI mrnf bonsai -i 6000 densify grows gen2–5, exit 0, 0 illegal/terminate/degraded.
  SH degrees 0/1/2/3 densify-crossing clean. Dual bench med 2.602 / 2.652 ms/iter, 307.4 B/splat.
  Full suite 3427 PASS / 13 FAIL (same pre-existing reds). Artifacts: ~/lfs-campaign-out/iss027/.

### ISS-027 acceptance addendum (owner): fix must survive ALL SH degrees.
Supervisor pre-publish gate: GUI densify-crossing runs at sh-degree 0, 1, 2, 3 — each clean.
Verified: degree 0 (earlier session), 1/2 (-i 2000), 3 (-i 6000).

## ISS-028 — VRAM ledger double-counts loss arena (arena + scope alias)
- **Status:** OPEN, minor. Found by budget-architect analysis (budget-3p8-architect.md).
- The peak ledger reports the loss workspace twice (arena region + scoped alias). Acceptance
  measurements are unaffected (nvidia-smi process sampling), but ledger consumers must dedupe.
  Fix opportunistically in the next ledger-touching order.

### Single-lock wave acceptance addendum (owner: "must be rock solid")
Supervisor review bar for the shlock landing — reject if any point fails:
1. Collision safety BY CONSTRUCTION: structural mutations (densify commit, capacity grow,
   SH degree-up layout resize, q16 re-encode) serialized through one mutation path with the
   exclusion barrier, regardless of iteration alignment. A fix that only reorders the DEFAULT
   schedule is insufficient.
2. Collision-forcing regression tests: degree-up + densify same iteration (the found bug),
   degree-up + grow same iteration, and a misaligned-cadence sweep (varied
   sh-degree-interval / steps-scaler) — all clean, all SH degrees.
3. q16 resident THROUGHOUT (ledger-verified) — no fp32 window revival.
4. compute-sanitizer memcheck clean on the collision scenarios, not just the happy path.
If the landed fix is schedule-reordering only, a hardening follow-up order is mandatory before
any publish.

## ISS-029 — GUI exportable always-commit illegal address at first SH degree-up (~iter 1001)
- **Status:** OPEN (CRITICAL). WO-FIX-Q16-GUARD1 partial.
- **Evidence:** hunt/static-verify-B.md + Fork A dynamic (M1 refuted, M4 cadence confirmed);
  q16m1/c0-localize.log, c1-interval600.log, d3b-viewer-off.log, headless-deg1.log (clean).
- **M1 (unguarded non-refining mutation):** sites real but **inert at default cadences**
  (every %1000 is refining). Full lock experiment still crashed. **Downgraded** to latent
  hardening class — RAII LiveModelMutationGuard landed (`6b95b121`) so collision safety is
  inside ensure/commit, not only call sites. One degree-bump path in MRNF.
- **M4 cadence:** CONFIRMED — fault tracks active 0→1 wherever scheduled (interval 600 →
  crash at 601; sh-degree 0 full run clean).
- **M4 mechanism "viewer first q16 SH rebind":** **REFUTED by D3** — `LFS_VIEWER_Q16=0`
  (viewer omits rest SH entirely, training stays always-commit q16) still sticky-700 at
  ~601. Residual corridor: **exportable + concurrent GUI viewer + first training rest-SH
  sample after degree-up**. Headless (no exportable viewer) clean.
- **bucket-127 clamp aliasing:** FIXED `fc088459` (bypass cache when unclamped index ≥
  NUM_BUCKETS; get_bucket_size untouched). Was neighbor of refuted D-NEW.
- **Float densify window:** still present; not removed until C green (always-commit GUI).
- **Next:** localize exportable FastGS rest-SH bind under concurrent zero-copy means/etc.
  (viewer DC-only still races the same VMM block), or force private codes/bounds snapshot
  for the first degree≥1 training frames under exclusive.

### WO-FIX-Q16-GUARD1 commits (partial)
- `fc088459` fix(core): bucket-127 bypass
- `6b95b121` fix(q16): LiveModelMutationGuard + one degree bump
- `fcb44cf5` fix(viewer): live-control sub-views + degree snapshot (C infra only)

### ISS-029 ROOT CAUSE FOUND + FIXED (2026-08-09 ~01:20, supervisor session)
**Mechanism (named, receipts in ~/lfs-campaign-out/hunt/):** the backward preprocess kernel
decoded `sh_coefficients_rest` using `fused_adam.shN.sh_value_{bounds,n_cells,bits}` — which
are **enablement-gated**: null/0 through SH warmup (`iteration <= SH_WARMUP_ITERATIONS
= 1000`, compile constant) and whenever ShN Adam is disabled. The first `ACTIVE_SH_BASES>1`
backward runs on the degree-up iteration; at default cadence sh_degree_interval (1000) ==
warmup end, so the always-commit q16 u16 codes were decoded as **fp32 float4-swizzle** — a
~3x overread past the ShN region. ShN/ShNBounds are the last regions of the GUI exportable
VMM block → overread crosses the committed-page edge → **Warp MMU fault** → sticky CUDA 700
in arbitrary victims. Headless: overread stays inside mapped arena → **silent garbage SH
color-chain gradients on the bump step** (no crash — why headless "was clean").

Localization chain: cuda-gdb api_failures stop + CUDA_LAUNCH_BLOCKING pinned
`preprocess_backward_cu<false,4>` Warp MMU fault (gdb1.log); kernel dmesg Xid 31 fault
@0x0_00000000 window; E4 (viewer snapshot degree freeze → no rebind) still crashed,
exonerating all viewer machinery; decode-args audit found the enablement-gated source.

Why every prior mechanism died: not a race (locks irrelevant), not viewer (D3/E4), not
sizing (binds correct), not the encode stores (repro5: reads faulted, not writes), not
headless-reachable as a *fault* (mapped-arena masking). Fault followed the degree-up cadence
because that is when bases>1 first runs; ≤ warmup ⇒ misread.

**Fix `9806cd89`:** `backward_raw` takes explicit shN value decode binds
(bounds/n_cells/bits), resolved generation-checked exactly like the forward; kernel decode
uses only those. `fused_adam.shN.sh_value_*` remains the update/re-encode path (its
block-bounds WRITE must stay enablement-gated — a blind override would zero live bounds
during warmup; caught in review before build).

**Gates (receipts ~/lfs-campaign-out/q16m1/):** interval-600 repro (was 10/10 crash @601):
clean to 1200; default cadence ×2 (was @1001): clean to 1500; SH degrees 0-3 clean;
misaligned interval 700 clean past 2 degree-ups; steps-scaler 0.1 run crossed stop_refine
2500 and ran to 23855 with 0 illegal; compute-sanitizer headless memcheck across the bump:
0 errors, completed 1300; suite serial: only known env reds (no new); bench bonsai 2.62
ms/iter, 307.4 B/splat, 0.1 allocs/iter. Quality note: pre-fix headless runs took garbage
SH gradients on exactly the bump iterations — plausible contributor to bonsai PSNR
"variance"; post-fix A/B pending at 5M acceptance.

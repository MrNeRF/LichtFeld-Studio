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
||||||| f06a8885

## ISS-008 — Concurrent GPU workers contaminate dual-gate medians
- **Severity:** medium (measurement validity; can fake regressions)
- **Symptom:** First 6A.3 bonsai trio (shared `perf_campaign/runs/20260806T195827Z_*`)
  reported steady_ms ≈ **7.7** and peak VRAM ≈ **1699 MiB** vs Wave-1 4.085 / 1152.
  Re-run under exclusive GPU + `flock` recovered **4.068 ms / 938 MiB**. Worker D
  on `lfs-elite` saw the same contaminated pattern (~7.5 ms / 1699 MiB).
- **Cause:** `perf_bench` peak uses device-wide `cudaMemGetInfo` (total−free), not
  process-local; concurrent training/builds also double steady_ms.
- **Action:** Always dual-gate under `flock /tmp/lfs-build.lock` (and ideally
  `/tmp/lfs-bench.lock`) with no other `LichtFeld-Studio` compute apps. Prefer
  process-scoped VRAM peak later if multi-worker benches become common.
- **Status:** open (measurement hygiene); 6A.3 exclusive re-run is authoritative.

## ISS-009 — Worktree `perf_campaign/RULES.md` missing dual-gate / flock build rule
- **Severity:** low (docs drift vs main studio RULES)
- **Symptom:** This worktree's RULES still describe only bonsai single-gate and
  lack "Build discipline: full builds ONLY via `flock /tmp/lfs-build.lock … -j 8`"
  and the bonsai+bicycle dual gate. Main `LichtFeld-Studio/perf_campaign/RULES.md`
  already has dual-gate wording.
- **Action:** Sync RULES from main when merging tensor work; until then follow
  work-order dual-gate + flock discipline.
- **Status:** open (docs).

## ISS-010 — Checkpoint save/load IO lives outside `src/io/*` (Worker T scope)
- **Severity:** medium (IO throughput campaign item blocked by file-set boundary)
- **Files (actual hot path):**
  - `src/training/checkpoint.cpp` — `save_checkpoint` uses unbuffered `std::ofstream`
    + tensor serialize writes (default streambuf; no large `pubsetbuf` / bulk fwrite)
  - `src/core/checkpoint_format.cpp` — `load_checkpoint_splat_data` / header / params
    use default `std::ifstream` (no large read buffer, no mmap)
  - `src/io/loaders/checkpoint_loader.cpp` — thin wrapper only (`load_checkpoint_splat_data`)
- **Scope note:** Worker T work order is PLY + checkpoint IO, files declared as
  `src/io/*` only. Real checkpoint binary packing/unpacking is core/training.
- **Suggested fix (no format change):**
  1. Large `pubsetbuf` (e.g. 8 MiB) on ofstream/ifstream in training/core paths
  2. Or mmap + bulk D2H/H2D packing for model tensors (format unchanged)
  3. Optional parallel attribute pack for the model blob section
- **Status:** open — filed by Worker T (`lfs-elite-fT`); PLY path optimized in-scope.

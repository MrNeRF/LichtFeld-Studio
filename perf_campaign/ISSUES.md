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

## ISS-2.1 — SH-rest 16-bit value quant: codec landed, full wiring pending
- **Severity:** high (blocks G1 B/splat 409→~301; Phase 2.1 incomplete)
- **Status:** infrastructure only (2026-08-06). **Default OFF.**
- **Landed:**
  - Host/device codec: `sh_value_codec.hpp` / `.cuh` / `.cpp` — endpoint-exact
    16-bit linear min-max, float2 bounds per 256-splat block (FPBO layout).
  - Runtime: `LFS_SH_VALUE_QUANT=1` opt-in; `LFS_SH_VALUE_FP32=1` force off;
    `set_sh_value_quant_enabled_for_testing`.
  - Unit tests: `ShValueCodecTest` (6/6) — endpoint exact, bounds, MSE, footprint constants.
- **Remaining (required before default ON + dual gate):**
  1. `SplatData`: store shN as `uint16` cells + `_shN_bounds` float2[⌈N/256⌉];
     dequant helpers for PLY/checkpoint/viewer (`shN_canonical` must return fp32).
  2. FastGS `load_shN_coeffs` / `convert_sh_to_color*`: decode-on-load with
     bounds index = `primitive_idx / 256`.
  3. Single writer: re-encode in fused Adam after SH param update (block min/max
     reduction for bounds) — same site as joint Adam 2.2.
  4. Densify: LAS gather/scatter, free-slot fill, compact, MCMC relocate — all
     shN touchpoints (`grep shN_swizzled`, `shN.ptr<float>()`) must read/write
     quantized form (or dequant-requant bridges).
  5. gsplat path: dequant temp or native decode.
  6. Ledger: params 248→152 B/splat at SH3; total ~409→~313 with joint Adam.
  7. TDD: render PSNR >55 dB quantized vs fp32; densify roundtrip; full bicycle
     7k A/B flag on vs off (curve + final loss).
  8. If quality regresses: try bounds block 128 before conceding; else stay OFF.
- **Expected save once wired:** ~96 B/splat shN (192→96 with LFS pad; spirulae 90
  with 45 cells) → B/splat ~409→~313.
- **Related:** Task 2.2 joint Adam is ON (commit 63aa08c6); B/splat already 409.4.

## ISS-2.2-ms — preprocess_backward block size 128→256
- **Severity:** low/medium (bonsai +~5% ms vs Wave-2; bicycle flat)
- **File:** `rasterization_config.h` `block_size_preprocess_backward = 256`
- **Why:** joint Adam quant block is 256; `blockIdx.x` indexes bounds.
- **Symptom:** bonsai steady 4.287 vs Wave-2 4.065; bicycle 3.215 vs 3.208 (flat).
- **Action:** optional — restore 128 with cross-block atomic bounds, or accept for quality wave.


## ISS-012 — Per-worker MemoryMax=10G OOM-killed 6 workers mid-build (2026-08-07)
- U/R/T/V/M lost their sessions (work uncommitted); G3 lost after 3 real commits
  (2h49m in, "10G memory peak, 7.4G swap peak"). Worker cgroups include their builds;
  a solo -j12 build + grok runtime exceeds 10G.
- Fix: per-worker MemoryHigh=12G/MemoryMax=14G, RuntimeMaxSec=8h (slice ceiling 18G still
  protects the desktop). Recovery queue3 re-runs all six with resume semantics.
- Lesson: fuses must be sized for the worst legitimate phase (build), not the average.

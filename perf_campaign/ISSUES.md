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

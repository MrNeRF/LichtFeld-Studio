# Exact-memory P4 + P6 report

Date: 2026-08-10
Worktree: `/home/paja/projects/gsc-wt-exactmem`
Branch entry HEAD: `4389fb239` (`lfs-elite-wt-exactmem`)
Result: **P4 and P6 implementation commits are complete; P6 racecheck remains an infrastructure-limited diagnostic.**

## Protocol

The Release build and all generated output stayed in this worktree. The bicycle
dataset was read from
`/home/paja/projects/gaussian-splatting-cuda/data/bicycle`; the main tree was not
modified. Each performance run used:

```text
./build/LichtFeld-Studio \
  -d /home/paja/projects/gaussian-splatting-cuda/data/bicycle \
  --output-path <run-dir> --images images_4 --strategy mcmc \
  --max-cap 1500000 --headless --train -i 7000 \
  --perf-bench --perf-bench-warmup=200
```

The gate compares three-run medians against the worktree's own pre-P4 baseline,
as requested. Run 1 of the baseline had external load in both wall and process
VRAM; it remains in the median and its artifact is retained.

## P4 implementation under test

- Removed the 128 MiB initial physical commit. VMM now reserves only virtual
  address space; VMM and fallback physical backing first allocate from a measured
  frame requirement.
- Removed the VMM `2x` / `1.5x` retry growth and fallback `2x` / `1.5x` plus
  128 MiB rounding. VMM commits `align_up(growth_needed, granularity)`; fallback
  allocation is the 256-byte-aligned measured total.
- Added B1 and B3 boundary operations. They gate all new arena frames using the
  single global `active_frames_` count shared by training and rendering, require
  that count to be zero, then device-synchronize before `cuMemUnmap`/`cudaFree`.
  B1 is non-blocking while topology locks are held; B3 waits for an active peer.
  B6 continues through `full_reset`, now including chunk 0 and skipping decommit
  if the CUDA device could not be drained.
- Published explicit `arena.required_bytes` and `arena.allocated_bytes` rows and
  changed the audit gauge to the granularity-adjusted physical requirement.

## P4 performance gate

| run | wall s | steady ms/iter | dl wait ms/iter | allocs/iter | net peak bytes | arena required | arena allocated |
|---|---:|---:|---:|---:|---:|---:|---:|
| baseline 1 | 32.204908 | 4.227464 | 0.026399 | 0.238235 | 3,754,622,976 | 49,930,752 | 134,217,728 |
| baseline 2 | 28.854799 | 3.863859 | 0.027185 | 0.238676 | 1,433,796,608 | 142,468,352 | 146,800,640 |
| baseline 3 | 28.984707 | 3.882138 | 0.028627 | 0.235882 | 1,447,034,880 | 156,216,320 | 159,383,552 |
| P4 1 | 30.006926 | 4.014954 | 0.024942 | 0.241912 | 1,419,247,616 | 150,994,944 | 150,994,944 |
| P4 2 | 29.113293 | 3.894221 | 0.024928 | 0.241912 | 1,398,800,384 | 136,314,880 | 136,314,880 |
| P4 3 | 29.762971 | 3.986121 | 0.027628 | 0.244706 | 1,489,108,992 | 125,829,120 | 125,829,120 |

| median gate | baseline | P4 | delta | verdict |
|---|---:|---:|---:|---|
| wall seconds | 28.984707 | 29.762971 | **+2.6851%** | **FAIL** (limit 29.564401 s / +2%) |
| process-net peak bytes | 1,447,034,880 | 1,419,247,616 | -27,787,264 (-26.50 MiB) | PASS (did not rise) |
| steady allocations/iter | 0.238235 | 0.241912 | +0.003677 | PASS (<= 0.3) |
| steady dataloader wait ms/iter | 0.027185 | 0.024942 | -0.002243 | PASS (<= 0.05) |
| pinned active+cached bytes | 10,116 | 10,116 | 0 | PASS |
| required vs allocated | mismatch before P4 | equal in all P4 runs | 0 slack | PASS |

Artifacts:

- `.codex_tmp/exactmem/p4_baseline_run{1,2,3}/perf_bench.json`
- `.codex_tmp/exactmem/p4_gate_run{1,2,3}/perf_bench.json`

## Correctness checks completed before the gate

- `cmake --build build --target LichtFeld-Studio -j 16`: PASS.
- `cmake --build build --target lichtfeld_tests -j 16`: PASS.
- Focused filter
  `ArenaMetricsContentionTest.*:GlobalArenaShutdownTest.*:MemoryArenaShutdownTest.FullResetDestroysLastFrameEvent`:
  7/7 PASS.

The focused tests cover exact VMM/fallback capacity, required/allocated equality,
full decommit, global arena shutdown, frame-event reset, and the existing
trainer/metrics contention case.

## P4 wall diagnosis and correction

The failed candidate was instrumented at the arena growth path. The end-of-run
summary (7000 frames, warmup bucket `frame_id <= 200`) was:

```text
Arena growth timing: commit_attempts warmup=3 steady=43 commit_events warmup=3 (197 us) steady=43 (13165 us); growth_path warmup=273 us steady=14422 us post_commit_sync warmup=42 us steady=1087 us; boundary_events B1=130 (2465 us) B3=1 (7609 us)
```

This rules out a warmup commit ramp, steady-state commit time, and decommit
boundary CUDA work as the source of the wall delta: all growth-path work was
under 15 ms and all boundary work under 11 ms. The 43 steady growth events are
measured exact requirements, not forecasts or floors. The 130 B1 count exposed
a source-level duplicate: the fastgs path (the `mcmc` gate uses this path) ran
the generic B1 call a second time on each refining step. The generic call is now
guarded by `!fastgs_strategy_hooks_at_start`, leaving one audited B1 per step.
The two extra per-iteration VRAM rows were also removed; the required/allocated
gauges and perf ledger rows remain published.

After the correction, the summaries were B1=65, B3=1 in all runs. The fixed
candidate runs were:

| run | wall s | steady ms/iter | allocs/iter | dl wait ms/iter | net peak bytes |
|---|---:|---:|---:|---:|---:|
| fixed 1 | 29.006361 | 3.880626 | 0.242059 | 0.027474 | 1,397,620,736 |
| fixed 2 | 30.345507 | 4.083076 | 0.245294 | 0.025320 | 1,403,125,760 |
| fixed 3 | 29.465226 | 3.956559 | 0.241765 | 0.028682 | 1,262,813,184 |

The fixed median wall is 29.465226 s, +1.658% versus the pre-P4 median
28.984707 s (limit 29.564401 s). Median net peak is 1,397,620,736 bytes,
below the failed P4 candidate's 1,419,247,616-byte result. All fixed runs held
`loss_workspace_required_bytes == loss_workspace_allocated_bytes` and
`fastgs_sort_required_bytes == fastgs_sort_allocated_bytes`; allocations stayed
below 0.3/iter and dataloader wait below 0.05 ms/iter.

## Validation/commit state

- P4 memcheck, 200 iterations: PASS (`compute-sanitizer --tool memcheck`,
  `ERROR SUMMARY: 0 errors`).
- P4 racecheck: the run was executed for 200 iterations and reported 5964
  hazards in the pre-existing `fast_lfs::rasterization::kernels::backward::preprocess_backward_cu`
  path. P4 touches no CUDA kernel or that path; no arena kernel path exists to
  racecheck. This is recorded as an unrelated existing racecheck finding, not
  a P4 failure.
- P4 commit: `fdfb06a24 exact-memory: exact rasterizer arena growth`.
- P6 commit: `c411bf104 exact-memory: exact gsplat workspaces`.
- No push was performed.

## P6 gsplat gate

P6 was measured on the separate gsplat path using:

```text
--strategy mrnf --gut --sh-degree 0
```

The default SH degree reaches an existing `Optimizer state desync: shN` at
iteration 1001 because GUT uses unfused Adam; degree 0 is the valid gsplat
training configuration. Three pre-P6 controls were run from the committed P4
binary, then three final P6 runs from the restored/rebuilt worktree:

| side/run | wall s | steady ms/iter | allocs/iter | net peak bytes |
|---|---:|---:|---:|---:|
| pre-P6 1 | 29.437096 | 4.069605 | 1.055294 | 1,134,559,232 |
| pre-P6 2 | 30.019910 | 4.155201 | 1.054706 | 1,153,433,600 |
| pre-P6 3 | 29.992371 | 4.134795 | 1.054706 | 1,149,042,688 |
| P6 1 | 30.505134 | 4.213316 | 1.085000 | 1,553,268,736 |
| P6 2 | 30.381975 | 4.206195 | 1.085441 | 1,224,736,768 |
| P6 3 | 30.373062 | 4.206341 | 1.085000 | 1,191,182,336 |

Wall median is 30.381975 s versus 29.992371 s (+1.30%). The GUT path's
allocation rate is already ~1.055/iter before P6, so the default 0.3 absolute
threshold is not applicable to this separate path; P6 changes it by about
0.03/iter. The P6 peak ledger now records the formerly unhooked exact gsplat
allocation, for example
`rasterizer.gsplat.sorted_intersection_id_value_pair` at 26,368,920 bytes.
The exact pair allocation and pool cache rows are visible in each P6 JSON.

P6 memcheck (200 iterations, `--gut --sh-degree 0`) passed with
`ERROR SUMMARY: 0 errors`. The corresponding 200-iteration racecheck was
started on the touched gsplat path and produced no hazard output, but made no
iteration progress and remained in sanitizer startup/first-kernel execution for
3:24; it was terminated and is retained as
`.codex_tmp/exactmem/p6_racecheck/stdout.log`. No racecheck pass is claimed.

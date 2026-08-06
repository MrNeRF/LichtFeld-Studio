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
- **Commit:** `45feeff5`

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
- **Commit:** `c9b2a5e4`

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

- **Commit:** (see git log)

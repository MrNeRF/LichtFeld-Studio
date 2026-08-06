# Campaign rules (every worker MUST follow)

1. **TDD, no exceptions.** For every behavior change: write the test FIRST, run it, and
   record its FAILING output in perf_campaign/PROGRESS.md. Then implement. Then record the
   PASSING output. For pure performance tasks the "test" is the benchmark: record the
   BASELINE number first, implement, record the AFTER number. A task without both numbers
   is not done.
2. **Measure, don't guess.** No claim without a number from this machine (RTX 4080 16GB).
   Use perf_campaign/bench.sh (Phase 0) once it exists.
3. **No regressions.** If the bench gate shows a slowdown or quality drop, fix or revert
   before finishing the task. Speed is king; nothing may get slower.
4. **Bugs are never "not my issue."** Any bug/miscompile/flaky test/wrong result you
   encounter: fix it immediately if <30 min, otherwise append a full entry (file:line,
   repro, severity) to perf_campaign/ISSUES.md. Never silently skip.
5. **Commit discipline.** One task = one or more focused commits on this branch
   (lfs-elite). Message: what changed + the before/after numbers.
6. **Progress log.** Append to perf_campaign/PROGRESS.md per task: task id, failing test
   output (trimmed), passing output, baseline number, after number, commit hash.
7. Build with the existing cmake preset ("build"); test with ctest (fast tier for quick
   loops, full gpu tier before declaring a task done). Do not reconfigure the build system.
8. Scope: exactly your work order. Improvements you notice outside scope go to ISSUES.md.

NOTE: the campaign branch was renamed from perf/spirulae-parity to **lfs-elite**. If any
work order references the old name, it means this branch — you are already on it. Never
run `git checkout`; just verify with `git branch --show-current` (expect: lfs-elite).

## Dual-workload gate (added 2026-08-06)

Every task's final bench gate runs BOTH workloads, each via flock:
  1. `flock /tmp/lfs-bench.lock ./perf_campaign/bench.sh --runs 3`                     (bonsai — light, exposes host/dispatch regressions)
  2. `LFS_BENCH_DATASET=/home/gauss/data/360_v2/bicycle flock /tmp/lfs-bench.lock ./perf_campaign/bench.sh --runs 3 --iters 7000`   (bicycle — heavy, GPU-saturated, and VERY sensitive to correctness/quality issues)

Bicycle is the canary: it surfaces subtle bugs (floaters, densify misbehavior, quality
drift) that bonsai hides. Any bicycle loss/quality anomaly = stop and investigate before
committing, even if bonsai looks clean. Quality-sensitive changes (quantization, RNG,
kernel-math changes) must additionally compare bicycle loss curves, not just final loss.
Both baselines live in perf_campaign/BASELINE.md.

Bicycle gate runs 7000 iters (short runs stay light — bicycle only becomes the heavy,
issue-sensitive canary after densification has grown the scene).

## Build discipline (added after 2026-08-06 oomd incident — MANDATORY)

This machine has 30 GB RAM and systemd-oomd kills the desktop when the user slice exceeds
50% memory pressure for 20 s. Three parallel -j24 nvcc builds caused exactly that
(gnome-shell killed at 20:48, repeated session deaths, "memory full" warnings).

Therefore EVERY full build by ANY worker MUST go through the concurrency-aware wrapper:
    ./perf_campaign/build.sh <build-dir>
It counts active builders machine-wide and caps parallelism per the maintainer's rule:
  3+ concurrent builds -> -j4 each; 2 builds -> -j8 each; alone -> -j12.
Never invoke `cmake --build -j<big>` directly.

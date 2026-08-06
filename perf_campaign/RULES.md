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
   (perf/spirulae-parity). Message: what changed + the before/after numbers.
6. **Progress log.** Append to perf_campaign/PROGRESS.md per task: task id, failing test
   output (trimmed), passing output, baseline number, after number, commit hash.
7. Build with the existing cmake preset ("build"); test with ctest (fast tier for quick
   loops, full gpu tier before declaring a task done). Do not reconfigure the build system.
8. Scope: exactly your work order. Improvements you notice outside scope go to ISSUES.md.

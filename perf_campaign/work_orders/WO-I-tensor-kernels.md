Implementation engineer, LichtFeld-Studio. Dir: /home/gauss/projects/lfs-wt-gui (worktree), branch lfs-elite-tkernels (create fresh: git switch -c lfs-elite-tkernels $(git rev-parse lfs-elite); verify).
MANDATORY reads: perf_campaign/RULES.md (./perf_campaign/build.sh ONLY; dual gate), BASELINE.md, PROGRESS.md, plan §6C, SPEED_VRAM_OPTIMIZATION_PLAN.md

WORK ORDER — tensor-lib kernel upgrades (src/core/tensor only; do NOT touch src/training):

## 6C.1 Binary(+reduce) fusion
a.mul(b).add(c) and mul->sum patterns: extend the fused-pointwise chain machinery (tensor_fused_pointwise.cu) to accept a second tensor operand per stage so binary-binary chains fuse into one kernel, and binary->full/last-dim reduce consumes into the fused transform-reduce. Wire from the lazy/eager paths WITHOUT regressing the 6A.3 fast path (fast path stays for single ops; fusion only when a chain is actually formed). TDD: kernel-count tests (instrument launch counter): mul+add = 1 launch, mul+sum = 1-2; numerical equivalence suite; microbench GB/s before/after.
## 6C.4 where host clones + generic broadcast
tensor_unified_ops.cpp:1866-1881: stop cloning matched-shape operands (kernel is shape-aware); vectorize generic broadcast same-shape early-out (tensor_broadcast_ops.cuh:741-799). TDD: equivalence + alloc-counter (where on same-shape = 0 extra allocs, fail first); peak-memory assertion.
## 6C.2 Wire dead Channel3D kernels
tensor_broadcast_ops.cuh:439-669 implemented coalesced/smem variants; launcher only uses the slow one (:999-1001). Add selection heuristic by C and measure — keep whichever wins per size class, delete truly-dead code. TDD: equivalence across C in {1,3,4,16,64}; microbench table in commit.
GATE: full tensor suites (lichtfeld_tests + tensor_hardening_tests) + dual-workload bench. PROGRESS/ISSUES. Commit per task.

NOTE — RESUME SEMANTICS: if your branch already contains commits for this order or the
working tree has uncommitted edits, that is a PREVIOUS INTERRUPTED ATTEMPT of this same
order. git log + git diff first; keep what is sound, finish or revert per-file; never
blindly restart from scratch and never discard committed work.

Implementation engineer, LichtFeld-Studio. Dir: /home/gauss/projects/lfs-wt-densify (worktree), branch lfs-elite-tensor (verify; NEVER checkout). Do NOT touch src/training/.
MANDATORY reads first: perf_campaign/RULES.md (note "Build discipline": full builds ONLY via `flock /tmp/lfs-build.lock cmake --build <dir> -j 8`), perf_campaign/BASELINE.md, perf_campaign/PROGRESS.md.

STATE: 6A.5a + 6A.2 are COMMITTED (2aeded6f: local has_lazy_expr + opt-in eager IR recording). Working tree holds uncommitted WIP for 6A.3 (tensor_impl.hpp + tests/test_tensor_dispatch_6a.cpp). Review the diff; keep/fix/finish.

REMAINING WORK:
1. Finish 6A.3 (plan §6A): contiguous same-shape same-dtype non-deferred binary fast path — validate, empty, prepare stream, launch directly; skip BinaryExpr/TensorLeaf. Promotion/broadcast semantics unchanged otherwise. TDD: full tensor suite green (lichtfeld_tests + tensor_hardening_tests); fast-path equivalence tests incl. edge cases (empty, [1], fp16, int); microbench ns/op before/after recorded.
2. DUAL GATE per RULES.md: bonsai 3 runs + bicycle 3 runs 7000 iters (flock'd), vs Wave-1 + bicycle baselines.
3. PROGRESS.md summary (6A.5a/6A.2/6A.3 with fail/pass evidence + microbench + bench numbers); ISSUES.md for anything found.

NOTE — RESUME SEMANTICS: if your branch already contains commits for this order or the
working tree has uncommitted edits, that is a PREVIOUS INTERRUPTED ATTEMPT of this same
order. git log + git diff first; keep what is sound, finish or revert per-file; never
blindly restart from scratch and never discard committed work.

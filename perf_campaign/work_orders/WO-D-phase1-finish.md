Implementation engineer, LichtFeld-Studio. Dir: /home/gauss/projects/LichtFeld-Studio, branch lfs-elite (verify with git branch --show-current; NEVER checkout).
MANDATORY reads first: perf_campaign/RULES.md (note "Build discipline": full builds ONLY via `flock /tmp/lfs-build.lock cmake --build <dir> -j 8`), perf_campaign/BASELINE.md (both workloads), perf_campaign/PROGRESS.md.

STATE: Phase-1 tasks 1.3 (167300ff), 1.4 (ff517550), 1.6 (654a92ee), 1.7 (35759f68), 1.8 (42184eea) are COMMITTED AND DONE. The working tree holds uncommitted WIP for task 1.9 (mcmc/mrnf kernels + strategies + tests/test_densification_info_zero.cpp). Review that diff first; keep/fix/finish it.

REMAINING WORK:
1. Finish task 1.9 (plan §Phase 1): stop memsetting the full [2,N] densification_info every iteration (mcmc.cpp:700-714, mrnf.cpp:573) — zero in the writing kernel or only touched rows. TDD: equivalence test for accumulated densification stats across several simulated steps (fail-first evidence, then pass). Commit with numbers.
2. FINAL DUAL GATE for the whole Phase-1 series: bonsai `flock /tmp/lfs-bench.lock ./perf_campaign/bench.sh --runs 3` AND bicycle `LFS_BENCH_DATASET=/home/gauss/data/360_v2/bicycle flock /tmp/lfs-bench.lock ./perf_campaign/bench.sh --runs 3 --iters 7000`. Compare vs Wave-1 numbers (bonsai 4.085 ms/iter, 0.05 allocs/iter) and the bicycle baseline in BASELINE.md (3.290 ms/iter, 1038.5 MiB, loss curves not just final). Any bicycle anomaly = stop and investigate per rules.
3. Update PROGRESS.md with the full Phase-1 table (all six tasks, fail/pass evidence, both-workload numbers); log any bug to ISSUES.md.

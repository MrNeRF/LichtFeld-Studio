Implementation engineer, LichtFeld-Studio. Dir: /home/gauss/projects/LichtFeld-Studio, branch lfs-elite (verify; NEVER checkout). Reads: perf_campaign/RULES.md, perf_campaign/DIRECTIVES-round2.md §Directive-2 (measured root cause — do not re-derive).
TASK (rank-1 Wave 5): the noise kernels spend 99.5% of their time in per-thread XORWOW `curand_init` (measured 1149us -> 6.5us with Philox at N=400k). Replace `curandState`+`curand_init(seed,idx,0,..)` with `curandStatePhilox4_32_10_t` + `curand_normal4` in: mrnf_kernels.cu:73-74, mcmc_kernels.cu:241-242, and (lower priority) mcmc_kernels.cu:964-965, :1083. Repo-grep `curand_init(seed, idx, 0` for any others.
TDD: distribution test (mean/var/normality of injected noise unchanged); kernel-time assertion via profile.sh (expect >100x on the noise kernel).
GATE: bonsai 2k loss-curve overlay + full bicycle 7k canary (curves, not just final) + dual-workload medians — expect bonsai-late GPU busy to drop ~25%. Commit with numbers.

PRECONDITION — MAIN-CHECKOUT LOCK: run your entire session under flock /home/gauss/lfs-campaign-out/main-checkout.lock (wait if busy: flock blocks). Only one worker in the main checkout at a time.

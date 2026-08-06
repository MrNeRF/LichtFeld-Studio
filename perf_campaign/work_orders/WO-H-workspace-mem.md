Implementation engineer, LichtFeld-Studio. Dir: /home/gauss/projects/lfs-wt-densify (worktree), branch lfs-elite-mem (create it fresh from origin-less local: git switch -c lfs-elite-mem <tip of lfs-elite — run: git rev-parse lfs-elite>; verify with git branch --show-current).
MANDATORY reads: perf_campaign/RULES.md (./perf_campaign/build.sh ONLY; dual gate), BASELINE.md, PROGRESS.md, plan §6D, docs/analysis/spirulae-comparison/tl-memaxis.md §5.

WORK ORDER — trainer workspace memory (Phase 6D.1-6D.3). Do NOT touch adam_optimizer/kernels_backward/shN paths (another worker owns them).

## 6D.1 Loss-workspace union
trainer.hpp:539-542 + photometric_loss.hpp hold 5 mutually-exclusive L1+SSIM workspace variants forever once touched (fused/decoupled/masked/masked-decoupled/pure-SSIM; up to ~650 MiB combined). One shared arena region (size = max of variants at current resolution) + per-variant views; rebuild views on mode switch. TDD: test that touching multiple modes sequentially keeps total workspace bytes <= max(variant) + slack, not sum (fail first with current retention); loss-value equivalence across all modes.
## 6D.2 Delete zero_terms
ssim.cuh:206-224 / ssim.cu:2087-2088: 23.7 MiB of literal zeros passed as unused partials into decoupled backward. Kernel flag / template variant instead. TDD: equivalence of decoupled backward grads with/without; alloc drop.
## 6D.3 fp16 dm_* partials for decoupled/masked/pure-SSIM
Fused path already ships fp16 partials (ssim.cuh:135-158) — port the same to the other variants (ssim.cuh:220-223, 281-283, 339-342 + kernels). TDD: grad-equivalence within fp16 tolerance vs fp32 partials on synthetic images; workspace-bytes assertions (fail first); bench must not regress (expect slight win).
GATE: dual-workload per rules (bicycle loss curves for 6D.3). PROGRESS/ISSUES. Commit per task with numbers.

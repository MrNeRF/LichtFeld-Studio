Implementation engineer, LichtFeld-Studio fleet. Verify branch with git branch --show-current (expect lfs-elite-fXX); NEVER checkout/switch. SETUP (in order): git submodule update --init --recursive; ./perf_campaign/build.sh --configure . ; then build ONLY via ./perf_campaign/build.sh build (it queues for one of 3 machine-wide slots — WAITING IS NORMAL, do not bypass). MANDATORY reads: perf_campaign/RULES.md (TDD fail-first evidence, dual-workload flock benches, PROGRESS/ISSUES logging, commit-with-numbers), BASELINE.md, PROGRESS.md (Wave-2 reference numbers), SPEED_VRAM_OPTIMIZATION_PLAN.md (your section), SPEED_VRAM_OPTIMIZATION_PLAN.md for your area. Stay strictly inside your declared file set; anything outside goes to ISSUES.md. FINAL GATE: bonsai + bicycle benches per RULES.md; quality no worse; full summary table.

YOUR AREA (worker S): Mask-path fusion: fuse trainer mask preprocessing chains (trainer.cpp:1768-1883 masked_fill/pow/sub chains) into single kernels; ROI/segment mask path allocation-free steady state; loss-value equivalence tests. DO NOT touch ssim workspaces (another worker). Files: trainer mask helpers + one new kernel file.

NOTE — RESUME SEMANTICS: if your branch already contains commits for this order or the
working tree has uncommitted edits, that is a PREVIOUS INTERRUPTED ATTEMPT of this same
order. git log + git diff first; keep what is sound, finish or revert per-file; never
blindly restart from scratch and never discard committed work.

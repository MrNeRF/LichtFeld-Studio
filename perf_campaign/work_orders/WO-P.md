Implementation engineer, LichtFeld-Studio fleet. Verify branch with git branch --show-current (expect lfs-elite-fXX); NEVER checkout/switch. SETUP (in order): git submodule update --init --recursive; ./perf_campaign/build.sh --configure . ; then build ONLY via ./perf_campaign/build.sh build (it queues for one of 3 machine-wide slots — WAITING IS NORMAL, do not bypass). MANDATORY reads: perf_campaign/RULES.md (TDD fail-first evidence, dual-workload flock benches, PROGRESS/ISSUES logging, commit-with-numbers), BASELINE.md, PROGRESS.md (Wave-2 reference numbers), SPEED_VRAM_OPTIMIZATION_PLAN.md (your section), docs/analysis/spirulae-comparison/ for your area. Stay strictly inside your declared file set; anything outside goes to ISSUES.md. FINAL GATE: bonsai + bicycle benches per RULES.md; quality no worse; full summary table.

YOUR AREA (worker P): Allocator hygiene (3.3+3.4+3.7): route bare cudaMallocs through pool/immediate-params (shape metadata tensor.cpp:1290-1298, masking d_count tensor_masking_ops.cpp:1772, NaN-check buffers); free fully-empty slabs on trim_cached_memory (gpu_slab_allocator.hpp:283-298); null-owner empty tensors instead of 1-byte CUDA sentinels (tensor_unified_ops.cpp:371-379). Files: src/core/tensor pool internals.

NOTE — RESUME SEMANTICS: if your branch already contains commits for this order or the
working tree has uncommitted edits, that is a PREVIOUS INTERRUPTED ATTEMPT of this same
order. git log + git diff first; keep what is sound, finish or revert per-file; never
blindly restart from scratch and never discard committed work.

Implementation engineer, LichtFeld-Studio. Dir: /home/gauss/projects/lfs-fleet/wt-L (worktree — reuse; create branch: git switch -c lfs-elite-fW $(git rev-parse lfs-elite); verify).
Reads: perf_campaign/RULES.md (build.sh only, dual gate), TENSOR_LIB_FINDINGS.md (Theme A — THE hazard for this order), docs/analysis/spirulae-comparison/tl-memaxis.md §4/§6, tl-kernels.md reduce sections.

WORK ORDER — two structural tensor-lib memory/speed items, each behind its own gate:

## W.1 Zero-stride expand / broadcast_to views
Today expand/broadcast_to ALWAYS materialize (tensor_shape_ops.cpp:143-172, tensor_broadcast.cpp:23-77 — same-shape even clones). Implement stride-0 views with a CORRECTNESS FIREWALL: only op paths explicitly verified stride-aware may consume stride-0 views directly; every other op materializes at its boundary (extend the existing TensorLeaf/contiguous_read firewall — do NOT let stride-0 leak into Theme-A kernels). Start with the verified-safe consumers (broadcast binary kernels are already shape-indexed; elementwise via the firewall) and keep a per-op allowlist.
TDD: correctness suite comparing view-based vs materialized results for every allowlisted op incl. edge cases (stride-0 x strided mix, in-place rejection on views must throw); alloc-counter tests proving expand no longer allocates; Theme-A canary tests (feed stride-0 into non-allowlisted ops -> must materialize, never corrupt).
Gate: full tensor suites + dual-workload bench (no regression; expect neutral-to-small wins).

## W.2 Strided reduce that beats transpose+copy
tensor_unified_ops.cpp:1321-1375: non-last-dim reduce with inner_size>=256 does permute+contiguous (full tensor copy) because the old strided kernel was slower (~74us vs ~15us per in-code comment). Build a proper strided/column reduce (coalesced inner-dim access, vectorized, SM-capped — worker N's modernized reduce infra in tensor_warp_reduce.cu is the foundation) and A/B per shape class. KEEP the transpose path behind a runtime heuristic for shape classes where it still wins — the gate is per-shape-measured, not ideological: ship heuristic = argmin(measured).
TDD: correctness vs reference across shapes/dtypes incl. Float16 (now supported); microbench table (shape x method us) committed in PROGRESS.md; alloc/peak assertions (no transpose copy on shapes where new kernel wins).
Gate: no shape class regresses vs today; dual-workload bench neutral+.

Commit per item with numbers. Summary table at end.

TDD DISCIPLINE (non-negotiable, per RULES.md): for EVERY behavior above, the sequence is
strictly: (1) write the test, (2) RUN it and record the FAILING output in PROGRESS.md
(alloc-counter tests fail because expand still allocates; view-equivalence tests fail
because the API doesn't exist; strided-reduce shape tests fail against the missing kernel;
firewall canaries fail loudly), (3) implement, (4) record the PASSING run. A commit whose
message lacks the fail evidence for its tests does not count as done.

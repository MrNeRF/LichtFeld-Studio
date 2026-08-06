# Tensor Library — Deep Assessment Findings (2026-07-14)

Source: 4 parallel Codex (gpt-5.6-sol, max effort) static assessments of `src/core/tensor/`
(60 files, ~35k LOC), split by area. **Static analysis only — no runtime execution.**
Every finding below is a concrete, falsifiable claim with a reproducing input; they must be
**verified against the libtorch oracle** before/while fixing (that verification is the first
gate of the scheduled hardening effort).

---

## STATUS UPDATE (post test-cleanup merge)

The test-suite cleanup merged to master as **`540afae56` "Test cleanup (#1413)"**. While cleaning
the tests it also **fixed a small, torch-verified SUBSET** of these findings. **The hardening
round must branch off CURRENT master (which already has #1413) and re-verify each reproducer —
some now pass; do NOT re-fix them.**

**✅ Already FIXED on master by #1413** (verified against torch by the test suite):
- **Gather-class strided-index bug** — multi-input gather now applies correct per-input strides
  via `index_stride_op` + `make_transform_iterator` (was a single permutation over a zip of
  stride-1 iterators). Fixes Theme-A gather / Theme-F "SWR #6 zip gather ignores strides".
- **`has_nan()` / `has_inf()` split** — were conflated (both called `has_nan_or_inf_gpu`, so
  `has_nan()` returned true for Inf and vice-versa). Now separate `has_nan_gpu` / `has_inf_gpu`
  with `matches_special_value(v, check_nan)` distinguishing isnan vs isinf.
- **`clamp` contract + NaN propagation** — assert relaxed `isfinite`→`!isnan` (±Inf bounds are
  now valid, matches torch); the kernel now propagates NaN (`isnan(v) ? v : clamp(...)`) instead
  of suppressing it. Addresses the clamp portion of Theme E.
- **`expand` rank / `-1` handling** and **bool-conversion semantics** — matched to torch (Area 1).

**❌ Still OPEN (the systematic bulk — this is the hardening target).** Verified NOT fixed by the
merge (the diff added no stride-awareness / materialization / reduction ownership / lazy value
semantics): the broader strided masking/scatter/movement bugs (`masked_select`/`masked_fill_`,
`index_select`/`scatter_`/`index_copy_`/`index_add_`, `cat`/`stack`, `uniform_`/`multinomial`/
`diag`, `index_put_`, NN bias); ALL reduction-correctness bugs (multi-axis sum/mean/prod, Int64
promotion); ALL dtype-dispatch holes (`Float16` no-op, `Prod` uninitialized, HWC→CHW); ALL
stream/race hazards; the rest of the IEEE/NaN semantics (fused max/min, relu/log/sqrt, affine
folding, randint/multinomial overflow); lazy-eval aliasing; rank-10 ceiling; pinned-fallback.
So **roughly 6 of ~62 fixed, ~56 remain.** Each ❌ item below is unfixed unless marked ✅.

## Verdict

| Area | Scope | Maturity | Robustness | Silent-wrong-result classes |
|---|---|---|---|---|
| 1 | core class, API, dtype, broadcasting | 2/5 | 2/5 | 14 |
| 2 | compute ops (arith/reduction/broadcast/nn) | 2/5 | 1/5 | 25 |
| 3 | indexing/movement/gather/scatter/strided | 2/5 | 2/5 | 8 (+2 hazards) |
| 4 | memory / lazy-IR / expression / random | 2/5 | 2/5 | 15 |
| **Total** | | **2/5** | **~2/5** | **~62 classes** |

**Research-grade, not production-grade.** Four independent assessments converged on the same
root cause (Area 4's phrasing): *"safety properties — contiguity, offset handling, stream
ordering, dtype promotion, IEEE behavior — are implemented as local choices rather than
library-wide invariants."* The gather bug found and fixed during the test cleanup was **one of
~62 instances of the same class**, not an isolated slip.

The library is genuinely capable where it counts (see "Good parts"), and every bug is an
**incomplete-coverage bug, not an architectural dead-end** — fixable systematically, not by
rewrite. The hot training path is largely contiguous/fp32, which is why the trainer produces
good results and these went unnoticed; but densification touches masks/gather/scatter on
potentially strided data, so a subset is plausibly reachable in real training.

## Root causes (the ~62 bugs cluster into 6 themes)

1. **Strided / non-contiguous view corruption** — the dominant cluster. Kernels assume
   contiguous storage and ignore tensor strides; a legal view (transpose/slice/crop) fed to an
   op reads/writes the wrong physical elements.
2. **Non-exhaustive dtype dispatch that fails *silently*** — `default:` branches do nothing
   (return unwritten memory) instead of throwing. `Float16` and `Prod` are the notable holes.
3. **Reduction correctness** — multi-axis suffix assumptions, double normalization, integer
   accumulation overflow / missing promotion.
4. **Stream / race hazards** — specialized ops omit input-stream dependencies; shared
   `thread_local`/staging buffers reused across concurrent kernels.
5. **Numerical / IEEE semantics** — NaN suppression, wrong Inf identities, algebraic folding
   that changes overflow/zero-division results; CUDA vs CPU disagree.
6. **Lazy-evaluation aliasing** — deferred expressions capture live (mutable) storage.

Plus bounds/overflow: an undocumented **rank-10 ceiling** (rank-11 → host/device buffer
overwrite) and signed-overflow in `randint` range math.

---

## Confirmed silent-wrong-result bugs (with reproducers)

### Theme A — strided / non-contiguous views (largest cluster)

- **`zero_()` on a non-contiguous crop** clears whole rows. A 32×32 crop of a 64×64 mask clears
  16 complete rows instead of the quadrant (`tensor.cpp` zero_ path). Its in-tree test does not
  verify the mask, so it passes. *(Area 1, worst example)*
- **`cat`/`stack` copy `numel*element_size` raw bytes and ignore strides.**
  `t=[[1,2,3],[4,5,6]].transpose(0,1); cat({t,t},0)` copies physical `1,2,3,4,5,6` → wrong rows.
  `tensor_unified_ops.cpp:2086-2317`, `tensor_movement_ops.cpp:277-346`. *(Area 2, Critical)*
- **`masked_select` / `masked_fill_` linearly scan data+mask, ignore strides**; in-place fill
  can overwrite storage outside the view. Transposed `[[1,2,3],[4,5,6]]`→`[3,2]`, mask logical
  pos 3 → reads/writes physical element 3. `tensor_masking_ops.cu:35-64,371-413`. *(Areas 2+3)*
- **`index_select` / `gather` / `scatter_` / `index_copy_` / `index_add_` assume contiguous
  input, source, dest, and Int32 index.** `view=base.t(); view.index_select(0,[1])` selects
  physical rows, not logical. `tensor_masking_ops.cu:530-1038`. *(Area 3 SWR #4, "systematic view corruption")*
- **`gather` also mis-reads a strided Int32 *index* tensor** — validation materializes indices
  but the kernel reads Int32 directly/densely. `idx=idx_full[:,0]` (stride 2) → `[30,10,20]`
  instead of `[30,20,10]`. `tensor_masking_ops.cpp:84-127,477-499`. *(Area 2, gather-class)*
- **`index_put_` copies a view result to the allocation base, not the view's strided cells.**
  `v=base.slice(0,2,5); v.index_put_([1],[9])` overwrites the first base elements.
  `tensor_masking_ops.cpp:1325-1427`. *(Area 2, Critical)*
- **`uniform_` / `multinomial` / `diag` ignore strides.** `v=base.t().slice(...).squeeze();
  v.uniform_(2,4)` writes raw offsets (corrupts sibling storage, skips logical elements);
  `multinomial` samples from different probabilities than host validation saw; `diag` reads raw
  `[1,2]` not logical `[1,3]`. `tensor_random_ops.cu`, `tensor_matrix_ops.cu`. *(Area 4 SWR-04/05/06)*
- **NN bias not normalized; `_out` kernels write linearly and accept overlapping output.**
  Strided bias read as `bias[c]`; `linear_out(x, output=x)` gives GEMM input/output overlap.
  `tensor_nn_ops.cpp/.cu`. *(Area 2, Critical)*

### Theme B — silent dtype-dispatch holes (returns garbage)

- **All strided copy/scatter/upload dispatches omit `Float16`; `default:` does nothing.**
  A CUDA half `[2,3]`→transpose→`.contiguous()` **allocates output and launches no kernel →
  returns unwritten memory as a valid tensor.** Same for non-contiguous CPU-half → CUDA upload.
  `tensor_strided_ops.cu:138-768`, reachable via `tensor.cpp:916-932`. *(Area 3 SWR #1, Critical)*
- **HWC→CHW permute-upload `return`s even when the dtype switch launched nothing** → uninitialized
  CUDA tensor for Int32/Bool/Int64 image-like tensors. `tensor_strided_ops.cu:528-557`. *(Area 3 SWR #7)*
- **Multi-axis `Prod` is selected but the launcher has no `Prod` case** → `[2,2,2].prod({0,1})`
  returns allocator contents. `tensor_ops.cu:884-899`. *(Area 2, Critical uninitialized)*
- **Int32 partial reductions return untouched `empty` storage on CPU.**
  `Int32 [[1,2],[3,4]].sum({0})` → uninitialized. *(Area 2, Critical)*

### Theme C — reduction correctness

- **Multi-axis suffix `sum` is wrong unless reduced axes are a physical suffix.**
  `arange(1,9).reshape({2,2,2}).sum({0,1})` → **`[10,26]` instead of `[16,20]`.**
  `tensor_ops.cu:789-899`, `tensor_warp_reduce.cu:1327-1431`. *(Area 2, Critical, High likelihood)*
- **Suffix multi-axis `Mean` is divided twice** (inside kernel and again in caller).
  `[2,3,4].mean({1,2})` returns each mean ÷12 again. *(Area 2, Critical)*
- **`cat` RGB+alpha fast path ignores tensor 1 and hard-codes alpha=1.0.**
  `cat({rgb, alpha[[0.2]]}, -1)` → `[r,g,b,1]` not `[r,g,b,0.2]`. `tensor_ops.cu:1806-1825`. *(Area 2, Critical)*
- **Integer sum accumulates in 32 bits (no Int64 promotion); integer Mean truncates.**
  `[INT_MAX,1].sum()` wraps. *(Area 2)*

### Theme D — stream / race hazards

- **Specialized ops (`mm`, `dot`, `diag`, `multinomial`) omit input-stream dependencies** — launch
  on consumer stream without `sync_to_stream` on inputs → stale/uninitialized reads when the input
  was produced on a gated nonblocking stream. `tensor_matrix_ops.cu`, `tensor_dot_optimized.cu`,
  `tensor_random_ops.cu`. *(Area 4 SWR-03, Critical race)*
- **One `thread_local` shape buffer per launcher reused while prior nonblocking-stream kernels may
  still read it** (broadcast/`where`). `tensor_masking_ops.cu:254-267`. *(Area 3 SWR #9)*
- **Overlapping input/output neither rejected nor staged; several kernels promise `__restrict__`.**
  `dst=base.t(); dst.copy_from(base)` → nondeterministic self-corruption. *(Area 3 SWR #10)*
- **This is likely the "order-dependent corruption" the test cleanup independently bisected**
  (a test that passes alone but corrupts after a specific allocation/view history).

### Theme E — numerical / IEEE semantics (CUDA ≠ CPU ≠ torch)

- **Fused max/min use finite identities and suppress NaN** (`fmaxf`/`fminf`). All-`-FLT_MAX` max
  returns `-FLT_MAX` not `-Inf`; all-NaN max returns the finite sentinel — *wrong exactly when
  instability diagnostics matter.* `tensor_fused_pointwise.cu:167-262`. *(Area 4 SWR-08)*
- **CUDA relu/log/sqrt/rsqrt suppress NaN and silently clamp domain errors; CPU/torch don't.**
  `relu(NaN)` → 0 on CUDA. *(Area 4 SWR-09)*
- **Affine folding changes IEEE sequential semantics.** `x=full(FLT_MAX); x.mul(2).div(2)` should
  be `+Inf`, folding returns `FLT_MAX`; `x.add(1).div(0)` should be `+Inf`, fold yields NaN.
  `lazy_executor.cpp:369-429`. *(Area 4 SWR-07)*
- **`randint` range overflow:** `int range = high-low` overflows for wide Int32 ranges → UB.
  **`multinomial` weight sum overflows to Inf after host validation passed** → collapses the
  distribution to the last index. *(Area 4 SWR-10/11)*

### Theme F — lazy-eval aliasing & misc

- **Deferred expression observes later source mutation.**
  `y = x.add(1); x.fill_(5); y.to_vector()` → **6 instead of 2** (closure holds a shallow handle
  to live storage; the 4KB defer threshold triggers it). `tensor_expr_impl.hpp:27-43`. *(Area 4 SWR-01, Critical)*
- **Int32 scalar math promotion wrong.** `from_vector({1}).div(2.0f)` → Int32 `[1]` not `[1.5]`;
  `add(0.5f)` casts scalar to int and adds zero. *(Area 4 SWR-02)*
- **CUDA `row[i]` returns `float&` to a single reusable staging slot** — `float& a=row[0];
  float& b=row[1]; a=5;` makes `a` alias column 1. `tensor_row_proxy.cpp:40-59`. *(Area 4 SWR-13)*
- **Undocumented rank-10 ceiling** → rank-11 gather/`where` overwrites host stack / device local
  arrays (memory corruption). `tensor_masking_ops.cu` reserves 10 dims. *(Area 3, Critical)*
- **Strided upload assumes GPU-addressable host memory** but the pinned allocator can return a
  plain `malloc` fallback → illegal device access / CUDA fault. *(Area 3, Critical contract break)*
- **Negative scatter indices: CPU normalizes, CUDA silently skips** → device-dependent result.
  *(Area 3 SWR #8)*

---

## Good parts (this is not incompetent code)

- **Pinned allocator is production-quality**: tracks backend provenance across mode changes,
  records multi-stream use, quarantines on event-record failure, caps/evicts its cache, and has
  multi-stream/LRU tests (`pinned_memory_allocator.cpp:235-489`).
- **Slab allocator** has a sensible same-stream/cross-stream reuse model.
- **`TensorLeaf` materializes non-contiguous operands** before out-of-place expression eval — a
  real correctness firewall that protects most arithmetic (the strided bugs are concentrated in
  in-place/kernel paths that bypass it).
- **Shape/reserve math uses checked products** (no silent size overflow).
- Boundary validation is often good; the gap is that kernels don't *enforce* the contracts the
  boundary checks assume.

## Fix strategy (for the scheduled hardening — verify vs libtorch FIRST)

0. **Branch off CURRENT master (has #1413).** Re-run every ✅-marked reproducer first — they now
   pass; skip them. The ~56 ❌ items are the target. Do not re-fix or revert the merged subset.
1. **Reproduce each remaining top bug against libtorch** with the input above; drop any static
   false-positive before fixing. Leave a torch-asserting regression test for each real one.
2. **Centralize the invariants, don't one-off-patch** (per every area's recommendation):
   - One **exhaustive, fail-loud dtype dispatch** — no silent `default:`.
   - A **materialization firewall** at the masking/indexing/movement boundary (normalize
     non-contiguous read operands + index tensors), with stride-aware kernels for in-place view
     destinations where materialization can't preserve semantics.
   - One **reduction result-dtype + normalization ownership point**; Int64 integer accumulation.
   - Route every specialized launcher through one `prepare_inputs_for_stream` helper.
   - Define numeric semantics (NaN/Inf/domain) once; stop algebraic folding of user-visible FP by
     default.
   - Lazy expressions get **value semantics** (COW snapshot / data-version guard).
   - One enforced **max-rank constant**.
3. **Keep the contiguous fast path** so correctness costs only apply to non-contiguous inputs —
   speed must be neutral-or-better (hard benchmark gate: no op may regress).

## Caveats

- **Static analysis** — confirmed by code-reading, not execution. Counts (~62) may include
  false-positives; the reproducers make each falsifiable.
- **Confidence raised by #1413:** the gather-class, the `has_nan`/`has_inf` conflation, and the
  clamp NaN/±Inf semantics were all **independently confirmed at runtime and fixed against the
  libtorch oracle** during the test cleanup — so this class of finding is real, not a static
  artifact. That makes the ~56 still-open items worth taking seriously.
- **The test suite is now a better oracle for the hardening:** #1413 added torch-asserting
  regression tests and CTest fast/slow/gpu tiers, so the hardening can validate each fix against
  `ctest -L fast` on a suite that actually checks masks/strides/dtypes (the old suite passed even
  when the code was wrong — e.g. the `zero_()` test never verified its mask).
- Full per-area reports (with complete tables) preserved at
  `.codex_tmp/claude-vulkan/tl-assess-{1,2,3,4}-full.md`.

---

## ADDENDUM (2026-07-23) — Deferred-materialization / pointer-stability defects D1/D3/D4

Source: the 2026-07-22 Windows training-start crash investigation
(`docs/windows_training_start_crash_2026_07.md`) plus a 12-agent evidence pass
(`.codex_tmp/tensorlib-hardening/d3-campaign/out/`). These are **lifetime/identity** defects in
the lazy machinery, orthogonal to the ~56 silent-wrong-result classes above. D2 (stale lazy-cache
entries surviving `emplace`) is already fixed in `8b2268869` (`insert_or_assign` + gtest).

- **D3 (crash class):** `ptr()`/`data_ptr()`/`storage_ptr()` on a deferred tensor trigger
  `materialize_if_deferred`, which replaces `data_`/`data_owner_`/`shape_`/`strides_`/`state_`
  wholesale (`tensor.cpp:504-559`). Any raw device pointer or host `dims().data()` pointer
  captured before a sibling operand's materialization in the same call expression dangles
  (C++ unspecified argument-evaluation order). Exactly one defended site exists
  (`tensor_expr_impl.hpp:347-353`); the audits enumerate ~35-40 unprotected logical patterns
  (~95 dtype-expanded HIGH lines) across the op layer — non-broadcast binary, masking, index/
  gather/scatter, cdist, matmul (worklist tables in `out/audit-{expr,internal,tensor-cpp}.md`).
  Standalone defects found en route: `create_strided_view` lacks the deferred guard that
  `create_view` has (`tensor_impl.hpp:993-1002` — silently copies null storage);
  `TensorAccessor` caches a raw pointer for its lifetime (`tensor_impl.hpp:1042-1093`);
  `contiguous_read` is **not** a materialization barrier (deferred tensors claim
  `is_contiguous_=true` with `data_==nullptr`).
- **D1 (identity desync):** copy ctor/assignment deep-copy `TensorState` — including
  `deferred_expr_node_id` and a full `std::function` materializer copy — while the lazy
  registry holds ONE weak state per node id; materialization unregisters by id, desyncing
  siblings (`tensor.cpp:611-689`, `lazy_executor.cpp:539-599`). Production amplifier: the
  deferred branches of `permute`/`slice`/`create_view`/`broadcast_to` copy the deferred tensor
  into their materializer lambdas (Scenario E, `out/d1-registry.md`). Consequences: double
  materialization (compute + VRAM waste), silently divergent storages for aliases, registry
  owner-expiry pruning when the original dies before its copies.
- **D4 (stream invisibility):** non-owning tensors (`from_blob` → raw ctor) are invisible to
  `record_stream` (silent no-op, `tensor.cpp:817-828`); external-owner tensors miss the pool
  map (silent no-op); untracked frees fall through to bare `cudaFreeAsync` with no cross-stream
  ordering (`memory_pool.hpp:309-319`). Today's production sites are papered over by
  same-stream discipline and sync-before-free (`out/d4-nonowning.md` inventory); the hole is
  structural. `prepare_inputs_for_stream` orders reads but never records the use with the
  allocator — free-side ordering rests entirely on home-stream metadata.

**Design decision** (draft, pending adversarial review):
`.codex_tmp/tensorlib-hardening/d3-campaign/DESIGN.md` — shared `LazyExprState` (D1 root fix,
also removes the per-copy `std::function` allocation), centralized operand pinning at the op
layer (D3), loud diagnostics on silently-ignored `record_stream` + breadcrumbs for silent pool
eviction frees, home-stream stamping for `from_blob` (D4 minimal). Options (b′) node-id-owned
stable storage and (d) guard-object `ptr()` deferred until the reporter's necrology log
(instrumented build `8b2268869`, CI pending) confirms or refutes the storage-move mechanism.
Speed gate: 3× 7k-smoke wall ≤ baseline×1.05 + exact splat-trajectory match + 5× microbench
(`out/test-bench.md` §5).

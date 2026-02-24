# Tensor Library Elite Plan

Date: 2026-02-23
Scope: `src/core/tensor/` and `src/core/include/core/tensor*.hpp`
Goal: make the tensor engine transparently lazy and materially faster without user-visible API changes.

## 1. Product Direction

We will move to a user-invisible lazy execution model:
- same public `Tensor` API
- same semantics and numerical results
- eager validation, lazy compute
- automatic materialization at boundaries only
- aggressive fusion and temp-buffer reuse under the hood

Non-negotiables:
1. No API break in `src/core/include/core/tensor.hpp`.
2. No required `.eval()` by users.
3. No semantic regressions for views/in-place/indexing.
4. Deterministic behavior for stateful/random ops.
5. Global kill switch and staged rollout.

## 2. Current Strengths (Keep and Build On)

1. Broad operation coverage + large test surface.
- `tests/test_tensor_*.cpp` and many focused regression/benchmark suites.

2. Memory system has strong foundations.
- Layered pooling and tracking in `src/core/tensor/internal/memory_pool.hpp`.
- Stream-aware deferred free queue in `src/core/tensor/internal/deferred_free_queue.hpp`.
- Pinned allocator with event-based reuse safety in `src/core/tensor/pinned_memory_allocator.cpp`.

3. Important CUDA optimization building blocks already exist.
- Warp/CUB reductions in `src/core/tensor/tensor_warp_reduce.cu`.
- Strided copy/upload kernels in `src/core/tensor/tensor_strided_ops.cu`.
- View/stride metadata support in `src/core/tensor/tensor_shape_ops.cpp`.

## 3. Findings (What Is Bad / Risky Today)

### 3.1 Critical correctness and contract issues

1. Public docs claim lazy/fused behavior, but most public ops return materialized `Tensor` eagerly.
- `src/core/include/core/tensor.hpp:14`
- `src/core/include/core/tensor.hpp:29`
- `src/core/tensor/internal/tensor_impl.hpp:1095`
- `src/core/tensor/internal/tensor_impl.hpp:1130`
- `src/core/tensor/internal/tensor_expr_impl.hpp:16`

2. `where`/ternary path has dtype risk for non-float outputs.
- `src/core/tensor/tensor_unified_ops.cpp:1263`
- `src/core/tensor/tensor_unified_ops.cpp:1289`

3. Multi-index `index_put_` can mis-handle index dtype (int32 assumptions on pointers).
- `src/core/tensor/tensor_masking_ops.cpp:1109`
- `src/core/tensor/tensor_masking_ops.cpp:1134`

4. `TensorRowProxy` CUDA assignment semantics are unsafe (host ref path).
- `src/core/tensor/tensor_row_proxy.cpp:33`
- `src/core/tensor/tensor_row_proxy.cpp:47`

### 3.2 Performance and runtime architecture issues

1. Stream semantics are inconsistent; many launches use `nullptr` stream.
- `src/core/tensor/internal/tensor_impl.hpp:394`
- `src/core/tensor/internal/tensor_expr_impl.hpp:50`
- `src/core/tensor/tensor_matrix_ops.cpp:63`
- `src/core/tensor/tensor_unified_ops.cpp:651`
- `src/core/tensor/internal/cuda_stream_context.hpp:29`

2. Hidden synchronization in hot paths.
- `src/core/tensor/tensor.cpp:751`
- `src/core/tensor/tensor.cpp:1732`
- `src/core/tensor/tensor_masking_ops.cpp:1183`
- `src/core/tensor/tensor_unified_ops.cpp:1767`
- `src/core/tensor/tensor_unified_ops.cpp:1851`

3. Frequent CUDA<->CPU roundtrips in operational paths.
- dtype conversions in `src/core/tensor/tensor.cpp:844`
- masking fallback in `src/core/tensor/tensor_masking_ops.cpp:958`
- nonzero fallback in `src/core/tensor/tensor_masking_ops.cpp:1319`
- min/max indices in `src/core/tensor/tensor_advanced_ops.cpp:109`
- slice copy in `src/core/tensor/tensor_shape_ops.cpp:299`

4. Reduction support gaps/partial behavior.
- `src/core/tensor/tensor_ops.cu:1018`
- `src/core/tensor/tensor_ops.cu:1222`
- `src/core/tensor/tensor_unified_ops.cpp:902`

5. Allocator-related sync stalls in some paths.
- `src/core/tensor/internal/gpu_arena_allocator.hpp:111`
- `src/core/tensor/internal/deferred_free_queue.hpp:117`
- `src/core/tensor/internal/size_bucketed_pool.hpp:150`

### 3.3 Maintainability friction

1. Dtype promotion logic duplicated.
- `src/core/tensor/internal/tensor_impl.hpp:49`
- `src/core/tensor/tensor_unified_ops.cpp:23`

2. Stream metadata not consistently propagated to view-like tensors.
- `src/core/tensor/internal/tensor_impl.hpp:604`
- `src/core/tensor/tensor_shape_ops.cpp:96`

3. Large explicit CUDA instantiation blocks raise compile-time and maintenance cost.
- `src/core/tensor/tensor_ops.cu` (large explicit instantiation region).

## 4. Target End-State Architecture

1. `Tensor` is a thin handle around `TensorState`.
2. `TensorState` carries shape/stride/dtype/device, alias/version metadata, optional storage, optional expr root.
3. Pure ops build DAG nodes (`ExprNode`) instead of immediate execution.
4. Central planner performs topological scheduling, liveness, and buffer reuse.
5. Fusion engine fuses elementwise/broadcast/cast/where chains automatically.
6. Materialization occurs only at defined boundaries.
7. Stream model is explicit and consistent end-to-end.

## 5. Execution Modes and Rollout

Environment variable: `TENSOR_LAZY_MODE`
- `off`: legacy eager execution.
- `shadow`: run lazy planning/execution checks alongside eager for parity diagnostics.
- `on`: lazy execution enabled.

Rollout policy:
1. Start with `off` default.
2. Enable `shadow` in CI/nightly.
3. Enable `on` per-op-family once parity and perf gates pass.
4. Keep immediate kill switch to `off`.

## 6. PR-by-PR Backlog

### PR1: Baseline + mode flags + telemetry scaffold
Goal: observability and switchboard with no behavior change.
- Add lazy mode config parser and runtime counters.
- Add minimal API to read mode and counters.
- Wire tests for mode parsing/default behavior.
Success criteria:
- zero behavior change in `off`
- counters available
- CI green

### PR2: Introduce `TensorState` handle split
Goal: prepare for deferred compute without API break.
- Move mutable runtime state to shared state object.
- Keep copy/move semantics unchanged.
Success criteria:
- all existing tests pass
- no runtime regressions > 2%

### PR3: Expression IR foundation
Goal: explicit DAG representation.
- Add `ExprNode`, op attributes, inferred shape/dtype/device.
Success criteria:
- graph can be built for pure ops
- inference tests pass

### PR4: Lazy build path for pure pointwise ops
Goal: defer execution for unary/binary/cast/comparison/where (build-only).
Success criteria:
- parity tests eager vs lazy (forced materialization) pass

### PR5: Central materialization boundaries
Goal: define exactly where lazy tensors execute.
Boundaries include:
- `item()`, host memory access, `to(Device::CPU)`, explicit sync, mutation hazards, external interop
Success criteria:
- no accidental eager execution outside boundaries

### PR6: Planner/executor V1 (non-fused)
Goal: execute DAG with topological scheduling + CSE.
Success criteria:
- shared subgraph computed once
- deterministic ordering

### PR7: Fusion V1 (contiguous elementwise)
Goal: fuse unary/binary/cast chains into single launch.
Success criteria:
- launch count drop >= 50% on targeted chain benchmark
- correctness parity

### PR8: Fusion V2 (scalar + unary pointwise chain extension)
Goal: extend V1 scalar fusion to mixed scalar+unary chains on CPU/CUDA.
Success criteria:
- mixed unary/scalar chains fuse with parity and reduced launches

### PR9: Reduction integration
Goal: make reductions graph-native boundaries with producer fusion.
Success criteria:
- parity for covered fused paths (CUDA Float32 full + last-dim segmented), with correct fallback elsewhere

### PR10: Alias/view/in-place correctness hardening
Goal: enforce versioning, copy-on-write only when needed, mutation barriers.
Success criteria:
- strict aliasing/view regression suite passes

### PR11: Stream consistency pass
Goal: remove implicit null/default-stream launch ambiguity.
Success criteria:
- stream-order tests pass on non-default streams

### PR12: Memory planner integration
Goal: graph-lifetime temp buffer reuse via existing pool infra.
Success criteria:
- temporary allocation bytes drop 30-60% on chain workloads

### PR13: Stateful op rules
Goal: preserve determinism around RNG/stateful operations.
Success criteria:
- seed-based reproducibility tests pass in lazy/on

### PR14: Heuristics + production rollout
Goal: eager fallback for tiny workloads and staged enablement.
Success criteria:
- no small-tensor regression > 5%
- large-chain gains preserved

### PR15: Broadcast + where fusion (pending)
Goal: fuse broadcast-heavy and ternary (`where`) chains in the lazy executor.
Success criteria:
- broadcast-heavy benchmarks show 1.5x+ speedups
- where-heavy pipelines reduce launches with parity

## 7. Test Plan (Must Add)

1. `where` dtype correctness matrix across Bool/Int32/Int64/Float16/Float32.
2. `index_put_` with Int64 indices on CUDA.
3. `TensorRowProxy` assignment semantics on CUDA.
4. Stream-ordering tests with two non-default CUDA streams.
5. Lazy vs eager parity suite (`shadow` mode) across representative op graphs.
6. Perf gates:
- elementwise chain
- broadcast chain
- reduction with precompute
- small tensor microbench

## 8. KPI Dashboard

Primary:
1. Kernel launches per training step.
2. Temporary bytes allocated per step.
3. GPU busy time / achieved occupancy proxy.
4. End-to-end iteration time on representative workloads.

Targets:
1. 50-80% fewer launches on pointwise-heavy pipelines.
2. 30-60% fewer temp bytes.
3. 1.5-3x speedup on long pointwise+broadcast chains.
4. <= 5% small-tensor overhead via eager heuristics.

## 9. Immediate Implementation Sequence (starting now)

1. Implement PR1 scaffolding (`TENSOR_LAZY_MODE`, counters, snapshot/reset API, tests).
2. Land without behavior change.
3. Start PR2 (`TensorState`) once PR1 compiles and tests pass.

## 10. Progress Log

### 2026-02-23 - Completed

1. PR1 baseline lazy runtime scaffold (done).
- Added `TENSOR_LAZY_MODE` runtime (`off|shadow|on`), mode parser, mode override for tests.
- Added telemetry counters (expr nodes, materializations, eager fallbacks, kernel launches, bytes).
- Added Tensor API hooks for mode/telemetry inspection and reset.
- Added `tests/test_tensor_lazy_runtime.cpp`.

2. PR2 TensorState split (done).
- Moved mutable runtime fields (`capacity`, `logical_size`, alignment flags, stream, tracking/name) into shared `TensorState`.
- Updated constructors/assignments and in-place/capacity-sensitive paths to use `state_`.
- Verified no behavior regression in targeted reserve/cat/move/capacity suites.

3. PR3 expression IR foundation bootstrap (done for shadow metadata path).
- Added `internal/lazy_ir.hpp` + `tensor/lazy_ir.cpp` runtime graph store.
- Added graph metadata hooks on Tensor (`has_lazy_expr`, `lazy_expr_id`, `lazy_expr_info`).
- Instrumented expression evaluators to record unary/binary/scalar/permutation nodes in `shadow/on`.
- In `on`, expression evaluation now records eager fallback telemetry (still eager execution for correctness).
- Added `tests/test_tensor_lazy_ir.cpp`.

4. Validation completed after PR3 bootstrap.
- Build: `cmake --build build-tests --target lichtfeld_tests -j8` passed.
- Tests: `TensorLazyRuntimeTest.*` + `TensorLazyIrTest.*` passed.
- Regression subset: reserve/in-place cat/move/capacity/lazy runtime suite (87 tests) passed.

5. PR4 deferred execution path for `lazy_mode=on` (implemented).
- `TensorExpr<Derived>::operator Tensor()` now returns deferred tensors in `on` mode instead of immediate eval.
- Added deferred tensor creation/materialization plumbing:
  - `Tensor::make_deferred_expr_tensor(...)`
  - `Tensor::materialize_if_deferred()`
  - `TensorState` deferred fields (`has_deferred_expr`, re-entry guard, deferred materializer functor)
- Added lazy-IR deferred node kind and recorder (`LazyOpKind::Deferred`, `lazy_ir_record_deferred`).
- Updated lazy-IR test semantics for `on` mode:
  - pre-boundary: deferred node exists, eager fallback remains 0
  - boundary (`ptr()`): materialization occurs and eager fallback increments

6. PR4 boundary hardening pass (implemented).
- Added deferred materialization guards to major user-visible data boundaries and mutators:
  - `contiguous()`, `to(Device)`, `to(DataType)`, `item()`
  - `zero_()`, `fill_()`, `fill_(..., stream)`, `copy_from()`, `reserve()`
  - vector export helpers (`to_vector*`, `debug_values`)
  - raw-storage view/mutation paths that needed explicit guards (`create_view`, `index_put_`, `append_zeros`)
  - in-place masking paths with raw storage writes (`index_put_` overloads, `append_zeros`)
- Hardened pointer helpers:
  - `data_ptr()` / `storage_ptr()` now return `nullptr` for invalid tensors instead of doing pointer math on null.
  - `item<T>()` now materializes deferred tensors before raw read.

7. Validation completed after PR4 implementation + hardening.
- Build: `cmake --build build-tests --target lichtfeld_tests -j8` passed.
- Tests: `./build-tests/tests/lichtfeld_tests --gtest_filter='TensorLazyRuntimeTest.*:TensorLazyIrTest.*'` passed.
- Regression subset: `./build-tests/tests/lichtfeld_tests --gtest_filter='TensorLazyRuntimeTest.*:TensorReserveInplaceCat.*:InplaceCat.*:TensorInplaceCapacityTest.*:TensorMoveTest.*'` passed (87/87).

8. PR4 follow-up: keep view/movement chains lazy (implemented).
- Deferred tensors now stay deferred through metadata-only transforms:
  - `create_view` deferred chaining in `tensor_impl.hpp`
  - `permute`/`slice` deferred chaining in `tensor_shape_ops.cpp`
  - `transpose` via `movement(Transpose)` routes deferred path through lazy `permute`
  - `broadcast_to` deferred chaining in `tensor.cpp`
- Added test coverage:
  - `TensorLazyIrTest.OnModeKeepsDeferredThroughViewChain`
- Validation:
  - `./build-tests/tests/lichtfeld_tests --gtest_filter='TensorLazyRuntimeTest.*:TensorLazyIrTest.*'` passed (15 tests).
  - `./build-tests/tests/lichtfeld_tests --gtest_filter='TensorViewTest.*:SliceMMBugTest.*:InterleavedSliceCopyTest.*:ItemTypeSafetyTest.*'` passed (57 tests).
  - Reserve/capacity/move regression subset still passed (87 tests).

9. PR5 boundary diagnostics: reason-tagged eager fallback telemetry (implemented).
- Added telemetry counters by boundary class:
  - `eager_fallback_host_read`
  - `eager_fallback_device_transfer`
  - `eager_fallback_mutation`
  - `eager_fallback_interop`
  - `eager_fallback_other`
- Added scoped fallback attribution:
  - `internal::LazyFallbackReason` + `LazyFallbackReasonScope`
  - Nested scopes preserve an already-classified outer boundary reason.
- Wired scopes across boundary materialization paths:
  - Host-read boundaries (`item`, `to_vector*`, `debug_values`, pointer reads)
  - Device transfer boundaries (`to(Device)`)
  - Mutation boundaries (`zero_`, `fill_`, `copy_from`, `reserve`, `index_put_`, `append_zeros`, `scatter_`, `index_copy_`, `index_add_`, `append_gather`, etc.)
  - Interop boundaries (`data_ptr`/`storage_ptr`)
- Extended lazy IR tests to assert reason-specific counters for:
  - host-read, transfer, mutation, interop boundaries
  - `Other` boundaries (`contiguous`, `to(DataType)`)
  - nested-boundary attribution (outer reason preserved)
  - mutation-family parity for `index_put_` variants and append/index edge paths
- Validation:
  - `cmake --build build-tests --target lichtfeld_tests -j8` passed.
  - `./build-tests/tests/lichtfeld_tests --gtest_filter='TensorLazyRuntimeTest.*:TensorLazyIrTest.*'` passed (21 tests).
  - `./build-tests/tests/lichtfeld_tests --gtest_filter='TensorViewTest.*:SliceMMBugTest.*:InterleavedSliceCopyTest.*:ItemTypeSafetyTest.*'` passed (57 tests).
  - `./build-tests/tests/lichtfeld_tests --gtest_filter='TensorLazyRuntimeTest.*:TensorReserveInplaceCat.*:InplaceCat.*:TensorInplaceCapacityTest.*:TensorMoveTest.*'` passed (87 tests).

10. PR6 planner/executor V1 skeleton bootstrap (started).
- Added planner/executor scaffold:
  - `src/core/tensor/internal/lazy_executor.hpp`
  - `src/core/tensor/lazy_executor.cpp`
- Added initial APIs:
  - `lazy_planner_build_plan_for_tensor(...)` (builds root/topology debug artifact)
  - `lazy_planner_execute_plan_for_tensor(...)` (currently delegates to root materializer)
- Wired deferred materialization to flow through planner executor path in `lazy_mode=on` with no semantic change:
  - `Tensor::materialize_if_deferred()` now routes through planner skeleton.
- Added tests:
  - `TensorLazyIrTest.OffModePlannerSkeletonDisabled`
  - `TensorLazyIrTest.OnModePlannerSkeletonBuildsPlanForDeferredRoot`
- Validation:
  - `cmake --build build-tests --target lichtfeld_tests -j8` passed.
  - `./build-tests/tests/lichtfeld_tests --gtest_filter='TensorLazyRuntimeTest.*:TensorLazyIrTest.*'` passed (21 tests).
  - Existing regression subsets remained green (57 + 87 tests filters).

11. PR6 planner topology/determinism expansion (implemented).
- Extended lazy IR query surface:
  - `lazy_ir_node_info(node_id)`
  - `lazy_ir_collect_topological_subgraph(root_node_id)`
  - `lazy_ir_record_deferred(..., input_ids)` now records explicit dependency edges.
- Extended deferred tensor creation plumbing:
  - `Tensor::make_deferred_expr_tensor(..., lazy_input_ids)` carries parent node ids.
- Wired deferred dependency edges through view-like chains:
  - `create_view` (`tensor_impl.hpp`)
  - `permute` / `slice` (`tensor_shape_ops.cpp`)
  - `broadcast_to` (`tensor.cpp`)
- Planner now builds a real topological plan from the lazy-IR root instead of root-only artifacts.
- Added planner topology tests:
  - `TensorLazyIrTest.OnModePlannerTopologicalOrderRespectsDependencies`
  - `TensorLazyIrTest.OnModePlannerTopologyIsDeterministic`
  - `TensorLazyIrTest.OnModePlannerSharedSubgraphVisibleAcrossPlans`
- Correctness fixes discovered while hardening PR6:
  - Fixed deferred expr metadata capture in `TensorExpr::operator Tensor()` by snapshotting shape/device/dtype before moving expr into lambda (avoids argument-evaluation-order moved-from bug).
  - Fixed deferred input-edge loss by capturing `lazy_expr_id()` before copying deferred sources (copy constructor assigns new debug id, which otherwise dropped dependency ids).
- Validation:
  - `cmake --build build-tests --target lichtfeld_tests -j8` passed.
  - `./build-tests/tests/lichtfeld_tests --gtest_filter='TensorLazyRuntimeTest.*:TensorLazyIrTest.*'` passed (24 tests).
  - `./build-tests/tests/lichtfeld_tests --gtest_filter='TensorViewTest.*:SliceMMBugTest.*:InterleavedSliceCopyTest.*:ItemTypeSafetyTest.*'` passed (57 tests).
  - `./build-tests/tests/lichtfeld_tests --gtest_filter='TensorLazyRuntimeTest.*:TensorReserveInplaceCat.*:InplaceCat.*:TensorInplaceCapacityTest.*:TensorMoveTest.*'` passed (87 tests).

12. PR6 execution-context cache scaffolding (implemented, non-fused).
- Added planner execution context with per-materialization node cache:
  - `lazy_executor_context_active()`
  - `lazy_executor_lookup_cached_materialization(...)`
  - `lazy_executor_cache_materialization(...)`
- Planner executor now establishes a thread-local context for root execution and keeps nested deferred materializations inside the same cache scope.
- Added deferred-node-id hint to `TensorState`:
  - `deferred_expr_node_id` is set at deferred-node creation and preserved through tensor copies.
  - `Tensor::lazy_expr_id()` now uses this hint for deferred tensors, preventing lazy-id loss after copy construction.
- `Tensor::materialize_if_deferred()` now:
  - tries cache lookup first (when lazy mode is on)
  - falls back to materializer execution on cache miss
  - caches successful materializations by lazy node id
- Added PR6 execution parity test:
  - `TensorLazyIrTest.OnModePlannerExecutorCachesSharedSubgraphWithinMaterialization`
  - verifies shared-subgraph graph triggers fewer fallback evals than equivalent non-shared graph.
- Validation:
  - `cmake --build build-tests --target lichtfeld_tests -j8` passed.
  - `./build-tests/tests/lichtfeld_tests --gtest_filter='TensorLazyRuntimeTest.*:TensorLazyIrTest.*'` passed (25 tests).
  - `./build-tests/tests/lichtfeld_tests --gtest_filter='TensorViewTest.*:SliceMMBugTest.*:InterleavedSliceCopyTest.*:ItemTypeSafetyTest.*'` passed (57 tests).
  - `./build-tests/tests/lichtfeld_tests --gtest_filter='TensorLazyRuntimeTest.*:TensorReserveInplaceCat.*:InplaceCat.*:TensorInplaceCapacityTest.*:TensorMoveTest.*'` passed (87 tests).

13. PR6 topo-driven executor flow (implemented, still non-fused).
- Added deferred-node materializer registry in planner runtime:
  - `lazy_executor_register_deferred_materializer(node_id, materializer)`
  - `lazy_executor_unregister_deferred_materializer(node_id)`
- Deferred tensors now register planner-visible node materializers at creation time, and unregister when materialized.
- Planner executor now executes registered nodes in plan topological order inside the execution context before root fallback materializer delegation.
- Root node is now resolved from cache when already executed by topo pass; fallback root materializer is only used when root cache is absent.
- This keeps semantics unchanged while making execution genuinely plan-driven and enabling future node-level execution/fusion work.
- Validation:
  - `cmake --build build-tests --target lichtfeld_tests -j8` passed.
  - `./build-tests/tests/lichtfeld_tests --gtest_filter='TensorLazyRuntimeTest.*:TensorLazyIrTest.*'` passed (25 tests).
  - `./build-tests/tests/lichtfeld_tests --gtest_filter='TensorViewTest.*:SliceMMBugTest.*:InterleavedSliceCopyTest.*:ItemTypeSafetyTest.*'` passed (57 tests).
  - `./build-tests/tests/lichtfeld_tests --gtest_filter='TensorLazyRuntimeTest.*:TensorReserveInplaceCat.*:InplaceCat.*:TensorInplaceCapacityTest.*:TensorMoveTest.*'` passed (87 tests).

14. PR5 mutation-boundary parity expansion + PR6 lifecycle hardening (implemented).
- Extended PR5 mutation-family boundary coverage:
  - Added `index_put_` parity tests for:
    - single-index Int64 path
    - single-index empty/no-op edge path
    - multi-index Int64 path
    - multi-index mismatch/no-op edge path
  - Added append/index edge-path boundary test:
    - `append_gather` non-1D indices early-return path
- Correctness hardening in multi-index `index_put_`:
  - Normalized row/col indices to Int64 in the 2D multi-index path.
  - Fixed pointer dispatch to read Int64 index tensors correctly (CPU + CUDA fallback path).
  - Added explicit Float32 guard for this legacy branch to avoid dtype-unsafe pointer use.
- PR6 registry lifecycle cleanup:
  - Deferred materializer registry entries now carry weak owner tokens.
  - Registry prunes expired unmaterialized deferred nodes automatically on lookup/register/count.
  - Deferred materializer registration now binds to deferred tensor state ownership.
- Added PR6 diagnostics tests:
  - `OnModePlannerRegistryPrunesExpiredUnmaterializedDeferredNodes`
  - `OnModeMixedTransferThenHostBoundaryMaterializesOnlyOnce`
- Validation:
  - `cmake --build build-tests --target lichtfeld_tests -j8` passed.
  - `./build-tests/tests/lichtfeld_tests --gtest_filter='TensorLazyRuntimeTest.*:TensorLazyIrTest.*'` passed (34 tests).
  - `./build-tests/tests/lichtfeld_tests --gtest_filter='TensorViewTest.*:SliceMMBugTest.*:InterleavedSliceCopyTest.*:ItemTypeSafetyTest.*'` passed (57 tests).
  - `./build-tests/tests/lichtfeld_tests --gtest_filter='TensorLazyRuntimeTest.*:TensorReserveInplaceCat.*:InplaceCat.*:TensorInplaceCapacityTest.*:TensorMoveTest.*'` passed (87 tests).

15. PR6 planner diagnostics instrumentation + fan-out/re-entry tests (implemented).
- Added internal planner/executor diagnostics snapshot API:
  - `LazyExecutorDiagnosticsSnapshot` with:
    - `planned_nodes`
    - `executed_nodes`
    - `cache_hits`
    - `cache_misses`
    - `root_fallbacks`
  - APIs:
    - `lazy_executor_reset_diagnostics_for_testing()`
    - `lazy_executor_diagnostics_snapshot_for_testing()`
- Wired diagnostics accounting through executor runtime:
  - plan construction path accumulates `planned_nodes`
  - topological materializer execution increments `executed_nodes`
  - execution-context cache lookups count hits/misses
  - root materializer fallback path counts `root_fallbacks`
- Extended lazy IR test coverage for PR6 diagnostics:
  - `OnModePlannerDiagnosticsCaptureFanOutExecution`
  - `OnModeRepeatedBoundaryAddsNoPlannerDiagnosticsAfterMaterialization`
  - `OnModePlannerDiagnosticsTrackRootFallbackWhenPlanHasNoRoot`
- Validation:
  - `cmake --build build-tests --target lichtfeld_tests -j8` passed.
  - `./build-tests/tests/lichtfeld_tests --gtest_filter='TensorLazyRuntimeTest.*:TensorLazyIrTest.*'` passed (37 tests).
  - `./build-tests/tests/lichtfeld_tests --gtest_filter='TensorViewTest.*:SliceMMBugTest.*:InterleavedSliceCopyTest.*:ItemTypeSafetyTest.*'` passed (57 tests).
  - `./build-tests/tests/lichtfeld_tests --gtest_filter='TensorLazyRuntimeTest.*:TensorReserveInplaceCat.*:InplaceCat.*:TensorInplaceCapacityTest.*:TensorMoveTest.*'` passed (87 tests).

16. PR6 overhead benchmark guardrail + long-run growth guardrails (implemented).
- Extended planner diagnostics with growth-focused peaks:
  - `max_registry_entries`
  - `max_context_cache_entries`
- Executor runtime now tracks:
  - peak live deferred registry size after prune-aware registration
  - peak per-execution context cache entry count
- Added PR6 overhead benchmark guardrail test:
  - `OnModeOverheadBenchmarkGuardrailVsOffMode`
  - Runs fixed CPU chain microbenchmark in `off` vs `on` and enforces broad catastrophe guardrail (`on/off < 3.0`).
  - Current measured run: `off_avg_us=317.167`, `on_avg_us=373.167`, `ratio=1.17656`.
- Added long-run growth guardrail tests:
  - `OnModeRegistryGrowthGuardrailInLongCreateDropLoop`
  - `OnModeContextCacheGrowthGuardrailBoundedByPlannedNodes`
- Validation:
  - `cmake --build build-tests --target lichtfeld_tests -j8` passed.
  - `./build-tests/tests/lichtfeld_tests --gtest_filter='TensorLazyRuntimeTest.*:TensorLazyIrTest.*'` passed (40 tests).
  - `./build-tests/tests/lichtfeld_tests --gtest_filter='TensorViewTest.*:SliceMMBugTest.*:InterleavedSliceCopyTest.*:ItemTypeSafetyTest.*'` passed (57 tests).
  - `./build-tests/tests/lichtfeld_tests --gtest_filter='TensorLazyRuntimeTest.*:TensorReserveInplaceCat.*:InplaceCat.*:TensorInplaceCapacityTest.*:TensorMoveTest.*'` passed (87 tests).

17. PR6 optional runtime diagnostics dump hook (implemented).
- Added optional single-line planner diagnostics dump on materialization:
  - Emits one top-level summary line per planner execution/materialization entry.
  - Includes root id, source (`cache|fallback`), planned/executed nodes, cache hits/misses, root fallbacks, and peak registry/context sizes.
- Added env-gated enablement:
  - `TENSOR_LAZY_EXEC_DIAGNOSTICS=1|true|on|yes|enabled` enables dumps.
  - Disabled by default.
- Added cached env parsing + test-only controls:
  - `lazy_executor_set_debug_dump_override_for_testing(...)`
  - `lazy_executor_clear_debug_dump_cache_for_testing()`
  - `lazy_executor_debug_dump_enabled_for_testing()`
- Added validation test:
  - `OnModePlannerDebugDumpOverrideControlsFlag`
- Updated overhead guardrail observed run:
  - `off_avg_us=318.417`, `on_avg_us=335.083`, `ratio=1.05234`.
- Validation:
  - `cmake --build build-tests --target lichtfeld_tests -j8` passed.
  - `./build-tests/tests/lichtfeld_tests --gtest_filter='TensorLazyRuntimeTest.*:TensorLazyIrTest.*'` passed (41 tests).
  - `./build-tests/tests/lichtfeld_tests --gtest_filter='TensorViewTest.*:SliceMMBugTest.*:InterleavedSliceCopyTest.*:ItemTypeSafetyTest.*'` passed (57 tests).
  - `./build-tests/tests/lichtfeld_tests --gtest_filter='TensorLazyRuntimeTest.*:TensorReserveInplaceCat.*:InplaceCat.*:TensorInplaceCapacityTest.*:TensorMoveTest.*'` passed (87 tests).

### PR7: Fusion V1 — GPU Affine Fold for Scalar Pointwise Chains (2026-02-23)

Replaced interpretive CPU-only scalar pointwise loop with algebraic affine folding (`y = a*x + b`).
Extended fusion to GPU tensors via vectorized float4 FMA kernel.

**Changes:**
- `src/core/tensor/tensor_fused_pointwise.cu` — NEW: vectorized affine transform kernel (float4 + scalar fallback)
- `src/core/tensor/internal/tensor_ops.hpp` — declared `launch_fused_affine_transform`
- `src/core/tensor/CMakeLists.txt` — added .cu to build
- `src/core/tensor/lazy_executor.cpp` — replaced interpretive loop with `fold_affine()` + GPU/CPU dispatch, removed CPU-only gate
- `tests/test_tensor_lazy_ir.cpp` — added 5 tests (GPU parity, GPU reduces-launches, CPU affine fold, GPU identity, div-zero IEEE 754); fixed pre-existing temp-expiry bug in `OnModePointwiseFusionReducesLaunchesWithParity`

**Tests:** 45 TensorLazyIrTest + 87 related suites = all pass.

### PR8: Unary Op Fusion — Extend Pointwise Chain Kernel (2026-02-23)

Extended fusion system to handle mixed chains of scalar + unary ops in a single GPU kernel launch. Pure scalar chains still use the affine fast path (`fmaf`).

**Changes:**
- `src/core/tensor/internal/lazy_executor.hpp` — extended `LazyPointwiseOpKind` with 15 unary op kinds (Abs=10 through Round=24)
- `src/core/tensor/internal/tensor_ops.hpp` — added `FusedPointwiseOp`, `FusedPointwiseOpChain` (16 max ops), `launch_fused_pointwise_chain`
- `src/core/tensor/tensor_fused_pointwise.cu` — added `apply_pointwise_op` device switch (19 op kinds), vec4 + scalar chain kernels, host launcher
- `src/core/tensor/lazy_executor.cpp` — added `is_pure_scalar_chain` routing (pure→affine, mixed→chain kernel), `apply_pointwise_op_cpu` for CPU path, chain length >16 falls back gracefully
- `src/core/tensor/internal/tensor_impl.hpp` — added `LFS_DEFINE_UNARY_OP_FUSABLE` macro, converted 15 ops (neg, abs, sign, reciprocal, exp, log, sqrt, rsqrt, square, sigmoid, relu, tanh, floor, ceil, round)
- `tests/test_tensor_lazy_ir.cpp` — added 8 tests (CPU/GPU pure unary, CPU/GPU mixed, launch count reduction, single unary, interleaved scalar+unary, chain length boundary)

**Tests:** 56 TensorLazyIrTest + 57 TensorViewTest + 84 reserve/inplace = all pass.

### PR9: Reduction Integration — Fused Transform-Reduce (2026-02-23)

Implemented fused transform-reduce: pointwise chains feed directly into reductions without materializing intermediates. Added benchmarks and refactored training hot paths.

**Kernel changes:**
- `src/core/tensor/internal/tensor_ops.hpp` — declared `launch_fused_transform_reduce` (full reduce) and `launch_fused_segmented_transform_reduce` (last-dim reduce) with `FusedPointwiseOpChain` + `ReduceOp`
- `src/core/tensor/tensor_fused_pointwise.cu` — implemented fused transform-reduce kernels (CUB DeviceReduce for full, warp-shuffle for segmented)
- `src/core/tensor/lazy_executor.cpp` — lazy executor now detects pointwise chain → reduction pattern and routes to fused kernels instead of separate materialize + reduce
- `src/core/tensor/internal/lazy_executor.hpp` — added `fused_reduce_launches` diagnostics counter

**Benchmark:**
- `tests/test_fusion_benchmark.cpp` — added 6 benchmarks:
  - Fused full-reduce with chain (100K–4M elements): 1.1–1.9x vs LibTorch
  - Fused segmented reduce with chain (sum, last-dim): 1.2–2.0x
  - All reduction ops (sum/mean/max/min/prod on [2048,1024]): 1.6–4.8x
  - MSE loss `(pred-target).square().mean()`: up to 7.9x
  - L2 norm per row `x.square().sum({-1}).sqrt()`: up to 9.8x
  - Gaussian activation reduce `(-x.abs()).exp().sum({-1})`: up to 3.9x

**Training refactors (x*x → .square() enables fusion):**
- `src/training/components/sparsity_optimizer.cpp` — `(diff * diff).sum()` → `diff.square().sum()` (fused square+sum)
- `src/training/strategies/adc.cpp` — `(rot * rot).sum(-1)` → `rot.square().sum(-1)` (fused square+segmented_sum)
- `src/training/trainer.cpp` — inlined `l1_diff` intermediate in masked L1 loss
- `src/training/metrics/metrics.cpp` — `diff * diff` → `(pred - target).square()` (fused square+mean in unmasked PSNR)

**Tests:** 6 new benchmarks pass, 251 existing tests pass (75 ADC/prune + 176 loss/metrics).

### PR10: Alias/View/In-Place Correctness Hardening (2026-02-23)

Added storage generation counter to detect stale views after base tensor reallocation. Fixed TensorRowProxy thread-safety bug.

**Core mechanism:**
- `StorageMeta` struct with atomic generation counter, shared between base and views
- Lazy allocation: meta created on first view creation (`propagate_view_meta`), not on every tensor allocation — zero overhead for non-view tensors
- `bump_storage_generation()` called in `reserve()` before replacing `data_owner_`
- Debug assertions (`assert_view_not_stale()`) in `ptr<T>()` and `data_ptr()` — compiles to nothing in release

**Changes:**
- `src/core/tensor/internal/tensor_impl.hpp` — `StorageMeta` struct, `storage_meta_`/`view_generation_snapshot_` members, helper methods, assertions in data access, `cuda_staging_` member in `TensorRowProxy`
- `src/core/tensor/tensor.cpp` — propagation in copy/move ctors/assignment, `bump_storage_generation()` + `init_storage_meta()` in `reserve()`
- `src/core/tensor/tensor_shape_ops.cpp` — `propagate_view_meta()` in `permute()`, `slice()` (both overloads)
- `src/core/tensor/tensor_movement_ops.cpp` — `propagate_view_meta()` in `transpose` case
- `src/core/tensor/tensor_row_proxy.cpp` — replaced `thread_local static float` with `mutable float cuda_staging_` member
- `tests/test_tensor_storage_version.cpp` — NEW: 22 tests (storage meta lifecycle, view staleness detection, row proxy correctness)
- `tests/CMakeLists.txt` — added new test file

**Tests:** 22 new tests pass, 167 existing view/memory/capacity tests pass, 170 broader tensor tests pass, 30 expression template tests pass.

### PR11: Stream Consistency Pass (2026-02-23)

Replaced implicit nullptr-stream launches with explicit stream propagation. Added PyTorch-compatible thread-local stream context.

**Changes:**
- `src/core/tensor/internal/cuda_stream_context.hpp` — simplified to pure RAII `CUDAStreamGuard`, exports `getCurrentCUDAStream()`/`setCurrentCUDAStream()`
- `src/core/tensor/cuda_stream_context.cpp` — NEW: thread-local stream management implementation
- `src/core/tensor/tensor_matrix_ops.cpp` — matmul kernels use `result.stream()` instead of `nullptr`
- `src/core/tensor/tensor_broadcast.cpp` — broadcast operations propagate stream
- `src/core/tensor/tensor_unified_ops.cpp` — extensive stream propagation for load/store ops
- Factory functions (`empty`, `zeros`, `ones`, `full`, `rand`, `randn`) pick up thread-local stream
- View operations preserve parent tensor's stream
- `tests/test_tensor_stream.cpp` — NEW: 11 tests (default stream, factory pickup, view inheritance, ordering, guard restore, in-place ops)

**Tests:** 11 new stream tests pass, existing regression suites green.

### PR12: Memory Planner Integration — Early Cache Release (2026-02-23)

Added liveness-driven early cache release to `execute_topological_nodes()`. During lazy execution, intermediates are now freed from the context cache as soon as their last consumer executes, letting `SizeBucketedPool` recycle GPU memory for subsequent allocations. No new allocation paths or materializer signature changes.

**Mechanism:**
- `compute_release_schedule()` builds a step→[dead_node_ids] map from the plan topology. Skips internal fused nodes and the root. O(N*M) where N=nodes, M=avg inputs.
- After each execution step, dead entries are erased from `cached_materializations`. When erased, `shared_ptr<void>` refcount drops → pool dealloc → memory available for reuse.
- Gated by `lazy_executor_memory_planner_enabled()` (on by default, same pattern as fusion gate).

**Measured impact** (from diagnostics dump on test workloads):
- 5-node reshape chain: `ctx_peak=2` (down from 5), 4 early releases, peak 48 bytes vs 120 naive
- 10-step 1MB chain: `peak_cache_bytes=1048576` vs 10485760 naive (10x reduction)

**Changes:**
- `src/core/tensor/internal/lazy_ir.hpp` — added `buffer_bytes` to `LazyExprDebugInfo`
- `src/core/tensor/lazy_ir.cpp` — added `buffer_bytes` to `LazyExprNode`, computed in `register_node_locked`, propagated in `to_debug_info`
- `src/core/tensor/internal/lazy_executor.hpp` — added `buffer_bytes` to `LazyPlanNodeDebug`, added `early_releases`/`early_release_bytes`/`peak_cache_bytes` to `LazyExecutorDiagnosticsSnapshot`, added memory planner gate API
- `src/core/tensor/lazy_executor.cpp` — `compute_release_schedule()`, `LazyExecutorMemoryPlannerState`, early release in `execute_topological_nodes()` via `try_release_dead`/`track_cache_insert` lambdas, diagnostics tracking, memory planner gate, extended debug dump line
- `tests/test_tensor_memory_planner.cpp` — NEW: 6 tests (liveness computation, early release fires, peak bytes reduction, root not released, multi-step chain releases, fusion coexistence)
- `tests/CMakeLists.txt` — added new test file

**Tests:** 6 new TensorMemoryPlannerTest pass, 76 existing TensorLazy{Runtime,Ir}Test pass, 141 regression tests pass.

### PR13: Stateful Op Rules (2026-02-24)

**Goal:** Codify that RNG/stateful ops must never be deferred, with telemetry to verify.

**Changes:**
- `src/core/tensor/internal/lazy_config.hpp` — added `stateful_op_eager` to `LazyTelemetrySnapshot`, declared `telemetry_record_stateful_op_eager()`
- `src/core/tensor/lazy_config.cpp` — wired atomic counter into reset/snapshot
- `src/core/tensor/tensor_unified_ops.cpp` — record telemetry in `LoadOp::Random`, `Normal`, `Randint`, `Bernoulli`
- `tests/test_tensor_lazy_stateful_ops.cpp` — NEW: 11 tests (eager checks, seed reproducibility, GPU variant, telemetry counts, shadow mode)
- `tests/CMakeLists.txt` — registered new test file

**Tests:** 11 TensorLazyStatefulOpsTest pass.

### PR14: Size Heuristic for Tiny Tensors (2026-02-24)

**Goal:** Skip deferral for tensors below 4096 bytes where overhead dominates.

**Changes:**
- `src/core/tensor/internal/lazy_config.hpp` — added `SizeHeuristic = 6` to `LazyFallbackReason`, `eager_fallback_size_heuristic` to telemetry
- `src/core/tensor/lazy_config.cpp` — wired counter and fallback recording
- `src/core/tensor/internal/lazy_executor.hpp` — declared size heuristic API
- `src/core/tensor/lazy_executor.cpp` — implemented `LazyExecutorSizeHeuristicState`, env vars `TENSOR_LAZY_SIZE_HEURISTIC`/`TENSOR_LAZY_SIZE_THRESHOLD`
- `src/core/tensor/internal/tensor_expr_impl.hpp` — size check in `operator Tensor()` before deferral
- `tests/test_tensor_lazy_ir.cpp` — 5 new size heuristic tests, `LazyRuntimeGuard` disables heuristic by default
- `tests/test_tensor_memory_planner.cpp` — `LazyRuntimeGuard` updated
- `tests/test_tensor_lazy_stateful_ops.cpp` — `LazyRuntimeGuard` updated

**Tests:** 5 SizeHeuristic tests pass, 87 full lazy suite pass, 113 regression tests pass.

### Next active step

Complete PR15 (broadcast + where fusion), then widen reduction fusion coverage beyond CUDA Float32 before production rollout.

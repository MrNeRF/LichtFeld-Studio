# Node performance and evaluation-device validation

2026-10-03, unstaged changes above `b46e167cf`. Apple M5 Max, Metal and
Vulkan through MoltenVK. CUDA implementation updated, but not compiled or run
on this Mac. No changes committed or staged.

## Results

Standalone and per-level times are completion-synchronized wall-clock
milliseconds, not GPU timestamps. Live-app numbers are the worker's reported
node/stack wall times; the MCP calls wait for the completed result, but those
individual timers exclude the worker's final publication-fence wait.
Remove Floaters uses relative radius 3 and Min Neighbours 3 on one million
splats. The standalone benchmark retains the original positions/scales but
uses degree-zero SH; the live application measurements retain the full loaded
payload and run through the modifier worker.

| Measurement | Before | Metal after | Vulkan after |
| --- | ---: | ---: | ---: |
| Garden, standalone complete node, three runs | — | 25.59–28.78 | 32.00–40.22 |
| Bicycle, standalone complete node, three runs | — | 51.02–54.69 | 62.20–70.67 |
| Garden, live app complete node, three runs | 14,888.47 measured | 25.34–38.42 | — |
| Bicycle, live app complete node, three runs | 18,900 user report | 48.78–54.88 | — |

Output counts remain **983,779 garden / 966,016 bicycle**, identical across
the optimization stages and Metal/Vulkan. In the live app the corresponding
whole-stack times, including payload preparation, were 36.26–54.64 ms
and 59.33–71.49 ms. Min Neighbours 2 gave 24.11–25.91 ms on garden and
52.65–52.97 ms on bicycle, with 988,589 / 975,015 survivors respectively.

The isolated live app used MCP port 45695. Graph construction and parameter
changes used `nodes_*` tools, followed by `nodes_evaluate(wait=true)`; returned
worker results were saved. The scene tensor backend was confirmed as Metal.

### Causes and changes

1. Saturation already returned immediately at `max_count`; it was not missing.
   Counts searched a corner cell before the query's own cell. Searching the
   current cell first reduced the garden diagnostic node to 724.94–732.96 ms.
2. Count queries also inherited host waits after every 8,192-point build/query
   batch. Keeping their build and query ordered on the worker queue, without
   those waits, reduced garden to 59.53–62.68 ms. Boolean radius queries retain
   their existing bounded-batch scheduling.
3. Cells were twice the radius wide. Radius-wide cells still require exactly
   27 cells for an exact query, but substantially reduce unrelated candidates.
   This produced the final results above. It also reduced a wide-scale,
   unsaturated Neighbour Count stress test from 1,802 / 2,414 ms to approximately
   282 / 377 ms on Metal / Vulkan.

No approximation or candidate cap was added to radius counts. Query masks still
restrict each octave to its own points while all points remain references.
There are still at most eight levels. Each level needs its own grid because
the cell width changes; these builds are now asynchronously queue-ordered.
No device-wide synchronization was added.

### Per-level measurements

Second diagnostic pass; each measurement includes hash construction, querying
and a scalar readback that waits for completion. Radii and query masks are
unchanged across the optimization stages.

Garden:

| Level | Queries | Radius | Center-first only, Metal | No batch waits, Metal | Final Metal | Final Vulkan |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 15 | 0.000092536 | 36.071 | 0.239 | 0.275 | 0.360 |
| 1 | 112 | 0.000822151 | 38.259 | 0.225 | 0.225 | 0.431 |
| 2 | 3,521 | 0.00659514 | 40.442 | 0.304 | 0.305 | 1.292 |
| 3 | 400,221 | 0.0527621 | 62.729 | 2.147 | 1.518 | 1.572 |
| 4 | 536,591 | 0.422095 | 279.558 | 27.204 | 6.023 | 9.190 |
| 5 | 58,074 | 3.37587 | 189.480 | 30.046 | 11.300 | 12.525 |
| 6 | 1,433 | 26.9389 | 40.166 | 0.677 | 1.077 | 1.463 |
| 7 | 33 | 97.1589 | 36.319 | 0.581 | 0.587 | 0.781 |

Bicycle:

| Level | Queries | Radius | Final Metal | Final Vulkan |
| --- | ---: | ---: | ---: | ---: |
| 0 | 25 | 0.000174314 | 0.214 | 0.400 |
| 1 | 403 | 0.00139487 | 0.253 | 0.419 |
| 2 | 51,242 | 0.0111618 | 0.802 | 0.958 |
| 3 | 492,289 | 0.0892946 | 1.817 | 2.458 |
| 4 | 382,261 | 0.714346 | 6.930 | 9.927 |
| 5 | 72,137 | 5.71446 | 5.242 | 5.572 |
| 6 | 1,633 | 44.6202 | 32.650 | 37.889 |
| 7 | 10 | 71.2901 | 1.098 | 0.661 |

### Points to Splats

Auto radius uses the new public `point_neighbor_spacing` operation. It searches
27 cells, expands once if fewer than three neighbours were found, and bounds
each bucket walk to 128 entries. This is an approximate local three-neighbour
spacing estimate, not exact global k-NN. Dense/colliding buckets can therefore
affect the estimate, and GPU atomic insertion order can change approximate
results. Radius remains half the estimated spacing, clamped to 0.25–4 times
the median. Cell width comes from the robust extent of at most 4,096 samples;
no `cdist`, quadratic distance matrix or per-element host loop remains.

Complete conversion, three synchronized runs using garden positions as point
clouds (not the original COLMAP cloud):

| Points | Metal | Vulkan |
| ---: | ---: | ---: |
| 140,000 | 4.87–6.11 | 6.94–9.33 |
| 200,000 | 7.19–7.73 | 8.58–9.45 |
| 1,000,000 | 35.02–35.76 | 37.68–38.54 |

The original 138,766-point COLMAP file was not supplied/located; its exact
10.7-second reported case remains unmeasured. The 160k Extract example was
not separately timed; it uses the same corrected radius-count path.

## Device handling and regression coverage

Evaluation defaults to GPU when the selected backend is available, regardless
of CPU input payloads. An explicit C++ CPU override retains offline/CPU test
coverage. Inputs and outputs are normalized at the evaluator boundary, including
Object Info, conversions and host-library outputs. Join also aligns each operand
to its first input's device/backend as a safety net.

The worker retains uploaded source geometry. `GeometryDeviceCache` caches mesh
attributes and referenced albedo textures by immutable mesh identity, edit
generation and target backend. Read-only mesh snapshots retain identity and
generation. CPU point colours are normalized after upload, on the GPU. Mesh
Transform/Delete/Separate/Join preserve the uploaded textures. GPU-origin mesh
sources are copied away from renderer storage after resolving their texture
cache; publication still uses separate storage and the worker completion fence.

Tests cover CPU mesh plus texture upload/reuse/invalidation, CPU mesh-derived
splats joined with GPU splats, CPU-origin points converted and cached on GPU,
and actual worker Object Info → Transform → Mesh to Splats → Join. An observer
node asserts that the transform's mesh tensors are on GPU. Original stored
CPU payloads remain unchanged. CPU/Metal/Vulkan tests also cover exact counts
against brute force, masks/collisions, local spacing, expansion and empty inputs.

The broader spatial parity check exposed an existing `Tensor::eye` aggregate
initializer that put `0.0f` into the newly added random-seed field rather than
the constant-value variant. Its one-line fix uses designated initialization;
the existing Metal/Vulkan spatial parity test now passes.

## Remaining host paths

Audit scope: every `.cpu()`, `to_vector`, `data_ptr`, `ptr`, scalar read,
`nonzero`/`masked_select`, and host loop in `src/core/nodes` and
`src/visualizer/nodes`.

| Path | Remaining host work / reason |
| --- | --- |
| Simplify | The existing host simplification library; intentionally retained. Its output is transferred to the evaluation device before downstream execution. |
| Stored Selection | Base64 encode/decode and packed-byte JSON serialization; packing/unpacking now uses tensor operations. Only packed bytes and one count are read back on capture. |
| Relative radius | Two scalar bounds plus at most eight scalar level radii; masks, hash construction and counts remain GPU operations. |
| Auto point radius | Three robust extent scalars; spacing and median/clamping stay on GPU. |
| Mesh to Splats | One total-area scalar to choose output count. Host loops only over material/submesh records; sampling, interpolation, texture gathering and rotations are GPU operations. |
| Colour Correct | Two percentile scalars and fixed-size colour-matrix constants. Sample indices now generated on device. |
| Filtering/Delete/Separate/Decimate | Output-cardinality reads from `nonzero` / `masked_select` / count reductions; tensor filtering and sorting remain GPU operations. Decimate's threshold no longer reads back. Mesh filtering reads one count per submesh to maintain CPU material ranges. |
| Inside Mesh | GPU bounds compaction and tiled GPU ray tests; host loops dispatch tiles, not individual point/triangle calculations. Compaction requires its output size. |
| Scalar socket adaptation | One to three values from context-independent fields converted to scalar/vec3 socket values; no geometry-sized host work. |
| Preview statistics | One selected-count reduction per evaluated preview socket, on the worker. |
| Source/payload ownership | Initial uploads, immutable CPU captures and renderer-separated GPU copies. Mesh material/image metadata retains its CPU representation for scene/render/export ownership; pixel sampling does not run on CPU. |
| CPU adapters/fallback | Explicit CPU evaluation, one-time CPU `geometry_from_point_cloud` colour normalization for callers outside worker evaluation, and no-GPU fallback. The worker bypasses that CPU colour conversion and normalizes after upload. |
| Graph/scene management | Registry/JSON/undo/topology/attribute-name/material metadata loops and fixed 3×3/4×4 transform constants, not elementwise geometry processing. |

No geometry-sized `to_vector`/raw-pointer host iteration remains in these node
evaluation paths. CPU and CUDA spatial kernels share their algorithm; CUDA was
reviewed but cannot be validated on this machine.

## Validation

Build command:

```sh
SDKROOT=/Library/Developer/CommandLineTools/SDKs/MacOSX26.5.sdk \
  ninja -C build lfs_core lichtfeld_tests LichtFeld-Studio lfs_py
```

Exit 0. Existing linker warnings remain about Python archives built for macOS
26.6 while targeting 26.0. Final incremental-build tail:

```text
[5/18] Staging Vulkan rasterizer shaders
[6/18] Refreshing lfs_core ABI stamp
[7/9] Copying lfs_plugins Python module
[8/9] Syncing visualizer runtime resources
```

Both `--tensor-backend=metal` and `--tensor-backend=vulkan` used this broader
filter, including parameterized CPU/Metal/Vulkan node tests and spatial parity:

```text
Nodes*:*Nodes*:*RadiusNeighbor*:*radius_neighbor*:*Modifier*:*McpNode*:TensorMetal.SpatialSelectionMatchesVulkan
```

Metal tail:

```text
[==========] 159 tests from 8 test suites ran. (2979 ms total)
[  PASSED  ] 156 tests.
[  SKIPPED ] 3 tests, listed below:
[  SKIPPED ] Backends/NodesCore.CpuMeshUploadsOnceAndJoinsGpuSplats/CPU
[  SKIPPED ] Backends/NodesCoreScale.MillionSplatsColourSelectionCleanupAndTorus/CPU
[  SKIPPED ] Backends/NodesCoreScale.SpatialOperationsStayInteractive/CPU
```

Vulkan tail:

```text
[==========] 159 tests from 8 test suites ran. (3015 ms total)
[  PASSED  ] 156 tests.
[  SKIPPED ] 3 tests, listed below:
[  SKIPPED ] Backends/NodesCore.CpuMeshUploadsOnceAndJoinsGpuSplats/CPU
[  SKIPPED ] Backends/NodesCoreScale.MillionSplatsColourSelectionCleanupAndTorus/CPU
[  SKIPPED ] Backends/NodesCoreScale.SpatialOperationsStayInteractive/CPU
```

GPU-only performance guardrails each assert less than 2,000 ms. The fixture
contains one million randomly jittered, clustered splats spanning sixteen
scale octaves. Exact printed timings from the final Vulkan-flag run:

```text
[NodesSpatialPerformance Metal] Remove Floaters relative 1M 13.3516 ms
[NodesSpatialPerformance Metal] Neighbour Count relative 1M 283.157 ms
[NodesSpatialPerformance Metal] Points to Splats auto 140000 1.54758 ms
[NodesSpatialPerformance Metal] Points to Splats auto 200000 1.83375 ms
[NodesSpatialPerformance Metal] Points to Splats auto 1000000 7.816 ms
[NodesSpatialPerformance Vulkan] Remove Floaters relative 1M 17.7977 ms
[NodesSpatialPerformance Vulkan] Neighbour Count relative 1M 376.639 ms
[NodesSpatialPerformance Vulkan] Points to Splats auto 140000 4.46246 ms
[NodesSpatialPerformance Vulkan] Points to Splats auto 200000 5.37404 ms
[NodesSpatialPerformance Vulkan] Points to Splats auto 1000000 10.2131 ms
```

Python command used the supplied scratchpad virtualenv, `LFS_TEST_BUILD_DIR=build`,
and `tests/python/test_nodes.py tests/python/test_localization_contracts.py -q`:

```text
tests/python/test_localization_contracts.py ..........................   [100%]
============================== 35 passed in 1.76s ==============================
```

Gate tails:

```text
Backend neutrality: 0 violation(s).
no new error-debt violations
No likely hardcoded UI strings found.
clang-format: 28 changed C++/CUDA/ObjC++ files clean
```

`git diff --check` also passed. No localized UI strings changed.

## Files and local evidence

Changed production/test files:

```text
src/core/CMakeLists.txt
src/core/include/core/detail/gpu_backend_ops.hpp
src/core/include/core/mesh_data.hpp
src/core/include/core/nodes/device.hpp (new)
src/core/include/core/nodes/evaluator.hpp
src/core/include/core/nodes/types.hpp
src/core/include/core/tensor_spatial.hpp
src/core/nodes/device.cpp (new)
src/core/nodes/builtin_cleanup.cpp
src/core/nodes/builtin_conversion.cpp
src/core/nodes/builtin_geometry.cpp
src/core/nodes/builtin_input.cpp
src/core/nodes/builtin_splat.cpp
src/core/nodes/evaluator.cpp
src/core/tensor/backend/cuda/cuda_ops_spatial.cpp
src/core/tensor/backend/cuda/kernels/tensor_spatial.cu
src/core/tensor/backend/cuda/kernels/tensor_spatial.hpp
src/core/tensor/backend/facade_entries.def
src/core/tensor/backend/metal/kernels.metal
src/core/tensor/backend/metal/metal_backend_ops.hpp
src/core/tensor/backend/metal/metal_ops.mm
src/core/tensor/backend/vulkan/shaders/radius_neighbors.slang
src/core/tensor/backend/vulkan/vk_backend_ops.hpp
src/core/tensor/backend/vulkan/vk_ops_spatial.cpp
src/core/tensor/internal/point_spatial.hpp
src/core/tensor/tensor_spatial.cpp
src/core/tensor/tensor_unified_ops.cpp
src/visualizer/nodes/modifier_evaluation_worker.cpp
src/visualizer/nodes/modifier_evaluation_worker.hpp
src/visualizer/nodes/modifier_manager_evaluation.cpp
tests/test_nodes_core.cpp
tests/test_nodes_modifier_manager.cpp
docs/docs/development/node-graph/design.md
docs/docs/development/node-graph/v6-validation.md (this report)
```

Ignored local evidence is in `build/node-v6-*`: final `radius-cells-*` logs,
`final-runtime-{garden,bicycle}.jsonl`, `metal-tests.log`, `vulkan-tests.log`,
`pytest.log`, gate logs, build logs and `host-audit.txt`. The reproducible
diagnostic harnesses are `build/node_v6_benchmark.cpp` and
`build/node_v6_runtime.py`. Earlier `center-metal` and `async-metal` logs retain
the intermediate measurements. `build-examples/` was not touched.

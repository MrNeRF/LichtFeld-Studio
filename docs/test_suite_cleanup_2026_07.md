# Test suite cleanup, July 2026

Marker: `TEST-SUITE-CONSOLIDATE`

## Result

The cleanup turns the monolithic native test executable into three bounded CTest
tiers while keeping one executable and one source of truth for each behavior:

- `fast;gpu`: deterministic native unit and component contracts; no training or
  large real-data fixture. The `fast` label also includes a bounded Python API
  and panel contract set.
- `slow;gpu`: tiny training/resume smoke tests and bounded integration fixtures.
- `nightly;gpu`: real-data round trips, stress/fuzz cases, optional codecs, and
  the one inherited-stream race that is not deterministic under concurrent GPU
  load.

Python follows the same ownership model: 36 focused modules are in `fast`, 24
concurrency/lifecycle and plugin integration modules are in `slow`, and seven
real-data, package, scene, and exhaustive Torch-interoperability modules are in
`nightly`. GPU-marked cases in the otherwise-fast tensor modules have their own
`fast;gpu` invocation, so the default tier includes the core Python CUDA
contracts without rerunning the CPU cases. Every tracked `tests/python/test_*.py` module belongs to
exactly one of those lists.

Run the developer/CI tier with:

```sh
cmake --build build --target lichtfeld_tests -j4
ctest --test-dir build -L fast --output-on-failure
```

Run the curated full suite, excluding scheduled nightly coverage, with:

```sh
ctest --test-dir build -LE nightly --output-on-failure
```

Run scheduled coverage with:

```sh
ctest --test-dir build -L nightly --output-on-failure
```

Each label invokes the filtered GTest suite once. Registering every GTest case as
an individual CTest process made process startup dominate the thousands of
sub-second tensor tests. Tier timeouts are 60 seconds (`fast`), 300 seconds
(`slow`), and 1,800 seconds (`nightly`).

## Measured baseline and final runtime

As requested, the runtime baseline was measured after the safe static-removal
commit, after building the Python extension and making the project's real
`./data/bicycle` data visible to the checkout. It ran the native test binary once
so GTest could report per-case timing without CTest process startup noise.

| Measurement | Tests | Result | GTest time | Wall time |
|---|---:|---:|---:|---:|
| Post-static-removal full suite | 3,106 | 2,973 pass, 67 fail, 65 skip, 1 disabled | 40.871 s | 41.80 s |
| Same suite, per-test purge removed only | 3,106 | 2,968 pass, 72 fail, 65 skip, 1 disabled | 33.856 s | 34.84 s |
| Curated `ctest -L fast` | PENDING | PENDING | PENDING | PENDING |
| Curated full, excluding nightly | PENDING | PENDING | PENDING | PENDING |

The listener-only comparison removes 6.96 seconds of wall time (16.7%). The five
additional failures without it were tests that depended on observing global
free VRAM or on another test having purged process-wide caches. Those tests were
rewritten around owned storage or removed when a deterministic owner already
covered the contract.

The slowest original cases were:

| Test | Time |
|---|---:|
| checkpoint resume, MCMC SH0 | 6.816 s |
| PPISP varying-image-size memory probe | 5.134 s |
| pointwise scalar fusion timing | 3.068 s |
| checkpoint resume, MCMC SH1 | 2.018 s |
| out-of-core PLY to RAD | 1.652 s |
| arena metrics contention | 1.454 s |
| long fusion chain timing | 1.286 s |
| large multidimensional NaN/Inf check | 1.147 s |

The original run's 67 failures were recorded by exact GTest name:

<details>
<summary>Baseline failure list</summary>

```text
TensorMemoryTest.MoveSemantics
TensorMemoryTest.InvalidTensorOperations
TensorViewTest.InvalidView
TensorViewTest.InvalidSlice
TensorViewTest.Expand
TensorViewTest.ExpandErrors
TensorAdvancedTest.Linspace
TensorAdvancedTest.Stack
TensorAdvancedTest.ErrorHandlingInvalidTensor
TensorStressTest.MaxMemoryAllocation
TensorStressTest.NormalizationStability
TensorMatrixTest.InvalidMatMulDimensionMismatch
TensorMatrixTest.InvalidBatchMatMulDimensionMismatch
TensorMatrixTest.InvalidDotProductDimensionMismatch
TensorBitwiseTest.BitwiseNotOnNonBoolFails
TensorBitwiseTest.BitwiseOrBroadcast
TensorBitwiseTest.BitwiseOrOnNonBoolFails
TensorRandomAdvancedTest.MultinomialInvalidInputs
TensorRandomAdvancedTest.MultinomialTooManySamplesWithoutReplacement
TensorRandomAdvancedTest.MultinomialAsIndices
TensorClampTest.ClampWithInfinity
TensorConversionsShapesTest.ReshapeInvalidSize
TensorVsTorchTest.KMeansPlusPlusInitialization
TypePromotionTest.SystematicAllCombinations
MemoryLeak.MultinomialRepeatedCalls
UsdFormatTest.RoundtripPreservesValues
TrainingSetupRegressionTest.ApplyLoadedDatasetKeepsFullInitPointCloudUntilTrainingStarts
PythonIntegrationTest.SceneCameraExposesVisualizerRenderContract
VRAMResizeTest.MultipleResizeCyclesNoLeak
ImageResizeParallelTest.LoadImagesWithResize
TensorSlicingSafetyTest.SliceOutOfBounds
TensorLazyIrTest.OnModeDefersUntilBoundaryAndMaterializes
TensorLazyIrTest.OnModeIndexPutMultiDimMismatchBoundaryMaterializes
TensorLazyIrTest.OnModeIndexAddEdgeBoundaryMaterializes
TensorLazyIrTest.OnModeAppendGatherNon1DIndexBoundaryMaterializes
TensorLazyIrTest.OnModeAppendGatherEdgeBoundaryMaterializes
TensorLazyRuntimeTest.ErankExpressionMatchesTorchOnInheritedStream
PythonIOTest.LoadCOLMAPDataset
SceneValidityTest.SceneManagerBlocksActiveCameraSubtreeAndAllowsInactiveCameraRemoval
SceneValidityTest.SceneManagerRemovesAllowedActiveCameraSubtreeWithoutClearingTrainer
ViewportFrameLifecycleServiceTest.ResizeActiveDefersFullRefreshUntilDebounceCompletes
ViewportFrameLifecycleServiceTest.PassiveWindowResizeDefersFullRefreshUntilDebounceCompletes
ViewportFrameLifecycleServiceTest.ExplicitRefreshDeferralCompletesAfterStableFrames
ViewportTest.WasdAdvanceSupportsFlatAdditionalSpeedInVisualizerSpace
InputControllerFocusTest.ToolControlActivationShortcutsResolveAcrossModesAtRuntime
NvCodecImageLoaderJpeg2k16Bit.RoundTrips2160pGrayAndRgbLossless
PipelinedImageLoaderSidecarJpeg2k.DepthFirstTouchTranscodesThenHotDecodes
PipelinedImageLoaderSidecarJpeg2k.CorruptFilesystemSidecarFallsBackAndRepairs
MeshLoaderTest.CanLoadOBJ
MeshLoaderTest.CannotLoadNonMesh
MeshLoaderTest.SupportedExtensions
MeshLoaderTest.LoadCubeOBJ
MeshLoaderTest.LoadCubeHasNormals
MeshLoaderTest.LoadCubeVertexBounds
MeshLoaderTest.LoadCubeIndexBounds
MeshLoaderTest.LoadNonexistentFile
MeshLoaderTest.LoaderName
MeshLoaderTest.LoaderPriority
LodPageDequant.KernelMatchesCpuDecodeOnConvertedFile
LodPageDequant.KernelMatchesCpuDecodeDegree2
LodPageDequant.KernelMatchesCpuDecodeDegree1
LodPageDequant.KernelMatchesCpuDecodeDegree0
CheckpointStrategies/CheckpointResumeTest.TrainSaveLoadResume/mcmc_1
CheckpointStrategies/CheckpointResumeTest.TrainSaveLoadResume/mcmc_2
CheckpointStrategies/CheckpointResumeTest.TrainSaveLoadResume/mcmc_3
CheckpointStrategies/CheckpointResumeTest.TrainSaveLoadResume/mrnf_3
CheckpointStrategies/CheckpointResumeTest.TrainSaveLoadResume/igs__3
```

</details>

## Size

| Scope | Files | Lines |
|---|---:|---:|
| Original native and Python tests | 298 | 134,435 |
| Curated native and Python tests | 223 | 98,307 |
| Removed | 75 | 36,128 |

The line totals cover top-level `tests/*.cpp` and `tests/python/*.py`. Build-system
and product fixes are reported separately by Git.

## Real tensor-library defects found

Suspicious tensor failures were investigated against the implementation before
changing their tests. Eight were genuine library defects:

1. `launch_zip_gather_2` and `launch_zip_gather_3` accepted strides but ignored
   them. Strided permutation iterators now honor every input stride, with exact
   two- and three-input regression cases.
2. Boolean OR rejected broadcastable shapes with an eager same-shape assertion,
   even though the underlying binary operator supports broadcasting. The stale
   restriction was removed and broadcast values are checked.
3. `clamp` rejected `-infinity` and `+infinity`, preventing valid one-sided
   clamps. It now rejects NaN and reversed bounds while preserving infinite
   endpoint semantics.
4. `expand` converted `-1` to `size_t` before resolving the keep-dimension
   sentinel, producing overflow rather than the documented shape. The sentinel
   is resolved before constructing `TensorShape`, and a leading `-1` is rejected.
5. A fused `gather_lazy(...).map(...)` clamped invalid indices while the unfused
   `take` path rejected them. Fused and unfused evaluation now share the same
   negative-index and out-of-bounds contract.
6. `Tensor::full` rejected NaN and infinity even for floating-point tensors.
   Float32 and Float16 constants now preserve non-finite values, while integer
   and Bool construction continues to reject them.
7. Tensor-valued Bool-mask assignment was hard-coded to Float32. CPU and CUDA
   masked scatter now support every tensor dtype, including the Int32 histogram
   path that exposed the defect.
8. Mixed `UInt8` arithmetic ignored the documented promotion hierarchy and
   silently promoted every pair except `UInt8 + UInt8` to Float32. Promotion now
   follows `Bool < UInt8 < Int32 < Int64 < Float16 < Float32` symmetrically, and
   one focused test checks the dtype and values of add/subtract/multiply for all
   36 dtype pairs.

`TensorLazyRuntimeTest.ErankExpressionMatchesTorchOnInheritedStream` passes by
itself and with its neighboring lazy suites but can fail in the full process
under concurrent GPU load. That is evidence of an inherited-stream or allocator
ordering defect, not permission to erase the test. It remains intact in the
nightly tier so the unresolved race stays visible.

## Tensor coverage assurance

The consolidation removed names and duplicated setup, not distinct tensor
behavior. Focused owners still cover:

- construction, shapes, strides, views, slices, expand/reshape/permute,
  contiguous conversion, move/copy/storage ownership, and reserve/capacity;
- all supported CPU dtype conversion pairs, CUDA round trips, promotion, Bool
  masks, and the production mural conversion workflow;
- scalar, pointwise, broadcast, comparison, Bool/bitwise, clamp, NaN/Inf, random,
  reduction, matrix, normalization, and extended unary operations against exact
  values or Torch references;
- indexing, negative indices, gather/take, fused lazy gather, strided zip gather,
  index-select/copy/add/put, masking, nonzero, sort, cumsum, and deleted-row
  filtering;
- cat/stack/split, serialization, lazy IR/stateful operations, streams and
  multistream ownership, allocator policy, and zero-dimensional tensors;
- `cdist`, k-means++ selection, MCMC relocation/histograms, optimizer state,
  densification tensor paths, and representative 100k-row indexing boundaries.

The exhaustive Python/Torch interoperability matrix was deliberately retained;
it is GPU compatibility coverage, not benchmark noise.

## Removed and merged coverage

The first cleanup commit removes only files verified by reading to be one of:

- pure timing/profile programs with no behavioral oracle;
- commented-out, proof-of-install, or manual dump/reproducer code;
- an exact duplicate whose named behavior remains in a focused owner;
- Python tests that pass whether the operation succeeds, fails, or does nothing.

A final oracle audit found no active C++ test source without an assertion path.
Every Python test has an assertion, an expected exception, or a focused helper
oracle; the only three without an inline assertion compile documented plugin
examples, where successful compilation is the contract.

Later consolidation maps duplicated behaviors to focused owners. Important
examples:

| Removed cluster | Surviving owner and salvaged behavior |
|---|---|
| tensor Torch/debug/compatibility dumping grounds | focused dtype, advanced, indexing, reduction, view, movement, ordering/distance, k-means, and lazy-gather suites; every distinct edge was moved before deletion |
| MCMC memory/reproducer files | `test_mcmc.cpp`, focused indexing/optimizer/relocation tests, and checkpoint strategy-state round trips |
| fusion/gather/permute timing files | exact operation owners and `test_zip_gather.cpp`; timing-only loops were discarded |
| pipelined-loader benchmark | `test_pipelined_loader.cpp`, using bounded real image/mask cases for resize, cache-key, ordering, and optional-mask contracts |
| PPISP Torch-only finite-difference tests | native-vs-Torch value/gradient owners; tests that never called native code were removed |
| SOG HTML duplicate | the format/export owners that assert emitted structure and round trip |
| mesh external-fixture test | generated minimal triangle with exact topology/material assertions |
| global VRAM probes | owned-storage, allocator policy, multistream trimming, and slicing-lifetime tests |
| Python GUI catch-all | tensor primitives, operator-property isolation, and typed panel-enum contracts moved to their focused modules; overlapping fixture and property cases removed |
| Python edge/lifetime/cache dumping grounds | exact panel-label fallback and operator-kwarg isolation moved to focused owners; modules that swallowed every exception, never invoked their callbacks, or asserted tautologies were removed |

The catch-all review also found four non-tensor product defects:

- `lf.ops.invoke()` left invocation kwargs on its cached Python operator
  instance. A later call without that kwarg inherited stale state. The binding
  now removes call-local attributes after invocation, matching the retained-UI
  operator binding, and the focused operator-property suite owns both success
  and failure-path regressions.
- Direct `OptimizationParams.means_lr` assignment bypassed property-registry
  notifications even though `set()` notified correctly. That binding now uses
  the registry path, and both callback APIs assert their old/new values.
- Decorator-style training hooks survived normal Python interpreter shutdown.
  Only the embedded app finalizer cleared them, so a standalone extension run
  could destroy nanobind callbacks after Python teardown and segfault. The
  module now registers idempotent hook cleanup with `atexit`, and hook tests
  clear process-global registrations after each case.
- Python property callbacks had the same lifetime hazard: decorator
  subscriptions were retained by the process-global property registry with no
  way to unsubscribe them. The binding now tracks only its Python subscription
  ids, removes them on explicit unsubscribe, clears the remainder at `atexit`,
  and isolates registrations between property tests.

## Slow coverage retiering

- The original checkpoint cross product performed about 12,600 training
  iterations. PR coverage is now direct MCMC/MRNF/IGS+ strategy-state round
  trips plus one real-data MCMC smoke from iteration 2 to 4. The 1,200 to 2,100
  real resume case remains nightly.
- Real image, COLMAP, RAD/LOD, USD/USDZ, UV-runner, video, and Python integration
  cases are slow rather than hidden among tensor units.
- Large stress/fuzz allocations, real PLY/SPZ/SH round trips, and optional JPEG2K
  codec coverage are nightly.
- Contention/event-pool tests keep the same synchronization and accounting
  invariants with bounded worker and iteration counts.

Real-data tests continue to read from `./data/`; tensor operation tests were not
replaced with synthetic substitutes for required production-data workflows.

## Deliberately retained

- Exhaustive tensor and Torch compatibility behavior, including odd dtypes,
  Bool semantics, non-contiguous views, and error contracts. This is correctness
  coverage and was consolidated only where an exact owner existed.
- The inherited-stream Erank test, because its load-dependent failure may expose
  a real ordering defect.
- `test_vram_profiler_metrics.cpp`, despite its profile-like name, because it
  asserts gauge, histogram, live-byte, and iteration-accounting contracts and
  contains no benchmark-only timing loop.
- Focused Python concurrency, hot-reload, operator-property, and plugin-lifecycle
  tests with observable state or ordering assertions. Broad edge/lifetime/cache
  dumping grounds were removed after their real contracts were moved here.
- Real-data round trips and optional-codec tests. They are expensive, so they
  moved to slow/nightly rather than disappearing.

## `test_main.cpp` isolation result

The old listener synchronized CUDA, cleared the CUDA error, emptied Torch's
caching allocator, trimmed the LichtFeld tensor pool, synchronized again, and did
that both before and after every test. No test required that sequence for
correctness. It cost 6.96 seconds across one baseline run, concealed tests that
observed process-global memory, and defeated realistic allocator reuse.

The listener is removed. Pinned-memory prewarming and the explicit shutdown
order remain because they are process-lifecycle correctness, not per-test cache
purging.

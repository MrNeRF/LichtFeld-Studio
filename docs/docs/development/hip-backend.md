# Native HIP backend

HIP is an optional Tensor and training backend on Windows and Linux. It does not
require CUDA, Torch, or PyTorch. Rendering uses the separately selected rendering
backend. Enabling HIP does not change the defaults of other backends.

## Developer build

Use a complete ROCm development SDK and the application's normal dependencies:

```sh
cmake -S . -B build-hip -G Ninja \
  -DLFS_ENABLE_CUDA=OFF \
  -DLFS_HIP_RUNTIME=ON -DLFS_TENSOR_HIP=ON -DLFS_HIP_TRAINING_RUNTIME=ON \
  -DLFS_HIP_ROOT=/absolute/path/to/sdk -DLFS_HIP_ARCHITECTURE=gfx1150
cmake --build build-hip --target LichtFeld-Studio
```

CUDA can also remain enabled in the same build. The HIP code is compiled with
the SDK's compiler independently of the application's C++ and CUDA compilers.
Use separate build directories for different target/profile configurations.

## Portable builds

`LFS_HIP_PORTABLE=ON` includes the declared wave32 targets in the generic runtime
and training modules. `LFS_HIP_ARCHITECTURES` can explicitly restrict the target
list, for example `-DLFS_HIP_ARCHITECTURES="gfx1103;gfx1150"`. An explicit list
takes precedence over the portable default. Unsupported targets fail configure.

The generic modules contain one device image per requested target. They are not
an emulation layer. The HIP runtime selects the image for the actual GPU.
HIPRTC uses the selected device's architecture in both its compiler options and
expression cache identity, independently of the process's current device.

Multiarchitecture builds use conservative FastGS kernels. With
`LFS_HIP_ARCH_PROFILES=ON`, a build containing `gfx1150` additionally includes a
separate validated `gfx1150` training profile. Single-target developer builds
retain their compile-time profile choice. `LFS_HIP_VALIDATED_TUNING=OFF` disables
the specialized kernel options, including cached options inherited from an
earlier configuration. Common stable sort and workspace improvements remain
available in the conservative path.

## Packaging and module selection

The `lfs_hip_package` target stages the native modules in `hip/` beside the
executable, with architecture-specific training modules under
`hip/profiles/<architecture>/`. SDK runtime dependencies are staged beside the
executable. The same layout is included by `cmake --install`.

The application prefers this adjacent package, then its developer build path.
Training selects a matching architecture profile when available. A profile
which cannot load or has incompatible metadata falls back to the generic
module and logs a CLI warning. The failed profile is not retried on every
operation. Fatal device errors after GPU work begins are not silently replayed.

`LFS_HIP_DISABLE_ARCH_PROFILES=1` selects the generic module in a package with
separate profiles. It cannot change the kernels in a single-profile developer
binary. `LFS_HIP_TRAINING_MODULE` is an explicit diagnostic override: a failed
override reports an error rather than silently choosing another file.

HIP affine bilateral backward also selects a stride-five shared histogram on
the actual selected `gfx1150` device. Both layouts are compiled into the HIP
color kernels: other devices and `LFS_HIP_DISABLE_ARCH_PROFILES=1` retain stride
four. Exposure/chroma retains stride four on every device. This selection does
not increase the shared-memory requirement of guide grids larger than 64 bins;
those grids retain the generic layout even on the validated device. It does
not alter the original CUDA kernels, parameterization, sampling, or RGB-gradient
calculation. Histogram accumulation remains FP32 and can differ in atomic
summation order. Validate both settings with the bilateral contract, including
its global-atomic fallback and shared-histogram partial-tile cases.

## Support boundaries

The trainer currently requires wave32. Portable builds cover RDNA target IDs
declared in `cmake/HipArchitectures.cmake`; GCN/CDNA wave64 targets are rejected.
Compiling a target is not a claim of runtime or numerical validation on that
hardware. GPU/driver availability remains subject to the selected SDK and OS.
The tested device profile is `gfx1150`; other targets use conservative kernels.

The profile's native FP32 atomics are restricted to session-owned backward
helper gradients allocated by `FastFrame`. Model storage ownership does not
establish ownership of output bindings. Densification statistics, edge/impact
scores, camera gradients and regularization outputs therefore retain ordinary
HIP `atomicAdd`, including when the model itself uses shared Vulkan storage.
Do not broaden native-atomic dispatch to imported or fine-grained allocations.

Fused L1/SSIM backward selects the per-plane 32x8 HIP kernel only for Float32
targets on `gfx1150`, with at least 1,048,576 pixels per image and a supported
dispatch grid. UInt8 targets, small images, masked/decoupled loss and other
devices retain the canonical kernel. FP16 partial storage and the forward loss
are unchanged. The adapted kernel records its upstream PR and immutable source
revision in `application_photo_plane_kernel.hpp`.

`LFS_HIP_DISABLE_PLANE_SSIM=1` disables this specialization; the existing global
`LFS_HIP_DISABLE_ARCH_PROFILES=1` also disables it. Set these before process
startup: an executable and its HIP DLL can have separate CRT environments.
`LFS_HIP_VALIDATION_PHOTO_DISPATCH=1` traces the actual Float32 fused dispatch
and should be disabled for timing. The dispatch neither allocates another
photometric workspace nor changes the target dtype.

## Completion status and remaining work

### Execution sequence

Complete the following gates locally before replacing the portable package.
Keep experiment reports and measurements outside source commits.

1. **Strategy correctness:** reproduce the known MRNF failure configuration past
   its failure boundary, preserving the configured optimization schedule when
   stopping the test early. Exercise IGS pruning, released-slot reuse, frozen
   parameters, opacity reset, SH transitions, checkpoint and resume. Retain the
   exhaustive half and typed broadcast contracts informed by the AMD port.
2. **AMD-port candidates:** compare group32 reductions on the complete generic
   backward, including gradients and timing, before adopting a microbenchmark
   gain. Preserve the validated OCKL profile when it wins. Audit subgroup masks,
   ballots and shuffles as a separate wave64 portability task; compiling a target
   is not hardware validation and must not silently enable unsupported kernels.
   Preserve the half conversion and complete stream-dependency regression tests.
3. **Viewer independence:** reduce redundant HIP-to-other-device publication,
   with explicit mutation invalidation and immutable in-flight snapshots. Verify
   unchanged views, native training writes, topology changes, same-device shared
   memory and cross-device staging, both idle and during training.
4. **Startup throughput:** separate cold/warm HIPRTC, ingest, prepared CPU cache,
   transfers and training work. Optimize measured bottlenecks while preserving
   cache budgets, memory bounds and correct stream ordering.
5. **Telemetry:** expose device-matched utilization and meaningful process/device
   memory budgets, including validity and shared-memory semantics on integrated
   GPUs. Keep these independent of CUDA device availability.
6. **Distribution:** validate offload compression, remove actual duplication,
   repair repeatable DLL/license packaging and preserve the supported target list.
7. **Final release gate:** run the build matrix and the established comparable
   2000-step protocol once the changes stabilize; verify quality, memory, speed,
   UI restart/device selection and the new complete portable package.

The backend is integrated, but it is **not yet validated as a complete training
replacement**. A successful build, populated operation tables, isolated kernel
contracts, or a short MCMC run do not establish complete strategy coverage.

| Area | Current boundary | Required completion evidence |
|---|---|---|
| MCMC | Application training and mutation paths have dedicated tests; portable smoke tests cover only a short run. | Retain existing checks and exercise the final packaged modules through mutation, save and resume. |
| MRNF | The known reported configuration passes beyond its failure boundary with the original schedule and an early stop. Application contracts also cover SH transitions, repeated refinement/Morton, checkpoint and resume. | Retain the composed Tensor regressions and complete the final comparable quality/performance run. Native callback contracts alone are insufficient. |
| IGS / ImprovedGSPlus | Application lifecycle coverage includes frozen parameters, explicit pruning and slot reuse, SH transitions, opacity resets, appearance, checkpoint/resume and process shutdown. | Retain lifecycle coverage and complete the final comparable quality/performance run. Short accelerated contracts do not establish long-run convergence or validation on other GPUs. |
| Tensor dtype/layout coverage | Binary broadcast supports matching Float32/Float16 operands, native integer arithmetic/comparisons, and Bool predicates. Bool arithmetic is rejected. | Audit actual strategy call sites and shared Tensor contracts, including unequal shapes, strided/expanded views, empty inputs and integer/Bool predicates. Preserve exact integer/Bool semantics; do not cast indices through Float32. |
| GPU utilization | The HIP UI path explicitly returns an unavailable sample. | Add device-matched sampling independently of CUDA availability; distinguish a real 0% sample from unavailable data. |
| GPU memory budget | HIP allocator statistics and `hipMemGetInfo` exist, but the UI does not expose a complete native process-budget path. | Connect meaningful process/device/budget measurements and their validity to the UI; distinguish allocated, reserved and driver-budget values. |
| Training startup | Short cold/warm diagnostics confirm prepared CPU cache population costs and near-zero loader waits after the first image pass. The serialized HIP queue and appearance stages still need separate attribution. | Optimize measured preparation and queue bottlenecks without changing sampling or quality; retain synchronized diagnostics and unprofiled confirmation. |
| Viewer/Tensor independence | Idle HIP publications are cached with bounded payloads and explicit invalidation. Short application checks cover camera reuse, edits, same/mixed GPU publication and pause transitions. | Complete bounded asynchronous training previews and interaction-latency measurements; static publication tests alone do not establish active-training responsiveness. |
| Hardware/platform validation | Runtime evidence is limited to the available GPU; Windows cross-target compilation is not execution on those targets. | Run the manual build matrix and preserve a conservative path for untested hardware. Wave64 remains outside the implemented scope. |

The known configuration behind the reported broadcast rejection is covered by
an integrated early-stop test; typed storage-level kernels alone would not
establish that the composed MRNF path is complete. Other dtype restrictions in
`HipBackendOps` must be compared with the shared API and real callers before
being classified as missing functionality. Do not turn every shape/type guard
into an unconditional promise of support or remove validation to silence a
failure. IGS reserves its free mask to model capacity: HIP slot fill accepts
that reserved suffix, validates that live rows fit, and must not modify the suffix.

### Correctness completion order

1. Capture the MRNF failure's operation, input/output dtypes, shapes, strides,
   selected device and phase. Add a focused regression case for the composed
   path, then implement the correct HIP operation without changing CUDA, Metal
   or Vulkan dispatch or semantics.
2. Check non-floating broadcast and related operation coverage against MCMC,
   MRNF and IGS call sites. In particular cover Bool mask combinations, integer
   indexing/comparison and promotion, selection, scatter, reductions and sorting
   where the strategies actually require them. A specialized direct path may
   coexist with a correct generic implementation; an implicit CPU fallback is
   not completion of native HIP support.
3. Add short application tests for each strategy with accelerated schedules that
   actually execute split/prune/compaction/slot reuse, frozen masks, SH changes,
   configured losses and checkpoint/resume. These runs are correctness checks,
   not comparable timing or quality measurements. Include the user's failing
   MRNF configuration with bilateral grid and PPISP rather than validating only
   a reduced configuration.
4. Recheck the packaged CUDA+HIP build and HIP-only build with the same regression
   cases, default profile selection and conservative fallback. Check failures
   before and after mutation separately; do not replay an already-mutated step.
5. Use the established undistorted 1920/SH0/1M-cap/2000-step protocol for the final
   comparable check, not after each small change. Preserve the numerical,
   quality and visual checks; report remaining uncertainty explicitly.

### GPU telemetry and memory-budget completion

`src/visualizer/gui/gpu_memory_query.cpp` currently marks HIP native metrics
unavailable and returns `-1` for utilization. The HIP device-info path does not
publish a process budget equivalent to the Vulkan budget extension. Pool
statistics alone cannot replace that OS/driver budget.

- Match telemetry to the selected physical GPU using stable device identity,
  not enumeration position. Tensor and rendering devices can differ; label
  their roles and do not accidentally show NVIDIA utilization for AMD training.
- On Windows investigate the existing DXGI process-budget path and a suitable
  AMD/OS utilization provider. On Linux investigate the supported AMD telemetry
  interfaces. Provider choice and dependency availability need verification;
  these are implementation tasks, not completed integrations.
- For integrated GPUs distinguish firmware-reserved memory, shared host memory,
  current OS budget and process allocation. A large `hipMemGetInfo` total is not
  the amount of dedicated VRAM and is not an appropriate standalone UI budget.
- Keep driver-budget polling separate from HIP allocator accounting. Include
  cached blocks and algorithm workspaces in their own categories; avoid double
  counting imported Vulkan storage. A staging copy on another GPU is separate
  storage and must be attributed accordingly.
- Handle unavailable/stale samples explicitly, throttle polling, and invalidate
  identity-specific caches when devices change. Test active and idle training,
  both same-GPU and mixed-GPU rendering, without requiring a CUDA device.

### Slow initial training steps

Treat the reported initial slowdown as unresolved; do not dismiss approximately
100 slow steps as expected warmup. Separate the optimizer's configured learning
rate warmup from runtime initialization costs. Record steps 1-10, 11-50, 51-100
and a later steady window, keeping the actual strategy and settings in the record.

Measure expression-cache misses and HIPRTC compilation/loading, native module
initialization, CPU decode, resize/undistort and prepared-image cache population,
allocation/pool growth, host/device transfers, forward, losses, backward,
optimizer, strategy post-backward, and viewer synchronization. Include FSR and
cross-GPU preview transfers when active. Compare fresh-process cold and warm
runs with fixed inputs; exclude concurrent builds from performance evidence.
Only then choose precompilation, bounded prewarming, scheduling, caching or kernel
changes. Keep CPU image-offload and configured resize behavior intact.

### Mixed-GPU viewer independence

The acceptance matrix covers every supported Tensor/render backend pairing,
including different physical devices within one backend. Device equality is
established by stable physical identity, never by matching backend names or
enumeration indices. Unsupported combinations must report their limitation
explicitly rather than silently selecting another GPU.

| Producer/consumer relationship | Publication requirement |
|---|---|
| Same physical GPU, compatible storage | Preserve the existing shared-memory and fence path where supported. |
| Same physical GPU, incompatible APIs/storage | Use a validated conversion/import path or bounded staging. |
| Different GPUs, same backend | Validate peer-copy support for the actual device pair; otherwise use bounded asynchronous host staging. |
| Different GPUs and backends | Use validated external/peer interoperability where available; otherwise bounded asynchronous host staging. |

For each supported row verify stopped, paused and active training, model edits,
resize/densification, project replacement, memory pressure and fallback. Camera
navigation must be able to render the latest completed coherent preview while a
new preview is being produced; it must not wait for every training step or
accumulate an unbounded preview queue. Measure interaction latency and frame-time
distribution in addition to training throughput. Preserve existing paths until
their replacements pass equivalent correctness and regression checks. Hardware
pairs unavailable locally remain unverified and must be recorded as such.

Treat the reported GeForce Vulkan renderer plus 890M HIP Tensor configuration
as an unresolved performance issue. Separate device selectors do not prove that
the render loop is independent of Tensor execution. Cross-GPU staging is an
expected compatibility path, but its presence alone does not explain or justify
slow viewing of a relatively small scene.

- Confirm effective renderer and Tensor device identities after restart; do not
  infer active devices solely from saved preferences.
- Reproduce camera motion with training stopped, paused and active. Keep scene,
  viewport size and reconstruction settings fixed; measure native rendering and
  FSR separately. Do not assume the reported observation occurred during training.
- Trace where render inputs live, when a Tensor-backed scene is converted or
  copied for Vulkan, bytes transferred, queue waits and CPU staging work. Identify
  any HIP work or global synchronization on the viewer's critical path.
- For an unchanged model, verify that camera-only motion reuses renderer-local
  storage instead of copying the full model from HIP on every frame. During
  training, inspect generation tracking, preview publication cadence and whether
  GPU/CPU waits serialize training and presentation unnecessarily.
- Compare same-device and mixed-device configurations without weakening device
  identity checks or importing incompatible storage. Any snapshot reuse,
  asynchronous staging or throttling must preserve lifetimes, mutation ordering,
  coherent preview data and visible/logged fallback behavior.

Completion requires responsive idle viewing and measured, bounded preview-update
cost during training. Do not claim that this independence is already verified,
or that the discrete renderer makes the Tensor device irrelevant to rendering.

The HIP VkSplat idle-input path retains publication leases per buffered view.
Camera-only frames can reuse those buffers, including host-staged publications
on a different GPU. The cache accounts for at most 256 MiB of retained payloads
across views; driver allocation padding and in-flight leases are additional.
Over-budget publications remain usable but are not retained. Failed optional
publications are retried. Normal submission fences and frame-retirement leases
remain required on both cache hits and misses.

Model snapshots, explicit splat dirtiness, view retirement and renderer reset
invalidate retained entries. Training and LOD page inputs bypass the cache;
selection/query preparation conservatively invalidates it. This does not solve
active-training preview synchronization. CUDA, Metal and Vulkan Tensor inputs
continue through their existing publication paths.

HIP model publication also excludes viewer reads while strategy post-backward
and allocator migration can replace Tensor objects, including non-refining
steps. A pending pause retains live synchronization until the worker acknowledges
its step boundary; the earlier UI state transition alone is insufficient to
enable idle caching. These guards are HIP-specific and do not change the native
CUDA, Metal or Vulkan synchronization policy.

For isolated diagnosis, `LFS_HIP_DISABLE_VIEWER_PUBLICATIONS_CACHE=1` restores
per-frame HIP publication. `LFS_HIP_VALIDATION_PUBLICATION_TRACE=1` emits debug
messages identifying the view slot, field, cache mode and actual loader calls.
Keep tracing disabled for performance measurements. The headless HIP viewer
interop contract verifies cached leases, source reallocation, invalidation,
failed-load retries, budget exhaustion and GPU consumption after invalidation;
application-level frame and edit validation remains a separate requirement.
Pass a consumer device index followed by `--large` to
`lfs_hip_viewer_interop_contract` to exercise publication, reallocation and
consumer copyback with more than 1 MiB of logical tensor data. Omitting the
flag retains the small contract. The contract initializes logging so backend
selection and staging fallbacks are visible in its CLI output.

The current HIP adapter admits one context queue: `wait_for` validates its token
and synchronizes that queue. HIP/Vulkan shared storage also uses host-completed
ownership transfers, and its `ready` path synchronizes HIP. The portable ingest
worker creates a separate Tensor work queue only for Vulkan. Consequently,
simply enabling another worker or adding a queue scope cannot establish HIP
overlap. Completing asynchronous HIP publication requires explicit queue/storage
dependencies and bounded snapshot ownership, plus failure and teardown tests;
do not remove the current waits without replacing their correctness guarantees.

The next active-training implementation should publish complete preview
snapshots at producer step boundaries and expose only completed generations.
Navigation renders the last completed snapshot while the next one is pending.
Do not mix fields from different steps, reuse a consumer-owned buffer, or hold
the live-model lock across host staging. Bound outstanding generations and bytes;
when capacity is exhausted, coalesce refresh requests and retain the last good
preview instead of queuing more copies. Training stop, in-place edits, project
replacement, densification and device changes need explicit generation handling.
Final saved/exported model data must still come from the authoritative model.

## Distribution size reduction plan

The current universal package contains an uncompressed device-code image for
each declared target, plus a complete optimized `gfx1150` training module.
Distribution-size work is outstanding; the large package is not an optimized
release-size baseline. Benchmark reports and measured artifact sizes belong in
external development records, not in source commits.

1. **Native offload compression:** the installed SDK compiler advertises
   `--offload-compress`. Apply it to the relevant HIP device bundles, verify
   actual compression of the final artifacts, and test generic/profile loading,
   fallback and HIPRTC on Windows and Linux. Measure disk size, initialization
   time, host-memory peak and training correctness; a recognized compiler flag
   alone is not validation. Do not recompress prebuilt SDK DLLs with this flag.
2. **Remove profile duplication:** inspect host/device sections and determine
   how much common rasterization, utility and training code the generic and
   optimized modules duplicate. Separate only the specialized kernels if this
   materially reduces the payload. Preserve module lifetime, version/build-set
   checks and generic fallback; an ABI redesign needs its own compatibility tests.
3. **Inspect linking:** inventory exported symbols, sections, instantiated
   kernels and repeated device libraries. Check dead-code elimination and
   identical-code folding where supported. Preserve C ABI exports and any
   dynamically looked-up kernel; do not blindly strip or disable optimizations.
4. **Optional family packages:** an installer may deliver the common application
   and only the selected HIP architecture/family. Keep the universal offline
   package as a separate option; reducing target coverage is a product choice,
   not a transparent optimization. Generic-family targets or SPIR-V need a
   separate compatibility/performance investigation before replacing native
   targets.
5. **SDK dependencies:** keep `amd_comgr.dll` and the matching runtime components
   required by the current HIPRTC expression path. Cache hits do not eliminate
   cold-start compilation. Optional download moves these bytes out of the base
   package but does not remove their installed cost for HIP users. Do not assume
   a compatible copy comes with the graphics driver.
6. **Packaging correctness:** the generic Windows library-exclusion regex can
   exclude build/dependency directories whose names contain `windows`. Validate
   recursive PE imports, loader search directories and Python DLL placement with
   a reduced PATH. Fix the actual packaging rule before relying on manual DLL
   completion for repeatable distributions. License-reference extraction must
   not open web URLs as Windows UNC paths; preserve full local license coverage.

For each candidate compare the same feature set, target list and build type.
Record uncompressed installed bytes and compressed download bytes separately.
Do not promise a tenfold reduction or remove COMGR, supported architectures,
training functionality or quality checks merely to reach a size target.

References: [AMD code portability and compression](https://rocm.docs.amd.com/projects/llvm-project/en/docs-7.14.0/conceptual/code-portability.html),
[Clang offload compression](https://clang.llvm.org/docs/ClangCommandLineReference.html),
[HIPRTC in SDK 10.1](https://rocm.docs.amd.com/projects/HIP/en/docs-10.1.0/how-to/hip_rtc.html).

## Verification

`cmake/hip-native` builds only native modules for cross-target compilation checks
without the host application. It requires the SDK, Slang compiler, GLM headers,
Python, and the same source checkout. No GPU is required to compile it.

`tests/hip/profile_policy` verifies compile-time profile selection without an
SDK. `tests/hip/module_policy` verifies module paths and target matching without
a GPU. Native numerical and application contracts require a compatible GPU.
The module-policy test also checks the photometric dtype/path/device gates and
dispatch limits. `tests/hip/photo_plane_parity.py <contract-executable> <module>
<new-fixture-directory> --expect-plane` compares fresh generic, selected and
global-opt-out processes on a validated device. It checks complete gradients,
loss and workspace sizes, including UInt8 preservation, valid padding and odd
multi-tile image sizes. Keep the generated fixture directory outside tracked
sources; the comparison requires no Torch dependency.

Four dedicated GitHub workflows cover Windows/Linux, HIP-only/CUDA+HIP, each in
Release and Debug. They are exclusively `workflow_dispatch`; existing workflows
are unchanged. These jobs verify builds and CPU policy checks, not GPU quality.
Benchmark measurements and development reports are kept outside source commits.

The application trainer contract accepts `--strategy=mcmc`, `--strategy=mrnf`
or `--strategy=igs+`. Its `--appearance` option enables both bilateral grid and
PPISP. Combine these with `--exercise-mutations` for an explicitly accelerated
correctness run, and pass the resulting project as `resume_project` to exercise
checkpoint restoration. Accelerated runs are not performance or quality baselines.
`--float-targets` uses the existing high-precision image-loading option to
exercise the Float32 loss dispatch. It is a diagnostic fixture option; omit it
for the standard UInt8 comparison protocol.

`--profile` enables PerfBench with zero warmup so diagnostics include the first
step. Use the HIP validation step/phase/loader trace controls to distinguish
startup from later windows. Synchronized diagnostic phases are not an unprofiled
throughput benchmark; keep those controls out of normal application launches.

`--stop-after=N` stops after committed progress reaches N while retaining the
configured total iteration schedule. The checkpoint is checked against the
actual completed iteration. `--exercise-lifecycle` requires a fresh IGS run
with `--exercise-mutations`: it freezes a prefix, prunes both frozen and
trainable rows, verifies frozen exclusion and slot reuse, accelerates opacity
reset and checks all frozen parameters. Morton is disabled only in this mode
to keep row identity stable; the mutation contract tests it separately.
Use `--require-frozen` on resume to require the frozen prefix and check its
parameters again. Tests use the normal process-exit hooks before asserting HIP
shutdown, so Q16 run scratch and cached HIP frozen masks release in order.

When testing a development build containing a staged `hip/` directory, verify
the module actually selected beside the executable. Rebuilding the native module
in its intermediate target directory does not refresh an older staged copy.
The isolated adapter contract takes the runtime DLL path explicitly.

The adapter contract enumerates every binary16 encoding for conversions to all
supported dtypes and for `pow`/`mod` with finite and nonfinite right operands.
It checks signed zero, NaN/infinity classification and offset guards. The CPU
math oracle quiets signaling half NaNs before libm evaluation, matching HIP's
half-to-float conversion; NaN payload equality is not required. `mod` must match
exactly, while `pow` permits at most one binary16 ULP after rounding.

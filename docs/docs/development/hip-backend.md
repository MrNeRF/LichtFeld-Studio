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

## Support boundaries

The trainer currently requires wave32. Portable builds cover RDNA target IDs
declared in `cmake/HipArchitectures.cmake`; GCN/CDNA wave64 targets are rejected.
Compiling a target is not a claim of runtime or numerical validation on that
hardware. GPU/driver availability remains subject to the selected SDK and OS.
The tested device profile is `gfx1150`; other targets use conservative kernels.

## Completion status and remaining work

The backend is integrated, but it is **not yet validated as a complete training
replacement**. A successful build, populated operation tables, isolated kernel
contracts, or a short MCMC run do not establish complete strategy coverage.

| Area | Current boundary | Required completion evidence |
|---|---|---|
| MCMC | Application training and mutation paths have dedicated tests; portable smoke tests cover only a short run. | Retain existing checks and exercise the final packaged modules through mutation, save and resume. |
| MRNF | The real application trainer has a dedicated short mutation/refinement contract with optional bilateral grid and PPISP. The original reported configuration still needs a matching reproduction. | Preserve the composed Tensor regression coverage and pass the original failing phase plus subsequent refinement transitions. Native callback contracts alone are insufficient. |
| IGS / ImprovedGSPlus | Registered through the normal strategy factory; consumes HIP MCMC, MRNF and refinement operations. A short application-level mutation contract exercises the strategy; this is not complete lifecycle/quality validation. | Exercise initialization, edge/error scoring, split, prune, slot reuse, frozen masks, opacity reset, SH changes, save and resume. Do not infer full strategy correctness from callback contracts or MCMC tests. |
| Tensor dtype/layout coverage | Binary broadcast supports matching Float32/Float16 operands, native integer arithmetic/comparisons, and Bool predicates. Bool arithmetic is rejected. | Audit actual strategy call sites and shared Tensor contracts, including unequal shapes, strided/expanded views, empty inputs and integer/Bool predicates. Preserve exact integer/Bool semantics; do not cast indices through Float32. |
| GPU utilization | The HIP UI path explicitly returns an unavailable sample. | Add device-matched sampling independently of CUDA availability; distinguish a real 0% sample from unavailable data. |
| GPU memory budget | HIP allocator statistics and `hipMemGetInfo` exist, but the UI does not expose a complete native process-budget path. | Connect meaningful process/device/budget measurements and their validity to the UI; distinguish allocated, reserved and driver-budget values. |
| Training startup | Very slow initial roughly 100 steps reported in the application; the cause is not established. | Profile cold and warm startup with per-phase timing before selecting an optimization. |
| Viewer/Tensor independence | Slow 3D viewing reported with Vulkan rendering on a GeForce and HIP Tensor processing on the 890M, even with a relatively low-detail scene. Whether training was active in this specific observation is not established. | Verify the effective devices and profile idle viewing separately from active training, including transfers, synchronization and unchanged-frame reuse. |
| Hardware/platform validation | Runtime evidence is limited to the available GPU; Windows cross-target compilation is not execution on those targets. | Run the manual build matrix and preserve a conservative path for untested hardware. Wave64 remains outside the implemented scope. |

The reported application broadcast rejection still requires an integrated
reproduction; adding typed storage-level kernels alone does not establish that
the composed MRNF path is complete. Other dtype restrictions in
`HipBackendOps` must be compared with the shared API and real callers before
being classified as missing functionality. Do not turn every shape/type guard
into an unconditional promise of support or remove validation to silence a
failure. IGS uses Bool masks, but an identical IGS failure has not been reproduced.

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

Four dedicated GitHub workflows cover Windows/Linux, HIP-only/CUDA+HIP, each in
Release and Debug. They are exclusively `workflow_dispatch`; existing workflows
are unchanged. These jobs verify builds and CPU policy checks, not GPU quality.
Benchmark measurements and development reports are kept outside source commits.

The application trainer contract accepts `--strategy=mcmc`, `--strategy=mrnf`
or `--strategy=igs+`. Its `--appearance` option enables both bilateral grid and
PPISP. Combine these with `--exercise-mutations` for an explicitly accelerated
correctness run, and pass the resulting project as `resume_project` to exercise
checkpoint restoration. Accelerated runs are not performance or quality baselines.

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

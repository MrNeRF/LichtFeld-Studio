# Native HIP backend

HIP is an optional Tensor and training backend on Windows and Linux. It does not
require CUDA, Torch, or PyTorch. Rendering uses the separately selected rendering
backend. Enabling HIP does not change the defaults of other backends.

## SDK and build prerequisites

The four opt-in HIP CI workflows pin the ROCm SDK Python distribution to
**10.1.0** and use Python 3.13 for the isolated SDK environment. The HIP API
inside this SDK has its own version (7.16); these are different version schemes.
The application retains its normal Python dependency and does not use Torch.
Use the SDK built for the host OS. The SDK supplies build tools and libraries;
it does not install an AMD display/kernel driver.

On Windows, use an x64 Visual Studio developer shell and install the SDK outside
the checkout:

```powershell
$sdkEnv = "$env:LOCALAPPDATA/LichtFeld/rocm-sdk-10.1"
py -3.13 -m venv $sdkEnv
& "$sdkEnv/Scripts/python.exe" -m pip install --index-url https://stable.repo.amd.com/rocm/whl-next/ 'rocm[devel]==10.1.0'
& "$sdkEnv/Scripts/rocm-sdk.exe" init
$env:LFS_HIP_ROOT = & "$sdkEnv/Scripts/python.exe" -c 'import sysconfig; from pathlib import Path; print(Path(sysconfig.get_paths()["purelib"])/"_rocm_sdk_devel")'
```

On Linux, install the normal upstream build prerequisites, then use a separate
SDK environment. CI compiles on Ubuntu 24.04 with GCC/G++ 14 and Python 3.13:

```sh
python3.13 -m venv "$HOME/.cache/lichtfeld/rocm-sdk-10.1"
sdk_env="$HOME/.cache/lichtfeld/rocm-sdk-10.1"
"$sdk_env/bin/python" -m pip install --index-url https://stable.repo.amd.com/rocm/whl-next/ 'rocm[devel]==10.1.0'
"$sdk_env/bin/rocm-sdk" init
export LFS_HIP_ROOT="$("$sdk_env/bin/python" -c 'import sysconfig; from pathlib import Path; print(Path(sysconfig.get_paths()["purelib"])/"_rocm_sdk_devel")')"
```

The mixed CUDA+HIP presets also need the upstream CUDA toolchain. The manual CI
installs CUDA 12.8 automatically for those jobs. HIP-only presets need no CUDA
installation. A final portable package ships HIP runtime libraries, not the
SDK's Python environment, headers, compiler executables or development archives.

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

The existing build presets and default CUDA/Vulkan portable configuration are
preserved. ROCm distribution uses separate build directories and presets:

| OS | CUDA + HIP portable | HIP-only portable |
|---|---|---|
| Windows | `windows-rocm-portable` | `windows-rocm-only-portable` |
| Linux | `linux-rocm-portable` | `linux-rocm-only-portable` |

These inherit the normal Windows/Linux release profiles and enable compressed
universal HIP modules and the matching optimized device profile. For example:

```sh
cmake --preset windows-rocm-portable
cmake --build --preset windows-rocm-portable
cmake --install build-windows-rocm-portable --prefix build-windows-rocm-portable/dist
```

Use the Linux preset names on Linux, or the `rocm-only` variants to omit CUDA.
Set `LFS_HIP_ROOT` first. Local iterative host builds can explicitly add
`-DLFS_FAST_COMPILE=ON`; compiler cache is inherited from the normal build preset.
Native HIP kernels retain their optimized compilation flags.

FSR retains its separate upstream opt-in. To include it as in the Windows
nightly package, add `-DLFS_ENABLE_AMD_FSR3=ON` and point
`LFS_AMD_FSR3_ROOT` at the existing FidelityFX SDK checkout. ROCm presets do
not change the FSR defaults of existing presets.

The four workflows `hip-windows.yml`, `hip-ubuntu.yml`,
`hip-cuda-windows.yml` and `hip-cuda-ubuntu.yml` remain **workflow_dispatch only**.
They automatically install/cache the pinned SDK without Torch, initialize it,
locate its headers, and compile all declared HIP targets in Release and Debug.
They also check compressed target coverage, single-target profile isolation,
module policy and runtime packaging. These are compile/CPU checks, not evidence
of GPU execution on the hosted runners. Normal upstream CI remains unchanged.

`LFS_HIP_PORTABLE=ON` includes the declared wave32 and wave64 targets in the generic runtime
and training modules. `LFS_HIP_ARCHITECTURES` can explicitly restrict the target
list, for example `-DLFS_HIP_ARCHITECTURES="gfx1103;gfx1150"`. An explicit list
takes precedence over the portable default. Unsupported targets fail configure.

The generic modules contain one device image per requested target. They are not
an emulation layer. The HIP runtime selects the image for the actual GPU.
HIPRTC uses the selected device's architecture in both its compiler options and
expression cache identity, independently of the process's current device.

Multiarchitecture builds use conservative FastGS kernels. With
`LFS_HIP_ARCH_PROFILES=ON`, a build containing `gfx1150` additionally includes a
separate validated `gfx1150` training profile. Every raster, color, mask and FastGS
object in that profile is compiled only for `gfx1150`; the generic module keeps
all requested architectures. Single-target developer builds
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

The trainer accepts physical wave32 and wave64 while preserving logical 32-lane
tiles in the rasterizers. Portable builds cover the RDNA and GCN/CDNA target IDs
declared in `cmake/HipArchitectures.cmake`. FastGS and 3DGUT both use the native
HIP training adapter, including checkpoint and resume.
Compiling a target is not a claim of runtime or numerical validation on that
hardware. GPU/driver availability remains subject to the selected SDK and OS.
The tested device profile is `gfx1150`; other targets use conservative kernels.
Wave64 target admission is a portability path, not a claim of validation on every
GCN/CDNA board. Physical wave64 kernel tests on RDNA exercise subgroup semantics;
they do not replace tests on native wave64 hardware.

For SDK-only development, `cmake/hip-native` accepts
`LFS_HIP_TEST_WAVE_SIZE=32` or `64`. This option is absent from normal application
builds. A forced wave64 test on RDNA requires `LFS_HIP_VALIDATED_TUNING=OFF`
and `LFS_HIP_ARCH_PROFILES=OFF`. It uses conservative rocPRIM device primitives
and a private copy of rocPRIM headers to align host dispatch with the forced
physical wave. The installed SDK is never modified. The adaptation checks its
source anchor and rejects an unreviewed SDK layout. No test override or adapted
header is needed for normal GCN/CDNA compilation.

`tests/hip/logical_subgroup_contract.hip` takes the expected physical wave size
as its sole argument. Run the matching binary for each wave size; it checks
independent 32-lane halves, masks, ballots, shuffles, divergence, cooperative
tiles, partial blocks and multidimensional thread layouts. Application tests
must additionally cover FastGS and 3DGUT gradients, sorting, indexing, image
preparation, real training and checkpoint resume. The trainer fixture accepts
`--rasterizer=3dgs` or `--rasterizer=3dgut`; short mutation checks are correctness
tests and must not be presented as performance benchmarks.

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

### CPU and GPU test entry points

The four manual HIP workflows run the SDK-independent CPU suite before compiling
the application and the `lfs_hip_compile_contracts` target. The latter compiles
the host-side HIP contracts, including the native viewer consumers; it does not
run GPU work. Hosted GitHub runners are not GPU validation.

Run the CPU suite independently with a C++23 compiler, Python and the application's
GLM headers. It links neither Core nor HIP/CUDA runtime libraries:

```sh
cmake -S tests/hip/cpu -B build-hip-cpu -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DLFS_CPU_DEPENDENCY_INCLUDE=/absolute/path/to/dependencies/include
cmake --build build-hip-cpu
ctest --test-dir build-hip-cpu --output-on-failure --no-tests=error
```

The suite covers producer/consumer exclusion and recovery with synthetic storage,
GPU identity collisions, independent renderer/tensor restart policy, native
memory-sample mapping, architecture validation, Windows packaging exclusions,
compression policy and deterministic adaptation of the current upstream kernel
sources. An optional `-DTEST_SDK=/absolute/path/to/sdk` additionally checks the
compiler's positive offload-compression capability without executing a GPU.

For local GPU execution, configure the application with
`-DLFS_HIP_REGISTER_GPU_TESTS=ON`, build `lfs_hip_compile_contracts`, then run
`ctest --test-dir <application-build> -R "^hip[.]gpu[.]" --output-on-failure`. HIP tests carry
both `hip` and `gpu` labels and run serially. The three contracts requiring MoGe
weights/reference data, LPIPS weights, or a fresh checkpoint result directory
retain their explicit CLI arguments and are compiled but not automatically
registered. The dataset trainer and performance probes also keep their documented
manual protocols. Do not infer dataset quality or performance from these checks.

`lfs_hip_secondary_viewer_contract <Vulkan-device> [shared|staged]` exercises the
actual point-cloud renderer and reference-image publication with HIP tensors.
Run it once on the same physical adapter and once with a distinct Vulkan adapter
when available. Set `LFS_VK_VALIDATION=1` before startup to request validation layers;
check that the layer is available and inspect errors as well as the exit status.

### Shared policies on CPU-only runners

Backend selection and restart behavior also run independently of HIP in
`tests/gpu_policy`. The existing
Windows, Linux and macOS build workflows run this SDK-independent suite using
their configured compiler and GLM headers, with an isolated pytest environment.
This covers five configurations: Windows Release/Debug, Linux Release/Debug,
and macOS Release. It compiles the shared policy both with
and without HIP declarations and runs synthetic CUDA/Vulkan/Metal/HIP device
scenarios, dependency-exclusion policy, and the common Python preferences,
rendering and keymap tests. The four manual HIP workflows include this same suite;
their architecture, native identity, synchronization and telemetry contracts stay
HIP-specific. Their manual triggers are unchanged.

The Python cases use only pytest and source modules; the native extension is not
loaded. These tests carry CPU labels and complement the existing application and
backend-native suites. They do not replace real rendering, allocation or training
checks, and their success must not be described as hardware validation.

### Secondary viewer consumers and memory telemetry

Point-cloud rendering prepares Tensor uploads/conversions before acquiring the
HIP/Vulkan consumer scope, then keeps shared buffers Vulkan-owned through command
recording and submission. It retains direct shared vertex buffers on a matching
adapter. Reference images have a different lifetime: their frame is submitted
after UI callbacks which may themselves use HIP. They publish a completed,
immutable CPU snapshot into renderer-owned storage on generation changes, instead
of retaining external HIP ownership across UI callbacks. Cached generations do
not repeat that upload. The ordinary native Vulkan splat rendering path is
unchanged by this reference-image policy.

HIP memory display maps only samples matched to its selected adapter. A valid
native process budget is distinct from whole-device usage; missing telemetry
stays unknown and does not borrow CUDA telemetry. CPU tests cover valid, absent,
stale-invalid and over-budget samples. These tests must not require telemetry to
be absent on a machine where Windows DXGI supplies it.

The neutrality gate recognizes HIP operation adapters and native implementations
at the same boundary as the existing CUDA/Vulkan implementations. Its named HIP
policy seams allow public backend selection only: private Tensor includes and
raw HIP API calls remain forbidden in neutral consumers. Reviewed catch handlers
document their C-ABI status conversion, optional fallback, or failure-path owner
retention with the existing census annotation. The kernel-launch gate recognizes
HIP error checks without relaxing the existing CUDA check window.

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
   gain. Preserve the validated OCKL profile when it wins. Keep the logical 32-lane subgroup contracts on physical wave64; compiling a target
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
| GPU utilization | Windows samples OS GPU engine counters matched by LUID/node; Linux reads the selected PCI device's AMDGPU busy counter. Missing samples remain unavailable. | Retain physical-device matching and active-load tests; validate Linux runtime and additional hardware separately. |
| GPU memory budget | Windows publishes the selected HIP adapter's DXGI LOCAL process usage and budget to the status bar and performance HUD. Allocator statistics remain separate. | Preserve validity and identity checks. Linux process-budget support remains unavailable rather than substituting whole-device VRAM or inventing a shared-memory limit. |
| Training startup | Short cold/warm diagnostics confirm prepared CPU cache population costs and near-zero loader waits after the first image pass. The serialized HIP queue and appearance stages still need separate attribution. | Optimize measured preparation and queue bottlenecks without changing sampling or quality; retain synchronized diagnostics and unprofiled confirmation. |
| Viewer/Tensor independence | Idle publications are cached. A single live HIP model on a non-shared rendering device uses bounded, complete asynchronous CPU snapshots; native shared storage retains its existing path. | Preserve lifecycle, allocation-fallback and real-model attribute checks. Extend interaction measurements to additional hardware and scene modes that retain synchronized publication. |
| Hardware/platform validation | Runtime evidence is limited to the available GPU; Windows cross-target compilation is not execution on those targets. | Run the manual build matrix and preserve a conservative path for untested hardware. Wave64 subgroup support is implemented; execution on native wave64 hardware remains unverified. |

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

The HIP device-info path publishes Windows DXGI LOCAL process usage and budget
using the runtime's exact LUID/node. This OS-managed segment can include shared
memory on integrated GPUs; neither the BIOS reservation nor `hipMemGetInfo`
defines that budget. Failed or stale queries are marked unavailable. Budget
polling is throttled independently from the HIP submission queue and allocator.
The status bar and performance HUD show process usage against this budget;
tooltips identify the measurement source. Whole-device usage is not inferred.

Windows utilization uses English PDH GPU Engine counters, matched to the same
LUID/node, summing clients per engine and selecting the busiest engine. It needs
two samples and runs in the background performance sampler; initialization does
not sleep or synchronize HIP. Linux reads `gpu_busy_percent` at the runtime's
exact PCI sysfs path. Missing support returns `-1`, distinct from a valid zero.
Linux currently has no equivalent process-budget integration.

`lfs_hip_os_metrics_contract <HIP runtime module> [require-native]` rejects null
and unknown Windows identities, checks a real GPU load, device identity and, in
an application build, the GUI dispatch and tooltip. The optional argument makes
native budget and nonzero load samples mandatory on the validation host.

- Match telemetry to the selected physical GPU using stable device identity,
  not enumeration position. Tensor and rendering devices can differ; label
  their roles and do not accidentally show NVIDIA utilization for AMD training.
- Windows uses only system DXGI/PDH libraries, independently of CUDA. Validate
  Linux's AMDGPU counter on Linux hardware; investigate process-level memory
  support separately without relabeling device-wide accounting as a budget.
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
overlap. Do not remove the established native shared-storage waits without
replacing their ownership guarantees.

### Asynchronous training previews

For a single live model on a renderer without HIP shared storage, the trainer
captures all attributes on its producer queue before subsequent parameter
writes. One packed readback is in flight at a time. Completed CPU snapshots
retain matching count, SH layout, bounds and deletion mask; their explicit
generation prevents false cache hits when allocator addresses are recycled.
Navigation can reuse the last complete generation while training advances or
waits for ingest. Final saves and exports continue to use the live model.

Capture is demand-driven, limited to one admission per 100 ms and stops after
two seconds without viewer requests. The 256 MiB host payload budget counts the
pinned slot and CPU generations still owned by readers. Payloads larger than a
third of that budget use synchronized publication. Driver padding, allocation
metadata and the separate bounded Vulkan publication cache are additional.
Immutable consumer buffers are reused across frame slots, rather than copied
again for each buffered frame. A partial or over-budget capture never replaces
the authoritative model or repeats an optimizer update.

The default path preserves native HIP/Vulkan shared allocations on the same
physical GPU. Aggregates, split views, point clouds, LOD, appearance-corrected
views and exact surface queries retain their existing synchronization policy.
Pause, stop, edits and project replacement return to the authoritative scene;
captured generations remain owned until their readers release them.

`LFS_HIP_ASYNC_VIEWER=0` restores synchronized active-training publication.
Capture failures and consumer snapshot-allocation failures emit a CLI fallback
message. Consumer allocation rejection retains the last viewport output and
retries the live-model path, with a cooldown before another optional capture.
`LFS_HIP_VALIDATION_HOST_PREVIEW_OOM=1` injects a pre-submit rejection for
application-level fallback tests. Keep it unset for normal use and benchmarks.
`LFS_HIP_VALIDATION_ASYNC_PREVIEW_TRACE=1` logs actual snapshot consumption;
tracing must be disabled for latency comparisons.

`lfs_hip_preview_snapshot_contract [--large]` checks complete FP32/FP16
generations, producer storage replacement, retained readers, reset, cooldown and
budget rejection. The application trainer's `--verify-preview` compares every
captured attribute byte against its stopped real model, including SH codes and
bounds. Consumer read/write tests remain in `lfs_hip_viewer_interop_contract`.

## Distribution size reduction plan

The separate ROCm portable presets compress device-code images for every declared
target and include a matching optimized `gfx1150` training module. Windows
universal native modules have passed isolated functional contracts, cold HIPRTC,
short application training, preferred-profile selection, global opt-out and
corrupt-profile CLI fallback. Binary inspection checks every embedded target;
these checks do not certify execution on untested GPUs. Complete application
installation and native Linux/hardware validation remain separate gates.
The large existing package
is not an optimized release-size baseline. Benchmark reports and measured artifact sizes belong in
external development records, not in source commits.

1. **Native offload compression:** `LFS_HIP_OFFLOAD_COMPRESSION=ON` applies
   `--offload-compress` to the Core runtime, every raster/color/mask/FastGS object,
   and generic/profile training modules. Default OFF preserves existing builds.
   Configuration rejects a selected compiler without support. Target lists,
   tuning and math flags are unchanged. Isolated Windows SDK builds pass native
   contracts, exact image comparisons, cold/warm HIPRTC and short application
   strategy tests with compressed modules. A multiarchitecture/profile configure
   check verifies all generated compiler/linker commands; it is not execution on
   other GPUs. Before default enablement, validate the complete universal package,
   generic/profile fallback, Linux runtime, startup/memory and final training
   quality. Do not recompress prebuilt SDK DLLs with this compiler flag.
2. **Profile target isolation:** profile raster/color/mask objects are compiled only for the profile architecture, while the generic module retains the universal target list. This prevents foreign architecture payloads from entering the profile. Further inspect host/device sections and determine
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
6. **Packaging correctness:** opt-in Windows HIP portable builds use an anchored,
   case-insensitive match of the actual `SystemRoot`, including mixed path
   separators, instead of excluding every path containing `windows`/`System32`.
   Other build modes retain their existing dependency filters. License-reference
   extraction resolves local paths/basenames only, including references found in
   URLs, and retains complete local notice bodies. Preserve these regressions.
   Validate the complete install, recursive PE imports, Python DLL placement,
   explicitly bundled SDK dependencies and SDK notices with a reduced PATH.
   The OS dependency resolver may find a driver runtime in System32; do not use
   that result to remove the matching SDK DLL explicitly staged for the package.

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


### Independent image preparation

HIP Tensor work queues retain their existing shared native submission stream.
The loader can instead own a bounded private native image producer on the same
physical HIP device. Its stream identity is checked against the training stream;
its GPU buffers never escape to ordinary Tensor storage. Completed CPU RGB bytes
are published for the existing optional JPEG blob cache, followed by a reduced
image upload and byte-preserving layout conversion on the training queue.
This does not imply concurrent hardware execution or independent memory bandwidth.

The producer handles cached UInt8 RGB forward undistortion only. Full-resolution
interleaved decoder ownership is retained without a CPU planar copy; the producer
uses existing image conversion, warp and strided-copy kernels. JPEG quality,
chroma, invalidation keys, cache budget and warp/quantization semantics are unchanged.
Float/16-bit inputs, alpha/masked preparation and requests without prepared-cache
eligibility retain the original path. The generic Tensor queue contract and the
CUDA, Metal and Vulkan implementations are unchanged.

Automatic enablement is limited to the validated gfx1150 architecture profile.
`LFS_HIP_INDEPENDENT_IMAGE_PREPARE=0` forces the original path; `=1` explicitly
exercises the generic producer on another supported HIP device.
`LFS_HIP_DISABLE_ARCH_PROFILES=1` disables automatic architecture selection.
A producer has a 512 MiB total GPU-buffer budget, reuses its scratch allocations,
and reports its owned storage to Core accounting. Producer work is serialized;
CPU JPEG encoding can overlap the next producer request. Shutdown joins loader
workers before releasing the producer and its native stream.

On producer failure, the loader disables it for that loader lifetime, emits one
CLI transfer-fallback warning and prepares the retained source on the original
training queue. Concurrent requests retain shared producer ownership until their
work finishes. Cache encoding remains optional: its failure preserves the already
prepared image. `LFS_HIP_VALIDATION_IMAGE_PREPARE_OOM=1` injects a failure after
private GPU preparation, before CPU publication, for fallback validation only.
The image contract covers both CHW/HWC publication, all supported forward lens
models, exact bytes against the original path, reuse, concurrent requests,
budget rejection, recovery and owned-storage release. Its optional JPEG argument
also exercises a full-resolution source with max-side 1920 preparation.

### Runtime payload and notices

Windows HIP packaging retains HIP, HIPRTC and COMGR as explicit dynamic roots,
then resolves their recursive SDK DLL imports. A build-directory manifest
passes the complete file list to staging without shell/list splitting. OpenCL is included when the SDK
actually imports it. When the SDK dependency inspection tool is unavailable,
packaging keeps the conservative dependency set. An incomplete SDK or an
unresolved import fails packaging instead of producing a broken distribution.
System and graphics-driver files are not copied as SDK dependencies.

Linux runtime staging follows soname symlink chains and preserves the aliases,
avoiding a separate physical copy of the same library for each soname. SDK
LICENSE, COPYING and NOTICE files are retained under the installed data directory
in `licenses/rocm`, with their component directories. Development headers,
compiler binaries and the SDK Python environment are not part of this payload.

The four ROCm portable presets opt into device-code compression. Existing
upstream presets and workflows retain their original options. The target graph
check verifies that generic objects retain all declared targets and that each
profile object contains only its named target. This build-time check does not
replace execution on other hardware or validation of the final installed app.

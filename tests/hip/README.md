# Isolated HIP bring-up

This directory is a standalone CMake project. The application does not include
it and does not advertise an incomplete HIP tensor backend. Probes are off by
default; the disabled configuration needs no SDK, compiler, or GPU.

On Windows, run the following from a Visual Studio x64 developer shell, using
an explicitly selected ROCm SDK root and a CMake version of at least 3.30:

```powershell
cmake -S tests/hip -B build/hip-probes -G Ninja `
  -DLFS_HIP_BUILD_PROBES=ON `
  -DLFS_HIP_ROOT=C:/local/rocm-sdk `
  -DLFS_HIP_ARCHITECTURE=gfx1150
cmake --build build/hip-probes
ctest --test-dir build/hip-probes --output-on-failure
```

The compiler runs in an explicit build rule. Neither the application's host
compiler nor its CUDA flags change. A kernel smoke checks affine arithmetic,
gather with offsets and a partial workgroup, untouched prefix/suffix guards,
and 32 Adam updates against CPU parameters and moments. The current smoke
deliberately requires one HIP adapter with a `gfx1150` target. It reports the
runtime version and the loaded DLL path on Windows.

To also compile a HIP DLL with Clang and call it from an MSVC/CUDA host, configure
with these additional options:

```powershell
-DLFS_HIP_PROBE_CUDA_COEXISTENCE=ON `
-DLFS_HIP_CUDA_ROOT="C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.8"
```

This optional test keeps CUDA device data live, runs the HIP kernel through a
C/POD interface, checks the CUDA bytes afterward, and verifies that CUDA remains
writable. It needs an NVIDIA GPU and an AMD HIP GPU. It establishes runtime and
compiler coexistence, not a concurrent multi-GPU training performance claim.

For the experimental Windows Vulkan sharing probe, also pass
`-DLFS_HIP_PROBE_VULKAN_INTEROP=ON`. Vulkan headers and its import library must be
discoverable by `find_package(Vulkan)`; `Vulkan_INCLUDE_DIR` and `Vulkan_LIBRARY`
can identify an existing SDK explicitly. This test compiles with MSVC and the
selected HIP headers, then loads that SDK's runtime by its absolute path.

It matches HIP PCI identity to the Windows adapter address behind Vulkan's LUID
when HIP does not supply a usable LUID. Nonzero PCI domains and ambiguous matches
are rejected. It checks one model buffer shared by both APIs, external ownership
handoffs, and 32 GPU roundtrips. A native D3D12 fence handles the HIP-to-Vulkan
handoff; a monotonic 64-bit control word and `hipStreamWaitValue64` additionally
protect the return to HIP. The latter API is marked Beta by the selected SDK and
is capability-checked. The ordinary imported-semaphore wait was insufficient in
the local runtime; it must not be treated as a safe standalone implementation.
Host waits at the end of each test cycle allow verification and command reuse;
they are not inserted between the two GPU handoffs being verified.

This is an experimental synchronization proof, not a production failure-recovery
protocol. Queue failure, cancellation, resource exhaustion, growing model storage,
and different driver releases still need explicit tests. The application must
retain a bounded host-completion path until those contracts are established.

For an isolated SDK installed with AMD's Python packages, initialize the devel
files with `python -m rocm_sdk init` and obtain its root with
`python -m rocm_sdk path --root`. Keep compiler temporary files in a writable
directory; the Windows sandbox's default temporary directory can reject the
device link output. Do not alter system drivers to run these probes.

These tests do not implement the application's tensor interface, rasterizer,
checkpoint integration, or viewer synchronization. Passing them must not be
reported as passing a complete HIP training run or beating Vulkan.

The probes also build the reusable private module in
`src/core/tensor/backend/hip`. On Windows, `HipRuntime.OwnershipQueuesAndBounds`
loads it from an MSVC executable and checks device selection errors, context
ownership, range/overflow rejection, offset guards, stream ordering, in-place
operations, and cleanup with live buffers and queued work. It checks 1,048,595
float values, including a non-full final block. A GPU absence is a failure, not
a skipped pass. This test does not cover driver resets or allocation exhaustion.

In the application build, the foundation module is controlled by the separate
`LFS_HIP_RUNTIME=OFF` option. Enabling it requires an explicit SDK root and GPU
architecture and does not register a selectable tensor/training backend.

### Experimental raster training

Add `-DLFS_HIP_PROBE_GSPLAT_TRAINING=ON` and point
`LFS_HIP_GLM_INCLUDE_DIR` to a directory containing `glm/glm.hpp`.
Python 3 is required only for this option (`Python3_EXECUTABLE` may select it).
The build generates HIP adaptations of this checkout's gsplat forward/backward
into its build directory, preserving the application CUDA sources. Only RGB
instantiations are built. Host allocation infrastructure is omitted; the test
provides its own small buffers, nonblocking stream and conservative tile lists.
Wave masks use HIP's required 64-bit type, and tiled reductions use shuffle XOR.
The executable rejects hardware reporting a wave size other than 32.

`HipTraining.GsplatGradientsAndConvergence` checks 6 Gaussians;
`HipTraining.GsplatBatchTailsAndConvergence` checks 263 translucent Gaussians,
crossing both the 128-forward and 256-backward staging boundaries. Both use a
35x29 image (partial edge tiles), compare RGB against an independent CPU renderer
with double-precision geometry, check central finite differences in all five
parameter families, then optimize means, rotations, log scales, RGB and opacity
with FP32 Adam for 400 steps. The target comes from the CPU renderer. Gradient
tolerances combine 1e-6 absolute error with 2% relative error; the finite-difference
epsilon is 1e-4. The loss must fall below 2% of its initial value and all parameters,
gradients and rendered pixels must remain finite. PPM renders are saved in each
test's separate results directory.

These are synthetic correctness tests, not dataset benchmarks. They omit
projection/binning/sorting, image ingest, SSIM, SH conversion, refinement,
checkpointing, memory budgets, viewer interop and the application Tensor adapter.
Reported wall time includes four checkpoint renders/readbacks in addition to
the 400 training steps. CUDA/Mac application regression and full-resolution HIP
performance cannot be inferred from these tests. This path is not yet callable
through the training UI or the reusable runtime module's C ABI.

`HipTraining.GsplatDynamicBinningAndConvergence` adds native HIP projection,
tile counts, hipCUB int64 prefix scan, intersection emission, radix sort and tile
offset construction before every render of the 263-Gaussian fixture. It validates
counts, scan, keys, Gaussian IDs, tile ranges and sentinels against a CPU rectangle
enumeration, then runs the same image/gradient/convergence checks and an all-culled
frame (zero intersections, zero RGB and zero parameter gradients). Projection uses
conservative geometric extents without opacity-aware shrinking. The HIP adaptation
clamps floating tile coordinates before unsigned conversion at image edges; this
change does not modify the original CUDA source.

This third test includes binning cost in its wall time. It still omits application
Tensor/ingest/trainer/viewer integration and all dataset benchmarks. Its tiny-scene
allocation bound is `Gaussians * tiles + 1`, including an explicit sentinel slot;
this is intentionally a fixture bound, **not a production allocation policy**.
Real images require bounded adaptive intersection capacity and UMA budgeting.

### Training module C ABI

`LFS_HIP_PROBE_TRAINING_SESSION=ON` builds `src/training/hip` and loads its training
DLL from an MSVC host. The generator now lives in `src/training/hip/prepare_gsplat.py`;
tests and the module share the same kernel object target. The module has its own
adaptive intersection storage and allocation budget, not the tiny fixture's
worst-case capacity.

`HipTraining.SessionCAbiBudgetAndConcurrency` checks the independent CPU render,
SH0 activation/clamp derivative, model uploads/readback, invalid devices/cameras/
rates/targets, budget failures, transactional frame replacement, first-update
consistency after recovery, 400-step convergence, two sessions on separate host
threads, cancellation/resume, translated camera and empty frame, and destruction.
It saves PPMs in `training-session-results`. With
`LFS_HIP_PROBE_CUDA_COEXISTENCE=ON`, the host also keeps a 1 MiB NVIDIA allocation
alive, verifies all its bytes and writes to it after unloading the HIP DLL.

`LFS_HIP_PROBE_LARGE_SESSION=ON` adds `HipTraining.SessionOneMillionModelBudget`.
It checks rejection at a 128 MiB budget, then allocation and a step for one million
model entries at 512 MiB. Only 263 entries are visible at 35x29: the test proves
allocation/count scaling and preservation of invisible entries, not throughput
for a million visible Gaussians. Logical buffer counters exclude driver heaps.

The session C ABI is callable from host code, but is not yet wired into the
application Tensor/trainer/UI. Its current SH0/MSE/pinhole feature limits are in
`src/training/hip/README.md`. No full dataset benchmark or CUDA/Mac training
regression run is implied by these standalone tests.

With both `LFS_HIP_PROBE_TRAINING_SESSION=ON` and
`LFS_HIP_PROBE_VULKAN_INTEROP=ON`, `HipTraining.VulkanSharedModelOneMillion` tests the
product module's shared-model import on Windows. It uses the SDK HIP runtime and
Vulkan device-local storage (64 MiB plus allocation padding), six Vulkan GPU copies
of all parameter families, guarded tail bytes and allocation lifetime after session
destruction. Three sessions train for 100 steps: shared, private and independent
private control. Only 263 of a million entries are visible, at 35x29.

The first shared/private update must agree within 1e-5. After 100 atomic-gradient
updates, max raw-parameter delta must remain below 1e-3, max RGB delta below 1/255,
RGB MSE between renders below 1e-8, and target MSE within 1% (plus 1e-9 absolute)
of the range of the two private controls. Target loss must improve at least 10x.
All parameters/images must be finite. The alias check comparing Vulkan GPU readback
with HIP readback of the same session remains byte exact. No performance comparison
is derived from this guarded test, which performs large diagnostic readbacks and
trains three sessions. Queue handoffs use host completion, not beta stream waits.

### Unmasked L1/SSIM

`HipTraining.PhotometricCpuValueAndGradient` runs 60 input/weight/padding cases
against independent double-precision 2D convolution and gradient accumulation,
with finite-difference checks of the CPU derivative. Shapes include 1x1, odd and
partial tiles and dimensions cropped independently. Inputs cover mixed L1 signs
and exact ties. Relative gradient L2 error must stay below 1% (the inherited
FP16 partials are intentional); scalar loss error must stay below 5e-5. Trailing
gradient guards must remain untouched.

`HipTraining.PhotometricSessionLossAndRecovery` checks 48 actual session scalar
reductions against the CPU oracle, default MSE, seven invalid loss descriptors,
full/partial workspace allocation failures, restoration of the budget, released
SSIM storage on pure L1, resizing, unchanged parameters during diagnostics and
an exact-match L1 step preserving every parameter. Three 300-step runs check
convergence for weights 0, 0.2 and 1 with a live CUDA allocation in the process.

`HipTraining.VulkanSharedModelPhotometric` extends the one-million-entry sharing
test to L1 plus 0.2 SSIM. The same base rates are multiplied by
`max(0.02, 0.95^step)` in all three sessions. The former fixed MSE fixture rates
oscillate near L1's cusp; the scheduler is test input, not a silent rate change
inside the module. The first-update and byte-exact shared alias checks remain.
After 100 steps, compare target MSE and photometric objective with both private
controls (1% plus the stated absolute floors), RGB RMSE below 1/255 and max RGB
delta below 4/255. Raw-parameter equality is not required for this new nonsmooth
trajectory. The existing MSE test retains its previous stricter thresholds.

### Spherical harmonics

`HipTraining.SphericalHarmonicsValueGradientAndLayout` runs 100 combinations of
active/layout degree 0–3, sizes 1/31/32/33/263, mask tails and absent optional
directions/masks/storage. It compares colors and canonical coefficient gradients
against independent double-precision real-SH polynomials and viewing-direction
gradients against finite differences. Output guards and inactive coefficients
are checked. CPU tolerances are 2e-6 (colors/coefficients) and 3e-6 (directions).

`HipTraining.ShSessionBudgetActivationAndConvergence` tests canonical/swizzled
round trips, inactive SH0 identity, invalid/nonfinite inputs, partial allocation
failure, clearing/reconfiguration before training, deferred first allocation,
camera translation, degree changes, zero rates, inactivity with live moments,
color-clamp derivatives and a culled zero direction. The first update with zero
SH-rest is compared with the existing SH0 path. Three 200-step SH-rest-only MSE
runs and an SH3/L1+0.2 SSIM run compare against CPU-generated targets with geometry
and SH0 frozen. This is not multiview or real-dataset quality validation.

With the existing large-session option, `HipTraining.ShSessionOneMillionBudget`
checks that one-million-entry SH3 state is rejected transactionally at 512 MiB,
then initializes SH1 and completes a step in that same session and budget. Only
263 entries are visible at 35x29. This is a memory/admission test, not a throughput
measurement. SH-rest remains private HIP storage; full higher-SH viewer sharing
is not asserted by any of these tests.

### Application layout and resident model

`HipTraining.CoreSplatStorageLayoutContract` compiles the real core storage
headers with the host compiler and checks mapping and ten rejected contracts.
`HipTraining.VulkanApplicationLayoutAndGuards` creates a Control describing the
actual imported Vulkan allocation: core family order, leading 64 KiB padding,
one million live rows and 50,000 reserved rows per region. It uses the product
layout adapter, uploads an initializer, destroys that session, then attaches a
new training session without host model inputs. Nine bad resident models are
rejected (NaN in each family, null quaternion, scale overflow/underflow and
infinite opacity), with bitwise preservation checked after each failure.
Seven malformed external layouts are also rejected.

The attached session performs 100 MSE updates against two private controls using
the existing strict quality thresholds. Vulkan copies all five live families for
bytewise comparison with HIP and checks every gap/reserved row/Vulkan buffer tail
against its sentinel, including immediately before and after attachment. The
owner allocation survives teardown. Only 263 Gaussians are visible at 35x29;
this remains an integration/storage/quality test, not a real-dataset benchmark.

### Real application Tensor integration

`HipTensor.CoreBackendPointwiseAndOwnership` uses the actual HIP GpuBackendOps
adapter and project CPU functors: 68 pointwise variants, all 17 binary broadcast
operators, NaN/infinity/signed zero, byte predicates, offsets, arbitrary read
strides and padded output, malformed layouts, reductions, fusion and explicit
unsupported-operation rejection. It checks queue-ordered temporary reuse, a
1 KiB cache bound, direct allocations, inactive pointer classification and trim
preserving live allocations.

The root project's three EXCLUDE_FROM_ALL targets additionally link real lfs_core:

- `lfs_hip_tensor_smoke`: factories, lazy fusion, clone, CPU transfers, sum,
  transpose/contiguous, matrix/vector broadcasting and in-place scalar dispatch;
  combined builds run CUDA control expressions before and after the HIP scope.
- `lfs_hip_training_admission`: the actual training registry rejects missing HIP
  families without initializing HIP or selecting another backend/default.
- `lfs_hip_tensor_training_smoke <absolute-training-module>`: native allocation
  validation, budget recovery, stale-generation/offset/reallocation rejection,
  optimizer poisoning propagated to the core Control, then 400 steps through
  TensorModel against two private sessions. It verifies all five parameter
  aliases bytewise, padding/reserved rows and owner lifetime. Tensor producers
  are enqueued before attachment to exercise the explicit queue handoff.

The integration fixture has six active entries among 263 logical rows, SH0,
35x29, fixed count and a test-only learning-rate schedule after step 200. Strict
image thresholds remain max < 1/255 and MSE < 1e-8. An earlier dense 263-active
fixture showed divergent long-run optimizer trajectories even between private
controls; it is documented as unresolved quality validation, not counted as a
passing equivalence test. These fixtures do not replace multiview dataset tests.

Test the combined build normally and with child-process environment
`CUDA_VISIBLE_DEVICES=-1`, `HIP_VISIBLE_DEVICES=0`. The first variable alone also
hides AMD devices from HIP. This verifies no visible CUDA devices, not absence
of the installed CUDA driver/runtime. No global environment changes are needed.

The core adapter test also checks FP32 sum/mean/min/max reductions against double
CPU oracles: extents 3, 32, 33, 263, 4096, 4097 and 100003; multiple inner/outer
sizes; long-axis partitioning; columns; nonadjacent and transposed axes; offset
and output guards; NaN/infinity/signed zero; empty sum/mean/extrema; duplicate and
out-of-range axes, overlap, unsupported kinds and overflow. Root Tensor smoke
additionally executes matrix column sum and row mean with keepdim.

Conversion/index coverage includes 49 dtype pairs (263-element tails, offsets,
output sentinels, NaN/Inf/signed zero, saturation boundaries, modulo and half
subnormals/ties), and 252 gather/select configurations: seven value dtypes, two
index dtypes, three axes, gather/select and three boundary modes. It checks
strided layouts, full Int64 extrema, duplicates, bit preservation including half,
invalid-index transactional rejection/recovery, overlap and descriptor errors.
The checked narrowing cast separately tests invalid indices and preserved output.

Root Tensor smoke also checks half/byte roundtrips against CPU, native Int64 row
selection, multidimensional and transposed-index gather, and full-width Int64
Clamp/Wrap. Combined builds execute existing CUDA gather/index_select controls
before and after the HIP scope. This is correctness validation, not timing.

Count tests compare Float32, Bool and UInt8 against CPU for 0, 1, 263, 1,048,595
and 16,777,219 elements, including all nonzero beyond FP32's exact integer range,
noncanonical nonzero bytes, NaN/Inf/signed zeros and rejected dtype/overflow.
Actual Tensor smoke checks float counts, predicate-generated masks and a
noncontiguous matrix view. These additions retain the 400-step storage controls.

Indexed-write coverage adds 108 copy/add configurations: copy across seven dtypes,
add across Float32/Int32, two index dtypes, three axes, unique and 263 repeated
indices. CPU oracles verify every byte of output, holes and sentinels. Duplicate
copy sources agree at each component; no deterministic write order is asserted.
Tests also cover invalid-index transactional preservation/recovery, empty writes,
overlap with source/indices, noninjective layouts, shape overflow and unsupported
add dtypes. Fifteen strided-write cases cover seven dtypes, normal/immediate
submission and Int32-to-Float32 rounding, plus malformed layouts.

Real Tensor smoke checks aliased source and index snapshots, repeated sums,
rank-one updates, invalid updates preserving values, and transposed-view
writeback. Combined builds run existing CUDA copy/add controls before/after HIP.

Random coverage checks the Philox zero-key/counter known-answer vector and exact
CPU mappings for uniform, Bernoulli and randint with two full-width seeds.
131,073 draws per distribution/seed check mean/variance, normal lag correlation
and integer-bin frequencies. Repeated seeds, 263-element prefixes, empty and
misaligned outputs, tails and sentinels are checked. Edge cases include zero
standard deviation with signed mean, constant uniform, Bernoulli 0/1, normal
mean/std affine transformation, widest finite uniform bounds, full signed Int32
range, invalid parameters and byte overflow. Random quality checks are synthetic,
not multiview training-quality validation.

Actual Tensor smoke checks normal seed reset/advance, explicit seed isolation,
normal transposed-view writeback and Bernoulli/randint factories. Combined builds
also check the existing CUDA odd-normal reset behavior before and after HIP.

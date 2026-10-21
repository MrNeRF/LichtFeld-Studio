# Opt-in native HIP training

The native modules provide HIP rasterization and Tensor/trainer adapters without
Torch/PyTorch dependencies. Build support remains opt-in with `LFS_TENSOR_HIP=ON`.
Explicit GPU preferences and the application trainer are integrated; training
admission checks the operations required by the selected configuration. HIP is
excluded from automatic backend candidacy. Original CUDA, Metal and Vulkan
kernel sources are unchanged.

The current branch is being aligned to upstream/dev. Native compilation and
historical integration results must be distinguished from validation on this new
base. POPSpa callbacks are not yet implemented or advertised on HIP; old checkpoint
reading and migration remain upstream responsibilities and must stay compatible.
See the [alignment record](../../../docs/docs/development/rocm-dev-alignment-2026-10-21.it.md),
[frozen historical results](../../../docs/docs/development/rocm-historical-handoff-2026-10-21.it.md)
and [AMD GSplat references and driver update](../../../docs/docs/development/rocm-gsplat-amd-references-2026-10-21.it.md).
The sections below retain the standalone session contracts and the implementation
history; they do not establish that every new upstream feature is supported.

In an application build, enable both `LFS_HIP_RUNTIME=ON` and
`LFS_HIP_TRAINING_RUNTIME=ON` (both default to OFF). Set `LFS_HIP_ROOT`, one explicit
`LFS_HIP_ARCHITECTURE` and, if needed, `LFS_HIP_GLM_INCLUDE_DIR`. Python 3 prepares
private HIP kernel sources. `cmake/HipRasterKernels.cmake` invokes the selected
ROCm Clang explicitly and does not replace the application's CXX/CUDA compiler.
Tests and the module use the same generated objects. Original CUDA files remain
unchanged. Linux compilation has not yet been validated; Windows gfx1150/wave32
is the tested configuration.

`training_api.h` contains only C-compatible descriptors and opaque handles.
The host loads the DLL/SO, checks `lfs_hip_train_abi()==1`, creates a session,
prepares a frame and calls `step`/`render`/`read_parameter`. Allocation and release
stay inside the module; no C++ container, HIP/CUDA header or stream crosses the ABI.
All calls on a session must be serialized except `set_cancel`, which is atomic.
Join callers, successfully destroy sessions and then unload the module. A failed
destroy retains the handle; it does not authorize unloading live GPU code.

Standalone fixed-count session contract:

- Fixed Gaussian count, float32 means, wxyz quaternions, log scales, SH0 and raw
  sigmoid opacity. Parameters occupy one allocation with 256-byte-aligned regions.
- SH0 activation `max(0, C0 * coefficient + 0.5)` and the corresponding clamp
  derivative. Persistent FP32 Adam moments; one fused update across base attributes.
- Optional SH1–SH3 with FP32 SH-rest in the native 32-primitive/float4 layout,
  explicit degree activation, canonical upload/readback and a coalesced physical
  parameter/moment update. Inactive bases preserve values and moments. Default
  SH0 does not allocate SH-rest state.
- Global pinhole camera with a rigid row-major world-to-camera matrix, linear
  RGB CHW target and black background. MSE remains the default; unmasked fused
  L1/SSIM is selectable, with configurable SSIM weight and valid padding.
  Frame dimensions may change.
- Native projection, int64 count scan, intersection emission, hipCUB radix sort,
  tile offsets, forward and backward on one nonblocking stream.

Not integrated yet: main-trainer TrainingOps, ingest/cache, GUI selection,
Masked/decoupled photometric paths, refinement/frozen masks/regularizers, checkpoint/optimizer-state
resume, mixed-device transfers and the application Vulkan viewer. Private sessions
use `hipMalloc`. Experimental shared sessions can now train directly in a dedicated
Vulkan Win32 allocation, but this is not yet connected to the application's model
storage or viewer.

The host supplies a hard byte budget for session buffers. This includes model,
gradients, moments, frame, binning and scratch allocations, plus temporary overlap
while replacing storage. It does not measure driver heaps, runtime overhead or
total UMA/VRAM consumption. `info` reports live and historical peak bytes.
`set_budget` may increase the limit, or reduce it as far as current live bytes.

Intersection capacity comes from the actual GPU count. It grows with bounded
headroom and the sort processes only live entries plus a sentinel. The module
never reserves `Gaussian count * image tile count`. A host count completion is
currently required each frame; optimizing this requires an overflow-safe GPU
protocol rather than a worst-case reservation.

Invalid input, budget exhaustion and cancellation before Adam preserve parameters
and optimizer step. Frame replacement is transactional: failed replacement keeps
the old frame. In-flight raster/gradient work is drained before a recoverable
error returns. Resource failures after Adam dispatch, nonfinite optimizer results
and GPU errors poison the session rather than silently switching backend. There
is no rollback of a partially executed optimizer kernel. Real device-loss and
driver-OOM recovery are not validated by the allocation-budget tests.

The standalone `tests/hip` project has `LFS_HIP_PROBE_TRAINING_SESSION=ON` for
MSVC-to-DLL correctness, budget, cancellation and concurrency checks, and the
separate `LFS_HIP_PROBE_LARGE_SESSION=ON` memory test. The latter allocates a million
model entries with only 263 visible at 35x29; it is not a dataset throughput test.
When CUDA coexistence testing is enabled, a live NVIDIA allocation is verified
after session destruction and unloading of the HIP training DLL.

`session.hpp` / `session.cpp` provide the host-compiler adapter, built as
`lfs_hip_training_host` when the parent project enables C++. It loads an explicit
absolute module path, verifies the private ABI, and resolves every required entry
point before creating a session. No HIP/CUDA headers or stream casts enter the
host adapter. Errors retain the training status and native code in `TrainingError`;
there is no implicit backend fallback.

Operations on a session serialize, while cancellation can run concurrently.
Explicit `close()` reports teardown failure and permits retry; repeated successful
close is harmless and subsequent operations are rejected. If destructor teardown
fails, it logs to stderr and deliberately retains the module's OS reference with
the live session. This avoids unloading code with potentially live GPU work;
actual device-loss fault injection remains untested. Callers must join threads
before destroying the C++ object. The adapter does not cache or duplicate model
arrays. The optional TensorModel bridge described below connects native Tensor
storage; the main trainer remains unconnected.

`HipTraining.HostAdapterLifetimeAndRecovery` exercises this exact host source via
MSVC, including frame replacement budget failure, cancellation/resume, concurrent
step/info calls, model readback, convergence and teardown, while CUDA memory stays
live. Linux loading code exists but has not been compiled or run on Linux yet.

### Experimental shared model storage (Windows)

`model_layout` queries aligned offsets and sizes for the five parameter families
before allocating Vulkan storage. The shared-session constructor imports a dedicated
`OPAQUE_WIN32` allocation bound at offset zero, with no private allocation for the
five base parameter families. Optional SH-rest remains private HIP storage; sharing
the complete higher-degree model with the application renderer is not implemented.
Gradients, moments and workspace remain private HIP storage. The full imported
allocation is charged once to the session budget, including allocation padding.
These counters do not measure the driver's total UMA footprint.

The descriptor carries Vulkan's device LUID and node mask. The HIP module checks
them against the selected HIP adapter: direct LUID matching when available, or
Windows LUID-to-PCI resolution when HIP omits the LUID. Ambiguous/mismatched identities
fail; there is no vendor/name match or private-allocation fallback. The caller must
provide the actual allocation owner's identity; the descriptor cannot independently
prove the origin of an arbitrary NT handle.

The Vulkan owner retains the NT handle through creation and retains the allocation
through session destruction. It releases queue ownership to EXTERNAL and waits for
Vulkan completion before a HIP operation, and only reacquires after the synchronous
HIP operation completes. No concurrent rendering/training access or asynchronous
semaphore protocol is provided. The beta GPU memory-wait probe is not used here.
External mapping/import cleanup is inside the HIP module. Driver/device-loss cleanup
failures have not been fault-injected; normal teardown and allocation-budget cleanup
are tested. Shared import on Linux is explicitly rejected pending implementation.

`HipTraining.VulkanSharedModelOneMillion` compares 100 training steps with two private
HIP controls, verifies Vulkan GPU readback of all trained families byte-for-byte,
checks a trailing guard and preservation of the Vulkan allocation after session
destruction. It checks a wrong physical GPU, invalid identity/size and budget failure.
One million model entries have only 263 visible at 35x29. Atomic gradient accumulation
is not bitwise deterministic even for private/private runs; quality checks include
first-update agreement and explicit parameter/image/target-MSE tolerances. This is
a storage and numerical comparison, not a real-dataset throughput benchmark.

### Unmasked photometric loss

`set_loss` validates the complete configuration before changing state. MSE keeps
its existing kernel and default behavior. Fused L1/SSIM uses kernels generated
directly from this checkout's `src/training/kernels/ssim.cu`, without changing the
CUDA source, formulas, Gaussian coefficients, convolution order or FP16 partials.
`prepare_photometric.py` extracts only the unmasked forward/backward functions and
their helpers; the generated file is private HIP build output.

Valid padding matches the existing fused path: each dimension larger than ten
loses five center pixels at either end. Smaller dimensions keep all centers and
retain zero padding outside the image. L1 is normalized over those centers as
well. Weight zero skips SSIM kernels/workspace entirely and uses a direct L1
gradient with zero subgradient at exact equality. Positive weights allocate
22 bytes per RGB pixel (three FP16 partial arrays and one FP32 channel-mean map),
within the session budget. Workspace is lazy, reused for the frame shape and
released when returning to pure L1 or MSE. Partial-allocation failures unwind
without changing the model or advancing Adam.

`loss()` evaluates on GPU, reusing the image-gradient allocation for reduction,
and reads back a scalar instead of downloading the full image. It does not
advance Adam or modify parameters/moments. This diagnostic still runs the normal
raster path, including its existing host count completion; it is not a timing
shortcut or a per-step profiling synchronization added to training.

Tests cover an independent double-precision 2D CPU value/gradient oracle,
finite differences, mixed gradient signs and exact L1 ties, odd shapes, partial
tiles, padding, scalar reductions, invalid configurations, complete/partial
workspace budget failures, resizing and convergence for L1, 0.2 SSIM and pure
SSIM. The shared-model photometric test uses an identical decaying rate schedule
for shared and two private controls. L1 sign changes make independent trajectories
less reproducible than MSE; the new test checks objective/image quality and keeps
the original MSE test's stricter bounds unchanged. Neither is a dataset benchmark.

### SH1–SH3 session state

`set_sh` accepts canonical RGB SH-rest coefficients and a maximum/active degree.
It validates input and allocates replacement state transactionally, including the
temporary canonical upload in the byte budget. Parameters and FP32 moments use
the native 32-primitive/float4-swizzled layout; the update traverses physical storage
for coalesced parameter/moment accesses and gathers canonical raster gradients.
Only one persistent SH-rest parameter copy is retained. Readback reuses gradient
scratch to gather canonical coefficients without another persistent model copy.

No SH-rest state is allocated by default. Initial configuration may be deferred
until the first nonzero SH degree, even after SH0 training has started. Replacing
an existing learned store is rejected after the first optimizer update; before
training it may be replaced or cleared. `set_sh_degree` changes active degree/rate
without reallocating. Inactive bases preserve both parameters and moments. Adam
bias correction uses the session's global step, including on first activation.
Checkpoint/optimizer-state restoration and SH-storage growth are still absent.

Forward/backward come from the repository's native gsplat SH kernels. The session
detaches viewing-direction gradients, matching the existing gsplat training caller;
the kernel's optional direction derivative is separately validated. Visibility
masks exclude culled/zero-direction Gaussians before normalization. The existing
lower color clamp is applied before rasterization and differentiated once. The
default SH0 forward remains the original path.

The shared-model API still imports only the five base families. SH-rest is private
HIP storage, not yet an exportable SH-rest region for the application renderer.
No Q16/FP16 SH-rest or compressed moments are implemented. The memory tests report
owned buffers, not dedicated VRAM or total UMA residency.

### Parameter and optimizer layouts

Base parameters retain the public 256-byte-aligned export layout. Gradients and
FP32 Adam moments use an independent private layout with 16-byte region alignment.
Backward kernels use optimizer offsets; Adam resolves parameter and optimizer
offsets separately. No exported offsets or C ABI have changed. This is preparation
for accepting application-owned layouts with reserved capacity and different
region ordering; the explicit-offset import and core layout adapter are described below. In particular,
the main trainer, immutable viewer leases and model growth are still
not connected to the HIP session.

The 263-entry session and shared-model cases exercise different parameter and
optimizer offsets, including higher SH, loss, cancellation, budget recovery and
Vulkan alias checks. This change makes no real-dataset throughput claim.

### External layouts and resident attachment

`lfs_hip_train_create_shared_layout` accepts explicit 256-byte-aligned offsets
and live counts for the five base families. Region order may differ from the
module's default; leading padding and reserved rows are preserved. Descriptor
validation checks counts, alignment, overflow, bounds and overlap before import.
Adam's private layout remains independent of the imported extent.

`lfs_hip_train_attach_shared_layout` imports already initialized parameters and
never uploads or reads back the model. A GPU validation kernel checks all live
values, quaternion norms and activated scales, reading back a single status.
Invalid contents reject attachment without modifying the allocation. This starts
fresh optimizer moments/step zero; it is not checkpoint optimizer restoration.
Both paths retain the existing explicit same-adapter identity and external queue
ownership protocol. The allocation owner must outlive the session.

The host-only `splat_storage_layout.hpp` adapter reads the actual core
`SplatExportableStorage::Control` region offsets/capacities. It currently admits
healthy, fixed SH0 storage and rejects exported CUDA VMM chunks. It neither
exports handles nor proves that a caller-supplied handle aliases that control;
the application integration must provide the matching dedicated Vulkan allocation,
retain ownership and prevent concurrent growth/generation changes. It is not yet
called by the main trainer; the layout adapter itself does not register Tensor
operations. The optional native Tensor adapter is described below.

### Native application Tensor seam (opt-in)

`LFS_TENSOR_HIP=ON` requires `LFS_HIP_RUNTIME=ON`; both default to OFF. It adds
an explicit HIP enum/registry entry and a host-compiled `GpuBackendOps` adapter
loading the isolated module. It admits Float32 pointwise/fused chains,
Float32/bool broadcasting, strided materialization/fill, copies, factories and
scalar reductions and FP32 sum/mean/min/max reductions by axes, columns or
strided segments. Other methods fail explicitly. HIP remains absent from the
compiled/default candidate list; an explicit `GpuBackendScope(HIP)` is required.
The real TrainingOps registry reports missing families instead of falling back.
Only the native default queue is accepted; a CUDA stream is never reinterpreted.
The module checks the selected device architecture against its compiled target.

Temporary allocations use an exact-size, bounded cache (64 MiB by default).
Releasing an eligible temporary queues no synchronization; reuse follows earlier
consumers on the same queue. Active and reserved bytes are reported separately.
Direct allocations bypass the cache. Trim waits on the queue and releases only
cached buffers; native out-of-memory drains optional cached buffers and retries
once. The failure path is implemented but driver OOM is not fault-injected.

`TensorModel` connects an actual native HIP byte Tensor plus its core Control to
`lfs_hip_train_attach_device_layout`. The native descriptor validates allocation
base/range/type/device with HIP. The host retains a shallow allocation snapshot,
checks generation, offsets, capacity and allocation identity before every call,
and waits for Tensor producers before entering the session's separate queue.
Session operations complete before returning. GPU/optimizer poisoning invalidates
the core Control. Callers still serialize mutations; concurrent viewer leases,
refinement, higher-SH sharing and checkpoint resume remain unimplemented.
A failed session teardown retains its module and model owner rather than freeing
an allocation that may still be in use.

The root project exposes EXCLUDE_FROM_ALL targets `lfs_hip_tensor_smoke`,
`lfs_hip_tensor_training_smoke`, and `lfs_hip_training_admission` when opted in.
They link the actual lfs_core DLL; standalone tests cover the adapter separately.
The storage test is a sparse synthetic 400-step case, not dataset throughput or
an assertion that HIP beats Vulkan. See the study document for results and the
dense-scene diagnostic limitation.

Axis reductions keep their output on the GPU. Up to 32 reduced elements use one
lane per output; larger axes use workgroups. Long axes with few outputs are split
into at most 4096 partial floats and combined by a second kernel. They reuse the
same fixed 16,388-byte context workspace as scalar reductions. Output shape,
axes, strides, overflow, bounds and input/output disjointness are validated before
launch; empty sum/mean are zero/NaN and empty extrema are rejected. Product,
variance, arg-extrema and other reduction kinds remain explicitly unsupported.
No throughput advantage is inferred from these correctness tests.

### Native conversions and indexing

The native adapter implements all 49 source/destination dtype combinations for
Float32, Float16, Int32, Int64, UInt8, Bool and UInt32. Float-to-integer casts
truncate and saturate with NaN-to-zero, byte casts truncate/modulo with nonfinite
to zero, bool casts test nonzero, and half conversion rounds to nearest even.
Identical-dtype copies preserve bits. Overlapping conversion ranges are rejected
except an identical-dtype exact alias. This does not enable arithmetic on every
dtype; unported integer/half arithmetic still fails explicitly.

Gather and index_select support all value widths, strided source/index layouts,
Int32/Int64 indices, repeated reads, and Assert/Clamp/Wrap boundaries. Assert
validates all indices on the GPU and reads one four-byte status before writing
output; rejection preserves every output byte. Clamp/wrap stay asynchronous.
Outputs must be disjoint from both inputs. Native indices are retained in the
application HIP gather/select path, preserving full Int64 values and strides.
Existing CUDA/Metal/Vulkan paths retain their previous index conversion behavior.
The standalone checked Int64-to-Int32 cast additionally rejects extents beyond
Int32 addressing; it never silently narrows an invalid index.

These additions are not scatter, refinement, model growth, or main-trainer
integration. They retain the same opt-in and admission restrictions.

Native `count_nonzero` for Float32 and Bool/UInt8 uses integer 64-bit partials and
one atomic contribution per workgroup. NaN/Inf count as nonzero and signed zeros
do not. Only an eight-byte scalar returns to the host. It reuses the fixed
context workspace and does not allocate temporary storage proportional to input
size. This completes counting, not masked compaction or pruning integration.

### Indexed writes and noncontiguous view updates

Native index_copy accepts all value widths and Int32/Int64 vector indices;
index_add accepts Float32/Int32 with atomic accumulation for repeated indices.
Source values are dense, destination layouts may have validated nonoverlapping
strides. Bounds, sizes, alignment, dtype and input/output disjointness are checked
before GPU mutation. All indices are validated first, reading one four-byte
status; an invalid index preserves the entire destination. Empty updates do no
work. They reuse the context workspace without model-sized auxiliary buffers.
Duplicate index_copy writes have unspecified ordering; differing duplicate
sources are not promised a deterministic last-writer result. Float accumulation
order is likewise not deterministic for overlapping contributions.

The application HIP index_copy_/index_add_ paths retain native Int64 indices and
snapshot a source or index Tensor sharing the destination's storage. Other
backends retain their existing paths. Frontend dtype restrictions remain the
existing ones even where the private native copy supports extra widths.

Native strided_scatter and its immediate variant pass metadata by value and
preserve element bits. A separate Int32-to-Float32 variant implements the existing
converted-write contract. Layout overlap, byte overflow, range and alignment are
validated; these operations do not support overlapping source/destination.
They complete writeback through the existing mutate_logical_view path, including
indexed updates of transposed views. Immediate describes metadata submission,
not a host synchronization. This is not general indexed scatter or refinement.

### Native random generation

Uniform, Bernoulli, Int32 randint and Float32 normal use a stateless
Philox4x32-10 generator with the same round constants as the project's shader.
Each counter block supplies four draws. Aligned full groups use four-value
stores; tails and four-byte-aligned views use bounded scalar stores. Prefixes
are stable across output sizes for a given distribution, seed and parameters.
This HIP sequence is not promised bit-identical to CUDA/Metal/Vulkan sequences.
No existing backend RNG implementation or global seed API is changed.

Uniform uses [low, high), Bernoulli emits 0/1, and randint uses the existing
multiply-high mapping over the full signed Int32 interval. Normal uses two
Box-Muller pairs per block. Constant distributions use fill. Invalid parameters,
ranges, alignment and byte overflow reject before writes. All draws stay on the
native queue with no per-element state allocation or host readback. Odd normals
need no scratch or extra synchronization; only the HIP frontend skips the odd
scratch allocation. Multinomial remains unsupported and is not advertised.


### Experimental real-dataset runner

With `LFS_TENSOR_HIP=ON`, explicitly build the excluded target
`lfs_hip_dataset_train`. It links the actual application I/O module and accepts:

```
lfs_hip_dataset_train MODULE DATASET NEW_OUTPUT STEPS MAX_WIDTH VIEWS SEED
```

`MODULE` is the absolute native HIP training module. `SEED` can be an SH0
Gaussian PLY, a `.licht` project with a bound training checkpoint, or the literal
`points`. Project seeds restore model parameters only, with fresh HIP Adam;
this is not optimizer/checkpoint resume. Frozen/deleted rows and higher SH are
rejected. `points` uses the sparse reconstruction with explicitly fixed initial
isotropic scale 0.01 world units, opacity 0.1 and SfM colors. This initialization
is diagnostic and is not the application's adaptive initialization.

The runner keeps one model in the actual HIP Tensor storage, attached through
`TensorModel`; the native session does not allocate a duplicate parameter model.
Decode uses the existing Media Ingest CPU codec. Distorted images are decoded
at native resolution into RGB bytes in RAM. The native HIP forward area/Lanczos
warp performs one combined resize/undistort, followed by the portable ingest
16-bit output grid on the returned CPU target. Only one native byte source is
uploaded at a time, transiently; it is never expanded into a full-resolution
float GPU image. After readback the runner verifies that active core HIP bytes
return to their pre-upload value. A source is limited to 256 MiB of RGB bytes
(two transient host byte buffers and one transient device byte buffer); the
prepared-target budget is separate. This is a bounded experimental
subset cache, not PipelinedImageLoader or its optional prepared JPEG blob cache.
HIP asynchronous Tensor uploads and SharedImageOps remain prerequisites for
connecting the full loader; unsupported paths still reject explicitly.
Prepared float RGB CHW targets stay in an explicit 512 MiB CPU subset cache;
`VIEWS` chooses evenly spaced cameras in application loader order. These views
are shuffled each epoch with seed 42. The native session budget is 2 GiB.
Neither budget is a process-wide or driver-wide memory limit. The image-prep
phase precedes timed training. Frame transfer and native step are recorded
separately; evaluation/readback and export are excluded from step times.

The configuration is fixed topology, SH0, black background, L1+SSIM (weight
0.2, valid padding), no refinement or extra losses. Means learning rate decays
exponentially from 0.00016 to approximately 0.0000016 over the requested steps;
quaternion/log-scale/SH0/raw-opacity rates are 0.001/0.005/0.0025/0.05.
These timings cannot be compared directly with MCMC/Fast training benchmarks.

A new output directory is required. Outputs include `initial.ply`, `trained.ply`,
`steps.csv`, `preparation.csv`, `summary.json` and before/after/target PPM images for every selected
view. Final PLY parameters are reloaded through the application loader and all
four non-rotation parameter families must compare exactly. Quaternions must
match unit-normalized originals within 3e-7, following the application's existing
PLY exporter contract. PLY exports, including `initial.ply`, normalize quaternion
norms; use the original project seed for exact optimizer initialization.
Reported PSNR/MSE use the training views, not a holdout set. Ctrl+C is observed between steps; a healthy model is
exported with an interrupted summary. Errors log explicitly and never switch
backend. The main application Trainer and GPU selectors remain unavailable for
HIP until the required TrainingOps families are implemented.

The explicit CPU reference uses the existing area/Lanczos kernel on up to eight
workers, with unchanged global coordinates and accumulation order per pixel.
The actual-core test compares every float against full-frame CPU resampling
for float inputs and for native byte inputs on a reduced destination grid.
`preparation.csv` separates calibration, decode/resize, planar layout, combined
warp and target ownership. These costs occur once before training. Rebinding
an identical immutable CameraFrame is skipped in the runner, including a
single-view run; the public borrowed-pointer set_frame API still validates and
uploads every call, so in-place caller changes are never silently cached.
The loader runs in a one-thread arena so the explicit experimental HIP scope
also applies to camera construction; no process backend preference is changed.
Native pinhole inverse rays are used for camera calibration. Unsupported GPU
image warps fail explicitly instead of entering the generic recursive fallback.

Set `LFS_HIP_TRAIN_PROFILE=1` **only in the child experiment's environment** to
print phase diagnostics. This allocates HIP timing events and deliberately
synchronizes phase boundaries; it is disabled by default. These diagnostic
phase times include dispatch/runtime gaps and must not be used as a regular
unprofiled throughput benchmark. In particular, unfenced event attribution was
ambiguous on the tested runtime; only synchronized phase reports are retained
as the basis for optimization decisions.

The initial real one-million-Gaussian/1920 test took about one second per
step in this 3DUT prototype. HIP-only dual-pixel staging now uses 128 splats
instead of 256 (12 KiB rather than 24 KiB LDS), with staging bounds adapted
together. A matched 30-step/two-view comparison without profiling measured
1038.564 -> 791.448 ms per step after five warmup steps. Training-view MSE was
0.0528338158365 vs 0.0528338372662; a repeat of the old path gives
0.0528337684407. Floating-point atomic ordering is not bitwise deterministic.
Native HIP preparation of the two native 6048x8064 images to 1440x1920 takes
about 1.45 seconds total, compared with 36.57 seconds on the CPU reference.
This improves startup, not per-step raster throughput. A 200-step comparison
kept training-view MSE coherent (CPU 0.0170283651, HIP 0.0170281178), with
about 700 ms per step in both runs after ten warmup steps.
The synchronized diagnostic attributes most training time to raster backward,
then forward and sorting. The path is not yet a replacement
for the application's faster Fast/3DGS trainer. Training a small fixed model
successfully does not establish quality parity or production-scale performance.


### Native image preparation controls

`LFS_HIP_CPU_IMAGE_PREP=1` selects the CPU reference explicitly. The default
uses the HIP pinhole RGB warp through real Tensor image operations, not a CPU
fallback inside the backend. Input is contiguous UInt8/F32 CHW; output is F32.
The private C ABI validates dimensions, ownership, alignment, overlap,
calibration and bounded work before queuing. Identity uses a grid-stride copy
or byte normalization, including inputs above the launch-grid cap. Forward
sampling reuses the existing CUDA formulas in a private generated copy, with
explicit projection FMA and output-row limits. Nonrepresentable projected
coordinates are skipped before any float-to-int conversion. Original CUDA,
Metal and Vulkan kernels remain unchanged.

The native footprint is limited to 64, with at most 2,097,152 quadrature samples
per output-row batch. Requests outside those work limits use the CPU reference
at the runner boundary, before any native image allocation, with an explicit
CLI warning and per-view/summary counters. This is image preparation only;
training stays HIP. Invalid calibration, ownership and actual GPU errors are
not hidden by that fallback. The native API itself rejects out-of-bounds work.

`LFS_HIP_VALIDATE_IMAGE_PREP=1` additionally computes the independent CPU reference
for each GPU-prepared image and compares every float before 16-bit quantization.
This diagnostic deliberately includes CPU reference cost in preparation time;
do not use it to measure normal loading speed. `preparation.csv` includes source
upload, warp/readback, live/reserved core HIP bytes and reference errors. Those
pool counters exclude the training module's own allocations and must not be
summed as if all logical model accounting represented separate physical memory.
A 64 MiB generic free-buffer pool may retain a reduced output buffer; it cannot
retain the tested 139.5 MiB native source.

The full PipelinedImageLoader cache/prefetch protocol, asynchronous upload queues,
SharedImageOps registration and main Trainer remain separate integration work.
The runner does not claim to implement those by providing the forward warp.

### Explicit FastGS and resident SH bring-up

`LFS_HIP_FAST_RUNTIME` builds the private wave32 FastGS implementation. It is off
by default. The dataset runner selects it only with `LFS_HIP_FAST_RASTER=1` and
logs that choice. The previous world-ray path remains the default. Experimental
kernel switches live under `LFS_HIP_FAST_*`; benchmark their exact configuration,
since their gains are not additive and some variants are slower.

FP32 model parameters and the FP32 Adam reference remain available. The explicit
`LFS_HIP_FUSED_ADAM=1` mode compresses optimizer moments and fuses updates; it is
not a switch to FP16 model parameters. Cancellation happens before mutating a
fused step. Failures after mutation poison the session instead of silently
retrying a partially applied update. This mode currently primarily saves memory.

`Session::attach_resident_sh` borrows FP32 float4-swizzled SH1–SH3 inside the same
allocation as the base parameters. The full allocation is accounted once; no
second SH value buffer is allocated. It validates degrees, alignment, ranges,
overlap and finite logical coefficients before replacing state. Padding is not
treated as coefficients. `TensorModel` additionally requires an explicit Float32
SH Tensor view, retains its allocation, and validates source/control generation
and both Tensor views before work. Compressed SH and topology growth are not
accepted by this boundary. Standalone tests cover FP32/fused updates, alias
readback, padding, invalid replacement and ownership through close.

This still does not admit HIP to the main Trainer: complete Tensor operations,
growth strategies, checkpoints and asynchronous loader/viewer integration remain
required before automatic backend selection can be enabled.

### Native prefix scans

The HIP Tensor adapter now implements inclusive `cumsum` for the public Float32
and Int32 contract, including arbitrary axes. The runtime additionally accepts
Int64/UInt32 without changing the public contract of existing backends. Dense
vectors use hipCUB DeviceScan and a reusable, accounted context workspace;
multiple/strided segments use hipCUB BlockScan with explicit carry between
256-element tiles. Writable layouts must not overlap. Ownership, byte spans,
alignment, axis and type are checked before any launch. Empty views are no-ops;
integer overflow uses defined unsigned modulo accumulation. FP32 parallel scans
can differ from sequential double accumulation by rounding.

This is infrastructure for sampling/selection. It does not enable the growth
strategy or alter the fixed-topology training benchmark above.

### Native stable sorting and workspace reclamation

Float32 Tensor sorting now returns Int64 indices relative to the selected axis.
Dense vectors use hipCUB radix sort; other axes pack independent segments, sort
them with segmented radix sort, then restore the original layout. Ascending and
descending order preserve the relative order of equal finite values. Public
Tensor cloning keeps the source unchanged. Arbitrary NaN payload ordering is
not a portable cross-backend guarantee.

All views, sizes, directions, alignment and value/index overlap are checked
before allocation or mutation. Scratch grows only when needed, remains private
to the context queue, and is included in current/peak memory accounting. Explicit
`trim` and the existing unused-reservation threshold release scan/sort scratch
after its queued users finish. The small fixed reduction workspace retains its
previous lifetime. Tests cover sort indices/stability on every axis, more than
one million values, invalid descriptors, output guard regions, and repeated
scan/sort/reduce/trim cycles followed by workspace recreation.


### Verifica locale 19 ottobre: selezione, copie e specializzazione RGB

Sono ora native HIP anche masked_select, nonzero/nonzero_bool e cat/stack (60 metodi Tensor ancora non implementati). La selezione usa hipCUB DeviceSelect e trasferisce alla CPU solo il conteggio; capacità insufficiente non modifica la destinazione. Le copie concatenate validano tutta la lista prima di lanciare il primo trasferimento. Le viste HIP vuote ammettono operazioni di zero byte senza allocazione, mantenendo il rifiuto di puntatori estranei, offset invalidi e copie non vuote su puntatore nullo.

La verifica di finitezza dei target CPU mantiene il controllo IEEE di ogni elemento, ma usa una riduzione sui bit che il compilatore vettorizza. Nel confronto 30 step la fase frame media scende da 8,1603 a 4,0992 ms. Questa media include i riusi senza caricamento: non misura il tempo del singolo upload cambiato. Test aggiuntivi coprono zeri con segno, subnormali, estremi finiti, Inf e NaN, anche modificando lo stesso buffer CPU tra chiamate.

`LFS_HIP_FAST_RGB_SPECIALIZATION` è opt-in e specializza il backward privato quando depth ed edge sono assenti; normal e statistiche rimangono indipendenti. Non modifica i kernel originali CUDA/Metal/Vulkan. Le combinazioni analitiche raster sono passate da 20 a 72.

| Confronto ripetuto ABBA | Generale, ms/step | RGB, ms/step | Riduzione |
| --- | ---: | ---: | ---: |
| 30 step, scartati i primi 5 | 142,026 | 133,941 | 5,69% |
| 200 step, scartati i primi 10 | 122,370 | 117,871 | 3,68% |

Protocollo: due viste, lato massimo 1920, un milione di gaussiane effettive, SH0, topologia fissa, ottimizzatore fused, quattro esecuzioni per durata. Memoria live circa 713,59 MB e picco 877,03 MB in entrambe le varianti. Il primo confronto lungo ha PSNR 17,681687 contro 17,681559 dB. Le immagini sono state confrontate visivamente; la differenza media è circa 0,04–0,05 livelli su 255, dello stesso ordine delle ripetizioni della stessa variante. Non è parità bit a bit, né una verifica del training completo con crescita o viste holdout. Non confrontare direttamente questi tempi con la baseline Vulkan storica, che usa un protocollo diverso.

Scartato un esperimento con due shard dei gradienti: circa 150,35 ms/step nel run breve e 64 MB aggiuntivi. I sorgenti sperimentali sono stati ripristinati dal manifest prima dell'esperimento.

Verifica finale della configurazione RGB: 25/25 test nativi; 9/9 matrice Tensor (HIP, CUDA+HIP, CUDA nascosta). Artefatti in `.local/rocm-tensor-study`: `rgb-final-native-suite.log`, `selection-final-core-matrix.json`, `rgb-specialization-repeated-200.json`, `sdk-sources/audit/rgb-quality-repeat-context.json`. Le opzioni restano disattivate per default; l'ammissione al Trainer principale resta chiusa.


### Tuning rocPRIM nel training, confronto ripetuto da 200 step

L'opzione `LFS_HIP_FAST_RADIX_TUNING` (default OFF, ammessa solo con target gfx1150) usa configurazioni pubbliche rocPRIM senza patch dell'SDK: onesweep a 8 bit, ranking match, histogram 256x12, sort 256x16, soglia merge zero. Il wrapper privato conserva i DoubleBuffer e aggiorna i selettori soltanto dopo successo; query ed esecuzione usano la stessa configurazione. Le altre architetture e i sorgenti originali restano invariati.

Con RGB e le altre opzioni già attive in entrambe le varianti, confronto ABBA con un milione di gaussiane, due viste a 1920, SH0, fused, 200 step, esclusi i primi 10:

| Variante | Run A, ms/step | Run B, ms/step | Media |
| --- | ---: | ---: | ---: |
| Sort di riferimento | 119,672 | 121,309 | 120,490 |
| Sort configurato | 115,103 | 115,395 | 115,249 |

Riduzione media dell'iterazione: 4,35%. Non sommare percentuali di confronti eseguiti in momenti diversi. Memoria live circa 713,59 contro 714,91 MB, picco 877,03 contro 879,67 MB: il miglioramento comporta circa 1,32 MB live e 2,64 MB di picco aggiuntivi.

MSE finale riferimento 0,0170542657 / 0,0170538768; configurato 0,0170541306 / 0,0170543009. Confronto visivo target/riferimento/configurato/differenza x8 senza degradazioni evidenti; differenza media per pixel RGB tra le prime coppie 0,0417 e 0,0491 livelli su 255, massimo 10/25. Differenze dello stesso ordine delle ripetizioni della stessa implementazione, senza promessa di identità bit a bit. Rimangono i limiti del test a due viste e topologia fissa.

Artefatti: `sdk-sources/audit/radix-training-repeated-200.json`, `radix-training-200-analysis.json`, `radix-training-200-quality.png`. Cinque test mirati di matematica/raster/convergenza/SH superati prima del benchmark.

Verifica finale anche con radix tuning attivo: 25/25 test nativi superati (`sdk-sources/audit/radix-final-native-suite.log`). Generatore idempotente per byte e mtime su 22 file, identici ai sorgenti compilati (`radix-generator-validation.json`). I dati live/picco sopra sono contabilità della sessione, non una misura della VRAM fisica residente. Nessuna modifica ai sorgenti originali FastGS/CUDA e ai kernel condivisi rispetto al checkpoint precedente.


### 19 ottobre: confronto SDK e riserva temporanea limitata

La variante pubblica `rocprim::warp_reduce<float,32,true>` è stata confrontata con la riduzione device-library dopo la fine delle attività concorrenti: media ABBA 122,912 contro 120,767 ms/step, senza vantaggi. Ripristinati esattamente i tre sorgenti dal manifest SHA256 dell'esperimento; conservati i test matematici aggiuntivi che verificano ogni lane con contributi sparsi e negativi. Non è stato modificato il driver e non si adotta questa variante SDK. Evidenze: `sdk-sources/audit/sdk-reduce-clean-training-repeated-200.json` e `sdk-reduce-clean-decision.json`. Il precedente confronto interrotto resta escluso.

Nel runtime privato FastGS, il riuso dei buffer temporanei usava capacità esatta: incrementi piccoli del massimo numero di intersezioni comportavano una nuova allocazione e la liberazione del vecchio buffer grande. La nuova prova `LFS_HIP_FAST_ARENA_GROWTH=1`, disattivata per default, aggiunge headroom soltanto quando un buffer già esistente deve crescere, per richieste di almeno 1 MiB. Riserva massima min(6,25% della richiesta, 8 MiB), soltanto se il budget ammette anche il vecchio buffer ancora vivo. Se non c'è margine contabile sufficiente, alloca la dimensione esatta precedente; non altera il budget e non prealloca un heap massimo. Un fallimento reale hipMalloc continua a propagarsi secondo il contratto esistente: non è una nuova garanzia di successo in condizioni di pressione sulla memoria fisica.

Confronto ABBA, stesso binario e stesso protocollo da 200 step, due viste 1920, un milione di gaussiane, SH0/fused, scartati i primi 10:

| Variante | Run A, ms/step | Run B, ms/step | Media |
| --- | ---: | ---: | ---: |
| Capacità esatta | 122,188 | 121,104 | 121,646 |
| Riserva limitata | 111,522 | 112,646 | 112,084 |

Riduzione media 7,86%. Iterazioni >150 ms: 13+13 nei riferimenti e 0+0 con riserva; p95 166,280/153,934 contro 124,103/133,373 ms. Contabilità sessione live 714,91 contro 727,88 MB (+12,97 MB), picco 879,67 contro 889,79 MB (+10,12 MB). Non sono misure della VRAM fisica residente. Non sommare questi guadagni alle percentuali di esperimenti precedenti condotti in momenti diversi.

MSE finale riferimento 0,0170544695 e 0,0170538663; riserva 0,0170542817 e 0,0170542960. Controllo visivo target/riferimento/riserva/differenza x8 senza degradazioni evidenti; differenza media RGB 0,0434/0,0483 livelli su 255, massimi 13/11. Non è parità bit a bit e non verifica crescita della topologia, holdout o il Trainer principale.

Artefatti: `sdk-sources/audit/arena-growth-training-repeated-200.json`, `arena-growth-analysis.json`, `arena-growth-quality.png`, `arena-growth-targeted-tests.log` (5/5). CUDA, Metal e Vulkan originali non modificati; HIP resta opt-in e non ammesso al Trainer principale.

Verifica finale con riserva attiva: 25/25 test nativi passati (`sdk-sources/audit/arena-growth-final-native-suite.log`). Verificata anche la rimozione completa della prova SDK: tutti i 22 sorgenti generati sono byte per byte uguali al riferimento precedente (`sdk-reduce-removal-parity.json`). Nessun benchmark GPU è rimasto in esecuzione.


### 19 ottobre: riuso dell'adiunto colore nel backward

`LFS_HIP_FAST_FACTOR_ADJOINT` (default OFF) riusa `dot(color, grad_c) + depth * grad_d + dot(normal, grad_n)` nel calcolo di dL/dalpha e nella ricorrenza della trasmittanza. La derivata è algebricamente equivalente, ma cambia l'ordine delle operazioni FP32: non si dichiara identità bit a bit. Nessun fast-math globale, quantizzazione, cambio di layout o memoria aggiuntiva. Modifiche solo nelle copie HIP private; tutti gli altri backend conservano i sorgenti originali.

Con riserva temporanea, RGB e radix già attivi in entrambe le varianti, confronto ABBA da 200 step per run, scartati i primi 10, un milione di gaussiane reali, due viste a 1920, SH0/fused:

| Variante | Run A ms/step | Run B ms/step | Media |
| --- | ---: | ---: | ---: |
| Ricorrenza precedente | 110,040 | 109,269 | 109,655 |
| Adiunto riusato | 106,606 | 105,052 | 105,829 |

Guadagno medio 3,49%; memoria sessione live circa 727,885 MB e picco 889,786 MB invariati. Il codice macchina del kernel RGB None/cull0/senza canali ausiliari contiene 1.398 istruzioni statiche contro 1.426: questa è dimensione del codice, non un conteggio delle istruzioni dinamiche eseguite (`factor-adjoint-isa-analysis.json`, disassemblati conservati).

MSE finali precedenti 0,0170542327 / 0,0170545959, nuovi 0,0170547098 / 0,0170542654. Controllo visivo delle immagini senza degradazioni evidenti. Differenza media RGB 0,0417 e 0,0544 livelli su 255; ripetizioni della stessa variante circa 0,041–0,055. Le differenze puntuali massime non sono usate come prova di equivalenza (fino a 44 nella coppia, 50 fra ripetizioni). I confronti restano limitati a due viste, SH0 e topologia fissa; non provano qualità su holdout o Trainer completo.

Quattro alternative sono state scartate dopo altrettanti confronti ABBA da 200 step: reciproco nativo raffinato con controllo del dominio (112,711 -> 131,902 ms), reciproco nel solo dominio già garantito dal raster (114,949 -> 131,265 ms), riuso del limite dell'ellisse e conica normalizzata (107,744 -> 108,024 ms, nessun vantaggio), riduzione wave shuffle al posto della device library (107,115 -> 131,407 ms). Correttezza e test mirati passavano, ma non bastano a giustificare una regressione prestazionale. Ripristinati i sorgenti dai manifest; i binari e le misure delle prove restano separati. Non applicare queste percentuali ad altri confronti.

L'SDK installato definisce `__fdividef(x,y)` come `x/y`: il nome da solo non prova una divisione approssimata più veloce. Questo ha motivato una prova isolata, non una modifica accettata dell'SDK o del driver. I test matematici aggiuntivi conservati verificano la divisione della trasmittanza su 65.536 input entro un errore relativo 3e-7 e la somma in ogni lane. Il culling precomputato aveva passato 65.536 confronti esatti, compresi i casi alla soglia degenere, ma anche quel codice sperimentale è stato rimosso.

Artefatti in `.local/rocm-tensor-study/sdk-sources/audit`: `backward-optimization-comparison.json`, `factor-adjoint-training-repeated-200.json`, `factor-adjoint-analysis.json`, `factor-adjoint-quality.png`, `backward-final-source-parity.json`. In totale cinque confronti da quattro run di 200 step (4.000 step dataset); tutti con GPU serializzata. Nessuna sostituzione di SDK/driver o modifica dei sorgenti CUDA/Metal/Vulkan.

Verifica finale della configurazione conservata: 25/25 test nativi passati (`sdk-sources/audit/factor-adjoint-final-native-suite.log`), compresi budget/recupero, coesistenza CUDA e condivisione HIP–Vulkan. Generatore verificato idempotente per byte e mtime su 22 file, identici ai sorgenti effettivamente compilati (`factor-adjoint-generator-validation.json`). Tutti i benchmark di questo confronto sono conclusi.


### 19 ottobre: disposizione locale dei sotto-tile e nuovi confronti SDK

La variante con un pixel per thread è stata verificata e scartata: 110,661 -> 130,946 ms/step nel confronto ABBA da quattro run di 200 step. LLVM riporta 80 -> 61 VGPR, gruppi 128 -> 256 thread e zero spill SGPR/VGPR in entrambi i kernel RGB; ridurre i registri non ha quindi prodotto un guadagno. Non si attribuisce automaticamente tutto il costo alle atomiche: non è stato misurato un contatore hardware che lo dimostri. Cinque sorgenti ripristinati dal manifest `one-pixel-source-reference`; nessuna opzione one-pixel conservata nel repository. Metadati e decisione in `one-pixel-reference-backward.json`, `one-pixel-candidate-backward.json`, `one-pixel-decision.json` sotto `sdk-sources/audit`.

Le nuove opzioni private `LFS_HIP_FAST_ADJACENT_PIXELS` e `LFS_HIP_FAST_FORWARD_ADJACENT`, entrambe default OFF, mantengono quattro wave e due pixel per thread. Cambiano l'abbinamento dei sotto-tile da (0,4)/(1,5)/(2,6)/(3,7) a (0,2)/(1,3)/(4,6)/(5,7): ogni coppia copre una regione locale 8x8. Nessun cambio di formato, allocazione o matematica del singolo contributo; la somma dei gradienti può cambiare ordine FP32. Il backward adiacente richiede esplicitamente lo staging wave già disponibile. Verificata anche la copertura geometrica senza duplicati su 289 forme di bordo (0..16 per asse), oltre ai test raster su tile completi/parziali.

Il solo backward aveva un vantaggio medio piccolo (109,314 -> 106,490 ms) con riferimenti variabili; non si usa quel dato da solo per dichiarare una percentuale acquisita. La combinazione forward/backward è stata confrontata direttamente con il riferimento precedente, dopo un run separato di riscaldamento da 60 step, escluso dai dati:

| Variante | Run A ms/step | Run B ms/step | Media |
| --- | ---: | ---: | ---: |
| Coppie distanti precedenti | 107,717 | 106,847 | 107,282 |
| Coppie adiacenti forward/backward | 104,524 | 104,300 | 104,412 |

Protocollo invariato: due viste, lato massimo 1920, un milione di gaussiane effettive, SH0/fused, 200 step per run, primi 10 esclusi. Vantaggio 2,67% della combinazione, non sommare ad altre percentuali storiche. Contabilità sessione invariata: circa 727,885 MB live, 889,786 MB picco. MSE finale riferimento 0,0170541998 / 0,0170546030; combinazione 0,0170540168 / 0,0170537455.

Le immagini iniziali esportate a 8 bit sono identiche; questa non è una prova di identità dei buffer float. Dopo 200 step: differenza media RGB 0,0423/0,0491 livelli su 255, confrontabile con le ripetizioni della stessa variante (circa 0,044–0,052). Massimi locali fino a 45 contro 44 fra ripetizioni del riferimento. Controllo visivo senza peggioramenti evidenti. Rimangono i limiti a due viste, SH0 e topologia fissa: nessuna garanzia estrapolata a holdout o strategie complete.

Confronto SDK aggiuntivo: sette configurazioni rocPRIM onesweep (8 bit, ranking match, histogram 256x12), usando API DoubleBuffer, chiavi uint64/valori uint32, 46 bit e input sintetico non ordinato ripristinato in GPU prima di ogni campione. Il ripristino resta fuori dagli eventi cronometrati. Verificati esattamente ordinamento, corrispondenza degli indici e stabilità con il 20% di chiavi duplicate, per un milione e 6.224.786 elementi. Ogni eseguibile è stato eseguito in ordine diretto e inverso; quattro warmup più venti campioni. Il microbenchmark usa conteggio size_t, mentre il chiamante training usa int: non si trasferiscono automaticamente questi tempi al training.

| Configurazione sort | 1M, ms | 6,225M, ms | Scratch 6,225M byte |
| --- | ---: | ---: | ---: |
| 256x16 attuale | 2,149 | 13,044 | 1.570.816 |
| 128x16 | 2,279 | 13,693 | 3.127.296 |
| 128x24 | 2,729 | 16,382 | 2.089.984 |
| 128x32 | 2,384 | 13,278 | 1.570.816 |
| 256x24 | 2,738 | 16,159 | 1.052.672 |
| 256x32 | 2,345 | 12,923 | 792.576 |
| 512x16 | 2,202 | 13,082 | 792.576 |

Mantenuto 256x16: la variante 256x32 guadagna meno dell'1% sul caso grande ma peggiora sul piccolo; non giustifica un cambio del training. I valori di scratch DoubleBuffer non vanno confusi con quelli dell'API out-of-place del precedente microbenchmark, che richiede ulteriore workspace. Nessuna patch di SDK o driver applicata.

Verifica finale combinazione adiacente: 25/25 test nativi passati, inclusi budget/recupero, coesistenza CUDA e condivisione HIP–Vulkan. Generatore idempotente per byte e mtime su 22 file, identici ai sorgenti compilati. Artefatti in `sdk-sources/audit`: `paired-adjacent-training-repeated-200.json`, `paired-adjacent-analysis.json`, `paired-adjacent-quality.png`, `paired-adjacent-final-native-suite.log`, `paired-adjacent-generator-validation.json`, `adjacent-pixels-layout-verification.json`, `radix-double-repeated.json`, `radix-double-analysis.json`. Sorgenti originali CUDA/Metal/Vulkan invariati; HIP e nuove ottimizzazioni rimangono opt-in.


### 19 ottobre: verifiche aggiuntive sulle riduzioni del backward

Nuovo profilo della configurazione conservata, 30 step, primi cinque esclusi: blending backward 60,55 ms, proiezione backward 5,78 ms, forward 20,39 ms, binning/sort 18,81 ms, loss/gradient 6,11 ms, Adam 1,18 ms. La profilazione sincronizza le fasi: i tempi non sono direttamente confrontabili con il throughput senza profiling e le misure annidate non vanno sommate due volte. Non prova che un componente specifico del driver o le atomiche siano l'unica causa.

Tre nuove varianti private, tutte con cinque test mirati passati, sono state confrontate ciascuna con quattro run ABBA da 200 step dopo warmup separato da 60 step. Stesso dataset/protocollo delle misure precedenti (due viste, 1920, 1M, SH0/fused, primi dieci step esclusi):

| Esperimento | Riferimento ms | Variante ms | Decisione |
| --- | ---: | ---: | --- |
| Riduzione dedicata a due lane | 105.929 | 106.656 | Scartata |
| Culling preliminare solo AABB | 105.462 | 110.863 | Scartata |
| Riduzione nativa anche con una sola lane | 104.269 | 104.787 | Scartata |

Nessuna variante ha prodotto un guadagno materiale ripetibile. Conservati binari e risultati separati; generatore ripristinato byte per byte dal riferimento. Nessuna nuova opzione sperimentale mantenuta. Questi confronti aggiungono 2.400 step misurati, più 180 di warmup. Il riferimento rimane circa 104–106 ms: dimezzarlo è un obiettivo non ancora dimostrato, che richiederebbe ulteriori cambiamenti nell'organizzazione del raster, non soltanto queste scorciatoie. Nessuna modifica SDK/driver o ai sorgenti CUDA/Metal/Vulkan.

Artefatti: `sdk-sources/audit/adjacent-next-profile.json`, `sparse-two-analysis.json`, `bbox-only-analysis.json`, `full-reduce-analysis.json` e rispettivi `paired-*-training-repeated-200.json`/`*-tests.log`. Ogni percentuale vale solo per il proprio confronto alternato. Le prove scartate non richiedono dichiarazioni di equivalenza qualitativa né vengono adottate sulla sola base della MSE.

Verifica finale dopo il ripristino: 25/25 test nativi passati (`after-reduction-final-native-suite.log`); 22 file generati identici a quelli compilati, generatore identico al riferimento e idempotente per byte/mtime (`after-reduction-generator-validation.json`). Tutti i benchmark e le compilazioni di questo ciclo sono terminati.


### 19 ottobre: completamento del riuso degli helper tra fasi HIP

L'audit CUDA/HIP ha trovato due adattamenti incompleti nel gestore della sessione sperimentale: callback della fase forward senza riuso nel backward; helper dimensionati sul totale delle gaussiane anziché sul workset visibile. Il raster e le formule erano portati, ma la gestione delle allocazioni non seguiva queste ottimizzazioni del percorso CUDA.

`LFS_HIP_FAST_PHASE_HELPERS=1` abilita ora il dimensionamento sul workset visibile (una riga reale minima per costruire puntatori validi anche con workset vuoto) e il riuso della più grande allocazione temporanea della fase forward quando sufficiente. La capacità deriva dal Buffer posseduto, senza superare il budget né creare una seconda proprietà. Indici ordinati trattenuti, stato dei tile, visibilità e attributi proiettati persistenti non sono candidati al riuso. Il forward è già completato sulla coda della sessione prima dell'azzeramento degli helper; tutto il preprocessing backward continua a visitare le righe del modello, incluso il decadimento dei momenti delle gaussiane invisibili. Se nessun buffer è sufficiente, rimane un buffer dedicato con sostituzione transazionale e controllo del budget. Default OFF; nessuna modifica dei kernel matematici o dei backend originali.

Il solo riuso con dimensione full-N ha eliminato 64 MB, senza un miglioramento di throughput (106,270 -> 107,526 ms nel suo confronto ABBA); la combinazione con workset compatto è stata poi confrontata separatamente:

| Variante | Primo run ms | Secondo run ms | Media ms |
| --- | ---: | ---: | ---: |
| Riferimento precedente | 111.174 | 111.005 | 111.089 |
| Helper compatti e riuso | 109.973 | 108.997 | 109.485 |

Guadagno 1,44% in questo confronto, non sommare alle percentuali storiche né sostituire la precedente misura da 104,41 ms ottenuta in un'altra serie. Stessi parametri: due viste, massimo 1920, un milione di gaussiane, SH0/fused, quattro run da 200 step con primi dieci esclusi e warmup separato da 60 step. Contabilità allocazioni della sessione: live circa 727,885 -> 663,885 MB, picco 889,786 -> 825,786 MB, risparmio 64.000.000 byte. Non sono misure di residenza fisica VRAM.

Immagini iniziali esportate identiche a 8 bit. Dopo training, differenza media RGB 0,0449/0,0495 livelli su 255; riferimento contro sua ripetizione 0,0478/0,0509. Massimi delle coppie 9/11, ripetizioni del riferimento 12/39. Controllo visivo senza peggioramenti evidenti. Restano i limiti a due viste, topologia fissa e SH0: non prova qualità su holdout o Trainer completo.

Aggiunto test di otto transizioni vuoto/visibile con learning rate zero, per verificare recupero del buffer dedicato, riuso, output vuoto, parametri e salute della sessione. Il test di esaurimento budget ora forza il fallimento a freddo prima delle allocazioni forward, senza assumere che il backward debba allocare separatamente gli helper. Suite completa con riuso attivo: 25/25; test di convergenza, budget e cancellazione anche con opzione assente: passato.

L'audit ha inoltre contato 60 metodi Tensor ancora esplicitamente non implementati: generano errori, non fallback CPU silenziosi. Non sono chiamati nel percorso di raster/training misurato. Rimangono lavoro reale per Trainer/TrainingOps, strategie di crescita/potatura, checkpoint e integrazione loader/UI; non descrivere il backend come integrato completamente.

Artefatti: `sdk-sources/audit/phase-helper-analysis.json`, `compact-phase-helper-analysis.json`, `compact-phase-helper-quality.png`, `compact-phase-helper-tests.log`, `compact-phase-helper-final-native-suite.log`, `compact-phase-helper-default-session.log`, `hip-porting-audit-phase-helpers.json`. Nessuna patch di SDK o driver e nessun push/PR.


### 19 ottobre: compilazione matematica selettiva dei kernel privati

Il confronto delle impostazioni di build ha trovato FastGS CUDA Release con `--use_fast_math`, mentre le unità HIP private usavano O3 e alcune intrinseche esplicite. Non è una prova che l'intero insieme fast-math sia opportuno in HIP. La nuova opzione `LFS_HIP_FAST_APPROX_FUNCTIONS` (default OFF) abilita esclusivamente `-fapprox-func -freciprocal-math -ffp-contract=fast` sulle due unità FastGS e sui relativi probe matematici/raster. Nessun `-ffast-math`, nessuna assunzione generale di assenza di NaN/Inf, nessuna modifica del denormal mode o dei flag degli altri target. I controlli della sessione rimangono compilati separatamente con le impostazioni precedenti. Un probe con lo stesso SDK produce IR con i controlli `fcmp uno`/confronto infinito ancora presenti e divisione `arcp contract afn`, senza `nnan`, `ninf` o riassociazione generale (`selective-math-flags.ll`). Questo controlla le assunzioni del compilatore, non dimostra identità numerica delle funzioni approssimate.

Con riuso compatto attivo in entrambe le varianti, quattro run ABBA per ciascuna durata, warmup separato di 60 step e primi dieci step esclusi:

| Durata | Riferimento ms | Variante ms | Riduzione |
| --- | ---: | ---: | ---: |
| 200 step | 107.984 | 103.471 | 4.18% |
| 500 step | 99.856 | 97.193 | 2.67% |

Le durate restano confronti distinti: il costo cambia con l'evoluzione del modello e non si attribuisce il passaggio 103 -> 97 ms interamente a un'ulteriore ottimizzazione. Stesso dataset, 1M gaussiane, due viste, 1920, SH0/fused; nessuna modifica di precisione dei parametri o dei codec dell'ottimizzatore. Contabilità live/picco circa 663,885/825,786 MB, sostanzialmente invariata rispetto alla variante compatta.

A 200 step MSE finali riferimento 0,0170544740 / 0,0170541809, nuove 0,0170538446 / 0,0170540807. A 500 step riferimento 0,0104700122 / 0,0104654174, nuove 0,0104633190 / 0,0104664847. I valori di ogni run rimangono nei JSON grezzi. Differenze medie RGB tra varianti dopo 500 step 0,7788/0,7142 livelli su 255; fra ripetizioni del riferimento 0,7355/0,7070, della candidata 0,7118/0,7021. Massimi locali rispettivamente 70/87, 81/80 e 89/87: nessuna pretesa di equivalenza bit per bit o garanzia su ogni pixel. Anche le immagini iniziali cambiano leggermente: massimo un livello su 255, media circa 0,000013–0,000015. Controllo visivo delle quattro immagini per vista senza peggioramenti evidenti. Rimangono i limiti di due viste, SH0, topologia fissa, assenza di holdout e integrazione Trainer non completata.

Cinque test mirati passati (comprese 72 combinazioni raw raster/gradienti, matematica e convergenza FP32/fused SH0–SH3). Suite nativa completa: 25/25, compresi condivisione del modello HIP–Vulkan e coesistenza CUDA. Aggiunto poi un test dedicato con 10.007 gaussiane visibili nello stesso tile e immagine 1x1: verifica che lo scratch insufficiente richieda helper dedicati, che il rifiuto per budget conservi tutte e cinque le famiglie dei parametri e il contatore, e che ripristino del budget e successivo riuso senza margine funzionino. Il test ampliato passa con riuso sia attivo sia disattivo. Nessuna modifica produttiva successiva alla suite completa: solo esplicitazione dell'inizializzazione delle liste CMake e commenti.

Metadati SDK del kernel RGB: VGPR 80 -> 79, SGPR 46 -> 43; zero spill SGPR/VGPR, gruppi da 128 e wave32 invariati. Questo non prova un aumento di occupancy né attribuisce da solo il guadagno a una specifica risorsa hardware. Nessuna patch di SDK/driver o ricompilazione del compilatore.

Artefatti sotto `sdk-sources/audit`: `approx-functions-analysis.json`, `approx-functions-long-analysis.json`, relative immagini `*-quality.png`, `approx-functions-tests.log`, `approx-functions-final-native-suite.log`, `phase-helper-dedicated-fallback.log`, `phase-helper-dedicated-default.log`, `selective-math-flags.ll`, `approx-functions-{reference,candidate}-backward.json`. Build locale di studio: `build-probes-approx-functions.cmd`; il precedente script adiacente forza ora esplicitamente l'opzione matematica OFF, così non eredita accidentalmente il valore ON dalla cache CMake. Entrambe le nuove ottimizzazioni rimangono opt-in e locali, nessun push/PR.


### 19 ottobre: aggregazione per tile, SSIM e buffer esclusivi del raster

Tre famiglie di esperimenti isolate non sono state adottate:

| Esperimento | Riferimento ms | Variante ms | Esito |
| --- | ---: | ---: | --- |
| Gradienti aggregati tra quattro wave | 97.931 | 102.730 | Scartato |
| SSIM con canali su blocchi indipendenti | 99.578 | 100.605 | Scartato |

Ogni confronto training usa quattro run ABBA da 200 step, primi dieci esclusi e warmup separato da 60 step, con 1M gaussiane, due viste, 1920 e SH0/fused. L'aggregazione dei gradienti aveva cinque test mirati passati e manteneva il clipping per wave prima della somma, ma introduceva sincronizzazioni e non giustificava il costo. Anche SSIM parallela passava i quattro test mirati, incluso l'oracolo CPU, senza migliorare il training. La variabilità osservata non viene trasformata in un guadagno.

Probe SSIM indipendenti su immagini sintetiche CHW 1440x1920: quattro warmup e venti campioni con eventi GPU, in ordine alternato. Con canali indipendenti forward/backward medi 3,468/3,456 ms contro 3,051/3,270 ms del riferimento. Provati anche blocchi larghi 32x16, con canali seriali/paralleli, e una disposizione swizzled XOR della memoria condivisa. Nessun vantaggio convincente: il layout largo seriale resta vicino al rumore (6,484 ms complessivi contro 6,660 del proprio riferimento, con deriva nei due run), gli altri sono più lenti. Nessun contatore hardware ha provato che i conflitti LDS siano il collo di bottiglia. In tutti i probe, tutti i partial FP16 e gradienti FP32 sono identici byte per byte fra riferimento e varianti; ciò non basta a giustificare un cambio senza un guadagno misurato. Queste varianti restano solo negli artefatti di studio. Sorgenti, template, CMake e test photometrici ripristinati; nessuna nuova opzione SSIM nel repository.

Il secondo audit ha invece individuato sette buffer privati usati soltanto da 3DGUT che restavano allocati scegliendo FastGS: color (12N byte), radii (8N), means2d (8N), depths (4N), conics (12N), counts (4N), prefix (8N). La nuova variabile `LFS_HIP_FAST_TRIM_UNUSED=1`, default assente/OFF, li rilascia quando si seleziona esplicitamente FastGS prima di associare un frame. Parametri condivisi/importati, gradienti dell'ottimizzatore e scratch della scan rimangono separati; nessun rilascio di memoria di proprietà del Tensor/renderer.

Tornando a 3DGUT prima del frame, i sette sostituti sono costruiti completamente sotto lo stesso budget prima del commit. Un fallimento restituisce ResourceExhausted lasciando FastGS e la contabilità precedente intatti. Il preprocessing 3DGUT ha anche una guardia che ricostruisce i buffer se mancanti. Il parametro color di Adam riceve nullptr solo su FastGS, dove il ramo che potrebbe leggerlo è già disabilitato. Default e kernel degli altri backend non cambiano.

Confronto ABBA da 200 step del solo rilascio:

Riferimento 101.198 -> variante 100.553 ms, circa 0.64%. Non viene dichiarata un'accelerazione significativa da una differenza inferiore all'1%. Il beneficio verificato è di 56.000.000 byte in meno, live circa 663,885 -> 607,885 MB e picco 825,786 -> 769,786 MB. Sommati al precedente riuso degli helper: 120 MB di allocazioni in meno rispetto a prima delle due modifiche di memoria; non sono misure della residenza fisica VRAM.


Le immagini iniziali esportate sono identiche a 8 bit. Dopo 200 step differenza media RGB 0,0467/0,0494 livelli su 255, contro 0,0468/0,0504 tra ripetizioni del riferimento e 0,0427/0,0540 tra ripetizioni della variante. Massimi locali rispettivamente 49/42, 8/45 e 49/48: non si dichiara identità del singolo pixel. Controllo visivo senza peggioramenti evidenti. Restano i limiti del test su due viste, SH0 e topologia fissa.

Test aggiunto: verifica il rilascio esatto di 56 byte per gaussiana, il fallimento immediato e a ricostruzione parziale per budget insufficiente, il rollback della contabilità, il successivo ripristino e un vero step 3DGUT. Suite completa con rilascio e riuso helper attivi: 25/25 passati; sessione predefinita senza entrambe le variabili: passata. Dopo l'aggiunta del caso di ricostruzione parziale viene rieseguito il test di sessione in entrambe le modalità. Nessun kernel produttivo cambiato dopo la suite completa.

Artefatti sotto `sdk-sources/audit`: `block-aggregate-analysis.json`, `photo-channels-analysis.json`, `photo-{channels,wide,swizzle}-probe-{results,parity}.json`, `block-photo-experiments-restoration.json`, `block-photo-final-generator-validation.json`, `trim-geometry-analysis.json`, `trim-geometry-quality.png`, `trim-geometry-tests.log`, `trim-geometry-final-native-suite.log`, `trim-geometry-default-session.log`, `trim-geometry-partial-rollback.log`, `trim-geometry-partial-default.log`. Verificati 22 file FastGS identici a quelli compilati e idempotenti per byte/mtime, più identità del sorgente photometric generato e assenza delle varianti scartate. Tutto locale e opt-in, nessun push/PR né modifica di SDK o driver.

Esito finale: anche il test ampliato di rollback parziale passa con entrambe le modalità. Benchmark, probe, compilazioni e verifiche di questo ciclo sono conclusi.

### 19 ottobre: backward RGB a quattro pixel con riduzione protetta

La nuova opzione privata `LFS_HIP_FAST_FOUR_PIXELS`, default OFF, usa 64 thread (due wave32) e quattro pixel per thread nel backward RGB. Richiede staging indipendente, disposizione adiacente e scatter delle riduzioni; CMake rifiuta combinazioni incomplete. Depth, normali, statistiche di densificazione/edge e modalità uncull di riferimento mantengono il kernel precedente. Guardie statiche vincolano il kernel a tile 16x16 e alla specializzazione RGB senza canali ausiliari. Forward, optimizer, codec dei parametri e sorgenti CUDA/Metal/Vulkan non cambiano.

La prima variante manteneva sempre due riduzioni separate: 98,140 -> 98,924 ms, senza guadagno, e non è stata adottata. La variante conservata unisce le riduzioni solo quando ogni lane dimostra che la somma assoluta di tutti i campi delle due vecchie quad è inferiore a clamp/64. Ogni somma originale di 32 lane resta sotto clamp/2, con ampio margine di arrotondamento: il clipping è allora un'identità. Altrimenti si riducono e limitano separatamente le due quad originali 8x8 prima di sommarle. NaN/Inf rendono falso il confronto; nessuna assunzione finite-only. Questo evita la cancellazione di gradienti saturi di segno opposto prima del clipping. L'ordine delle somme FP32 può cambiare: nessuna promessa di identità bit per bit.

Confronti ABBA, warmup separato da 60 step, primi dieci step esclusi, nessuna compilazione durante le misure. Stesso dataset statua, 1M gaussiane, due viste, lato lungo 1920, SH0, fused Adam, budget 2 GiB, identici flag di memoria:

| Durata per run | Riferimento ms | Variante ms | Riduzione |
| --- | ---: | ---: | ---: |
| 200 step | 98,556 | 92,788 | 5,85% |
| 500 step | 92,318 | 87,213 | 5,53% |

Le durate restano protocolli distinti; non si sommano questi guadagni alle percentuali storiche. Le singole medie a 500 step sono 92,905/91,731 ms riferimento e 86,713/87,713 ms candidata. Contabilità sostanzialmente invariata: circa 607,885 MB live, 769,786 MB picco, più target CPU da 66,355 MB. Non sono misure di residenza fisica VRAM.

MSE finali a 500 step: riferimento 0,0104620315 / 0,0104713229, nuova 0,0104674607 / 0,0104666952. Differenze medie RGB fra varianti 0,7239/0,7128 livelli su 255; fra ripetizioni del riferimento 0,6902/0,6995, della candidata 0,7576/0,7172. Massimi locali 77/89 contro 97/89 e 96/93. Ispezionate entrambe le viste e crop a risoluzione nativa, senza peggioramenti evidenti. A 200 step differenza media circa 0,048 livelli, comparabile alle ripetizioni. Restano due viste, SH0, topologia fissa e nessun holdout: non è una garanzia per ogni dataset né una misura comparabile alla vecchia baseline Vulkan.

Suite completa sulla variante: 25/25, incluse convergenza FP32/fused SH0–SH3, budget/cancellazione, memoria HIP–Vulkan e coesistenza CUDA. Aggiunte poi 24 coppie ottimizzato/riferimento (48 invocazioni) al probe raster: due larghezze parziali, gradienti piccoli/intermedi/saturanti positivi e negativi, anche con segni opposti fra quad e canali. Confrontati 16 helper più 14 gradienti dei parametri; si controlla anche che il caso saturante raggiunga davvero tale regime. Tutte le coppie passano insieme ai 72 casi analitici originali. Ricompilata l'opzione OFF: altri cinque test mirati passati, compresi clipping e convergenza. Nel relativo oggetto il simbolo del nuovo kernel è assente; il vecchio conserva 79 VGPR/43 SGPR, gruppo 128, zero spill.

Metadati SDK della variante: 107 VGPR, 55 SGPR, wave32, gruppo massimo 64, zero spill e scratch privato zero. I maggiori registri non provano un incremento di occupancy. Nessuna patch di driver/SDK. Il readback temporaneo dei bounding box proiettati è stato rimosso integralmente prima dei benchmark: quei box sono limiti superiori, non conteggi di contributi accettati.

Artefatti sotto `sdk-sources/audit`: `four-pixel-guarded-{200,500}-analysis.json`, immagini `*-quality.png`, `four-pixel-gated-all-tests.log`, `four-pixel-mixed-clipping-tests.log`, `four-pixel-option-off-tests.log`, `four-pixel-guarded-backward.json`, `four-pixel-layout-proof.json` e `four-pixel-generator-validation.json`. Provati 256 pixel unici, 289 forme di bordo e conservazione delle quad originali. I 22 file generati sono idempotenti per byte/mtime e corrispondono ai compilati. Build di studio `build-probes-approx-functions.cmd`; lo script precedente forza la nuova opzione OFF per non ereditarla dalla cache. Tutto locale, nessun push/PR, ammissione al Trainer principale ancora chiusa.

Verifica conclusiva: anche dopo le guardie statiche e i test con segni misti, il raster e la sessione con variabili di memoria assenti passano (2/2). Build ottimizzata ripristinata con opzione ON; generazione finale verificata nuovamente su 22 file. Tre combinazioni CMake incomplete respinte come previsto. Artefatti aggiuntivi: `four-pixel-final-default-session-tests.log`, `four-pixel-config-guards.json`, `four-pixel-option-off-symbol-check.json`. Nessun benchmark o compilazione di questo ciclo rimane attivo.


### 19 ottobre: letture uniformi dei payload nel backward HIP

Aggiunta l'opzione privata `LFS_HIP_FAST_UNIFORM_BROADCAST`, default OFF. Richiede native wave e staging indipendente, con rifiuto CMake delle combinazioni incomplete. Soltanto le letture dei payload dal batch, con indice sorgente `k` uniforme per l'intera wave32 attiva, usano `__builtin_amdgcn_readlane`. Il generico shuffle, i percorsi con indice variabile per lane, le riduzioni e il clipping non cambiano. Le componenti FP32 viaggiano come bit unsigned, senza conversioni numeriche. Il batch conserva l'ordine dei contributi accettati. Anche il kernel precedente a due pixel usa la nuova lettura quando esplicitamente abilitata, inclusi i suoi canali ausiliari. Nessuna nuova allocazione, modifica dell'optimizer, del codec dei parametri o dell'ammissione al Trainer principale. Nessuna modifica di CUDA, Metal o Vulkan.

Confronti ABBA, warmup separato 60 step, primi dieci esclusi, nessuna compilazione concorrente. Statua, 1M gaussiane, due viste, 1920, SH0/fused Adam, budget 2 GiB, stessi flag di memoria. Il riferimento è la precedente variante a quattro pixel con guardia a somma assoluta, ricampionata nello stesso confronto:

| Durata per run | Riferimento ms | Variante ms | Riduzione |
| --- | ---: | ---: | ---: |
| 200 step | 92,806 | 90,901 | 2,05% |
| 500 step | 87,031 | 85,428 | 1,84% |

A 500 step riferimento 87,0077/87,0537 ms, candidata 85,4086/85,4467 ms. Le durate restano protocolli distinti e le percentuali non si sommano alle precedenti. Memoria circa 607,885 MB live, 769,786 MB picco, target CPU 66,355 MB, invariata entro le piccole differenze dei conteggi visibili. Nessuna misura di residenza fisica VRAM. I tempi restano del probe di training a topologia fissa e non del Trainer applicativo completo.

MSE finali a 500 step: riferimento 0,0104650310 / 0,0104621376, candidata 0,0104653568 / 0,0104718077. La media della candidata è circa 0,048% superiore in questo piccolo campione: non si dichiara un miglioramento qualitativo né equivalenza esatta. A 200 step riferimento 0,0170539867 / 0,0170538349, candidata 0,0170534138 / 0,0170542992. I quattro export iniziali (due viste per ciascuna durata) sono identici byte per byte. Dopo 500 step differenze medie RGB fra varianti 0,7844/0,7245 livelli su 255, fra ripetizioni del riferimento 0,7590/0,6899 e della candidata 0,7331/0,7308. Massimi locali rispettivamente 79/73, 78/78 e 65/82. Ispezionate entrambe le viste e crop nativi senza peggioramenti evidenti. Restano i limiti di due viste, SH0, topologia fissa e nessun holdout; gli atomici non deterministici impediscono di presumere identità del training dal solo trasporto esatto dei payload.

Il test matematico verifica esattamente i bit float e unsigned per tutte le 32 lane sorgenti e blocchi da 32/64/128/512 thread, con indici distinti per wave nello stesso blocco. Copre finite, subnormali, estremi, zeri con segno, infiniti e payload NaN: 23.552 risultati float e altrettanti unsigned. Passa sia con la variante attiva sia in un eseguibile separato senza alcun flag di ottimizzazione matematica/native wave. Suite completa 25/25, inclusi i 72 casi analitici raster, le 24 coppie di gradienti/clipping, convergenza FP32/fused SH0–SH3, budget/cancellazione, HIP–Vulkan e coesistenza CUDA. Ricompilata anche l'opzione OFF: cinque test mirati passati.

Nel kernel RGB a quattro pixel le istruzioni statiche `ds_bpermute_b32` passano da 47 a 35, con 12 `v_readlane_b32`; non sono conteggi dinamici. VGPR 107 -> 106, SGPR 55 -> 63, zero spill e scratch privato zero. Questi numeri non dimostrano un incremento di occupancy; il guadagno dichiarato proviene dai confronti di training.

Due ulteriori studi non sono stati adottati:

| Esperimento | Riferimento ms | Variante ms | Esito |
| --- | ---: | ---: | --- |
| Guardia del clipping con limite IEEE a OR di bit | 92,549 | 92,699 | Nessun guadagno |
| Radix onesweep a 10 bit, istogramma con due stripe | 92,486 | 91,676 | Meno dell'1%, non giustifica la dipendenza sperimentale |

Entrambi hanno cinque test mirati passati e quattro run da 200 step. La guardia IEEE ha anche 100.000 vettori di verifica del limite, inclusi NaN/Inf, ma il sorgente conservato usa la precedente somma assoluta. Il radix 10 bit standard non compila: l'istogramma richiede 114.688 byte LDS oltre il limite 65.536; cambiare soltanto il block size non risolve. Nella copia isolata dei soli header rocPRIM si è reso configurabile `atomic_stripes` dell'helper onesweep, senza modificare l'SDK installato. Probe con conteggi int32, chiavi FP32 complete e 20% di pareggi: valori/indici e stabilità esatti a 1M e 6.224.786 elementi. Su quest'ultima taglia mediane ripetute 12,872 ms per 8 bit/4 stripe, 12,032 per 10 bit/2 stripe, 11,797 per 10 bit/1 stripe; sul milione di elementi 2 stripe era migliore di 1. Il vantaggio isolato del sort non si trasferisce interamente al training. La variante a due stripe aggiungeva circa 5,08 MB live e 10,16 MB di picco rispetto al riferimento, quindi è stata esclusa dal codice finale. Tutte le modifiche rocPRIM restano nei soli artefatti, il radix produttivo privato è tornato a 8 bit e l'hash dell'header SDK installato è invariato. Nessuna patch di driver o dipendenza da percorsi di studio nel repository.

Artefatti sotto `sdk-sources/audit`: `paired-uniform-broadcast-{200,long-500}-analysis.json`, immagini di qualità, `uniform-broadcast-tests.log`, `uniform-full-native-suite.log`, `uniform-default-math-test.log`, `uniform-option-off-tests.log`, `uniform-broadcast-backward.json`, `uniform-broadcast-instructions.json`, `uniform-initial-image-parity.json`, `uniform-config-and-sdk-guards.json`, `four-pixel-bit-bound-analysis.json`, `radix-bits-analysis.json`, `radix-stripes-sdk-patch.json` e `wide-radix-training-analysis.json`. I JSON mantengono tutti i run, compresi quelli senza beneficio.

Build di studio aggiornata: `build-probes-uniform-broadcast.cmd`. Gli script precedenti mantenuti forzano la nuova opzione OFF per evitare eredità accidentali dalla cache CMake. Tutto locale e opt-in, nessun push/PR.

Verifica conclusiva: ricompilata e ripristinata la variante ottimizzata, passano anche matematica, raster e convergenza con le variabili di memoria assenti (3/3). Verificati nuovamente 22 file generati idempotenti per byte/mtime e identici ai compilati, guardia assoluta e radix 8 bit ripristinati, sorgente training privo del readback diagnostico. Artefatti finali `uniform-final-default-session-tests.log` e `uniform-final-generator-validation.json`. Nessun training, probe o compilazione di questo ciclo rimane attivo.


### 19 ottobre: tile RGB intero per wave nel forward e backward

Aggiunte due opzioni private, entrambe default OFF: `LFS_HIP_FAST_EIGHT_PIXELS` e `LFS_HIP_FAST_FORWARD_EIGHT_PIXELS`. Un gruppo da 32 thread copre tutti i 256 pixel del tile 16x16, con otto pixel per thread. Il backward richiede la disposizione/clipping già usata dal percorso a quattro pixel; il forward richiede le letture uniformi native. CMake rifiuta combinazioni incomplete. Le due opzioni sono indipendenti e sono state provate anche separatamente.

Nel backward rimangono quattro accumulatori logici, uno per ciascuna quad originale 8x8: pixel (0,2), (1,3), (4,6), (5,7). Il limite assoluto include tutti i campi delle quattro quad. Solo se ogni lane resta sotto clamp/64 si possono unire le riduzioni; altrimenti ogni quad viene ridotta e limitata separatamente prima della somma. Non si cancella un gradiente saturo prima del suo clipping. Il prefisso inverso comune del tile può essere più lungo di quello delle precedenti mezze tile, ma le guardie `last` conservano la sequenza dei contributi accettati di ciascun pixel. La modifica è confinata al backward RGB privo di canali ausiliari.

Nel forward, RGB senza profondità/normali usa batch da 32 primitive in registri privati. Ogni lane carica una primitiva e le letture uniformi vengono eseguite prima di qualsiasi `continue` divergente dei pixel: la lane sorgente rimane attiva anche se i suoi pixel sono fuori immagine o già terminati. La sincronizzazione di blocco è sostituita dal ballot dell'unica wave; non rimane staging LDS. Le modalità di test con culling disattivato/negativo, canali ausiliari o batch esplicitamente richiesti diversi da 32 mantengono il vecchio forward. Il batch predefinito può cambiare senza cambiare l'ordine di blending di ciascun pixel. Nessuna nuova allocazione e nessuna modifica al layout dei parametri, all'optimizer o ai kernel CUDA/Metal/Vulkan.

#### Confronto principale pulito

Statua, 1M gaussiane, due viste, lato lungo 1920, SH0/fused Adam, budget 2 GiB, shuffle 42, stessi flag di memoria. Quattro run ABBA da 500 step, primi dieci esclusi, warmup separato da 300 step. Nessuna compilazione, query dei contatori GPU o analisi immagini durante questa serie. Il riferimento è la precedente build a quattro pixel con letture uniformi, non una media storica ricostruita:

| Configurazione | Run 1 ms | Run 2 ms | Media ms |
| --- | ---: | ---: | ---: |
| Riferimento | 84,5179 | 84,8220 | 84,6700 |
| Tile intero | 80,4164 | 80,4282 | 80,4223 |

Riduzione misurata **5,02%**. I p95 sono circa 97,11/97,05 ms per il riferimento e 91,57/91,37 ms per la variante. Nessuno dei quattro run contiene step oltre 150 ms. Live/picco circa 607,885/769,786 MB, target CPU 66,355 MB, senza nuove allocazioni. Non sono misure di residenza fisica VRAM né tempi del Trainer applicativo completo: rimane il probe a topologia fissa.

La prima serie combinata da 200 step è stata mantenuta nei dati ma esclusa come confronto principale: riferimento 90,884/95,543 ms, candidata 85,617/91,792 ms; tre step della seconda candidata superavano 150 ms, con massimo 403,295 ms. Il carico in istanze era quasi identico. Il controllo dei processi non ha mostrato altre build o sessioni LichtFeld, e la RAM disponibile era abbondante; non si attribuisce la deriva a una causa hardware provata. Le query WMI dei contatori erano in sola lettura. Anche la successiva serie esplorativa da 500 step, con diagnostica durante parte dell'esecuzione, resta distinta: 87,181 -> 83,215 ms. Non si selezionano singoli campioni veloci né si sommano le percentuali delle varianti isolate.

#### Esperimenti isolati e risorse

Tutti i confronti seguenti hanno quattro run da 200 step e warmup separato da 60 step; i loro tempi non sono confrontabili direttamente con la serie da 500:

| Variante isolata | Riferimento ms | Variante ms | Decisione |
| --- | ---: | ---: | --- |
| Backward otto pixel, vincolo minimo 8 gruppi | 90,824 | 88,715 | Superato: 23 spill, 80 byte privati |
| Backward otto pixel, vincolo minimo 4 gruppi | 90,940 | 88,721 | Conservato: zero spill |
| Forward otto pixel, staging condiviso da 128 | 91,260 | 92,170 | Scartato |
| Forward otto pixel, staging in registri da 32 | 90,788 | 88,735 | Conservato |

La variante backward senza spill usa 154 VGPR/71 SGPR, il forward 87 VGPR/96 SGPR. Entrambi hanno wave32, gruppo massimo 32, scratch privato zero e zero spill; il forward ha anche LDS zero. La prima variante backward aveva il limite di registri indotto dal vincolo più aggressivo. Eliminare gli spill non ha prodotto da solo un ulteriore guadagno dimostrato rispetto a quella variante: è stato scelto il percorso senza scratch a prestazioni comparabili. Queste risorse statiche non provano da sole un incremento di occupancy.

#### Qualità e regressioni

Nel confronto pulito, MSE finali riferimento 0,0104660488 / 0,0104637243, candidata 0,0104667142 / 0,0104674683. Nella serie esplorativa da 500 step erano invece riferimento 0,0104687815 / 0,0104684523 e candidata 0,0104680233 / 0,0104644252. Su tutti e quattro i run di ciascun tipo la media è rispettivamente 0,0104667517 e 0,0104666577: nessuna pretesa di miglioramento qualitativo o equivalenza esatta.

Le coppie di export iniziali controllate nelle due serie da 500 step sono identiche byte per byte nei PPM a 8 bit. Dopo il training pulito, differenze medie RGB fra varianti 0,7474/0,7165 livelli su 255, fra ripetizioni del riferimento 0,7219/0,7344 e della candidata 0,7370/0,7316. Massimi locali 60/78 contro 77/83 e 82/94. Ispezionate le due viste e crop nativi senza peggioramenti evidenti. Anche il controllo visivo della serie esplorativa è coerente. Restano due viste, SH0, topologia fissa e nessun holdout; l'ordine delle somme atomiche e delle riduzioni FP32 non è deterministico.

Il test raster include ora quattro disposizioni dei segni (uniforme, X, Y, diagonale), due larghezze parziali e sei scale positive/negative: **48 coppie** riferimento/ottimizzato, oltre ai 72 casi analitici originali. Confronta 16 helper più 14 gradienti dei parametri e verifica che i casi saturanti raggiungano davvero quel regime. Prova geometrica su 256 pixel unici e 289 forme di bordo, conservando tutte le quad di clipping. Suite nativa completa: **25/25**, compresi convergenza FP32/fused SH0–SH3, caso denso 1x1, budget/rollback, cancellazione, alias dei parametri, HIP–Vulkan e coesistenza CUDA. Ricompilate entrambe le opzioni OFF: cinque test mirati passati. Ripristinate ON: matematica, raster e convergenza con le variabili di memoria assenti passano (3/3).

Verificati 22 file generati idempotenti per byte/mtime e identici ai compilati. Il corpo generato del precedente backward a quattro pixel e quello del forward originale sono ancora identici ai rispettivi riferimenti; il sorgente host `training.hip` è invariato e privo del vecchio readback diagnostico. Gli altri backend non sono stati modificati da questo ciclo. Nessuna patch di SDK o driver, nessun cambiamento dell'ammissione al Trainer principale.

Artefatti sotto `sdk-sources/audit`: `paired-fulltile-clean-500-analysis.json`, relativa immagine `*-quality.png`, `fulltile-clean-variability.json`, `fulltile-quality-all-500.json`, `fulltile-initial-image-parity.json`, `fulltile-combined-200-variability.json`, `eight-pixel-{spill,relaxed}-analysis.json`, `forward-eight-{shared128,register}-analysis.json`, `fulltile-native-suite.log`, `fulltile-options-off-tests.log`, `fulltile-default-session-tests.log`, `fulltile-config-guards.json`, `fulltile-final-{forward,backward}.json`, `eight-pixel-layout-proof.json`, `fulltile-final-generator-validation.json`. Conservati anche sorgenti e binari degli esperimenti scartati.

Build locale corrente: `build-probes-fulltile-combined.cmd`. Gli script precedenti mantenuti forzano esplicitamente le nuove opzioni OFF; quello del solo backward forza il nuovo forward OFF. Tutto locale e opt-in, nessun push/PR. Nessun benchmark, query dei contatori o compilazione di questo ciclo rimane in esecuzione.

Possibile prossimo studio, non implementato né misurato: i payload proiettati sono ancora tre array separati da 16 byte (`PackedMeanBBox`, conica/opacità, colore). Un layout privato interleavato da 64 byte potrebbe ridurre letture sparse, al costo teorico di circa 16 byte per elemento visibile più allineamento. Andrebbero preservati gli indici e i pareggi del sort, il layout dei Tensor condivisi e tutti i consumatori, compreso il colore nel preprocess backward. Prima di giudicarlo servono test di stride/padding su più righe e dimensioni non multiple, budget/rollback, confronto numerico multi-gaussiana e misure contro la build a tile intero; non si presume che sia più veloce.


## 2026-10-19 — Layout interlacciato dei dati proiettati: esperimento scartato

Verifica successiva alla versione accettata con forward/backward RGB a otto pixel per thread. Solo generazione privata HIP: nessuna modifica ai sorgenti CUDA/Metal/Vulkan, al modello o ai parametri di training. Geometria/bbox, conica/opacita e colore accorpati in righe da 64 byte (16 di padding), poi da 48 byte. Ordine, indici, formule e ordinamento delle primitive conservati.

Protocollo: stesso dataset e seed, 1M gaussiane, resize 1920, SH0, due viste 0/41, topologia fissa, 500 step per run, ordine A-B-B-A, primi 10 step esclusi; riscaldamento separato da 300 step per serie. Nessuna compilazione o diagnostica GPU durante le misure. Le misure storiche da 80,42 ms restano valide nel loro confronto; i riferimenti contemporanei qui sono circa 81,4 ms.

| Layout | Riferimento medio | Variante media | Variazione | Decisione |
|---|---:|---:|---:|---|
| 64 byte | 81.411 ms | 83.932 ms | +3.097% | Scartato |
| 48 byte | 81.403 ms | 82.703 ms | +1.597% | Scartato |

Layout 64 byte, run A-B-B-A: 81.300382, 84.207633, 83.656337, 81.521513 ms. MSE finali, stesso ordine: 0.0104637185701, 0.0104617196927, 0.0104665740414, 0.0104646173913.

Layout 48 byte, run A-B-B-A: 81.295482, 82.686732, 82.719399, 81.510491 ms. MSE finali, stesso ordine: 0.010471740809, 0.0104663278135, 0.0104661917585, 0.0104618248888.

La variante da 64 byte passa da circa 607,885/769,786 MB (live/picco) a 622,056/783,958 MB: circa +14,172 MB. La variante da 48 byte mantiene il footprint precedente, entro poche centinaia di byte di variazione. Sono byte contabilizzati dalla sessione, non una misura della residenza fisica VRAM.

Validazione sperimentale 64 byte: suite nativa 25/25 e sessione con PHASE_HELPERS/TRIM_UNUSED/ARENA_GROWTH attivi 1/1. Variante 48 byte: matematica, raster e sessione con ottimizzazioni di memoria 3/3. Aggiunto nel prototipo un controllo GPU con 0, 3 e 257 righe, scritture alternate e confronto byte completo: righe non scritte, padding e buffer confinanti devono restare intatti. Test ripetuto nei due layout.

Qualita: immagini iniziali PPM a 8 bit identiche tra i quattro run di ogni serie sulle due viste. Dopo 500 step, differenza media assoluta riferimento-variante 0,750/0,682 livelli (64 byte) e 0,757/0,745 (48 byte), confrontabile con le differenze fra ripetizioni dello stesso modulo. Ispezionati montaggi di entrambe le viste e ritagli centrali a risoluzione nativa: nessun peggioramento evidente. Questi controlli non dimostrano equivalenza numerica universale; due viste SH0 e topologia fissa.

Esito: nessun nuovo guadagno di prestazioni. Entrambi i prototipi rimossi dal codice del branch ripristinando esattamente i quattro file modificati rispetto agli snapshot di inizio esperimento; conservati sorgenti sperimentali, DLL, test e risultati nella cartella locale rocm-tensor-study. La build ripristina il layout precedente e tutte le ottimizzazioni gia accettate, senza introdurre un nuovo flag nel prodotto.

Artefatti locali: `sdk-sources/audit/payload-final-analysis.json`, `payload48-final-analysis.json`, `payload-quality-comparisons.json`, `payload-quality.png`, `payload48-quality.png`; `payload-native-suite.log`, `payload-optimized-session.log`, `payload48-tests.log`; sorgenti in `payload48-source/`, snapshot `payload64-prepare_fast.py`/`payload64-fast_raster.hip`, DLL `fast-payload64.dll`/`fast-payload48.dll`. Prossimi tentativi dovranno riguardare altri colli di bottiglia: questo accorpamento non e una ottimizzazione utile nel carico misurato.

Verifica finale del ripristino: quattro file di codice byte-identici agli snapshot iniziali, 22 file generati idempotenti (byte e timestamp), cache CMake priva del flag sperimentale e matematica/raster/sessione 3/3 passati con ottimizzazioni di memoria attive.


## 2026-10-19 — Pressione sui registri del backward: quattro esperimenti scartati

Profilo preliminare della versione accettata: 40 step, primi 10 esclusi, 1M gaussiane, resize 1920, SH0, viste 0/41. Con sincronizzazioni diagnostiche attive: projection 3,497 ms, count/scan 0,592 ms, binning/sort 19,948 ms, forward 17,884 ms, gradient clear 0,673 ms, loss/gradient 6,544 ms, backward 50,768 ms, Adam 1,465 ms. Sottofasi backward misurate separatamente: blend 43,140 ms, projection backward 6,143 ms. Questi numeri includono il costo del profiling e non sono una scomposizione dei tempi non profilati da 80-86 ms.

Quattro prototipi confinati alla generazione privata HIP; nessun cambiamento ai sorgenti originali CUDA/Metal/Vulkan, al modello o al media ingest. Tutte le opzioni sperimentali erano opt-in. Nessuna modifica al compilatore o SDK installato.

1. Gradienti RGB immutabili degli otto pixel in LDS, righe per canale/lane senza conflitti di banco e accessi esclusivamente della lane proprietaria: nessuna sincronizzazione hardware richiesta. Il compilatore porta i VGPR da 154 a 178, senza spill, con 3072 byte LDS per blocco.

2. Stesso storage LDS ma letture dopo ammissione del contributo e controllo alpha: 179 VGPR, nessuno spill, 3072 byte LDS. Entrambi i tentativi peggiorano il risultato e sono rimossi.

3. Elaborazione sequenziale dei quattro vecchi domini di clipping: una coppia di pixel per volta, un accumulatore temporaneo e un aggregato dei soli gruppi dimostrati sicuri. Il test conservativo per gruppo usa la somma dei moduli dei campi < clamp/256 in tutte le 32 lane: ogni vecchia riduzione resta sotto clamp/8 e la somma dei quattro gruppi sotto clamp/2. NaN/Inf falliscono il confronto stretto. I gruppi non sicuri sono ridotti e clippati separatamente. Questo cambia il raggruppamento di alcune somme FP32; i test numerici sono quindi indispensabili. I VGPR scendono a 142, senza spill o LDS, ma il tempo peggiora.

4. Hint `__builtin_amdgcn_sched_barrier(0)` fra coppie di pixel, senza cambiare le formule o la riduzione del riferimento. Si tratta di regioni per lo scheduling del compilatore, non di una barriera di sincronizzazione fra thread. Fonte primaria: https://llvm.org/docs/AMDGPUUsage.html e https://github.com/llvm/llvm-project/blob/main/llvm/include/llvm/IR/IntrinsicsAMDGPU.td . Rimangono 154 VGPR, nessuno spill o LDS. Il risultato temporale e entro le oscillazioni dei riferimenti, quindi scartato.

Protocollo temporale per ogni prototipo: riscaldamento separato da 60 step, quattro run da 200 step A-B-B-A, primi 10 esclusi, stesso seed e impostazioni del profilo, profiling disattivato. Ogni confronto e seriale, senza compilazioni durante le misure. I valori non vanno confrontati direttamente con il benchmark storico da 500 step, perche lo stato del modello evolve.

| Variante | VGPR | LDS/blocco | Riferimento | Variante | Variazione |
|---|---:|---:|---:|---:|---:|
| LDS eager | 178 | 3072 B | 85.776 ms | 106.352 ms | +23.989% |
| LDS admitted only | 179 | 3072 B | 85.592 ms | 99.241 ms | +15.947% |
| Streamed quads | 142 | 0 B | 85.849 ms | 87.002 ms | +1.343% |
| Scheduler regions | 154 | 0 B | 85.915 ms | 85.738 ms | -0.206% |

Lettura: valori positivi significano piu lenti. Il -0,206% dello scheduling non e un miglioramento convincente: i due riferimenti variano da 85,777 a 86,053 ms, mentre le varianti sono 85,611 e 85,866 ms. Nessun prototipo adottato. Allocazioni globali della sessione sostanzialmente invariate, circa 607,885 MB live e 769,786 MB di picco; il conteggio non misura la residenza fisica VRAM.

Controlli eseguiti su ciascuno dei quattro prototipi: matematica, raster/clipping e FastSession con convergenza e recupero/cancellazione, 3/3 passati, con PHASE_HELPERS/TRIM_UNUSED/ARENA_GROWTH abilitati. I run da 200 step producono immagini e MSE per entrambe le viste; sono esperimenti di screening delle prestazioni, non una prova universale di qualita/convergenza su altri dataset. Nessun confronto lungo aggiuntivo da 500 step, perche non e emerso un vantaggio da validare.

Aggiunto e conservato un test mirato al clipping misto: ciascuno dei quattro domini 8x8 a turno riceve un gradiente enorme, gli altri gradienti piccoli, due larghezze non multiple del tile (17/35), altezza 29, entrambi i segni. Il riferimento senza culling e confrontato con il percorso ottimizzato su tutti i campi di gradiente. Il test controlla anche che sia effettivamente raggiunta la saturazione. Sono 16 nuove coppie, 32 invocazioni; insieme ai casi precedenti il probe raster esegue 200 invocazioni.

Stato finale: helper fast_four_pixels.py e HipFastKernels.cmake ripristinati byte per byte dagli snapshot di inizio esperimento. Tutti gli esperimenti conservati solo come sorgenti/DLL/risultati nella cartella locale rocm-tensor-study. Rimangono nel branch soltanto il nuovo test sul clipping e questo rapporto. Nessuna nuova ottimizzazione del runtime accettata.

Artefatti: `sdk-sources/audit/register-baseline-profile.json`, `register-baseline-backward-subphases.json`, `register-final-comparisons.json`, metadati `register-{lds,lds-lazy,streamed,schedule}-backward.json`; run e log `paired-register-*-screen-*`, `register-*-tests.log`; snapshot `register-*-fast_four_pixels.py` e `register-*-HipFastKernels.cmake`, DLL `fast-register-{lds-eager,lds-lazy,streamed,schedule}.dll`. Il profilo identifica il blend backward come priorita per un prossimo intervento algoritmico; la riduzione dei soli registri non ha dato benefici nel carico misurato.

Conferma finale: build precedente ripristinata; matematica, raster con saturazione mista verificata e sessione 3/3 passati con ottimizzazioni di memoria attive. Due file di codice byte-identici agli snapshot, 22 file generati idempotenti (byte e timestamp), nessun flag sperimentale residuo nella cache CMake. `git diff --check` passa.


## 2026-10-19 — Riduzione distribuita dei gradienti RGB: guadagno confermato

Intervento nella sola generazione privata del backward HIP RGB a otto pixel per thread. I singoli contributi dei pixel, il loro ordine per primitiva, il criterio conservativo di fusione dei quattro vecchi domini e i clamp dei domini non fondibili restano invariati. Con piu di una lane contribuente, una riduzione a farfalla distribuisce fra le lane i 16 campi, riducendo progressivamente i campi mantenuti. La lane finale riceve il campo (lane & 15). Il caso con una sola lane contribuente conserva il codice precedente. Cambia il raggruppamento di somme FP32: non si rivendica identita numerica universale. Nessun passaggio a FP16, nessun nuovo fast-math globale.

Opzione `LFS_HIP_FAST_BUTTERFLY_FIELDS`, default OFF, richiede il backward a otto pixel e la distribuzione delle atomiche gia presenti. Solo la sessione nativa sperimentale a topologia fissa; questa modifica non completa l'integrazione nel Trainer principale. Nessun cambiamento ai backend CUDA, Metal o Vulkan.

Protocollo principale: dataset/seed invariati, 1M gaussiane, resize massimo 1920, SH0, viste 0/41, L1/SSIM 0,2, cache target CPU 66.355.200 byte e budget sessione 2 GiB. Riscaldamento separato da 300 step, quattro run da 500 step A-B-B-A, primi 10 esclusi da ogni run, nessuna compilazione durante le misure, profiling disattivato.

| Misura | Riferimento | Nuova riduzione |
|---|---:|---:|
| Step nativo medio | 80.972107 ms | 75.352106 ms |
| Binding del frame medio | 4.851958 ms | 4.865571 ms |
| Step + binding medio | 85.824065 ms | 80.217677 ms |

Riduzione del tempo dello step: **6,9407%**. Il totale step+binding scende di circa **6,53%**. Il tempo di preparazione iniziale delle immagini e separato da queste misure.

Run da 500 step A-B-B-A: 81.165838, 75.355593, 75.348619, 80.778375 ms. Entrambi i candidati sono vicini a 75,35 ms; nessuno dei 1960 step inclusi supera 150 ms. Lo screening preliminare da quattro run di 200 step (60 di warmup separato) aveva dato 85.995380 -> 79.733520 ms (-7.2816%). I due protocolli non sono confrontabili direttamente come tempi assoluti: cambiano durata ed evoluzione del modello.

Memoria globale allocata invariata: circa 607,885 MB live e 769,786 MB di picco; nessuna nuova allocazione o duplicazione del modello, nessun nuovo LDS. Sono conteggi della sessione, non misure della residenza fisica VRAM. Shader RGB: 154 -> 160 VGPR, 71 -> 75 SGPR, zero spill e zero memoria privata. Il guadagno viene dal lavoro della riduzione; il numero di registri isolato non predice il tempo.

Validazione: suite nativa **25/25** con PHASE_HELPERS/TRIM_UNUSED/ARENA_GROWTH attivi, inclusi convergenza, budget, rollback, cancellazione, SH, alias, modello HIP-Vulkan condiviso e convivenza con CUDA. A opzione disattivata, matematica/raster/FastSession e i due test di alias SH **5/5**. Dopo il ripristino della build migliore, matematica/raster/FastSession senza quelle opzioni ambientali **3/3**.

Nuovo oracolo GPU delle riduzioni: 32 configurazioni (blocchi 32/64/128/512, 8 famiglie di input), 5888 risultati di lane confrontati con somme CPU indipendenti. Include campi distinti, contribuenti sparsi, segni opposti, cancellazioni, decimali, NaN, Inf, subnormali e grandi valori finiti. Verifica tutti i 16 campi e le wave multiple. Il probe raster mantiene i test analitici e di clipping misto e aggiunge due casi 1025x769 con gaussiana isotropa estesa: forzano l'emissione cooperativa con una sola sorgente attiva. Totale finale 202 invocazioni raster, passate sia ON sia OFF. I due casi estesi sono stati aggiunti dopo la suite completa, e poi verificati nuovamente nei controlli finali pertinenti.

Qualita: PPM iniziali a 8 bit byte-identici fra tutti e quattro i run su entrambe le viste. MSE finali A-B-B-A: 0.0104758583845, 0.0104642206253, 0.0104637116595, 0.0104611014709. Media riferimento 0.0104684799277; media candidato 0.0104639661424. Nessuna rivendicazione di miglioramento della qualita.

Differenze medie assolute riferimento-candidato: 0,720865/0,734279 livelli su 255 nelle due viste; riferimento-ripetizione 0,730587/0,729528; candidato-ripetizione 0,706780/0,687503. Ispezionati montaggio delle due viste e ritagli centrali a risoluzione nativa: nessun peggioramento evidente. Limite: due viste SH0, topologia fissa, 500 step, non una verifica holdout o di tutti i dataset.

Prova del perimetro: rispetto al generatore di partenza cambiano solo `hip_fast_wave.hpp` e `kernels_backward.cuh` fra i 22 file generati. Tutti i file forward e i corpi dei backward originali a due/quattro pixel sono byte-identici. A flag OFF anche i 23.284 byte di codice macchina del kernel a otto pixel coincidono con il riferimento (SHA-256 `f2d10c1b35f07cb098a6dcf6024223496f6bcc081ee971bcdc6829dc34922124`).

Altri esperimenti di questo giro:

| Esperimento | Esito sullo step | Decisione |
|---|---:|---|
| Taglio dei test geometrici oltre il prefisso di ogni sottotile | -0.0225% | Scartato |
| Scambi DS_SWIZZLE espliciti fra lane | +4.5350% | Scartato |
| Maschera delle sorgenti attive e soglia collettiva nel binning | +0.5481% | Scartato |

Gli scambi DS_SWIZZLE erano bit-identici nei test rispetto al percorso generico e riducevano i VGPR a 155, ma peggioravano il tempo. Maschere e indici del prototipo cooperativo preservavano ordine delle sorgenti/scritture; anche questo prototipo passa i test, ma non porta un vantaggio nel dataset. Codice e opzioni di questi prototipi rimossi dal branch; rimangono snapshot/DLL/risultati locali e il caso di test esteso.

Verifica separata del lavoro CPU: microbenchmark locale host-only su 33.177.600 byte, stessa classificazione IEEE-754 intera del controllo portabile, quattro ripetizioni in ordine alternato e un prototipo AVX2 con verifica delle feature CPU/OS. Risultato medio 0.566670 ms portabile e 0.481441 ms AVX2; 32896 controlli di offset, dimensioni e rifiuto di Inf/NaN anche nel tail. Non e una misura del binding completo. Il risparmio isolato e circa 0,085 ms e non giustifica ora un nuovo percorso nel runtime: nessuna modifica a training.hip o alle opzioni host. Il resto del binding richiede una misurazione diretta di copia e sincronizzazioni; non si attribuiscono tutti i suoi 4,86 ms alla scansione CPU.

Profilo diagnostico finale: riscaldamento separato e quattro run da 80 step A-B-B-A. La serie mostra deriva anche in fasi byte-identiche: ad esempio il count/scan varia da 0,600 a 1,289 ms. Conservata integralmente, esclusa dall'attribuzione quantitativa dei guadagni per fase. I tempi di profiling includono sincronizzazioni e intervalli host; non sostituiscono il confronto pulito da 500 step. Il riferimento non profilato e stabile e nessuna parte della serie principale e stata scartata selettivamente.

Artefatti: `sdk-sources/audit/butterfly-fields-clean-analysis.json`, `butterfly-fields-screen-analysis.json`, `butterfly-fields-quality.json/.png`, `butterfly-generated-scope.json`, `butterfly-option-off-code-parity.json`, `butterfly-phase-profile-analysis.json`, `host-validation-study.json`; log `butterfly-fields-native-suite.log`, `butterfly-fields-off-tests.log`, `butterfly-fields-final-default-tests.log`; build migliore `build-probes-butterfly-fields.cmd`, DLL di riferimento accettata `fast-butterfly-fields-accepted.dll`. I vecchi script di studio ora forzano esplicitamente OFF il nuovo flag per evitare contaminazione della cache CMake.

Prossime possibilita da studiare, non implementate: costo reale di upload/sincronizzazione dei frame; riduzione esatta dei bit delle chiavi di profondita tramite intervallo intero (offset dal minimo, senza quantizzare), solo se il costo della misura min/max e inferiore al passaggio radix risparmiato e preservando ordinamento stabile, fallback e budget. Projection, scan, loss e Adam mantengono il codice precedente: non si introducono cambiamenti senza un vantaggio verificato.

Verifica finale: quattro sorgenti della variante accettata preservati byte per byte; training.hip invariato; 22 file generati idempotenti per byte/timestamp; flag accettato ON nella build locale e default OFF nel CMake; flag dei prototipi rimossi dalla cache. Suite 25/25, OFF 5/5, configurazione finale senza opzioni ambientali 3/3 confermate dai log.


## 2026-10-19 — Screening successivo e memoria host del frame

Tre prototipi isolati scartati: compressione esatta delle chiavi di profondita con conversioni separate (-0,0574%, rumore), stessa compressione fusa in emissione/estrazione (+1,8284% tempo), riduzione dei soli nove campi RGB (nessun vantaggio iniziale, deriva del riferimento finale; non quantificabile come piccolo guadagno). Ogni build supera matematica/raster/FastSession 3/3; oracoli aggiuntivi per ordine stabile delle chiavi e somme dei nove campi. Nessuna quantizzazione. Codice/opzioni dei prototipi rimossi; sorgenti e runtime ripristinati byte-identici alla variante accettata, artefatti conservati.

Misura host isolata: 33.177.600 byte; RAM ordinaria + validazione + copia/attesa 3,0161 ms, HIP pinned 1,7014 ms, staging esplicito incluso 2,8547 ms. L'allocatore CPU usa pinned CUDA solo se CUDA e disponibile e selezionato o gia attivo; altrimenti usa malloc. Caso di convivenza: sullo stesso buffer temporaneo cudaHostAlloc, quattro blocchi A-B-B-A senza/con/con/senza registrazione HIP mostrano copia+attesa 2,701764 -> 1,135422 ms, dati verificati esatti. Registrazione 1,5422 ms e rilascio 0,9741 ms una tantum. Sono probe di trasferimento; non una misura del guadagno di training. La validazione del probe incrociato scansiona una copia shadow, quindi si confrontano solo copia e attesa.

Nessuna registrazione introdotta nel runtime o negli allocatori. Possibile seguito: registrazione legata al proprietario dell'allocazione CPU, con budget, alias/pagine sovrapposte/riuso del pool, concorrenza, piu sessioni/device, rilascio prima del proprietario CUDA e fallback nel log; non registrare per ogni frame, non rimuovere registrazioni altrui, non memorizzare la validazione dei frame C ABI per solo puntatore. Prima di adottarla servono confronti con i veri owner del runner e controlli dei contratti. Nessun cambiamento ai percorsi CUDA, Metal e Vulkan.

Rapporto locale completo: `.local/rocm-tensor-study/optimization-followup-report.it.md`. Risultati: `sdk-sources/audit/{range-keys-screen-analysis,range-keys-fused-screen-analysis,nine-fields-analysis,frame-transfer-study,frame-cross-vendor-abba,optimization-cycle-final-scope}.json`. Il risultato accettato rimane quello della riduzione a 16 campi, 80,9721 -> 75,3521 ms nel confronto da 500 step precedentemente confermato.


Verifica indipendenza CUDA (2026-10-19): probe autonomo compilato esclusivamente con HIP, allocazione hipHostMalloc/rilascio hipHostFree, trasferimenti e letture di verifica esatti. Con CUDA_VISIBLE_DEVICES=-1 e HIP_VISIBLE_DEVICES=0, il controllo finale GetModuleHandle conferma assenza di nvcuda.dll e cudart64_12.dll. Senza HIP_VISIBLE_DEVICES, HIP interpreta anche CUDA_VISIBLE_DEVICES e nasconde la stessa GPU AMD: il primo tentativo e quindi fallito per visibilita, non per dipendenza CUDA. Nessuna modifica permanente alle variabili di sistema. La serie dei tempi di questo ultimo probe mostra deriva ed e usata solo come prova di funzionamento/correttezza e assenza di moduli NVIDIA, non per un nuovo guadagno. Non equivale a un test dell'intera app su un sistema senza CUDA installato.

Direzione dell'integrazione: allocazione host HIP propria (hipHostMalloc), budget e proprietario espliciti, fallback a RAM ordinaria con log; nessuna dipendenza CUDA. La registrazione di memoria gia esistente e solo un'ottimizzazione facoltativa di interoperabilita, non il percorso richiesto per ROCm. Il pool pinned CUDA e i backend gia esistenti mantengono il comportamento corrente.

Validazione finale dopo rimozione dei tre prototipi: suite nativa 25/25, 122,13 secondi, inclusi budget/rollback, SH, alias, modello HIP-Vulkan condiviso e convivenza CUDA. Quattro sorgenti della variante accettata e training.hip byte-identici alle ancore; opzioni RANGE_KEYS e NINE_FIELDS eliminate dalla cache. Nessun processo di test o compilazione di questo giro rimane attivo.


## 2026-10-19 — Probe diretti di driver, SDK e matematica

Solo eseguibili di prova locali; nessuna nuova modifica a kernel, API, allocatori o backend. Runtime ordinario di sistema 70260201 e runtime nel pacchetto SDK 71460850: entrambi riconoscono gfx1150, host mapping supportato e managed/pageable access non supportato. Confronto SDK isolato senza installazioni o cambi permanenti. Binari identici, ma serie A-B-B-A con deriva dei riferimenti: nessun guadagno quantitativo attribuito al runtime nuovo. Test di sessione gia compilato passato anche nel confronto SDK (SH0..3, FP32/joint-fused, budget/rollback/cancellazione e scratch), senza adottarlo nel progetto.

Matematica: 262.144 campioni per famiglia, sei famiglie, oracoli CPU e casi speciali. Exp, reciproco e sigmoid hanno codice macchina byte-identico fra percorso corrente e builtin espliciti; nessuna accelerazione possibile riscrivendoli cosi. Rsqrt protetto migliora il loop sintetico ma riguarda soprattutto SH superiori, non e un guadagno di training SH0. Log protetto non migliora; sqrt esplicito piu lento. Oggetti reali forward/backward senza istruzioni FP64 accidentali, atomiche gradienti FP32 hardware e dual issue gia emesso.

Fattorizzazione della covarianza: meno operazioni ma arrotondamenti diversi, culling differente nei casi anisotropi estremi anche con Jacobiani pinhole; non adottata. Atomiche sintetiche: stride 9 migliore su indici contigui, stride 16 corrente migliore sugli indici dispersi; serve la distribuzione reale prima di modificare il layout.

Kernel L1/SSIM reale non modificato, target 1440x1920: hipHostMallocNonCoherent mappato 6,290693 ms contro copia+loss 6,864074 ms nello screening; tutti i 24 confronti finali di gradienti/mappa bit-identici nel probe. Memoria mappata default/coherent invece peggiora. Non e una misura dell'intera iterazione o della qualita del training. Il contratto C ABI borrowed non permette di introdurre alias persistenti impliciti: serve proprieta esplicita. Primo passo conservativo: target host HIP proprio, budget e fallback a RAM ordinaria con log, mantenendo la copia attuale e senza dipendere da CUDA.

Rapporto completo locale: `.local/rocm-tensor-study/driver-sdk-math-report.it.md`, con log, limiti, confronto fra runtime e prova bytewise del perimetro. Nessuna patch SDK/driver giustificata per ora. Nessuna ricompilazione del progetto richiesta da questi esperimenti.


## 2026-10-19 — Test rsqrt sulle funzioni SH reali

SH0 di controllo e SH1..3, 32.768 primitive, coefficienti rest nonnulli, layout swizzled: colore, gradienti posizione/SH0 e dei coefficienti contro riferimento e oracoli CPU double/differenze finite. Helper generico protetto: piccoli arrotondamenti nel calcolo composto, non adottato. Specializzazione limitata ai due punti con soglie gia presenti: tutti i 7.471.104 componenti del probe coincidono bitwise nei casi ordinari/speciali/vicini alle soglie, ripetuto in due processi. Errori massimi SH3 vs CPU: colore 2,277e-7, posizione 1,070e-7, coefficienti 2,673e-7.

Nessun guadagno robusto: throughput wall-clock del solo gradiente posizione SH1 +1,79%, SH2 -0,05%, SH3 +0,98%; non sono tempi del training. Forward e gradienti coefficienti hanno codice macchina byte-identico, il gradiente posizione elimina solo 28/32 byte. Il -32% scalare non si trasferisce al carico SH reale. Prototipo conservato solo negli artefatti, nessuna nuova opzione o modifica ai kernel del branch.

Le serie a eventi GPU interleaved sono escluse: osservati tempi negativi. Riproduttore minimo indipendente con stream esplicito: 2 negativi su 2048 coppie sia runtime 70260201 sia 71460850, query hipSuccess, non riparati da sincronizzazione esplicita. Causa nel percorso runtime/driver ancora da identificare; nessuna patch o issue pubblicata. Le misure step_ms del runner sono chrono host e non dipendono da questi timestamp; anche il confronto finale SH usa timer host e sincronizzazione.

Rapporto locale: `.local/rocm-tensor-study/sh-rsqrt-validation.it.md`; dati e SHA-256 in `sdk-sources/audit/sh-rsqrt-*.json`, riproduttore `hip-event-order-probe.hip`. Sorgenti accettati e training.hip byte-preservati; CUDA/Metal/Vulkan invariati.


## 2026-10-19 — Immagini CPU con proprietario HIP (opt-in)

Il runner sperimentale a topologia fissa supporta `LFS_HIP_HOST_TARGETS=1`: CameraFrame conserva la sua copia immutabile in un'allocazione propria hipHostMallocDefault, tramite proprietario condiviso e modulo mantenuto caricato fino al rilascio. Nessuna dipendenza CUDA aggiunta. Budget pinned separato, massimo 512 MiB, riducibile con `LFS_HIP_HOST_TARGET_BUDGET` (byte); nessuna seconda copia persistente della cache CPU. Set_frame conserva la copia sincrona al buffer GPU e la validazione corrente: nessun alias host persistente implicito.

Fallback a RAM ordinaria con motivo CLI per budget insufficiente, allocazione fallita o modulo precedente senza simboli opzionali. Budget malformato gestito come errore dell'opzione facoltativa. Il test di errore nativo ha rilevato e fatto correggere lo slot last-error HIP residuo che poteva contaminare un kernel successivo; ora verificato nello stesso processo prima di una loss valida.

Suite nativa definitiva 26/26 (121,03 s), inclusi proprieta/copie/lifetime del modulo, budget/riuso, quattro thread, frame NaN rifiutato senza danneggiare il frame preparato, loss valida dopo il rilascio del buffer host. Verificati anche training reali di fallback: budget di un byte, vecchio modulo butterfly e budget malformato. Kernel, matematica, set_frame e optimizer byte-preservati rispetto allo snapshot prima della prova; CUDA/Metal/Vulkan non modificati da questo intervento.

Screening ABBA a 200 passi: caricamento frame medio 4,7572 -> 3,1898 ms; tempo totale step+frame 90,2249 -> 87,1663 ms. Questo screening precede il monitor continuo delle compilazioni e rimane preliminare. Due successivi riscaldamenti contaminati da compilazioni esterne sono stati esclusi; la conferma richiede 30 secondi iniziali senza compilatori e monitor a 250 ms durante ogni esecuzione.

Rapporto e prove locali: `.local/rocm-tensor-study/host-target-training-report.it.md`, `host-target-final-suite.log`, `host-target-scope-audit.json`, `host-target-fallback-results.json`. Non e ancora il trainer applicativo completo; il mapping diretto nella loss resta separato e non implementato.

Conferma conclusa, serie `host-target-confirm-clean2`: quattro training da 500 passi ABBA, due viste 1920, 1M/SH0, tutti senza compilatori rilevati. Frame 5,2052 -> 3,3226 ms (-36,17%); step 75,4096 -> 75,9514 ms (+0,72%, oscillazioni); totale 80,6148 -> 79,2740 ms (-1,66%). MSE medio finale 0,0104681250 -> 0,0104699783 (+0,0177% relativo), differenze dei render dello stesso ordine delle ripetizioni e nessun peggioramento visivo strutturale osservato nelle due viste e nei dettagli; nessuna identita bitwise del training dichiarata. Cache CPU 66.355.200 byte in entrambi i casi; picco GPU contabilizzato identico 769.785.936 byte, non una misura completa del driver/processo. Percorso conservato opt-in, modulo locale `fast-butterfly-host-targets-accepted.dll`. Dettagli e limiti in `host-target-training-report.it.md`, analisi `host-target-confirm-analysis.json`, immagini `host-target-confirm-quality-{full,detail}.png`.


## Ripetizioni delle nuove organizzazioni globali dei tile, con controllo delle interferenze

Esperimenti nuovi rispetto al registro precedente: cambiano l'assegnazione globale dei tile, non le formule per pixel o le precedenti riduzioni intrawave. Prima prova: quattro classi di parita, per distanziare i tile contemporanei. Seconda: macroblocchi 4x4, per favorire localita. Terza: quattro tile indipendenti in un workgroup di 128 thread, conservando una wave32 e otto pixel per lane per ciascun tile. Nessuna LDS/barriera fra le wave nel kernel privato; guard delle wave finali prima di leggere i range. I tre prototipi sono soltanto negli artefatti locali e **non sono adottati**.

Copertura delle permutazioni: stesso corpo intero verificato su CPU (70.480.909 visite) e GPU (121 griglie, 112.896 visite, identita CPU/GPU); griglie dispari, strette, bordi. Packing: 4.101 dimensioni, 8.608.071 visite e 256 pixel unici per tile. Ogni variante ha passato matematica, raster/clipping e FastSession SH0..3 (3/3).

La prima serie pack4 aveva riferimenti 77,97 e 86,11 ms, deriva 10,43%: non e una misura affidabile. La prima serie macro4 mostrava -0,65%, troppo piccolo rispetto alla variabilita. L'utente ha richiesto la ripetizione per possibili lavori concorrenti. Il controllo ha rilevato realmente build FFmpeg/AV1 Windows (cl/CMake/Ninja/link), build Linux in Docker/WSL, ctest, script Python, compressione 7z e carico Defender. Non si attribuisce retroattivamente ogni vecchia oscillazione a una causa specifica.

Ripetizione finale: `tile-repeat-background-v3`, stessi quattro binari archiviati verificati SHA-256, due viste 0/41 del dataset statua_budda_villaggio, 1M gaussiane, SH0/fused, massimo 1920, L1/SSIM 0,2; RAM host HIP in entrambe le varianti. Riscaldamento separato di 60 passi, come nello screening originale; ciascun confronto usa quattro esecuzioni da 200 passi in ordine A-B-B-A, primi dieci esclusi. I 300 passi di warmup tentati in v2 sono stati esclusi per sovrapposizioni reali e non entrano nei risultati.

Monitor a circa un secondo: nomi di compilatori/test, script Python, CPU per processo (inclusa WSL) e motori GPU per PID tramite contatori Windows PDH. Prima dell'avvio si richiedono 30 secondi senza i carichi rilevati. Le esecuzioni con conflitti sono conservate, escluse e ripetute; il monitor deve vedere la nostra GPU in ogni campione della finestra misurata, senza errori o lacune oltre tre secondi. Dati GPU temporaneamente non disponibili fuori dalla finestra sono espliciti nella telemetria. Limite dichiarato prima dei confronti: deriva massima 3% fra i riferimenti, altrimenti ripetizione dell'intero gruppo. Il monitor e passivo ma non gratuito: costo CPU registrato per ogni campione; stesso controllo in tutte le varianti. Non equivale a isolamento completo del sistema o a una prova statistica per differenze sotto l'1%.

Sono state accettate 12 esecuzioni, scartate 6 nella serie finale; tutte le finestre misurate accettate hanno copertura GPU completa e nessun conflitto rilevato. Le esecuzioni valide seguono l'ordine ABBA; fra esse possono esserci tentativi scartati e attese di quiete, documentati nei log, quindi non si descrive la serie come ininterrotta.

| Variante | Riferimento step ms | Candidato step ms | Variazione | Deriva riferimenti |
|---|---:|---:|---:|---:|
| Ordine globale a quattro parita | 77.815206 | 78.652347 | +1.076% | 1.791% |
| Ordine globale in macroblocchi 4x4 | 77.220376 | 77.824554 | +0.782% | 0.173% |
| Quattro tile indipendenti per workgroup | 77.521736 | 83.167962 | +7.283% | 0.470% |

Valori positivi indicano maggiore lentezza. I risultati piccoli di phase2/macro4 non dimostrano una regressione universale, ma non offrono un beneficio da adottare; il -0,65% precedente di macro4 non si conferma. Pack4 e sfavorevole anche nella ripetizione: circa +7,28%. La query del runtime sui kernel reali stima 24 wave32 attive per multiprocessore nel riferimento e 20 in pack4; metadati 160 -> 167 VGPR, 75 -> 68 SGPR, nessuno spill o scratch privato. Sono stime e risorse statiche, non contatori dell'occupancy effettivamente osservata: non provano da sole la causa del rallentamento.

Target PPM identici per SHA-256 all'interno di ogni confronto. MSE finali e dati originali conservati integralmente nel JSON; niente garanzia di identita bitwise del training o su holdout. Non e stato necessario un nuovo confronto lungo da 500 passi per adottare una variante: nessuna ha mostrato un guadagno credibile. I tempi sono chrono host del runner, non eventi HIP/PDH. Sono tempi di step; non includono preparazione iniziale, binding immagini o esportazione e non vanno confrontati direttamente con i precedenti 500 passi.

Entrambi i generatori sono ripristinati byte per byte agli snapshot accettati. Nessun nuovo flag o modifica runtime adottata, nessuna patch SDK/driver, nessuna modifica CUDA/Metal/Vulkan in questo giro. La selezione di immagini host HIP precedentemente accettata rimane opt-in. Modello/trainer applicativo completo e refinement restano fuori dal runner sperimentale.

Prove locali: `.local/rocm-tensor-study/tile-repeat-final-analysis.json`, `tile-repeat-background-v3-{manifest,results,comparisons}.json`, telemetrie `*.telemetry.json`, `tile-repeat-restoration-and-binaries.json`, `tile-pack4-{coverage,safety-audit,backward}.json`, `tile-pack-occupancy.log`, log di test `tile-order-tests.log`, `tile-order-macro4-tests.log`, `tile-pack4-tests.log`. I confronti iniziali e i tentativi v1/v2 rimangono disponibili e non sono mischiati alla ripetizione finale. API di riferimento per i campioni wildcard: https://learn.microsoft.com/en-us/windows/win32/api/pdh/nf-pdh-pdhgetformattedcounterarrayw .

Verifica finale del ripristino: matematica, raster/clipping, FastSession SH0..3/convergenza/cancellazione e OwnedHostTargets **4/4 passati**. I due generatori coincidono byte per byte con gli snapshot accettati; anche l'ELF GPU backward ricostruito e byte-identico all'immagine accettata prima degli esperimenti. Prove: `tile-repeat-restored-validation.json`, `tile-repeat-restored-build.log`, `tile-repeat-restored-tests.log`. Nessuna nuova ottimizzazione adottata.


## Verifica a 2.000 passi e ripresa integrazione — 20 ottobre 2026

Nuova esecuzione HIP valida, 1M fisso, SH0, 1440x1920, due viste 0/41,
seed dal checkpoint funzionale `final-adam-training-cuda/project.licht`, Adam
azzerato. Finestra 201–1900: step 65,764643 ms, frame 2,646883 ms, somma
68,411526 ms; mediana step 65,0292, p95 73,0267. Totali delle 2.000
iterazioni: 134,198585 s di step e 5,312534 s di bind, esclusi preparazione,
render di controllo ed export. Nessun conflitto rilevato nella prova accettata;
primo preriscaldamento contaminato conservato ed escluso.

MSE training 0,0736427444693 → 0,00655564343391. Controllo visivo di entrambe
le viste: statua e basamento ricostruiti nettamente meglio del seed, vegetazione
ancora sfumata e striata. Sono due viste usate per ottimizzare: non è prova di
qualità multi-vista, generalizzazione o parità con il trainer applicativo.
Picco sessione 769.785.168 byte, target host 66.355.200 byte; non confondere
contabilità sessione con residenza fisica complessiva. Nessun fallback ingest.

Confronti storici, finestre e metriche originali sono conservati in
`.local/rocm-tensor-study/backend-2000-comparison-oct20.it.md` e nel JSON
omonimo. CUDA e Mac non sono stati rieseguiti per performance. Il controllo
AMD Vulkan successivo da checkpoint misura 114,591 ms di step + 9,936 ms di
loader: non usare il riferimento precedente 229,98 ms per rappresentare lo
stato più ottimizzato. Protocolli e seed differiscono, quindi nessun rapporto
HIP/Vulkan viene dichiarato un confronto equivalente di training completo.

Primo incremento per integrazione: `HipBackendOps::dot_product`, `diag` e
`eye` sono ora nativi tramite i primitivi HIP esistenti. Dot usa temporaneo
GPU con ciclo di vita protetto e riduzione sulla stessa coda; è un percorso
corretto di base, non una nuova ottimizzazione fused. Nessun trasferimento CPU,
nessun fallback ad altri backend. Diag/eye controllano estensioni complete
prima delle scritture; diag rifiuta overlap. Contratti reali passati per offset,
padding, matrice rettangolare, identità vuota, overflow, memoria di altro
backend, errore senza modifica output. Il controllo di operazioni ancora
non implementate usa ora SGEMM, invece del dot appena portato.

Verifiche: build MSVC lfs_core riuscita, Tensor smoke con CUDA nascosta e
con CUDA visibile riusciti (inclusi controlli CUDA di correttezza prima/dopo
scope HIP); contratto backend isolato passato; ponte modello/camera/sessione
400 step passato; admission del trainer incompleto ancora respinta senza
inizializzare altri backend. Log `integration-matrix-*.log`. Nessun kernel
CUDA/Metal/Vulkan modificato da questo incremento. Il kernel HIP del benchmark
precede queste aggiunte e non ne dipende.

### Lavoro applicativo ancora necessario, in ordine di dipendenza

1. Completare i contratti Tensor richiesti dal trainer e dalle strategie:
   scritture indicizzate/maschere, multinomial, riduzioni fuse, sincronizzazione
   e readback; aggiungere test su capacità oltre il conteggio vivo e owner.
2. Esporre vere famiglie `TrainingOps` HIP (oggi tabella vuota intenzionale):
   Fast/Gsplat, photometric, Adam, SH, TrainingImage, SharedImage; per MCMC
   anche Mcmc e Refine. Non aggirare `required_training_families`.
3. Collegare crescita/prune/relocation con conservazione di momenti e codec,
   generazioni del modello, capacity e budget; poi restore completo checkpoint
   con stato Adam e scheduling. Oggi un PLY non è un checkpoint riprendibile.
4. Collegare ingest/cache asincroni applicativi e tutte le immagini con budget
   RAM, invalidazione, cancellazione e fallback visibile nel log. Il limite
   sperimentale 512 MiB/due viste non costituisce ingest applicativo completo.
5. Collegare storage condiviso al viewer Vulkan sulla stessa scheda e copie
   esplicite su schede diverse, con owner/sync, resize/growth e recovery.
6. Preferenze opt-in, rilevamento device e indicatore riavvio coerenti con il
   backend realmente attivo. Ammissione solo per capacità effettivamente coperte.
7. Ripetere infine 2.000 step con stesso seed, split, MCMC/refinement, stato
   iniziale, cache, schedule e valutatore del trainer; controllare qualità e
   viewer, CUDA e Mac. Soltanto allora formulare una classifica diretta.

Questa lista resta aperta: le tre operazioni aggiunte non equivalgono a piena
integrazione, e non autorizzano a promuovere HIP nelle preferenze o nei default.


## Integrazione applicativa in corso — blocco I/O verificato (20 ottobre)

Richiesta esplicita: completare il collegamento al trainer, non promuovere il
runner fisso come training applicativo. Il benchmark a due viste resta separato.

Collegati e verificati nelle API reali `lfs_core`:
- `TensorUpload`: copia CPU immediata in staging HIP pinned, trasferimento
  asincrono sulla coda HIP, owner fino al completamento, riuso/crescita e guardia
  dello slot occupato. La modifica della sorgente CPU/span dopo enqueue non
  altera il trasferimento.
- `TensorReadback`: snapshot, offset e sorgenti strided, riuso dello staging,
  destinazione preparata e durata del contesto/modulo oltre il facade.
- `TensorCompletion`: marker HIP di completamento, ready ripetibile e obblighi
  backend-wide. Prima il capture di un Tensor HIP non registrava il lavoro:
  questa omissione applicativa è ora corretta.
- `TensorFence`/`TensorWorkQueue`: marker riutilizzabili, attese sulla stessa
  coda senza attesa CPU al record, callback host native HIP e ticket execute.
  Le code logiche HIP sono serializzate su una coda nativa: non dichiarare
  parallelismo fra code. Gli handle CUDA/target estranei sono rifiutati.
- `TensorReadbackRing`: slot host pinned HIP, append di più range con offset,
  padding, seal, riuso e staging GPU solo se richiesto esplicitamente; nessuna
  seconda copia del modello GPU per il normale readback.
- Shutdown rifiutato con allocazioni, queue lease, trasferimenti o fence vivi.
  Gli errori di teardown mantengono caricati codice e risorse ancora in uso.

Prove passate, CUDA nascosta e HIP device 0: contratto backend nativo completo
(1,52 s), smoke applicativo con tutte le API precedenti, contratto ABI di
readback, training core/private 400 step (tutte le alias byte-exact, righe
riservate intatte), admission incompleta ancora rifiutata. Ripetuto anche il
controllo di correttezza CUDA prima/dopo lo scope HIP, con CUDA visibile: passa.
Nessun benchmark CUDA ripetuto. Log `integration-pipeline-*.log`; snapshot
verificato dei 12 file I/O in `integration-io-accepted-sources-oct20` e manifest
SHA256 omonimo. Le successive aggiunte FastGS non fanno parte di questo snapshot.

In lavorazione: C ABI FastGS applicativa con queue lease del Tensor runtime,
parametri/momenti del trainer presi in prestito e workspace nativo bounded.
`LFS_HIP_APPLICATION_FAST` registra solo questa famiglia nei build che abilitano
esplicitamente Tensor HIP, training runtime e Fast runtime. La tabella resta
incompleta: non abilita ancora un training principale né preferenze/default.
La compilazione/confronto del nuovo FastGS applicativo è ancora da verificare;
non confondere il codice aggiunto con un risultato di porting già superato.


### Application TrainingOps progress, 20 October: five families verified

The real HIP application table now has opt-in implementations of FastGS,
MCMC, refinement, photometric loss and standalone Adam. This does not admit
the complete application trainer: SH storage, image/ingest families, session
and viewer/shared storage still need integration and acceptance. No complete
2,000-step comparison is claimed from these family tests.

All five application contracts pass sequentially on HIP 0 with
HIP_VISIBLE_DEVICES=0 and CUDA_VISIBLE_DEVICES=-1. Fast forward is identical
to the validated private renderer; one-step fused Adam max error 1.1920929e-7.
Standalone Adam CPU update max error over three updates is 6.15e-8; frozen/crop
rows, FP32/F16/Q16 SH and 8/16-bit row/swizzle moment resets preserve unrelated
blocks and padding. All six photometric loss paths agree with an independent
double-precision 2-D CPU reference (max loss delta 5.77e-8); sampled finite-
difference gradient max absolute delta 1.2e-6, with unchanged test tolerances.
Byte targets/Bool masks, fixed/automatic normalization, empty masks, metric/
error maps and workspace reuse/reset pass. MCMC sampling is deterministic
for fixed HIP seed, first-row frequency .24925 vs .25 expected; copy/update,
noise statistics, frozen rows and fold/reset pass. Refinement's eight ops pass,
including exact positive median vs sorted CPU reference over 10,007 values,
counts, clipping, scores, split, slot reuse, pruning and teardown.

Core native code modules are retained until the Tensor context has drained
its queue; the module cache is weak and cannot retain the context cyclically.
GPU I/O and the new typed fills/clamp stay HIP-specific. Existing CUDA,
Metal and Vulkan operation implementations have not been edited. This is
source isolation evidence, not a new full macOS regression run.

Evidence: application-{fast,mcmc,refine,photo,adam}-contract.log,
application-families-host-build.log, application-adam-native-build.log.
SHA256 manifest: application-five-families-accepted-oct20.json.


### Integrazione applicativa, 20 ottobre: sette famiglie verificate

Sono ora collegati e verificati FastGS, MCMC, refinement, loss fotometrica,
Adam, tutte le operazioni SH e sessione. I contratti girano sequenzialmente
su HIP 0 con CUDA nascosta; non costituiscono ancora un training applicativo
completo. Il gate di ammissione resta chiuso alle famiglie mancanti.

SH passa per rest=3/8/15: swizzle, gather float/byte, scatter, azzeramento,
chunk Q16 con sentinelle, decode FP32/F16/Q16 a intervalli, rounding host
identico per serializzazione Q16, compattazione di tre run, aggiornamento
dei soli blocchi toccati con ordine canonico indiretto, copia diretta in
arena senza duplicare la concatenazione. Le viste raw HIP accettano solo
intervalli di allocazioni Tensor vive e registrate; il proprietario dell'arena
deve vivere almeno quanto la vista, come per i raw-view esistenti.

La sessione usa workspace FastSaved privati: non presta memoria di un frame
in uso. Gli scratch opzionali mantengono il percorso Tensor, analogamente
a Metal. Verificati discovery senza inizializzazione, memoria HIP reale,
controllo StreamQuery senza sincronizzazione della coda sana, rifiuto target
estranei, timeout, trim con Tensor vivi e shutdown. Il profiling hook non
avvia una cattura SDK; i benchmark applicativi usano i normali timer host.

Prossimi passaggi necessari: TrainingImage e SharedImage/ingest, funzioni
ausiliarie richieste dalla configurazione, storage condiviso/importazioni
HIP-Vulkan, preferenze/discovery, build trainer completa e validazione
checkpoint/crescita/pruning/resume. Solo dopo: confronto a 2.000 step con
le impostazioni comuni e verifica qualitativa, senza riciclare il benchmark
privato a topologia fissa come confronto equivalente.

Snapshot completo dei file verificati: application-seven-families-accepted-
sources-oct20; manifest SHA256: application-seven-families-accepted-oct20.json.
Log aggiuntivi: application-sh-contract.log e application-session-contract.log.


### 20 ottobre: ottava famiglia applicativa verificata

TrainingImage ora usa la coda e le allocazioni Tensor HIP per EMA delle camere,
ROI, sfondi, Canny, normalizzazione e quantizzazione. Contratto GPU con CUDA
nascosto: EMA/hash/quantizzazione esatti; resize contro riferimento CPU double
con errore massimo 2.41964841e-7; Canny contro convoluzione/NMS CPU indipendente
con errore massimo 4.66818609e-7 su dimensioni 35x37, oltre ai controlli UInt8,
maschera nulla, crop traslato/inverso e rifiuto di input CPU. Tutti gli otto
contratti applicativi ripetuti in sequenza passano con lo stesso runtime.

La famiglia SharedImage del core rimane non registrata: sono in implementazione
le conversioni native, prima di completare resize Lanczos e undistortion. Il
trainer completo rimane correttamente non ammesso. Nessun nuovo dato di velocità
end-to-end deriva da questi contratti; il confronto equivalente a 2000 step resta
da eseguire dopo il collegamento effettivo del trainer.


### Primitive native per ingest: verifica separata

Nuovo seam nel runtime Tensor del core: sedici conversioni ImageConversion,
sentinelle di decodifica, split RGBA float/byte e tre trasformazioni maschere.
I kernel di formato derivano dai sorgenti esistenti, senza modificarli. Primo
contratto nativo: sedici conversioni contro riferimenti CPU, tutti i 65536 alpha
UInt16 con divisione arrotondata esatta, sentinelle mutate/non mutate, split
RGB/alpha, normali sRGB ruotate e rifiuto di overlap/contesto estraneo/disallineamento.
La prima verifica precede l'aggiunta della terza trasformazione maschere; il
contratto esteso e il ponte Core Tensor vengono verificati a parte. La tabella
SharedImage pubblica non viene dichiarata disponibile finche' resize e lens
non sono completi: questi risultati non equivalgono ancora al loader integrato.


Verifica estesa completata: anche quantizzazione maschere a byte-grid e ponte
tra il seam di conversione e veri Tensor Core HIP passano. Ripetuti con il nuovo
core tutti gli otto contratti applicativi e il contratto runtime di ownership,
queue/readback e warp: tutti passano, CUDA nascosto. Questo non registra ancora
SharedImage e non rende disponibile il trainer completo. Risultati in
application-image-contract.log e shared-image-runtime-contract.log.


### Resize condiviso: cinque percorsi nativi e ponte Tensor verificati

Portate le funzioni device Lanczos RGB HWC (1-4 canali, byte/float), grayscale,
Float32 CHW e resize validity-aware per depth/normal. Workspace coefficienti
espliciti nelle allocazioni Tensor Core, senza allocator GPU esterno. Verificati
up/down, canali/layout, prior non validi, alias e stride insufficienti. I valori
CPU dei coefficienti seguono la specifica FP32 esistente; la convoluzione CPU
rimane indipendente in double. Un primo confronto interamente double aveva
esposto la sensibilita' numerica della normalizzazione in ingrandimento ~5x:
corretto il riferimento al contratto FP32, non aumentata la tolleranza 2e-5.
Errore massimo 1.51839911e-6. Verificati anche HWC/CHW e depth sui Tensor reali.
Tutti gli otto contratti applicativi ripetuti dopo il nuovo core passano.

SharedImage resta non registrata: manca ancora la correzione ottica completa,
oltre alle successive famiglie/configurazioni, interop/viewer e trainer finale.
Nessun confronto end-to-end nuovo a 2000 step e nessuna dichiarazione di
integrazione completa deriva da questi test. Log: shared-resize-runtime-contract.log.


### Collegamento applicativo: SharedImage ed ExtraLoss verificati

Registrata la tabella HIP SharedImage completa. Contratto reale: tutte le sedici
conversioni, sentinelle, cinque modelli ottici, immagini UInt8/Float32 a 1/3/4
canali, identita' esatta, maschere/depth/normal su piu' batch di righe, cinque
resize, rifiuto di input estranei e teardown passano. Nessun fallback CPU nel
percorso HIP; i confronti CPU sono esclusivamente nel test.

Registrate anche regolarizzazioni scale/opacity e backward ADMM. Riferimenti CPU
su 263, 10007 e 300001 elementi, riduzione grid-stride, accumulo/overwrite,
value-only, peso nullo e scratch insufficiente passano. Massimo errore loss
3.9929419e-8; gradiente 7.44256656e-9. Sono dieci famiglie applicative verificate,
non un'ammissione del trainer completo. Avviata build del trainer reale in
build-application con CUDA disabilitato, runtime HIP opt-in e due compilazioni.
Morton e codec SH Core sono in preparazione e non ancora verificati/registrati.

### Integrazione applicativa: undici famiglie verificate, trainer end-to-end ancora in verifica

La tabella HIP collega ora Fast, MCMC, Refine, Photometric, Adam, SH, Session, TrainingImage, SharedImage, ExtraLoss e Morton. Tutti gli undici contratti sono stati ricompilati ed eseguiti con lo stesso runtime nativo e CUDA nascosta. Anche il codec SH del core passa i confronti CPU per SH1/SH2/SH3, FP32/F16/Q16, gather/scatter, offset, alias, padding ed espansione SH0.

Morton verifica chiavi intere a 30 bit, stabilita dei pareggi e bit bassi, momenti compressi a 8/16 bit, uguaglianza bit per bit fra gruppi e buffer completo, permutazione del vero SplatData e copie da intervalli nativi posseduti dal core. L'errore massimo osservato nel riordino SH Q16 e 2.2649765e-6 (entro il limite della quantizzazione); FP32/F16 seguono il riferimento CPU esatto. Il test ha inoltre individuato un dispatch globale errato negli helper condivisi: nel solo percorso HIP ora viene rispettato il contesto di esecuzione HIP esplicito.

Il rilevamento hardware usa le entry point di inventario del modulo, senza creare allocatore o contesto Tensor; disponibilita e conteggio ripetuti passano il controllo di assenza di effetti sul backend configurato. La configurazione HIP esplicita e consentita nelle build opt-in; i candidati automatici restano invariati. Il controllo delle famiglie mancanti resta obbligatorio.

La prima build completa della libreria lfs_training con LFS_ENABLE_CUDA=OFF e riuscita. E stato aggiunto lfs_hip_application_trainer, che usa loadTrainingDataIntoScene, initializeTrainingModel e il vero Trainer::train: il vecchio runner a topologia fissa non viene usato per dichiarare completata l'integrazione. Le prove end-to-end di questo nuovo eseguibile sono ancora in corso: non sono ancora dimostrati checkpoint/ripresa, interop viewer, tutti i moduli opzionali o il confronto equivalente a 2000 step. Nessun nuovo tempo comparativo viene dichiarato su questa base.
### Pipeline reale del trainer: prime prove superate

Il nuovo eseguibile di verifica lfs_hip_application_trainer usa le API effettive dell'applicazione: importazione delle 82 camere del dataset, initializeTrainingModel, Trainer::initialize/train, salvataggio headless, riapertura del ProjectDocument e verifica del CKPT associato (iterazione e numero di gaussiani). La build ha LFS_ENABLE_CUDA=OFF e le prove nascondono CUDA. Non usa il vecchio runner HIP a topologia fissa.

Prove riuscite a larghezza massima 320, CPU offload, MCMC e correzione ottica attiva: 2 step SH0; 620 step SH0 con crescita da 19555 a 20532 gaussiani; ripresa 620->720 con ulteriore crescita a 21558; ripresa 2->4; prova accelerata SH0->SH3, crescita e Morton ogni due step fino a 8, seguita da ripresa 8->10. Tutti i casi riusciti salvano e riaprono il checkpoint. La prova accelerata e esplicitamente marcata NOT A BENCHMARK e non fornisce tempi confrontabili.

Queste prove hanno individuato e corretto tre collegamenti effettivamente mancanti: riduzioni Tensor fuse/segmentate necessarie all'inizializzazione, calibrazione dei trasferimenti checkpoint senza timestamp GPU HIP, e scrittura SH per snapshot in scratch UInt8 anziche solo Tensor Float32. Le riduzioni restano fuse sul dispositivo (nessun intermedio grande quanto il modello), con al massimo il workspace di riduzione gia previsto dal core. Il contratto numerico copre quattro riduzioni, segmenti brevi/lunghi/divisi, lease rhs, guardie e facade Tensor pubblica. Il contratto SH aggiunge destinazioni byte allineate per FP32/F16/Q16, equivalenza esatta con destinazioni tipizzate, guardie e rifiuto dei disallineamenti.

La calibrazione HIP usa tempo monotono sincronizzato dopo il completamento dei trasferimenti e lo segnala nel log CLI; non viene presentata come tempo del dispositivo. Le annotazioni dei range del profiler sono opzionali e disabilitate esplicitamente per HIP, senza dipendenza NVTX/CUDA. I percorsi degli altri backend per calibrazione e annotazioni restano quelli esistenti.

Rimangono da completare: tutti i moduli opzionali ancora non registrati, selezione e identita delle GPU, interop del viewer e gestione delle configurazioni miste, verifiche estese di qualita e confronto equivalente a 2000 step con i parametri storici. Non e ancora una dichiarazione di integrazione completa o di assenza globale di regressioni.
### 21 ottobre: app GUI nativa, preferenze e Structure

La build completa LichtFeld-Studio + modulo Python e compilata con CUDA disabilitato. Verificati nell'app via MCP: Tensor HIP e identita ROCm DLPack (device type 10), rifiuto degli import estranei prima di consumare capsule, inventario lazy, cambi backend/GPU con indicatore di riavvio e sua rimozione tornando alla configurazione attiva. Screenshot della sezione Hardware & GPUs e del badge arancione Restart needed conservati nello studio locale. Non e ancora implementato l'import di allocazioni ROCm estranee.

Il training GUI ha individuato due connessioni mancanti: selezione dell'allocatore generico dello storage splat e sostituzione della timeline del consumer. Correzioni verificate da contratti su crescita in-place, dati precedenti, slack opacity/quaternion, overflow rifiutato, sostituzione di semafori e consumer senza device. Ripetuti con successo tutti i 20 contratti precedenti. La GUI ha poi completato 12 iterazioni MCMC a max-width 320, cap 30000, SH0, CPU offload e undistort, con thin_structure_weight=0.5 e densify_structure_weight=1, salvando project.licht senza nuovi errori di rendering. La loss sui gradienti non viene attivata da 12 iterazioni; e verificata separatamente dal contratto numerico.

Registrata la tredicesima famiglia Structure: ridge a quattro scale con workspace per bande, pesi fotometrici/densificazione e loss Sobel con backward accumulato. Shader del progetto abbassato a HIP senza SDK CUDA, parametri per dispatch e scratch condiviso per blocco. Il generatore ordina le letture XOR dei conteggi prima della barriera per evitare race fra wave. Contratto PASS: riferimenti CPU per pesi e loss/gradiente analitico, tile dispari, accumulo, equivalenza delle bande, normalizzazione ridge, input UInt8 rispetto a Float32 normalizzato, maschere tutte valide e tutte nulle. Non sono state modificate le implementazioni numeriche CUDA/Metal/Vulkan.

Verifiche UI aggiuntive: 152 test Python, policy GPU HIP on/off (16+13), viewer GPU memory (3 PASS e un caso CUDA non compilato SKIP). La telemetria HIP non usa i valori NVML della GeForce; memoria condivisa riportata dal driver non e descritta come VRAM dedicata.

Il controllo CLI preliminare con seed storico e evaluation ha scartato tutte le viste per mancanza del warp inverso HIP. Quel run e FALLITO e non vale come confronto di qualita. Il ricampionamento inverso e ora in implementazione/verifica prima di eseguire il benchmark da 2000 step. Nessun nuovo tempo comparativo dichiarato. Rimangono otto famiglie applicative non registrate (Gsplat, Geometry, Masks, Mrnf, Bilateral, PPISP, Controller, Blob), zero-copy HIP/Vulkan sulla stessa scheda e completamento delle verifiche di compatibilita. Il viewer usa ancora fallback host sincronizzato, esplicitamente segnalato nella CLI, anche sulla stessa scheda.
### 21 ottobre: evaluation reale e NN Core

Il warp inverso HIP e ora collegato per colore con maschera di validita, maschere, depth HW/CHW a un canale e normal. Il contratto confronta raggi analitici e copertura CPU, bordi/tile dispari, cinque modelli ottici e tutti i modi con Vulkan. Passa. La prima evaluation completa del trainer reale valuta tutte le 11 viste e salva le immagini, PSNR/SSIM/LPIPS e progetto; e uno smoke test a 2 step, non una misura di qualita finale.

Completati i cinque kernel NN Core prima mancanti: im2col, col2im, resize, attivazioni e griglia; il pooling nativo precedente resta invariato. GEMM FP32 nativa copre prodotto normale, peso trasposto, batch e bias/ReLU: tile condivisi per matrici e un kernel per poche righe, senza librerie CUDA. Il contratto verifica riferimenti CPU, batch, dimensioni dispari, K nullo, guardie, padding, canali di ingresso/uscita diversi nel fold, cinque attivazioni, nove combinazioni di resize e descrittori/alias rifiutati senza scritture. Tutti i 23 contratti applicativi passano con la stessa build.

L'uscita headless HIP senza script ora lascia distruggere trainer, scena e generazioni ritirate prima del teardown in main. Il controllo reale a 3 step con crescita e checkpoint termina con successo e senza l'avviso sulle allocazioni vive. I percorsi di uscita CUDA/macOS e quelli con script mantengono la scelta preesistente. Gli eventi temporali HIP restano volutamente non esposti: i riproduttori precedenti in sh-rsqrt-validation.it.md avevano gia dimostrato tempi negativi del driver; non vanno presentati come GPU time attendibile.

Il primo tentativo da 2000 step e stato fermato ed escluso: applicare come override di ripresa il training_config.json esportato, che contiene step non scalati, cambia il calendario effettivo. Un preparatore CPU C++ usa OptimizationParameters::apply_step_scaling e rende espliciti i valori risolti con steps_scaler=1: start_refine=33, refine_every=7, stop_refine=1667, Morton=333. Il nuovo run real-hip-historical-2000-oct21-r2 verifica questi valori anche nel file effettivamente scritto dall'app, usa seed SHA256 4273c94a3c29daffc84c490651510d91ac71ba9cfb9eba8aa3abc0fced3f63f7, cap 1000000, SH0, max-width 1920, CPU offload e evaluation sulle stesse 11 viste. E ancora in corso; nessun tempo finale o giudizio di qualita finale e dichiarato. I primi campioni del trainer reale sono piu lenti del runner isolato precedente e richiedono profiling/ottimizzazione dell'integrazione, non una dichiarazione di equivalenza con quel runner.
### Porting integrato: Blob, Controller, MRNF, Bilateral e PPISP (21 ottobre)

Aggiunte e registrate cinque famiglie complete, arrivando a 18/21: Blob, Controller, MRNF, Bilateral e PPISP. Le tre famiglie ancora assenti sono Gsplat/3DGUT, Masks e Geometry; la condivisione HIP/Vulkan dello stesso modello senza duplicazione resta da realizzare. Non dichiarare completo il backend.

Blob: downsample CPU indipendente, immagine uniforme senza falsi picchi, sweep analitico con nove viste di supporto, picchi/bitmap confrontati con Vulkan. Controller: contratto dello slot esposizione, accumulo dei gradienti ripetuto, ReLU e gradiente opzionale. MRNF: tutti gli otto callback verificati, bounds ordinati sulla GPU, Gumbel denso/sparso deterministico, noise congelato/invisibile, decay, pruning e compattazione confrontati con CPU. Training MRNF reale da dataset fresco: 80 step, crescita 19.555 -> 30.000 gaussiane, checkpoint finale e uscita 0. Un checkpoint MCMC non puo essere ripreso come MRNF: il rifiuto strategy mismatch e corretto.

Bilateral: kernel del repository preparati in copie HIP isolate; affini CHW/HWC contro forward e gradienti analitici CPU, exposure-chroma contro Vulkan, TV/Adam/momenti/proiezione/offset contro CPU, percorsi in-place e trasferimenti limitati. PPISP: inizializzazione neutra, forward/backward contro Vulkan, differenze finite dei parametri, coordinate crop, backward in-place, regolarizzazione/Adam/project-mean CPU e gruppi/output opzionali. Correzione esposizione reale con Bilateral+PPISP: 24 step, crescita a 23.766 gaussiane, progetto salvato e uscita 0. La rete controller completa ha rivelato ulteriori ingressi NN legacy del Core: bias_add/bias_relu/relu sono ora nativi e testati; max_pool2d/adaptive_avg_pool2d sono in fase di collegamento/verifica. Non confondere il test del callback Controller con la rete completa nel trainer.

Core: Int32 -> Bool usa il kernel di conversione HIP, evitando un broadcast intero non supportato. Riduzioni native Bool e intere con accumuli interi, assi arbitrari e viste trasposte; verificati Int64 oltre 2^53 tramite descrittore nativo, identita vuote tramite facade e ingressi pubblici Bool sum_scalar/any e Int32 sum. Le limitazioni pubbliche condivise delle riduzioni Int64/any-all su Float32 non sono state cambiate.

Ottimizzazione MCMC sperimentale: la variante RGB multi-pixel prima copriva solo DensificationType::None. Estesa a MCMC sotto LFS_HIP_FAST_DENSIFICATION_PIXELS, senza cambiare i kernel CUDA/Metal/Vulkan originali. Tre step, 263 gaussiane, viste 19x17/64x48/1x1, gradienti piccoli e molto grandi: parametri identici e statistiche densificazione entro 2e-5 assoluto+relativo rispetto al modulo precedente. ABBA sul trainer reale, checkpoint finale da un milione, 220 step per processo e 142 di warmup: A1 187.802576 ms, B1 149.579704 ms, B2 149.563194 ms, A2 187.839890 ms. Media 187.821233 -> 149.571449 ms, -20.365%, speedup 1.256x. Attesa loader circa 0.06 ms in tutti i casi. La nuova variante non ha ancora la verifica completa di qualita a 2.000 step e resta esplicita/sperimentale; non dichiarare raggiunto il tempo Vulkan ottimizzato.

Diagnostica per fasi: 78 campioni di CPU steady_clock con sincronizzazione della coda, non eventi temporali HIP: pre-forward 1.678 ms, forward 64.715 ms, loss 11.519 ms, backward 106.181 ms, optimizer 0.640 ms. Nel backward, blend ~95.857 ms e proiezione ~7.840 ms su tutti i 220 step. La coda HIP condivisa puo includere lavoro del loader nelle finestre diagnostiche.

Un eseguibile di test terminato dopo un errore iniziale resta nello stato Windows di terminazione e mantiene mapping di DLL. GPU verificata sana tramite contratto Core indipendente. File mappati conservati con nomi archivio, build normale ripristinata; nessun riavvio e nessun processo di altre chat toccato. Il temporaneo CMAKE_PROJECT_INCLUDE per il nome alternativo Core e stato rimosso dalla configurazione locale.

### Oct 21 - real controller, geometry, masks and long MCMC verification

The complete PPISP controller network now finishes its real 24-step activation test (activation at step 5), saves its project and exits successfully. The missing legacy Core bias/ReLU and pooling entry points use native HIP operations and pass CPU references.

Geometry: all five callbacks are connected. Four loss operations pass weighted/unweighted and inactive-coverage cases against Vulkan, including gradients and diagnostic prefixes; independent CPU cosine and anchor projection references pass. Anchor scratch is bounded and released only after confirmed queue completion. No CUDA SDK/runtime is used. Masks: all five callbacks are connected; photometric, opacity and alpha penalties pass independent CPU references, while small/large/near-clipped/invalid mesh faces and point coverage match Vulkan, including inverse sample maps. Distorted mesh preparation exposed Core masked_fill; its HIP implementation preserves unselected raw storage and supports the seven facade dtypes. Additional dtype/broadcast tests are queued for the current rebuild.

A complete real MCMC candidate run (same seed and resolved schedule, 2,000 steps, max_width 1920, cap 1M, SH0, 11 evaluation views) finished with exit 0. CPU step window 201-1900, 1,700 samples: previous mean 328.718765 ms, candidate 213.357566 ms (-35.094%); medians 297.312950 -> 204.939900 ms. This whole-run result includes the evolving topology; the earlier resumed ABBA experiment isolated a 20.365% improvement. The two measurements are not interchangeable.

Quality averages: PSNR 18.208645 -> 18.290804, SSIM 0.437029 -> 0.439301, LPIPS 0.577670 -> 0.576331. All 11 views were inspected in side-by-side sheets including the historical CUDA render; no overall visual degradation was observed. This is evidence for this dataset/configuration, not a guarantee for every scene. Core pool peaks are unchanged (used 577,335,508 bytes; reserved 586,527,264 bytes). Driver global shared-memory accounting is not isolated process VRAM. Full wall time, including evaluation, was 1009.409 -> 815.698 seconds and must not be presented as ordinary training-step time.

The validated MCMC multi-pixel path is now the HIP default when eligible; LFS_HIP_FAST_DENSIFICATION_PIXELS=0 explicitly selects the former diagnostic path. Original CUDA, Metal and Vulkan kernels are unchanged. The candidate did not yet beat the historical optimized Vulkan training baseline.

3DGUT: the complete application adapter and native entry points have been written for RGB, D, ED, RGB_D, RGB_ED and RGB_D_N, SH0-3, camera rays, geometry gradients, density/error/edge/share accumulation, resource errors and cache lifetime. One/three/four/eight channel kernels and geometry features are being compiled and tested; do not count this family as validated until those tests pass. Checkpoint-resume and the consolidated serial suite remain pending. Same-device HIP/Vulkan shared storage, physical GPU inventory unification and broader platform validation also remain open.

Per-view qualification of the long comparison: PSNR improves on 6/11 views, SSIM on 7/11, LPIPS on 5/11; the respective averages improve. Individual views are not identical: worst PSNR delta -0.111305 dB, worst SSIM delta -0.00305748, worst LPIPS delta +0.00511187. These small view-level fluctuations and the absence of an overall visual degradation must not be described as bitwise parity or improvement on every image. Full per-view deltas and binary hashes are retained in the local mcmc-pixels-full-quality-summary-oct21.json artifact.


### Verifica di integrazione, 21 ottobre: inventario, SH3 e memoria esterna

- `full-integration-oct21-r7-results.json`: 33 contratti seriali superati sui binari ricompilati. Gsplat include ora quattro confronti aggiuntivi su ritagli SH2 (pinhole, fisheye, equirettangolare e thin-prism), con output e gradienti confrontati con Vulkan.
- `fast-sh3-application-smoke-oct21-r4` e `gsplat-application-smoke-oct21-r3`: trainer reali, 64 step, SH0-3, refinement, cap 30.000 e checkpoint; entrambi exit 0 e senza l'avviso di allocazioni HIP trattenute alla chiusura. La causa era lo scratch statico della ricodifica Q16: lo slot HIP viene ora svuotato attraverso il pre-shutdown hook; il ramo preesistente degli altri backend resta invariato.
- L'inventario C++ verifica la Radeon HIP/Vulkan come `pci:0000:c8:00.0` e mantiene distinta la GeForce `pci:0000:c6:00.0`. Le scelte conservano i propri UUID API. Superati 13 test Python di inventario, comprese collisioni, identificatori ambigui e memoria condivisa.
- `external-memory-contract-oct21-r1.log`: nuova API del runtime per importare memoria esterna, 128 cicli bidirezionali reali HIP/Vulkan sulla stessa allocazione, ownership Vulkan EXTERNAL con completamenti host, controlli di range, piattaforma, contesto, handle rilasciati e allocazioni normali successive. Questo prova il seam nativo, non ancora la pubblicazione condivisa del modello nel viewer.
- Il collegamento Core degli import e la contabilita senza duplicazione del pool sono in compilazione/verifica. La pubblicazione del modello nel viewer resta da collegare con le barriere di ownership e i reader; non va descritta come completata sulla base del solo test nativo.

Riferimento della sincronizzazione e import: https://rocmdocs.amd.com/projects/HIP/en/latest/how-to/hip_runtime_api/external_interop.html . I completamenti host sono un percorso supportato; non si assume che i semafori esterni funzionino su un driver che li ha rifiutati nelle prove precedenti.


### Integrazione HIP/Vulkan, verifica locale del 21 ottobre

La suite `full-integration-oct21-r12-results.json` supera 36 contratti seriali sugli stessi binari: include le 21 famiglie del trainer, import esterno Core, coordinamento concorrente e pubblicazione condivisa del viewer. Il confronto MCMC ottimizzato/non ottimizzato usa due politiche dello stesso modulo corrente; il prodotto non richiede moduli HIP precedenti.

L'import Core conserva il proprietario Vulkan fino all'ultima vista Tensor ed esclude i byte importati dal pool HIP. Il contratto del viewer verifica scritture bidirezionali GPU, offset, capienza, sostituzione del modello con lease ancora vivo, 32 alternanze HIP/Vulkan e il blocco di una chiamata nativa del trainer durante la registrazione del consumer. Il coordinatore protegge registrazione e invio dei comandi e alterna i ruoli in contesa. Il renderer usa questi scope solo nelle build HIP; le implementazioni degli altri backend non sono sostituite.

Il caricamento nell'app reale conferma `HIP/Vulkan shared storage active`. La prova GUI da 2.000 step usa il CKPT estratto senza modifiche dal progetto iniziale comune (SHA-256 del CKPT `55485e15b62870b8a32d960800693f299a49d58233b174b0a9a6db7135c6cd2e`), dataset locale, undistort, larghezza massima 1920, SH0, cap 1M, cache CPU e steps_scaler=1. Il percorso GUI dei checkpoint richiede il CKPT separato; il progetto originale e il suo checkpoint restano immutati. Il test ha superato 1.200 step e la crescita a 1M, con due cambi di camera e un salvataggio concorrente; il completamento e la chiusura non sono ancora verificati a questo punto. I tempi GUI con viewport attivo non sostituiscono la baseline headless.

Restano aperti l'import DLPack di allocazioni ROCm esterne, le operazioni Tensor generali ancora esplicitamente non supportate, le verifiche delle altre piattaforme e il superamento prestazionale della baseline Vulkan. I contratti superati non dimostrano assenza universale di regressioni.


Aggiornamento della prova GUI: completati 2.000 step, crescita a 1M, valutazione e checkpoint finale (generation 4); due cambi camera e salvataggio concorrente riusciti. Questa prova e' esclusivamente di integrazione: il caricamento GUI del CKPT riimporta il dataset e produce una suddivisione train/test e una preparazione delle camere differenti dal progetto conservato nel protocollo headless. Le metriche GUI non entrano nella tabella comparativa storica.

Il controllo reale successivo ha rilevato due lacune ancora aperte: la selezione richiedeva `filter_points`, non ancora portato, e la chiusura GUI segnalava allocazioni HIP vive. Aggiunti adattatore nativo del filtro, generazione dai predicati Core condivisi e contratto per etichette, offset/stride, tabelle vuote/alias, forme invertite, tre proiezioni e input temporanei accodati. Compilazione e ripetizione GUI in corso; lo shutdown e' in diagnosi. Non descrivere l'integrazione complessiva come conclusa prima di questi controlli.


Ulteriore verifica GUI: la cache globale dell'overlay dei frustum tratteneva 82 camere e 164 piccole allocazioni HIP (6.232 byte). Il rilascio e' ora esplicito alla chiusura solo quando il backend attivo e' HIP; con diagnostica abilitata, le sessioni GUI successive non segnalano piu' risorse HIP vive. CUDA e gli altri backend non entrano nel nuovo ramo.

`point-filter-contract-oct21-r2.log` verifica anche update_labels e and_live: modi Add/Remove/Replace, lock, categorie, indici ripetuti, alias, stride e offset. `point-filter-contract-oct21-r3.log` aggiunge l'istogramma di gruppi UInt8/Bool con bin Int32, bin zero escluso, input vuoto e guardie sui bin adiacenti. Questi contratti passano su CPU e HIP; la suite r14 resta il riferimento di 37 casi antecedente alle ultime aggiunte. Il successivo test dell'app ha isolato un confronto Int32 NotEqualTensor mancante (op 30); aggiunta implementazione nativa dei predicati Int32/Int64/UInt32/Float16, attualmente in compilazione con test di precisione intera e speciali floating point. Non considerare ancora passata la selezione GUI finale.


Verifica conclusiva della prima integrazione GUI: `full-integration-oct21-r18-results.json` supera 37 contratti seriali con gli ultimi binari. I predicati tipizzati sono confrontati con CPU: Int32 oltre 2^24, Int64 oltre 2^53, UInt32 e FP16 con NaN/infinito. Le promozioni scalari pubbliche restano quelle della libreria; non sono state ampliate le API comuni per adattarle al test.

Nella sessione GUI r8 sul modello da 1M: selezione rettangolare 718.183 elementi; rimozione 336.521; undo 718.183; redo 336.521; intersezione con la regione rimossa zero; aggiunta di quella regione 381.662; clear zero. Ogni modifica e' verificata con selection_get, inclusi indici e conteggi. Il nodo camera correttamente esclude il modello dal filtro; il nodo Model lo include. Controllato il viewport attraverso catture della finestra. L'uscita normale non riporta errori HIP o allocazioni trattenute. Il modulo nativo del trainer mantiene SHA-256 `be44344559a43b4019ecb9efa3de954872e4c3532fbffa0c5401b32f33d4f936`: i kernel del trainer non sono stati modificati per queste correzioni del viewer.

Il lavoro prosegue su project_points e mark_points_2d, necessari alle selezioni riferite alle camere del dataset e alle regioni 2D. Adattatori scritti riutilizzando PointProjection/pointInRegion/rounded math condivisi; contratto aggiunto per tre modelli di camera, trasformazioni, visibilita, indici fuori range, code, stride, regioni disco/rettangolo/poligono/dischi e valori non finiti. Questi due nuovi port sono ancora in compilazione: non sono inclusi nei 37 casi r18.

Verifica successiva del 21 ottobre: la suite r20 supera 38 contratti seriali, inclusi project_points e mark_points_2d. Nella sessione GUI r9 sulla camera 0 del dataset, rettangolo e poligono equivalente selezionano gli stessi 7.906 indici; il pennello seleziona 2.563 indici, tutti contenuti nel rettangolo. Clear e chiusura normale passano senza risorse HIP trattenute. Questi risultati riguardano integrazione e correttezza, non nuovi guadagni di training.

Esperimento separato, non abilitato di default: LFS_HIP_FAST_SPLIT_MCMC_STATS=1 divide il backward RGB dalle statistiche di densita/errore MCMC. L'obiettivo e' ridurre i registri vivi; due passaggi implicano letture aggiuntive. Occorrono contratto numerico e ABBA del trainer reale prima di qualsiasi accettazione prestazionale. La baseline primaria resta 213,358 ms/iterazione nel confronto headless da 2.000 step.

Esito dell'esperimento split MCMC: scartato e rimosso dal generatore di produzione. Sei casi numerici passano con due livelli di gradiente, ma ABBA reale dallo stesso checkpoint 1M (2.000 -> 2.220, 142 warmup, 78 steady) misura A1 145,970274 / B1 164,117023 / B2 164,904163 / A2 146,319896 ms. Medie: 146,145085 contro 164,510593 ms, peggioramento +12,5666%. Attesa loader steady 0,059-0,064 ms. Non e' una nuova misura del protocollo completo da 2.000 step. Pool picco A1/B1/B2 618.711.288 byte, A2 610.416.892 byte; conteggi allocazioni Core, non VRAM fisica per processo. Loss finale circa 0,14714 in tutte le prove; non sostituisce una valutazione qualitativa completa. Artefatti split-stats-abba-summary-oct21.json, split-stats-contract-oct21-results.json, snapshot del generatore e DLL sperimentale nella cartella locale rocm-tensor-study. Nessun backend esistente modificato da questo esperimento.

Controllo supplementare sullo stesso checkpoint finale (SHA-256 f0894923c0f26b35756ec9fd3f96b6d6907664be15b7481f8621dc7cfed5030b): V-H-H-V, resize 1920 SH0, cap 1M, 2000 -> 2220, 142 warmup/78 steady. HIP 144,170741 e 145,559773 ms; Vulkan di questo branch 256,554869 e 262,056609 ms. Questa versione Vulkan non e' il riferimento ottimizzato piu' recente gia' registrato, e la topologia e' piena fin dall'inizio: non sostituisce la baseline storica 114,591 ms + loader, non dimostra raggiunto l'obiettivo prestazionale, non e' il test completo con crescita da 2.000 step. I quattro run hanno lo stesso hash iniziale e completano senza errori; qualita' completa non rivalutata in questo controllo. Riepilogo same-checkpoint-hip-vulkan-oct21-summary.json.

Nuova ipotesi circoscritta al resume HIP: MCMC inizializza statistiche transitorie anche riprendendo dopo stop_refine, ma normalmente le libera solo allo step esatto stop_refine. Il prototipo LFS_HIP_STOPPED_MCMC_CLEANUP=1 esegue la stessa transizione scaduta su HIP alla prima post_backward utile, senza cambiare rumore, parametri o momenti e senza cambiare gli altri backend. ABBA reale in corso; nessun guadagno ancora accettato. Il ciclo fresh che attraversa stop_refine non e' oggetto di questa ottimizzazione.


Congelamento del 21 ottobre su richiesta dell'utente: prove fermate, valutazione qualitativa del candidato resume incompleta. Riferimento esaustivo: rocm-historical-handoff-2026-10-21.it.md nella documentazione development. Il prototipo resume e il suo contratto non sono accettati; il contratto non e' stato compilato. Si procede ora a studio e allineamento locale all'ultimo upstream/dev, preservando questo stato storico.

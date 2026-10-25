# AMD HIP/ROCm Tensor backend

The optional HIP backend implements Core's `GpuBackendOps` contract. All HIP
execution, allocation and SDK loading stays inside the Tensor implementation.
Public Tensor headers do not include HIP SDK headers, and the host compiler does
not link a HIP import library. The private native module uses the selected ROCm
compiler independently of the application's C++ and CUDA compilers.

## Build

`LFS_TENSOR_HIP` defaults to `OFF`. Standard CUDA, Vulkan and Metal build defaults
and backend preference order remain unchanged. The option supports Windows and
Linux; macOS rejects it at configuration time.

Add these arguments to a normal application configuration:

```sh
-DLFS_TENSOR_HIP=ON -DLFS_HIP_ROOT=/absolute/path/to/rocm-sdk -DLFS_HIP_ARCHITECTURE=gfx1150
```

A HIP-only Tensor validation build can additionally use `LFS_ENABLE_CUDA=OFF`
and `LFS_BUILD_TRAINER=OFF`; the native module never requires a CUDA toolkit or
PyTorch. Normal application dependencies, Python and the project's Slang compiler
are still required. `LFS_HIP_CORE_SLANGC` can explicitly select that compiler.
The development SDK reference is ROCm SDK 10.1.0; its package version is separate
from the HIP runtime API version. HIPRTC is required for fused expressions.

`LFS_HIP_ARCHITECTURES` accepts an explicit semicolon-separated list and takes
precedence over the single development target. `LFS_HIP_PORTABLE=ON` requests the
complete declared architecture list in `cmake/HipArchitectures.cmake`.
`LFS_HIP_OFFLOAD_COMPRESSION=ON` requires compiler support and fails configuration
rather than silently ignoring an unsupported request. These are compilation
targets, not a claim that every corresponding GPU/driver has been qualified.
Wave32 and wave64 code objects can be compiled; each requires hardware validation.

This Tensor integration does not enable the HIP trainer, add preferences controls,
or define distribution profiles. Those integrations must independently admit
supported capabilities rather than infer them from Tensor availability.

## Selection and execution

`GpuBackend::HIP` is available only in HIP-enabled builds. Existing backend enum
values remain stable. `GpuBackendScope` or `TensorExecutionTarget::default_queue`
selects HIP through the same Tensor facade used by other backends.
`TensorBackendOptions::hip_device` accepts a nonnegative decimal device ordinal;
an empty value selects the first compatible device. Inventory and availability
checks do not create an execution context. Selection is frozen at startup with
the existing backend options. An incompatible explicit selection does not switch
to a different GPU or CPU backend.

The adapter owns one nonblocking HIP queue for the selected device and serializes
host submissions. It rejects CUDA streams and foreign storage before dispatch.
Each native call restores the calling thread's previous HIP device. Tensor
readback, upload, fences, completion and timestamp facilities use HIP-owned
resources; CUDA availability is irrelevant to those paths.

Generic pointwise, strided, indexing, reduction, scan, sorting, GEMM and sampling
operations use the private native operators. Image, geometry and NN operations
are also implemented inside the Tensor backend. Shared source generators adapt
project-owned mathematical kernels and Slang programs without changing the CUDA,
Metal or Vulkan implementations. Fused expression source is lowered separately
and compiled with HIPRTC; unsupported inputs fail explicitly.

## Ownership and memory

HIP device allocations use the appended `MemoryDomain::HipDevice`. The allocator
caches exact requested sizes up to a bounded cache limit and trims optional cached
allocations before retrying an allocation failure once. Live tensors are retained.
Memory information separates live allocations, cached allocations and native
algorithm workspaces. Device heap statistics from APIs using the same physical
GPU must not be summed.

Storage views validate backend, owner, byte ranges, dtype and layout before native
use. Borrowed raw views require a live allocation and retain the existing raw-view
lifetime contract. External-owner views retain the supplied owner and do not
account the same borrowed allocation twice.

The private C ABI is exact-versioned. The loader checks the revision before
resolving the typed export inventory and accepts only absolute module paths.
Windows uses Unicode paths and explicit module/default dependency directories;
Linux uses local, eager symbol resolution. SDK dependencies must be installed
correctly. Development builds copy only the runtime DLL families used by the
module, rather than the full SDK.

Explicit shutdown rejects outstanding Tensor allocations and transfer/queue
leases. Cleanup failure retains the native context and loaded module, reports a
CLI diagnostic, and permits retry. Transfer/event cleanup uses preallocated
retirement nodes so a failed destructor release can be retried before context
teardown. An unrecoverable driver failure retains ownership until process exit
rather than unloading code beneath pending GPU work.

## Validation

Build `tensor_operation_contracts` and run `TensorOperationContracts.HIP` for
required GPU execution; an unavailable required backend is a failure. The same
operation and edge-case tests run against CPU and each compiled GPU backend.
CPU runners execute the CPU contracts and runner validation without claiming GPU
coverage. The SDK-only and SDK-free checks are documented in
`cmake/hip-native/README.md` and `cmake/hip-host/README.md`.

The HIP compile CI is path-filtered, uses a pinned SDK and source-generation
compiler, builds declared native targets, and checks module exports without a
GPU. Windows hardware execution and Linux compilation cover different properties;
compilation alone does not validate device results or driver behavior.

## Integration and compatibility checks

Existing HIP implementations must be checked against the current Tensor contract
before reuse. A previously working application build does not establish that an
extracted operator module, its adapter and their tests agree on every dtype,
descriptor or ownership rule.

### ABI and operation semantics

ABI revision 4 unified the native operator contract; this adapter's expanded
module requires revision 5. These are exact module versions, not independently
negotiated versions for individual operators. A loader must reject a different
revision before binding exports.
Changing a descriptor or the meaning of an operation number requires reviewing
all producers and consumers, not just updating the version constant.

Every native reduction entry point uses `LfsHipReduceOp`. Core `ReduceOp` values
must be mapped explicitly: their ordinal values are not interchangeable. Test
Min and Max with different expected answers, and Any and All with mixed inputs.
Native dtype support does not expand the public frontend's supported operations.
For example, public axis reductions do not currently admit Int64, and public
Int32 reductions do not admit Any/All; native tests cover the additional storage
types separately.

Scalar comparisons must follow the conversion contract of their entry point.
Include fractional scalars, scalar-left expressions, boundary integers and
predicates with byte output. Bool semantics and UInt8 storage must not be assumed
interchangeable merely because both occupy one byte. Check generated pointwise,
fused expressions and GEMM activations separately when changing shared functors;
HIP device ReLU canonicalizes negative zero to positive zero.

### Workspace and lifetime

Selection must validate insufficient output capacity before writing the output.
An exact-sized result must not require a retained full-input-sized result copy.
The library's algorithm workspace still needs accounting: this is not a promise
of constant total workspace. Measure live, cached and algorithm memory separately,
including cleanup after allocation, synchronization and release failures.

Use the shared range and alias helpers after validating ownership and range
bounds. Exact in-place permissions depend on the operation; a shifted overlap,
broadcast alias or foreign-context buffer must not inherit permission from a
different operation with the same element width.

Reduction refactoring must preserve integer accumulator precision, Half output
rounding and the final-pass accumulation order. Cover empty axes, strided inputs,
short/long dispatch thresholds and split reductions. A bitwise comparison over
selected cases is useful regression evidence, not a general equivalence proof
for all devices or input values.

### Verification scope

Validate both the native entry point and the public Tensor call site: a correct
kernel cannot compensate for an adapter passing the wrong enum or dtype. Run
shared contracts on CPU and available GPU backends, and check HIP-disabled builds
and CUDA+HIP coexistence when changing shared integration files. Preserve the
existing behavior of other backends rather than broadening their contracts to
make a new test pass.

Loader and recovery tests must be repeatable in the same build directory. Use
unique owned Unicode-path fixtures, verify cleanup and preserve pre-existing
files. A test that passes only after manually deleting a fixture is incomplete.

Record the tested commit, SDK, compiler, configuration, architecture set and
runtime dependencies with validation results outside the maintained source
documentation. Distinguish compilation, CPU-only module checks, GPU operator
execution, full application integration and training quality/performance. A
result from an earlier snapshot or another hardware configuration must not be
presented as validation of the current branch. Experimental and diagnostic build
switches require evidence that the build actually consumes them.

# AMD HIP/ROCm Tensor backend

The optional HIP backend implements Core's `GpuBackendOps` contract. All HIP
execution, allocation and SDK loading stays inside the Tensor implementation.
Public Tensor headers do not include HIP SDK headers, and the host compiler does
not link a HIP import library. The private native module uses the selected ROCm
compiler independently of the application's C++ and CUDA compilers.

The shared responsibilities of all four implementations are documented in
[Tensor backend contracts](tensor-backend-contracts.md). That contract also
covers allocation ownership, targets, compatibility APIs and validation.

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

`GpuBackend::HIP = 3` and the dense `kGpuBackends` table are present in every
build, including builds without HIP. `kCompiledGpuBackends` alone controls which
backends are advertised; disabled HIP reports no devices and cannot construct
backend operations. Existing enum values and default backend preference remain
stable. `GpuBackendScope` or `TensorExecutionTarget::default_queue`
selects HIP through the same Tensor facade used by other backends.
`TensorBackendOptions::hip_device` accepts a nonnegative decimal device ordinal;
an empty value selects the first compatible device. Inventory and availability
checks do not create an execution context. Selection is frozen at startup with
the existing backend options. An incompatible explicit selection does not switch
to a different GPU or CPU backend.

CMake generates the compiled-backend inventory and its registration source from
one set of build options. The shared indexed driver registry does not include
backend SDK headers or construct concrete backend implementations. Hardware
inventory, startup selection and activation belong to each backend driver.
Activated operation pointers are cached per thread; steady Tensor dispatch uses
that indexed pointer and the existing operation's virtual call. CUDA worker
binding remains inside its driver.

Upload buffers, fences, completion snapshots, timestamp pools and packed readback
slots use SDK-free ownership contracts implemented by all four GPU backends.
Each shared transfer owns one native implementation rather than separate fields
for every GPU API. Concrete HIP types, native error classification and architecture
compatibility checks stay in `backend/hip`. Unsupported optional timestamps and
external imports are reported explicitly by the native factory.

The adapter owns one nonblocking HIP queue for the selected device and serializes
host submissions. It rejects CUDA streams and foreign storage before dispatch.
Checked index and mask-cardinality failures have distinct private error codes.
They are reported after synchronization; queue destruction logs them and releases
its lease. A native synchronization failure still retains the queue ownership.
Each native call restores the calling thread's previous HIP device. Tensor
readback, upload, fences, completion and timestamp facilities use HIP-owned
resources; CUDA availability is irrelevant to those paths.
Synchronous scalar results use reusable context-owned staging; a failed copy or
wait leaves the caller's result unchanged and retains staging for safe cleanup.
Staged readbacks capture preceding execution errors at their queue position.
Later work cannot invalidate an earlier snapshot, and consuming the context's
error cannot validate a failed snapshot. Packed slots retain errors across
appends and check them before publishing bytes, including after a queue wait.

Generic pointwise, strided, indexing, reduction, scan, sorting, GEMM and sampling
operations use the private native operators. Image, geometry and NN operations
are also implemented inside the Tensor backend. Shared source generators adapt
project-owned mathematical kernels and Slang programs. Each backend retains its
native emitters and runtime ownership. Fused expression source is lowered separately
and compiled with HIPRTC; unsupported inputs fail explicitly.

GPU work queues and execution scopes own one native implementation. Queue
creation, scope binding, host callbacks, consumer timeline imports and completion
snapshots use the same contract on CUDA, Vulkan, Metal and HIP. Public scope
objects retain opaque state inline and allocate no scope object on the heap.
Unavailable native features fail explicitly rather than exposing another
backend's handles.

GPU indexing accepts Int32 and Int64 with a common aliasing and selected-count
contract. Adapters whose kernels require Int32 own their conversion scratch;
clamp and wrap normalize wide values before narrowing. Indexed writes snapshot
overlapping inputs before mutation. Prepared readback retains its source and
CPU destination, preserves native ordering, and checks execution errors before
publishing bytes, including zero-byte completion snapshots.

Checked-index capture validation and Debug completion checks belong to the native
implementation. CUDA consumes its device-fault record after synchronization;
the other implementations preserve their native queue error checks.

Vulkan expression cache identity includes the launch's push-constant size.
The loader checks that size before reusing an artifact; an incompatible cached
layout is recompiled with a log warning. A freshly compiled artifact rejected by
its loader raises an error rather than publishing a null kernel.

## Ownership and memory

HIP device allocations use the appended `MemoryDomain::HipDevice`. The allocator
caches exact requested sizes up to a bounded cache limit and trims optional cached
allocations before retrying an allocation failure once. Live tensors are retained.
Memory information separates live allocations, cached allocations and native
algorithm workspaces. Device heap statistics from APIs using the same physical
GPU must not be summed.

Native allocations return their accounting, teardown and diagnostic-label policy
with the allocation descriptor. Shared allocation and deletion consume that
policy instead of deriving ownership from backend names.

Memory-pressure queries and reclaim clients use the native driver for the
allocation's memory domain. Recovery checks that domain rather than the current
Tensor backend; an unavailable CUDA device does not prevent another backend's
pressure episode from recovering. CUDA retains its native OOM error-state
handling and allocator retry behavior.

Storage views validate backend, owner, byte ranges, dtype and layout before native
use. Borrowed raw views require a live allocation and retain the existing raw-view
lifetime contract. External-owner views retain the supplied owner and do not
account the same borrowed allocation twice.

Before native shutdown, cached buffers are released individually and removed
from the cache only after successful release. A later teardown failure cannot
leave cached handles referring to already destroyed native buffers.

Python identifies GPU tensors through their registered backend name. DLPack
capability checks and export preparation belong to the Tensor library; CUDA
pointer queries, stream ordering and Vulkan external-memory imports stay in
native implementations selected by CMake. HIP and Metal currently report this
protocol as unavailable. Both Python DLPack entry points reject HIP export
explicitly; callers can use `.cpu()` before exporting. HIP storage is never passed
to CUDA's pointer or stream APIs for DLPack export. CPU exports ignore GPU consumer
streams.

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
The existing Linux Release job also configures the main tree with HIP enabled,
CUDA and training disabled, reuses its installed dependencies, and builds the
Tensor contracts plus the production Python Tensor binding contracts. It runs
only their CPU checks on the hosted runner. The HIP hardware tests remain
explicitly separate.

`tensor_binding_contracts` builds a small test extension from the production
`PyTensor` source. `TensorPythonBindings.CPU` checks the CPU bridge;
`TensorPythonBindings.HIP` requires an AMD device and checks HIP identity and
DLPack rejection, including empty tensors.

For adapter cleanup fault tests, build the instrumented native module as described
in `cmake/hip-native/README.md`, configure the main tree with
`LFS_HIP_ADAPTER_FAULT_MODULE` pointing to that absolute module path, build
`hip_adapter_recovery`, and run `HipAdapter.Recovery` on a compatible AMD device.
These hooks exist only in the separately instrumented module.

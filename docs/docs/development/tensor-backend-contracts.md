# Tensor backend contracts

CUDA, Vulkan, Metal and HIP implement one Tensor execution contract. Backend
selection belongs to registration and native implementations; shared Tensor
operations describe work without inspecting a backend name or including SDK
headers. Public compatibility API names do not grant permission to implement
their SDK behavior in shared source.

## Registration and dispatch

CMake generates the compiled backend lists and registration references. A build
references only the native register functions it enables. Stable identities are
CUDA `0`, Vulkan `1`, Metal `2` and HIP `3`; being representable does not mean a
backend is compiled or available.

The registry has indexed driver slots. Inventory and selection do not construct
Tensor execution contexts. Native activation performs checks such as CUDA worker
device binding and Metal OS availability. Activated operation pointers are cached
per thread. The steady operation path must not add a registered getter, lock,
allocation or capability probe before its virtual operation.

Native drivers own device and target policy, inventory, pressure services,
protocol capabilities and context shutdown. Unsupported native features must be
reported explicitly; a foreign SDK must never receive another backend's handles.

An unavailable driver has no native target, device selection or renderer-storage
access. Legacy API defaults belong to their compatibility implementations.
Immutable thread-binding policy may be cached per thread; dynamic availability
must not be cached with it. A required device binding still runs at every scope,
including nested scopes, so external device changes cannot bypass rebinding.

## Targets and transfer ownership

Shared upload, fence, completion, queue, elapsed-time and readback objects own one
native implementation. Queue creation and borrowing, scope binding, consumer
timeline imports and host callbacks belong to that implementation.

Target scopes restore the previous state on success and exceptions and retain
their binding tokens inline. Lazy materialization and conversions use native
target-binding policy. Stream retirement and native event ownership stay beside
the native runtime, including compatibility entry points.

Synchronizing a tensor to its existing home target is a no-op. Shared code may
skip native dispatch for that case; different targets must still use the native
ordering policy. Fused expressions preserve the execution-to-home ordering of
every output before subsequent work consumes it on another target.

Transfers retain every resource still in use. Prepared downloads retain their
source and destination and check the producer's execution errors before exposing
bytes. A zero-byte completion still represents an error snapshot. Queue cleanup
must distinguish a checked execution error from a failure to drain; a drain
failure cannot justify releasing live storage or unloading its module.

One completion factory consumes the native record, supplied target/value,
timeline source and release policy. A zero-valued record still performs native
capture. Imported timelines can retain an existing obligation without recording
another native marker; these choices belong to the native caller.

## Allocations

The allocation-only descriptor carries storage, home target, optional external
owner and provenance. For example, the CUDA activation arena can supply retained
storage without making shared allocation code identify CUDA or access the arena.
The hot `StorageRef` layout does not grow to accommodate these allocation details.

Native allocation flags carry accounting, labeling and teardown policy. Shared
adoption and deleters consume that policy. They must not derive ownership from
backend names or dereference allocator metadata after shutdown.

If host-side owner construction fails, adoption must release the unowned native
allocation and balance its accounting before propagating the exception. Borrowed
storage remains owned by its external owner. The normal deleter resolves the
backend on the releasing thread; an allocation-time adapter reference must not
bypass a required thread binding during later destruction.

Exact allocation is an intent: CUDA selects its exact pool path, while other
adapters preserve their native sizing behavior. Empty allocations retain logical
shape and placement without activating an allocator or creating a context;
backend selection and availability validation still apply.

Allocation profiling remains optional. Disabled profiling must not construct
shape descriptors or issue native profiling calls solely for exact allocations.

Memory-pressure recovery follows the failing allocation's domain, rather than
the currently selected Tensor backend. An unavailable CUDA runtime must not
prevent another backend's pressure episode from recovering.

Device domains map explicitly to their backend. Host domains do not query a GPU
or inherit a device safety reserve. Native recovery policy handles terminal
runtime failures without treating unmapped domains as CUDA.

## Operation semantics

All adapters accept the common index widths, aliasing rules and selected-count
contract. Native adapters own any narrowing scratch. Clamp and wrap normalize
wide indices before narrowing, including the extreme signed values. Indexed
writes snapshot overlapping inputs before mutation.

Bool conversion means nonzero, including fractional floating values and NaN;
converting through an integer cast is not equivalent. Checked-index capture
validation and completion errors belong to the native implementation.

Random operations preserve each native algorithm and seed contract. Moving SDK
state must retain lazy initialization, reseeding, offsets, stream ordering and
odd-count handling. The existing explicit CUDA generator APIs retain their
cuRAND handle-address contract and report unavailable without CUDA; other
backends use their own Tensor random operators.

Expression address interpretation and native input-pairing optimizations also
belong to the adapters. Shared expression code performs the common shape,
aliasing and launch validation. Source fingerprints and specialization identity
must distinguish incompatible artifacts. Vulkan additionally validates the
push-constant layout before reusing a cached kernel and recompiles incompatible
artifacts with a log warning. A rejected freshly compiled artifact is an error,
not a usable null kernel.

## Validation

`tensor_operation_contracts` shares operations across CPU and every compiled GPU
backend. `test_tensor_backend_semantics.cpp` has private access for low-level
ownership and service checks; public operation tests use public Tensor APIs.
`tensor_queue_contracts` checks ordering, retained storage and callbacks.

Hosted runners without GPUs execute CPU contracts and runner checks while still
compiling the enabled native interfaces. A required GPU backend that is missing
must fail explicitly. A skipped GPU case is not evidence of GPU correctness.

Hardware validation must cover worker first use, inventory without activation,
shutdown/reopen, failure cleanup and representative before/after submission costs
when direct calls move behind an interface. Build flags and timing dispersion
matter: fastcompile measurements do not establish optimized Release performance.

Run `tools/check_backend_neutrality.py`, the error-debt gate and clang-format on
the change. The neutrality gate also checks the shared Tensor dispatch files
that implement this contract. Its lexical checks supplement compilation and
review; they do not prove ownership or ordering correctness.

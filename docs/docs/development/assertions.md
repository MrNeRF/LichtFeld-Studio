# Assertion and contract policy

LichtFeld checks invalid input at the earliest boundary and keeps redundant
per-element checks out of release hot paths. All contract failures must say:

1. what invariant broke in plain words;
2. the observed values needed to diagnose it; and
3. where it broke. The shared assertion primitive appends `file:line`
   automatically.

For example:

```cpp
LFS_ASSERT_MSG(slot < descriptor_ring.size(),
               std::format("descriptor slot must be in range "
                           "(slot={}, ring_size={})",
                           slot, descriptor_ring.size()));
```

Messages such as `"invalid state"`, `"bad shape"`, or a rule without the
actual state, shape, dtype, handle, index, or count do not meet this policy.

## Shared vocabulary

`src/core/include/core/assert.hpp` and `core/cuda_error.hpp` use the same
`LFS FAILURE REPORT` envelope, but they are different diagnostic families.
Generic contract failures contain the contract, failed expression, detection
site, context, and host stack. CUDA runtime failures add the native CUDA error,
pre-call attribution, device/runtime and VRAM snapshots, CUDA breadcrumbs, and
the asynchronous-error hint. Each path then follows its established
throw/return behavior.

| Check | Use it for | Cost class | Release behavior |
|---|---|---|---|
| `LFS_ASSERT(condition)` | Self-explanatory public/API contract | One branch | Always enabled; throws `std::runtime_error` on failure |
| `LFS_ASSERT_MSG(condition, message)` | Public contract needing observed values | One branch; message is failure-only | Always enabled; logs the shared report and throws |
| `LFS_DEBUG_ASSERT[_MSG]` | Redundant internal invariant after an always-on boundary | Zero in release | Compiles out completely under `NDEBUG`; CUDA device code uses native `assert` |
| `LFS_CUDA_CHECK(call)` | Any `cudaError_t`-returning runtime call | Checked call, pre-call non-clearing error sample, comparison, and relaxed breadcrumb write | Always enabled; logs the shared report and throws on failure |
| `LFS_CUDA_CHECK_MSG(call, fmt, ...)` | CUDA call needing allocation, shape, stream, or phase context | Same happy path; context formatting is failure-only | Always enabled; same failure semantics as `LFS_CUDA_CHECK` |
| `LFS_CUDA_BREADCRUMB("tag")` / `LFS_CUDA_BREADCRUMB_STREAM("tag", stream)` | Key allocation, free, copy, or launch boundary without its own runtime call; use the stream form when ownership is known | Fixed array write plus relaxed atomic increment | Always enabled; static string literals only, no allocation or lock |

`LFS_CUDA_CHECK` accepts only expressions whose result type is exactly
`cudaError_t`; accidental use with a pointer, integer, or boolean fails at
compile time. CUDA calls already made inside callback-shaped APIs route their
status into the same reporter through the central status adapter. Do not add a
local `CHECK_CUDA`, `CUDA_CHECK`, `check_cuda_result`, or one-line logger.

Do not use a debug assertion as the only validation of caller-controlled data.
Validate once at the public or subsystem boundary, then use debug assertions
for per-element bounds, tracker consistency, and already-proven loop
invariants.

## What one pasted failure block contains

The reporter emits one log record, so concurrent logger output cannot split the
diagnostic into unrelated lines. A shortened CUDA example is annotated below:

```text
========== LFS FAILURE REPORT ==========
Family: CUDA runtime error                         # this example is CUDA-specific
Error: cudaErrorMemoryAllocation (2): out of memory
Failed expression: cudaMalloc(&new_buffer, bytes) # expression that returned the status
Detection site: .../memory_arena.cu:1390 (...)    # where it was detected
Context: requested_bytes=2147483648
Attribution: pre-existing CUDA error detected BEFORE this call — this site is NOT the origin.
Thread: 845109...                                  # stable hash for this process
CUDA device: 0 / device count: 1
VRAM: free=612 MiB, used=23459 MiB, total=24071 MiB
Host stack trace:
  #0 lfs::core::RasterizerMemoryArena::grow_arena(...)
CUDA breadcrumbs (most recent first):
  #8841 arena.grow at .../memory_arena.cu:1350 thread=... stream=0x0
  #8840 tensor.pool.allocate at .../memory_pool.hpp:69 thread=... stream=0x...
Hint: CUDA reports async errors at the next sync point. Set LFS_CUDA_SYNC_DEBUG=1 ...
========================================
```

Tensor contract failures use the generic envelope: contract, failed expression,
detection site, context, and host stack. They deliberately do not query the CUDA
runtime or include CUDA breadcrumbs. The stack is captured only on failure and
must retain the public caller; a dtype error that names only `masked_select` is
not a complete report.

`cudaMemGetInfo`, current-device, and device-count queries are guarded while
building the report. A damaged context is printed as `unavailable` with the
query's own CUDA status rather than causing a second crash.

## Diagnostic controls

The canonical CMake and runtime control surface is documented in
[Developer flags and diagnostics](flags). Do not add an environment read at a
call site; add a narrowly justified `LFS_` entry through the shared accessor and
document it there.

Normal CUDA reports sample `cudaPeekAtLastError` before the checked call. If it
was already non-success, the report says explicitly that the detection site is
not the origin. With sync debug enabled, stream/device synchronization is also
performed before and after the call so asynchronous execution failures surface
at the nearest boundary.

At startup, LichtFeld pre-opens a per-process crash log and prints its path.
`std::terminate` reports the active exception type, `what()`, and host stack
through the generic reporter, flushes the logger, then aborts. POSIX
SIGSEGV/SIGABRT/SIGFPE/SIGBUS handling writes only to the pre-opened descriptor,
uses `backtrace_symbols_fd`, restores the default disposition, and re-raises.
Windows installs `SetUnhandledExceptionFilter` as a best-effort address trace;
symbolization remains a postmortem debugger step.

## Adding a new check

1. Put shape, dtype, device, size, state, and ownership rules at the public
   boundary with `LFS_ASSERT_MSG`; include the observed values.
2. Wrap every checked CUDA runtime expression directly in `LFS_CUDA_CHECK` or
   `LFS_CUDA_CHECK_MSG`. Do not save and reword the status in a local macro.
3. At a high-value tensor allocation/free/copy/launch boundary with no runtime
   call of its own, add one static-literal `LFS_CUDA_BREADCRUMB`, or
   `LFS_CUDA_BREADCRUMB_STREAM` when the owning stream is available.
4. Preserve the caller's established failure control flow. If a subsystem must
   recover or fall back, use the central status adapter with its explicit
   log-only disposition; do not silently clear the CUDA status.
5. Add a failure-path test that asserts the sections belonging to that failure
   family and the public caller in the host stack. Do not test only the short
   exception string.

## Vulkan domain wrappers

The Vulkan call sites retain their local error-handling idioms while sharing
the central debug primitive.

- Rasterizer `_THROW_ERROR`, `_THROW_ERROR_ALWAYS`, and `_CHECK_FATAL` remain
  always-on failures. `LFS_VK_DEBUG_ASSERT(condition, fmt, ...)` is the
  rasterizer spelling of `LFS_DEBUG_ASSERT_MSG`; use it only for redundant hot
  invariants. Its format arguments have zero release cost.
- Visualizer `LFS_VK_CHECK_MSG(expr, fmt, ...)` evaluates a `VkResult`
  expression exactly once. On failure it includes the expression,
  `vkResultToString(result)`, the numeric result, caller context, and source
  location, then follows the call site's existing `false`/`lastError()` path.
  It is always-on because Vulkan API failures are runtime failures, not
  debug-only invariants.
- Handles in messages use hexadecimal form. Include the actual range, sizes,
  counts, enum names, frame/slot/image indices, semaphore values, and expected
  state whenever they determine why the call failed.

Vulkan validation defaults, fatal routing, shader source information, and the
source-built validation-layer launcher are documented in
[Developer flags and diagnostics](flags#source-built-vulkan-validation-workflow).

## Release versus debug placement

Keep these checks always-on:

- public sizes, shapes, dtypes, devices, handles, indices, and state-machine
  transitions;
- file/header/payload bounds and import/export schema contracts;
- allocation, command submission, synchronization, and external-interop
  boundaries; and
- conditions that protect a following dereference, array access, driver call,
  or kernel dispatch.

Keep these checks debug-only after an always-on boundary has established the
contract:

- per-element loop indices and computed offsets;
- barrier/layout tracker membership;
- redundant internal ring-slot and packing consistency; and
- expensive read-back or round-trip verification of data just emitted by a
  writer.

Never add a per-splat or per-pixel release assertion. A release check outside
the loop should prove the range once.

An import parser is a boundary even while it iterates: each record originates
outside the process, so record-local bounds, finiteness, and schema checks stay
always-on. By contrast, a writer rereading the complete file it just emitted is
redundant verification and belongs in a debug-only round-trip validator.

## ABI tripwire

CMake generates `lfs_core_abi_stamp.h` from the core implementation and build
inputs. The application compares its compiled stamp with the loaded
`lfs_core` stamp before argument parsing or CUDA initialization. A mismatch
prints both stamps, tells the user to remove stale binaries and rebuild, and
exits with status 2. This is an always-on startup boundary and must remain the
first executable check.

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

`src/core/include/core/assert.hpp` owns the common failure formatter and the
release/debug compile-out policy.

- `LFS_ASSERT(condition)` and `LFS_ASSERT_MSG(condition, message)` are
  always-on boundary contracts. They throw `std::runtime_error` in every build
  type. Prefer the message form for any condition whose runtime values are not
  fully evident from the expression.
- `LFS_DEBUG_ASSERT(condition)` and
  `LFS_DEBUG_ASSERT_MSG(condition, message)` are redundant internal
  invariants. Host failures use the same self-describing formatter. CUDA
  device code uses native `assert`, because device code cannot format or throw.
  Both macros compile to no code when `NDEBUG` is defined; neither the
  condition nor message arguments are evaluated.

Do not use a debug assertion as the only validation of caller-controlled data.
Validate once at the public or subsystem boundary, then use debug assertions
for per-element bounds, tracker consistency, and already-proven loop
invariants.

## Vulkan domain wrappers

The Vulkan call sites retain their local error-handling idioms while sharing
the central debug primitive.

- Rasterizer `_THROW_ERROR`, `_THROW_ERROR_ALWAYS`, and `_CHECK_FATAL` remain
  always-on failures. `LFS_VK_DEBUG_ASSERT(condition, fmt, ...)` is the
  rasterizer spelling of `LFS_DEBUG_ASSERT_MSG`; use it only for redundant hot
  invariants. Its format arguments have zero release cost.
- Visualizer `LFS_VK_CHECK(expr)` and
  `LFS_VK_CHECK_MSG(expr, fmt, ...)` evaluate a `VkResult` expression exactly
  once. On failure they include the expression, `vkResultToString(result)`, the
  numeric result, caller context, and source location, then follow the call
  site's existing `false`/`lastError()` path. They are always-on because Vulkan
  API failures are runtime failures, not debug-only invariants.
- Handles in messages use hexadecimal form. Include the actual range, sizes,
  counts, enum names, frame/slot/image indices, semaphore values, and expected
  state whenever they determine why the call failed.

Vulkan validation is opt-in through the existing validation-layer setting.
Set `LFS_VK_VALIDATION_FATAL=1` in CI or a development run to route a validation
ERROR through the fatal path and abort with the driver message. The default is
off. Startup logs report whether validation layers, debug utils, and fatal
validation routing are active.

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

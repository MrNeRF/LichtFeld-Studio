# Native HIP runtime foundation

This independent developer project builds a private HIP module without configuring
the application, changing its C++ compiler, or requiring CUDA or PyTorch. It does
not register an application Tensor backend or enable training. Standard application
builds do not traverse this project and retain their existing configuration.

With CMake 3.30 or newer, Ninja, Python 3 and a complete ROCm development SDK:

```sh
cmake -S cmake/hip-native -B build-hip-native -G Ninja \
  -DLFS_HIP_ROOT=/absolute/path/to/rocm-sdk \
  -DLFS_HIP_ARCHITECTURE=gfx1150
cmake --build build-hip-native
```

The SDK must provide `include/hip` and `lib/llvm/bin/clang++` (`clang++.exe` on
Windows). SDK package and HIP API version numbers are separate identifiers.
SDK 10.1.0 is the development reference. GPU execution was exercised on Windows
with gfx1150 hardware; other targets still need hardware qualification.
`LFS_HIP_ARCHITECTURES` accepts
an explicit semicolon-separated target list. The declared targets are not a
hardware support matrix: the selected SDK compiler and installed driver must
support the requested devices. This project supports native Windows and Linux
builds and rejects cross-compilation. Context creation rejects a GPU whose
architecture has no matching compiled code object. Wave size is reported by the
driver; the smoke kernels do not assume wave32 or wave64. Compression is optional
and fails configuration if explicitly requested from an unsupported compiler.

After building, `ctest --test-dir build-hip-native --output-on-failure` runs CPU
configuration checks and loads the module to verify its ABI and rejection of null
arguments. These checks require the SDK runtime dependencies, but do not query
devices or launch kernels. A missing module or dependency is a failure, not a skip.

GPU execution is opt-in and requires a compatible device and installed driver:

```sh
cmake -S cmake/hip-native -B build-hip-native -DLFS_HIP_NATIVE_GPU_TESTS=ON
cmake --build build-hip-native
ctest --test-dir build-hip-native --output-on-failure
```

GPU checks use ordinal 0 by default. Set `LFS_HIP_NATIVE_TEST_DEVICE` to a different
nonnegative ordinal when testing a specific compatible GPU; no automatic fallback
to another device occurs.

The test uses an absolute module path and, on Windows, an explicit SDK DLL directory
kept alive for the duration of the test. It does not modify PATH or the process-wide
DLL search policy. This developer setup does not package SDK libraries. Application
loading and distribution integration are separate work.

The private versioned C ABI exposes inventory, contexts, owned buffers, blocking
host transfers, synchronization, fill and affine smoke kernels. No HIP types or C++
exceptions cross the ABI. Calls on one context must be serialized by the caller;
contexts and handles must remain valid until their destruction succeeds. Buffer
ownership and ranges are checked before use. Context destruction drains queued
work and frees remaining buffers; a driver cleanup failure retains the context for
an explicit retry. Errors use caller-owned storage and never trigger CPU fallback.
Fill and affine calls enqueue work and check launch status immediately; they do
not wait for completion. Host transfers and explicit synchronization wait for the
context queue. Callers must keep borrowed host storage alive until a transfer
completes; a failed synchronization must not be interpreted as successful drainage.

The mathematical smoke checks cover non-aligned lengths and exact FP32 values;
they are not performance benchmarks or training-quality claims. Public Tensor
contracts, advanced operators, HIPRTC, renderer interop and trainer integration
belong to later changes. CPU-only configuration regressions can run without an SDK:

```sh
python tests/python/test_hip_runtime_configuration.py
```

On Linux, the SDK can be installed into an isolated development environment using
[AMD's installation instructions](https://rocm.docs.amd.com/en/develop/install/rocm.html):

```sh
python3 -m venv /path/to/hip-sdk-env
. /path/to/hip-sdk-env/bin/activate
python -m pip install --index-url https://stable.repo.amd.com/rocm/whl-next/ 'rocm[devel]==10.1.0'
cmake -S cmake/hip-native -B build-hip-native -G Ninja \
  -DLFS_HIP_ROOT="$(rocm-sdk path --root)" \
  -DLFS_HIP_ARCHITECTURE=gfx1150
cmake --build build-hip-native
ctest --test-dir build-hip-native --output-on-failure
```

Linux compilation also needs the host C++ standard-library development headers.
Explicit architecture selection allows compilation on machines without an AMD
GPU, including WSL or CI runners. The native module does not load Python or
PyTorch, but still requires the SDK shared libraries installed in that environment.
This is a developer setup, not a redistributable runtime bundle.

## Owned asynchronous transfers

Contexts can create reusable pinned staging slots for asynchronous upload,
readback and queue markers. A staged upload copies the caller's input into
module-owned host storage before returning, so the original input can immediately
be reused. A readback snapshots device data on the same context queue. Subsequent
operations on that queue remain ordered with both transfers.

`readback_wait` waits for completion without consuming it; `readback_poll` reports
readiness and consumes a completed snapshot. Passing a null destination discards
the snapshot. A non-null destination must provide its writable size through
`destination_bytes`; insufficient space fails without writing or consuming the
snapshot, allowing a retry with a larger buffer. Host subranges may overlap the
staging allocation: both upload snapshotting and poll use overlap-safe copies.
A slot remains busy until completion is consumed, even after a
successful wait. Slots that have allocated data storage cannot be reused as
zero-byte queue markers; dedicated markers can be re-recorded to track newer
work on the same queue.

For packed readback, reserve a host slot and append device ranges on the same
queue. Both append bounds and the poll copy size use the requested reservation,
not the capacity retained from an earlier, larger transfer. The returned host
address is borrowed: wait before reading it, and do not keep it across capacity
growth or slot destruction. No host pointer may be retained across context or
slot destruction.

Destroy staging slots before their context. Context destruction fails while
slots remain registered, preserving the context for cleanup and retry. Slot
destruction drains pending transfers before releasing events or pinned storage;
if cleanup fails, retry destruction rather than reusing the partially cleaned
slot. A failed event recording retains the pending transfer and falls back to
stream completion checks, so pinned memory is not freed while a copy may use it.

An actual copy-submission failure is latched separately from event-recording
failure: wait and poll keep returning the copy error and never present old or
partial staging bytes as a successful snapshot. Once the queue has completed,
`readback_poll(context, slot, nullptr, 0, ...)` consumes that failed result but
still returns the error with `ready=0`. The slot may then be reused. Packed
appends cannot continue after a failed copy until the failed result is discarded.

Growing an idle staging slot can discard its previous contents and cached
allocation before attempting the replacement allocation. Failure does not retain
the old capacity, but the slot remains owned and can be retried or destroyed.
All these operations stay behind the private Tensor runtime boundary; they do
not register a backend or change the application's image-loading path.

With GPU tests explicitly enabled, the build also creates a separate
`lfs_hip_runtime_faults` module and registers `HipNative.FaultRecovery`. Its
test-only forced-include header simulates individual SDK failures for copies,
pinned allocation/release, event creation/recording/destruction and stream
drainage. The tests check recovery, preservation of valid copies after event
errors, rejection of invalid snapshots and balanced resource counts. The normal
runtime contains none of these injection entry points; CPU-only builds do not
build the instrumented module. These tests exercise API error handling on a
real device, not physical device-loss or GPU-hang recovery.

# HIP host adapter checks

This standalone project compiles the SDK-free host module loader and native
context/buffer ownership layer. It needs a host C++20 compiler and CMake 3.30,
but no HIP SDK, CUDA toolkit, application build or GPU for its default tests.
It does not register an application backend or enable training.

```sh
cmake -S cmake/hip-host -B build-hip-host -G Ninja -DCMAKE_BUILD_TYPE=Release
cmake --build build-hip-host
ctest --test-dir build-hip-host --output-on-failure
```

Five deliberately incomplete libraries test early rejection of ABI revisions
1/2/3/4 and missing revision-5 exports. The host resolves the ABI first, then binds
the typed export table. Module paths must be absolute. Windows uses a Unicode
path and explicit DLL-directory/default-directory lookup, rather than the
working directory. SDK dependencies must be installed or packaged appropriately.

For a CPU-only export check against a real revision-5 module, configure
`LFS_HIP_HOST_TEST_MODULE` with its absolute path. On Windows the test can add an
explicit SDK dependency directory through `LFS_HIP_HOST_TEST_RUNTIME_DIR` for
its lifetime, without changing the process-wide DLL search policy.

GPU context tests are opt-in with `LFS_HIP_HOST_GPU_TESTS=ON` and require that
explicit real module and device ordinal 0. They exercise typed dispatch, buffer
bounds/ownership, serialized caller threads and a buffer retaining its context
and module after the context wrapper is destroyed. The optional
`LFS_HIP_HOST_FAULT_MODULE` selects the separately built instrumented module for
allocation/release and context-cleanup recovery checks. No production fault hooks
are required or resolved by the host loader.

The host serializes native calls on each context. Buffer leases pin the context
and loaded module. Explicit close rejects live buffers and retains state on a
native cleanup failure for retry. If final context destruction fails, a
preallocated retirement node retains both native handle and module; it reports
the failure to stderr. `retry_retired_contexts` retries cleanup. Unrecoverable
driver failures may retain that ownership until process exit; unloading code
beneath an undrained context is never treated as successful cleanup.

The opt-in public Tensor facade adapter and shared contracts are documented in
`docs/docs/development/hip-tensor-backend.md`. This standalone project remains
independent of the application and GPU SDK at compile time.

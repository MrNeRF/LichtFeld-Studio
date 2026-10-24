# Shared tensor operation contracts

`tensor_operation_contracts` exercises the same public Tensor operations on the
CPU and each compiled GPU backend. It is available with `BUILD_FORMAT_TESTS`,
`BUILD_VISUALIZER_TESTS` or `BUILD_TESTS` and does not require LibTorch. The
application's normal build dependencies still apply.

```sh
cmake --build <build> --target tensor_operation_contracts --parallel 2
ctest --test-dir <build> -R '^TensorOperationContracts[.]CPU$' --output-on-failure --no-tests=error
```

The Windows Release/Debug, Linux Release/Debug and macOS workflows run the CPU
selection. Loading a CUDA-enabled Core library on Linux without a driver needs
the same driver stub used by the existing CPU tests. These runs do not establish
GPU coverage.

On a machine with the relevant GPU, run the registered backend contract, for
example:

```sh
ctest --test-dir <build> -R '^TensorOperationContracts[.]Vulkan$' --output-on-failure --no-tests=error
```

The other current backend names are `CUDA` and `Metal`. Each GPU CTest entry
requires that backend to be available and fails if it is missing. By contrast,
running the executable without a required backend skips unavailable GPU cases.
An empty GoogleTest selection is an error. Test listing remains available with
`--gtest_list_tests` and does not probe GPU availability.

`tensor_backend_contract.hpp` provides the shared fixture and enumerates
`kCompiledGpuBackends` without creating contexts during test discovery. Selection
is scoped to each test; the process default is not changed. The existing semantic
edge cases use the same fixture and remain part of the full test executable.

New operation contracts should call public Tensor APIs, verify device/backend
placement and use analytic results or an independent host calculation. Exact
movement and representable arithmetic use exact comparisons. Floating-point
algorithms use a justified, backend-independent tolerance. Random sampling tests
check deterministic degenerate distributions rather than requiring identical
random sequences across different generators. Sort tests verify keys and a valid
permutation without imposing an undocumented order for equal keys.

The semantic suite also contains low-level dispatch and error tests; only that
translation unit receives private Tensor access in this target. Native stream,
interop, shader diagnostics and cross-backend transfer tests remain in their
dedicated suites. The existing CUDA/Vulkan parity tests are retained as additional
comparisons, not used as the reference for the shared operation contracts.

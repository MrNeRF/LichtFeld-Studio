# Shared GPU configuration policies on CPU

This standalone project tests the public selection/restart policy with synthetic
adapters. It links neither Core nor CUDA, HIP, Vulkan or Metal runtimes. One
executable builds without HIP declarations, and another enables the HIP data
structures without requiring its SDK. Cases cover backend changes, independent
renderer/Tensor selections, active identity restoration, unavailable devices,
inactive options, duplicate names and enumeration reordering.

```sh
cmake -S tests/gpu_policy -B build-gpu-policy -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DLFS_CPU_DEPENDENCY_INCLUDE=/path/to/dependencies/include
cmake --build build-gpu-policy
ctest --test-dir build-gpu-policy --output-on-failure --no-tests=error
```

A C++23 compiler and GLM headers are required. Explicitly enabling
`LFS_CPU_FETCH_GLM=ON` downloads GLM 1.0.3 from its official archive only when the
headers are absent; the archive hash matches the application's vcpkg version.
No other dependency is fetched by this project.

`LFS_CPU_PYTHON_UI_TESTS=ON` adds the shared Python preference, rendering and keymap
contracts. Install pytest (`pytest>=7,<10`) in the selected interpreter and pass
`Python3_EXECUTABLE` if necessary. CTest disables unrelated pytest plugins and
points the fixtures away from any application build, so this is a synthetic UI
suite, not native binding or GPU validation.

The existing Windows, Linux and macOS build workflows run these contracts in a
separate CPU executable without linking the application or GPU SDKs. All five
configurations run it: Windows Release/Debug, Linux Release/Debug and macOS
Release. The helper
`tools/run_gpu_policy_cpu.py --application-build <build>` reuses the configured
compiler and GLM headers and creates an isolated pytest environment. The manual HIP build
workflows include the same project in their CPU suite. All tests carry the `cpu`
label; none carries `gpu`. Native allocation, kernel, rendering and interop tests
remain in their respective backend suites. These CPU checks do not establish
runtime correctness on hardware.

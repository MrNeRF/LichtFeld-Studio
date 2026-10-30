# SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
# SPDX-License-Identifier: GPL-3.0-or-later
cmake_minimum_required(VERSION 3.30)
project(LichtFeldHipNative LANGUAGES NONE)
# SDK-only Tensor runtime check. No application, CUDA toolkit, or GPU is required.
get_filename_component(_root "${CMAKE_CURRENT_LIST_DIR}/../.." ABSOLUTE)
# Defined before adding the runtime so an explicitly requested GPU test build
# can also compile its separate fault-injection module.
option(LFS_HIP_NATIVE_GPU_TESTS "Register SDK-only HIP GPU boundary tests" OFF)
add_subdirectory("${_root}/src/core/tensor/backend/hip" runtime)

enable_testing()
find_package(Python3 REQUIRED COMPONENTS Interpreter)
add_test(NAME HipNative.CiFilter
    COMMAND "${Python3_EXECUTABLE}" "${_root}/tests/python/test_hip_native_ci_filter.py")
set_tests_properties(HipNative.CiFilter PROPERTIES LABELS "cpu;hip" TIMEOUT 60)
add_test(NAME HipNative.Configuration
    COMMAND "${Python3_EXECUTABLE}" "${_root}/tests/python/test_hip_runtime_configuration.py")
set_tests_properties(HipNative.Configuration PROPERTIES
    ENVIRONMENT "CMAKE_COMMAND=${CMAKE_COMMAND}" LABELS "cpu;hip" TIMEOUT 60)
add_test(NAME HipNative.PointwiseCatalog
    COMMAND "${Python3_EXECUTABLE}" "${_root}/tests/python/test_hip_pointwise_catalog.py")
set_tests_properties(HipNative.PointwiseCatalog PROPERTIES LABELS "cpu;hip" TIMEOUT 60)
add_test(NAME HipNative.ModuleBoundary
    COMMAND "${Python3_EXECUTABLE}" "${_root}/tests/hip/test_native_runtime.py"
        "${LFS_HIP_RUNTIME_LIBRARY}" --runtime-dir "${LFS_HIP_ROOT}/bin" --abi-only)
set_tests_properties(HipNative.ModuleBoundary PROPERTIES LABELS "cpu;hip" TIMEOUT 30)
add_test(NAME HipNative.TensorModuleBoundary
    COMMAND "${Python3_EXECUTABLE}" "${_root}/tests/hip/test_tensor_operations.py"
        "${LFS_HIP_RUNTIME_LIBRARY}" --runtime-dir "${LFS_HIP_ROOT}/bin" --abi-only)
set_tests_properties(HipNative.TensorModuleBoundary PROPERTIES LABELS "cpu;hip" TIMEOUT 30)

# GPU execution is explicit; merely configuring or compiling needs no AMD GPU.
if(LFS_HIP_NATIVE_GPU_TESTS)
    set(LFS_HIP_NATIVE_TEST_DEVICE "0" CACHE STRING "HIP ordinal for explicit native GPU tests")
    add_test(NAME HipNative.RuntimeBoundary
        COMMAND "${Python3_EXECUTABLE}" "${_root}/tests/hip/test_native_runtime.py"
            "${LFS_HIP_RUNTIME_LIBRARY}" --runtime-dir "${LFS_HIP_ROOT}/bin"
            --device "${LFS_HIP_NATIVE_TEST_DEVICE}")
    set_tests_properties(HipNative.RuntimeBoundary PROPERTIES LABELS "hip;gpu" TIMEOUT 120)
    add_test(NAME HipNative.FaultRecovery
        COMMAND "${Python3_EXECUTABLE}" "${_root}/tests/hip/test_native_runtime.py"
            "${LFS_HIP_FAULT_LIBRARY}" --runtime-dir "${LFS_HIP_ROOT}/bin"
            --device "${LFS_HIP_NATIVE_TEST_DEVICE}" --faults)
    set_tests_properties(HipNative.FaultRecovery PROPERTIES LABELS "hip;gpu" TIMEOUT 120)
    add_test(NAME HipNative.TensorOperations
        COMMAND "${Python3_EXECUTABLE}" "${_root}/tests/hip/test_tensor_operations.py"
            "${LFS_HIP_RUNTIME_LIBRARY}" --runtime-dir "${LFS_HIP_ROOT}/bin"
            --device "${LFS_HIP_NATIVE_TEST_DEVICE}")
    set_tests_properties(HipNative.TensorOperations PROPERTIES LABELS "hip;gpu" TIMEOUT 120)
    add_test(NAME HipNative.TensorWorkspaceRecovery
        COMMAND "${Python3_EXECUTABLE}" "${_root}/tests/hip/test_tensor_operations.py"
            "${LFS_HIP_FAULT_LIBRARY}" --runtime-dir "${LFS_HIP_ROOT}/bin"
            --device "${LFS_HIP_NATIVE_TEST_DEVICE}" --faults)
    set_tests_properties(HipNative.TensorWorkspaceRecovery PROPERTIES LABELS "hip;gpu" TIMEOUT 120)

endif()

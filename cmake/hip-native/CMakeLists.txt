# SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
# SPDX-License-Identifier: GPL-3.0-or-later
cmake_minimum_required(VERSION 3.30)
project(LichtFeldHipNative LANGUAGES NONE)
# SDK-only Tensor runtime check. No application, CUDA toolkit, or GPU is required.
get_filename_component(_root "${CMAKE_CURRENT_LIST_DIR}/../.." ABSOLUTE)
add_subdirectory("${_root}/src/core/tensor/backend/hip" runtime)

enable_testing()
find_package(Python3 REQUIRED COMPONENTS Interpreter)
add_test(NAME HipNative.Configuration
    COMMAND "${Python3_EXECUTABLE}" "${_root}/tests/python/test_hip_runtime_configuration.py")
set_tests_properties(HipNative.Configuration PROPERTIES
    ENVIRONMENT "CMAKE_COMMAND=${CMAKE_COMMAND}" LABELS "cpu;hip" TIMEOUT 60)

# GPU execution is explicit; merely configuring or compiling needs no AMD GPU.
option(LFS_HIP_NATIVE_GPU_TESTS "Register SDK-only HIP GPU boundary tests" OFF)
if(LFS_HIP_NATIVE_GPU_TESTS)
    add_test(NAME HipNative.RuntimeBoundary
        COMMAND "${Python3_EXECUTABLE}" "${_root}/tests/hip/test_native_runtime.py"
            "${LFS_HIP_RUNTIME_LIBRARY}" --runtime-dir "${LFS_HIP_ROOT}/bin")
    set_tests_properties(HipNative.RuntimeBoundary PROPERTIES LABELS "hip;gpu" TIMEOUT 120)
endif()

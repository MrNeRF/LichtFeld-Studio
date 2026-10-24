# SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
# SPDX-License-Identifier: GPL-3.0-or-later
# Independent of the application and its CXX/CUDA compiler configuration.
if(CMAKE_CROSSCOMPILING)
    message(FATAL_ERROR "The native HIP runtime does not support cross-compilation")
endif()
if(NOT CMAKE_SYSTEM_NAME MATCHES "^(Windows|Linux)$")
    message(FATAL_ERROR "The native HIP runtime supports Windows and Linux")
endif()
set(LFS_HIP_ROOT "" CACHE PATH "Explicit ROCm development SDK root")
set(LFS_HIP_ARCHITECTURE "" CACHE STRING "Explicit HIP GPU target, e.g. gfx1150")
include("${CMAKE_CURRENT_LIST_DIR}/../../../../../cmake/HipArchitectures.cmake")
lfs_hip_architectures()
if(WIN32)
    set(_compiler "${LFS_HIP_ROOT}/lib/llvm/bin/clang++.exe")
    set(_library "${CMAKE_BINARY_DIR}/lfs_hip_runtime.dll")
else()
    set(_compiler "${LFS_HIP_ROOT}/lib/llvm/bin/clang++")
    set(_library "${CMAKE_CURRENT_BINARY_DIR}/liblfs_hip_runtime.so")
    set(_platform_flags -fPIC "-Wl,-rpath,${LFS_HIP_ROOT}/lib" "-Wl,-rpath,${LFS_HIP_ROOT}/lib64")
endif()
if(NOT EXISTS "${_compiler}" OR NOT EXISTS "${LFS_HIP_ROOT}/include/hip/hip_runtime.h")
    message(FATAL_ERROR "LFS_HIP_ROOT must identify a complete ROCm development SDK")
endif()
find_package(Python3 REQUIRED COMPONENTS Interpreter)
set(_pointwise_source "${CMAKE_CURRENT_SOURCE_DIR}/../../../include/core/detail")
set(_pointwise_generated "${CMAKE_CURRENT_BINARY_DIR}/pointwise-generated")
set(_pointwise_outputs "${_pointwise_generated}/hip_pointwise_functors.hpp"
    "${_pointwise_generated}/hip_pointwise_f32.inc" "${_pointwise_generated}/hip_pointwise_f16.inc")
add_custom_command(OUTPUT ${_pointwise_outputs}
    COMMAND "${Python3_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/prepare_pointwise.py"
        --source "${_pointwise_source}" --output "${_pointwise_generated}"
    DEPENDS prepare_pointwise.py "${_pointwise_source}/tensor_functors.hpp" "${_pointwise_source}/pointwise_ops.def"
    VERBATIM)
add_custom_command(OUTPUT "${_library}"
    COMMAND "${_compiler}" -x hip -std=c++20 -O3 -shared
        ${_platform_flags} ${LFS_HIP_COMPRESSION_FLAGS}
        "-DLFS_HIP_COMPILED_ARCH=\"${LFS_HIP_COMPILED_ARCHITECTURES}\""
        -DLFS_HIP_RUNTIME_EXPORTS "-I${_pointwise_generated}" ${LFS_HIP_OFFLOAD_FLAGS}
        "--hip-path=${LFS_HIP_ROOT}"
        "${CMAKE_CURRENT_SOURCE_DIR}/runtime.hip" -o "${_library}"
    DEPENDS runtime.hip runtime_api.h architecture_runtime.hpp ${_pointwise_outputs}
        "${_compiler}" "${LFS_HIP_ROOT}/include/hip/hip_runtime.h"
    VERBATIM)
add_custom_target(lfs_hip_runtime_build ALL DEPENDS "${_library}")
set(LFS_HIP_RUNTIME_LIBRARY "${_library}" PARENT_SCOPE)

if(LFS_HIP_NATIVE_GPU_TESTS)
    get_filename_component(_root "${CMAKE_CURRENT_SOURCE_DIR}/../../../../.." ABSOLUTE)
    set(_fault_header "${_root}/tests/hip/runtime_faults.hpp")
    get_filename_component(_runtime_directory "${_library}" DIRECTORY)
    if(WIN32)
        set(_fault_library "${_runtime_directory}/lfs_hip_runtime_faults.dll")
    else()
        set(_fault_library "${_runtime_directory}/liblfs_hip_runtime_faults.so")
    endif()
    add_custom_command(OUTPUT "${_fault_library}"
        COMMAND "${_compiler}" -x hip -std=c++20 -O3 -shared
            ${_platform_flags} ${LFS_HIP_COMPRESSION_FLAGS}
            "-DLFS_HIP_COMPILED_ARCH=\"${LFS_HIP_COMPILED_ARCHITECTURES}\""
            -DLFS_HIP_RUNTIME_EXPORTS "-I${_pointwise_generated}" ${LFS_HIP_OFFLOAD_FLAGS}
            "--hip-path=${LFS_HIP_ROOT}" "-I${CMAKE_CURRENT_SOURCE_DIR}"
            -include "${_fault_header}" "${CMAKE_CURRENT_SOURCE_DIR}/runtime.hip" -o "${_fault_library}"
        DEPENDS runtime.hip runtime_api.h architecture_runtime.hpp "${_fault_header}" ${_pointwise_outputs}
            "${_compiler}" "${LFS_HIP_ROOT}/include/hip/hip_runtime.h"
        VERBATIM)
    add_custom_target(lfs_hip_runtime_faults ALL DEPENDS "${_fault_library}")
    set(LFS_HIP_FAULT_LIBRARY "${_fault_library}" PARENT_SCOPE)
endif()

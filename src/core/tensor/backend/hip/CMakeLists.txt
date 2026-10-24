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
add_custom_command(OUTPUT "${_library}"
    COMMAND "${_compiler}" -x hip -std=c++20 -O3 -shared
        ${_platform_flags} ${LFS_HIP_COMPRESSION_FLAGS}
        "-DLFS_HIP_COMPILED_ARCH=\"${LFS_HIP_COMPILED_ARCHITECTURES}\""
        -DLFS_HIP_RUNTIME_EXPORTS ${LFS_HIP_OFFLOAD_FLAGS}
        "--hip-path=${LFS_HIP_ROOT}"
        "${CMAKE_CURRENT_SOURCE_DIR}/runtime.hip" -o "${_library}"
    DEPENDS runtime.hip runtime_api.h architecture_runtime.hpp
        "${_compiler}" "${LFS_HIP_ROOT}/include/hip/hip_runtime.h"
    VERBATIM)
add_custom_target(lfs_hip_runtime_build ALL DEPENDS "${_library}")
set(LFS_HIP_RUNTIME_LIBRARY "${_library}" PARENT_SCOPE)

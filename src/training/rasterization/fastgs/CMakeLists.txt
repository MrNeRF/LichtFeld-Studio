# SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
#
# SPDX-License-Identifier: GPL-3.0-or-later

cmake_minimum_required(VERSION 3.30)

set(FASTLFS_SOURCES
        # rasterizer
        ${CMAKE_CURRENT_SOURCE_DIR}/rasterization/src/rasterization_api.cu
        ${CMAKE_CURRENT_SOURCE_DIR}/rasterization/src/forward.cu
        ${CMAKE_CURRENT_SOURCE_DIR}/rasterization/src/backward.cu
        # optimizer
        ${CMAKE_CURRENT_SOURCE_DIR}/optimizer/src/adam_api.cu
        ${CMAKE_CURRENT_SOURCE_DIR}/optimizer/src/adam.cu
)

# One unified library - renamed from fastgs_backend to fastlfs_backend
add_library(fastlfs_backend STATIC ${FASTLFS_SOURCES})

# PTX-only builds require separable compilation OFF
if(LFS_DEFER_DEVICE_SYMBOLS)
    set(_sep_comp OFF)
else()
    set(_sep_comp ON)
endif()

set_target_properties(fastlfs_backend PROPERTIES
        CUDA_ARCHITECTURES "${LichtFeld-Studio_CUDA_ARCH}"
        CUDA_SEPARABLE_COMPILATION ${_sep_comp}
        POSITION_INDEPENDENT_CODE ON
        CUDA_RESOLVE_DEVICE_SYMBOLS ON
)

target_include_directories(fastlfs_backend
        PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/rasterization/include
        ${CMAKE_CURRENT_SOURCE_DIR}/optimizer/include
        ${CMAKE_CURRENT_SOURCE_DIR}/utils
        ${CMAKE_SOURCE_DIR}/include
        ${CUDAToolkit_INCLUDE_DIRS}
        PRIVATE
)

# Pure CUDA libraries only
target_link_libraries(fastlfs_backend
        PUBLIC
        CUDA::cudart
        CUDA::curand
        ${CUDA_CUDA_LIBRARY}
        lfs_core_cuda  # CUDA utilities from core module (memory arena)
)

target_compile_options(fastlfs_backend PRIVATE
  # CUDA device code + MSVC host compiler flags (Windows only)
  $<$<AND:$<COMPILE_LANGUAGE:CUDA>,$<CXX_COMPILER_ID:MSVC>,$<CONFIG:Debug>>:-O0 -g -G -lineinfo -Xcompiler=/Od -Xcompiler=/Z7>
  $<$<AND:$<COMPILE_LANGUAGE:CUDA>,$<CXX_COMPILER_ID:MSVC>,$<CONFIG:Release>>:-O3 --use_fast_math --expt-relaxed-constexpr --diag-suppress=20012,186,221 -Xcompiler=/O2 -Xcompiler=/DNDEBUG>

  # CUDA on Windows: Pass /utf-8 to MSVC when invoked by nvcc
  $<$<AND:$<COMPILE_LANGUAGE:CUDA>,$<CXX_COMPILER_ID:MSVC>>:
    -Xcompiler=/utf-8
    -Xcompiler=/D_CRT_SECURE_NO_WARNINGS
    --diag-suppress=27  # Suppress "character value out of range" warnings from fmt
  >

  # C++ MSVC
  $<$<AND:$<COMPILE_LANGUAGE:CXX>,$<CXX_COMPILER_ID:MSVC>,$<CONFIG:Debug>>:/Od /Z7>
  $<$<AND:$<COMPILE_LANGUAGE:CXX>,$<CXX_COMPILER_ID:MSVC>,$<CONFIG:Release>>:/O2 /DNDEBUG>

  # C++ no-MSVC (GCC/Clang/AppleClang/IntelLLVM)
  $<$<AND:$<COMPILE_LANGUAGE:CXX>,$<NOT:$<CXX_COMPILER_ID:MSVC>>,$<CONFIG:Debug>>:-O0 -g>
  $<$<AND:$<COMPILE_LANGUAGE:CXX>,$<NOT:$<CXX_COMPILER_ID:MSVC>>,$<CONFIG:Release>>:-O3 -DNDEBUG>
)

# Windows: Disable fmt Unicode check for CUDA files (nvcc doesn't properly propagate /utf-8)
if(MSVC)
    target_compile_definitions(fastlfs_backend PRIVATE
        $<$<COMPILE_LANGUAGE:CUDA>:FMT_UNICODE=0>
    )
endif()

if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    target_compile_definitions(fastlfs_backend PRIVATE _DEBUG DEBUG_BUILD)
elseif(CMAKE_BUILD_TYPE STREQUAL "Release")
    target_compile_definitions(fastlfs_backend PRIVATE RELEASE_BUILD)
endif()

# SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
#
# SPDX-License-Identifier: GPL-3.0-or-later

# Add forward-only gsplat backend subdirectory
add_subdirectory(gsplat_fwd)

set(RASTERIZER_SOURCES
        ${CMAKE_CURRENT_SOURCE_DIR}/rasterization/src/forward.cu
        ${CMAKE_CURRENT_SOURCE_DIR}/rasterization/src/gsplat_forward.cu
        ${CMAKE_CURRENT_SOURCE_DIR}/rasterization/src/rasterization_api_tensor.cu
)

add_library(lfs_rasterizer STATIC ${RASTERIZER_SOURCES})

# PTX-only builds require separable compilation OFF
if(LFS_DEFER_DEVICE_SYMBOLS)
    set(_sep_comp OFF)
else()
    set(_sep_comp ON)
endif()

set_target_properties(lfs_rasterizer PROPERTIES
        CUDA_ARCHITECTURES "${LichtFeld-Studio_CUDA_ARCH}"
        CUDA_SEPARABLE_COMPILATION ${_sep_comp}
        POSITION_INDEPENDENT_CODE ON
        CUDA_RESOLVE_DEVICE_SYMBOLS ON
)

target_include_directories(lfs_rasterizer
        PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/rasterization/include
        ${CMAKE_CURRENT_SOURCE_DIR}/utils
        PRIVATE
        ${CMAKE_SOURCE_DIR}/include
        ${CMAKE_CURRENT_SOURCE_DIR}  # For gsplat_fwd headers (gsplat_fwd/Ops.h, gsplat_fwd/Common.h)
        ${CUDAToolkit_INCLUDE_DIRS}
)

target_link_libraries(lfs_rasterizer
        PUBLIC
        CUDA::cudart
        CUDA::curand
        CUDA::cuda_driver
        spdlog::spdlog
        lfs_core_cuda       # Required for shared memory arena
        gsplat_fwd          # Forward-only GUT rasterization backend (no training dependency)
)

target_compile_options(lfs_rasterizer PRIVATE
        # CUDA
        $<$<AND:$<COMPILE_LANGUAGE:CUDA>,$<CONFIG:Debug>>:-O0 -g -G -lineinfo --expt-relaxed-constexpr>
        $<$<AND:$<COMPILE_LANGUAGE:CUDA>,$<CONFIG:Release>>:-O3 --use_fast_math --expt-relaxed-constexpr --diag-suppress=20012,186,221>

        # CUDA on Windows: Pass /utf-8 to MSVC when invoked by nvcc
        $<$<AND:$<COMPILE_LANGUAGE:CUDA>,$<CXX_COMPILER_ID:MSVC>>:
            -Xcompiler=/utf-8
            -Xcompiler=/D_CRT_SECURE_NO_WARNINGS
            --diag-suppress=27  # Suppress "character value out of range" warnings from fmt
        >

        # C++ MSVC
        $<$<AND:$<COMPILE_LANGUAGE:CXX>,$<CXX_COMPILER_ID:MSVC>,$<CONFIG:Debug>>:/Od /Z7>
        $<$<AND:$<COMPILE_LANGUAGE:CXX>,$<CXX_COMPILER_ID:MSVC>,$<CONFIG:Release>>:/O2 /DNDEBUG>

        # C++ non-MSVC
        $<$<AND:$<COMPILE_LANGUAGE:CXX>,$<NOT:$<CXX_COMPILER_ID:MSVC>>,$<CONFIG:Debug>>:-O0 -g>
        $<$<AND:$<COMPILE_LANGUAGE:CXX>,$<NOT:$<CXX_COMPILER_ID:MSVC>>,$<CONFIG:Release>>:-O3 -DNDEBUG>
)

# Windows: Disable fmt Unicode check for CUDA files (nvcc doesn't properly propagate /utf-8)
if(MSVC)
    target_compile_definitions(lfs_rasterizer PRIVATE
        $<$<COMPILE_LANGUAGE:CUDA>:FMT_UNICODE=0>
    )
endif()

if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    target_compile_definitions(lfs_rasterizer PRIVATE _DEBUG DEBUG_BUILD)
elseif(CMAKE_BUILD_TYPE STREQUAL "Release")
    target_compile_definitions(lfs_rasterizer PRIVATE RELEASE_BUILD)
endif()

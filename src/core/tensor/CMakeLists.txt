# SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
# SPDX-License-Identifier: GPL-3.0-or-later

# ============================================================================
# lfs_tensor_kernels - CUDA kernels for tensor operations
# The C++ tensor sources are now compiled as part of lfs_core (SHARED).
# Only the CUDA .cu files remain here as a STATIC library linked privately.
# ============================================================================

set(LFS_TENSOR_CUDA_SOURCES
    tensor_ops.cu
    tensor_warp_reduce.cu
    tensor_matrix_ops.cu
    tensor_broadcast_ops.cu
    tensor_masking_ops.cu
    tensor_random_ops.cu
    tensor_strided_ops.cu
    tensor_dot_optimized.cu
    tensor_nn_ops.cu
)

if(LFS_TENSOR_CUDA_SOURCES)
    add_library(lfs_tensor_kernels STATIC ${LFS_TENSOR_CUDA_SOURCES})

    # For PTX-only builds, disable separable compilation to avoid device linking issues
    if(LFS_DEFER_DEVICE_SYMBOLS)
        set(_separable_compilation OFF)
    else()
        set(_separable_compilation ON)
    endif()

    set_target_properties(lfs_tensor_kernels PROPERTIES
            CUDA_ARCHITECTURES "${LichtFeld-Studio_CUDA_ARCH}"
            CUDA_SEPARABLE_COMPILATION ${_separable_compilation}
            POSITION_INDEPENDENT_CODE ON
            CUDA_RESOLVE_DEVICE_SYMBOLS ON
            CUDA_STANDARD 20
            CUDA_STANDARD_REQUIRED ON
    )

    target_include_directories(lfs_tensor_kernels
            PUBLIC
            ${CMAKE_CURRENT_SOURCE_DIR}/../include
            ${CMAKE_BINARY_DIR}/include
            PRIVATE
            ${CMAKE_CURRENT_SOURCE_DIR}
            ${CUDAToolkit_INCLUDE_DIRS}
    )

    target_link_libraries(lfs_tensor_kernels
            PUBLIC
            CUDA::cudart
            CUDA::curand
            spdlog::spdlog
    )

    target_compile_options(lfs_tensor_kernels PRIVATE
            $<$<AND:$<COMPILE_LANGUAGE:CUDA>,$<CXX_COMPILER_ID:MSVC>,$<CONFIG:Debug>>:-O0 -g ${LFS_CUDA_DEVICE_DEBUG_FLAGS} -lineinfo --extended-lambda --expt-relaxed-constexpr -Xcompiler=/Od -Xcompiler=/Z7 -Xcompiler=/utf-8 -Xcompiler=/D_CRT_SECURE_NO_WARNINGS -Xcompiler=/bigobj --diag-suppress=27>
            $<$<AND:$<COMPILE_LANGUAGE:CUDA>,$<CXX_COMPILER_ID:MSVC>,$<CONFIG:Release>>:-O3 -use_fast_math --extended-lambda --expt-relaxed-constexpr --ptxas-options=-v -Xcompiler=/O2 -Xcompiler=/DNDEBUG -Xcompiler=/utf-8 -Xcompiler=/D_CRT_SECURE_NO_WARNINGS -Xcompiler=/bigobj --diag-suppress=27>
            $<$<AND:$<COMPILE_LANGUAGE:CUDA>,$<NOT:$<CXX_COMPILER_ID:MSVC>>,$<CONFIG:Debug>>:-O0 -g ${LFS_CUDA_DEVICE_DEBUG_FLAGS} -lineinfo --extended-lambda --expt-relaxed-constexpr>
            $<$<AND:$<COMPILE_LANGUAGE:CUDA>,$<NOT:$<CXX_COMPILER_ID:MSVC>>,$<CONFIG:Release>>:-O3 -use_fast_math --extended-lambda --expt-relaxed-constexpr --ptxas-options=-v>
    )

    target_compile_definitions(lfs_tensor_kernels PRIVATE LFS_CORE_EXPORTS)

    if(MSVC)
        target_compile_definitions(lfs_tensor_kernels PRIVATE
            $<$<COMPILE_LANGUAGE:CUDA>:FMT_UNICODE=0>
        )
    endif()

    # Windows NVCC 12.8 ICE workaround
    if(WIN32)
        set_source_files_properties(
            tensor_broadcast_ops.cu
            tensor_ops.cu
            PROPERTIES
            COMPILE_FLAGS "-O0"
        )
    endif()
endif()

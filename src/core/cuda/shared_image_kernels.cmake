# SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
# SPDX-License-Identifier: GPL-3.0-or-later

# Core-owned leaf target: IO and training share these unchanged image kernels.
if(LFS_HAS_CUDA)
add_library(lfs_shared_image_cuda STATIC
        ${CMAKE_SOURCE_DIR}/src/io/cuda/image_format_kernels.cuh
        ${CMAKE_SOURCE_DIR}/src/io/cuda/image_format_kernels.cu
)

target_include_directories(lfs_shared_image_cuda
        PUBLIC
        ${CMAKE_SOURCE_DIR}/src/io
        PRIVATE
        ${CMAKE_SOURCE_DIR}/src/core/include
        ${CMAKE_SOURCE_DIR}/src
)

target_link_libraries(lfs_shared_image_cuda
        PRIVATE
        lfs_diagnostics
        CUDA::cudart
)

set_target_properties(lfs_shared_image_cuda PROPERTIES
        CUDA_SEPARABLE_COMPILATION OFF
        CUDA_STANDARD 20
        CUDA_STANDARD_REQUIRED ON
        POSITION_INDEPENDENT_CODE ON
)

# CUDA compiler options for lfs_shared_image_cuda
target_compile_options(lfs_shared_image_cuda PRIVATE
        # CUDA device code + MSVC host compiler flags (Windows only)
        $<$<AND:$<COMPILE_LANGUAGE:CUDA>,$<CXX_COMPILER_ID:MSVC>,$<CONFIG:Debug>>:-O0 -g --extended-lambda --expt-relaxed-constexpr -Xcompiler=/utf-8 -Xcompiler=/D_CRT_SECURE_NO_WARNINGS --diag-suppress=27>
        $<$<AND:$<COMPILE_LANGUAGE:CUDA>,$<CXX_COMPILER_ID:MSVC>,$<CONFIG:Release>>:-O3 -use_fast_math --extended-lambda --expt-relaxed-constexpr -Xcompiler=/DNDEBUG -Xcompiler=/utf-8 -Xcompiler=/D_CRT_SECURE_NO_WARNINGS --diag-suppress=27>

        # CUDA device code for non-Windows
        $<$<AND:$<COMPILE_LANGUAGE:CUDA>,$<NOT:$<CXX_COMPILER_ID:MSVC>>,$<CONFIG:Debug>>:-O0 -g --extended-lambda --expt-relaxed-constexpr>
        $<$<AND:$<COMPILE_LANGUAGE:CUDA>,$<NOT:$<CXX_COMPILER_ID:MSVC>>,$<CONFIG:Release>>:-O3 -use_fast_math --extended-lambda --expt-relaxed-constexpr>
)

if(MSVC)
    target_compile_definitions(lfs_shared_image_cuda PRIVATE
            $<$<COMPILE_LANGUAGE:CUDA>:FMT_UNICODE=0>
    )
endif()
endif()

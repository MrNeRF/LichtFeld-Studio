# SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
# SPDX-License-Identifier: GPL-3.0-or-later

# ============================================================================
# lfs_tensor - LichtFeld Studio Tensor Library
# ============================================================================
#
# A high-performance tensor library with CPU/CUDA support, featuring:
# - Expression templates for lazy evaluation and kernel fusion
# - Optimized CUDA kernels (2.4x faster rasterization)
# - Broadcasting, reductions, masking, and advanced operations
# - Warp-level reductions inspired by tiny-cuda-nn
#
# Public API:
#   #include "core/tensor.hpp"
#
# All implementation details are private in internal/ directory.
# ============================================================================

# C++ sources (can use C++23)
set(LFS_TENSOR_SOURCES
    tensor.cpp              # Main tensor implementation (includes expr templates)
    tensor_factory.cpp      # Factory methods (empty, zeros, ones, rand, etc.) - moved from header
    tensor_utils.cpp        # Minimal utilities (MemoryInfo)
    tensor_matrix_ops.cpp   # Matrix operations (matmul, transpose, etc.)
    tensor_unified_ops.cpp  # Unified operations (load, unary, binary, reduce, ternary)
    tensor_movement_ops.cpp # Movement operations (reshape, permute, etc.)
    tensor_random_ops.cpp   # Random generation operations
    tensor_broadcast.cpp    # Broadcasting implementation
    tensor_shape_ops.cpp       # Shape functions (reshape, slice, etc.)
    tensor_masking_ops.cpp     # Masking and indexing operations
    tensor_advanced_ops.cpp    # Advanced operations
    tensor_row_proxy.cpp       # TensorRowProxy implementations (moved from header)
    pinned_memory_allocator.cpp # Pinned memory allocator for fast CPU-GPU transfers (used by tensor)
    offset_allocator.cpp       # OffsetAllocator for O(1) GPU memory sub-allocation
    cuda_stream_pool.cpp       # CUDA stream pool for parallel execution
)

# CUDA sources (limited to C++20)
set(LFS_TENSOR_CUDA_SOURCES
    tensor_ops.cu           # CUDA kernels for tensor operations
    tensor_warp_reduce.cu   # Optimized warp-level reduction kernels (tiny-cuda-nn inspired)
    tensor_matrix_ops.cu    # CUDA kernels for matrix operations
    tensor_broadcast_ops.cu # CUDA kernels for broadcasting
    tensor_masking_ops.cu   # CUDA kernels for masking/indexing
    tensor_random_ops.cu    # CUDA kernels for random ops
    tensor_strided_ops.cu   # CUDA kernels for strided tensor operations
    tensor_dot_optimized.cu # Optimized dot product and scalar reduction kernels
)

# Create CUDA library for tensor operations (C++20)
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
            CUDA_STANDARD 20  # CUDA only supports up to C++20
            CUDA_STANDARD_REQUIRED ON
    )

    target_include_directories(lfs_tensor_kernels
            PUBLIC
            ${CMAKE_CURRENT_SOURCE_DIR}/../include  # core/include for core headers
            ${CMAKE_BINARY_DIR}/include
            PRIVATE
            ${CMAKE_CURRENT_SOURCE_DIR}  # For internal/ headers
            ${CUDAToolkit_INCLUDE_DIRS}
    )

    target_link_libraries(lfs_tensor_kernels
            PUBLIC
            CUDA::cudart
            CUDA::curand      # For random number generation
            spdlog::spdlog
    )

    target_compile_options(lfs_tensor_kernels PRIVATE
            # CUDA device code + MSVC host compiler flags (Windows only)
            $<$<AND:$<COMPILE_LANGUAGE:CUDA>,$<CXX_COMPILER_ID:MSVC>,$<CONFIG:Debug>>:-O0 -g -G -lineinfo --extended-lambda --expt-relaxed-constexpr -Xcompiler=/Od -Xcompiler=/Z7 -Xcompiler=/utf-8 -Xcompiler=/D_CRT_SECURE_NO_WARNINGS --diag-suppress=27>
            $<$<AND:$<COMPILE_LANGUAGE:CUDA>,$<CXX_COMPILER_ID:MSVC>,$<CONFIG:Release>>:-O3 -use_fast_math --extended-lambda --expt-relaxed-constexpr --ptxas-options=-v -Xcompiler=/O2 -Xcompiler=/DNDEBUG -Xcompiler=/utf-8 -Xcompiler=/D_CRT_SECURE_NO_WARNINGS --diag-suppress=27>

            # CUDA device code for non-Windows
            $<$<AND:$<COMPILE_LANGUAGE:CUDA>,$<NOT:$<CXX_COMPILER_ID:MSVC>>,$<CONFIG:Debug>>:-O0 -g -G -lineinfo --extended-lambda --expt-relaxed-constexpr>
            $<$<AND:$<COMPILE_LANGUAGE:CUDA>,$<NOT:$<CXX_COMPILER_ID:MSVC>>,$<CONFIG:Release>>:-O3 -use_fast_math --extended-lambda --expt-relaxed-constexpr --ptxas-options=-v>
    )

    # Windows: Add compile definition to tell fmt that UTF-8 is enabled for CUDA files
    if(MSVC)
        target_compile_definitions(lfs_tensor_kernels PRIVATE
            $<$<COMPILE_LANGUAGE:CUDA>:FMT_UNICODE=0>
        )
    endif()

    # NOTE: Removed -O1 workaround as it conflicts with -Xcompiler=/O2
    # The proper fix is to use consistent optimization flags via -Xcompiler

    # Windows NVCC 12.8 ICE workaround: DISABLE optimization for complex templates
    # These files have complex functors with arrays that trigger ICE at ANY optimization level
    # Must use -O0 (no optimization) to avoid "unexpected expression with aggregate type!" error
    # This is a severe nvcc 12.8 compiler bug with aggregate types in template-heavy code
    if(WIN32)
        set_source_files_properties(
            tensor_broadcast_ops.cu
            tensor_ops.cu
            PROPERTIES
            COMPILE_FLAGS "-O0"
        )
    endif()
endif()

# Main tensor library (C++23)
add_library(lfs_tensor STATIC ${LFS_TENSOR_SOURCES})

target_include_directories(lfs_tensor
        PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/../include  # core/include for core headers
        ${CMAKE_SOURCE_DIR}/src       # For <core/tensor/internal/*.hpp> includes
        ${CMAKE_BINARY_DIR}/include
        ${CUDAToolkit_INCLUDE_DIRS}
        PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}  # For internal/ headers (relative paths)
)

target_link_libraries(lfs_tensor
        PUBLIC
        TBB::tbb
        nlohmann_json::nlohmann_json
        glm::glm
        spdlog::spdlog
        CUDA::cudart
        CUDA::curand
        PRIVATE
        lfs_tensor_kernels
)

# Platform-specific settings
if(UNIX)
    target_link_libraries(lfs_tensor PUBLIC dl)
endif()

# Compiler options
if(MSVC)
    target_compile_options(lfs_tensor PRIVATE
            $<$<CONFIG:Debug>:/Od /Z7>
            $<$<CONFIG:Release>:/O2 /DNDEBUG>
    )
    target_compile_definitions(lfs_tensor PRIVATE _USE_MATH_DEFINES NOMINMAX)
else()
    target_compile_options(lfs_tensor PRIVATE
            $<$<CONFIG:Debug>:-O0 -g -fno-omit-frame-pointer -DDEBUG>
            $<$<CONFIG:Release>:-O3 -DNDEBUG -march=native>
    )
endif()

# AVX2 support
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    include(CheckCXXCompilerFlag)
    check_cxx_compiler_flag("-mavx2" COMPILER_SUPPORTS_AVX2)

    if(COMPILER_SUPPORTS_AVX2)
        target_compile_options(lfs_tensor PRIVATE
                $<$<COMPILE_LANGUAGE:CXX>:-mavx2 -mfma>
        )
        target_compile_definitions(lfs_tensor PRIVATE HAS_AVX2_SUPPORT)
        message(STATUS "✓ AVX2 support enabled for lfs_tensor")
    else()
        message(WARNING "✗ Compiler does not support AVX2")
    endif()
endif()

# OpenMP support for multi-threaded CPU operations
find_package(OpenMP)
if(OpenMP_CXX_FOUND)
    target_link_libraries(lfs_tensor PUBLIC OpenMP::OpenMP_CXX)
    message(STATUS "✓ OpenMP support enabled for lfs_tensor (multi-threaded .contiguous())")
else()
    message(WARNING "✗ OpenMP not found - .contiguous() will use single-threaded code")
endif()

# Set properties
set_target_properties(lfs_tensor PROPERTIES
        CXX_STANDARD 23
        CXX_STANDARD_REQUIRED ON
        POSITION_INDEPENDENT_CODE ON
        EXPORT_COMPILE_COMMANDS ON
)

# Precompiled header
target_precompile_headers(lfs_tensor
        PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/../include/core/tensor.hpp
)

# Configure build type
if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    target_compile_definitions(lfs_tensor PRIVATE DEBUG_BUILD)
elseif(CMAKE_BUILD_TYPE STREQUAL "Release")
    target_compile_definitions(lfs_tensor PRIVATE RELEASE_BUILD)
endif()

# Allocation profiling
if(ENABLE_ALLOCATION_PROFILING)
    target_compile_definitions(lfs_tensor PUBLIC ENABLE_ALLOCATION_PROFILING=1)
    message(STATUS "  • Allocation profiling: ENABLED for lfs_tensor")
else()
    target_compile_definitions(lfs_tensor PUBLIC ENABLE_ALLOCATION_PROFILING=0)
endif()

message(STATUS "╔════════════════════════════════════════════════════════════════╗")
message(STATUS "║  lfs_tensor - LichtFeld Studio Tensor Library                 ║")
message(STATUS "╚════════════════════════════════════════════════════════════════╝")
message(STATUS "  • Tensor kernels: CUDA C++20 (lfs_tensor_kernels)")
message(STATUS "  • Core library: C++23 (lfs_tensor)")
message(STATUS "  • Public API: include/core/tensor.hpp")
message(STATUS "  • Implementation: src/core/tensor/internal/")
message(STATUS "  • Build type: ${CMAKE_BUILD_TYPE}")
message(STATUS "  • Precompiled headers: ENABLED (tensor.hpp)")
message(STATUS "  • Using Thrust (bundled with CUDA) - no libtorch dependency")

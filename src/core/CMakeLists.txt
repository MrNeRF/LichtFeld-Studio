# SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
# SPDX-License-Identifier: GPL-3.0-or-later

# ============================================================================
# core module - LibTorch-free core + tensor library (SHARED)
# ============================================================================

# Tensor CUDA kernels (STATIC, linked privately)
add_subdirectory(tensor)

# Core CUDA library (STATIC, linked privately)
add_subdirectory(cuda)

# Event bridge shared library
add_subdirectory(event_bridge)

# Logger shared library (leaf dep, breaks lfs_core ↔ lfs_event_bridge cycle)
add_subdirectory(logger)

# Tensor C++ sources (merged from src/core/tensor/)
set(TENSOR_SOURCES
        tensor/cuda_event_pool.cpp
        tensor/cuda_stream_context.cpp
        tensor/gpu_slab_allocator.cpp
        tensor/size_bucketed_pool.cpp
        tensor/lazy_config.cpp
        tensor/lazy_ir.cpp
        tensor/lazy_executor.cpp
        tensor/tensor.cpp
        tensor/tensor_serialization.cpp
        tensor/tensor_factory.cpp
        tensor/tensor_utils.cpp
        tensor/tensor_matrix_ops.cpp
        tensor/tensor_unified_ops.cpp
        tensor/tensor_movement_ops.cpp
        tensor/tensor_random_ops.cpp
        tensor/tensor_broadcast.cpp
        tensor/tensor_shape_ops.cpp
        tensor/tensor_masking_ops.cpp
        tensor/tensor_advanced_ops.cpp
        tensor/tensor_row_proxy.cpp
        tensor/tensor_nn_ops.cpp
        tensor/pinned_memory_allocator.cpp
        tensor/offset_allocator.cpp
        tensor/tensor_exports.cu
)

# Core module sources
set(CORE_SOURCES
        abi.cpp
        crash_handler.cpp
        cuda_error.cpp
        cuda_error_typed.cpp
        device_fault.cpp
        error.cpp
        error_bus.cpp
        error_envelope.cpp
        error_latch.cpp
        error_reporter.cpp
        guarded_task.cpp
        failure_report.cpp
        cuda/memory_arena_owner.cpp
        memory_pressure.cpp
        operator_callbacks.cpp
        animatable_property.cpp
        argument_parser.cpp
        camera.cpp
        checkpoint_format.cpp
        image_loader.cpp
        cuda_version.cpp
        image_io.cpp
        mapped_file.cpp
        parameters.cpp
        property_registry.cpp
        reactive_store.cpp
        scene.cpp
        splat_simplify.cpp
        bhatt_lod.cpp
        octree_lod.cpp
        splat_data.cpp
        splat_data_mirror.cpp
        splat_data_transform.cpp
        splat_exportable_storage.cpp
        tensor_debug.cpp
        training_snapshot.cpp
        mesh_data.cpp
)

add_library(lfs_core SHARED ${CORE_SOURCES} ${TENSOR_SOURCES})

target_compile_definitions(lfs_core PRIVATE LFS_CORE_EXPORTS)

target_include_directories(lfs_core
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/include
        ${CMAKE_BINARY_DIR}/include
        ${CUDAToolkit_INCLUDE_DIRS}
    PRIVATE
        ${CMAKE_SOURCE_DIR}/src
        ${CMAKE_CURRENT_SOURCE_DIR}/tensor  # For internal/ headers
)
target_include_directories(lfs_core SYSTEM PRIVATE
    ${CMAKE_SOURCE_DIR}/external
)

target_link_libraries(lfs_core
    PUBLIC
        lfs_logger
        lfs_diagnostics
        lfs_event_bridge
        lfs_geometry
        TBB::tbb
        nlohmann_json::nlohmann_json
        glm::glm
        Threads::Threads
        CUDA::cudart
        CUDA::curand
        spdlog::spdlog
    PRIVATE
        lfs_core_cuda
        lfs_tensor_kernels
        taywee::args
        lfs_oiio
        OpenMeshCore
)

set_property(SOURCE image_io.cpp APPEND PROPERTY COMPILE_DEFINITIONS FMT_HEADER_ONLY=1)

# OpenMP for tensor multi-threaded operations
find_package(OpenMP)
if(OpenMP_CXX_FOUND)
    target_link_libraries(lfs_core PRIVATE OpenMP::OpenMP_CXX)
    message(STATUS "[ok] OpenMP support enabled for lfs_core")
endif()

# Platform-specific settings
if(UNIX)
    target_link_libraries(lfs_core PUBLIC dl)
endif()

if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU" AND CMAKE_CXX_COMPILER_VERSION VERSION_GREATER_EQUAL 12)
    target_link_libraries(lfs_core PRIVATE stdc++exp)
endif()

# Compiler options
if(MSVC)
    target_compile_options(lfs_core PRIVATE
        $<$<AND:$<COMPILE_LANGUAGE:CXX>,$<CONFIG:Debug>>:/Od /Z7>
        $<$<AND:$<COMPILE_LANGUAGE:CXX>,$<CONFIG:Release>>:/O2 /DNDEBUG>
        $<$<AND:$<COMPILE_LANGUAGE:CUDA>,$<CONFIG:Debug>>:-O0 -g -lineinfo --extended-lambda --expt-relaxed-constexpr -Xcompiler=/Od -Xcompiler=/Z7 -Xcompiler=/utf-8 -Xcompiler=/D_CRT_SECURE_NO_WARNINGS -Xcompiler=/bigobj --diag-suppress=27>
        $<$<AND:$<COMPILE_LANGUAGE:CUDA>,$<CONFIG:Release>>:-O3 -use_fast_math --extended-lambda --expt-relaxed-constexpr -Xcompiler=/O2 -Xcompiler=/DNDEBUG -Xcompiler=/utf-8 -Xcompiler=/D_CRT_SECURE_NO_WARNINGS -Xcompiler=/bigobj --diag-suppress=27>
    )
    target_compile_definitions(lfs_core PRIVATE _USE_MATH_DEFINES NOMINMAX
        $<$<COMPILE_LANGUAGE:CUDA>:FMT_UNICODE=0>
    )
    target_link_options(lfs_core PRIVATE /ignore:4217)
else()
    target_compile_options(lfs_core PRIVATE
        $<$<AND:$<COMPILE_LANGUAGE:CXX>,$<CONFIG:Debug>>:-O0 -g -fno-omit-frame-pointer -DDEBUG>
        $<$<AND:$<COMPILE_LANGUAGE:CXX>,$<CONFIG:Release>>:-O3 -DNDEBUG -march=native>
        $<$<AND:$<COMPILE_LANGUAGE:CUDA>,$<CONFIG:Debug>>:-O0 -g -lineinfo --extended-lambda --expt-relaxed-constexpr>
        $<$<AND:$<COMPILE_LANGUAGE:CUDA>,$<CONFIG:Release>>:-O3 -use_fast_math --extended-lambda --expt-relaxed-constexpr>
    )
endif()

# AVX2 support
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    include(CheckCXXCompilerFlag)
    check_cxx_compiler_flag("-mavx2" COMPILER_SUPPORTS_AVX2)

    if(COMPILER_SUPPORTS_AVX2)
        target_compile_options(lfs_core PRIVATE
            $<$<COMPILE_LANGUAGE:CXX>:-mavx2 -mfma>
        )
        target_compile_definitions(lfs_core PRIVATE HAS_AVX2_SUPPORT)
    endif()

    # The splat simplifier is sensitive to tiny FP contraction differences.
    set_source_files_properties(
        ${CMAKE_CURRENT_SOURCE_DIR}/splat_simplify.cpp
        PROPERTIES COMPILE_OPTIONS "-ffp-contract=off;-fno-tree-vectorize;-fno-unsafe-math-optimizations;-frounding-math;-ffloat-store;-fexcess-precision=standard$<$<BOOL:${COMPILER_SUPPORTS_AVX2}>:;-mno-fma;-mno-avx2>")
endif()

# Visibility: hide by default, export only LFS_CORE_API symbols
set_target_properties(lfs_core PROPERTIES
    CXX_STANDARD 23
    CXX_STANDARD_REQUIRED ON
    CUDA_STANDARD 20
    CUDA_STANDARD_REQUIRED ON
    CUDA_ARCHITECTURES "${LichtFeld-Studio_CUDA_ARCH}"
    CUDA_SEPARABLE_COMPILATION OFF
    POSITION_INDEPENDENT_CODE ON
    EXPORT_COMPILE_COMMANDS ON
    CXX_VISIBILITY_PRESET hidden
    VISIBILITY_INLINES_HIDDEN ON
    LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}
    DEBUG_POSTFIX d
)
get_property(_IS_MULTI_CONFIG GLOBAL PROPERTY GENERATOR_IS_MULTI_CONFIG)
if(WIN32 AND _IS_MULTI_CONFIG)
    set_target_properties(lfs_core PROPERTIES
        LIBRARY_OUTPUT_DIRECTORY              ${CMAKE_BINARY_DIR}/$<CONFIG>
        RUNTIME_OUTPUT_DIRECTORY              ${CMAKE_BINARY_DIR}/$<CONFIG>
        RUNTIME_OUTPUT_DIRECTORY_DEBUG        ${CMAKE_BINARY_DIR}/Debug
        RUNTIME_OUTPUT_DIRECTORY_RELEASE      ${CMAKE_BINARY_DIR}/Release
        RUNTIME_OUTPUT_DIRECTORY_RELWITHDEBINFO ${CMAKE_BINARY_DIR}/RelWithDebInfo
        RUNTIME_OUTPUT_DIRECTORY_MINSIZEREL   ${CMAKE_BINARY_DIR}/MinSizeRel
    )
endif()

# Allocation profiling
target_compile_definitions(lfs_core PUBLIC
    LFS_ALLOCATION_PROFILING_ENABLED=$<BOOL:${LFS_ENABLE_ALLOCATION_PROFILING}>)

message(STATUS "+------------------------------------------------------------------+")
message(STATUS "|  lfs_core - Core + Tensor Library (SHARED)                       |")
message(STATUS "+------------------------------------------------------------------+")
message(STATUS "  - Namespace: lfs::core")
message(STATUS "  - Includes: tensor, core utilities, parameters, splat data")

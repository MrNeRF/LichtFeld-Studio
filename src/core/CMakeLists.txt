# SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
# SPDX-License-Identifier: GPL-3.0-or-later

# ============================================================================
# core module - LibTorch-free core + tensor library (SHARED)
# ============================================================================

# Tensor CUDA kernels (STATIC, linked privately)
if(LFS_HAS_CUDA)
    add_subdirectory(tensor)
endif()

option(LFS_TENSOR_VULKAN "Build the Vulkan tensor backend" ON)
if(LFS_TENSOR_VULKAN)
    include(tensor/backend/vulkan/CMakeLists.txt)
endif()

# Neural inference kernels (STATIC, linked privately)
if(LFS_HAS_CUDA)
    add_subdirectory(nn)
endif()

# Core CUDA library (STATIC, linked privately)
if(LFS_HAS_CUDA)
    add_subdirectory(cuda)
endif()

# Event bridge shared library
add_subdirectory(event_bridge)

# Logger shared library (leaf dep, breaks lfs_core ↔ lfs_event_bridge cycle)
add_subdirectory(logger)

# Tensor C++ sources (merged from src/core/tensor/)
set(TENSOR_SOURCES
        tensor/backend/cuda/runtime/cuda_event_pool.cpp
        tensor/backend/cuda/runtime/cuda_stream_context.cpp
        tensor/backend/cuda/runtime/stream_lifetime.cpp
        tensor/backend/cuda/runtime/gpu_slab_allocator.cpp
        tensor/backend/cuda/runtime/size_bucketed_pool.cpp
        tensor/lazy_config.cpp
        tensor/lazy_ir.cpp
        tensor/lazy_executor.cpp
        tensor/expression_program.cpp
        tensor/expression_codegen.cpp
        tensor/expression_cache.cpp
        tensor/tensor_fused.cpp
        tensor/tensor_backend.cpp
        tensor/tensor_gpu_elapsed.cpp
        tensor/tensor_backend_selftest.cpp
        tensor/backend/backend_registry.cpp
        tensor/backend/portable_nn_ops.cpp
        tensor/backend/facade_trace.cpp
        tensor/backend/cuda/cuda_ops_pointwise.cpp
        tensor/backend/cuda/cuda_ops_expression.cpp
        tensor/backend/cuda/cuda_ops_reduce.cpp
        tensor/backend/cuda/cuda_ops_scan_sort.cpp
        tensor/backend/cuda/cuda_ops_matrix_nn.cpp
        tensor/backend/cuda/cuda_ops_random.cpp
        tensor/backend/cuda/cuda_ops_spatial.cpp
        tensor/backend/cuda/cuda_ops_histogram.cpp
        tensor/backend/cuda/cuda_ops_ppisp.cpp
        tensor/backend/cuda/cuda_ops_environment.cpp
        tensor/backend/cuda/cuda_ops_image.cpp
        tensor/backend/cuda/cuda_ops_splat.cpp
        tensor/backend/cuda/cuda_ops_sh.cpp
        tensor/backend/cuda/cuda_ops_labels.cpp
        tensor/backend/cuda/cuda_ops_filters.cpp
        tensor/backend/cuda/cuda_ops_export.cpp
        tensor/backend/cuda/cuda_ops_index_mask.cpp
        tensor/backend/cuda/cuda_ops_movement.cpp
        tensor/backend/cuda/cuda_services.cpp
        tensor/backend/vulkan/vma_implementation.cpp
        tensor/tensor.cpp
        tensor/tensor_method_bodies.cpp
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
        tensor/tensor_spatial.cpp
        tensor/tensor_projection.cpp
        tensor/tensor_point_region.cpp
        tensor/tensor_histogram.cpp
        tensor/tensor_ppisp.cpp
        tensor/tensor_environment.cpp
        tensor/tensor_image.cpp
        tensor/tensor_splat.cpp
        tensor/tensor_sh.cpp
        tensor/tensor_labels.cpp
        tensor/tensor_filters.cpp
        tensor/tensor_export.cpp
        tensor/tensor_readback.cpp
        tensor/backend/tensor_upload.cpp
        tensor/tensor_rad.cpp
        tensor/backend/cuda/cuda_ops_rad.cpp
        tensor/backend/cuda/kernels/rad_page.cu
        tensor/tensor_row_proxy.cpp
        tensor/tensor_nn_ops.cpp
        tensor/backend/cuda/runtime/pinned_memory_allocator.cpp
        tensor/backend/cuda/kernels/tensor_exports.cu
        tensor/backend/cuda/kernels/where_scalar.cu
        nn/ops.cpp
        nn/weight_file.cpp
        nn/weight_file_open.cpp
        nn/activation_arena.cpp
        nn/models/moge2.cpp
        nn/models/sam2.cpp
        nn/models/lpips.cpp
        nn/models/romav1.cpp
)

if(NOT LFS_HAS_CUDA)
    list(FILTER TENSOR_SOURCES EXCLUDE REGEX "tensor/backend/cuda/")
    list(APPEND TENSOR_SOURCES tensor/backend/cuda/runtime/pinned_memory_allocator.cpp)
    list(REMOVE_ITEM TENSOR_SOURCES nn/models/romav1.cpp)
endif()

if(LFS_TENSOR_VULKAN)
    list(APPEND TENSOR_SOURCES
        tensor/backend/vulkan/vk_context.cpp
        tensor/backend/vulkan/vk_memory.cpp
        tensor/backend/vulkan/vk_recorder.cpp
        tensor/backend/vulkan/vk_pipelines.cpp
        tensor/backend/vulkan/vk_ops_pointwise.cpp
        tensor/backend/vulkan/vk_ops_expression.cpp
        tensor/backend/vulkan/vk_ops_movement.cpp
        tensor/backend/vulkan/vk_ops_reduce.cpp
        tensor/backend/vulkan/vk_ops_index.cpp
        tensor/backend/vulkan/vk_ops_mask.cpp
        tensor/backend/vulkan/vk_ops_sort.cpp
        tensor/backend/vulkan/vk_ops_matrix.cpp
        tensor/backend/vulkan/vk_ops_nn.cpp
        tensor/backend/vulkan/vk_ops_random.cpp
        tensor/backend/vulkan/vk_ops_spatial.cpp
        tensor/backend/vulkan/vk_ops_histogram.cpp
        tensor/backend/vulkan/vk_ops_ppisp.cpp
        tensor/backend/vulkan/vk_ops_environment.cpp
        tensor/backend/vulkan/vk_ops_image.cpp
        tensor/backend/vulkan/vk_ops_splat.cpp
        tensor/backend/vulkan/vk_ops_sh.cpp
        tensor/backend/vulkan/vk_ops_rad.cpp
        tensor/backend/vulkan/vk_ops_labels.cpp
        tensor/backend/vulkan/vk_ops_filters.cpp
        tensor/backend/vulkan/vk_ops_export.cpp
        tensor/backend/vulkan/vk_services.cpp
    )
    if(LFS_HAS_CUDA)
        list(APPEND TENSOR_SOURCES tensor/backend/vulkan/vk_cuda_bridge.cpp)
    endif()
endif()

# The GPU export driver, shared by the Vulkan and Metal backends. Its host edge
# refinement uses the same independent double rounding as decimation.
if(LFS_TENSOR_VULKAN OR APPLE)
    list(APPEND TENSOR_SOURCES tensor/backend/export_pipeline.cpp)
    set_source_files_properties(tensor/backend/export_pipeline.cpp PROPERTIES
        COMPILE_OPTIONS "$<$<CXX_COMPILER_ID:GNU,Clang,AppleClang>:-ffp-contract=off>")
endif()

# Metal tensor backend: the kernel sources are embedded and compiled at
# runtime, so the build needs no Metal toolchain.
if(APPLE)
    set(_lfs_metal_kernels_cpp ${CMAKE_CURRENT_BINARY_DIR}/lfs_metal_kernels.cpp)
    set(_lfs_metal_sources tensor/backend/metal/kernels.metal tensor/backend/metal/export.metal
                           tensor/backend/metal/nn.metal)
    list(TRANSFORM _lfs_metal_sources PREPEND ${CMAKE_CURRENT_SOURCE_DIR}/ OUTPUT_VARIABLE _lfs_metal_inputs)
    string(REPLACE ";" "$<SEMICOLON>" _lfs_metal_inputs "${_lfs_metal_inputs}")
    add_custom_command(
        OUTPUT ${_lfs_metal_kernels_cpp}
        COMMAND ${CMAKE_COMMAND}
            "-DINPUTS=${_lfs_metal_inputs}"
            -DOUTPUT=${_lfs_metal_kernels_cpp}
            -P ${CMAKE_CURRENT_SOURCE_DIR}/tensor/backend/metal/embed_kernels.cmake
        DEPENDS ${_lfs_metal_sources} tensor/backend/metal/embed_kernels.cmake
        COMMENT "Embedding Metal tensor kernels"
        VERBATIM)
    list(APPEND TENSOR_SOURCES
        tensor/backend/metal/expression_msl.cpp
        tensor/backend/metal/metal_context.mm
        tensor/backend/metal/metal_ops.mm
        ${_lfs_metal_kernels_cpp}
    )
    if(LFS_TENSOR_VULKAN)
        list(APPEND TENSOR_SOURCES tensor/backend/metal/metal_vulkan_interop.mm)
    endif()
endif()

# Core module sources
set(CORE_SOURCES
        abi.cpp
        alloc_counter.cpp
        crash_handler.cpp
        training_churn_metrics.cpp
        cuda_diagnostic_modes.cpp
        cuda_error.cpp
        cuda_error_typed.cpp
        device_fault.cpp
        validated_index_token.cpp
        error.cpp
        error_bus.cpp
        exif.cpp
        error_envelope.cpp
        error_latch.cpp
        error_reporter.cpp
        guarded_task.cpp
        host_metrics.cpp
        failure_report.cpp
        cuda/memory_arena_owner.cpp
        cuda/vmm_device_buffer.cpp
        memory_pressure.cpp
        memory_domain.cpp
        operator_callbacks.cpp
        animatable_property.cpp
        camera.cpp
        checkpoint_format.cpp
        image_loader.cpp
        cuda_version.cpp
        image_io.cpp
        image_codecs.cpp
        mapped_file.cpp
        optimization_properties.cpp
        parameters.cpp
        property_registry.cpp
        provenance.cpp
        reactive_store.cpp
        scene.cpp
        selection_ops.cpp
        session_breadcrumb.cpp
        uuid.cpp
        splat_simplify.cpp
        bhatt_lod.cpp
        octree_lod.cpp
        splat_data.cpp
        splat_data_mirror.cpp
        splat_data_transform.cpp
        splat_exportable_storage.cpp
        sh_value_quant.cpp
        shareable_allocation_limit.cpp
        tensor_debug.cpp
        training_snapshot.cpp
        mesh_data.cpp
        system_info.cpp
        user_paths.cpp
        legacy_settings_migration.cpp
)

if(NOT LFS_HAS_CUDA)
    list(REMOVE_ITEM CORE_SOURCES
        cuda_error.cpp
        cuda_error_typed.cpp
        device_fault.cpp
        cuda/memory_arena_owner.cpp
        cuda/vmm_device_buffer.cpp
        splat_exportable_storage.cpp)
    list(APPEND CORE_SOURCES
        cuda_error_no_cuda.cpp
        splat_exportable_storage_no_cuda.cpp)
    list(APPEND TENSOR_SOURCES
        tensor/tensor_cuda_interop_no_cuda.cpp)
endif()

add_library(lfs_core SHARED ${CORE_SOURCES} ${TENSOR_SOURCES})

if(LFS_TENSOR_VULKAN)
    add_dependencies(lfs_core lfs_tensor_vulkan_shaders)
    target_sources(lfs_core PRIVATE ${LFS_TENSOR_SHADER_TABLE})
    target_compile_definitions(lfs_core PRIVATE LFS_TENSOR_VULKAN=1)
endif()
if(APPLE)
    target_compile_definitions(lfs_core PRIVATE LFS_TENSOR_METAL=1)
    target_compile_options(lfs_core PRIVATE $<$<COMPILE_LANGUAGE:OBJCXX>:-fobjc-arc>)
    target_link_libraries(lfs_core PRIVATE "-framework Metal" "-framework Foundation")
endif()

add_dependencies(lfs_core lfs_core_abi_stamp)
target_include_directories(lfs_core BEFORE PRIVATE "${LFS_CORE_ABI_INCLUDE_DIR}")

target_compile_definitions(lfs_core PRIVATE LFS_CORE_EXPORTS LFS_TENSOR_PRIVATE_ACCESS)

target_include_directories(lfs_core
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/include
        ${CMAKE_BINARY_DIR}/include
        ${CUDAToolkit_INCLUDE_DIRS}
    PRIVATE
        ${CMAKE_SOURCE_DIR}/src
        ${CMAKE_CURRENT_SOURCE_DIR}/tensor  # For internal/ headers
        ${CMAKE_CURRENT_SOURCE_DIR}/nn
)
target_include_directories(lfs_core SYSTEM PRIVATE
    ${CMAKE_SOURCE_DIR}/external
)

find_package(Python3 COMPONENTS Interpreter REQUIRED)
set(_expression_catalog "${CMAKE_CURRENT_BINARY_DIR}/expression_catalog.cpp")
if(LFS_HAS_CUDA)
add_library(lfs_expression_catalog OBJECT tensor/backend/cuda/expression_catalog.cu)
set_target_properties(lfs_expression_catalog PROPERTIES
    CUDA_PTX_COMPILATION ON
    CUDA_ARCHITECTURES "${LichtFeld-Studio_CUDA_ARCH}"
    CUDA_STANDARD 20 CUDA_STANDARD_REQUIRED ON)
target_include_directories(lfs_expression_catalog PRIVATE
    "$<TARGET_PROPERTY:lfs_tensor_kernels,INCLUDE_DIRECTORIES>")
target_compile_options(lfs_expression_catalog PRIVATE --keep-device-functions
    "$<TARGET_PROPERTY:lfs_tensor_kernels,COMPILE_OPTIONS>")
target_compile_definitions(lfs_expression_catalog PRIVATE
    "$<TARGET_PROPERTY:lfs_tensor_kernels,COMPILE_DEFINITIONS>")
add_custom_command(OUTPUT "${_expression_catalog}"
    COMMAND "${Python3_EXECUTABLE}" "${CMAKE_SOURCE_DIR}/tools/embed_expression_ptx.py"
        --ptx $<TARGET_OBJECTS:lfs_expression_catalog>
        --tensor "${CMAKE_CURRENT_SOURCE_DIR}/tensor"
        --output "${_expression_catalog}"
        --depfile "${_expression_catalog}.d"
    DEPFILE "${_expression_catalog}.d"
    DEPENDS lfs_expression_catalog $<TARGET_OBJECTS:lfs_expression_catalog>
            "${CMAKE_SOURCE_DIR}/tools/embed_expression_ptx.py"
    COMMAND_EXPAND_LISTS VERBATIM)
target_sources(lfs_core PRIVATE "${_expression_catalog}" tensor/backend/cuda/expression_ptx.cpp)
else()
    add_custom_command(OUTPUT "${_expression_catalog}"
        COMMAND "${Python3_EXECUTABLE}" "${CMAKE_SOURCE_DIR}/tools/embed_expression_ptx.py"
            --tensor "${CMAKE_CURRENT_SOURCE_DIR}/tensor"
            --output "${_expression_catalog}"
            --depfile "${_expression_catalog}.d"
        DEPFILE "${_expression_catalog}.d"
        DEPENDS "${CMAKE_SOURCE_DIR}/tools/embed_expression_ptx.py"
        VERBATIM)
    target_sources(lfs_core PRIVATE "${_expression_catalog}")
endif()
if(LFS_TENSOR_VULKAN)
    target_sources(lfs_core PRIVATE tensor/backend/vulkan/expression_spirv.cpp)
    target_link_libraries(lfs_core PRIVATE lfs_tensor_spirv)
endif()

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
        $<$<BOOL:${LFS_HAS_CUDA}>:CUDA::cudart>
        $<$<BOOL:${LFS_HAS_CUDA}>:CUDA::curand>
        spdlog::spdlog
        Vulkan::Vulkan
        GPUOpen::VulkanMemoryAllocator
    PRIVATE
        $<$<BOOL:${LFS_HAS_CUDA}>:lfs_core_cuda>
        $<$<BOOL:${LFS_HAS_CUDA}>:lfs_tensor_kernels>
        $<$<BOOL:${LFS_HAS_CUDA}>:lfs_nn_kernels>
        JPEG::JPEG
        PNG::PNG
        TIFF::TIFF
        WebP::webp
        unofficial::tinyexr::tinyexr
        OpenMeshCore
)
# OpenMP for tensor multi-threaded operations
find_package(OpenMP)
if(OpenMP_CXX_FOUND)
    target_link_libraries(lfs_core PRIVATE OpenMP::OpenMP_CXX)
    message(STATUS "[ok] OpenMP support enabled for lfs_core")
endif()

# Platform-specific settings
if(UNIX)
    target_link_libraries(lfs_core PUBLIC ${CMAKE_DL_LIBS})
endif()

if(WIN32)
    target_link_libraries(lfs_core PRIVATE bcrypt)
    # Public tensor headers use min/max names, so consumers must suppress the
    # Windows macros before any transitive windows.h include is processed.
    target_compile_definitions(lfs_core PUBLIC NOMINMAX)
endif()

if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU" AND CMAKE_CXX_COMPILER_VERSION VERSION_GREATER_EQUAL 12)
    target_link_libraries(lfs_core PRIVATE stdc++exp)
endif()

# Compiler options
if(MSVC)
    target_compile_options(lfs_core PRIVATE
        $<$<AND:$<COMPILE_LANGUAGE:CXX>,$<CONFIG:Debug>>:/Od>
        $<$<AND:$<COMPILE_LANGUAGE:CXX>,$<CONFIG:Release>>:/O2 /DNDEBUG>
        $<$<AND:$<COMPILE_LANGUAGE:CUDA>,$<CONFIG:Debug>>:-O0 -g --extended-lambda --expt-relaxed-constexpr -Xcompiler=/utf-8 -Xcompiler=/D_CRT_SECURE_NO_WARNINGS -Xcompiler=/bigobj --diag-suppress=27>
        $<$<AND:$<COMPILE_LANGUAGE:CUDA>,$<CONFIG:Release>>:-O3 -use_fast_math --extended-lambda --expt-relaxed-constexpr -Xcompiler=/DNDEBUG -Xcompiler=/utf-8 -Xcompiler=/D_CRT_SECURE_NO_WARNINGS -Xcompiler=/bigobj --diag-suppress=27>
    )
    target_compile_definitions(lfs_core PRIVATE _USE_MATH_DEFINES
        $<$<COMPILE_LANGUAGE:CUDA>:FMT_UNICODE=0>
    )
    target_link_options(lfs_core PRIVATE /ignore:4217)
else()
    target_compile_options(lfs_core PRIVATE
        $<$<AND:$<COMPILE_LANGUAGE:CXX>,$<CONFIG:Debug>>:-O0 -g -fno-omit-frame-pointer -DDEBUG>
        $<$<AND:$<COMPILE_LANGUAGE:CXX>,$<CONFIG:Release>>:-O3 -DNDEBUG -march=native>
        $<$<AND:$<COMPILE_LANGUAGE:CUDA>,$<CONFIG:Debug>>:-O0 -g --extended-lambda --expt-relaxed-constexpr>
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

set_source_files_properties(tensor/backend/cuda/kernels/rad_page.cu PROPERTIES
    COMPILE_OPTIONS "$<$<COMPILE_LANGUAGE:CUDA>:--fmad=false>")

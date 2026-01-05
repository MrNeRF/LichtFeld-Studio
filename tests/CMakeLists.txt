# SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
# SPDX-License-Identifier: GPL-3.0-or-later

# Testing module - comprehensive tests for all components

# Find GTest
find_package(GTest CONFIG REQUIRED)

# =============================================================================
# Unicode Path Test Only (no CUDA dependencies)
# =============================================================================
if(BUILD_UNICODE_TEST_ONLY)
    # Standalone Unicode path test without CUDA
    add_executable(unicode_path_test
        test_unicode_paths_windows.cpp
    )

    target_include_directories(unicode_path_test PRIVATE
        ${CMAKE_SOURCE_DIR}/include
        ${CMAKE_BINARY_DIR}/include
        ${CMAKE_SOURCE_DIR}/src
    )

    target_link_libraries(unicode_path_test PRIVATE
        GTest::gtest
        GTest::gtest_main
    )

    # C++23 standard
    set_target_properties(unicode_path_test PROPERTIES
        CXX_STANDARD 23
        CXX_STANDARD_REQUIRED ON
    )

    # Enable Windows long path support (paths > MAX_PATH/260 chars)
    if(WIN32 AND MSVC)
        # Add manifest file as a source to enable long path awareness
        target_sources(unicode_path_test PRIVATE
            ${CMAKE_CURRENT_SOURCE_DIR}/unicode_path_test.manifest
        )
    endif()

    # Register with CTest
    include(GoogleTest)
    gtest_discover_tests(unicode_path_test
        WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
        PROPERTIES TIMEOUT 30
    )

    message(STATUS "Building standalone Unicode path test (no CUDA)")
    return()  # Skip the rest of the file
endif()

# Find LibTorch for tensor comparison tests
# Set Torch_DIR based on platform
get_filename_component(PROJ_ROOT_DIR "${CMAKE_SOURCE_DIR}" ABSOLUTE)
if(WIN32)
    if(CMAKE_BUILD_TYPE STREQUAL "Release" OR NOT CMAKE_BUILD_TYPE)
        set(Torch_DIR "${PROJ_ROOT_DIR}/external/release/libtorch/share/cmake/Torch")
    else()
        set(Torch_DIR "${PROJ_ROOT_DIR}/external/debug/libtorch/share/cmake/Torch")
    endif()
else()
    set(Torch_DIR "${PROJ_ROOT_DIR}/external/libtorch/share/cmake/Torch")
endif()
find_package(Torch REQUIRED)
message(STATUS "Tests: LibTorch ${Torch_VERSION} found for tensor comparison tests")

# =============================================================================
# Separate test sources into regular tests and benchmarks
# =============================================================================

# Benchmark sources (performance measurement focused)
set(BENCHMARK_FILES
    benchmark_background_blend.cpp
    benchmark_adc.cpp
    benchmark_grad_alpha_layout.cpp
    benchmark_pipelined_loader.cpp
    benchmark_slice_assignment.cpp
    benchmark_strided_tensors.cpp
    benchmark_scheduler.cpp
    benchmark_losses.cpp
    # profile_adam_optimizer.cpp  # Disabled - missing header
    test_tensor_benchmark.cpp
    test_memory_pool_benchmark.cpp
    test_broadcast_benchmark.cpp
    test_reduction_benchmark.cpp
    test_fusion_benchmark.cpp
    test_gather_benchmark.cpp
    test_zip_gather_benchmark.cpp
    test_transform_iterator_benchmark.cpp
    test_cub_reduction_benchmark.cpp
    test_scalar_reduction_benchmark.cpp
    test_permute_upload_benchmark.cpp
    test_isolation.cpp
    test_allocator_benchmark.cpp
)

# Regular test sources (unit tests, integration tests)
set(TEST_FILES
    test_loss_gradients.cpp
    test_regularization_comparison.cpp
    test_gut_rasterizer_gradients.cpp
    test_gut_manual_vs_autograd.cpp
    test_gut_autograd_vs_manual.cpp
    test_gut_gradient_comparison.cpp
    test_activation_gradients.cpp
    test_tensor_basic.cpp
    test_tensor_ops.cpp
    test_tensor_memory.cpp
    test_tensor_reduction.cpp
    test_isolation.cpp
    test_tensor_views.cpp
    test_tensor_math.cpp
    test_tensor_utils.cpp
    test_tensor_torch_compat.cpp
    test_tensor_advanced.cpp
    test_tensor_stress.cpp
    test_tensor_matrix.cpp
    test_slice_mm_bug.cpp
    test_tensor_random.cpp
    test_tensor_broadcast.cpp
    test_tensor_masking.cpp
    test_tensor_masked_select_rows.cpp
    test_tensor_bitwise.cpp
    test_tensor_accessor.cpp
    test_tensor_random_advanced.cpp
    test_tensor_indexing_advanced.cpp
    test_tensor_clamping.cpp
    test_tensor_compat.cpp
    test_tensor_cumsum_convenience.cpp
    test_tensor_conversions_shape.cpp
    test_tensor_reserve_inplace_cat.cpp
    test_kmeans.cpp
    test_critical_fixes.cpp
    test_cuda_device_state.cpp
    test_cuda_streams.cpp
    test_tensor_ops_comparison.cpp
    test_sort_minimal.cpp
    test_tensor_debug.cpp
    test_bugs_found.cpp
    test_tensor_vs_torch.cpp
    test_tensor_move.cpp
    test_mcmc.cpp
    test_mcmc_add_new_gs_comparison.cpp
    test_mcmc_relocate_optimizer_state_bug.cpp
    test_expression_templates.cpp
    test_lfs_adam_optimizer.cpp
    test_lfs_scheduler.cpp
    test_lfs_losses.cpp
    test_mcmc_kernels.cpp
    test_tensor_bool.cpp
    test_dataloader_lfs.cpp
    test_bilateral_grid.cpp
    test_sparsity_debug.cpp
    test_adc.cpp
    test_dtype_operations_comprehensive.cpp
    test_tensor_bugs.cpp
    test_densification_performance.cpp
    test_densification_profile.cpp
    test_bug_regressions.cpp
    test_torch_comparisons.cpp
    test_mcmc_logit_verification.cpp
    test_tensor_reserve_capacity.cpp
    test_tensor_cat_inplace.cpp
    test_tensor_inplace_cat_debug.cpp
    test_tensor_slice_assignment_reserve.cpp
    test_tensor_mcmc_memory_profile.cpp
    test_mcmc_tensor_ops.cpp
    test_mcmc_memory_ops.cpp
    test_mcmc_critical_bugs.cpp
    test_mcmc_dead_mask.cpp
    test_relocate_gs_edge_cases.cpp
    test_bool_reduction.cpp
    test_tensor_index_select_bool.cpp
    test_tensor_index_select_uint8.cpp
    test_tensor_uint8_inversion.cpp
    test_tensor_serialization.cpp
    test_tensor_cat_reduction_bug.cpp
    test_tensor_inplace_capacity.cpp
    test_sog_format.cpp
    test_sog_html_export.cpp
    test_gsplat_rasterizer.cpp
    test_spz_format.cpp
    test_unicode_paths_windows.cpp
    test_interleaved_slice_copy.cpp
    test_adc_tensor_ops.cpp
    test_tensor_zero_dimension.cpp
    test_mask_loss.cpp
    test_checkpoint_resume.cpp
    test_nan_inf_gpu_check.cpp
    test_mcmc_nan_fix.cpp
    test_vram_resize.cpp
    test_stride_bugs.cpp
    test_bool_any_all.cpp
    test_fused_l1_ssim.cpp
    test_add_new_gs_tensor_ops.cpp
    test_cpu_dtype_conversions.cpp
    test_cpu_large_tensor_bugs.cpp
    test_curand_buffer_overflow.cpp
    test_adc_comparison.cpp
    # test_export_comparison.cpp  # Standalone tool with its own main()
    test_gradient_accumulation.cpp
    test_image_resize_parallel.cpp
    test_tensor_cat_bug.cpp
    test_tensor_slicing_safety.cpp
    test_tensor_struct_init.cpp
    test_torch_conversion.cpp
    test_fastgs_kernels.cpp
    test_fastgs_fuzz.cpp
)

# Build test sources list
set(TEST_SOURCES test_main.cpp)
foreach(TEST_FILE ${TEST_FILES})
    if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/${TEST_FILE}")
        list(APPEND TEST_SOURCES ${TEST_FILE})
    endif()
endforeach()

# Build benchmark sources list
set(BENCHMARK_SOURCES test_main.cpp)
foreach(BENCH_FILE ${BENCHMARK_FILES})
    if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/${BENCH_FILE}")
        list(APPEND BENCHMARK_SOURCES ${BENCH_FILE})
    endif()
endforeach()

# Create test executable
add_executable(lichtfeld_tests ${TEST_SOURCES})

# Set CUDA architectures
set_target_properties(lichtfeld_tests PROPERTIES
    CUDA_ARCHITECTURES "${LichtFeld-Studio_CUDA_ARCH}"
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_RESOLVE_DEVICE_SYMBOLS ON
    CXX_STANDARD 23
    CXX_STANDARD_REQUIRED ON
)

# Include directories
target_include_directories(lichtfeld_tests PRIVATE
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_BINARY_DIR}/include
    ${CMAKE_SOURCE_DIR}/tests
    ${CMAKE_SOURCE_DIR}/src/training   # For training headers
    ${CMAKE_SOURCE_DIR}/src            # For module headers
    ${CMAKE_SOURCE_DIR}/external/spz   # SPZ library headers
    ${CUDAToolkit_INCLUDE_DIRS}
    ${OPENGL_INCLUDE_DIRS}
)

# Define TEST_DATA_DIR and PROJECT_ROOT_PATH for tests
target_compile_definitions(lichtfeld_tests PRIVATE
    TEST_DATA_DIR="${CMAKE_SOURCE_DIR}/data"
    PROJECT_ROOT_PATH="${CMAKE_SOURCE_DIR}"
)

# Link libraries
target_link_libraries(lichtfeld_tests PRIVATE
    # LFS modules
    lfs_visualizer
    lfs_rendering
    lfs_training
    lfs_core
    lfs_geometry
    lfs_io
    lfs_tensor

    # Backend libraries
    fastlfs_backend
    spz_lib  # Niantic SPZ format library

    # External libraries
    GTest::gtest
    ${TORCH_LIBRARIES}
    ${OPENGL_LIBRARIES}
    OpenImageIO::OpenImageIO

    # CUDA libraries
    CUDA::cudart
    CUDA::curand
    CUDA::cupti

    # System libraries
    spdlog::spdlog
)

# Compiler options for CUDA debugging
target_compile_options(lichtfeld_tests PRIVATE
    $<$<AND:$<COMPILE_LANGUAGE:CUDA>,$<CONFIG:Debug>>:-G -lineinfo>
    $<$<AND:$<COMPILE_LANGUAGE:CXX>,$<CONFIG:Debug>>:-O0 -g -fno-omit-frame-pointer>
    $<$<AND:$<COMPILE_LANGUAGE:CXX>,$<CONFIG:Release>>:-O3 -DNDEBUG>
)

# =============================================================================
# Benchmark executable
# =============================================================================
add_executable(lichtfeld_benchmarks ${BENCHMARK_SOURCES})

set_target_properties(lichtfeld_benchmarks PROPERTIES
    CUDA_ARCHITECTURES "${LichtFeld-Studio_CUDA_ARCH}"
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_RESOLVE_DEVICE_SYMBOLS ON
    CXX_STANDARD 23
    CXX_STANDARD_REQUIRED ON
)

target_include_directories(lichtfeld_benchmarks PRIVATE
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_BINARY_DIR}/include
    ${CMAKE_SOURCE_DIR}/tests
    ${CMAKE_SOURCE_DIR}/src/training
    ${CMAKE_SOURCE_DIR}/src
    ${CMAKE_SOURCE_DIR}/external/spz
    ${CUDAToolkit_INCLUDE_DIRS}
    ${OPENGL_INCLUDE_DIRS}
)

target_compile_definitions(lichtfeld_benchmarks PRIVATE
    TEST_DATA_DIR="${CMAKE_SOURCE_DIR}/data"
    PROJECT_ROOT_PATH="${CMAKE_SOURCE_DIR}"
)

target_link_libraries(lichtfeld_benchmarks PRIVATE
    lfs_visualizer
    lfs_rendering
    lfs_training
    lfs_core
    lfs_geometry
    lfs_io
    lfs_tensor
    fastlfs_backend
    spz_lib
    GTest::gtest
    ${TORCH_LIBRARIES}
    ${OPENGL_LIBRARIES}
    OpenImageIO::OpenImageIO
    CUDA::cudart
    CUDA::curand
    CUDA::cupti
    spdlog::spdlog
)

target_compile_options(lichtfeld_benchmarks PRIVATE
    $<$<AND:$<COMPILE_LANGUAGE:CUDA>,$<CONFIG:Debug>>:-G -lineinfo>
    $<$<AND:$<COMPILE_LANGUAGE:CXX>,$<CONFIG:Debug>>:-O0 -g -fno-omit-frame-pointer>
    $<$<AND:$<COMPILE_LANGUAGE:CXX>,$<CONFIG:Release>>:-O3 -DNDEBUG>
)

# =============================================================================
# CTest registration
# =============================================================================
include(GoogleTest)

# Register unit tests with CTest
gtest_discover_tests(lichtfeld_tests
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
    PROPERTIES TIMEOUT 30
    DISCOVERY_MODE PRE_TEST
)

# Register benchmarks with CTest (longer timeout for benchmarks)
gtest_discover_tests(lichtfeld_benchmarks
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
    PROPERTIES TIMEOUT 300
    DISCOVERY_MODE PRE_TEST
    TEST_PREFIX "benchmark."
)

# Custom targets for convenience
add_custom_target(run_tests
    COMMAND ${CMAKE_CTEST_COMMAND} --output-on-failure -V -R "^(?!benchmark\\.)"
    DEPENDS lichtfeld_tests
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
    COMMENT "Running unit tests (excluding benchmarks)"
)

add_custom_target(run_benchmarks
    COMMAND ${CMAKE_CTEST_COMMAND} --output-on-failure -V -R "^benchmark\\."
    DEPENDS lichtfeld_benchmarks
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
    COMMENT "Running benchmarks"
)

# Platform-specific settings
if(WIN32)
    # Copy Torch DLLs for testing on Windows
    file(GLOB TORCH_DLLS "${Torch_DIR}/../../../lib/*.dll")
    foreach(TEST_TARGET lichtfeld_tests lichtfeld_benchmarks)
        foreach(TORCH_DLL ${TORCH_DLLS})
            add_custom_command(
                TARGET ${TEST_TARGET}
                POST_BUILD
                COMMAND ${CMAKE_COMMAND} -E copy_if_different "${TORCH_DLL}"
                "$<TARGET_FILE_DIR:${TEST_TARGET}>")
        endforeach()
        if(TARGET nvimgcodec)
            add_dependencies(${TEST_TARGET} nvimgcodec)
            add_custom_command(TARGET ${TEST_TARGET} POST_BUILD
                COMMAND ${CMAKE_COMMAND} -E copy_if_different
                    "$<TARGET_FILE:nvimgcodec>"
                    "$<TARGET_FILE_DIR:${TEST_TARGET}>"
                COMMENT "Copying nvImageCodec library for ${TEST_TARGET}"
            )
        endif()
        if(TARGET nvjpeg_ext)
            add_dependencies(${TEST_TARGET} nvjpeg_ext)
            add_custom_command(TARGET ${TEST_TARGET} POST_BUILD
                COMMAND ${CMAKE_COMMAND} -E make_directory "$<TARGET_FILE_DIR:${TEST_TARGET}>/extensions"
                COMMAND ${CMAKE_COMMAND} -E copy_if_different
                    "$<TARGET_FILE:nvjpeg_ext>"
                    "$<TARGET_FILE_DIR:${TEST_TARGET}>/extensions/"
                COMMENT "Copying nvjpeg extension for ${TEST_TARGET}"
            )
        endif()
        if(TARGET nvjpeg2k_ext)
            add_dependencies(${TEST_TARGET} nvjpeg2k_ext)
            add_custom_command(TARGET ${TEST_TARGET} POST_BUILD
                COMMAND ${CMAKE_COMMAND} -E make_directory "$<TARGET_FILE_DIR:${TEST_TARGET}>/extensions"
                COMMAND ${CMAKE_COMMAND} -E copy_if_different
                    "$<TARGET_FILE:nvjpeg2k_ext>"
                    "$<TARGET_FILE_DIR:${TEST_TARGET}>/extensions/"
                COMMENT "Copying nvjpeg2k extension for ${TEST_TARGET}"
            )
        endif()
    endforeach()
endif()

# Status messages
list(LENGTH TEST_SOURCES TEST_COUNT)
list(LENGTH BENCHMARK_SOURCES BENCH_COUNT)
math(EXPR TEST_COUNT "${TEST_COUNT} - 1")  # Subtract test_main.cpp
math(EXPR BENCH_COUNT "${BENCH_COUNT} - 1")  # Subtract test_main.cpp

message(STATUS "Tests configured with ${CMAKE_BUILD_TYPE} build type")
message(STATUS "  Unit tests: ${TEST_COUNT} files -> lichtfeld_tests")
message(STATUS "  Benchmarks: ${BENCH_COUNT} files -> lichtfeld_benchmarks")
message(STATUS "  Build tests: ninja lichtfeld_tests")
message(STATUS "  Build benchmarks: ninja lichtfeld_benchmarks")
message(STATUS "  Run tests: ninja run_tests")
message(STATUS "  Run benchmarks: ninja run_benchmarks")

# SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
# SPDX-License-Identifier: GPL-3.0-or-later

# Testing module - comprehensive tests for all components

# Find GTest
find_package(GTest CONFIG REQUIRED)

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

# Test sources
set(TEST_SOURCES
    test_main.cpp       # Custom main with pinned memory pre-warming
)

# Check if test files exist and add them
set(OPTIONAL_TEST_FILES
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
    test_tensor_views.cpp
    test_tensor_math.cpp
    test_tensor_utils.cpp
    test_tensor_torch_compat.cpp
    test_tensor_advanced.cpp
    test_tensor_stress.cpp
    test_tensor_matrix.cpp
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
    test_tensor_ops_comparison.cpp
    test_sort_minimal.cpp
    test_tensor_debug.cpp
    test_bugs_found.cpp
    test_tensor_vs_torch.cpp
    test_tensor_move.cpp
    test_mcmc.cpp
    test_mcmc_add_new_gs_comparison.cpp
    test_mcmc_relocate_optimizer_state_bug.cpp
    test_tensor_benchmark.cpp
    test_memory_pool_benchmark.cpp
    test_broadcast_benchmark.cpp
    test_reduction_benchmark.cpp
    test_expression_templates.cpp
    test_fusion_benchmark.cpp
    test_gather_benchmark.cpp
    test_zip_gather_benchmark.cpp
    test_transform_iterator_benchmark.cpp
    test_cub_reduction_benchmark.cpp
    benchmark_strided_tensors.cpp
    benchmark_grad_alpha_layout.cpp
    benchmark_background_blend.cpp
    test_scalar_reduction_benchmark.cpp
    test_permute_upload_benchmark.cpp
    test_lfs_adam_optimizer.cpp
    test_lfs_scheduler.cpp
    benchmark_scheduler.cpp
    test_lfs_losses.cpp
    benchmark_losses.cpp
    test_mcmc_kernels.cpp
    test_tensor_bool.cpp
    test_dataloader_lfs.cpp
    test_bilateral_grid.cpp
    test_sparsity_debug.cpp
    test_default_strategy.cpp
    benchmark_default_strategy.cpp
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
    test_interleaved_slice_copy.cpp
    test_default_strategy_tensor_ops.cpp
    test_tensor_zero_dimension.cpp
    test_mask_loss.cpp
    test_checkpoint_resume.cpp
    test_nan_inf_gpu_check.cpp
    test_mcmc_nan_fix.cpp
)

foreach(TEST_FILE ${OPTIONAL_TEST_FILES})
    if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/${TEST_FILE}")
        list(APPEND TEST_SOURCES ${TEST_FILE})
        message(STATUS "Found test file: ${TEST_FILE}")
    else()
        message(STATUS "Test file not found (skipping): ${TEST_FILE}")
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

# Enable testing
include(GoogleTest)
gtest_discover_tests(lichtfeld_tests
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
    PROPERTIES TIMEOUT 30
)

# Custom target to run tests with output
add_custom_target(run_tests
    COMMAND ${CMAKE_CTEST_COMMAND} --output-on-failure -V
    DEPENDS lichtfeld_tests
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
    COMMENT "Running all tests with verbose output"
)

# Platform-specific settings
if(WIN32)
    # Copy Torch DLLs for testing on Windows
    file(GLOB TORCH_DLLS "${Torch_DIR}/../../../lib/*.dll")
    foreach(TORCH_DLL ${TORCH_DLLS})
        add_custom_command(
            TARGET lichtfeld_tests
            POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy_if_different "${TORCH_DLL}"
            "$<TARGET_FILE_DIR:lichtfeld_tests>")
    endforeach()
endif()

message(STATUS "Tests configured with ${CMAKE_BUILD_TYPE} build type")
message(STATUS "  Test sources found: ${TEST_SOURCES}")
message(STATUS "  Build with: ninja lichtfeld_tests")
message(STATUS "  Run with: ninja run_tests or ctest --output-on-failure")

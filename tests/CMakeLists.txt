# SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
# SPDX-License-Identifier: GPL-3.0-or-later

# Testing module - comprehensive tests for all components

# Find GTest
find_package(GTest CONFIG REQUIRED)

# Note: WebP and LibArchive are already found by parent CMakeLists.txt
# and will be available through gs_core target

# Test sources - START WITH ONLY EXISTING FILES
set(TEST_SOURCES
    test_main.cpp       # Custom main with pinned memory pre-warming
    test_geometry.cpp
)

# Check if new test files exist and add them
set(OPTIONAL_TEST_FILES
    test_regularization.cpp
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
    test_tensor_bitwise.cpp
    test_tensor_accessor.cpp
    test_tensor_random_advanced.cpp
    test_tensor_indexing_advanced.cpp
    test_tensor_clamping.cpp
    test_tensor_compat.cpp
    test_tensor_cumsum_convenience.cpp
    test_tensor_conversions_shape.cpp
    test_kmeans.cpp
    test_critical_fixes.cpp
    test_tensor_ops_comparison.cpp
    test_sort_minimal.cpp
    test_tensor_debug.cpp
    test_bugs_found.cpp
    test_tensor_vs_torch.cpp
    test_tensor_move.cpp
    test_mcmc.cpp
    test_tensor_benchmark.cpp
    test_memory_pool_benchmark.cpp  # benchmarks CudaMemoryPool (now internal)
    test_broadcast_benchmark.cpp
    test_reduction_benchmark.cpp
    test_expression_templates.cpp
    test_fusion_benchmark.cpp
    test_gather_benchmark.cpp
    test_zip_gather_benchmark.cpp
    test_transform_iterator_benchmark.cpp
    test_cub_reduction_benchmark.cpp
    benchmark_strided_tensors.cpp
    test_scalar_reduction_benchmark.cpp
    test_permute_upload_benchmark.cpp
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
    ${CUDAToolkit_INCLUDE_DIRS}
    ${OPENGL_INCLUDE_DIRS}
)

# Link libraries - Fix linking order and add missing dependencies
target_link_libraries(lichtfeld_tests PRIVATE
    # Core libraries (in dependency order)
    gs_visualizer
    gs_rendering
    gs_training
    gs_training_kernels
    gs_loader
    gs_project
    gs_geometry
    gs_core
    gs_kernels
    
    # Backend libraries
    gsplat_backend
    fastgs_backend
    
    # External libraries (using custom test_main.cpp instead of gtest_main)
    GTest::gtest
    ${TORCH_LIBRARIES}
    ${OPENGL_LIBRARIES}
    
    # CUDA libraries
    CUDA::cudart
    CUDA::cublas
    CUDA::curand
    CUDA::cupti      # Required by PyTorch profiling

    # System libraries
    spdlog::spdlog
)

# Compiler options for CUDA debugging
target_compile_options(lichtfeld_tests PRIVATE
    $<$<AND:$<COMPILE_LANGUAGE:CUDA>,$<CONFIG:Debug>>:-G -lineinfo -Xcudafe --device-debug>
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

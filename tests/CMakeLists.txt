# SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
# SPDX-License-Identifier: GPL-3.0-or-later

# Testing module - comprehensive tests for all components

# =============================================================================
# Headless localization contracts (no LibTorch/CUDA)
# =============================================================================
if(BUILD_LOCALIZATION_TESTS)
    find_package(Python REQUIRED COMPONENTS Interpreter)
    add_test(
        NAME LocalizationContracts
        COMMAND ${Python_EXECUTABLE}
                ${CMAKE_SOURCE_DIR}/tests/python/test_localization_contracts.py
    )
    set_tests_properties(LocalizationContracts PROPERTIES
        WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
        TIMEOUT 30
    )
    add_custom_target(test_localization_contracts
        COMMAND ${CMAKE_CTEST_COMMAND} --output-on-failure -R LocalizationContracts
        DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/CTestTestfile.cmake
        USES_TERMINAL
    )
    message(STATUS "Registered headless localization contract tests (no LibTorch)")
    if(NOT BUILD_TESTS AND NOT BUILD_UNICODE_TEST_ONLY)
        return()
    endif()
endif()

# Find GTest
find_package(GTest CONFIG REQUIRED)

# Reuse nanobind headers for Python UI regression tests
if(NOT DEFINED NB_DIR)
    find_package(Python REQUIRED COMPONENTS Interpreter Development Development.Module Development.SABIModule)
    find_package(nanobind CONFIG REQUIRED)
endif()

if(COMMAND nanobind_build_library AND NOT TARGET nanobind-static)
    nanobind_build_library(nanobind-static)
endif()

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
    if(NOT BUILD_TESTS)
        return()  # Skip LibTorch-dependent tests when only Unicode coverage was requested.
    endif()
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

# Regular test sources (unit tests, integration tests)
set(TEST_FILES
    test_assert_hardening.cpp
    test_cuda_error.cpp
    test_exportable_storage.cpp
    test_vram_leak_regression.cpp
    test_cuda_error_typed.cpp
    test_cuda_launch_check.cpp
    test_tensor_launch_check.cpp
    test_device_fault.cpp
    device_fault_cuda_utils.cu
    test_selection_ops.cpp
    test_tensor_basic.cpp
    test_tensor_ops.cpp
    test_tensor_memory.cpp
    test_allocator_policy.cpp
    test_alloc_counter.cpp
    test_allocator_hygiene.cpp
    test_fastgs_sort_buffers.cpp
    test_fused_reg_loss.cpp
    test_fused_bg_blend.cpp
    test_photometric_hygiene.cpp
    test_persistent_masks.cpp
    test_fused_noise_injection.cpp
    test_densification_info_zero.cpp
    test_densify_events_4x.cpp
    test_training_state_ledger.cpp
    test_joint_adam_codec.cpp
    test_sh_value_codec.cpp
    test_sh_value_storage.cpp
    test_tensor_reduction.cpp
    test_tensor_reduction_alignment.cpp
    test_tensor_views.cpp
    test_tensor_advanced.cpp
    test_tensor_stress.cpp
    test_tensor_matrix.cpp
    test_slice_mm_bug.cpp
    test_tensor_random.cpp
    test_tensor_broadcast.cpp
    test_tensor_masking.cpp
    test_theme_a_strided_ops.cpp
    test_tensor_lazy_gather.cpp
    test_zip_gather.cpp
    test_tensor_ordering_distance.cpp
    test_tensor_kmeans_workflow.cpp
    test_tensor_masked_select_rows.cpp
    test_tensor_bitwise.cpp
    test_tensor_accessor.cpp
    test_tensor_random_advanced.cpp
    test_tensor_indexing_advanced.cpp
    test_tensor_clamping.cpp
    test_tensor_cumsum_convenience.cpp
    test_tensor_conversions_shape.cpp
    test_tensor_reserve_inplace_cat.cpp
    test_mcmc.cpp
    test_expression_templates.cpp
    test_lfs_scheduler.cpp
    test_mrnf_strategy.cpp
    test_compact_splats_peak.cpp
    test_adam_capacity_invariant.cpp
    test_sparsity_optimizer.cpp
    test_mcmc_tensor_ops.cpp
    test_training_cropbox_mask.cpp
    test_training_parameters.cpp
    test_tensor_index_select_uint8.cpp
    test_tensor_uint8_inversion.cpp
    test_tensor_uint8_masking.cpp
    test_tensor_serialization.cpp
    test_tensor_cat_reduction_bug.cpp
    test_tensor_inplace_capacity.cpp
    test_sog_format.cpp
    test_ply_error_taxonomy.cpp
    test_gsplat_rasterizer.cpp
    test_spz_format.cpp
    test_converter_usd.cpp
    test_usd_format.cpp
    test_nurec_usdz_format.cpp
    test_unicode_paths_windows.cpp
    test_interleaved_slice_copy.cpp
    test_densification_tensor_ops.cpp
    test_tensor_zero_dimension.cpp
    test_mask_loss.cpp
    test_mask_preprocess_fusion.cpp
    test_roi_weight_map.cpp
    test_roi_weighted_losses.cpp
    test_checkpoint_resume.cpp
    test_atomic_output.cpp
    test_training_terminal_state.cpp
    test_headless_run_coordinator.cpp
    test_argument_parser.cpp
    test_crash_handler.cpp
    test_logger.cpp
    test_system_info.cpp
    test_parameter_manager.cpp
    test_mcp_protocol.cpp
    test_mcp_training_state.cpp
    test_mcp_app_utils.cpp
    test_mcp_operator_schema.cpp
    test_mcp_shared_scene_tools.cpp
    test_mcp_sequencer_tools.cpp
    test_tcp_publisher.cpp
    test_view_info_json.cpp
    test_video_export_utils.cpp
    test_video_frame_extractor_naming.cpp
    test_video_hdr_logic.cpp
    test_visualizer_post_work.cpp
    test_panel_registry_animation_demand.cpp
    test_panel_layout_render_demand.cpp
    test_reactive_store.cpp
    test_video_color_convert.cpp
    test_python_integration.cpp
    test_python_lsp_client.cpp
    test_uv_runner.cpp
    test_nan_inf_gpu_check.cpp
    test_mcmc_nan_fix.cpp
    test_stride_bugs.cpp
    test_strided_fill.cpp
    test_expanded_tensor_ops.cpp
    test_bool_any_all.cpp
    test_fused_l1_ssim.cpp
    test_loss_workspace_union.cpp
    test_image_kernels.cpp
    test_add_new_gs_tensor_ops.cpp
    test_cpu_dtype_conversions.cpp
    test_gradient_accumulation.cpp
    test_image_resize_parallel.cpp
    test_tensor_slicing_safety.cpp
    test_tensor_deferred_d1_copy_registry.cpp
    test_tensor_deferred_d3_pin_operands.cpp
    test_tensor_lazy_ir.cpp
    test_tensor_dispatch_6a.cpp
    test_tensor_kernels_6c.cpp
)

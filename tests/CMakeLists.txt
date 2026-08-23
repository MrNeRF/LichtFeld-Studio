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
    if(NOT BUILD_TESTS AND NOT BUILD_UNICODE_TEST_ONLY AND NOT BUILD_FORMAT_TESTS)
        return()
    endif()
endif()

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
    if(COMMAND lfs_enable_fast_cxx_compile)
        lfs_enable_fast_cxx_compile(unicode_path_test)
    endif()

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

# MSVC resolves every object reference before /OPT:REF, while GNU ld can discard
# unreferenced function sections before resolution. Keep the dependency-heavy
# chapter coverage in the non-Windows composition.
set(LICHT_FORMAT_WINDOWS_FILES
    test_json_chapter_dom.cpp
    test_project_chapters.cpp
    test_project_container.cpp
    licht_format_test_runtime.cpp
    ${CMAKE_SOURCE_DIR}/src/core/error.cpp
    ${CMAKE_SOURCE_DIR}/src/core/memory_domain.cpp
    ${CMAKE_SOURCE_DIR}/src/core/optimization_properties.cpp
    ${CMAKE_SOURCE_DIR}/src/core/parameters.cpp
    ${CMAKE_SOURCE_DIR}/src/core/property_registry.cpp
    ${CMAKE_SOURCE_DIR}/src/core/uuid.cpp
    ${CMAKE_SOURCE_DIR}/src/io/json_chapter_dom.cpp
    ${CMAKE_SOURCE_DIR}/src/io/project/project_chapters.cpp
    ${CMAKE_SOURCE_DIR}/src/io/project/crc32c.cpp
    ${CMAKE_SOURCE_DIR}/src/io/project/project_file.cpp
    ${CMAKE_SOURCE_DIR}/src/io/project/project_container.cpp
    ${CMAKE_SOURCE_DIR}/src/io/project/project_writer.cpp
    ${CMAKE_SOURCE_DIR}/src/io/project/project_recovery.cpp
)

set(LICHT_FORMAT_NON_WINDOWS_FILES
    ${LICHT_FORMAT_WINDOWS_FILES}
    test_selection_chapter.cpp
    test_p8_compatibility.cpp
    licht_format_tensor_runtime.cpp
    ${CMAKE_SOURCE_DIR}/src/core/checkpoint_format.cpp
    ${CMAKE_SOURCE_DIR}/src/core/tensor/tensor_serialization.cpp
    ${CMAKE_SOURCE_DIR}/src/io/project/geometry_payload.cpp
    ${CMAKE_SOURCE_DIR}/src/io/project/project_document.cpp
    ${CMAKE_SOURCE_DIR}/src/io/project/selection_chapter.cpp
    ${CMAKE_SOURCE_DIR}/src/io/project/session_chapters.cpp
    ${CMAKE_SOURCE_DIR}/src/io/project/splat_chapter.cpp
    ${CMAKE_SOURCE_DIR}/src/visualizer/gui/layout_state.cpp
)
# test_project_document.cpp is excluded: it links headless recovery, training strategies, and visualizer parameter state.
# test_p5_session_chapters.cpp is excluded: it links the Python editor and TrainingManager runtime.

if(WIN32)
    set(LICHT_FORMAT_TARGET_FILES ${LICHT_FORMAT_WINDOWS_FILES})
else()
    set(LICHT_FORMAT_TARGET_FILES ${LICHT_FORMAT_NON_WINDOWS_FILES})
endif()

add_executable(lichtfeld_format_tests ${LICHT_FORMAT_TARGET_FILES})
target_include_directories(lichtfeld_format_tests PRIVATE
    ${CMAKE_SOURCE_DIR}/src/core/include
    ${CMAKE_SOURCE_DIR}/src/io/include
    ${CMAKE_SOURCE_DIR}/src
    ${CMAKE_SOURCE_DIR}/tests
    ${CMAKE_BINARY_DIR}/include
)
if(NOT WIN32)
    target_include_directories(lichtfeld_format_tests PRIVATE
        ${CMAKE_SOURCE_DIR}/src/diagnostics/include
        ${CMAKE_SOURCE_DIR}/src/training
        ${CMAKE_SOURCE_DIR}/src/training/include
        ${CMAKE_SOURCE_DIR}/src/visualizer
        ${CUDAToolkit_INCLUDE_DIRS}
    )
endif()
target_compile_definitions(lichtfeld_format_tests PRIVATE
    PROJECT_ROOT_PATH="${CMAKE_SOURCE_DIR}"
)
if(WIN32)
    target_compile_definitions(lichtfeld_format_tests PRIVATE
        LFS_CORE_EXPORTS
        LFS_LOGGER_EXPORTS
    )
else()
    target_compile_definitions(lichtfeld_format_tests PRIVATE
        LFS_FORMAT_TEST_TARGET
    )
endif()
target_link_libraries(lichtfeld_format_tests PRIVATE
    GTest::gtest_main
    nlohmann_json::nlohmann_json
    xxHash::xxhash
    zstd::libzstd
    Threads::Threads
)
if(NOT WIN32)
    target_link_libraries(lichtfeld_format_tests PRIVATE OpenSSL::Crypto)
endif()
if(MSVC)
    target_compile_options(lichtfeld_format_tests PRIVATE /Gy)
    target_link_options(lichtfeld_format_tests PRIVATE /OPT:REF)
else()
    target_compile_options(lichtfeld_format_tests PRIVATE
        -ffunction-sections -fdata-sections
    )
    target_link_options(lichtfeld_format_tests PRIVATE
        -Wl,--no-export-dynamic -Wl,--gc-sections
    )
endif()
if(WIN32)
    target_link_libraries(lichtfeld_format_tests PRIVATE bcrypt)
endif()
if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU" AND
   CMAKE_CXX_COMPILER_VERSION VERSION_GREATER_EQUAL 12)
    target_link_libraries(lichtfeld_format_tests PRIVATE stdc++exp)
endif()
set_target_properties(lichtfeld_format_tests PROPERTIES
    CXX_STANDARD 23
    CXX_STANDARD_REQUIRED ON
)
if(COMMAND lfs_enable_fast_cxx_compile)
    lfs_enable_fast_cxx_compile(lichtfeld_format_tests)
endif()
add_test(NAME lichtfeld_format_tests
    COMMAND $<TARGET_FILE:lichtfeld_format_tests> --gtest_color=no
)
set_tests_properties(lichtfeld_format_tests PROPERTIES
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
    LABELS "fast;format"
    TIMEOUT 300
)

if(NOT BUILD_TESTS)
    message(STATUS "Building CPU-only format tests (no CUDA); full suite disabled")
    return()  # lichtfeld_format_tests is defined above; the rest needs CUDA/Torch
endif()

# Reuse nanobind headers for Python UI regression tests
if(NOT DEFINED NB_DIR)
    find_package(Python REQUIRED COMPONENTS Interpreter Development Development.Module Development.SABIModule)
    find_package(nanobind CONFIG REQUIRED)
endif()

if(COMMAND nanobind_build_library AND NOT TARGET nanobind-static)
    nanobind_build_library(nanobind-static)
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
    licht_test_support.cpp
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
    test_joint_bounds_grow_only.cpp
    test_steady_alloc_invariant.cpp
    test_allocator_hygiene.cpp
    test_pinned_stream_teardown.cpp
    test_cuda_pool_stream_teardown.cpp
    test_stale_stream_teardown.cpp
    test_fastgs_sort_buffers.cpp
    test_warp_cull_blend.cpp
    test_warp_cull_bwd.cpp
    test_fused_reg_loss.cpp
    test_fused_bg_blend.cpp
    test_photometric_hygiene.cpp
    test_persistent_masks.cpp
    test_fused_noise_injection.cpp
    test_densification_info_zero.cpp
    test_densify_events_4x.cpp
    test_training_state_ledger.cpp
    test_vram_ledger_model.cpp
    test_host_metrics.cpp
    test_joint_adam_codec.cpp
    test_sh_value_codec.cpp
    test_sh_value_storage.cpp
    test_morton_reorder.cpp
    test_transform_bake_q16.cpp
    test_splat_data_clone.cpp
    test_tensor_reduction.cpp
    test_tensor_reduction_alignment.cpp
    test_tensor_views.cpp
    test_tensor_advanced.cpp
    test_tensor_stress.cpp
    test_tensor_matrix.cpp
    test_slice_mm_bug.cpp
    test_tensor_random.cpp
    test_tensor_broadcast.cpp
    test_zero_stride_expand.cpp
    test_strided_reduce_w2.cpp
    test_tensor_masking.cpp
    test_strided_tensor_ops.cpp
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
    test_dual_rep_optimizer.cpp
    test_gut_shN_joint_adam.cpp
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
    test_ply_io_throughput.cpp
    test_gsplat_rasterizer.cpp
    test_spz_format.cpp
    test_provenance.cpp
    test_converter_usd.cpp
    test_converter_licht.cpp
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
    test_component_config_serialization.cpp
    test_training_snapshot_service.cpp
    test_atomic_output.cpp
    test_geometry_payload.cpp
    test_selection_chapter.cpp
    test_project_document.cpp
    test_project_lifecycle.cpp
    test_p8_compatibility.cpp
    test_job_registry.cpp
    test_p5_session_chapters.cpp
    test_training_terminal_state.cpp
    test_headless_run_coordinator.cpp
    test_argument_parser.cpp
    test_user_paths.cpp
    test_window_state.cpp
    test_crash_handler.cpp
    test_gpu_teardown_order.cpp
    test_logger.cpp
    test_exif.cpp
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
    test_panel_registry_default_closed.cpp
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
    test_tensor_kernels_6c_elem.cpp
    test_tensor_lazy_stateful_ops.cpp
    test_fastgs_kernels.cpp
    test_fastgs_fuzz.cpp
    test_fastgs_tile_culling_boundary.cpp
    fastgs_tile_culling_boundary_cuda.cu
    test_sh_swizzle_layout.cpp
    test_python_io.cpp
    test_extended_unary_ops_vs_torch.cpp
    test_scene_validity.cpp
    test_uuid.cpp
    test_render_settings_defaults.cpp
    test_scene_upscaler_registry.cpp
    test_selection_rasterization_ops.cpp
    test_rendering_refactor_services.cpp
    test_theme_registry.cpp
    test_viewport.cpp
    test_input_controller_focus.cpp
    test_input_controller_dataset_load.cpp
    test_selection_service_interactions.cpp
    test_scene_graph_identity.cpp
    test_undo_history.cpp
    test_selection_command_dispatch.cpp
    test_selection_operator_modal.cpp
    test_operator_registry_props.cpp
    test_fastgs_analytical_gradients.cpp
    test_normal_prior_decode.cpp
    test_normal_auto_generate.cpp
    test_metrics_evaluator.cpp
    test_background_image.cpp
    test_nvcodec_jpeg2k_16bit.cpp
    test_compressed_blob_tier.cpp
    test_pipelined_loader.cpp
    test_pipelined_loader_ledger.cpp
    test_pipelined_loader_sidecar_jpeg2k.cpp
    test_operator_return_value.cpp
    test_sdl_rml_key_mapping.cpp
    test_rml_path_utils.cpp
    test_rml_static_style_boundaries.cpp
    test_sequencer_regressions.cpp
    test_tensor_storage_version.cpp
    test_modal_registry_regression.cpp
    test_tensor_nn_ops.cpp
    test_nn_ops.cpp
    test_moge2.cpp
    bench_nn_ops.cpp
    test_ppisp_controller.cpp
    test_ppisp_regularization.cpp
    test_ppisp_sidecar.cpp
    test_ppisp_checkpoint_roundtrip.cpp
    test_ppisp_cuda_vs_torch.cpp
    test_ppisp_banded_region.cpp
    test_ppisp_exposure_seed.cpp
    test_metrics_evaluator_appearance.cpp
    test_export_env_composite.cpp
    test_undistort.cpp
    test_viewport_region_utils.cpp
    test_viewport_interop_decision.cpp
    test_mesh_data.cpp
    test_mesh_offscreen_renderer.cpp
    test_openmesh_bridge.cpp
    test_mesh_loader.cpp
    test_colmap_image_layout.cpp
    test_missing_dataset_images.cpp
    test_colmap_points3d_text.cpp
    test_colmap_binary_error_taxonomy.cpp
    test_splat_simplify.cpp
    test_rml_text_input_handler.cpp
    test_rml_im_mode_layout.cpp
    test_status_bar_fit.cpp
    test_rotated_sh_correctness.cpp
    test_splat_raw_payload.cpp
    test_tensor_stream.cpp
    test_tensor_d4_from_blob_stream.cpp
    tensor_hardening_cuda_utils.cu
    test_cuda_event_pool.cpp
    test_tensor_multistream.cpp
    test_tensor_discovery.cpp
    test_tensor_discovery_fuzz.cpp
    test_mcmc_multinomial.cpp
    test_arena_metrics_contention.cpp
    test_memory_arena_shutdown.cpp
    test_tensor_memory_planner.cpp
    test_vram_profiler_metrics.cpp
    test_memory_pressure.cpp
    test_error_core.cpp
    test_error_reporter.cpp
    test_error_bus.cpp
    test_run_guarded.cpp
    test_tcp_responder.cpp
    test_error_envelope.cpp
)
list(APPEND TEST_FILES test_vksplat_input_packer.cpp)
list(APPEND TEST_FILES test_viewer_sh_zerocopy_triad.cpp)
# Phase 7A: bounded-wait machine + SubmissionState + QW-6 (GPU-free).
list(APPEND TEST_FILES test_vulkan_wait_bounded.cpp)
list(APPEND TEST_FILES test_submission_state_transitions.cpp)
list(APPEND TEST_FILES test_vksplat_failed_submit_no_publish.cpp)
# Epic #1496: computed barriers (GPU-free planner/tracker suites).
list(APPEND TEST_FILES test_vksplat_barrier_planner.cpp)
list(APPEND TEST_FILES test_vksplat_tagged_dispatch.cpp)
list(APPEND TEST_FILES test_vksplat_buffer_retire.cpp)
list(APPEND TEST_FILES test_vulkan_image_tracker.cpp)
list(APPEND TEST_FILES test_vulkan_gpu_census.cpp)
# Epic #1497: output image pool (GPU-free host bookkeeping).
list(APPEND TEST_FILES test_output_image_pool.cpp)
# Epic #1568 / #1567: output slot ring (GPU-free host bookkeeping).
list(APPEND TEST_FILES test_output_slot_ring.cpp)
# Epic #1577 / #1574: readback ticket ring (GPU-free host bookkeeping).
list(APPEND TEST_FILES test_readback_ticket_ring.cpp)
# Epic #1568 / #1566: payload-generic GpuResourcePool + subrect fit helper.
list(APPEND TEST_FILES test_gpu_resource_pool.cpp)
list(APPEND TEST_FILES test_viewport_scratch_bucket.cpp)
list(APPEND TEST_FILES test_lod_page_cache.cpp)
list(APPEND TEST_FILES test_lod_page_dequant.cpp)
list(APPEND TEST_FILES test_lod_upload_engine.cpp)
list(APPEND TEST_FILES test_octree_lod.cpp)
list(APPEND TEST_FILES test_ply_to_rad_lod.cpp)
list(APPEND TEST_FILES test_rad_gpu_encode.cpp)
list(APPEND TEST_FILES test_rad_meta_sidecar.cpp)

# Keep the source manifest strict: a stale entry should fail configuration
# instead of silently dropping coverage from the executable.
set(TEST_SOURCES test_main.cpp ${TEST_FILES})

# PyModalRegistry and its native-modal dispatch path for regression tests.
list(APPEND TEST_SOURCES
    ${CMAKE_SOURCE_DIR}/src/python/lfs/py_error.cpp
    ${CMAKE_SOURCE_DIR}/src/python/lfs/py_modal_registry.cpp
    ${CMAKE_SOURCE_DIR}/src/python/lfs/py_operator_properties.cpp
    ${CMAKE_SOURCE_DIR}/src/python/lfs/py_ui_modals.cpp
    ${CMAKE_SOURCE_DIR}/src/python/lfs/rml_im_mode_layout.cpp
    ${CMAKE_SOURCE_DIR}/src/app/mcp_sequencer_tools.cpp
    ${CMAKE_SOURCE_DIR}/src/app/mcp_operator_schema.cpp
    ${CMAKE_SOURCE_DIR}/src/app/converter.cpp
    ${CMAKE_SOURCE_DIR}/src/python/lfs/py_params.cpp
)

# Create test executable
add_executable(lichtfeld_tests ${TEST_SOURCES})

# USD (pxr) headers from vcpkg pull in deprecated <ext/hash_set> and emit
# #warning (-Wcpp). Third-party; suppress only for this translation unit.
if(NOT MSVC)
    set_source_files_properties(
        test_usd_format.cpp
        PROPERTIES COMPILE_OPTIONS "-Wno-cpp"
    )
endif()

# Python integration tests import the in-tree extension at runtime.
add_dependencies(lichtfeld_tests lfs_py)

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
    ${CMAKE_SOURCE_DIR}/src/app/include
    ${CMAKE_SOURCE_DIR}/src/visualizer # For visualizer-relative headers (e.g. gui/ui_widgets.hpp)
    ${CMAKE_SOURCE_DIR}/external/spz   # SPZ library headers
    ${NB_DIR}/include
    ${CUDAToolkit_INCLUDE_DIRS}
)

# Define TEST_DATA_DIR and PROJECT_ROOT_PATH for tests
target_compile_definitions(lichtfeld_tests PRIVATE
    TEST_DATA_DIR="${CMAKE_SOURCE_DIR}/data"
    PROJECT_ROOT_PATH="${CMAKE_SOURCE_DIR}"
)

# Large test assets are hosted in the public Cloudflare R2 bucket
# 'lichtfeld-test-assets' and fetched at configure time. A failed download only
# warns: tests guard on the file's existence and GTEST_SKIP when it is missing.
set(LFS_SIMPLIFY_TEST_PLY "${CMAKE_SOURCE_DIR}/data/kerstbol-isolated-rotated_137502.ply")
set(LFS_SIMPLIFY_TEST_PLY_SHA256 "7fc85d1b0f4b7ab26acaa988b397b41b4ecf83f90e78cfb91b65ed474cb671fc")
set(LFS_SIMPLIFY_TEST_PLY_OK FALSE)
if(EXISTS "${LFS_SIMPLIFY_TEST_PLY}")
    file(SHA256 "${LFS_SIMPLIFY_TEST_PLY}" LFS_SIMPLIFY_TEST_PLY_HASH)
    if(LFS_SIMPLIFY_TEST_PLY_HASH STREQUAL LFS_SIMPLIFY_TEST_PLY_SHA256)
        set(LFS_SIMPLIFY_TEST_PLY_OK TRUE)
    endif()
endif()
if(NOT LFS_SIMPLIFY_TEST_PLY_OK)
    message(STATUS "Downloading simplify regression asset kerstbol-isolated-rotated_137502.ply (33 MB)")
    file(DOWNLOAD
        "https://pub-c3fee24842994e5d866dbd9cc41c3f66.r2.dev/kerstbol-isolated-rotated_137502.ply"
        "${LFS_SIMPLIFY_TEST_PLY}.tmp"
        INACTIVITY_TIMEOUT 60
        TIMEOUT 300
        STATUS LFS_SIMPLIFY_TEST_PLY_STATUS
    )
    list(GET LFS_SIMPLIFY_TEST_PLY_STATUS 0 LFS_SIMPLIFY_TEST_PLY_RC)
    list(GET LFS_SIMPLIFY_TEST_PLY_STATUS 1 LFS_SIMPLIFY_TEST_PLY_ERR)
    if(LFS_SIMPLIFY_TEST_PLY_RC EQUAL 0)
        file(SHA256 "${LFS_SIMPLIFY_TEST_PLY}.tmp" LFS_SIMPLIFY_TEST_PLY_HASH)
        if(LFS_SIMPLIFY_TEST_PLY_HASH STREQUAL LFS_SIMPLIFY_TEST_PLY_SHA256)
            file(RENAME "${LFS_SIMPLIFY_TEST_PLY}.tmp" "${LFS_SIMPLIFY_TEST_PLY}")
            set(LFS_SIMPLIFY_TEST_PLY_OK TRUE)
        else()
            set(LFS_SIMPLIFY_TEST_PLY_ERR "SHA256 mismatch: got ${LFS_SIMPLIFY_TEST_PLY_HASH}")
        endif()
    endif()
    if(NOT LFS_SIMPLIFY_TEST_PLY_OK)
        file(REMOVE "${LFS_SIMPLIFY_TEST_PLY}.tmp")
        message(WARNING "Failed to download simplify regression asset "
            "kerstbol-isolated-rotated_137502.ply (${LFS_SIMPLIFY_TEST_PLY_ERR}); "
            "SplatSimplify.RealPlySimplifyDoesNotUndershootRequestedTarget will be skipped.")
    endif()
endif()

if(CMAKE_CXX_LINK_GROUP_USING_RESCAN_SUPPORTED)
    set(LFS_TEST_INTERNAL_LINK_LIBS
        "$<LINK_GROUP:RESCAN,lfs_mcp,lfs_tcp,lfs_visualizer,lfs_sequencer,lfs_rendering,lfs_training,lfs_core,lfs_geometry,lfs_io,lfs_video,fastlfs_backend,spz_lib,lfs_python_utils>"
    )
else()
    set(LFS_TEST_INTERNAL_LINK_LIBS
        lfs_mcp
        lfs_tcp
        lfs_visualizer
        lfs_sequencer
        lfs_rendering
        lfs_training
        lfs_core
        lfs_geometry
        lfs_io
        lfs_video
        fastlfs_backend
        spz_lib
        lfs_python_utils
    )
endif()

# Link libraries
target_link_libraries(lichtfeld_tests PRIVATE
    ${LFS_TEST_INTERNAL_LINK_LIBS}

    # External libraries
    GTest::gtest
    nanobind-static
    ${TORCH_LIBRARIES}
    OpenImageIO::OpenImageIO
    OpenMeshCore
    assimp::assimp

    # CUDA libraries
    CUDA::cudart
    CUDA::curand
    CUDA::cupti

    # System libraries
    spdlog::spdlog
    SDL3::SDL3
)

target_link_libraries(lichtfeld_tests PRIVATE lfs_vulkan_rasterizer Vulkan::Vulkan)

# Compiler options for CUDA debugging
target_compile_options(lichtfeld_tests PRIVATE
    $<$<AND:$<COMPILE_LANGUAGE:CUDA>,$<CONFIG:Debug>>:-lineinfo>
)
if(MSVC)
    target_compile_options(lichtfeld_tests PRIVATE
        $<$<AND:$<COMPILE_LANGUAGE:CXX>,$<CONFIG:Debug>>:/Od /Z7>
        $<$<AND:$<COMPILE_LANGUAGE:CXX>,$<CONFIG:Release>>:/O2 /DNDEBUG>
    )
else()
    target_compile_options(lichtfeld_tests PRIVATE
        $<$<AND:$<COMPILE_LANGUAGE:CXX>,$<CONFIG:Debug>>:-O0 -g -fno-omit-frame-pointer>
        $<$<AND:$<COMPILE_LANGUAGE:CXX>,$<CONFIG:Release>>:-O3 -DNDEBUG>
    )
endif()
if(COMMAND lfs_enable_fast_cxx_compile)
    lfs_enable_fast_cxx_compile(lichtfeld_tests)
endif()

# Tensor-library hardening is intentionally isolated from the default test
# tiers.  During the hardening phases these oracle tests are expected to expose
# known failures, and crash-prone reproducers must not terminate the monolithic
# test process before unrelated tests run.
add_executable(tensor_hardening_tests
    test_main.cpp
    tensor_hardening_cuda_utils.cu
    test_tensor_hardening_theme_a.cpp
    test_tensor_hardening_theme_b.cpp
    test_tensor_hardening_theme_c.cpp
    test_tensor_hardening_theme_d.cpp
    test_tensor_hardening_theme_e.cpp
    test_tensor_hardening_theme_f.cpp
)

set_target_properties(tensor_hardening_tests PROPERTIES
    CUDA_STANDARD 20
    CUDA_STANDARD_REQUIRED ON
    CXX_STANDARD 23
    CXX_STANDARD_REQUIRED ON
)

target_include_directories(tensor_hardening_tests PRIVATE
    ${CMAKE_SOURCE_DIR}/src
    ${CMAKE_SOURCE_DIR}/tests
    ${CUDAToolkit_INCLUDE_DIRS}
)

target_link_libraries(tensor_hardening_tests PRIVATE
    lfs_core
    GTest::gtest
    ${TORCH_LIBRARIES}
    CUDA::cudart
    CUDA::curand
)

if(MSVC)
    target_compile_options(tensor_hardening_tests PRIVATE
        $<$<AND:$<COMPILE_LANGUAGE:CXX>,$<CONFIG:Debug>>:/Od /Z7>
        $<$<AND:$<COMPILE_LANGUAGE:CXX>,$<CONFIG:Release>>:/O2 /DNDEBUG>
    )
else()
    target_compile_options(tensor_hardening_tests PRIVATE
        $<$<AND:$<COMPILE_LANGUAGE:CXX>,$<CONFIG:Debug>>:-O0 -g -fno-omit-frame-pointer>
        $<$<AND:$<COMPILE_LANGUAGE:CXX>,$<CONFIG:Release>>:-O3 -DNDEBUG>
    )
endif()
if(COMMAND lfs_enable_fast_cxx_compile)
    lfs_enable_fast_cxx_compile(tensor_hardening_tests)
endif()

# =============================================================================
# CTest registration
# =============================================================================
# The monolithic test executable still mixes quick contracts with real-data,
# training, stress, and optional-codec cases. Register each tier as one filtered
# invocation: per-case discovery would pay process startup for thousands of
# sub-second unit tests and make the "fast" CTest label needlessly slow.
string(CONCAT LFS_NIGHTLY_TEST_FILTER
    "*CheckpointResumeTest.TrainSaveLoadResume/nightly_*:"
    "TensorStressTest.*:"
    "TensorMemoryTest.VeryLargeTensor:"
    "TensorMemoryTest.StressTest*:"
    "TensorMatrixTest.VeryLargeMatMul:"
    "TensorMaskingTest.StressTest*:"
    "BoolAnyAllTest.LargeBoolSumPreservesDensificationCounts:"
    "TensorZeroDimensionTest.StressTest*:"
    "TensorCatReductionBugTest.*VeryLarge*:"
    "MaskedSelectRowsTest.*LargeScale*:"
    "CPUDtypeConversionTest.ProductionMuralPointCloudWorkflow:"
    "FastGSFuzzTest.RandomStress_*:"
    "FastGSFuzzTest.VeryLargeBatch_100k:"
    "NaNInfGPUCheckTest.*5M*:"
    "NaNInfGPUCheckTest.RandomStress_*:"
    "NaNInfGPUCheckTest.InfDetection_LargeTensors:"
    "NaNInfGPUCheckTest.MultiDim_LargeTensors:"
    "StridedFillTest.VeryLargeGrid_3D_Fill:"
    "TensorLazyRuntimeTest.ErankExpressionMatchesTorchOnInheritedStream:"
    "TensorLazyRuntimeTest.FusedSegmentedSquareSumMatchesTorchOnInheritedStream:"
    "PlyToRadLod.OutOfCore*:"
    "SpzFormatTest.RealPlyRoundtrip:"
    "PythonIOTest.LoadRealPlyFile:"
    "ShSwizzleLayout.CanonicalRoundTrip_RealData:"
    "RotatedShCorrectnessTest.*:"
    "NvCodecImageLoaderJpeg2k16Bit.*:"
    "PipelinedImageLoaderSidecarJpeg2k.*"
)

string(CONCAT LFS_SLOW_TEST_FILTER
    "*CheckpointResumeTest.TrainSaveLoadResume/tiny_*:"
    "TrainingSetupRegressionTest.*:"
    "PipelinedImageLoaderTest.*:"
    "ImageResizeParallelTest.*:"
    "PythonIOTest.LoadCOLMAPDataset:"
    "PythonIOTest.PipelinedLoader*:"
    "PythonIntegrationTest.SceneCameraExposesVisualizerRenderContract:"
    "PythonLspClientTest.*:"
    "UvRunnerTest.*:"
    "ViewportFrameLifecycleServiceTest.*:"
    "ArenaMetricsContentionTest.*:"
    "TensorMultiStreamTest.TrainerViewerHandshake*:"
    "CheckpointAtomicReplaceFaultTest.*:"
    "MemoryPressureTest.InjectedFailure*:"
    "PPISPControllerTest.TrainingLoopConverges:"
    "UndistortCameraTest.BicyclePinholeNoDist:"
    "VideoColorConvertTest.*FullHD*:"
    "PlyToRadLod.*:"
    "LodPageDequant.*:"
    "LodUploadEngine.*:"
    "RadMetaSidecar.*:"
    "OctreeLod.*:"
    "NurecUsdzFormatTest.*:"
    "UsdFormatTest.RoundtripPreservesValues"
)

set(LFS_NON_FAST_TEST_FILTER
    "${LFS_SLOW_TEST_FILTER}:${LFS_NIGHTLY_TEST_FILTER}:DiscoveryFuzz.*:DiscoverySweep.*")

add_test(NAME lichtfeld.fast
    COMMAND $<TARGET_FILE:lichtfeld_tests>
        "--gtest_filter=*-${LFS_NON_FAST_TEST_FILTER}"
        --gtest_color=no
)
set_tests_properties(lichtfeld.fast PROPERTIES
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
    LABELS "fast;gpu"
    TIMEOUT 60
)

add_test(NAME lichtfeld.slow
    COMMAND $<TARGET_FILE:lichtfeld_tests>
        "--gtest_filter=${LFS_SLOW_TEST_FILTER}-${LFS_NIGHTLY_TEST_FILTER}"
        --gtest_color=no
)
set_tests_properties(lichtfeld.slow PROPERTIES
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
    LABELS "slow;gpu"
    TIMEOUT 300
)

add_test(NAME lichtfeld.nightly
    COMMAND $<TARGET_FILE:lichtfeld_tests>
        "--gtest_filter=${LFS_NIGHTLY_TEST_FILTER}"
        --gtest_color=no
)
set_tests_properties(lichtfeld.nightly PROPERTIES
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
    LABELS "nightly;gpu"
    TIMEOUT 1800
)

add_test(NAME tensor.hardening
    COMMAND $<TARGET_FILE:tensor_hardening_tests>
        --gtest_color=no
        --gtest_death_test_style=threadsafe
)
set_tests_properties(tensor.hardening PROPERTIES
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
    LABELS "hardening;gpu"
    TIMEOUT 900
)

add_test(NAME lichtfeld.discovery
    COMMAND $<TARGET_FILE:lichtfeld_tests>
        "--gtest_filter=DiscoveryFuzz.*:DiscoverySweep.*"
        --gtest_color=no
)

add_test(NAME system_info
    COMMAND $<TARGET_FILE:lichtfeld_tests>
        "--gtest_filter=SystemInfoTest.*:SessionBreadcrumbTest.*:LoggerTailTest.*"
        --gtest_color=no
)
set_tests_properties(system_info PROPERTIES
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
    LABELS "fast"
    TIMEOUT 30
)
set_tests_properties(lichtfeld.discovery PROPERTIES
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
    LABELS "discovery;gpu"
    TIMEOUT 1800
)

# Keep every Python tier explicit so new modules cannot silently drift out of
# the curated suite.
set(LFS_PYTHON_FAST_TESTS
    ${CMAKE_CURRENT_SOURCE_DIR}/python/test_bug_report.py
    ${CMAKE_CURRENT_SOURCE_DIR}/python/test_event_keypad_mapping.py
    ${CMAKE_CURRENT_SOURCE_DIR}/python/test_export_panel.py
    ${CMAKE_CURRENT_SOURCE_DIR}/python/test_gizmo_context.py
    ${CMAKE_CURRENT_SOURCE_DIR}/python/test_histogram_panel.py
    ${CMAKE_CURRENT_SOURCE_DIR}/python/test_http.py
    ${CMAKE_CURRENT_SOURCE_DIR}/python/test_import_dialog_panels.py
    ${CMAKE_CURRENT_SOURCE_DIR}/python/test_io_hygiene_regressions.py
    ${CMAKE_CURRENT_SOURCE_DIR}/python/test_keymap_bindings.py
    ${CMAKE_CURRENT_SOURCE_DIR}/python/test_verified_dead_code_cleanup.py
    ${CMAKE_CURRENT_SOURCE_DIR}/python/test_layout_composition.py
    ${CMAKE_CURRENT_SOURCE_DIR}/python/test_manage_stubs.py
    ${CMAKE_CURRENT_SOURCE_DIR}/python/test_mesh.py
    ${CMAKE_CURRENT_SOURCE_DIR}/python/test_mesh2splat_panel.py
    ${CMAKE_CURRENT_SOURCE_DIR}/python/test_operator_properties.py
    ${CMAKE_CURRENT_SOURCE_DIR}/python/test_operator_rich_returns.py
    ${CMAKE_CURRENT_SOURCE_DIR}/python/test_plugin_api_surface.py
    ${CMAKE_CURRENT_SOURCE_DIR}/python/test_plugin_marketplace_panel.py
    ${CMAKE_CURRENT_SOURCE_DIR}/python/test_plugin_system.py
    ${CMAKE_CURRENT_SOURCE_DIR}/python/test_plugin_templates.py
    ${CMAKE_CURRENT_SOURCE_DIR}/python/test_plugin_validator.py
    ${CMAKE_CURRENT_SOURCE_DIR}/python/test_preferences_panel.py
    ${CMAKE_CURRENT_SOURCE_DIR}/python/test_registry.py
    ${CMAKE_CURRENT_SOURCE_DIR}/python/test_rendering_panel_regressions.py
    ${CMAKE_CURRENT_SOURCE_DIR}/python/test_rendering_panel_simplify.py
    ${CMAKE_CURRENT_SOURCE_DIR}/python/test_rmlui_image_sources.py
    ${CMAKE_CURRENT_SOURCE_DIR}/python/test_runtime_state.py
    ${CMAKE_CURRENT_SOURCE_DIR}/python/test_scripts_panel.py
    ${CMAKE_CURRENT_SOURCE_DIR}/python/test_selection_controls_panel.py
    ${CMAKE_CURRENT_SOURCE_DIR}/python/test_selection_groups_panel.py
    ${CMAKE_CURRENT_SOURCE_DIR}/python/test_splat_lod_hierarchy.py
    ${CMAKE_CURRENT_SOURCE_DIR}/python/test_store.py
    ${CMAKE_CURRENT_SOURCE_DIR}/python/test_tensor.py
    ${CMAKE_CURRENT_SOURCE_DIR}/python/test_tensor_property.py
    ${CMAKE_CURRENT_SOURCE_DIR}/python/test_tool_registry_activation.py
    ${CMAKE_CURRENT_SOURCE_DIR}/python/test_training_manager_regressions.py
    ${CMAKE_CURRENT_SOURCE_DIR}/python/test_tools_new.py
    ${CMAKE_CURRENT_SOURCE_DIR}/python/test_training_panel_regressions.py
    ${CMAKE_CURRENT_SOURCE_DIR}/python/test_transform_controls_panel.py
    ${CMAKE_CURRENT_SOURCE_DIR}/python/test_undo.py
    ${CMAKE_CURRENT_SOURCE_DIR}/python/test_viewport_overlay_hooks.py
)

set(LFS_PYTHON_SLOW_TESTS
    ${CMAKE_CURRENT_SOURCE_DIR}/python/test_alignment_overlay_regressions.py
    ${CMAKE_CURRENT_SOURCE_DIR}/python/test_asset_manager_panel.py
    ${CMAKE_CURRENT_SOURCE_DIR}/python/test_async_plugin_loading.py
    ${CMAKE_CURRENT_SOURCE_DIR}/python/test_capability_thread_safety.py
    ${CMAKE_CURRENT_SOURCE_DIR}/python/test_errors.py
    ${CMAKE_CURRENT_SOURCE_DIR}/python/test_histogram_localization_regressions.py
    ${CMAKE_CURRENT_SOURCE_DIR}/python/test_hooks.py
    ${CMAKE_CURRENT_SOURCE_DIR}/python/test_hot_reload_edge_cases.py
    ${CMAKE_CURRENT_SOURCE_DIR}/python/test_hot_reload_fixes.py
    ${CMAKE_CURRENT_SOURCE_DIR}/python/test_menu_schemas.py
    ${CMAKE_CURRENT_SOURCE_DIR}/python/test_menubar_resources.py
    ${CMAKE_CURRENT_SOURCE_DIR}/python/test_operator_hot_reload.py
    ${CMAKE_CURRENT_SOURCE_DIR}/python/test_operator_system.py
    ${CMAKE_CURRENT_SOURCE_DIR}/python/test_pano_coordinate_regressions.py
    ${CMAKE_CURRENT_SOURCE_DIR}/python/test_plugin_callbacks.py
    ${CMAKE_CURRENT_SOURCE_DIR}/python/test_plugin_concurrency.py
    ${CMAKE_CURRENT_SOURCE_DIR}/python/test_plugin_docs.py
    ${CMAKE_CURRENT_SOURCE_DIR}/python/test_plugin_error_recovery.py
    ${CMAKE_CURRENT_SOURCE_DIR}/python/test_property_schema.py
    ${CMAKE_CURRENT_SOURCE_DIR}/python/test_property_system.py
    ${CMAKE_CURRENT_SOURCE_DIR}/python/test_signals.py
    ${CMAKE_CURRENT_SOURCE_DIR}/python/test_toolbar_hooks.py
    ${CMAKE_CURRENT_SOURCE_DIR}/python/test_vksplat_orthographic_regressions.py
    ${CMAKE_CURRENT_SOURCE_DIR}/python/test_vksplat_output_lifetime_regressions.py
)

set(LFS_PYTHON_NIGHTLY_TESTS
    ${CMAKE_CURRENT_SOURCE_DIR}/python/test_io_load.py
    ${CMAKE_CURRENT_SOURCE_DIR}/python/test_io_save.py
    ${CMAKE_CURRENT_SOURCE_DIR}/python/test_packages.py
    ${CMAKE_CURRENT_SOURCE_DIR}/python/test_scene.py
    ${CMAKE_CURRENT_SOURCE_DIR}/python/test_scene_validity.py
    ${CMAKE_CURRENT_SOURCE_DIR}/python/test_tensor_pytorch_interop.py
    ${CMAKE_CURRENT_SOURCE_DIR}/python/test_unicode_path_bindings.py
)

add_test(NAME lichtfeld.python.fast
    COMMAND ${Python_EXECUTABLE} -m pytest -q
        -m "not gpu and not slow and not integration"
        ${LFS_PYTHON_FAST_TESTS}
)
set_tests_properties(lichtfeld.python.fast PROPERTIES
    WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
    LABELS "fast"
    TIMEOUT 60
)

add_test(NAME lichtfeld.python.slow
    COMMAND ${Python_EXECUTABLE} -m pytest -q
        ${LFS_PYTHON_SLOW_TESTS}
)
set_tests_properties(lichtfeld.python.slow PROPERTIES
    WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
    LABELS "slow;gpu"
    TIMEOUT 300
)

add_test(NAME lichtfeld.python.nightly
    COMMAND ${Python_EXECUTABLE} -m pytest -q
        ${LFS_PYTHON_NIGHTLY_TESTS}
)
set_tests_properties(lichtfeld.python.nightly PROPERTIES
    WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
    LABELS "nightly;gpu"
    TIMEOUT 1800
)

add_test(NAME lichtfeld.python.gpu
    COMMAND ${Python_EXECUTABLE} -m pytest -q
        -m gpu
        ${CMAKE_CURRENT_SOURCE_DIR}/python/test_tensor.py
        ${CMAKE_CURRENT_SOURCE_DIR}/python/test_tensor_property.py
)
set_tests_properties(lichtfeld.python.gpu PROPERTIES
    WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
    LABELS "fast;gpu"
    TIMEOUT 60
)

# Custom targets for convenience
add_custom_target(run_tests
    COMMAND ${CMAKE_CTEST_COMMAND} --output-on-failure -L fast
    DEPENDS lichtfeld_tests lichtfeld_format_tests
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
    COMMENT "Running the fast developer test tier"
)

add_custom_target(run_full_tests
    COMMAND ${CMAKE_CTEST_COMMAND} --output-on-failure -LE nightly
    DEPENDS lichtfeld_tests lichtfeld_format_tests
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
    COMMENT "Running fast and slow tests (excluding nightly)"
)

foreach(TEST_TARGET lichtfeld_tests)
    add_custom_command(TARGET ${TEST_TARGET} POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E make_directory "$<TARGET_FILE_DIR:${TEST_TARGET}>/resources/viewer"
        COMMAND ${CMAKE_COMMAND} -E copy_directory_if_different
            "${PROJECT_SOURCE_DIR}/src/visualizer/gui/resources/viewer"
            "$<TARGET_FILE_DIR:${TEST_TARGET}>/resources/viewer"
    )
endforeach()

# Platform-specific settings
if(WIN32)
    # Copy Torch DLLs for testing on Windows
    file(GLOB TORCH_DLLS "${Torch_DIR}/../../../lib/*.dll")
    foreach(TEST_TARGET lichtfeld_tests)
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
            if(WITH_DYNAMIC_NVJPEG2K AND DEFINED _lfs_nvjpeg2k_runtime_file AND EXISTS "${_lfs_nvjpeg2k_runtime_file}")
                add_custom_command(TARGET ${TEST_TARGET} POST_BUILD
                    COMMAND ${CMAKE_COMMAND} -E copy_if_different
                        "${_lfs_nvjpeg2k_runtime_file}"
                        "$<TARGET_FILE_DIR:${TEST_TARGET}>"
                    COMMENT "Copying nvjpeg2k runtime library for ${TEST_TARGET}"
                )
            endif()
        endif()
    endforeach()
endif()

# Standalone tensor launch-overhead bench (Phase 6B-2 P2 §9 pin).
# Built with the test target; NOT registered with ctest; not part of any
# commit list. Depends only on the public tensor API.
add_executable(bench_tensor_launch_overhead
    bench_tensor_launch_overhead.cpp
)
set_target_properties(bench_tensor_launch_overhead PROPERTIES
    CXX_STANDARD 23
    CXX_STANDARD_REQUIRED ON
)
target_include_directories(bench_tensor_launch_overhead PRIVATE
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_BINARY_DIR}/include
    ${CMAKE_SOURCE_DIR}/src
    ${CUDAToolkit_INCLUDE_DIRS}
)
target_link_libraries(bench_tensor_launch_overhead PRIVATE
    lfs_core
    CUDA::cudart
)

# Phase 6C-P3 negative nvcc TU fixtures (Ruling 1 / §1.11) live under
# tests/cuda_fixtures/ and are intentionally NOT listed in any target SOURCES.
# Positive nvcc TU: device_fault_cuda_utils.cu (compiled into lichtfeld_tests;
# includes the FOUR allowed headers and exercises DeviceFaultRecord +
# device_fault_try_record_first). A host gtest asserts the negative fixtures
# exist and contain the forbidden includes (see test_device_fault.cpp).

# Status messages
list(LENGTH TEST_FILES TEST_COUNT)

message(STATUS "Tests configured with ${CMAKE_BUILD_TYPE} build type")
message(STATUS "  Unit tests: ${TEST_COUNT} files -> lichtfeld_tests")
message(STATUS "  Build tests: ninja lichtfeld_tests")
message(STATUS "  Run tests: ninja run_tests")
message(STATUS "  Run full tests: ninja run_full_tests")
message(STATUS "  Bench (not ctest): bench_tensor_launch_overhead")
message(STATUS "  Phase 6C device-fault tests: test_device_fault.cpp + device_fault_cuda_utils.cu")

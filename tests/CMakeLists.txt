# SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
# SPDX-License-Identifier: GPL-3.0-or-later

# Testing module - comprehensive tests for all components

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

# Regular test sources (unit tests, integration tests)
set(TEST_FILES
    test_assert_hardening.cpp
    test_cuda_error.cpp
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
    test_sparsity_optimizer.cpp
    test_mcmc_tensor_ops.cpp
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
    test_checkpoint_resume.cpp
    test_atomic_output.cpp
    test_training_terminal_state.cpp
    test_headless_run_coordinator.cpp
    test_argument_parser.cpp
    test_crash_handler.cpp
    test_logger.cpp
    test_parameter_manager.cpp
    test_mcp_protocol.cpp
    test_mcp_app_utils.cpp
    test_mcp_shared_scene_tools.cpp
    test_mcp_sequencer_tools.cpp
    test_tcp_publisher.cpp
    test_view_info_json.cpp
    test_video_export_utils.cpp
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
    test_image_kernels.cpp
    test_add_new_gs_tensor_ops.cpp
    test_cpu_dtype_conversions.cpp
    test_gradient_accumulation.cpp
    test_image_resize_parallel.cpp
    test_tensor_slicing_safety.cpp
    test_tensor_lazy_ir.cpp
    test_tensor_lazy_stateful_ops.cpp
    test_fastgs_kernels.cpp
    test_fastgs_fuzz.cpp
    test_sh_swizzle_layout.cpp
    test_python_io.cpp
    test_extended_unary_ops_vs_torch.cpp
    test_scene_validity.cpp
    test_render_settings_defaults.cpp
    test_selection_rasterization_ops.cpp
    test_rendering_refactor_services.cpp
    test_theme_registry.cpp
    test_viewport.cpp
    test_input_controller_focus.cpp
    test_input_controller_dataset_load.cpp
    test_selection_service_interactions.cpp
    test_undo_history.cpp
    test_selection_command_dispatch.cpp
    test_selection_operator_modal.cpp
    test_operator_registry_props.cpp
    test_fastgs_analytical_gradients.cpp
    test_normal_prior_decode.cpp
    test_background_image.cpp
    test_nvcodec_jpeg2k_16bit.cpp
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
    test_ppisp_controller.cpp
    test_ppisp_regularization.cpp
    test_ppisp_sidecar.cpp
    test_ppisp_cuda_vs_torch.cpp
    test_ppisp_banded_region.cpp
    test_export_env_composite.cpp
    test_undistort.cpp
    test_viewport_region_utils.cpp
    test_mesh_data.cpp
    test_openmesh_bridge.cpp
    test_mesh_loader.cpp
    test_colmap_image_layout.cpp
    test_colmap_points3d_text.cpp
    test_colmap_binary_error_taxonomy.cpp
    test_splat_simplify.cpp
    test_rml_text_input_handler.cpp
    test_rotated_sh_correctness.cpp
    test_tensor_stream.cpp
    test_cuda_event_pool.cpp
    test_tensor_multistream.cpp
    test_tensor_discovery.cpp
    test_tensor_discovery_fuzz.cpp
    test_mcmc_multinomial.cpp
    test_arena_metrics_contention.cpp
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
# Phase 7A: bounded-wait machine + SubmissionState + QW-6 (GPU-free).
list(APPEND TEST_FILES test_vulkan_wait_bounded.cpp)
list(APPEND TEST_FILES test_submission_state_transitions.cpp)
list(APPEND TEST_FILES test_vksplat_failed_submit_no_publish.cpp)
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

# PyModalRegistry data layer (no ImGui) for lock-order regression tests.
list(APPEND TEST_SOURCES
    ${CMAKE_SOURCE_DIR}/src/python/lfs/py_modal_registry.cpp
    ${CMAKE_SOURCE_DIR}/src/app/mcp_sequencer_tools.cpp
    ${CMAKE_SOURCE_DIR}/src/app/converter.cpp
)

# Create test executable
add_executable(lichtfeld_tests ${TEST_SOURCES})

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
    imgui::imgui
    implot::implot
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
)

target_link_libraries(lichtfeld_tests PRIVATE lfs_vulkan_rasterizer Vulkan::Vulkan)

# Compiler options for CUDA debugging
target_compile_options(lichtfeld_tests PRIVATE
    $<$<AND:$<COMPILE_LANGUAGE:CUDA>,$<CONFIG:Debug>>:-lineinfo>
    $<$<AND:$<COMPILE_LANGUAGE:CXX>,$<CONFIG:Debug>>:-O0 -g -fno-omit-frame-pointer>
    $<$<AND:$<COMPILE_LANGUAGE:CXX>,$<CONFIG:Release>>:-O3 -DNDEBUG>
)

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

target_compile_options(tensor_hardening_tests PRIVATE
    $<$<AND:$<COMPILE_LANGUAGE:CXX>,$<CONFIG:Debug>>:-O0 -g -fno-omit-frame-pointer>
    $<$<AND:$<COMPILE_LANGUAGE:CXX>,$<CONFIG:Release>>:-O3 -DNDEBUG>
)

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
set_tests_properties(lichtfeld.discovery PROPERTIES
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
    LABELS "discovery;gpu"
    TIMEOUT 1800
)

# Keep every Python tier explicit so new modules cannot silently drift out of
# the curated suite.
set(LFS_PYTHON_FAST_TESTS
    ${CMAKE_CURRENT_SOURCE_DIR}/python/test_event_keypad_mapping.py
    ${CMAKE_CURRENT_SOURCE_DIR}/python/test_export_panel.py
    ${CMAKE_CURRENT_SOURCE_DIR}/python/test_gizmo_context.py
    ${CMAKE_CURRENT_SOURCE_DIR}/python/test_histogram_panel.py
    ${CMAKE_CURRENT_SOURCE_DIR}/python/test_http.py
    ${CMAKE_CURRENT_SOURCE_DIR}/python/test_import_dialog_panels.py
    ${CMAKE_CURRENT_SOURCE_DIR}/python/test_input_settings_panel.py
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
    DEPENDS lichtfeld_tests
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
    COMMENT "Running the fast developer test tier"
)

add_custom_target(run_full_tests
    COMMAND ${CMAKE_CTEST_COMMAND} --output-on-failure -LE nightly
    DEPENDS lichtfeld_tests
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
    COMMENT "Running fast and slow tests (excluding nightly)"
)

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

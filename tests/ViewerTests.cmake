# SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
# SPDX-License-Identifier: GPL-3.0-or-later

# Keep a native viewer suite available without linking the excluded trainer.
add_executable(lichtfeld_viewer_tests
    test_main.cpp
    test_training_disabled.cpp
    test_headless_vulkan_device_selection.cpp
    test_viewer_no_cuda.cpp
    test_tensor_completion.cpp
    test_tensor_point_ops.cpp
    test_tensor_expression.cpp
    test_tensor_expression_cuda.cu
    test_splat_codec_edits.cpp
    test_tensor_splat_affine.cpp
    test_tensor_sh_codec.cpp
    test_tensor_image_ops.cpp
    test_viewer_appearance.cpp
    test_export_env_composite.cpp
    test_export_band_pack.cpp
    test_selection_tensor_projection.cpp
    test_selection_screen_ops.cpp
    test_selection_group_mask.cpp
    test_selection_mask_ops.cpp
    test_selection_rasterization_ops.cpp
    test_selection_service_interactions.cpp
    test_selection_command_dispatch.cpp
    test_selection_operator_modal.cpp
    test_scene_graph_identity.cpp
    test_scene_combining_backends.cpp
    test_scene_snapshot_backend.cpp
    test_splat_sh_tensor_layout.cpp
    test_sh_quant_tensor_program.cpp
    test_splat_affine_transform.cpp
    test_splat_sh_affine.cpp
)
target_compile_features(lichtfeld_viewer_tests PRIVATE cxx_std_23)
target_include_directories(lichtfeld_viewer_tests PRIVATE
    ${CMAKE_SOURCE_DIR}/src
    ${CMAKE_SOURCE_DIR}/src/visualizer
    ${CMAKE_SOURCE_DIR}/src/app/include
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_BINARY_DIR}/include
)
target_compile_definitions(lichtfeld_viewer_tests PRIVATE
    PROJECT_ROOT_PATH="${CMAKE_SOURCE_DIR}"
    TEST_DATA_DIR="${CMAKE_SOURCE_DIR}/data"
)
get_target_property(_lfs_viewer_test_sources lichtfeld_viewer_tests SOURCES)
lfs_tensor_private_access(${_lfs_viewer_test_sources})
target_link_libraries(lichtfeld_viewer_tests PRIVATE
    ${LFS_APP_INTERNAL_LINK_LIBS}
    GTest::gtest
    CUDA::cudart
)
add_test(NAME lichtfeld_viewer_tests COMMAND lichtfeld_viewer_tests)
set_tests_properties(lichtfeld_viewer_tests PROPERTIES TIMEOUT 120)

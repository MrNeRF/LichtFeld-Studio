# SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
# SPDX-License-Identifier: GPL-3.0-or-later

include(${CMAKE_SOURCE_DIR}/cmake/CompileShaders.cmake)

add_library(lfs_onnx_vulkan STATIC
    execution_plan.cpp
    graph_validation.cpp
    onnx_parser.cpp
    operator_registry.cpp
    session.cpp
    tensor.cpp
    vulkan_runtime.cpp
    weights.cpp
    wire_reader.cpp
)
add_library(lfs::onnx_vulkan ALIAS lfs_onnx_vulkan)

target_include_directories(lfs_onnx_vulkan
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${CMAKE_CURRENT_BINARY_DIR}/generated
)

target_compile_features(lfs_onnx_vulkan PUBLIC cxx_std_23)
target_link_libraries(lfs_onnx_vulkan PRIVATE Vulkan::Vulkan)
set_target_properties(lfs_onnx_vulkan PROPERTIES
    CXX_STANDARD 23
    CXX_STANDARD_REQUIRED ON
    POSITION_INDEPENDENT_CODE ON
)

set(LFS_ONNX_SHADER_DIR "${CMAKE_CURRENT_SOURCE_DIR}/shaders")
set(LFS_ONNX_SHADER_OUTPUT_DIR "${CMAKE_CURRENT_BINARY_DIR}/generated")
compile_shader(lfs_onnx_vulkan "${LFS_ONNX_SHADER_DIR}/elementwise.comp" "${LFS_ONNX_SHADER_OUTPUT_DIR}/elementwise.comp.spv.h" kElementwiseSpv lfs::onnx_vulkan::shaders)
compile_shader(lfs_onnx_vulkan "${LFS_ONNX_SHADER_DIR}/transform.comp" "${LFS_ONNX_SHADER_OUTPUT_DIR}/transform.comp.spv.h" kTransformSpv lfs::onnx_vulkan::shaders)
compile_shader(lfs_onnx_vulkan "${LFS_ONNX_SHADER_DIR}/matmul.comp" "${LFS_ONNX_SHADER_OUTPUT_DIR}/matmul.comp.spv.h" kMatMulSpv lfs::onnx_vulkan::shaders)
compile_shader(lfs_onnx_vulkan "${LFS_ONNX_SHADER_DIR}/matmul_tiled.comp" "${LFS_ONNX_SHADER_OUTPUT_DIR}/matmul_tiled.comp.spv.h" kMatMulTiledSpv lfs::onnx_vulkan::shaders)
compile_shader(lfs_onnx_vulkan "${LFS_ONNX_SHADER_DIR}/matmul_small_k.comp" "${LFS_ONNX_SHADER_OUTPUT_DIR}/matmul_small_k.comp.spv.h" kMatMulSmallKSpv lfs::onnx_vulkan::shaders)
compile_shader(lfs_onnx_vulkan "${LFS_ONNX_SHADER_DIR}/matmul_large_n.comp" "${LFS_ONNX_SHADER_OUTPUT_DIR}/matmul_large_n.comp.spv.h" kMatMulLargeNSpv lfs::onnx_vulkan::shaders)
compile_shader(lfs_onnx_vulkan "${LFS_ONNX_SHADER_DIR}/matmul_128.comp" "${LFS_ONNX_SHADER_OUTPUT_DIR}/matmul_128.comp.spv.h" kMatMul128Spv lfs::onnx_vulkan::shaders)
compile_shader(lfs_onnx_vulkan "${LFS_ONNX_SHADER_DIR}/matmul_packed_b.comp" "${LFS_ONNX_SHADER_OUTPUT_DIR}/matmul_packed_b.comp.spv.h" kMatMulPackedBSpv lfs::onnx_vulkan::shaders)
compile_shader(lfs_onnx_vulkan "${LFS_ONNX_SHADER_DIR}/conv.comp" "${LFS_ONNX_SHADER_OUTPUT_DIR}/conv.comp.spv.h" kConvSpv lfs::onnx_vulkan::shaders)
compile_shader(lfs_onnx_vulkan "${LFS_ONNX_SHADER_DIR}/conv_1x1.comp" "${LFS_ONNX_SHADER_OUTPUT_DIR}/conv_1x1.comp.spv.h" kConv1x1Spv lfs::onnx_vulkan::shaders)
compile_shader(lfs_onnx_vulkan "${LFS_ONNX_SHADER_DIR}/conv_tiled.comp" "${LFS_ONNX_SHADER_OUTPUT_DIR}/conv_tiled.comp.spv.h" kConvTiledSpv lfs::onnx_vulkan::shaders)
compile_shader(lfs_onnx_vulkan "${LFS_ONNX_SHADER_DIR}/conv_3x3_edge.comp" "${LFS_ONNX_SHADER_OUTPUT_DIR}/conv_3x3_edge.comp.spv.h" kConv3x3EdgeSpv lfs::onnx_vulkan::shaders)
compile_shader(lfs_onnx_vulkan "${LFS_ONNX_SHADER_DIR}/conv_3x3_edge_32.comp" "${LFS_ONNX_SHADER_OUTPUT_DIR}/conv_3x3_edge_32.comp.spv.h" kConv3x3Edge32Spv lfs::onnx_vulkan::shaders)
compile_shader(lfs_onnx_vulkan "${LFS_ONNX_SHADER_DIR}/conv_transpose.comp" "${LFS_ONNX_SHADER_OUTPUT_DIR}/conv_transpose.comp.spv.h" kConvTransposeSpv lfs::onnx_vulkan::shaders)
compile_shader(lfs_onnx_vulkan "${LFS_ONNX_SHADER_DIR}/conv_transpose_tiled.comp" "${LFS_ONNX_SHADER_OUTPUT_DIR}/conv_transpose_tiled.comp.spv.h" kConvTransposeTiledSpv lfs::onnx_vulkan::shaders)
compile_shader(lfs_onnx_vulkan "${LFS_ONNX_SHADER_DIR}/conv_transpose_2x2.comp" "${LFS_ONNX_SHADER_OUTPUT_DIR}/conv_transpose_2x2.comp.spv.h" kConvTranspose2x2Spv lfs::onnx_vulkan::shaders)
compile_shader(lfs_onnx_vulkan "${LFS_ONNX_SHADER_DIR}/qkv_pack.comp" "${LFS_ONNX_SHADER_OUTPUT_DIR}/qkv_pack.comp.spv.h" kQkvPackSpv lfs::onnx_vulkan::shaders)
compile_shader(lfs_onnx_vulkan "${LFS_ONNX_SHADER_DIR}/attention.comp" "${LFS_ONNX_SHADER_OUTPUT_DIR}/attention.comp.spv.h" kAttentionSpv lfs::onnx_vulkan::shaders)
compile_shader(lfs_onnx_vulkan "${LFS_ONNX_SHADER_DIR}/reduce.comp" "${LFS_ONNX_SHADER_OUTPUT_DIR}/reduce.comp.spv.h" kReduceSpv lfs::onnx_vulkan::shaders)
compile_shader(lfs_onnx_vulkan "${LFS_ONNX_SHADER_DIR}/reduce_serial.comp" "${LFS_ONNX_SHADER_OUTPUT_DIR}/reduce_serial.comp.spv.h" kReduceSerialSpv lfs::onnx_vulkan::shaders)
compile_shader(lfs_onnx_vulkan "${LFS_ONNX_SHADER_DIR}/softmax.comp" "${LFS_ONNX_SHADER_OUTPUT_DIR}/softmax.comp.spv.h" kSoftmaxSpv lfs::onnx_vulkan::shaders)
compile_shader(lfs_onnx_vulkan "${LFS_ONNX_SHADER_DIR}/layer_norm.comp" "${LFS_ONNX_SHADER_OUTPUT_DIR}/layer_norm.comp.spv.h" kLayerNormSpv lfs::onnx_vulkan::shaders)

add_executable(lfs_onnx_inspect EXCLUDE_FROM_ALL tools/onnx_inspect.cpp)
target_compile_features(lfs_onnx_inspect PRIVATE cxx_std_23)
target_include_directories(lfs_onnx_inspect PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(lfs_onnx_inspect PRIVATE lfs_onnx_vulkan)

add_executable(lfs_onnx_smoke EXCLUDE_FROM_ALL tools/onnx_smoke.cpp)
target_compile_features(lfs_onnx_smoke PRIVATE cxx_std_23)
target_link_libraries(lfs_onnx_smoke PRIVATE lfs_onnx_vulkan)

add_executable(lfs_onnx_tests EXCLUDE_FROM_ALL tests/onnx_tests.cpp)
target_compile_features(lfs_onnx_tests PRIVATE cxx_std_23)
target_include_directories(lfs_onnx_tests PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(lfs_onnx_tests PRIVATE lfs_onnx_vulkan)
if(BUILD_TESTS)
    add_test(NAME onnx_vulkan COMMAND lfs_onnx_tests)
    set_tests_properties(onnx_vulkan PROPERTIES LABELS "fast;vulkan" TIMEOUT 60)
endif()

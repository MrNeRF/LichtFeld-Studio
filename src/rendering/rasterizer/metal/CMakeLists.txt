# SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
#
# SPDX-License-Identifier: GPL-3.0-or-later
if(NOT APPLE)
    return()
endif()
enable_language(OBJCXX)
file(READ "${CMAKE_CURRENT_SOURCE_DIR}/splat_preprocessor.metal" LFS_METAL_PREPROCESS_SOURCE)
file(READ "${CMAKE_CURRENT_SOURCE_DIR}/viewer_overlay.metal" LFS_METAL_OVERLAY_SOURCE)
string(REPLACE "@LFS_METAL_OVERLAY@" "${LFS_METAL_OVERLAY_SOURCE}" LFS_METAL_PREPROCESS_SOURCE "${LFS_METAL_PREPROCESS_SOURCE}")
set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS
    "${CMAKE_CURRENT_SOURCE_DIR}/viewer_overlay.metal")
set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS
    "${CMAKE_CURRENT_SOURCE_DIR}/splat_preprocessor.metal")
configure_file(shader_source.hpp.in generated/shader_source.hpp @ONLY)
file(READ "${CMAKE_CURRENT_SOURCE_DIR}/tile_rasterizer.metal" LFS_METAL_TILE_SOURCE)
string(REPLACE "@LFS_METAL_OVERLAY@" "${LFS_METAL_OVERLAY_SOURCE}" LFS_METAL_TILE_SOURCE "${LFS_METAL_TILE_SOURCE}")
set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS
    "${CMAKE_CURRENT_SOURCE_DIR}/tile_rasterizer.metal")
configure_file(tile_shader_source.hpp.in generated/tile_shader_source.hpp @ONLY)
add_library(lfs_metal_viewer STATIC splat_preprocessor.mm tile_rasterizer.mm)
target_include_directories(lfs_metal_viewer PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}"
    PRIVATE "${CMAKE_CURRENT_BINARY_DIR}/generated"
    "${CMAKE_CURRENT_SOURCE_DIR}/../../../core/include")
target_compile_features(lfs_metal_viewer PUBLIC cxx_std_20)
set_target_properties(lfs_metal_viewer PROPERTIES OBJCXX_STANDARD 20 OBJCXX_STANDARD_REQUIRED YES)
target_compile_options(lfs_metal_viewer PRIVATE -fobjc-arc)
target_compile_definitions(lfs_metal_viewer PRIVATE LFS_HAS_CUDA=0)
target_link_libraries(lfs_metal_viewer PUBLIC "-framework Metal" "-framework Foundation")

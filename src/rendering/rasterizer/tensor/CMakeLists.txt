# SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
# SPDX-License-Identifier: GPL-3.0-or-later

# Single-source splat rasterizer: Slang passes run through the tensor library
# on Metal and Vulkan. It replaces the native Metal and vksplat rasterizers
# pass by pass.
add_library(lfs_tensor_rasterizer STATIC splat_tile_binner.cpp)
target_compile_features(lfs_tensor_rasterizer PUBLIC cxx_std_23)
set_target_properties(lfs_tensor_rasterizer PROPERTIES POSITION_INDEPENDENT_CODE ON)
target_include_directories(lfs_tensor_rasterizer PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}")
target_link_libraries(lfs_tensor_rasterizer PUBLIC lfs_core)
lfs_add_gpu_program(lfs_tensor_rasterizer splat_scan SOURCE splat_scan.slang COMPUTE scan_blocks scan_add)
lfs_add_gpu_program(lfs_tensor_rasterizer splat_sort SOURCE splat_sort.slang
    COMPUTE sort_histogram sort_digit_scan sort_digit_offsets sort_scatter)
lfs_add_gpu_program(lfs_tensor_rasterizer splat_tiles SOURCE splat_tiles.slang
    COMPUTE tile_counts tile_status tile_instances tile_ranges)
set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/splat_types.slang")

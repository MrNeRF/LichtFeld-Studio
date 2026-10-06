# SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
# SPDX-License-Identifier: GPL-3.0-or-later

# Single-source splat rasterizer: Slang passes run through the tensor library
# on Metal and Vulkan. It replaces the native Metal and vksplat rasterizers
# pass by pass.
add_library(lfs_tensor_rasterizer STATIC splat_projector.cpp splat_tile_binner.cpp splat_rasterizer.cpp splat_lod_selector.cpp splat_selection_query.cpp point_cloud_renderer.cpp tensor_scene_temporal.cpp tensor_scene_texture.cpp)
target_compile_features(lfs_tensor_rasterizer PUBLIC cxx_std_23)
set_target_properties(lfs_tensor_rasterizer PROPERTIES POSITION_INDEPENDENT_CODE ON)
target_include_directories(lfs_tensor_rasterizer PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}")
target_link_libraries(lfs_tensor_rasterizer PUBLIC lfs_core)
# The viewer's near clip is the Vulkan reference's, read from the shared config.
file(READ "${CMAKE_CURRENT_SOURCE_DIR}/../slang/config.slang" _viewer_config)
string(REGEX MATCH "#define NEAR_CLIP ([0-9.]+)f" _near_clip "${_viewer_config}")
if(NOT _near_clip)
    message(FATAL_ERROR "Missing NEAR_CLIP in the shared viewer config")
endif()
target_compile_definitions(lfs_tensor_rasterizer PUBLIC LFS_SPLAT_NEAR_CLIP=${CMAKE_MATCH_1}f)
lfs_add_gpu_program(lfs_tensor_rasterizer splat_scan SOURCE splat_scan.slang COMPUTE scan_blocks scan_add)
lfs_add_gpu_program(lfs_tensor_rasterizer splat_sort SOURCE splat_sort.slang
    COMPUTE sort_histogram sort_digit_scan sort_digit_offsets sort_scatter)
lfs_add_gpu_program(lfs_tensor_rasterizer splat_sort32 SOURCE splat_sort.slang
    COMPUTE sort_histogram sort_digit_scan sort_digit_offsets sort_scatter DEFINES LFS_SORT_KEY32=1)
lfs_add_gpu_program(lfs_tensor_rasterizer splat_tiles SOURCE splat_tiles.slang
    COMPUTE tile_counts tile_status source_key_groups source_compact source_permutation tile_instances tile_ranges tile_depth_batches)
lfs_add_gpu_program(lfs_tensor_rasterizer splat_project SOURCE splat_project.slang COMPUTE project_splats)
lfs_add_gpu_program(lfs_tensor_rasterizer splat_lod SOURCE splat_lod.slang COMPUTE select_lod)
lfs_add_gpu_program(lfs_tensor_rasterizer splat_selection SOURCE splat_selection.slang COMPUTE polygon_coverage selection_query)
# Blend variants. The plain opaque view is specialized at compile time with
# relaxed math, as the native rasterizer does: serial (128), and depth-batch
# chunks (128|512) and prefix (128|512|1024). Generic blends per raster mode and
# group width read their feature flags per dispatch.
set(_blend_entries tile_blend tile_depth_compose)
lfs_add_gpu_program(lfs_tensor_rasterizer splat_blend_gs_fast SOURCE splat_blend.slang COMPUTE ${_blend_entries}
    DEFINES LFS_BLEND_MODE=0 LFS_BLEND_FLAGS=128 RELAXED_MATH)
lfs_add_gpu_program(lfs_tensor_rasterizer splat_blend_gs_batch SOURCE splat_blend.slang COMPUTE ${_blend_entries}
    DEFINES LFS_BLEND_MODE=0 LFS_BLEND_FLAGS=640 RELAXED_MATH)
lfs_add_gpu_program(lfs_tensor_rasterizer splat_blend_gs_prefix SOURCE splat_blend.slang COMPUTE ${_blend_entries}
    DEFINES LFS_BLEND_MODE=0 LFS_BLEND_FLAGS=1664 RELAXED_MATH)
# Depth display keeps precise math, but otherwise has the same hot-path flag
# stability as the opaque view. Specialize its serial, chunk and prefix passes
# so Slang can discard every overlay/portal/Spark/GUT branch before emitting MSL.
lfs_add_gpu_program(lfs_tensor_rasterizer splat_blend_gs_depth SOURCE splat_blend.slang COMPUTE ${_blend_entries}
    DEFINES LFS_BLEND_MODE=0 LFS_BLEND_FLAGS=2176)
lfs_add_gpu_program(lfs_tensor_rasterizer splat_blend_gs_depth_batch SOURCE splat_blend.slang COMPUTE ${_blend_entries}
    DEFINES LFS_BLEND_MODE=0 LFS_BLEND_FLAGS=2688)
lfs_add_gpu_program(lfs_tensor_rasterizer splat_blend_gs_depth_prefix SOURCE splat_blend.slang COMPUTE ${_blend_entries}
    DEFINES LFS_BLEND_MODE=0 LFS_BLEND_FLAGS=3712)
lfs_add_gpu_program(lfs_tensor_rasterizer splat_blend_gs32 SOURCE splat_blend.slang COMPUTE ${_blend_entries}
    DEFINES LFS_BLEND_MODE=0 LFS_BLEND_SINGLE_SIMD=1)
lfs_add_gpu_program(lfs_tensor_rasterizer splat_blend_gs64 SOURCE splat_blend.slang COMPUTE ${_blend_entries} DEFINES LFS_BLEND_MODE=0)
lfs_add_gpu_program(lfs_tensor_rasterizer splat_blend_points SOURCE splat_blend.slang COMPUTE ${_blend_entries} DEFINES LFS_BLEND_MODE=1)
lfs_add_gpu_program(lfs_tensor_rasterizer splat_blend_discs SOURCE splat_blend.slang COMPUTE ${_blend_entries} DEFINES LFS_BLEND_MODE=2)
lfs_add_gpu_program(lfs_tensor_rasterizer splat_blend_gut32 SOURCE splat_blend.slang COMPUTE ${_blend_entries}
    DEFINES LFS_BLEND_MODE=3 LFS_BLEND_SINGLE_SIMD=1)
lfs_add_gpu_program(lfs_tensor_rasterizer splat_blend_gut64 SOURCE splat_blend.slang COMPUTE ${_blend_entries} DEFINES LFS_BLEND_MODE=3)
# The ordinary GUT view always uses the legacy saturation convention (32),
# with the rasterizer selecting 64 or 32 threads from the previous frame's
# density. Exact variants recover the native function-constant specialization.
lfs_add_gpu_program(lfs_tensor_rasterizer splat_blend_gut_plain32 SOURCE splat_blend.slang COMPUTE ${_blend_entries}
    DEFINES LFS_BLEND_MODE=3 LFS_BLEND_FLAGS=160 RELAXED_MATH)
lfs_add_gpu_program(lfs_tensor_rasterizer splat_blend_gut_plain64 SOURCE splat_blend.slang COMPUTE ${_blend_entries}
    DEFINES LFS_BLEND_MODE=3 LFS_BLEND_FLAGS=32 RELAXED_MATH)
lfs_add_gpu_program(lfs_tensor_rasterizer splat_blend_gut_batch SOURCE splat_blend.slang COMPUTE ${_blend_entries}
    DEFINES LFS_BLEND_MODE=3 LFS_BLEND_FLAGS=672 RELAXED_MATH)
lfs_add_gpu_program(lfs_tensor_rasterizer splat_blend_gut_prefix SOURCE splat_blend.slang COMPUTE ${_blend_entries}
    DEFINES LFS_BLEND_MODE=3 LFS_BLEND_FLAGS=1696 RELAXED_MATH)
lfs_add_gpu_program(lfs_tensor_rasterizer splat_present SOURCE splat_present.slang COMPUTE present_viewer)
lfs_add_gpu_program(lfs_tensor_rasterizer point_cloud SOURCE point_cloud.slang
    COMPUTE extractPointDepth VERTEX pointVertex FRAGMENT pointColorFragment pointDepthFragment)
lfs_add_gpu_program(lfs_tensor_rasterizer tensor_scene_temporal_program SOURCE tensor_scene_temporal.slang
    COMPUTE scene_motion scene_spatial scene_temporal_resolve scene_temporal_resolve_samples)
# Parity tests load the passes directly against the native kernels.
target_include_directories(lfs_tensor_rasterizer PUBLIC "${CMAKE_CURRENT_BINARY_DIR}/gpu_programs/splat_project")
# Shared includes of the Slang passes; the program rules depend only on their sources.
cmake_path(ABSOLUTE_PATH CMAKE_CURRENT_SOURCE_DIR NORMALIZE OUTPUT_VARIABLE _rasterizer_dir)
cmake_path(GET _rasterizer_dir PARENT_PATH _rasterizer_dir)
set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS
    "${CMAKE_CURRENT_SOURCE_DIR}/splat_types.slang" "${CMAKE_CURRENT_SOURCE_DIR}/splat_overlay.slang"
    "${_rasterizer_dir}/slang/config.slang"
    "${_rasterizer_dir}/slang/portal_compact.inc"
    "${_rasterizer_dir}/slang/display_color.inc")

lfs_add_gpu_program(lfs_tensor_rasterizer tensor_scene_texture_program SOURCE tensor_scene_texture.slang
    COMPUTE scene_texture_pack scene_texture_unpack)

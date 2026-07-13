# SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
#
# SPDX-License-Identifier: GPL-3.0-or-later

add_library(lfs_vulkan_rasterizer STATIC
        src/buffer.cpp
        src/gs_pipeline.cpp
        src/gs_renderer.cpp
        src/perf_timer.cpp
        src/vma_implementation.cpp
)

target_include_directories(lfs_vulkan_rasterizer
        PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/src
        ${CMAKE_CURRENT_SOURCE_DIR}
        $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/src>
        $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/src/core/include>
)

target_link_libraries(lfs_vulkan_rasterizer
        PUBLIC
        Vulkan::Vulkan
        GPUOpen::VulkanMemoryAllocator
        PRIVATE
        lfs_diagnostics
)

target_compile_features(lfs_vulkan_rasterizer PUBLIC cxx_std_23)
set_target_properties(lfs_vulkan_rasterizer PROPERTIES
        POSITION_INDEPENDENT_CODE ON
)

target_compile_definitions(lfs_vulkan_rasterizer PRIVATE
        $<$<PLATFORM_ID:Windows>:NOMINMAX>
)

set(LFS_VULKAN_RASTERIZER_SPV_DIR "${CMAKE_CURRENT_BINARY_DIR}/shader")
set_property(TARGET lfs_vulkan_rasterizer PROPERTY
        LFS_VULKAN_RASTERIZER_SPV_DIR "${LFS_VULKAN_RASTERIZER_SPV_DIR}")
target_compile_definitions(lfs_vulkan_rasterizer PUBLIC
        LFS_VULKAN_RASTERIZER_DEV_SPV_DIR="${LFS_VULKAN_RASTERIZER_SPV_DIR}/"
)

# ---------------------------------------------------------------------------
# Shader compilation (Slang + GLSL → SPIR-V)
# ---------------------------------------------------------------------------

find_program(SLANGC
        NAMES slangc
        HINTS
                ${CMAKE_PREFIX_PATH}
                ${CMAKE_PREFIX_PATH}/tools/shader-slang
                ${VCPKG_INSTALLED_DIR}/${VCPKG_TARGET_TRIPLET}/tools/shader-slang
        PATH_SUFFIXES tools/shader-slang
        REQUIRED
)
find_program(GLSLANG
        NAMES glslang glslangValidator
        HINTS
                ${CMAKE_PREFIX_PATH}
                ${CMAKE_PREFIX_PATH}/tools/glslang
                ${VCPKG_INSTALLED_DIR}/${VCPKG_TARGET_TRIPLET}/tools/glslang
        PATH_SUFFIXES tools/glslang
        REQUIRED
)
message(STATUS "vulkan: slangc=${SLANGC}")
message(STATUS "vulkan: glslang=${GLSLANG}")

set(_VULKAN_SHADER_SRC ${CMAKE_CURRENT_SOURCE_DIR}/shader/src)
set(_VULKAN_GENERATED_OUT ${LFS_VULKAN_RASTERIZER_SPV_DIR}/generated)
set(_VULKAN_RADIX_OUT ${LFS_VULKAN_RASTERIZER_SPV_DIR}/radix_sort)

file(MAKE_DIRECTORY ${_VULKAN_GENERATED_OUT})
file(MAKE_DIRECTORY ${_VULKAN_RADIX_OUT})

file(GLOB _VULKAN_SLANG_ALL
        ${_VULKAN_SHADER_SRC}/slang/*.slang
        ${_VULKAN_SHADER_SRC}/slang/*.inc
)
set(_VULKAN_RADIX_DEPS
        ${_VULKAN_SHADER_SRC}/radix_sort/config.glsl
)

set(_VULKAN_SLANG_FLAGS
        -stage compute
        -O
        -fp-mode fast
        -line-directive-mode none
        -target spirv
)
set(_VULKAN_GLSL_FLAGS
        -V
        --target-env vulkan1.2
)
if(LFS_ENABLE_VULKAN_SHADER_DEBUG_INFO)
        list(APPEND _VULKAN_SLANG_FLAGS -g3)
        list(APPEND _VULKAN_GLSL_FLAGS -gVS)
endif()

set(VULKAN_RASTERIZER_SHADER_OUTPUTS "")

function(_vulkan_add_slang_shader OUT_NAME SRC_FILE)
        set(_defs "")
        foreach(D ${ARGN})
                list(APPEND _defs "-D${D}")
        endforeach()
        set(_out ${_VULKAN_GENERATED_OUT}/${OUT_NAME}.spv)
        add_custom_command(
                OUTPUT ${_out}
                COMMAND ${SLANGC}
                        ${_VULKAN_SHADER_SRC}/slang/${SRC_FILE}
                        ${_VULKAN_SLANG_FLAGS}
                        ${_defs}
                        -I${_VULKAN_SHADER_SRC}/slang
                        -o ${_out}
                DEPENDS ${_VULKAN_SLANG_ALL}
                COMMENT "slangc ${OUT_NAME}.spv"
                VERBATIM
        )
        set(VULKAN_RASTERIZER_SHADER_OUTPUTS ${VULKAN_RASTERIZER_SHADER_OUTPUTS} ${_out} PARENT_SCOPE)
endfunction()

function(_vulkan_add_glsl_shader OUT_NAME SRC_FILE)
        set(_defs "")
        foreach(D ${ARGN})
                list(APPEND _defs "-D${D}")
        endforeach()
        set(_out ${_VULKAN_RADIX_OUT}/${OUT_NAME}.spv)
        add_custom_command(
                OUTPUT ${_out}
                COMMAND ${GLSLANG}
                        ${_VULKAN_GLSL_FLAGS}
                        ${_defs}
                        ${_VULKAN_SHADER_SRC}/radix_sort/${SRC_FILE}
                        -o ${_out}
                DEPENDS ${_VULKAN_SHADER_SRC}/radix_sort/${SRC_FILE} ${_VULKAN_RADIX_DEPS}
                COMMENT "glslang ${OUT_NAME}.spv"
                VERBATIM
        )
        set(VULKAN_RASTERIZER_SHADER_OUTPUTS ${VULKAN_RASTERIZER_SHADER_OUTPUTS} ${_out} PARENT_SCOPE)
endfunction()

# Forward path only: trainer / backward shaders are not compiled here.

# cumsum.slang → 4 phases
_vulkan_add_slang_shader(cumsum_block_scan         cumsum.slang CUMSUM_PHASE=1)
_vulkan_add_slang_shader(cumsum_scan_block_sums    cumsum.slang CUMSUM_PHASE=2)
_vulkan_add_slang_shader(cumsum_add_block_offsets  cumsum.slang CUMSUM_PHASE=3)
_vulkan_add_slang_shader(cumsum_single_pass        cumsum.slang CUMSUM_PHASE=0)

# tile_shader.slang → 2 entries
_vulkan_add_slang_shader(generate_keys             tile_shader.slang EXPORT_MODE=0 ENTRY=1)
_vulkan_add_slang_shader(compute_tile_ranges       tile_shader.slang EXPORT_MODE=0 ENTRY=2)

# Two-stage sort helper shaders
_vulkan_add_slang_shader(seed_primitive_indices    seed_primitive_indices.slang)
_vulkan_add_slang_shader(apply_depth_ordering      apply_depth_ordering.slang)
_vulkan_add_slang_shader(visible_flags             visible_flags.slang)
_vulkan_add_slang_shader(prepare_visible_sort      prepare_visible_sort.slang)
_vulkan_add_slang_shader(prepare_tile_sort         prepare_tile_sort.slang)
_vulkan_add_slang_shader(compact_visible_primitives compact_visible_primitives.slang)
_vulkan_add_slang_shader(lod_map_indices           lod_map_indices.slang)
_vulkan_add_slang_shader(lod_select_threshold      lod_select_threshold.slang)
_vulkan_add_slang_shader(lod_compact_touch        lod_compact_touch.slang)

# vertex_shader.slang → projection_forward variants
_vulkan_add_slang_shader(projection_forward        vertex_shader.slang EXPORT_MODE=0)
_vulkan_add_slang_shader(projection_forward_3dgut  vertex_shader.slang EXPORT_MODE=0 LFS_3DGUT_PROJECTION=1)
# Canonical quantized LOD pool variants (lod_pool_quant.hpp)
_vulkan_add_slang_shader(projection_forward_quant       vertex_shader.slang EXPORT_MODE=0 LFS_QUANT_POOL=1)
_vulkan_add_slang_shader(projection_forward_quant_3dgut vertex_shader.slang EXPORT_MODE=0 LFS_3DGUT_PROJECTION=1 LFS_QUANT_POOL=1)

# selection_mask.slang → compact 1-byte-per-splat brush/rectangle/polygon masks
_vulkan_add_slang_shader(selection_mask            selection_mask.slang)

# selection_polygon_rasterize.slang → rasterizes a screen-space polygon into a
# coverage mask consumed by selection_mask in POLYGON_MASK mode.
_vulkan_add_slang_shader(selection_polygon_rasterize selection_polygon_rasterize.slang)

# alphablend_shader.slang → forward variants. The *_plain variants strip the
# editor overlay/selection path from the per-pixel inner loop; the host selects
# them whenever no overlay is active (the common viewing case).
_vulkan_add_slang_shader(rasterize_forward               alphablend_shader.slang EXPORT_MODE=0)
_vulkan_add_slang_shader(rasterize_forward_3dgut         alphablend_shader.slang EXPORT_MODE=0 LFS_3DGUT_RASTERIZE=1)
_vulkan_add_slang_shader(rasterize_forward_plain         alphablend_shader.slang EXPORT_MODE=0 LFS_RASTER_OVERLAYS=0)
_vulkan_add_slang_shader(rasterize_forward_3dgut_plain   alphablend_shader.slang EXPORT_MODE=0 LFS_3DGUT_RASTERIZE=1 LFS_RASTER_OVERLAYS=0)
_vulkan_add_slang_shader(rasterize_forward_light         alphablend_shader.slang EXPORT_MODE=0 LFS_SKIP_DENSE_TILES=1)
_vulkan_add_slang_shader(rasterize_forward_light_plain   alphablend_shader.slang EXPORT_MODE=0 LFS_RASTER_OVERLAYS=0 LFS_SKIP_DENSE_TILES=1)
_vulkan_add_slang_shader(tile_batch_counts               tile_batch_shader.slang TILE_BATCH_ENTRY=1)
_vulkan_add_slang_shader(tile_batch_descriptors          tile_batch_shader.slang TILE_BATCH_ENTRY=2)
_vulkan_add_slang_shader(rasterize_forward_batches       tile_batch_shader.slang TILE_BATCH_ENTRY=3 LFS_RASTER_OVERLAYS=1)
_vulkan_add_slang_shader(rasterize_forward_batches_plain tile_batch_shader.slang TILE_BATCH_ENTRY=3)
_vulkan_add_slang_shader(compose_tile_batches            tile_batch_shader.slang TILE_BATCH_ENTRY=4 LFS_RASTER_OVERLAYS=1)
_vulkan_add_slang_shader(compose_tile_batches_plain      tile_batch_shader.slang TILE_BATCH_ENTRY=4)

# HiGS viewer chain: position-only cull prepass, survivor projection with
# compact-slot outputs, visible-bounded sort/cumsum/key generation, and
# compact-slot overlay raster variants.
_vulkan_add_slang_shader(cull_splats                  cull_splats.slang EXPORT_MODE=0 CULL_ENTRY=1)
_vulkan_add_slang_shader(cull_prepare                 cull_splats.slang EXPORT_MODE=0 CULL_ENTRY=2)
_vulkan_add_slang_shader(projection_forward_survivors vertex_shader.slang EXPORT_MODE=0 LFS_PROJECTION_SURVIVORS=1)
_vulkan_add_slang_shader(projection_forward_quant_survivors vertex_shader.slang EXPORT_MODE=0 LFS_PROJECTION_SURVIVORS=1 LFS_QUANT_POOL=1)
_vulkan_add_slang_shader(prepare_visible_chain        prepare_visible_chain.slang)
_vulkan_add_slang_shader(copy_visible_indices         copy_visible_indices.slang)
_vulkan_add_slang_shader(cumsum_block_scan_indirect        cumsum.slang CUMSUM_PHASE=1 CUMSUM_INDIRECT=1)
_vulkan_add_slang_shader(cumsum_scan_block_sums_indirect   cumsum.slang CUMSUM_PHASE=2 CUMSUM_INDIRECT=1)
_vulkan_add_slang_shader(cumsum_add_block_offsets_indirect cumsum.slang CUMSUM_PHASE=3 CUMSUM_INDIRECT=1)
_vulkan_add_slang_shader(prepare_tile_sort_visible    prepare_tile_sort.slang LFS_VISIBLE_BOUNDED=1)

# HiGS macro-tile binning + raster/compose. The plain raster ships fp16
# staging; the fp32 variant doubles as validation reference and overlay path.
_vulkan_add_slang_shader(macro_coverage         macro_tile_shader.slang EXPORT_MODE=0 MACRO_ENTRY=1)
_vulkan_add_slang_shader(generate_macro_keys    macro_tile_shader.slang EXPORT_MODE=0 MACRO_ENTRY=2)
_vulkan_add_slang_shader(compute_macro_ranges   macro_tile_shader.slang EXPORT_MODE=0 MACRO_ENTRY=3)
_vulkan_add_slang_shader(macro_batch_counts     macro_tile_shader.slang EXPORT_MODE=0 MACRO_ENTRY=4)
_vulkan_add_slang_shader(macro_batch_prepare    macro_tile_shader.slang EXPORT_MODE=0 MACRO_ENTRY=5)
_vulkan_add_slang_shader(macro_raster           macro_raster.slang EXPORT_MODE=0)
_vulkan_add_slang_shader(macro_raster_fp32      macro_raster.slang EXPORT_MODE=0 LFS_HIGS_FP32=1)
_vulkan_add_slang_shader(macro_raster_overlays  macro_raster.slang EXPORT_MODE=0 LFS_HIGS_FP32=1 LFS_RASTER_OVERLAYS=1)
_vulkan_add_slang_shader(macro_compose          macro_compose.slang EXPORT_MODE=0)
_vulkan_add_slang_shader(macro_compose_overlays macro_compose.slang EXPORT_MODE=0 LFS_RASTER_OVERLAYS=1)

# Radix sort GLSL .comp
_vulkan_add_glsl_shader(upsweep_indirect    upsweep.comp)
_vulkan_add_glsl_shader(spine_indirect      spine.comp)
_vulkan_add_glsl_shader(downsweep_indirect  downsweep.comp)

add_custom_target(lfs_vulkan_rasterizer_shaders DEPENDS ${VULKAN_RASTERIZER_SHADER_OUTPUTS})
add_dependencies(lfs_vulkan_rasterizer lfs_vulkan_rasterizer_shaders)

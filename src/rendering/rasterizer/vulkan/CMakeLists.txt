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

file(GLOB _VULKAN_SLANG_ALL ${_VULKAN_SHADER_SRC}/slang/*.slang)
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
                        -V
                        --target-env vulkan1.2
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

# vertex_shader.slang → projection_forward variants
_vulkan_add_slang_shader(projection_forward        vertex_shader.slang EXPORT_MODE=0)
_vulkan_add_slang_shader(projection_forward_3dgut  vertex_shader.slang EXPORT_MODE=0 LFS_3DGUT_PROJECTION=1)

# selection_mask.slang → compact 1-byte-per-splat brush/rectangle/polygon masks
_vulkan_add_slang_shader(selection_mask            selection_mask.slang)

# selection_polygon_rasterize.slang → rasterizes a screen-space polygon into a
# coverage mask consumed by selection_mask in POLYGON_MASK mode.
_vulkan_add_slang_shader(selection_polygon_rasterize selection_polygon_rasterize.slang)

# alphablend_shader.slang → forward variants
_vulkan_add_slang_shader(rasterize_forward         alphablend_shader.slang EXPORT_MODE=0)
_vulkan_add_slang_shader(rasterize_forward_3dgut   alphablend_shader.slang EXPORT_MODE=0 LFS_3DGUT_RASTERIZE=1)

# Radix sort GLSL .comp
_vulkan_add_glsl_shader(upsweep    upsweep.comp)
_vulkan_add_glsl_shader(spine      spine.comp)
_vulkan_add_glsl_shader(downsweep  downsweep.comp)
_vulkan_add_glsl_shader(upsweep_indirect    upsweep.comp   RADIX_SORT_COUNT_BUFFER=1)
_vulkan_add_glsl_shader(spine_indirect      spine.comp     RADIX_SORT_COUNT_BUFFER=1)
_vulkan_add_glsl_shader(downsweep_indirect  downsweep.comp RADIX_SORT_COUNT_BUFFER=1)

add_custom_target(lfs_vulkan_rasterizer_shaders DEPENDS ${VULKAN_RASTERIZER_SHADER_OUTPUTS})
add_dependencies(lfs_vulkan_rasterizer lfs_vulkan_rasterizer_shaders)

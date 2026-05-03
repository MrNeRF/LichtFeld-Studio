# SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
#
# SPDX-License-Identifier: GPL-3.0-or-later

add_library(lfs_vksplat_fwd STATIC
        src/buffer.cpp
        src/gs_pipeline.cpp
        src/gs_renderer.cpp
        src/perf_timer.cpp
)

target_include_directories(lfs_vksplat_fwd
        PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/src
        ${CMAKE_CURRENT_SOURCE_DIR}
)

target_link_libraries(lfs_vksplat_fwd
        PUBLIC
        Vulkan::Vulkan
        GPUOpen::VulkanMemoryAllocator
)

target_compile_features(lfs_vksplat_fwd PUBLIC cxx_std_23)
set_target_properties(lfs_vksplat_fwd PROPERTIES
        POSITION_INDEPENDENT_CODE ON
)

target_compile_definitions(lfs_vksplat_fwd PRIVATE
        $<$<PLATFORM_ID:Windows>:NOMINMAX>
)

set(LFS_VKSPLAT_SPV_DIR "${CMAKE_CURRENT_SOURCE_DIR}/shader")
target_compile_definitions(lfs_vksplat_fwd PUBLIC
        LFS_VKSPLAT_SPV_DIR="${LFS_VKSPLAT_SPV_DIR}/"
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
message(STATUS "vksplat_fwd: slangc=${SLANGC}")
message(STATUS "vksplat_fwd: glslang=${GLSLANG}")

set(_VKS_SHADER_SRC ${CMAKE_CURRENT_SOURCE_DIR}/shader/src)
set(_VKS_GENERATED_OUT ${CMAKE_CURRENT_SOURCE_DIR}/shader/generated)
set(_VKS_RADIX_OUT ${CMAKE_CURRENT_SOURCE_DIR}/shader/radix_sort)

file(MAKE_DIRECTORY ${_VKS_GENERATED_OUT})
file(MAKE_DIRECTORY ${_VKS_RADIX_OUT})

file(GLOB _VKS_SLANG_ALL ${_VKS_SHADER_SRC}/slang/*.slang)
set(_VKS_RADIX_DEPS
        ${_VKS_SHADER_SRC}/radix_sort/config.glsl
)

set(_VKS_SLANG_FLAGS
        -stage compute
        -O
        -fp-mode fast
        -line-directive-mode none
        -target spirv
)

set(VKSPLAT_FWD_SHADER_OUTPUTS "")

function(_vks_add_slang_shader OUT_NAME SRC_FILE)
        set(_defs "")
        foreach(D ${ARGN})
                list(APPEND _defs "-D${D}")
        endforeach()
        set(_out ${_VKS_GENERATED_OUT}/${OUT_NAME}.spv)
        add_custom_command(
                OUTPUT ${_out}
                COMMAND ${SLANGC}
                        ${_VKS_SHADER_SRC}/slang/${SRC_FILE}
                        ${_VKS_SLANG_FLAGS}
                        ${_defs}
                        -I${_VKS_SHADER_SRC}/slang
                        -o ${_out}
                DEPENDS ${_VKS_SLANG_ALL}
                COMMENT "slangc ${OUT_NAME}.spv"
                VERBATIM
        )
        set(VKSPLAT_FWD_SHADER_OUTPUTS ${VKSPLAT_FWD_SHADER_OUTPUTS} ${_out} PARENT_SCOPE)
endfunction()

function(_vks_add_glsl_shader OUT_NAME SRC_FILE)
        set(_out ${_VKS_RADIX_OUT}/${OUT_NAME}.spv)
        add_custom_command(
                OUTPUT ${_out}
                COMMAND ${GLSLANG}
                        -V
                        --target-env vulkan1.2
                        ${_VKS_SHADER_SRC}/radix_sort/${SRC_FILE}
                        -o ${_out}
                DEPENDS ${_VKS_SHADER_SRC}/radix_sort/${SRC_FILE} ${_VKS_RADIX_DEPS}
                COMMENT "glslang ${OUT_NAME}.spv"
                VERBATIM
        )
        set(VKSPLAT_FWD_SHADER_OUTPUTS ${VKSPLAT_FWD_SHADER_OUTPUTS} ${_out} PARENT_SCOPE)
endfunction()

# Forward path only: trainer / backward shaders are not compiled here.

# cumsum.slang → 4 phases
_vks_add_slang_shader(cumsum_block_scan         cumsum.slang CUMSUM_PHASE=1)
_vks_add_slang_shader(cumsum_scan_block_sums    cumsum.slang CUMSUM_PHASE=2)
_vks_add_slang_shader(cumsum_add_block_offsets  cumsum.slang CUMSUM_PHASE=3)
_vks_add_slang_shader(cumsum_single_pass        cumsum.slang CUMSUM_PHASE=0)

# tile_shader.slang → 2 entries
_vks_add_slang_shader(generate_keys             tile_shader.slang EXPORT_MODE=0 ENTRY=1)
_vks_add_slang_shader(compute_tile_ranges       tile_shader.slang EXPORT_MODE=0 ENTRY=2)

# Tiny indirect-dispatch setup pass — reads cumsum tail from GPU and writes a
# VkDispatchIndirectCommand for compute_tile_ranges. Eliminates the per-frame
# CPU readback stall.
_vks_add_slang_shader(setup_dispatch_indirect   setup_dispatch_indirect.slang)

# vertex_shader.slang → projection_forward
_vks_add_slang_shader(projection_forward        vertex_shader.slang EXPORT_MODE=0)

# alphablend_shader.slang → forward only
_vks_add_slang_shader(rasterize_forward         alphablend_shader.slang EXPORT_MODE=0)

# Radix sort GLSL .comp
_vks_add_glsl_shader(upsweep    upsweep.comp)
_vks_add_glsl_shader(spine      spine.comp)
_vks_add_glsl_shader(downsweep  downsweep.comp)

add_custom_target(lfs_vksplat_fwd_shaders DEPENDS ${VKSPLAT_FWD_SHADER_OUTPUTS})
add_dependencies(lfs_vksplat_fwd lfs_vksplat_fwd_shaders)

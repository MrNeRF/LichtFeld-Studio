# SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
#
# SPDX-License-Identifier: GPL-3.0-or-later

# With the native Metal viewer, the Vulkan rasterizer is a comparison fixture,
# never a desktop backend.
if(LFS_TENSOR_METAL)
    if(NOT BUILD_TESTS)
        message(FATAL_ERROR "The macOS Vulkan reference rasterizer is test-only")
    endif()
    set(_VULKAN_RASTERIZER_TARGET lfs_vulkan_rasterizer_macos_reference)
    set(_VULKAN_REFERENCE_PROFILE ON)
else()
    set(_VULKAN_RASTERIZER_TARGET lfs_vulkan_rasterizer)
    set(_VULKAN_REFERENCE_PROFILE OFF)
endif()

add_library(${_VULKAN_RASTERIZER_TARGET} STATIC
        src/barrier_planner.cpp
        src/buffer.cpp
        src/compute_program_sets.cpp
        src/gs_pipeline.cpp
        src/gs_renderer.cpp
        src/stochastic.cpp
        src/perf_timer.cpp
        # Phase 7A: shared bounded-wait + SubmissionState + VulkanDispatch seam.
        # Lives under src/rendering/ so visualizer and tests include it without
        # depending on the visualizer module.
)

target_include_directories(${_VULKAN_RASTERIZER_TARGET}
        PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/src
        ${CMAKE_CURRENT_SOURCE_DIR}
        $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/src>
        $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/src/core/include>
)

target_link_libraries(${_VULKAN_RASTERIZER_TARGET}
        PUBLIC
        Vulkan::Vulkan
        GPUOpen::VulkanMemoryAllocator
        lfs_core
        lfs_vulkan_runtime
        PRIVATE
        lfs_diagnostics
)

target_compile_features(${_VULKAN_RASTERIZER_TARGET} PUBLIC cxx_std_23)
set_target_properties(${_VULKAN_RASTERIZER_TARGET} PROPERTIES
        POSITION_INDEPENDENT_CODE ON
)

target_compile_definitions(${_VULKAN_RASTERIZER_TARGET} PRIVATE
        $<$<PLATFORM_ID:Windows>:NOMINMAX>
)

# The SteadySplats STBN volume (1 MiB) is generated at build time instead of
# being checked in: about 10 s with optimization, deterministic, cached.
add_executable(lfs_generate_stbn tools/generate_stbn.cpp)
target_compile_features(lfs_generate_stbn PRIVATE cxx_std_20)
target_include_directories(lfs_generate_stbn PRIVATE ${CMAKE_SOURCE_DIR}/src/core/include)
target_compile_options(lfs_generate_stbn PRIVATE $<$<NOT:$<CXX_COMPILER_ID:MSVC>>:-O2>)
add_custom_command(
        OUTPUT "${CMAKE_CURRENT_BINARY_DIR}/stbn_data.inc"
        COMMAND lfs_generate_stbn "${CMAKE_CURRENT_BINARY_DIR}/stbn_data.inc"
        DEPENDS lfs_generate_stbn
        COMMENT "Generating the 128x128x64 STBN volume"
        VERBATIM)
add_custom_target(lfs_stbn_data DEPENDS "${CMAKE_CURRENT_BINARY_DIR}/stbn_data.inc")
add_dependencies(${_VULKAN_RASTERIZER_TARGET} lfs_stbn_data)
target_include_directories(${_VULKAN_RASTERIZER_TARGET} PRIVATE "${CMAKE_CURRENT_BINARY_DIR}")
# The Windows preflight compiles sources after building only lfs_shader_headers; the volume is one of their includes.
if(NOT TARGET lfs_shader_headers)
    add_custom_target(lfs_shader_headers)
endif()
add_dependencies(lfs_shader_headers lfs_stbn_data)

if(_VULKAN_REFERENCE_PROFILE)
    # Public because the reference adapters share the host-side batch layout.
    target_compile_definitions(${_VULKAN_RASTERIZER_TARGET} PUBLIC LFS_VULKAN_MACOS_REFERENCE=1)
    set(LFS_VULKAN_RASTERIZER_SPV_DIR "${CMAKE_CURRENT_BINARY_DIR}/macos-reference/shader")
else()
    set(LFS_VULKAN_RASTERIZER_SPV_DIR "${CMAKE_CURRENT_BINARY_DIR}/shader")
endif()
set_property(TARGET ${_VULKAN_RASTERIZER_TARGET} PROPERTY
        LFS_VULKAN_RASTERIZER_SPV_DIR "${LFS_VULKAN_RASTERIZER_SPV_DIR}")
target_compile_definitions(${_VULKAN_RASTERIZER_TARGET} PUBLIC
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

if(LFS_ENABLE_VULKAN_VALIDATION)
        set(_VULKAN_SPIRV_TOOLS_TRIPLET "${VCPKG_HOST_TRIPLET}")
        if(NOT _VULKAN_SPIRV_TOOLS_TRIPLET)
                set(_VULKAN_SPIRV_TOOLS_TRIPLET "${VCPKG_TARGET_TRIPLET}")
        endif()
        find_program(_VULKAN_SPIRV_VAL
                NAMES spirv-val
                HINTS
                        ${CMAKE_PREFIX_PATH}
                        ${CMAKE_PREFIX_PATH}/tools/spirv-tools
                        ${VCPKG_INSTALLED_DIR}/${_VULKAN_SPIRV_TOOLS_TRIPLET}/tools/spirv-tools
                PATH_SUFFIXES tools/spirv-tools
                NO_CACHE
                REQUIRED
        )
        message(STATUS "vulkan: spirv-val=${_VULKAN_SPIRV_VAL}")
endif()

set(_VULKAN_SHADER_SRC ${CMAKE_CURRENT_SOURCE_DIR}/shader/src)
# Includes shared with the tensor rasterizer passes.
set(_SHARED_SLANG_SRC ${CMAKE_CURRENT_SOURCE_DIR}/../slang)
set(_VULKAN_GENERATED_OUT ${LFS_VULKAN_RASTERIZER_SPV_DIR}/generated)
set(_VULKAN_RADIX_OUT ${LFS_VULKAN_RASTERIZER_SPV_DIR}/radix_sort)

file(MAKE_DIRECTORY ${_VULKAN_GENERATED_OUT})
file(MAKE_DIRECTORY ${_VULKAN_RADIX_OUT})

file(GLOB _VULKAN_SLANG_ALL
        ${_VULKAN_SHADER_SRC}/slang/*.slang
        ${_VULKAN_SHADER_SRC}/slang/*.inc
        ${_SHARED_SLANG_SRC}/*.slang
        ${_SHARED_SLANG_SRC}/*.inc
)
# The user-effect API and its hosts only build to IR; no SPIR-V recipe imports them.
list(FILTER _VULKAN_SLANG_ALL EXCLUDE REGEX "/lfs(_[a-z_]+)?\\.slang$")
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
if(_VULKAN_REFERENCE_PROFILE)
    list(APPEND _VULKAN_SLANG_FLAGS -DLFS_VULKAN_MACOS_REFERENCE=1)
endif()
set(_VULKAN_GLSL_FLAGS
        -V
        --target-env vulkan1.2
)
if(_VULKAN_REFERENCE_PROFILE)
    list(APPEND _VULKAN_GLSL_FLAGS -DLFS_VULKAN_MACOS_REFERENCE=1)
endif()
if(LFS_ENABLE_VULKAN_SHADER_DEBUG_INFO)
        list(APPEND _VULKAN_SLANG_FLAGS -g3)
        list(APPEND _VULKAN_GLSL_FLAGS -gVS)
endif()

set(VULKAN_RASTERIZER_SHADER_OUTPUTS "")

# User-effect package for the runtime compiler in src/visualizer/shaders: the
# API, the custom effect hosts and their imports as Slang IR plus manifest.json,
# staged and installed with the SPIR-V. Reflection stays in the build tree.
set(_lfs_user_shader_src ${_VULKAN_SHADER_SRC}/slang)
set(_lfs_user_shader_dir ${LFS_VULKAN_RASTERIZER_SPV_DIR}/user)
set(_lfs_user_shader_reflection ${CMAKE_CURRENT_BINARY_DIR}/user_shader_reflection)
set(_lfs_user_shader_ir_sources
        ${_lfs_user_shader_src}/reveal_noise.inc
        ${_lfs_user_shader_src}/lfs_helpers.slang
        ${_lfs_user_shader_src}/lfs.slang
        ${_lfs_user_shader_src}/lfs_reveal.slang
)
set(_lfs_user_shader_package lfs_user_shader_ir)

# One rule for host, API, import and test IR. Keep a durable copy of the
# compiler's dependency closure for manifests: generators may consume the .d.
function(lfs_add_shader_ir)
        cmake_parse_arguments(PARSE_ARGV 0 IR "REFLECT" "NAME;SOURCE" "DEFINES")
        list(TRANSFORM IR_DEFINES PREPEND "-D" OUTPUT_VARIABLE _defs)
        set(_compile ${SLANGC} ${_lfs_user_shader_src}/${IR_SOURCE} ${_defs}
                -matrix-layout-column-major -I${_lfs_user_shader_src} -I${_SHARED_SLANG_SRC})
        set(_module ${_lfs_user_shader_dir}/${IR_NAME}.slang-module)
        set(_depfile ${_lfs_user_shader_reflection}/${IR_NAME}.d)
        set(_outputs ${_module} ${_lfs_user_shader_reflection}/${IR_NAME}.deps)
        set(_reflect)
        if(IR_REFLECT)
                set(_reflection ${_lfs_user_shader_reflection}/${IR_NAME}.json)
                list(APPEND _outputs ${_reflection})
                set(_reflect COMMAND ${_compile} -target spirv -stage compute -entry main -skip-codegen
                        -reflection-json ${_reflection})
        endif()
        add_custom_command(
                OUTPUT ${_outputs}
                COMMAND ${CMAKE_COMMAND} -E make_directory ${_lfs_user_shader_dir} ${_lfs_user_shader_reflection}
                COMMAND ${_compile} -o ${_module} -depfile ${_depfile}
                COMMAND ${CMAKE_COMMAND} -E copy_if_different ${_depfile} ${_lfs_user_shader_reflection}/${IR_NAME}.deps
                ${_reflect}
                DEPENDS ${_lfs_user_shader_src}/${IR_SOURCE} ${SLANGC} ${CMAKE_CURRENT_FUNCTION_LIST_FILE}
                DEPFILE ${_depfile}
                COMMENT "slangc ${IR_NAME}.slang-module" VERBATIM)
        set_property(GLOBAL APPEND PROPERTY ${_lfs_user_shader_package}_OUTPUTS ${_outputs})
endfunction()

function(_lfs_add_custom_host HOST SRC_FILE)
        set(_defines ${ARGN} LFS_REVEAL=1 LFS_CUSTOM_EFFECT=1)
        lfs_add_shader_ir(NAME ${HOST} SOURCE ${SRC_FILE} DEFINES ${_defines} REFLECT)
        list(JOIN _defines "," _joined)
        set_property(GLOBAL APPEND PROPERTY LFS_USER_SHADER_HOSTS "${HOST}:${_joined}")
endfunction()

# Compiles one SPIR-V variant of a recipe into the shader outputs. DEPENDS
# lists inputs beyond the shared sources.
function(_vulkan_add_slang_variant OUT_NAME SRC_FILE)
        cmake_parse_arguments(PARSE_ARGV 2 _variant "" "" "DEPENDS")
        list(TRANSFORM _variant_UNPARSED_ARGUMENTS PREPEND "-D" OUTPUT_VARIABLE _defs)
        set(_out ${_VULKAN_GENERATED_OUT}/${OUT_NAME}.spv)
        set(_validation)
        if(_VULKAN_SPIRV_VAL)
                set(_validation COMMAND ${_VULKAN_SPIRV_VAL} ${_out})
        endif()
        add_custom_command(
                OUTPUT ${_out}
                COMMAND ${SLANGC} ${_VULKAN_SHADER_SRC}/slang/${SRC_FILE} ${_VULKAN_SLANG_FLAGS} ${_defs}
                        -I${_VULKAN_SHADER_SRC}/slang -I${_SHARED_SLANG_SRC} -o ${_out}
                ${_validation}
                DEPENDS ${_VULKAN_SLANG_ALL} ${_VULKAN_SPIRV_VAL} ${_variant_DEPENDS}
                COMMENT "slangc ${OUT_NAME}.spv" VERBATIM)
        set(VULKAN_RASTERIZER_SHADER_OUTPUTS ${VULKAN_RASTERIZER_SHADER_OUTPUTS} ${_out} PARENT_SCOPE)
endfunction()

# Lists a legacy recipe in slang_recipes.json, which
# tests/python/test_reveal_shader_default_runtime.py compiles from a base tree.
function(_vulkan_record_slang_recipe OUT_NAME SRC_FILE)
        list(TRANSFORM ARGN PREPEND "\"" OUTPUT_VARIABLE _defines)
        list(TRANSFORM _defines APPEND "\"")
        list(JOIN _defines ", " _defines)
        set_property(GLOBAL APPEND PROPERTY LFS_VULKAN_SLANG_RECIPES
                "{\"name\": \"${OUT_NAME}\", \"source\": \"${SRC_FILE}\", \"defines\": [${_defines}]}")
endfunction()

function(_vulkan_add_slang_shader OUT_NAME SRC_FILE)
        set(_banded "")
        set(_reveal "")
        set(_custom_name "")
        set(_custom_banded "")
        set(_custom_kind None)
        set(_chain None)
        _vulkan_record_slang_recipe(${OUT_NAME} ${SRC_FILE} ${ARGN})
        _vulkan_add_slang_variant(${OUT_NAME} ${SRC_FILE} ${ARGN})
        # Keep band-origin arithmetic out of interactive shaders. These variants
        # are loaded lazily when the viewer first exports a banded image.
        if(OUT_NAME MATCHES "^(projection_forward|rasterize_forward|macro_raster|macro_compose)"
           OR OUT_NAME STREQUAL "generate_keys_wave")
                _vulkan_record_slang_recipe(${OUT_NAME}_banded ${SRC_FILE} ${ARGN} LFS_BANDED_EXPORT_RENDER=1)
                _vulkan_add_slang_variant(${OUT_NAME}_banded ${SRC_FILE} ${ARGN} LFS_BANDED_EXPORT_RENDER=1)
                set(_banded "${OUT_NAME}_banded")
        endif()
        # Reveal shaders are selected only while a sequencer reveal is active.
        # The macOS reference profile draws no reveals and no user shaders.
        if(NOT _VULKAN_REFERENCE_PROFILE AND
           (OUT_NAME MATCHES "^projection_forward" OR OUT_NAME STREQUAL "cull_splats"
            OR OUT_NAME MATCHES "^rasterize_forward_3dgut" OR OUT_NAME MATCHES "^rasterize_outline_gut"))
                _vulkan_record_slang_recipe(${OUT_NAME}_reveal ${SRC_FILE} ${ARGN} LFS_REVEAL=1)
                _vulkan_add_slang_variant(${OUT_NAME}_reveal ${SRC_FILE} ${ARGN} LFS_REVEAL=1)
                set(_reveal "${OUT_NAME}_reveal")
                if(OUT_NAME MATCHES "^(projection_forward|rasterize_forward)")
                        set(_reveal_banded ${ARGN} LFS_REVEAL=1 LFS_BANDED_EXPORT_RENDER=1)
                        _vulkan_record_slang_recipe(${OUT_NAME}_reveal_banded ${SRC_FILE} ${_reveal_banded})
                        _vulkan_add_slang_variant(${OUT_NAME}_reveal_banded ${SRC_FILE} ${_reveal_banded})
                        set_property(GLOBAL APPEND PROPERTY LFS_SHADER_VARIANTS
                                "{\"${OUT_NAME}_reveal\", \"${OUT_NAME}_reveal_banded\", \"\", \"\", \"\", CustomKind::None, CustomChain::None},")
                endif()
                # Projection calls the effect and ships as a host; the viewport
                # bands only survivor and 3DGUT chains. Cull and the 3DGUT rasters
                # only read what it wrote, so their custom variants are SPIR-V
                # built for every effect and stay out of slang_recipes.json.
                if(OUT_NAME MATCHES "^projection_forward")
                        set(_custom_kind Host)
                        _lfs_add_custom_host(${OUT_NAME}_custom ${SRC_FILE} ${ARGN})
                        if(OUT_NAME MATCHES "_(survivors|3dgut)$")
                                _lfs_add_custom_host(${OUT_NAME}_custom_banded ${SRC_FILE} ${ARGN} LFS_BANDED_EXPORT_RENDER=1)
                        endif()
                else()
                        set(_custom_kind Spirv)
                        set(_custom ${ARGN} LFS_REVEAL=1 LFS_CUSTOM_EFFECT=1)
                        _vulkan_add_slang_variant(${OUT_NAME}_custom ${SRC_FILE} ${_custom}
                                DEPENDS ${_lfs_user_shader_ir_sources})
                        if(OUT_NAME MATCHES "^rasterize_forward")
                                _vulkan_add_slang_variant(${OUT_NAME}_custom_banded ${SRC_FILE} ${_custom}
                                        LFS_BANDED_EXPORT_RENDER=1 DEPENDS ${_lfs_user_shader_ir_sources})
                        endif()
                endif()
                set(_custom_name "${OUT_NAME}_custom")
        endif()
        if(_custom_name)
                if("LFS_PROJECTION_SURVIVORS=1" IN_LIST ARGN OR OUT_NAME STREQUAL "cull_splats")
                        set(_chain Survivors)
                elseif("LFS_3DGUT_PROJECTION=1" IN_LIST ARGN OR "LFS_3DGUT_RASTERIZE=1" IN_LIST ARGN)
                        set(_chain Gut)
                endif()
                # A banded draw requests this label even when its host is not built.
                if(_banded)
                        set(_custom_banded "${OUT_NAME}_custom_banded")
                endif()
        endif()
        if(_banded OR _reveal OR _custom_name)
                set_property(GLOBAL APPEND PROPERTY LFS_SHADER_VARIANTS
                        "{\"${OUT_NAME}\", \"${_banded}\", \"${_reveal}\", \"${_custom_name}\", \"${_custom_banded}\", CustomKind::${_custom_kind}, CustomChain::${_chain}},")
        endif()
        set(VULKAN_RASTERIZER_SHADER_OUTPUTS ${VULKAN_RASTERIZER_SHADER_OUTPUTS} PARENT_SCOPE)
endfunction()

function(_vulkan_add_glsl_shader OUT_NAME SRC_FILE)
        set(_defs "")
        foreach(D ${ARGN})
                list(APPEND _defs "-D${D}")
        endforeach()
        set(_out ${_VULKAN_RADIX_OUT}/${OUT_NAME}.spv)
        set(_spirv_validation)
        set(_spirv_validation_dependency)
        if(_VULKAN_SPIRV_VAL)
                set(_spirv_validation COMMAND ${_VULKAN_SPIRV_VAL} ${_out})
                set(_spirv_validation_dependency ${_VULKAN_SPIRV_VAL})
        endif()
        add_custom_command(
                OUTPUT ${_out}
                COMMAND ${GLSLANG}
                        ${_VULKAN_GLSL_FLAGS}
                        ${_defs}
                        ${_VULKAN_SHADER_SRC}/radix_sort/${SRC_FILE}
                        -o ${_out}
                ${_spirv_validation}
                DEPENDS ${_VULKAN_SHADER_SRC}/radix_sort/${SRC_FILE} ${_VULKAN_RADIX_DEPS} ${_spirv_validation_dependency}
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
_vulkan_add_slang_shader(radix_histogram_clear     radix_histogram_clear.slang)
_vulkan_add_slang_shader(expected_depth_finalize   expected_depth_finalize.slang)
_vulkan_add_slang_shader(stochastic_reuse          stochastic_reuse.slang)
_vulkan_add_slang_shader(stochastic_resolve        stochastic_resolve.slang)
_vulkan_add_slang_shader(selection_outline_coverage selection_outline_coverage.slang)
_vulkan_add_slang_shader(selection_outline_resolve selection_outline_resolve.slang)
_vulkan_add_slang_shader(selection_outline_seeds selection_outline_seeds.slang)
_vulkan_add_slang_shader(selection_outline_horizontal selection_outline_horizontal.slang)
_vulkan_add_slang_shader(selection_outline_background_init selection_outline_background.slang OUTLINE_BACKGROUND_INIT=1)
_vulkan_add_slang_shader(selection_outline_background_union selection_outline_background.slang)
_vulkan_add_slang_shader(selection_outline_tiles macro_tile_shader.slang MACRO_ENTRY=7)
_vulkan_add_slang_shader(selection_outline_mask selection_outline_mask.slang)
_vulkan_add_slang_shader(rasterize_outline alphablend_shader.slang LFS_SELECTION_OUTLINE=1)
_vulkan_add_slang_shader(rasterize_outline_gut alphablend_shader.slang LFS_3DGUT_RASTERIZE=1 LFS_SELECTION_OUTLINE=1)
_vulkan_add_slang_shader(rasterize_outline_plain alphablend_shader.slang LFS_RASTER_OVERLAYS=0 LFS_SELECTION_OUTLINE=1)
_vulkan_add_slang_shader(rasterize_outline_gut_plain alphablend_shader.slang LFS_3DGUT_RASTERIZE=1 LFS_RASTER_OVERLAYS=0 LFS_SELECTION_OUTLINE=1)

_vulkan_add_slang_shader(generate_keys_wave        tile_shader.slang ENTRY=1)
_vulkan_add_slang_shader(compute_tile_ranges       tile_shader.slang ENTRY=2)
_vulkan_add_slang_shader(compute_tile_ranges_and_batch_counts tile_shader.slang ENTRY=3)

# Two-stage sort helper shaders
_vulkan_add_slang_shader(apply_depth_ordering      apply_depth_ordering.slang)
_vulkan_add_slang_shader(visible_flags             visible_flags.slang)
_vulkan_add_slang_shader(prepare_visible_sort      prepare_visible_sort.slang)
_vulkan_add_slang_shader(prepare_tile_sort         prepare_tile_sort.slang)
_vulkan_add_slang_shader(compact_visible_primitives compact_visible_primitives.slang)
_vulkan_add_slang_shader(lod_map_indices           lod_map_indices.slang)
_vulkan_add_slang_shader(lod_select_threshold      lod_select_threshold.slang)
_vulkan_add_slang_shader(lod_compact_touch        lod_compact_touch.slang)

# vertex_shader.slang → projection_forward variants
_vulkan_add_slang_shader(projection_forward        vertex_shader.slang)
_vulkan_add_slang_shader(projection_forward_3dgut  vertex_shader.slang LFS_3DGUT_PROJECTION=1)
# Canonical quantized LOD pool variants (core/rad_pool_quant.hpp)
_vulkan_add_slang_shader(projection_forward_quant       vertex_shader.slang LFS_QUANT_POOL=1)
_vulkan_add_slang_shader(projection_forward_quant_3dgut vertex_shader.slang LFS_3DGUT_PROJECTION=1 LFS_QUANT_POOL=1)
# Standalone PLY/SOG IEEE f16 SH rest
_vulkan_add_slang_shader(projection_forward_shn_f16       vertex_shader.slang LFS_SHN_F16=1)
_vulkan_add_slang_shader(projection_forward_shn_f16_3dgut vertex_shader.slang LFS_3DGUT_PROJECTION=1 LFS_SHN_F16=1)
# Training-viewport exportable pad-dropped q16 SH rest (zero-copy dequant)
_vulkan_add_slang_shader(projection_forward_shn_q16       vertex_shader.slang LFS_SHN_Q16=1)
_vulkan_add_slang_shader(projection_forward_shn_q16_3dgut vertex_shader.slang LFS_3DGUT_PROJECTION=1 LFS_SHN_Q16=1)

# selection_mask.slang → compact 1-byte-per-splat brush/rectangle/polygon masks
_vulkan_add_slang_shader(selection_mask            selection_mask.slang)

# selection_polygon_rasterize.slang → rasterizes a screen-space polygon into a
# coverage mask consumed by selection_mask in POLYGON_MASK mode.
_vulkan_add_slang_shader(selection_polygon_rasterize selection_polygon_rasterize.slang)

# alphablend_shader.slang → forward variants. The *_plain variants strip the
# editor overlay/selection path from the per-pixel inner loop; the host selects
# them whenever no overlay is active (the common viewing case).
_vulkan_add_slang_shader(rasterize_forward               alphablend_shader.slang)
_vulkan_add_slang_shader(rasterize_forward_3dgut         alphablend_shader.slang LFS_3DGUT_RASTERIZE=1)
_vulkan_add_slang_shader(rasterize_forward_plain         alphablend_shader.slang LFS_RASTER_OVERLAYS=0)
_vulkan_add_slang_shader(rasterize_stochastic            alphablend_shader.slang LFS_STOCHASTIC_RASTER=1)
_vulkan_add_slang_shader(rasterize_stochastic_plain      alphablend_shader.slang LFS_STOCHASTIC_RASTER=1 LFS_RASTER_OVERLAYS=0)
_vulkan_add_slang_shader(rasterize_forward_3dgut_plain   alphablend_shader.slang LFS_3DGUT_RASTERIZE=1 LFS_RASTER_OVERLAYS=0)
# Spark RAD scenes get their own 3DGUT variants so LOD-parent kernels stay out of the ordinary loop.
_vulkan_add_slang_shader(rasterize_outline_gut_rad alphablend_shader.slang LFS_SELECTION_OUTLINE=1 LFS_3DGUT_RASTERIZE=1 LFS_SPARK_RAD_RASTER=1)
_vulkan_add_slang_shader(rasterize_outline_gut_plain_rad alphablend_shader.slang LFS_SELECTION_OUTLINE=1 LFS_3DGUT_RASTERIZE=1 LFS_RASTER_OVERLAYS=0 LFS_SPARK_RAD_RASTER=1)
_vulkan_add_slang_shader(rasterize_forward_3dgut_rad       alphablend_shader.slang LFS_3DGUT_RASTERIZE=1 LFS_SPARK_RAD_RASTER=1)
_vulkan_add_slang_shader(rasterize_forward_3dgut_plain_rad alphablend_shader.slang LFS_3DGUT_RASTERIZE=1 LFS_RASTER_OVERLAYS=0 LFS_SPARK_RAD_RASTER=1)
_vulkan_add_slang_shader(rasterize_forward_light         alphablend_shader.slang LFS_SKIP_DENSE_TILES=1)
_vulkan_add_slang_shader(rasterize_forward_light_plain   alphablend_shader.slang LFS_RASTER_OVERLAYS=0 LFS_SKIP_DENSE_TILES=1)
_vulkan_add_slang_shader(tile_batch_descriptors          tile_batch_shader.slang TILE_BATCH_ENTRY=2)
_vulkan_add_slang_shader(rasterize_forward_batches       tile_batch_shader.slang TILE_BATCH_ENTRY=3 LFS_RASTER_OVERLAYS=1)
_vulkan_add_slang_shader(rasterize_forward_batches_plain tile_batch_shader.slang TILE_BATCH_ENTRY=3)
_vulkan_add_slang_shader(compose_tile_batches            tile_batch_shader.slang TILE_BATCH_ENTRY=4 LFS_RASTER_OVERLAYS=1)
_vulkan_add_slang_shader(compose_tile_batches_plain      tile_batch_shader.slang TILE_BATCH_ENTRY=4)

# HiGS viewer chain: position-only cull prepass, survivor projection with
# compact-slot outputs, visible-bounded sort/cumsum/key generation, and
# compact-slot overlay raster variants.
_vulkan_add_slang_shader(cull_splats                  cull_splats.slang CULL_ENTRY=1)
_vulkan_add_slang_shader(cull_prepare                 cull_splats.slang CULL_ENTRY=2)
_vulkan_add_slang_shader(projection_forward_survivors vertex_shader.slang LFS_PROJECTION_SURVIVORS=1)
_vulkan_add_slang_shader(projection_forward_quant_survivors vertex_shader.slang LFS_PROJECTION_SURVIVORS=1 LFS_QUANT_POOL=1)
_vulkan_add_slang_shader(projection_forward_shn_f16_survivors vertex_shader.slang LFS_PROJECTION_SURVIVORS=1 LFS_SHN_F16=1)
_vulkan_add_slang_shader(projection_forward_shn_q16_survivors vertex_shader.slang LFS_PROJECTION_SURVIVORS=1 LFS_SHN_Q16=1)
_vulkan_add_slang_shader(prepare_visible_chain        prepare_visible_chain.slang)
_vulkan_add_slang_shader(copy_visible_indices         copy_visible_indices.slang)
_vulkan_add_slang_shader(prepare_stable_depth_sort    prepare_stable_depth_sort.slang)
_vulkan_add_slang_shader(cumsum_block_scan_indirect        cumsum.slang CUMSUM_PHASE=1 CUMSUM_INDIRECT=1)
_vulkan_add_slang_shader(cumsum_scan_block_sums_indirect   cumsum.slang CUMSUM_PHASE=2 CUMSUM_INDIRECT=1)
_vulkan_add_slang_shader(cumsum_add_block_offsets_indirect cumsum.slang CUMSUM_PHASE=3 CUMSUM_INDIRECT=1)
_vulkan_add_slang_shader(prepare_tile_sort_visible    prepare_tile_sort.slang LFS_VISIBLE_BOUNDED=1)

# HiGS macro-tile binning + raster/compose. Plain and overlay rasters each ship
# an fp16 (default) and fp32 staging variant — overlays no longer force fp32.
_vulkan_add_slang_shader(macro_coverage         macro_tile_shader.slang MACRO_ENTRY=1)
_vulkan_add_slang_shader(generate_macro_keys_wave macro_tile_shader.slang MACRO_ENTRY=2)
_vulkan_add_slang_shader(compute_macro_ranges   macro_tile_shader.slang MACRO_ENTRY=3)
_vulkan_add_slang_shader(macro_batch_prepare    macro_tile_shader.slang MACRO_ENTRY=5)
_vulkan_add_slang_shader(wave_partition         macro_tile_shader.slang MACRO_ENTRY=6)
_vulkan_add_slang_shader(wave_partition_visible macro_tile_shader.slang MACRO_ENTRY=6 LFS_VISIBLE_BOUNDED=1)
_vulkan_add_slang_shader(macro_raster                  macro_raster.slang)
_vulkan_add_slang_shader(macro_raster_fp32             macro_raster.slang LFS_HIGS_FP32=1)
_vulkan_add_slang_shader(macro_raster_fp32_lean        macro_raster.slang LFS_HIGS_FP32=1 LFS_HIGS_LEAN_STAGING=1)
_vulkan_add_slang_shader(macro_raster_overlays         macro_raster.slang LFS_RASTER_OVERLAYS=1)
_vulkan_add_slang_shader(macro_raster_overlays_lean    macro_raster.slang LFS_RASTER_OVERLAYS=1 LFS_HIGS_LEAN_STAGING=1)
_vulkan_add_slang_shader(macro_raster_overlays_fp32    macro_raster.slang LFS_HIGS_FP32=1 LFS_RASTER_OVERLAYS=1)
_vulkan_add_slang_shader(macro_raster_overlays_fp32_lean macro_raster.slang LFS_HIGS_FP32=1 LFS_RASTER_OVERLAYS=1 LFS_HIGS_LEAN_STAGING=1)
# Accurate transparent comparison is specific to the Mac test reference.
if(_VULKAN_REFERENCE_PROFILE)
    _vulkan_add_slang_shader(macro_raster_fp32_precise_alpha             macro_raster.slang LFS_HIGS_FP32=1 LFS_HIGS_PRECISE_ALPHA=1)
    _vulkan_add_slang_shader(macro_raster_fp32_lean_precise_alpha        macro_raster.slang LFS_HIGS_FP32=1 LFS_HIGS_LEAN_STAGING=1 LFS_HIGS_PRECISE_ALPHA=1)
    _vulkan_add_slang_shader(macro_raster_overlays_fp32_precise_alpha    macro_raster.slang LFS_HIGS_FP32=1 LFS_RASTER_OVERLAYS=1 LFS_HIGS_PRECISE_ALPHA=1)
    _vulkan_add_slang_shader(macro_raster_overlays_fp32_lean_precise_alpha macro_raster.slang LFS_HIGS_FP32=1 LFS_RASTER_OVERLAYS=1 LFS_HIGS_LEAN_STAGING=1 LFS_HIGS_PRECISE_ALPHA=1)
endif()
_vulkan_add_slang_shader(macro_compose          macro_compose.slang)
_vulkan_add_slang_shader(macro_compose_overlays macro_compose.slang LFS_RASTER_OVERLAYS=1)

# Radix sort GLSL .comp
_vulkan_add_glsl_shader(upsweep_indirect    upsweep.comp)
_vulkan_add_glsl_shader(spine_indirect      spine.comp)
_vulkan_add_glsl_shader(downsweep_indirect  downsweep.comp)

add_custom_target(${_VULKAN_RASTERIZER_TARGET}_shaders DEPENDS ${VULKAN_RASTERIZER_SHADER_OUTPUTS})
add_dependencies(${_VULKAN_RASTERIZER_TARGET} ${_VULKAN_RASTERIZER_TARGET}_shaders)

get_property(_shader_variants GLOBAL PROPERTY LFS_SHADER_VARIANTS)
list(JOIN _shader_variants "\n" _shader_variants)
file(CONFIGURE OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/shader_variants.inc CONTENT "${_shader_variants}\n")

get_property(_slang_recipes GLOBAL PROPERTY LFS_VULKAN_SLANG_RECIPES)
list(JOIN _slang_recipes ",\n    " _slang_recipes)
list(JOIN _VULKAN_SLANG_FLAGS "\", \"" _slang_flags)
file(CONFIGURE OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/slang_recipes.json
        CONTENT "{\n  \"flags\": [\"${_slang_flags}\"],\n  \"recipes\": [\n    ${_slang_recipes}\n  ]\n}\n")
set_target_properties(${_VULKAN_RASTERIZER_TARGET} PROPERTIES
        LFS_SLANG_RECIPES "${CMAKE_CURRENT_BINARY_DIR}/slang_recipes.json")

# User shaders run only where vksplat draws the viewport; the macOS reference
# profile is a test fixture for the Metal viewer and ships no user-shader package.
if(NOT _VULKAN_REFERENCE_PROFILE)
    # Both packages assemble only finished IR/reflection outputs. Direct ABI inputs
    # also cover clean builds before the compiler has written any depfiles.
    get_property(_lfs_custom_hosts GLOBAL PROPERTY LFS_USER_SHADER_HOSTS)
    list(JOIN _lfs_custom_hosts "|" _lfs_custom_host_list)
    set(_lfs_user_shader_abi ${CMAKE_BINARY_DIR}/generated/user_shaders/lfs_shader_abi.hpp)
    set(_lfs_user_shader_test_dir ${CMAKE_CURRENT_BINARY_DIR}/user_shader_test_package)
    foreach(_lfs_user_shader_package lfs_user_shader_ir lfs_user_shader_test_package)
            set(_manifest_args -DHOSTS=${_lfs_custom_host_list} -DABI_HEADER=${_lfs_user_shader_abi})
            set(_byproducts ${_lfs_user_shader_abi})
            if(_lfs_user_shader_package STREQUAL "lfs_user_shader_test_package")
                    set(_lfs_user_shader_dir ${_lfs_user_shader_test_dir})
                    set(_lfs_user_shader_reflection ${CMAKE_CURRENT_BINARY_DIR}/user_shader_test_reflection)
                    set(_manifest_args -DTEST_HOST=lfs_test_host)
                    set(_byproducts)
                    lfs_add_shader_ir(NAME lfs_test_host SOURCE lfs_test_host.slang REFLECT)
            else()
                    foreach(_name lfs_reveal spherical_harmonics)
                            lfs_add_shader_ir(NAME ${_name} SOURCE ${_name}.slang)
                    endforeach()
                    lfs_add_shader_ir(NAME utils SOURCE utils.slang DEFINES LFS_CUSTOM_EFFECT=1)
                    lfs_add_shader_ir(NAME utils_banded SOURCE utils.slang DEFINES LFS_CUSTOM_EFFECT=1 LFS_BANDED_EXPORT_RENDER=1)
            endif()
            foreach(_name lfs_helpers lfs)
                    lfs_add_shader_ir(NAME ${_name} SOURCE ${_name}.slang)
            endforeach()
            get_property(_ir_files GLOBAL PROPERTY ${_lfs_user_shader_package}_OUTPUTS)
            add_custom_command(
                    OUTPUT ${_lfs_user_shader_dir}/manifest.json
                    BYPRODUCTS ${_byproducts}
                    COMMAND ${CMAKE_COMMAND} -DSLANGC=${SLANGC}
                            -DSOURCE_DIR=${_lfs_user_shader_src} -DSHARED_SOURCE_DIR=${_SHARED_SLANG_SRC}
                            -DOUTPUT_DIR=${_lfs_user_shader_dir}
                            -DREFLECTION_DIR=${_lfs_user_shader_reflection} ${_manifest_args}
                            -P ${CMAKE_SOURCE_DIR}/cmake/GenerateUserShaderIR.cmake
                    DEPENDS ${_ir_files} ${SLANGC} ${CMAKE_SOURCE_DIR}/cmake/GenerateUserShaderIR.cmake
                            ${_SHARED_SLANG_SRC}/config.slang ${_lfs_user_shader_src}/reveal_noise.inc
                            ${_lfs_user_shader_src}/lfs_helpers.slang ${_lfs_user_shader_src}/lfs.slang
                    COMMENT "Assembling ${_lfs_user_shader_package} manifest" VERBATIM)
            add_custom_target(${_lfs_user_shader_package} DEPENDS ${_lfs_user_shader_dir}/manifest.json)
    endforeach()
    set(_lfs_user_shader_dir ${LFS_VULKAN_RASTERIZER_SPV_DIR}/user)
    add_dependencies(${_VULKAN_RASTERIZER_TARGET}_shaders lfs_user_shader_ir)
    add_dependencies(lfs_shader_headers lfs_user_shader_ir) # lfs_shader_abi.hpp

    set_target_properties(lfs_user_shader_ir PROPERTIES
            LFS_USER_SHADER_PACKAGE_DIR "${_lfs_user_shader_dir}"
            LFS_USER_SHADER_TEST_PACKAGE_DIR "${_lfs_user_shader_test_dir}"
            LFS_SPIRV_VAL "${_VULKAN_SPIRV_VAL}"
    )
endif()

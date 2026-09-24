set(LFS_TENSOR_SPV_DIR "${CMAKE_BINARY_DIR}/lfs_tensor_spv")
file(MAKE_DIRECTORY "${LFS_TENSOR_SPV_DIR}")

set(_tensor_host_triplet "${VCPKG_HOST_TRIPLET}")
if(NOT _tensor_host_triplet)
    set(_tensor_host_triplet "${VCPKG_TARGET_TRIPLET}")
endif()

find_program(LFS_TENSOR_SLANGC
    NAMES slangc
    HINTS
        ${CMAKE_PREFIX_PATH}
        ${CMAKE_PREFIX_PATH}/tools/shader-slang
        ${VCPKG_INSTALLED_DIR}/${_tensor_host_triplet}/tools/shader-slang
    PATH_SUFFIXES tools/shader-slang
    REQUIRED)

find_package(SPIRV-Headers CONFIG REQUIRED)
find_package(SPIRV-Tools CONFIG QUIET)

# Host tool run on every compiled module: declares the float-controls
# execution modes the backend contract requires, validates the result when the
# SPIRV-Tools library is available and writes the manifest from the binary.
add_library(lfs_tensor_spirv STATIC
    ${CMAKE_CURRENT_SOURCE_DIR}/tensor/backend/vulkan/spirv_module.cpp)
set_target_properties(lfs_tensor_spirv PROPERTIES
    POSITION_INDEPENDENT_CODE ON
    CXX_VISIBILITY_PRESET hidden
    VISIBILITY_INLINES_HIDDEN ON)
target_include_directories(lfs_tensor_spirv PRIVATE ${CMAKE_SOURCE_DIR}/src)
target_compile_definitions(lfs_tensor_spirv PRIVATE LFS_TENSOR_PRIVATE_ACCESS)
target_link_libraries(lfs_tensor_spirv PUBLIC SPIRV-Headers::SPIRV-Headers)
if(TARGET SPIRV-Tools-static)
    target_link_libraries(lfs_tensor_spirv PRIVATE SPIRV-Tools-static)
    target_compile_definitions(lfs_tensor_spirv PRIVATE LFS_SPIRV_FINALIZE_HAS_TOOLS=1)
elseif(TARGET SPIRV-Tools)
    target_link_libraries(lfs_tensor_spirv PRIVATE SPIRV-Tools)
    target_compile_definitions(lfs_tensor_spirv PRIVATE LFS_SPIRV_FINALIZE_HAS_TOOLS=1)
else()
    message(STATUS "tensor Vulkan shaders: SPIRV-Tools not found, finalized modules are not validated")
endif()
add_executable(lfs_tensor_spirv_finalize
    ${CMAKE_CURRENT_SOURCE_DIR}/tensor/backend/vulkan/tools/spirv_finalize.cpp)
target_include_directories(lfs_tensor_spirv_finalize PRIVATE ${CMAKE_SOURCE_DIR}/src)
target_compile_definitions(lfs_tensor_spirv_finalize PRIVATE LFS_TENSOR_PRIVATE_ACCESS)
target_link_libraries(lfs_tensor_spirv_finalize PRIVATE lfs_tensor_spirv)

file(GLOB LFS_TENSOR_SLANG_HEADERS CONFIGURE_DEPENDS
    "${CMAKE_CURRENT_SOURCE_DIR}/tensor/backend/vulkan/shaders/*.slangh")
list(APPEND LFS_TENSOR_SLANG_HEADERS
    "${CMAKE_CURRENT_SOURCE_DIR}/tensor/internal/point_filter.hpp")

set(LFS_TENSOR_SHADER_OUTPUTS)
function(_tensor_add_slang_kernel name src)
    set(_source "${CMAKE_CURRENT_SOURCE_DIR}/tensor/backend/vulkan/shaders/${src}")
    set(_raw "${LFS_TENSOR_SPV_DIR}/${name}.raw.spv")
    set(_spv "${LFS_TENSOR_SPV_DIR}/${name}.spv")
    set(_manifest "${LFS_TENSOR_SPV_DIR}/${name}.json")
    add_custom_command(
        OUTPUT ${_spv} ${_manifest}
        BYPRODUCTS ${_raw}
        COMMAND ${LFS_TENSOR_SLANGC} ${_source}
                -entry main -stage compute -O -fp-mode precise
                -line-directive-mode none -target spirv -profile glsl_460
                -emit-spirv-directly
                -I ${CMAKE_CURRENT_SOURCE_DIR}/tensor/backend/vulkan/shaders
                -o ${_raw}
        COMMAND lfs_tensor_spirv_finalize ${_raw} ${_spv} ${_manifest}
        DEPENDS ${_source} ${LFS_TENSOR_SLANG_HEADERS} ${LFS_TENSOR_SLANGC} ${ARGN}
                lfs_tensor_spirv_finalize
        COMMENT "slangc tensor/${name}.spv"
        VERBATIM)
    list(APPEND LFS_TENSOR_SHADER_OUTPUTS ${_spv} ${_manifest})
    set(LFS_TENSOR_SHADER_OUTPUTS ${LFS_TENSOR_SHADER_OUTPUTS} PARENT_SCOPE)
    list(APPEND LFS_TENSOR_SHADER_EMBED_ARGS ${name} ${_spv})
    set(LFS_TENSOR_SHADER_EMBED_ARGS ${LFS_TENSOR_SHADER_EMBED_ARGS} PARENT_SCOPE)
endfunction()
set(LFS_TENSOR_SHADER_EMBED_ARGS)

_tensor_add_slang_kernel(fill fill.slang)
_tensor_add_slang_kernel(arange arange.slang)
_tensor_add_slang_kernel(pointwise pointwise.slang)
_tensor_add_slang_kernel(pointwise_half pointwise_half.slang)
_tensor_add_slang_kernel(broadcast broadcast.slang)
_tensor_add_slang_kernel(convert convert.slang)
_tensor_add_slang_kernel(strided_copy strided_copy.slang)
_tensor_add_slang_kernel(cat_pad cat_pad.slang)
_tensor_add_slang_kernel(reduce reduce.slang)
_tensor_add_slang_kernel(count count.slang)
_tensor_add_slang_kernel(scan scan.slang)
_tensor_add_slang_kernel(index index.slang)
_tensor_add_slang_kernel(index_atomic index_atomic.slang)
_tensor_add_slang_kernel(mask mask.slang)
_tensor_add_slang_kernel(where where.slang)
_tensor_add_slang_kernel(sort sort.slang)
_tensor_add_slang_kernel(radix radix.slang)
_tensor_add_slang_kernel(gemm gemm.slang)
_tensor_add_slang_kernel(matrix_misc matrix_misc.slang)
_tensor_add_slang_kernel(cdist cdist.slang)
_tensor_add_slang_kernel(nn nn.slang)
_tensor_add_slang_kernel(inference inference.slang)
_tensor_add_slang_kernel(random random.slang)
_tensor_add_slang_kernel(radius_neighbors radius_neighbors.slang)
_tensor_add_slang_kernel(project_points project_points.slang)
_tensor_add_slang_kernel(point_region point_region.slang)
_tensor_add_slang_kernel(histogram_u8 histogram_u8.slang)
_tensor_add_slang_kernel(ppisp_apply ppisp_apply.slang)
_tensor_add_slang_kernel(environment_composite environment_composite.slang)
_tensor_add_slang_kernel(image_resample image_resample.slang)
_tensor_add_slang_kernel(update_labels update_labels.slang)
_tensor_add_slang_kernel(filter_points filter_points.slang)
_tensor_add_slang_kernel(affine_splat_geometry affine_splat_geometry.slang)
_tensor_add_slang_kernel(sh_codec sh_codec.slang)
_tensor_add_slang_kernel(sh_encode sh_encode.slang)
_tensor_add_slang_kernel(rad_page rad_page.slang
    ${CMAKE_CURRENT_SOURCE_DIR}/include/core/rad_packed_page.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/include/core/rad_dequant_math.hpp)
# The library carries its modules: one generated translation unit holds every
# finalized SPIR-V word stream and the facts the loader checks.
set(LFS_TENSOR_SHADER_TABLE "${LFS_TENSOR_SPV_DIR}/lfs_tensor_shader_table.cpp")
add_custom_command(
    OUTPUT ${LFS_TENSOR_SHADER_TABLE}
    COMMAND lfs_tensor_spirv_finalize embed ${LFS_TENSOR_SHADER_TABLE} ${LFS_TENSOR_SHADER_EMBED_ARGS}
    DEPENDS ${LFS_TENSOR_SHADER_OUTPUTS} lfs_tensor_spirv_finalize
    COMMENT "embed tensor shaders"
    VERBATIM)
add_custom_target(lfs_tensor_vulkan_shaders DEPENDS ${LFS_TENSOR_SHADER_OUTPUTS} ${LFS_TENSOR_SHADER_TABLE})

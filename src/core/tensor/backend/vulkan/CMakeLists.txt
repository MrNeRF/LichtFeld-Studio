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

find_program(LFS_TENSOR_SPIRV_VAL
    NAMES spirv-val
    HINTS
        ${CMAKE_PREFIX_PATH}
        ${CMAKE_PREFIX_PATH}/tools/spirv-tools
        ${VCPKG_INSTALLED_DIR}/${_tensor_host_triplet}/tools/spirv-tools
    PATH_SUFFIXES tools/spirv-tools
    NO_CACHE)

set(LFS_TENSOR_SHADER_OUTPUTS)
function(_tensor_add_slang_kernel name src)
    set(_source "${CMAKE_CURRENT_SOURCE_DIR}/tensor/backend/vulkan/shaders/${src}")
    set(_spv "${LFS_TENSOR_SPV_DIR}/${name}.spv")
    set(_manifest "${LFS_TENSOR_SPV_DIR}/${name}.json")
    set(_validation)
    if(LFS_TENSOR_SPIRV_VAL)
        set(_validation COMMAND ${LFS_TENSOR_SPIRV_VAL} --target-env vulkan1.3 ${_spv})
    endif()
    add_custom_command(
        OUTPUT ${_spv} ${_manifest}
        COMMAND ${LFS_TENSOR_SLANGC} ${_source}
                -entry main -stage compute -O -fp-mode precise
                -line-directive-mode none -target spirv -profile glsl_460
                -emit-spirv-directly -o ${_spv}
        ${_validation}
        COMMAND ${CMAKE_COMMAND} -DSOURCE=${_source} -DOUTPUT=${_manifest}
                -P ${CMAKE_CURRENT_SOURCE_DIR}/tensor/backend/vulkan/write_shader_manifest.cmake
        DEPENDS ${_source}
                ${CMAKE_CURRENT_SOURCE_DIR}/tensor/backend/vulkan/write_shader_manifest.cmake
                ${LFS_TENSOR_SLANGC}
        COMMENT "slangc tensor/${name}.spv"
        VERBATIM)
    list(APPEND LFS_TENSOR_SHADER_OUTPUTS ${_spv} ${_manifest})
    set(LFS_TENSOR_SHADER_OUTPUTS ${LFS_TENSOR_SHADER_OUTPUTS} PARENT_SCOPE)
endfunction()

_tensor_add_slang_kernel(fill fill.slang)
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
add_custom_target(lfs_tensor_vulkan_shaders DEPENDS ${LFS_TENSOR_SHADER_OUTPUTS})

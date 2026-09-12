cmake_minimum_required(VERSION 3.25)

project(lfs_vk_probes LANGUAGES CXX)

find_package(Vulkan REQUIRED)
find_package(VulkanMemoryAllocator CONFIG REQUIRED)

find_program(
    SLANGC
    NAMES slangc
    HINTS
        ${CMAKE_PREFIX_PATH}
        ${CMAKE_PREFIX_PATH}/tools/shader-slang
        ${VCPKG_INSTALLED_DIR}/${VCPKG_TARGET_TRIPLET}/tools/shader-slang
    PATH_SUFFIXES tools/shader-slang
    REQUIRED)

find_program(
    SPIRV_VAL
    NAMES spirv-val
    HINTS
        ${CMAKE_PREFIX_PATH}
        ${CMAKE_PREFIX_PATH}/tools/spirv-tools
        ${VCPKG_INSTALLED_DIR}/${VCPKG_HOST_TRIPLET}/tools/spirv-tools
        ${VCPKG_INSTALLED_DIR}/${VCPKG_TARGET_TRIPLET}/tools/spirv-tools
    PATH_SUFFIXES tools/spirv-tools
    NO_CACHE)

set(SHADER_OUTPUT_DIR ${CMAKE_CURRENT_BINARY_DIR}/shaders)
file(MAKE_DIRECTORY ${SHADER_OUTPUT_DIR})

function(add_probe_shader name)
    set(source ${CMAKE_CURRENT_SOURCE_DIR}/shaders/${name}.slang)
    set(output ${SHADER_OUTPUT_DIR}/${name}.spv)
    set(validation_command)
    if(SPIRV_VAL)
        set(validation_command COMMAND ${SPIRV_VAL} --target-env vulkan1.3 ${output})
    endif()
    add_custom_command(
        OUTPUT ${output}
        COMMAND
            ${SLANGC} ${source} -entry main -stage compute -O -fp-mode precise
            -line-directive-mode none -target spirv -profile glsl_460
            -emit-spirv-directly -o ${output}
        ${validation_command}
        DEPENDS ${source}
        COMMENT "slangc ${name}.spv"
        VERBATIM)
    list(APPEND PROBE_SHADER_OUTPUTS ${output})
    set(PROBE_SHADER_OUTPUTS ${PROBE_SHADER_OUTPUTS} PARENT_SCOPE)
endfunction()

add_probe_shader(abi_probe)
add_probe_shader(reduction_probe)
add_custom_target(probe_shaders ALL DEPENDS ${PROBE_SHADER_OUTPUTS})

add_executable(vk_probes probe_main.cpp)
add_dependencies(vk_probes probe_shaders)
target_compile_features(vk_probes PRIVATE cxx_std_20)
target_compile_definitions(
    vk_probes PRIVATE
    VMA_STATIC_VULKAN_FUNCTIONS=1
    VMA_DYNAMIC_VULKAN_FUNCTIONS=0
    PROBE_SHADER_DIR="${SHADER_OUTPUT_DIR}"
    PROBE_SLANGC="${SLANGC}"
    PROBE_SPIRV_VAL="$<IF:$<BOOL:${SPIRV_VAL}>,${SPIRV_VAL},not-found>")
target_link_libraries(
    vk_probes PRIVATE Vulkan::Vulkan GPUOpen::VulkanMemoryAllocator)
target_compile_options(
    vk_probes PRIVATE -Wall -Wextra -Wpedantic -Werror
                      -Wno-missing-field-initializers)

message(STATUS "probe slangc: ${SLANGC}")
if(SPIRV_VAL)
    message(STATUS "probe spirv-val: ${SPIRV_VAL}")
else()
    message(STATUS "probe spirv-val: not found - validation skipped")
endif()

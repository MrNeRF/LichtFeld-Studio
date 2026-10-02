# SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
# SPDX-License-Identifier: MIT
#
# Standalone Linux/macOS build for the AMD FidelityFX SDK v1.1.4 FSR3 upscaler.
# The SDK is Windows-only upstream, its shader compiler is a Windows .exe,
# and its ABI requires two-byte wchar_t; this project supplies the POSIX tool
# driver/shims and builds only the GLSL Vulkan upscaler path.

cmake_minimum_required(VERSION 3.23)
project(ffx_fsr3_posix LANGUAGES C CXX)

if(NOT CMAKE_SYSTEM_NAME STREQUAL "Linux" AND NOT APPLE)
    message(FATAL_ERROR "ffx_fsr3_posix supports Linux and macOS only")
endif()
if(NOT FFX_SDK_SOURCE_DIR)
    message(FATAL_ERROR "FFX_SDK_SOURCE_DIR must point to a staged FidelityFX SDK sdk directory")
endif()
if(NOT GLSLANG_EXECUTABLE OR NOT EXISTS "${GLSLANG_EXECUTABLE}")
    message(FATAL_ERROR "GLSLANG_EXECUTABLE must point to a glslang executable")
endif()

if(APPLE)
    if(CMAKE_OSX_ARCHITECTURES)
        set(_ffx_arch "${CMAKE_OSX_ARCHITECTURES}")
    else()
        set(_ffx_arch "${CMAKE_SYSTEM_PROCESSOR}")
    endif()
    if(_ffx_arch MATCHES "^(arm64|aarch64)$")
        set(_ffx_archive_suffix arm64)
    elseif(_ffx_arch MATCHES "^(x86_64|amd64)$")
        set(_ffx_archive_suffix x64)
    else()
        message(FATAL_ERROR "Unsupported macOS FSR architecture: ${_ffx_arch}")
    endif()
else()
    set(_ffx_archive_suffix x64)
endif()

find_package(Threads REQUIRED)
find_package(Vulkan REQUIRED)

get_filename_component(FFX_SDK_SOURCE_DIR "${FFX_SDK_SOURCE_DIR}" ABSOLUTE)
get_filename_component(GLSLANG_EXECUTABLE "${GLSLANG_EXECUTABLE}" ABSOLUTE)
set(FFX_TOOL_SOURCE_DIR "${FFX_SDK_SOURCE_DIR}/tools/ffx_shader_compiler")
set(FFX_SHIM "${CMAKE_CURRENT_SOURCE_DIR}/ffx_linux_shims.h")
set(FFX_SHADER_OUTPUT_DIR "${CMAKE_CURRENT_BINARY_DIR}/shaders/vk")

set(FFX_TOOL_INCLUDES
    "${FFX_TOOL_SOURCE_DIR}/src"
    "${FFX_TOOL_SOURCE_DIR}/libs/MD5"
    "${FFX_TOOL_SOURCE_DIR}/libs/SPIRV-Reflect"
    "${FFX_TOOL_SOURCE_DIR}/libs/tiny-process-library")

add_executable(ffx_shader_compiler
    tool/ffx_sc_linux.cpp
    tool/utils_linux.cpp
    "${FFX_TOOL_SOURCE_DIR}/src/glsl_compiler.cpp"
    "${FFX_TOOL_SOURCE_DIR}/libs/MD5/md5.cpp"
    "${FFX_TOOL_SOURCE_DIR}/libs/tiny-process-library/process.cpp"
    "${FFX_TOOL_SOURCE_DIR}/libs/tiny-process-library/process_unix.cpp")
target_compile_features(ffx_shader_compiler PRIVATE cxx_std_17)
target_compile_definitions(ffx_shader_compiler PRIVATE FFX_LINUX_SHADER_COMPILER)
target_compile_options(ffx_shader_compiler PRIVATE -O2 -fPIC -U_FORTIFY_SOURCE -D_FORTIFY_SOURCE=0 -include "${FFX_SHIM}")
target_include_directories(ffx_shader_compiler PRIVATE ${FFX_TOOL_INCLUDES})
target_link_libraries(ffx_shader_compiler PRIVATE Threads::Threads)

add_library(ffx_spirv_reflect OBJECT
    "${FFX_TOOL_SOURCE_DIR}/libs/SPIRV-Reflect/spirv_reflect.c")
target_compile_definitions(ffx_spirv_reflect PRIVATE FFX_LINUX_SHADER_COMPILER)
target_compile_options(ffx_spirv_reflect PRIVATE -O2 -fPIC -include "${FFX_SHIM}")
target_include_directories(ffx_spirv_reflect PRIVATE "${FFX_TOOL_SOURCE_DIR}/libs/SPIRV-Reflect")
target_link_libraries(ffx_shader_compiler PRIVATE ffx_spirv_reflect)

set(FFX_PERMUTATION_HEADERS)
file(GLOB FFX_SHADER_SOURCES "${FFX_SDK_SOURCE_DIR}/src/backends/vk/shaders/fsr3upscaler/*.glsl")
list(SORT FFX_SHADER_SOURCES)
foreach(shader IN LISTS FFX_SHADER_SOURCES)
    get_filename_component(pass "${shader}" NAME_WE)
    foreach(variant IN ITEMS base wave64 16bit wave64_16bit)
        if(variant STREQUAL base)
            set(name "${pass}")
        else()
            set(name "${pass}_${variant}")
        endif()
        list(APPEND FFX_PERMUTATION_HEADERS "${FFX_SHADER_OUTPUT_DIR}/${name}_permutations.h")
    endforeach()
endforeach()
set(FFX_SHADER_STAMP "${CMAKE_CURRENT_BINARY_DIR}/fsr3upscaler_shaders.stamp")
add_custom_command(
    OUTPUT "${FFX_SHADER_STAMP}"
    COMMAND "${CMAKE_COMMAND}"
        -DFFX_SC_EXECUTABLE=$<TARGET_FILE:ffx_shader_compiler>
        -DFFX_SDK_SOURCE_DIR=${FFX_SDK_SOURCE_DIR}
        -DGLSLANG_EXECUTABLE=${GLSLANG_EXECUTABLE}
        -DFFX_OUTPUT_DIR=${FFX_SHADER_OUTPUT_DIR}
        -P "${CMAKE_CURRENT_SOURCE_DIR}/generate_shaders.cmake"
    COMMAND "${CMAKE_COMMAND}" -E touch "${FFX_SHADER_STAMP}"
    DEPENDS ffx_shader_compiler ${FFX_SHADER_SOURCES}
            "${CMAKE_CURRENT_SOURCE_DIR}/generate_shaders.cmake"
    BYPRODUCTS ${FFX_PERMUTATION_HEADERS})
add_custom_target(ffx_fsr3upscaler_shaders DEPENDS "${FFX_SHADER_STAMP}")

set(FFX_SDK_INCLUDES
    "${FFX_SDK_SOURCE_DIR}/include"
    "${FFX_SDK_SOURCE_DIR}/include/FidelityFX/gpu"
    "${FFX_SDK_SOURCE_DIR}/src"
    "${FFX_SDK_SOURCE_DIR}/src/shared"
    "${FFX_SDK_SOURCE_DIR}/src/backends/shared"
    "${FFX_SDK_SOURCE_DIR}/src/backends/vk"
    "${FFX_SDK_SOURCE_DIR}/src/components"
    "${FFX_SHADER_OUTPUT_DIR}")

function(ffx_posix_sdk_options target)
    target_compile_features(${target} PRIVATE cxx_std_17)
    target_compile_definitions(${target} PRIVATE FFX_FSR3UPSCALER)
    target_compile_options(${target} PRIVATE -O2 -fPIC -fshort-wchar -include "${FFX_SHIM}")
    target_include_directories(${target} PRIVATE ${FFX_SDK_INCLUDES})
    add_dependencies(${target} ffx_fsr3upscaler_shaders)
endfunction()

add_library(ffx_fsr3upscaler_${_ffx_archive_suffix} STATIC
    "${FFX_SDK_SOURCE_DIR}/src/components/fsr3upscaler/ffx_fsr3upscaler.cpp"
    "${FFX_SDK_SOURCE_DIR}/src/shared/ffx_assert.cpp"
    "${FFX_SDK_SOURCE_DIR}/src/shared/ffx_message.cpp"
    "${FFX_SDK_SOURCE_DIR}/src/shared/ffx_object_management.cpp")
ffx_posix_sdk_options(ffx_fsr3upscaler_${_ffx_archive_suffix})

add_library(ffx_backend_vk_${_ffx_archive_suffix} STATIC
    "${FFX_SDK_SOURCE_DIR}/src/backends/vk/ffx_vk.cpp"
    "${FFX_SDK_SOURCE_DIR}/src/backends/shared/ffx_shader_blobs.cpp"
    "${FFX_SDK_SOURCE_DIR}/src/backends/shared/blob_accessors/ffx_fsr3upscaler_shaderblobs.cpp"
    "${FFX_SDK_SOURCE_DIR}/src/shared/ffx_assert.cpp"
    "${FFX_SDK_SOURCE_DIR}/src/shared/ffx_message.cpp"
    "${FFX_SDK_SOURCE_DIR}/src/shared/ffx_breadcrumbs_list.cpp"
    ffx_vk_noop.cpp)
ffx_posix_sdk_options(ffx_backend_vk_${_ffx_archive_suffix})
target_link_libraries(ffx_backend_vk_${_ffx_archive_suffix} PRIVATE Vulkan::Vulkan)

install(TARGETS ffx_fsr3upscaler_${_ffx_archive_suffix}
                ffx_backend_vk_${_ffx_archive_suffix}
    ARCHIVE DESTINATION lib)

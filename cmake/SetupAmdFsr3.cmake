# SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
#
# SPDX-License-Identifier: GPL-3.0-or-later

include_guard(GLOBAL)

option(LFS_ENABLE_AMD_FSR3
    "Compile the optional AMD FidelityFX FSR 3.1 Vulkan adapter (requires a user-provided SDK build)"
    OFF)
set(LFS_AMD_FSR3_ROOT "" CACHE PATH
    "Path to an AMD FidelityFX SDK 1.1.x checkout; never downloaded automatically")
set(LFS_AMD_FSR3_LIBRARY_DIR "" CACHE PATH
    "Optional directory containing prebuilt FidelityFX FSR 3.1 Vulkan libraries")
option(LFS_AMD_FSR3_BUILD_SDK
    "Build the user-provided FidelityFX FSR 3.1 Vulkan SDK automatically when needed"
    ON)

set(_lfs_fsr3_download_url "https://gpuopen.com/amd-fidelityfx-sdk-1/")

if(NOT LFS_ENABLE_AMD_FSR3)
    set(LFS_AMD_FSR3_AVAILABLE OFF CACHE INTERNAL
        "Whether the AMD FidelityFX FSR 3.1 Vulkan SDK is available to LichtFeld" FORCE)
    return()
endif()

if(NOT LFS_AMD_FSR3_ROOT)
    message(FATAL_ERROR
        "LFS_ENABLE_AMD_FSR3=ON requires LFS_AMD_FSR3_ROOT to point to an AMD "
        "FidelityFX SDK 1.1.x checkout obtained by the person compiling LichtFeld. "
        "The SDK is not downloaded or distributed by this project. Official download: "
        "${_lfs_fsr3_download_url}")
endif()
if(NOT CMAKE_SIZEOF_VOID_P EQUAL 8)
    message(FATAL_ERROR "The AMD FSR 3.1 adapter currently supports 64-bit builds only")
endif()
if(NOT WIN32 AND NOT CMAKE_SYSTEM_NAME STREQUAL "Linux")
    message(FATAL_ERROR
        "The AMD FSR 3.1 Vulkan adapter currently supports Windows and Linux only")
endif()

cmake_path(ABSOLUTE_PATH LFS_AMD_FSR3_ROOT
    NORMALIZE
    BASE_DIRECTORY "${CMAKE_SOURCE_DIR}"
    OUTPUT_VARIABLE _lfs_fsr3_root)
set(_lfs_fsr3_include "${_lfs_fsr3_root}/sdk/include")
foreach(_header IN ITEMS
        FidelityFX/host/ffx_fsr3upscaler.h
        FidelityFX/host/backends/vk/ffx_vk.h)
    if(NOT EXISTS "${_lfs_fsr3_include}/${_header}")
        message(FATAL_ERROR
            "AMD FidelityFX SDK at '${_lfs_fsr3_root}' is missing sdk/include/${_header}. "
            "Use SDK v1.1.4 with the Vulkan FSR 3.1 upscaler sources. Official download: "
            "${_lfs_fsr3_download_url}")
    endif()
endforeach()

set(_lfs_fsr3_library_hints)
if(LFS_AMD_FSR3_LIBRARY_DIR)
    cmake_path(ABSOLUTE_PATH LFS_AMD_FSR3_LIBRARY_DIR
        NORMALIZE
        BASE_DIRECTORY "${CMAKE_SOURCE_DIR}"
        OUTPUT_VARIABLE _lfs_fsr3_library_override)
    list(APPEND _lfs_fsr3_library_hints "${_lfs_fsr3_library_override}")
endif()
list(APPEND _lfs_fsr3_library_hints
    "${_lfs_fsr3_root}/bin/ffx_sdk"
    "${_lfs_fsr3_root}/sdk/bin/ffx_sdk"
    "${_lfs_fsr3_root}/build/bin/ffx_sdk"
    "${_lfs_fsr3_root}/build/sdk/bin/ffx_sdk")

unset(_lfs_fsr3_effect_library CACHE)
unset(_lfs_fsr3_backend_library CACHE)
find_library(_lfs_fsr3_effect_library
    NAMES ffx_fsr3upscaler_x64 ffx_fsr3upscaler_x86_64 ffx_fsr3upscaler
    PATHS ${_lfs_fsr3_library_hints}
    NO_DEFAULT_PATH)
find_library(_lfs_fsr3_backend_library
    NAMES ffx_backend_vk_x64 ffx_backend_vk_x86_64 ffx_backend_vk
    PATHS ${_lfs_fsr3_library_hints}
    NO_DEFAULT_PATH)
set(_lfs_fsr3_build_target "")
if(NOT _lfs_fsr3_effect_library OR NOT _lfs_fsr3_backend_library)
    if(LFS_AMD_FSR3_BUILD_SDK AND WIN32)
        include(ExternalProject)
        set(_lfs_fsr3_stage_root "${CMAKE_BINARY_DIR}/_deps/amd-fsr3-sdk")
        set(_lfs_fsr3_stage_source "${_lfs_fsr3_stage_root}/src")
        set(_lfs_fsr3_stage_binary "${_lfs_fsr3_stage_root}/build")
        set(_lfs_fsr3_stage_library_dir "${_lfs_fsr3_stage_source}/bin/ffx_sdk")
        set(_lfs_fsr3_effect_library
            "${_lfs_fsr3_stage_library_dir}/ffx_fsr3upscaler_x64.lib")
        set(_lfs_fsr3_backend_library
            "${_lfs_fsr3_stage_library_dir}/ffx_backend_vk_x64.lib")

        ExternalProject_Add(lfs_amd_fsr3_sdk
            PREFIX "${_lfs_fsr3_stage_root}/prefix"
            SOURCE_DIR "${_lfs_fsr3_stage_source}"
            BINARY_DIR "${_lfs_fsr3_stage_binary}"
            DOWNLOAD_COMMAND
                "${CMAKE_COMMAND}" -E rm -rf "${_lfs_fsr3_stage_source}"
            COMMAND
                "${CMAKE_COMMAND}" -E copy_directory
                "${_lfs_fsr3_root}/sdk" "${_lfs_fsr3_stage_source}"
            UPDATE_COMMAND ""
            PATCH_COMMAND ""
            CMAKE_GENERATOR "Visual Studio 17 2022"
            CMAKE_GENERATOR_PLATFORM x64
            CMAKE_ARGS
                -DFFX_API_BACKEND=VK_X64
                -DFFX_FSR3=ON
                -DFFX_FSR3UPSCALER=ON
                -DFFX_FI=OFF
                -DFFX_OF=OFF
                -DFFX_AUTO_COMPILE_SHADERS=ON
                -DFFX_BUILD_AS_DLL=OFF
            CMAKE_CACHE_ARGS
                -DVulkan_INCLUDE_DIR:PATH=${Vulkan_INCLUDE_DIR}
                -DVulkan_LIBRARY:FILEPATH=${Vulkan_LIBRARY}
            BUILD_COMMAND
                "${CMAKE_COMMAND}" --build "<BINARY_DIR>" --config Release
                --target ffx_fsr3upscaler_x64 ffx_backend_vk_x64 --parallel
            BUILD_BYPRODUCTS
                "${_lfs_fsr3_effect_library}"
                "${_lfs_fsr3_backend_library}"
            INSTALL_COMMAND ""
            USES_TERMINAL_CONFIGURE TRUE
            USES_TERMINAL_BUILD TRUE)
        set(_lfs_fsr3_build_target lfs_amd_fsr3_sdk)
        message(STATUS
            "AMD FidelityFX FSR 3.1: the user-provided SDK will be built automatically")
    else()
        set(_lfs_fsr3_platform_hint "")
        if(CMAKE_SYSTEM_NAME STREQUAL "Linux")
            set(_lfs_fsr3_platform_hint
                " The upstream v1.1.4 standalone generator is Visual Studio-oriented; "
                "provide Linux-built libraries through LFS_AMD_FSR3_LIBRARY_DIR.")
        endif()
        message(FATAL_ERROR
            "AMD FidelityFX SDK at '${_lfs_fsr3_root}' has no usable Vulkan FSR 3.1 "
            "libraries. Enable LFS_AMD_FSR3_BUILD_SDK on Windows, or set "
            "LFS_AMD_FSR3_LIBRARY_DIR to ffx_fsr3upscaler and ffx_backend_vk."
            "${_lfs_fsr3_platform_hint} Official download: ${_lfs_fsr3_download_url}")
    endif()
endif()

add_library(AMD::FidelityFXFsr3Upscaler STATIC IMPORTED GLOBAL)
set_target_properties(AMD::FidelityFXFsr3Upscaler PROPERTIES
    IMPORTED_LOCATION "${_lfs_fsr3_effect_library}"
    INTERFACE_INCLUDE_DIRECTORIES "${_lfs_fsr3_include}")
add_library(AMD::FidelityFXVkBackend STATIC IMPORTED GLOBAL)
set_target_properties(AMD::FidelityFXVkBackend PROPERTIES
    IMPORTED_LOCATION "${_lfs_fsr3_backend_library}"
    INTERFACE_INCLUDE_DIRECTORIES "${_lfs_fsr3_include}")

set(LFS_AMD_FSR3_ROOT_RESOLVED "${_lfs_fsr3_root}" CACHE INTERNAL
    "Validated AMD FidelityFX FSR 3.1 SDK root" FORCE)
set(LFS_AMD_FSR3_AVAILABLE ON CACHE INTERNAL
    "Whether the AMD FidelityFX FSR 3.1 Vulkan SDK is available to LichtFeld" FORCE)
set(LFS_AMD_FSR3_BUILD_TARGET "${_lfs_fsr3_build_target}" CACHE INTERNAL
    "Optional target that builds the user-provided FidelityFX SDK" FORCE)

message(STATUS
    "AMD FidelityFX FSR 3.1 Vulkan SDK: enabled from user-provided build ${_lfs_fsr3_root}")

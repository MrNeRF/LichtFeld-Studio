# SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
#
# SPDX-License-Identifier: GPL-3.0-or-later

include_guard(GLOBAL)

option(LFS_ENABLE_NVIDIA_DLSS
    "Compile the optional NVIDIA DLSS Vulkan adapter (requires a user-provided SDK)"
    OFF)
set(LFS_NVIDIA_DLSS_ROOT "" CACHE PATH
    "Path to an unpacked NVIDIA DLSS SDK; never downloaded automatically")
# NGX identifies a custom engine/application with one stable GUID. This is a
# public project identifier, not a credential and not a per-user value.
set(LFS_NVIDIA_DLSS_PROJECT_ID
    "7fc73d74-f126-4146-b028-4bc1026e5c3b"
    CACHE STRING "Stable NGX project ID used by LichtFeld Studio")
set(_lfs_dlss_download_url "https://developer.nvidia.com/rtx/dlss/get-started")

if(NOT LFS_ENABLE_NVIDIA_DLSS)
    set(LFS_NVIDIA_DLSS_AVAILABLE OFF CACHE INTERNAL
        "Whether the NVIDIA DLSS SDK is available to LichtFeld" FORCE)
    return()
endif()

if(NOT LFS_NVIDIA_DLSS_ROOT)
    message(FATAL_ERROR
        "LFS_ENABLE_NVIDIA_DLSS=ON requires LFS_NVIDIA_DLSS_ROOT to point to an "
        "SDK obtained and accepted by the person compiling LichtFeld. The SDK is "
        "not downloaded or distributed by this project. Official download: "
        "${_lfs_dlss_download_url}")
endif()
if(NOT LFS_NVIDIA_DLSS_PROJECT_ID MATCHES
        "^[0-9A-Fa-f][0-9A-Fa-f][0-9A-Fa-f][0-9A-Fa-f][0-9A-Fa-f][0-9A-Fa-f][0-9A-Fa-f][0-9A-Fa-f]-[0-9A-Fa-f][0-9A-Fa-f][0-9A-Fa-f][0-9A-Fa-f]-[0-9A-Fa-f][0-9A-Fa-f][0-9A-Fa-f][0-9A-Fa-f]-[0-9A-Fa-f][0-9A-Fa-f][0-9A-Fa-f][0-9A-Fa-f]-[0-9A-Fa-f][0-9A-Fa-f][0-9A-Fa-f][0-9A-Fa-f][0-9A-Fa-f][0-9A-Fa-f][0-9A-Fa-f][0-9A-Fa-f][0-9A-Fa-f][0-9A-Fa-f][0-9A-Fa-f][0-9A-Fa-f]$")
    message(FATAL_ERROR
        "LFS_NVIDIA_DLSS_PROJECT_ID must be a GUID-like NGX project ID, for example "
        "a0f57b54-1daf-4934-90ae-c4035c19df04.")
endif()

cmake_path(ABSOLUTE_PATH LFS_NVIDIA_DLSS_ROOT
    NORMALIZE
    BASE_DIRECTORY "${CMAKE_SOURCE_DIR}"
    OUTPUT_VARIABLE _lfs_dlss_root)
set(_lfs_dlss_include "${_lfs_dlss_root}/include")

foreach(_header IN ITEMS nvsdk_ngx.h nvsdk_ngx_vk.h nvsdk_ngx_helpers_vk.h)
    if(NOT EXISTS "${_lfs_dlss_include}/${_header}")
        message(FATAL_ERROR
            "NVIDIA DLSS SDK at '${_lfs_dlss_root}' is missing include/${_header}. "
            "Official download: ${_lfs_dlss_download_url}")
    endif()
endforeach()

if(NOT CMAKE_SIZEOF_VOID_P EQUAL 8)
    message(FATAL_ERROR "The NVIDIA DLSS adapter currently supports 64-bit builds only")
endif()

if(WIN32)
    set(_lfs_dlss_link_library
        "${_lfs_dlss_root}/lib/Windows_x86_64/x64/nvsdk_ngx_d.lib")
    set(_lfs_dlss_runtime
        "${_lfs_dlss_root}/lib/Windows_x86_64/rel/nvngx_dlss.dll")
elseif(CMAKE_SYSTEM_NAME STREQUAL "Linux")
    set(_lfs_dlss_link_library
        "${_lfs_dlss_root}/lib/Linux_x86_64/libnvsdk_ngx.a")
    file(GLOB _lfs_dlss_runtime_candidates
        "${_lfs_dlss_root}/lib/Linux_x86_64/rel/libnvidia-ngx-dlss.so.*")
    if(_lfs_dlss_runtime_candidates)
        list(SORT _lfs_dlss_runtime_candidates COMPARE NATURAL ORDER DESCENDING)
        list(GET _lfs_dlss_runtime_candidates 0 _lfs_dlss_runtime)
    endif()
else()
    message(FATAL_ERROR
        "The NVIDIA DLSS Vulkan adapter currently supports Windows and Linux only")
endif()

if(NOT EXISTS "${_lfs_dlss_link_library}")
    message(FATAL_ERROR
        "NVIDIA DLSS SDK at '${_lfs_dlss_root}' is missing its platform link library: "
        "${_lfs_dlss_link_library}. Official download: ${_lfs_dlss_download_url}")
endif()
if(NOT _lfs_dlss_runtime OR NOT EXISTS "${_lfs_dlss_runtime}")
    message(FATAL_ERROR
        "NVIDIA DLSS SDK at '${_lfs_dlss_root}' is missing its release runtime library. "
        "Official download: ${_lfs_dlss_download_url}")
endif()

add_library(NVIDIA::NGX STATIC IMPORTED GLOBAL)
set_target_properties(NVIDIA::NGX PROPERTIES
    IMPORTED_LOCATION "${_lfs_dlss_link_library}"
    INTERFACE_INCLUDE_DIRECTORIES "${_lfs_dlss_include}")

set(LFS_NVIDIA_DLSS_ROOT_RESOLVED "${_lfs_dlss_root}" CACHE INTERNAL
    "Validated NVIDIA DLSS SDK root" FORCE)
set(LFS_NVIDIA_DLSS_RUNTIME "${_lfs_dlss_runtime}" CACHE INTERNAL
    "NVIDIA DLSS runtime library copied only by opted-in builds" FORCE)
set(LFS_NVIDIA_DLSS_AVAILABLE ON CACHE INTERNAL
    "Whether the NVIDIA DLSS SDK is available to LichtFeld" FORCE)

message(STATUS "NVIDIA DLSS: enabled from user-provided SDK ${_lfs_dlss_root}")

function(lfs_stage_nvidia_dlss_runtime target)
    if(NOT TARGET "${target}")
        message(FATAL_ERROR
            "Cannot stage the NVIDIA DLSS runtime for missing target '${target}'")
    endif()
    add_custom_command(TARGET "${target}" POST_BUILD
        COMMAND "${CMAKE_COMMAND}" -E copy_if_different
            "${LFS_NVIDIA_DLSS_RUNTIME}"
            "$<TARGET_FILE_DIR:${target}>"
        COMMENT "Staging the user-provided NVIDIA DLSS runtime")
endfunction()

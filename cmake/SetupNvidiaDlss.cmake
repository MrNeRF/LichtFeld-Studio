# SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
#
# SPDX-License-Identifier: GPL-3.0-or-later

include_guard(GLOBAL)

option(LFS_ENABLE_NVIDIA_DLSS
    "Build the optional external NVIDIA DLSS scene-reconstruction plugin"
    OFF)
set(LFS_NVIDIA_DLSS_ROOT "" CACHE PATH
    "Path to an NVIDIA DLSS SDK checkout; the SDK is never fetched by CMake")

# Portable packages have a controlled dependency checkout and always include
# the optional plugin. Ordinary developer builds remain opt-in.
if(BUILD_PORTABLE)
    set(LFS_ENABLE_NVIDIA_DLSS ON CACHE BOOL
        "Build the optional external NVIDIA DLSS scene-reconstruction plugin" FORCE)
    if(NOT LFS_NVIDIA_DLSS_ROOT)
        set(LFS_NVIDIA_DLSS_ROOT "${CMAKE_SOURCE_DIR}/external/nvidia-dlss-sdk"
            CACHE PATH "Path to an NVIDIA DLSS SDK checkout" FORCE)
    endif()
endif()

if(NOT LFS_ENABLE_NVIDIA_DLSS)
    set(LFS_NVIDIA_DLSS_AVAILABLE OFF CACHE INTERNAL
        "Whether the external NVIDIA DLSS plugin can be built" FORCE)
    return()
endif()

if(NOT LFS_NVIDIA_DLSS_ROOT)
    message(FATAL_ERROR
        "LFS_ENABLE_NVIDIA_DLSS=ON requires LFS_NVIDIA_DLSS_ROOT to point to an "
        "SDK obtained separately from NVIDIA. CMake does not download or accept "
        "the NVIDIA SDK license on the user's behalf.")
endif()

cmake_path(ABSOLUTE_PATH LFS_NVIDIA_DLSS_ROOT
    NORMALIZE
    BASE_DIRECTORY "${CMAKE_SOURCE_DIR}"
    OUTPUT_VARIABLE _lfs_dlss_root)
set(_lfs_dlss_include "${_lfs_dlss_root}/include")

foreach(_header IN ITEMS
        nvsdk_ngx.h
        nvsdk_ngx_defs.h
        nvsdk_ngx_helpers.h
        nvsdk_ngx_helpers_vk.h
        nvsdk_ngx_vk.h)
    if(NOT EXISTS "${_lfs_dlss_include}/${_header}")
        message(FATAL_ERROR
            "NVIDIA DLSS SDK at '${_lfs_dlss_root}' is missing include/${_header}")
    endif()
endforeach()

if(NOT CMAKE_SIZEOF_VOID_P EQUAL 8)
    message(FATAL_ERROR "The NVIDIA DLSS plugin currently supports 64-bit builds only")
endif()

if(WIN32)
    set(_lfs_dlss_link_release
        "${_lfs_dlss_root}/lib/Windows_x86_64/x64/nvsdk_ngx_d.lib")
    set(_lfs_dlss_link_debug
        "${_lfs_dlss_root}/lib/Windows_x86_64/x64/nvsdk_ngx_d_dbg.lib")
    set(_lfs_dlss_runtime_release
        "${_lfs_dlss_root}/lib/Windows_x86_64/rel/nvngx_dlss.dll")
    set(_lfs_dlss_runtime_debug
        "${_lfs_dlss_root}/lib/Windows_x86_64/dev/nvngx_dlss.dll")
elseif(CMAKE_SYSTEM_NAME STREQUAL "Linux")
    set(_lfs_dlss_link_release
        "${_lfs_dlss_root}/lib/Linux_x86_64/libnvsdk_ngx.a")
    set(_lfs_dlss_link_debug "${_lfs_dlss_link_release}")
    file(GLOB _lfs_dlss_runtime_release_candidates
        "${_lfs_dlss_root}/lib/Linux_x86_64/rel/libnvidia-ngx-dlss.so.*")
    file(GLOB _lfs_dlss_runtime_debug_candidates
        "${_lfs_dlss_root}/lib/Linux_x86_64/dev/libnvidia-ngx-dlss.so.*")
    list(SORT _lfs_dlss_runtime_release_candidates COMPARE NATURAL ORDER DESCENDING)
    list(SORT _lfs_dlss_runtime_debug_candidates COMPARE NATURAL ORDER DESCENDING)
    if(_lfs_dlss_runtime_release_candidates)
        list(GET _lfs_dlss_runtime_release_candidates 0 _lfs_dlss_runtime_release)
    endif()
    if(_lfs_dlss_runtime_debug_candidates)
        list(GET _lfs_dlss_runtime_debug_candidates 0 _lfs_dlss_runtime_debug)
    endif()
else()
    message(FATAL_ERROR
        "The NVIDIA DLSS Vulkan plugin currently supports Windows and Linux only")
endif()

foreach(_required IN ITEMS
        _lfs_dlss_link_release
        _lfs_dlss_link_debug
        _lfs_dlss_runtime_release
        _lfs_dlss_runtime_debug)
    if(NOT DEFINED ${_required} OR NOT EXISTS "${${_required}}")
        message(FATAL_ERROR
            "NVIDIA DLSS SDK at '${_lfs_dlss_root}' is missing ${_required}: "
            "'${${_required}}'")
    endif()
endforeach()

add_library(NVIDIA::NGX UNKNOWN IMPORTED GLOBAL)
set_target_properties(NVIDIA::NGX PROPERTIES
    IMPORTED_CONFIGURATIONS "DEBUG;RELEASE;RELWITHDEBINFO;MINSIZEREL"
    IMPORTED_LOCATION_DEBUG "${_lfs_dlss_link_debug}"
    IMPORTED_LOCATION_RELEASE "${_lfs_dlss_link_release}"
    IMPORTED_LOCATION_RELWITHDEBINFO "${_lfs_dlss_link_release}"
    IMPORTED_LOCATION_MINSIZEREL "${_lfs_dlss_link_release}"
    INTERFACE_INCLUDE_DIRECTORIES "${_lfs_dlss_include}")

set(LFS_NVIDIA_DLSS_ROOT_RESOLVED "${_lfs_dlss_root}" CACHE INTERNAL
    "Validated NVIDIA DLSS SDK root" FORCE)
set(LFS_NVIDIA_DLSS_RUNTIME_RELEASE "${_lfs_dlss_runtime_release}" CACHE INTERNAL
    "NVIDIA DLSS release runtime" FORCE)
set(LFS_NVIDIA_DLSS_RUNTIME_DEBUG "${_lfs_dlss_runtime_debug}" CACHE INTERNAL
    "NVIDIA DLSS development runtime" FORCE)
set(LFS_NVIDIA_DLSS_AVAILABLE ON CACHE INTERNAL
    "Whether the external NVIDIA DLSS plugin can be built" FORCE)

message(STATUS
    "NVIDIA DLSS: external plugin enabled from user-provided SDK ${_lfs_dlss_root}")


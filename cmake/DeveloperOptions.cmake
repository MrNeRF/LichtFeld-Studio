# SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
#
# SPDX-License-Identifier: GPL-3.0-or-later

include_guard(GLOBAL)

set(_lfs_developer_default OFF)
set(_lfs_cuda_device_debug_default OFF)
if(NOT CMAKE_CONFIGURATION_TYPES)
    if(CMAKE_BUILD_TYPE MATCHES "^(Debug|RelWithDebInfo)$")
        set(_lfs_developer_default ON)
    endif()
    if(CMAKE_BUILD_TYPE STREQUAL "Debug")
        set(_lfs_cuda_device_debug_default ON)
    endif()
endif()

option(LFS_ENABLE_VULKAN_VALIDATION
    "Request Vulkan validation by default; LFS_VK_VALIDATION can override it at runtime"
    ${_lfs_developer_default})
option(LFS_ENABLE_VULKAN_SHADER_DEBUG_INFO
    "Embed source-level debug information in Vulkan SPIR-V"
    ${_lfs_developer_default})
option(LFS_ENABLE_CUDA_DEVICE_DEBUG
    "Compile Debug CUDA device code with -G"
    ${_lfs_cuda_device_debug_default})
option(LFS_ENABLE_CUDA_FAILURE_INJECTION
    "Compile the CUDA allocation-failure injection API used by tests"
    ${BUILD_TESTS})
option(LFS_ENABLE_ALLOCATION_PROFILING
    "Compile tensor allocation stack profiling"
    OFF)

option(LFS_DEV_IMPORT_SOURCE_PYTHON
    "Import built-in Python packages from src/python in non-portable builds"
    ON)
option(LFS_DEV_IMPORT_SOURCE_RESOURCES
    "Load RmlUI resources and locales from source in non-portable builds"
    ON)

unset(_lfs_developer_default)
unset(_lfs_cuda_device_debug_default)

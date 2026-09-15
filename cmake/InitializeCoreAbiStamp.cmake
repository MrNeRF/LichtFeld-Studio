# SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
# SPDX-License-Identifier: GPL-3.0-or-later

function(lfs_initialize_core_abi_stamps)
    # A dedicated directory also isolates an unnamed single-config build from
    # the legacy shared header on lfs_core's public include path.
    set(_lfs_include_root "${CMAKE_BINARY_DIR}/include/core-abi")
    file(REMOVE "${CMAKE_BINARY_DIR}/include/lfs_core_abi_stamp.h")

    set(LFS_SOURCE_DIR "${CMAKE_SOURCE_DIR}")
    set(LFS_CORE_ABI_TEMPLATE "${CMAKE_CURRENT_FUNCTION_LIST_DIR}/lfs_core_abi_stamp.h.in")
    set(LFS_PROJECT_VERSION "${PROJECT_VERSION}")
    set(LFS_CONFIGURED_GIT_COMMIT_HASH_SHORT "${GIT_COMMIT_HASH_SHORT}")
    set(LFS_SYSTEM_NAME "${CMAKE_SYSTEM_NAME}")
    set(LFS_SIZEOF_VOID_P "${CMAKE_SIZEOF_VOID_P}")
    set(LFS_CXX_COMPILER_ID "${CMAKE_CXX_COMPILER_ID}")
    set(LFS_CXX_COMPILER_VERSION "${CMAKE_CXX_COMPILER_VERSION}")
    set(LFS_CUDA_COMPILER_ID "${CMAKE_CUDA_COMPILER_ID}")
    set(LFS_CUDA_COMPILER_VERSION "${CMAKE_CUDA_COMPILER_VERSION}")
    set(LFS_CUDA_ARCHITECTURES "${CMAKE_CUDA_ARCHITECTURES}")
    set(LFS_VCPKG_TARGET_TRIPLET "${VCPKG_TARGET_TRIPLET}")

    # Dependencies can populate CMAKE_CONFIGURATION_TYPES even for Ninja.
    # Only the generator property determines whether these are actual configs.
    get_property(_lfs_multi_config GLOBAL PROPERTY GENERATOR_IS_MULTI_CONFIG)
    if(_lfs_multi_config)
        foreach(LFS_BUILD_CONFIG IN LISTS CMAKE_CONFIGURATION_TYPES)
            set(LFS_CORE_ABI_HEADER "${_lfs_include_root}/${LFS_BUILD_CONFIG}/lfs_core_abi_stamp.h")
            include("${CMAKE_CURRENT_FUNCTION_LIST_DIR}/GenerateCoreAbiStamp.cmake")
        endforeach()
    else()
        set(LFS_BUILD_CONFIG "${CMAKE_BUILD_TYPE}")
        set(LFS_CORE_ABI_HEADER "${_lfs_include_root}/${LFS_BUILD_CONFIG}/lfs_core_abi_stamp.h")
        include("${CMAKE_CURRENT_FUNCTION_LIST_DIR}/GenerateCoreAbiStamp.cmake")
    endif()

    set(LFS_CORE_ABI_INCLUDE_DIR "${_lfs_include_root}/$<CONFIG>" PARENT_SCOPE)
    set(LFS_CORE_ABI_HEADER "${_lfs_include_root}/$<CONFIG>/lfs_core_abi_stamp.h" PARENT_SCOPE)
endfunction()

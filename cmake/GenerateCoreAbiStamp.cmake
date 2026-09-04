# SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
# SPDX-License-Identifier: GPL-3.0-or-later

foreach(_lfs_required_variable IN ITEMS
        LFS_SOURCE_DIR
        LFS_CORE_ABI_TEMPLATE
        LFS_CORE_ABI_HEADER
        LFS_PROJECT_VERSION
        LFS_BUILD_CONFIG
        LFS_SYSTEM_NAME
        LFS_SIZEOF_VOID_P
        LFS_CXX_COMPILER_ID
        LFS_CXX_COMPILER_VERSION
        LFS_CUDA_COMPILER_ID
        LFS_CUDA_COMPILER_VERSION)
    if(NOT DEFINED ${_lfs_required_variable})
        message(FATAL_ERROR "${_lfs_required_variable} is required")
    endif()
endforeach()

if(NOT IS_DIRECTORY "${LFS_SOURCE_DIR}/src/core")
    message(FATAL_ERROR "LFS core source directory does not exist: ${LFS_SOURCE_DIR}/src/core")
endif()
if(NOT EXISTS "${LFS_CORE_ABI_TEMPLATE}")
    message(FATAL_ERROR "LFS core ABI template does not exist: ${LFS_CORE_ABI_TEMPLATE}")
endif()

# This scan intentionally happens at build time. Source/header content changes
# must update the ABI tripwire without making CMake regenerate the whole graph.
file(GLOB_RECURSE LFS_CORE_BUILD_INPUTS LIST_DIRECTORIES FALSE
    "${LFS_SOURCE_DIR}/src/core/*.cpp"
    "${LFS_SOURCE_DIR}/src/core/*.cu"
    "${LFS_SOURCE_DIR}/src/core/*.cuh"
    "${LFS_SOURCE_DIR}/src/core/*.hpp"
)
list(APPEND LFS_CORE_BUILD_INPUTS
    "${LFS_SOURCE_DIR}/src/core/CMakeLists.txt"
)
list(SORT LFS_CORE_BUILD_INPUTS)

execute_process(
    COMMAND git rev-parse --short HEAD
    WORKING_DIRECTORY "${LFS_SOURCE_DIR}"
    OUTPUT_VARIABLE _lfs_git_commit_hash_short
    OUTPUT_STRIP_TRAILING_WHITESPACE
    ERROR_QUIET
)
if(NOT _lfs_git_commit_hash_short AND
   DEFINED LFS_CONFIGURED_GIT_COMMIT_HASH_SHORT AND
   NOT LFS_CONFIGURED_GIT_COMMIT_HASH_SHORT STREQUAL "" AND
   NOT LFS_CONFIGURED_GIT_COMMIT_HASH_SHORT STREQUAL "unknown")
    set(_lfs_git_commit_hash_short "${LFS_CONFIGURED_GIT_COMMIT_HASH_SHORT}")
endif()
if(NOT _lfs_git_commit_hash_short AND
   DEFINED ENV{GITHUB_SHA} AND NOT "$ENV{GITHUB_SHA}" STREQUAL "")
    string(SUBSTRING "$ENV{GITHUB_SHA}" 0 7 _lfs_git_commit_hash_short)
endif()
if(NOT _lfs_git_commit_hash_short)
    set(_lfs_git_commit_hash_short "unknown")
endif()

set(_lfs_core_abi_material
    "${LFS_PROJECT_VERSION}|${_lfs_git_commit_hash_short}|${LFS_BUILD_CONFIG}|${LFS_SYSTEM_NAME}|${LFS_SIZEOF_VOID_P}|${LFS_CXX_COMPILER_ID}|${LFS_CXX_COMPILER_VERSION}|${LFS_CUDA_COMPILER_ID}|${LFS_CUDA_COMPILER_VERSION}|${LFS_CUDA_ARCHITECTURES}|${LFS_VCPKG_TARGET_TRIPLET}")
foreach(_lfs_core_build_input IN LISTS LFS_CORE_BUILD_INPUTS)
    file(SHA256 "${_lfs_core_build_input}" _lfs_core_build_input_hash)
    string(APPEND _lfs_core_abi_material "|${_lfs_core_build_input_hash}")
endforeach()

string(SHA256 LFS_CORE_ABI_STAMP "${_lfs_core_abi_material}")
get_filename_component(_lfs_core_abi_header_dir "${LFS_CORE_ABI_HEADER}" DIRECTORY)
file(MAKE_DIRECTORY "${_lfs_core_abi_header_dir}")

# configure_file only updates the timestamp when the generated content changes.
# That keeps no-op builds from recompiling the ABI consumers.
configure_file(
    "${LFS_CORE_ABI_TEMPLATE}"
    "${LFS_CORE_ABI_HEADER}"
    @ONLY
)

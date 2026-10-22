# SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
# SPDX-License-Identifier: GPL-3.0-or-later
cmake_minimum_required(VERSION 3.30)
project(HipProfilePolicy LANGUAGES NONE)

# Exercise the real policy without an SDK or kernel compilation. In particular,
# the caller must observe the effective packing flag, rather than a stale cache.
add_custom_target(lfs_hip_fast_kernels)
include("${CMAKE_CURRENT_LIST_DIR}/../../../cmake/HipFastKernels.cmake")
lfs_hip_fast_objects(_include _objects)
get_cmake_property(_variables VARIABLES)
foreach(_name IN LISTS _variables)
    if(_name MATCHES "^LFS_HIP_FAST_" AND NOT _name STREQUAL "LFS_HIP_FAST_RUNTIME")
        if(LFS_HIP_VALIDATED_TUNING)
            if(NOT ${_name})
                message(FATAL_ERROR "Expected enabled profile option: ${_name}")
            endif()
        elseif(${_name})
            message(FATAL_ERROR "Generic caller retained tuned option: ${_name}")
        endif()
    endif()
endforeach()

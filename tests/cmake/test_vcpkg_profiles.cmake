# SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
#
# SPDX-License-Identifier: GPL-3.0-or-later

# Script-mode checks only: no project(), compiler detection or vcpkg execution.
cmake_minimum_required(VERSION 3.30)
get_filename_component(_project_root "${CMAKE_CURRENT_LIST_DIR}/../.." ABSOLUTE)
include("${_project_root}/cmake/VcpkgProfiles.cmake")

if(DEFINED LFS_PROFILE_TEST_CASE)
    set(VCPKG_TARGET_TRIPLET "x64-${LFS_PROFILE_TEST_PLATFORM}")
    set(VCPKG_HOST_TRIPLET "${VCPKG_TARGET_TRIPLET}")
    set(VCPKG_OVERLAY_TRIPLETS "${_project_root}/cmake/triplets/release-only")
    set(CMAKE_BUILD_TYPE "Release")
    set(CMAKE_CONFIGURATION_TYPES "")

    if(LFS_PROFILE_TEST_CASE STREQUAL "relwithdebinfo")
        set(CMAKE_BUILD_TYPE "RelWithDebInfo")
    elseif(LFS_PROFILE_TEST_CASE MATCHES "^(ninja-cached-configs|relwithdebinfo-cached-configs)$")
        # OpenMesh writes this cache entry even with single-config Ninja.
        unset(CMAKE_CONFIGURATION_TYPES)
        set(CMAKE_CONFIGURATION_TYPES "Debug;Release;RelWithDebInfo" CACHE STRING "")
        if(LFS_PROFILE_TEST_CASE STREQUAL "relwithdebinfo-cached-configs")
            set(CMAKE_BUILD_TYPE "RelWithDebInfo")
        endif()
    elseif(LFS_PROFILE_TEST_CASE STREQUAL "minsizerel")
        set(CMAKE_BUILD_TYPE "MinSizeRel")
    elseif(LFS_PROFILE_TEST_CASE STREQUAL "debug")
        set(CMAKE_BUILD_TYPE "Debug")
    elseif(LFS_PROFILE_TEST_CASE STREQUAL "unknown")
        set(CMAKE_BUILD_TYPE "Profile")
    elseif(LFS_PROFILE_TEST_CASE STREQUAL "multi")
        set(CMAKE_CONFIGURATION_TYPES "Debug;Release")
    elseif(LFS_PROFILE_TEST_CASE STREQUAL "multi-empty-configs")
        # A fresh multi-config tree has no configuration list before project().
        unset(CMAKE_CONFIGURATION_TYPES)
    elseif(LFS_PROFILE_TEST_CASE STREQUAL "host-mismatch")
        set(VCPKG_HOST_TRIPLET "arm64-linux")
    elseif(LFS_PROFILE_TEST_CASE STREQUAL "custom-triplet")
        set(VCPKG_TARGET_TRIPLET "custom-release")
        set(VCPKG_HOST_TRIPLET "custom-release")
    elseif(LFS_PROFILE_TEST_CASE STREQUAL "standard-debug")
        set(CMAKE_BUILD_TYPE "Debug")
        unset(VCPKG_OVERLAY_TRIPLETS)
    elseif(LFS_PROFILE_TEST_CASE STREQUAL "standard-multi")
        set(CMAKE_CONFIGURATION_TYPES "Debug;Release")
        unset(VCPKG_OVERLAY_TRIPLETS)
    elseif(LFS_PROFILE_TEST_CASE STREQUAL "user-overlay")
        set(VCPKG_TARGET_TRIPLET "custom-debug")
        set(VCPKG_HOST_TRIPLET "custom-debug")
        set(CMAKE_BUILD_TYPE "Debug")
        set(VCPKG_OVERLAY_TRIPLETS "${_project_root}/custom-triplets")
    elseif(LFS_PROFILE_TEST_CASE STREQUAL "normalized-path")
        set(VCPKG_OVERLAY_TRIPLETS
            "${_project_root}/custom-triplets;${_project_root}/cmake/../cmake/triplets/release-only/")
    elseif(NOT LFS_PROFILE_TEST_CASE STREQUAL "release")
        message(FATAL_ERROR "Unknown test case: ${LFS_PROFILE_TEST_CASE}")
    endif()

    set(_original_overlays "${VCPKG_OVERLAY_TRIPLETS}")
    lfs_validate_vcpkg_profile()
    if(NOT "${VCPKG_OVERLAY_TRIPLETS}" STREQUAL "${_original_overlays}")
        message(FATAL_ERROR "Validation changed the user's overlay selection")
    endif()
    if(LFS_PROFILE_TEST_CASE MATCHES "^(release|relwithdebinfo|minsizerel|normalized-path|ninja-cached-configs|relwithdebinfo-cached-configs)$")
        get_property(_configure_deps DIRECTORY PROPERTY CMAKE_CONFIGURE_DEPENDS)
        set(_triplet "${_project_root}/cmake/triplets/release-only/${VCPKG_TARGET_TRIPLET}.cmake")
        if(NOT _triplet IN_LIST _configure_deps)
            message(FATAL_ERROR "Active triplet edits would not trigger regeneration")
        endif()
    endif()
    return()
endif()

set(_total_cases 0)
foreach(_platform windows linux)
    foreach(_case release relwithdebinfo ninja-cached-configs relwithdebinfo-cached-configs
                  minsizerel normalized-path standard-debug standard-multi user-overlay
                  debug unknown multi multi-empty-configs host-mismatch custom-triplet)
        # -G selects the real generator property in script mode without
        # configuring a project, finding Ninja or invoking any compiler.
        if(_case MATCHES "^(multi|multi-empty-configs|standard-multi)$")
            set(_generator "Ninja Multi-Config")
        else()
            set(_generator "Ninja")
        endif()
        execute_process(
            COMMAND "${CMAKE_COMMAND}"
                -G "${_generator}"
                "-DLFS_PROFILE_TEST_CASE=${_case}"
                "-DLFS_PROFILE_TEST_PLATFORM=${_platform}"
                -P "${CMAKE_CURRENT_LIST_FILE}"
            RESULT_VARIABLE _result
            OUTPUT_VARIABLE _stdout
            ERROR_VARIABLE _stderr)
        if(_case MATCHES "^(debug|unknown|multi|multi-empty-configs)$")
            set(_expected_error "requires a single-config")
        elseif(_case MATCHES "^(host-mismatch|custom-triplet)$")
            set(_expected_error "requires matching native")
        else()
            set(_expected_error "")
        endif()
        if(_expected_error)
            if(_result EQUAL 0 OR NOT _stderr MATCHES "${_expected_error}")
                message(FATAL_ERROR "${_platform}/${_case}: expected profile rejection\n${_stdout}${_stderr}")
            endif()
        elseif(NOT _result EQUAL 0)
            message(FATAL_ERROR "${_platform}/${_case}: ${_stdout}${_stderr}")
        endif()
        math(EXPR _total_cases "${_total_cases} + 1")
    endforeach()
endforeach()
message(STATUS "vcpkg profile checks passed (${_total_cases} cases; no configure or build)")

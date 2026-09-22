# SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
# SPDX-License-Identifier: GPL-3.0-or-later

include_guard(GLOBAL)

function(lfs_add_windows_version_resource target description)
    if(NOT WIN32)
        return()
    endif()

    get_target_property(target_type ${target} TYPE)
    if(target_type STREQUAL "EXECUTABLE")
        set(LFS_WINDOWS_FILE_TYPE 0x1L) # VFT_APP
    elseif(target_type STREQUAL "SHARED_LIBRARY" OR target_type STREQUAL "MODULE_LIBRARY")
        set(LFS_WINDOWS_FILE_TYPE 0x2L) # VFT_DLL, including Python extension modules
    else()
        message(FATAL_ERROR "Windows version resources require an executable or DLL: ${target}")
    endif()

    set(LFS_WINDOWS_VERSION_TWEAK 0)
    if(NOT "${PROJECT_VERSION_TWEAK}" STREQUAL "")
        set(LFS_WINDOWS_VERSION_TWEAK "${PROJECT_VERSION_TWEAK}")
    endif()
    set(LFS_WINDOWS_FILE_DESCRIPTION "${description}")
    set(LFS_WINDOWS_INTERNAL_NAME "$<TARGET_FILE_BASE_NAME:${target}>")
    set(LFS_WINDOWS_ORIGINAL_FILENAME "$<TARGET_FILE_NAME:${target}>")

    set(resource_dir "${CMAKE_CURRENT_BINARY_DIR}/windows-resources")
    configure_file(
        "${CMAKE_CURRENT_FUNCTION_LIST_DIR}/windows_version.rc.in"
        "${resource_dir}/${target}.rc.in"
        @ONLY
    )
    # Resolve configuration-specific filenames after all target properties are
    # known, including the Debug suffix and nanobind's Python ABI suffix.
    set(resource "${resource_dir}/$<CONFIG>/${target}.rc")
    file(GENERATE OUTPUT "${resource}" INPUT "${resource_dir}/${target}.rc.in" TARGET ${target})
    target_sources(${target} PRIVATE "${resource}")
endfunction()

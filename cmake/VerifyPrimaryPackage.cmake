# SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
# SPDX-License-Identifier: GPL-3.0-or-later

# A primary package may contain the in-tree AMD module, but must not pick up
# separately installed provider folders from a reused distribution prefix.
if(NOT DEFINED CMAKE_INSTALL_PREFIX OR CMAKE_INSTALL_PREFIX STREQUAL "")
    message(FATAL_ERROR "A primary package install prefix is required")
endif()
set(_lfs_primary_prefix "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}")
foreach(_relative IN ITEMS scene_upscalers bin/scene_upscalers lib/scene_upscalers)
    file(GLOB _lfs_provider_entries LIST_DIRECTORIES TRUE "${_lfs_primary_prefix}/${_relative}/*")
    foreach(_entry IN LISTS _lfs_provider_entries)
        cmake_path(GET _entry FILENAME _name)
        if(NOT _name STREQUAL "amd")
            message(FATAL_ERROR "Primary packages must omit externally installed providers. Use a fresh install prefix; '${_entry}' already exists.")
        endif()
    endforeach()
endforeach()

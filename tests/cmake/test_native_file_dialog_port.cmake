# SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
#
# SPDX-License-Identifier: GPL-3.0-or-later

# Script-mode contract check only: no configure, compiler, or vcpkg invocation.
cmake_minimum_required(VERSION 3.20)
get_filename_component(_project_root "${CMAKE_CURRENT_LIST_DIR}/../.." ABSOLUTE)

file(READ "${_project_root}/vcpkg-configuration.json" _configuration)
string(JSON _overlay_port_root GET "${_configuration}" overlay-ports 0)
if(NOT _overlay_port_root STREQUAL ".github/overlays")
    message(FATAL_ERROR "The repository overlay-port root changed unexpectedly")
endif()

set(_port_root
    "${_project_root}/${_overlay_port_root}/nativefiledialog-extended")
if(NOT EXISTS "${_port_root}/vcpkg.json" OR
   NOT EXISTS "${_port_root}/portfile.cmake")
    message(FATAL_ERROR "The native file dialog overlay port is incomplete")
endif()

file(READ "${_port_root}/vcpkg.json" _manifest)
string(JSON _port_name GET "${_manifest}" name)
string(JSON _port_version GET "${_manifest}" version)
if(NOT _port_name STREQUAL "nativefiledialog-extended" OR
   NOT _port_version STREQUAL "1.3.0")
    message(FATAL_ERROR "The native file dialog overlay does not match the pinned port")
endif()

file(READ "${_port_root}/portfile.cmake" _portfile)
if(NOT _portfile MATCHES
   "if\\(VCPKG_TARGET_IS_WINDOWS\\)(.|\n)*NFD_OVERRIDE_RECENT_WITH_DEFAULT=ON(.|\n)*endif\\(\\)")
    message(FATAL_ERROR
        "Windows dialogs must prefer an explicitly supplied default directory")
endif()

message(STATUS "native file dialog default-directory contract passed")

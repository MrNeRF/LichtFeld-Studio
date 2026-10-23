# SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
# SPDX-License-Identifier: GPL-3.0-or-later
cmake_minimum_required(VERSION 3.30)
project(HipPackageCommands LANGUAGES NONE)
set(CMAKE_INSTALL_BINDIR bin)
set(CMAKE_INSTALL_DATADIR share)
set(LFS_HIP_RUNTIME ON)
set(LFS_HIP_ROOT "${TEST_SDK}")
set(LFS_HIP_RUNTIME_LIBRARY "${TEST_SDK}/bin/amdhip64_fixture.dll")
add_custom_target(lfs_hip_runtime_build)
add_custom_target(HipPackageCommands)
include("${CMAKE_CURRENT_LIST_DIR}/../../../../cmake/HipPackaging.cmake")

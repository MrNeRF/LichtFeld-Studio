# SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
#
# SPDX-License-Identifier: GPL-3.0-or-later

# Standalone build for the LOD splat module + profiler.
# Deliberately independent of the root LichtFeld-Studio build, which requires
# CUDA; this module is portable C++20 and runs anywhere (incl. macOS).
#
#   cmake -S src/lod_splat -B build-lod -DCMAKE_BUILD_TYPE=Release
#   cmake --build build-lod -j
#   ./build-lod/lod_profile --validate
cmake_minimum_required(VERSION 3.20)
project(lfs_lod_splat LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

add_library(lfs_lod STATIC
    bhatt_ref.cpp
    hierarchy.cpp
    cut.cpp
    streaming.cpp
    rasterizer.cpp
    ply_io.cpp
    synthetic.cpp
)
find_package(Threads REQUIRED)
target_link_libraries(lfs_lod PUBLIC Threads::Threads)

add_executable(lod_profile profile_main.cpp)
target_link_libraries(lod_profile PRIVATE lfs_lod)

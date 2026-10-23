# SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
# SPDX-License-Identifier: GPL-3.0-or-later
cmake_minimum_required(VERSION 3.30)
project(HipModulePolicy LANGUAGES CXX)
set(CMAKE_CXX_SCAN_FOR_MODULES OFF)
add_executable(hip_module_policy module_policy.cpp)
target_include_directories(hip_module_policy PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/../../../src/core/include")
target_compile_features(hip_module_policy PRIVATE cxx_std_20)
enable_testing()
add_test(NAME hip_module_policy COMMAND hip_module_policy "${CMAKE_CURRENT_BINARY_DIR}/fixture")

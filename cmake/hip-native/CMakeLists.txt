# SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
# SPDX-License-Identifier: GPL-3.0-or-later
cmake_minimum_required(VERSION 3.30)
project(LichtFeldHipNative LANGUAGES NONE)
# Small SDK-only build for cross-target compilation checks. No host application,
# CUDA toolkit, Torch, display server, or AMD GPU is needed to compile it.
get_filename_component(_root "${CMAKE_CURRENT_LIST_DIR}/../.." ABSOLUTE)
set(LFS_TENSOR_HIP OFF)
add_subdirectory("${_root}/src/core/tensor/backend/hip" runtime)
add_subdirectory("${_root}/src/training/hip" training)

# SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
# SPDX-License-Identifier: GPL-3.0-or-later
cmake_minimum_required(VERSION 3.30)
if(NOT LFS_HIP_RUNTIME_FILES OR NOT LFS_HIP_RUNTIME_DEST)
    message(FATAL_ERROR "HIP runtime staging requires files and a destination")
endif()
# Preserve Linux soname links instead of copying the full library once per alias.
file(COPY ${LFS_HIP_RUNTIME_FILES} DESTINATION "${LFS_HIP_RUNTIME_DEST}" FOLLOW_SYMLINK_CHAIN)

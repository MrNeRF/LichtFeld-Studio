# SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
#
# SPDX-License-Identifier: GPL-3.0-or-later

# Shared per-camera depth-anchor fit + sidecar cache, used by both the trainer
# (startup fit / write-through cache) and the preprocess CLI (offline precompute).
add_library(lfs_depth_anchor STATIC
    depth_anchor_cache.cpp
    depth_anchor_cache.hpp
    depth_anchor_fit.cpp
)

target_include_directories(lfs_depth_anchor
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
        $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/src/training/include>
)

target_include_directories(lfs_depth_anchor SYSTEM PRIVATE
    ${CMAKE_SOURCE_DIR}/external         # For nlohmann/json.hpp
)

target_link_libraries(lfs_depth_anchor
    PUBLIC
        lfs_core
    PRIVATE
        lfs_io                           # ScopedAtomicOutputFile
)

target_compile_features(lfs_depth_anchor PUBLIC cxx_std_23)

set_target_properties(lfs_depth_anchor PROPERTIES
    CXX_STANDARD 23
    CXX_STANDARD_REQUIRED ON
    POSITION_INDEPENDENT_CODE ON
)

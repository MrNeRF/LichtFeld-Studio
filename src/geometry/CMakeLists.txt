# SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
# SPDX-License-Identifier: GPL-3.0-or-later

# geometry module - LibTorch-free geometry library

set(GEOMETRY_SOURCES
    euclidean_transform.cpp
    bounding_box.cpp
    geometry_misc.cpp
)

add_library(lfs_geometry STATIC ${GEOMETRY_SOURCES})

target_include_directories(lfs_geometry
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/include
        ${CMAKE_BINARY_DIR}/include
)

target_link_libraries(lfs_geometry
    PUBLIC
        glm::glm
)

set_target_properties(lfs_geometry PROPERTIES
    CXX_STANDARD 23
    CXX_STANDARD_REQUIRED ON
    POSITION_INDEPENDENT_CODE ON
    DEBUG_POSTFIX d
)

message(STATUS "╔════════════════════════════════════════════════════════════════╗")
message(STATUS "║  geometry - LibTorch-free Geometry Module                 ║")
message(STATUS "╚════════════════════════════════════════════════════════════════╝")
message(STATUS "  • Namespace: lfs::geometry")
message(STATUS "  • Contains: EuclideanTransform, BoundingBox")

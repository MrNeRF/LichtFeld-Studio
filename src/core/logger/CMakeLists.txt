# SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
# SPDX-License-Identifier: GPL-3.0-or-later

# Logger SHARED library - leaf dependency with zero upstream deps.
# Extracted from lfs_core to break the lfs_core ↔ lfs_event_bridge cycle.

add_library(lfs_logger SHARED
    ../logger.cpp
)

target_include_directories(lfs_logger
    PUBLIC
        ${CMAKE_SOURCE_DIR}/src/core/include
)

target_link_libraries(lfs_logger
    PRIVATE
        spdlog::spdlog
)

target_compile_definitions(lfs_logger
    PRIVATE
        LFS_LOGGER_EXPORTS
)

set_target_properties(lfs_logger PROPERTIES
    POSITION_INDEPENDENT_CODE ON
    CXX_STANDARD 23
    CXX_STANDARD_REQUIRED ON
    CXX_VISIBILITY_PRESET hidden
    VISIBILITY_INLINES_HIDDEN ON
    LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}
    RUNTIME_OUTPUT_DIRECTORY_DEBUG        ${CMAKE_BINARY_DIR}/Debug
    RUNTIME_OUTPUT_DIRECTORY_RELEASE      ${CMAKE_BINARY_DIR}/Release
    RUNTIME_OUTPUT_DIRECTORY_RELWITHDEBINFO ${CMAKE_BINARY_DIR}/RelWithDebInfo
    RUNTIME_OUTPUT_DIRECTORY_MINSIZEREL   ${CMAKE_BINARY_DIR}/MinSizeRel
)
if(WIN32)
    set_target_properties(lfs_logger PROPERTIES
        LIBRARY_OUTPUT_DIRECTORY              ${CMAKE_BINARY_DIR}/$<CONFIG>
        RUNTIME_OUTPUT_DIRECTORY              ${CMAKE_BINARY_DIR}/$<CONFIG>
    )
endif()

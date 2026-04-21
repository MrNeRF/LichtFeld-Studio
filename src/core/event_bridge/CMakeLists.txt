# SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
# SPDX-License-Identifier: GPL-3.0-or-later

# Event bridge SHARED library - solves singleton duplication between
# main executable and Python module

add_library(lfs_event_bridge SHARED
    event_bridge.cpp
    control_boundary.cpp
    command_center_bridge.cpp
    scoped_handler.cpp
    localization_manager.cpp
)

target_include_directories(lfs_event_bridge
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${CMAKE_SOURCE_DIR}/src
        ${CMAKE_SOURCE_DIR}/src/core/include
)

target_link_libraries(lfs_event_bridge
    PUBLIC
        CUDA::cudart
    PRIVATE
        lfs_logger
        spdlog::spdlog
        nlohmann_json::nlohmann_json
)

target_compile_definitions(lfs_event_bridge
    PRIVATE
        LFS_EVENT_BRIDGE_EXPORTS
)

set_target_properties(lfs_event_bridge PROPERTIES
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
    set_target_properties(lfs_event_bridge PROPERTIES
        LIBRARY_OUTPUT_DIRECTORY              ${CMAKE_BINARY_DIR}/$<CONFIG>
        RUNTIME_OUTPUT_DIRECTORY              ${CMAKE_BINARY_DIR}/$<CONFIG>
    )
endif()

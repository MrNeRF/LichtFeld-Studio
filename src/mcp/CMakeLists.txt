# SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
#
# SPDX-License-Identifier: GPL-3.0-or-later

# Shared on all platforms so the embedded Python module and the app
# observe the same tool/resource registries at runtime.
set(LFS_MCP_LIB_TYPE SHARED)

add_library(lfs_mcp ${LFS_MCP_LIB_TYPE}
    mcp_protocol.cpp
    mcp_tools.cpp
    shared_scene_tools.cpp
    mcp_server.cpp
    mcp_http_server.cpp
    mcp_training_context.cpp
    llm_client.cpp
)

target_include_directories(lfs_mcp
    PUBLIC
    ${CMAKE_SOURCE_DIR}/src
    ${CMAKE_SOURCE_DIR}/external
)

find_package(OpenSSL REQUIRED)

target_link_libraries(lfs_mcp
    PUBLIC
    nlohmann_json::nlohmann_json
    PRIVATE
    lfs_core
    lfs_event_bridge
    lfs_io
    lfs_training
    lfs_rendering
    lfs_python_utils
    OpenSSL::SSL
    OpenSSL::Crypto
)

target_compile_features(lfs_mcp PUBLIC cxx_std_23)

target_compile_definitions(lfs_mcp PRIVATE LFS_MCP_EXPORTS)

set_target_properties(lfs_mcp PROPERTIES
    POSITION_INDEPENDENT_CODE ON
    LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}
    RUNTIME_OUTPUT_DIRECTORY_DEBUG ${CMAKE_BINARY_DIR}
    RUNTIME_OUTPUT_DIRECTORY_RELEASE ${CMAKE_BINARY_DIR}
    RUNTIME_OUTPUT_DIRECTORY_RELWITHDEBINFO ${CMAKE_BINARY_DIR}
    RUNTIME_OUTPUT_DIRECTORY_MINSIZEREL ${CMAKE_BINARY_DIR}
)

if(WIN32)
    set_target_properties(lfs_mcp PROPERTIES
        CXX_VISIBILITY_PRESET hidden
        VISIBILITY_INLINES_HIDDEN ON
    )
endif()

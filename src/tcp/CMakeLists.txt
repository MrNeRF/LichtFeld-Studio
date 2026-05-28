# SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
#
# SPDX-License-Identifier: GPL-3.0-or-later

# tcp module - ZMQ-based event server (responder + publisher)

set(TCP_SOURCES
    tcp_server.cpp
    tcp_responder.cpp
    tcp_publisher.cpp
)

add_library(lfs_tcp STATIC ${TCP_SOURCES})

find_package(cppzmq CONFIG REQUIRED)

target_include_directories(lfs_tcp
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/include
        ${CMAKE_BINARY_DIR}/include
    PRIVATE
        ${CMAKE_SOURCE_DIR}/src
        ${CMAKE_SOURCE_DIR}/src/visualizer
        ${CMAKE_SOURCE_DIR}/src/training/include
        ${CMAKE_SOURCE_DIR}/src/training
        ${CMAKE_SOURCE_DIR}/src/io/include
        ${CMAKE_SOURCE_DIR}/external
)

target_link_libraries(lfs_tcp
    PRIVATE
        lfs_visualizer
        lfs_core
        cppzmq
        nlohmann_json::nlohmann_json
)

set_target_properties(lfs_tcp PROPERTIES
    CXX_STANDARD 23
    CXX_STANDARD_REQUIRED ON
    POSITION_INDEPENDENT_CODE ON
    DEBUG_POSTFIX d
)


message(STATUS "╔════════════════════════════════════════════════════════════════╗")
message(STATUS "║  tcp - ZMQ-based event servers                                 ║")
message(STATUS "╚════════════════════════════════════════════════════════════════╝")
message(STATUS "  • Namespace: lfs::tcp")
message(STATUS "  • Contains: ResponderServer, PublisherServer")

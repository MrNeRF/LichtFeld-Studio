# SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
# SPDX-License-Identifier: GPL-3.0-or-later
add_library(lfs_error SHARED
    ../error.cpp ../memory_domain.cpp ../failure_report.cpp ../error_reporter.cpp
    ../error_bus.cpp ../error_envelope.cpp ../error_latch.cpp ../guarded_task.cpp)
target_compile_definitions(lfs_error PRIVATE LFS_ERROR_EXPORTS)
target_include_directories(lfs_error PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/../include"
    PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/../..")
target_link_libraries(lfs_error PUBLIC nlohmann_json::nlohmann_json PRIVATE lfs_logger)
if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU" AND CMAKE_CXX_COMPILER_VERSION VERSION_GREATER_EQUAL 12)
    target_link_libraries(lfs_error PRIVATE stdc++exp)
endif()
set_target_properties(lfs_error PROPERTIES
    POSITION_INDEPENDENT_CODE ON CXX_STANDARD 23 CXX_STANDARD_REQUIRED ON
    CXX_VISIBILITY_PRESET hidden VISIBILITY_INLINES_HIDDEN ON
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}"
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}")
target_compile_features(lfs_error PUBLIC cxx_std_23)
if(MSVC)
    target_compile_options(lfs_error PRIVATE /utf-8 /W4)
    target_compile_definitions(lfs_error PUBLIC NOMINMAX)
endif()

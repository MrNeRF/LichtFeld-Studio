# SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
# SPDX-License-Identifier: GPL-3.0-or-later
add_library(lfs_file_io SHARED ../atomic_file.cpp)
target_compile_definitions(lfs_file_io PRIVATE LFS_FILE_IO_EXPORTS)
target_include_directories(lfs_file_io PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/../include")
target_link_libraries(lfs_file_io PUBLIC lfs_error)
target_compile_features(lfs_file_io PUBLIC cxx_std_23)
set_target_properties(lfs_file_io PROPERTIES POSITION_INDEPENDENT_CODE ON
    CXX_VISIBILITY_PRESET hidden VISIBILITY_INLINES_HIDDEN ON
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}" RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}")
if(MSVC)
    target_compile_options(lfs_file_io PRIVATE /utf-8 /W4)
    target_compile_definitions(lfs_file_io PUBLIC NOMINMAX)
endif()

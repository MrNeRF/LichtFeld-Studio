# SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
#
# SPDX-License-Identifier: GPL-3.0-or-later

# External dependencies (EXCLUDE_FROM_ALL to prevent third-party install rules)

# Zep - embeddable editor core
if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/zep/CMakeLists.txt")
    message(STATUS "Adding Zep from external/zep")
    add_subdirectory(zep EXCLUDE_FROM_ALL)
else()
    message(WARNING "Zep not found at external/zep")
endif()

# tree-sitter - editor-grade syntax parsing for dirty Python buffers.
set(LFS_TREE_SITTER_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/tree-sitter")
set(LFS_TREE_SITTER_PYTHON_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/tree-sitter-python")

set(LFS_TREE_SITTER_SOURCES
    "${LFS_TREE_SITTER_ROOT}/lib/src/alloc.c"
    "${LFS_TREE_SITTER_ROOT}/lib/src/get_changed_ranges.c"
    "${LFS_TREE_SITTER_ROOT}/lib/src/language.c"
    "${LFS_TREE_SITTER_ROOT}/lib/src/lexer.c"
    "${LFS_TREE_SITTER_ROOT}/lib/src/node.c"
    "${LFS_TREE_SITTER_ROOT}/lib/src/parser.c"
    "${LFS_TREE_SITTER_ROOT}/lib/src/point.c"
    "${LFS_TREE_SITTER_ROOT}/lib/src/query.c"
    "${LFS_TREE_SITTER_ROOT}/lib/src/stack.c"
    "${LFS_TREE_SITTER_ROOT}/lib/src/subtree.c"
    "${LFS_TREE_SITTER_ROOT}/lib/src/tree.c"
    "${LFS_TREE_SITTER_ROOT}/lib/src/tree_cursor.c"
    "${LFS_TREE_SITTER_ROOT}/lib/src/wasm_store.c"
)

set(LFS_TREE_SITTER_REQUIRED_FILES
    "${LFS_TREE_SITTER_ROOT}/lib/include/tree_sitter/api.h"
    "${LFS_TREE_SITTER_ROOT}/lib/src/wasm_store.h"
    "${LFS_TREE_SITTER_PYTHON_ROOT}/bindings/c/tree_sitter/tree-sitter-python.h"
    "${LFS_TREE_SITTER_PYTHON_ROOT}/src/parser.c"
    "${LFS_TREE_SITTER_PYTHON_ROOT}/src/scanner.c"
    "${LFS_TREE_SITTER_PYTHON_ROOT}/src/tree_sitter/parser.h"
    ${LFS_TREE_SITTER_SOURCES}
)

foreach(required_file IN LISTS LFS_TREE_SITTER_REQUIRED_FILES)
    if(NOT EXISTS "${required_file}")
        message(FATAL_ERROR "Required tree-sitter file is missing: ${required_file}")
    endif()
endforeach()

message(STATUS "Adding tree-sitter Python parser")

add_library(lfs_tree_sitter STATIC ${LFS_TREE_SITTER_SOURCES})
add_library(lfs::tree_sitter ALIAS lfs_tree_sitter)
target_include_directories(lfs_tree_sitter
    PUBLIC
        "${LFS_TREE_SITTER_ROOT}/lib/include"
    PRIVATE
        "${LFS_TREE_SITTER_ROOT}/lib/src"
)
target_compile_definitions(lfs_tree_sitter
    PRIVATE
        _POSIX_C_SOURCE=200112L
        _DEFAULT_SOURCE
        _BSD_SOURCE
        _DARWIN_C_SOURCE
)
set_target_properties(lfs_tree_sitter PROPERTIES
    POSITION_INDEPENDENT_CODE ON
    C_STANDARD 11
    C_STANDARD_REQUIRED ON
)

add_library(lfs_tree_sitter_python STATIC
    "${LFS_TREE_SITTER_PYTHON_ROOT}/src/parser.c"
    "${LFS_TREE_SITTER_PYTHON_ROOT}/src/scanner.c"
)
add_library(lfs::tree_sitter_python ALIAS lfs_tree_sitter_python)
target_include_directories(lfs_tree_sitter_python
    PUBLIC
        "${LFS_TREE_SITTER_PYTHON_ROOT}/bindings/c"
    PRIVATE
        "${LFS_TREE_SITTER_PYTHON_ROOT}/src"
)
set_target_properties(lfs_tree_sitter_python PROPERTIES
    POSITION_INDEPENDENT_CODE ON
    C_STANDARD 11
    C_STANDARD_REQUIRED ON
)

# OpenMesh - Half-edge mesh data structure
if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/OpenMesh/CMakeLists.txt")
    message(STATUS "Adding OpenMesh")
    set(OPENMESH_BUILD_SHARED OFF CACHE BOOL "" FORCE)
    set(OPENMESH_BUILD_UNIT_TESTS OFF CACHE BOOL "" FORCE)
    set(BUILD_TESTING OFF CACHE BOOL "" FORCE)
    add_subdirectory(OpenMesh EXCLUDE_FROM_ALL)
else()
    message(WARNING "OpenMesh not found at external/OpenMesh")
endif()

# nvImageCodec - GPU-accelerated image codec library
find_package(Boost REQUIRED)
if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/nvImageCodec/CMakeLists.txt")
    message(STATUS "Adding nvImageCodec")
    if(DEFINED DETECTED_COMPUTE_CAP)
        string(REPLACE "." "" _arch "${DETECTED_COMPUTE_CAP}")
        set(CUDA_TARGET_ARCHS "${_arch}" CACHE STRING "" FORCE)
    endif()
    set(BUILD_NVJPEG_EXT ON CACHE BOOL "" FORCE)
    set(BUILD_NVJPEG2K_EXT ON CACHE BOOL "" FORCE)
    set(WITH_DYNAMIC_NVJPEG OFF CACHE BOOL "" FORCE)
    set(WITH_DYNAMIC_NVJPEG2K OFF CACHE BOOL "" FORCE)
    add_subdirectory(nvImageCodec EXCLUDE_FROM_ALL)
else()
    message(WARNING "nvImageCodec not found - GPU image loading unavailable")
endif()

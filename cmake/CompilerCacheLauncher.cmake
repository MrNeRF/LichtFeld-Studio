# SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
#
# SPDX-License-Identifier: GPL-3.0-or-later

cmake_minimum_required(VERSION 3.30)

foreach(_lfs_required_variable IN ITEMS
        LFS_COMPILER_CACHE_PROGRAM
        LFS_COMPILER_CACHE_SOURCE_ROOT
        LFS_COMPILER_CACHE_BINARY_ROOT)
    if(NOT DEFINED ${_lfs_required_variable} OR "${${_lfs_required_variable}}" STREQUAL "")
        message(FATAL_ERROR "Compiler-cache launcher is missing ${_lfs_required_variable}")
    endif()
endforeach()

set(_lfs_first_compiler_argument -1)
math(EXPR _lfs_last_argument "${CMAKE_ARGC} - 1")
foreach(_lfs_argument_index RANGE 0 ${_lfs_last_argument})
    if("${CMAKE_ARGV${_lfs_argument_index}}" STREQUAL "--")
        math(EXPR _lfs_first_compiler_argument "${_lfs_argument_index} + 1")
        break()
    endif()
endforeach()

if(_lfs_first_compiler_argument LESS 0 OR
   _lfs_first_compiler_argument GREATER _lfs_last_argument)
    message(FATAL_ERROR "Compiler-cache launcher did not receive a compiler command")
endif()

set(_lfs_compiler_command "${LFS_COMPILER_CACHE_PROGRAM}")
foreach(_lfs_argument_index RANGE ${_lfs_first_compiler_argument} ${_lfs_last_argument})
    set(_lfs_argument "${CMAKE_ARGV${_lfs_argument_index}}")

    # Absolute compile definitions may intentionally describe runtime paths.
    # Normalize compiler filesystem arguments without changing those values.
    if(NOT _lfs_argument MATCHES "^-D")
        string(REPLACE
            "${LFS_COMPILER_CACHE_BINARY_ROOT}" "."
            _lfs_argument "${_lfs_argument}")
        string(REPLACE
            "${LFS_COMPILER_CACHE_SOURCE_ROOT}" ".lfs-cache-source"
            _lfs_argument "${_lfs_argument}")
    endif()

    string(REPLACE ";" "\\;" _lfs_argument "${_lfs_argument}")
    list(APPEND _lfs_compiler_command "${_lfs_argument}")
endforeach()

execute_process(
    COMMAND ${_lfs_compiler_command}
    WORKING_DIRECTORY "${LFS_COMPILER_CACHE_BINARY_ROOT}"
    RESULT_VARIABLE _lfs_compiler_result)

if(_lfs_compiler_result MATCHES "^[0-9]+$")
    cmake_language(EXIT ${_lfs_compiler_result})
endif()
message(FATAL_ERROR "Compiler-cache launcher failed: ${_lfs_compiler_result}")

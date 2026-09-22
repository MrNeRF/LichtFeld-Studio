# SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
#
# SPDX-License-Identifier: GPL-3.0-or-later

cmake_minimum_required(VERSION 3.30)

# CMake appends target and source options after configuration defaults. Keep
# the last option in each conflicting group, matching cl/nvcc semantics without
# their override warnings. This preserves the fast-C++ and per-source CUDA ICE
# workarounds while leaving diagnostic flags and all other arguments intact.
function(_lfs_flag_group argument output)
    if(argument MATCHES "^[-/]O([012dx])$")
        set(group host_optimization)
    elseif(argument MATCHES "^-O3$")
        set(group host_optimization)
    elseif(argument MATCHES "^[-/]Ob[0123]$")
        set(group inlining)
    elseif(argument MATCHES "^[-/]Z[7iI]$")
        set(group debug_format)
    elseif(argument MATCHES "^[-/]Zc:preprocessor-?$")
        set(group preprocessor)
    else()
        set(group "")
    endif()
    set(${output} "${group}" PARENT_SCOPE)
endfunction()

set(_lfs_first_argument -1)
math(EXPR _lfs_last_argument "${CMAKE_ARGC} - 1")
foreach(_lfs_index RANGE 0 ${_lfs_last_argument})
    if("${CMAKE_ARGV${_lfs_index}}" STREQUAL "--")
        math(EXPR _lfs_first_argument "${_lfs_index} + 1")
        break()
    endif()
endforeach()
if(_lfs_first_argument LESS 0 OR _lfs_first_argument GREATER _lfs_last_argument)
    message(FATAL_ERROR "MSVC flag normalization requires a compiler command after --")
endif()

foreach(_lfs_index RANGE ${_lfs_first_argument} ${_lfs_last_argument})
    _lfs_flag_group("${CMAKE_ARGV${_lfs_index}}" _lfs_group)
    if(NOT _lfs_group STREQUAL "")
        set(_lfs_last_${_lfs_group} ${_lfs_index})
    endif()
endforeach()

set(_lfs_command "execute_process(COMMAND")
foreach(_lfs_index RANGE ${_lfs_first_argument} ${_lfs_last_argument})
    set(_lfs_argument "${CMAKE_ARGV${_lfs_index}}")
    _lfs_flag_group("${_lfs_argument}" _lfs_group)
    if(NOT _lfs_group STREQUAL "" AND NOT _lfs_index EQUAL _lfs_last_${_lfs_group})
        continue()
    endif()
    # CMake lists cannot preserve an element ending in a backslash, as used by
    # MSVC's /Fd directory argument. Evaluate quoted references to the original
    # argv entries so backslashes, semicolons and empty arguments stay intact.
    string(APPEND _lfs_command " \"\${CMAKE_ARGV${_lfs_index}}\"")
endforeach()

string(APPEND _lfs_command " RESULT_VARIABLE _lfs_result)")
cmake_language(EVAL CODE "${_lfs_command}")
if(_lfs_result MATCHES "^[0-9]+$")
    cmake_language(EXIT ${_lfs_result})
endif()
message(FATAL_ERROR "Compiler invocation failed: ${_lfs_result}")

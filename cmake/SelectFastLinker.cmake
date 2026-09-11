# SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
#
# SPDX-License-Identifier: GPL-3.0-or-later

# The link graph here is dominated by a few very large outputs (the executable
# and liblfs_core / liblfs_mcp are hundreds of megabytes combined), so the
# default GNU ld costs several seconds on every incremental build. mold links
# the same graph roughly eight times faster and lld about three times faster.
# Selection is automatic and silently falls back to the toolchain default, so a
# machine without either linker installed still configures and builds.

set(LFS_LINKER "AUTO" CACHE STRING "Linker selection: AUTO, MOLD, LLD, GOLD or DEFAULT")
set_property(CACHE LFS_LINKER PROPERTY STRINGS AUTO MOLD LLD GOLD DEFAULT)

function(_lfs_select_fast_linker)
    if(NOT CMAKE_SYSTEM_NAME STREQUAL "Linux" OR LFS_LINKER STREQUAL "DEFAULT")
        return()
    endif()

    if(DEFINED CMAKE_LINKER_TYPE AND NOT CMAKE_LINKER_TYPE STREQUAL "")
        message(STATUS "Linker: honouring CMAKE_LINKER_TYPE=${CMAKE_LINKER_TYPE}")
        return()
    endif()

    # -fuse-ld=mold reached GCC in 12.1 and Clang in 12; older drivers only
    # understand the -B wrapper form, which CMAKE_LINKER_TYPE does not emit.
    set(_mold_supported FALSE)
    if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU" AND CMAKE_CXX_COMPILER_VERSION VERSION_GREATER_EQUAL 12.1)
        set(_mold_supported TRUE)
    elseif(CMAKE_CXX_COMPILER_ID MATCHES "Clang" AND CMAKE_CXX_COMPILER_VERSION VERSION_GREATER_EQUAL 12)
        set(_mold_supported TRUE)
    endif()

    # AUTO only ever picks mold. LLD and GOLD stay explicit opt-ins: silently
    # switching linker because one happens to be installed would change ELF
    # layout and static-archive symbol resolution on machines where nobody
    # validated it, and this project already ships workarounds for that class
    # of problem (see the --exclude-libs block in the top-level CMakeLists).
    if(LFS_LINKER STREQUAL "AUTO")
        set(_candidates "")
        if(_mold_supported)
            list(APPEND _candidates MOLD)
        endif()
    else()
        set(_candidates "${LFS_LINKER}")
        if(LFS_LINKER STREQUAL "MOLD" AND NOT _mold_supported)
            message(FATAL_ERROR "LFS_LINKER=MOLD needs GCC >= 12.1 or Clang >= 12, found ${CMAKE_CXX_COMPILER_ID} ${CMAKE_CXX_COMPILER_VERSION}")
        endif()
    endif()

    foreach(_candidate IN LISTS _candidates)
        string(TOLOWER "${_candidate}" _program)
        find_program(LFS_LINKER_PROGRAM_${_candidate} NAMES "ld.${_program}" "${_program}")
        if(LFS_LINKER_PROGRAM_${_candidate})
            set(CMAKE_LINKER_TYPE "${_candidate}" PARENT_SCOPE)
            message(STATUS "Linker: ${_candidate} (${LFS_LINKER_PROGRAM_${_candidate}})")
            return()
        endif()
    endforeach()

    if(NOT LFS_LINKER STREQUAL "AUTO")
        message(FATAL_ERROR "LFS_LINKER=${LFS_LINKER} requested but the linker was not found on PATH")
    endif()
    message(STATUS "Linker: toolchain default (install mold for faster incremental links)")
endfunction()

_lfs_select_fast_linker()

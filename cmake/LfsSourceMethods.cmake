# SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
# SPDX-License-Identifier: GPL-3.0-or-later

include_guard(GLOBAL)
include(CMakeParseArguments)

function(lfs_add_source_method)
    set(options)
    set(one_value_args NAME TARGET REGISTRAR)
    set(multi_value_args)
    cmake_parse_arguments(LFS_METHOD
        "${options}"
        "${one_value_args}"
        "${multi_value_args}"
        ${ARGN})

    if(LFS_METHOD_UNPARSED_ARGUMENTS)
        message(FATAL_ERROR
            "lfs_add_source_method received unknown arguments: ${LFS_METHOD_UNPARSED_ARGUMENTS}")
    endif()
    foreach(required_arg NAME TARGET REGISTRAR)
        if(NOT LFS_METHOD_${required_arg})
            message(FATAL_ERROR "lfs_add_source_method requires ${required_arg}")
        endif()
    endforeach()

    if(NOT LFS_METHOD_NAME MATCHES "^[a-z0-9-]+$")
        message(FATAL_ERROR
            "Source method NAME '${LFS_METHOD_NAME}' must match [a-z0-9-]+")
    endif()
    if(LFS_METHOD_NAME STREQUAL "3dgs")
        message(FATAL_ERROR "Source method NAME '3dgs' is reserved for the built-in trainer")
    endif()
    if(NOT LFS_METHOD_REGISTRAR MATCHES "^[A-Za-z_][A-Za-z0-9_]*$")
        message(FATAL_ERROR
            "Source method REGISTRAR '${LFS_METHOD_REGISTRAR}' must be a C identifier")
    endif()
    if(NOT TARGET "${LFS_METHOD_TARGET}")
        message(FATAL_ERROR
            "Source method '${LFS_METHOD_NAME}' references missing target '${LFS_METHOD_TARGET}'")
    endif()

    if(WIN32)
        target_compile_definitions("${LFS_METHOD_TARGET}" PRIVATE LFS_VIS_EXPORTS)
    endif()

    get_property(registered_names GLOBAL PROPERTY LFS_SOURCE_METHOD_NAMES)
    if(LFS_METHOD_NAME IN_LIST registered_names)
        message(FATAL_ERROR "Source method '${LFS_METHOD_NAME}' was registered more than once")
    endif()

    set_property(GLOBAL APPEND PROPERTY LFS_SOURCE_METHOD_NAMES "${LFS_METHOD_NAME}")
    set_property(GLOBAL APPEND PROPERTY LFS_SOURCE_METHOD_TARGETS "${LFS_METHOD_TARGET}")
    set_property(GLOBAL APPEND PROPERTY LFS_SOURCE_METHOD_REGISTRARS "${LFS_METHOD_REGISTRAR}")
endfunction()

# SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
#
# SPDX-License-Identifier: GPL-3.0-or-later

include_guard(GLOBAL)

find_program(LFS_GLSLC_EXECUTABLE glslc)
find_program(LFS_GLSLANG_VALIDATOR_EXECUTABLE glslangValidator)

if(LFS_GLSLC_EXECUTABLE)
    message(STATUS "Found glslc: ${LFS_GLSLC_EXECUTABLE}")
elseif(LFS_GLSLANG_VALIDATOR_EXECUTABLE)
    message(STATUS "Found glslangValidator: ${LFS_GLSLANG_VALIDATOR_EXECUTABLE}")
else()
    message(STATUS "glslc/glslangValidator not found; using the in-tree glslang shader compiler")
endif()

function(compile_shader target source output symbol)
    if(NOT TARGET "${target}")
        message(FATAL_ERROR "compile_shader target does not exist: ${target}")
    endif()
    if(NOT TARGET lfs_shader_compiler)
        message(FATAL_ERROR "compile_shader requires the lfs_shader_compiler target")
    endif()

    if(IS_ABSOLUTE "${source}")
        set(_source "${source}")
    else()
        set(_source "${CMAKE_CURRENT_SOURCE_DIR}/${source}")
    endif()

    if(IS_ABSOLUTE "${output}")
        set(_output "${output}")
    else()
        set(_output "${CMAKE_CURRENT_BINARY_DIR}/${output}")
    endif()

    string(MAKE_C_IDENTIFIER "${target}_${symbol}_shader" _shader_target)
    set_source_files_properties("${_output}" PROPERTIES GENERATED TRUE HEADER_FILE_ONLY TRUE)

    add_custom_command(
        OUTPUT "${_output}"
        COMMAND $<TARGET_FILE:lfs_shader_compiler>
                --input "${_source}"
                --output "${_output}"
                --symbol "${symbol}"
        DEPENDS "${_source}" lfs_shader_compiler
        COMMENT "Compiling shader ${source}"
        VERBATIM)

    add_custom_target("${_shader_target}" DEPENDS "${_output}")
    add_dependencies("${target}" "${_shader_target}")
    target_sources("${target}" PRIVATE "${_output}")
endfunction()

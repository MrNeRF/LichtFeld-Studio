# SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
#
# SPDX-License-Identifier: GPL-3.0-or-later

# Libvterm.cmake
# libvterm - a VT220/xterm terminal emulator library (MIT license)
# Located in external/libvterm as git submodule

set(LIBVTERM_SOURCE_DIR ${CMAKE_SOURCE_DIR}/external/libvterm)
set(LIBVTERM_ENC_DIR ${LIBVTERM_SOURCE_DIR}/src/encoding)

# update submodule libvterm
if(NOT EXISTS ${LIBVTERM_SOURCE_DIR}/src/encoding.c OR NOT EXISTS ${LIBVTERM_SOURCE_DIR}/include/vterm.h)
    message(FATAL_ERROR
        "libvterm sources are missing from external/libvterm. "
        "Initialize the submodule with: git submodule update --init --recursive external/libvterm"
    )
endif()

# Generate encoding tables if not present (pre-generated in submodule)
if(NOT EXISTS ${LIBVTERM_ENC_DIR}/DECdrawing.inc OR NOT EXISTS ${LIBVTERM_ENC_DIR}/uk.inc)
    find_package(Perl REQUIRED)
endif()

if(NOT EXISTS ${LIBVTERM_ENC_DIR}/DECdrawing.inc)
    execute_process(
        COMMAND ${PERL_EXECUTABLE} -CSD ${LIBVTERM_SOURCE_DIR}/tbl2inc_c.pl ${LIBVTERM_ENC_DIR}/DECdrawing.tbl
        OUTPUT_FILE ${LIBVTERM_ENC_DIR}/DECdrawing.inc
        WORKING_DIRECTORY ${LIBVTERM_SOURCE_DIR}
    )
endif()

if(NOT EXISTS ${LIBVTERM_ENC_DIR}/uk.inc)
    execute_process(
        COMMAND ${PERL_EXECUTABLE} -CSD ${LIBVTERM_SOURCE_DIR}/tbl2inc_c.pl ${LIBVTERM_ENC_DIR}/uk.tbl
        OUTPUT_FILE ${LIBVTERM_ENC_DIR}/uk.inc
        WORKING_DIRECTORY ${LIBVTERM_SOURCE_DIR}
    )
endif()

set(LIBVTERM_SOURCES
    ${LIBVTERM_SOURCE_DIR}/src/encoding.c
    ${LIBVTERM_SOURCE_DIR}/src/keyboard.c
    ${LIBVTERM_SOURCE_DIR}/src/mouse.c
    ${LIBVTERM_SOURCE_DIR}/src/parser.c
    ${LIBVTERM_SOURCE_DIR}/src/pen.c
    ${LIBVTERM_SOURCE_DIR}/src/screen.c
    ${LIBVTERM_SOURCE_DIR}/src/state.c
    ${LIBVTERM_SOURCE_DIR}/src/unicode.c
    ${LIBVTERM_SOURCE_DIR}/src/vterm.c
)

add_library(vterm STATIC ${LIBVTERM_SOURCES})

target_include_directories(vterm
    PUBLIC ${LIBVTERM_SOURCE_DIR}/include
    PRIVATE ${LIBVTERM_SOURCE_DIR}/src
)

set_target_properties(vterm PROPERTIES
    C_STANDARD 99
    POSITION_INDEPENDENT_CODE ON
)

if(CMAKE_C_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(vterm PRIVATE -w)
elseif(MSVC)
    target_compile_options(vterm PRIVATE /w)
endif()

# SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
#
# SPDX-License-Identifier: GPL-3.0-or-later

set(ZEP_ROOT ${CMAKE_CURRENT_SOURCE_DIR})

configure_file(
    ${ZEP_ROOT}/cmake/config_app.h.cmake
    ${CMAKE_CURRENT_BINARY_DIR}/config_app.h
)

set(ZEP_SOURCES
    ${ZEP_ROOT}/src/buffer.cpp
    ${ZEP_ROOT}/src/commands.cpp
    ${ZEP_ROOT}/src/display.cpp
    ${ZEP_ROOT}/src/editor.cpp
    ${ZEP_ROOT}/src/filesystem.cpp
    ${ZEP_ROOT}/src/glyph_iterator.cpp
    ${ZEP_ROOT}/src/indexer.cpp
    ${ZEP_ROOT}/src/keymap.cpp
    ${ZEP_ROOT}/src/line_widgets.cpp
    ${ZEP_ROOT}/src/mcommon/animation/timer.cpp
    ${ZEP_ROOT}/src/mcommon/file/path.cpp
    ${ZEP_ROOT}/src/mcommon/string/stringutils.cpp
    ${ZEP_ROOT}/src/mode.cpp
    ${ZEP_ROOT}/src/mode_repl.cpp
    ${ZEP_ROOT}/src/mode_search.cpp
    ${ZEP_ROOT}/src/mode_standard.cpp
    ${ZEP_ROOT}/src/mode_tree.cpp
    ${ZEP_ROOT}/src/mode_vim.cpp
    ${ZEP_ROOT}/src/range_markers.cpp
    ${ZEP_ROOT}/src/regress.cpp
    ${ZEP_ROOT}/src/scroller.cpp
    ${ZEP_ROOT}/src/splits.cpp
    ${ZEP_ROOT}/src/syntax.cpp
    ${ZEP_ROOT}/src/syntax_markdown.cpp
    ${ZEP_ROOT}/src/syntax_python.cpp
    ${ZEP_ROOT}/src/syntax_providers.cpp
    ${ZEP_ROOT}/src/syntax_rainbow_brackets.cpp
    ${ZEP_ROOT}/src/syntax_tree.cpp
    ${ZEP_ROOT}/src/tab_window.cpp
    ${ZEP_ROOT}/src/theme.cpp
    ${ZEP_ROOT}/src/window.cpp
)

add_library(Zep STATIC ${ZEP_SOURCES})
add_library(Zep::Zep ALIAS Zep)

target_compile_definitions(Zep PUBLIC ZEP_FEATURE_CPP_FILE_SYSTEM)

target_include_directories(Zep
    PUBLIC
        ${ZEP_ROOT}/include
    PRIVATE
        ${ZEP_ROOT}/src
        ${ZEP_ROOT}/src/mcommon
        ${CMAKE_CURRENT_BINARY_DIR}
)

set_target_properties(Zep PROPERTIES
    CXX_STANDARD 17
    CXX_STANDARD_REQUIRED ON
    POSITION_INDEPENDENT_CODE ON
)

# SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
# SPDX-License-Identifier: GPL-3.0-or-later
# Register all contracts against the existing root production targets.
cmake_file_api(QUERY API_VERSION 1 CODEMODEL 2)
if(WIN32)
    set(MEDIA_RUNTIME_LIBRARY_DIR "${VCPKG_INSTALLED_DIR}/${VCPKG_TARGET_TRIPLET}/$<$<CONFIG:Debug>:debug/>bin")
endif()
find_program(LFS_MEDIA_TEST_FFMPEG NAMES ffmpeg)
find_program(LFS_MEDIA_TEST_FFPROBE NAMES ffprobe)
foreach(_media_tool IN ITEMS FFMPEG FFPROBE)
    if(NOT LFS_MEDIA_TEST_${_media_tool})
        # CI may install the command-line fixture tools after configuration.
        string(TOLOWER "${_media_tool}" _media_tool_name)
        set(LFS_MEDIA_TEST_${_media_tool} "${_media_tool_name}")
    endif()
endforeach()
add_test(NAME MediaFixturePreparation
    COMMAND "${Python_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/test_fixture_preparation.py")
set_tests_properties(MediaFixturePreparation PROPERTIES
    LABELS "media;cpu;offline" TIMEOUT 180
    ENVIRONMENT "LFS_MEDIA_TEST_FFMPEG=${LFS_MEDIA_TEST_FFMPEG};LFS_MEDIA_TEST_FFPROBE=${LFS_MEDIA_TEST_FFPROBE}")
add_subdirectory(extractor)
add_subdirectory(studio)
add_custom_target(media_contracts DEPENDS
    media_extraction_runner media_shared_core_bridge media_test_bindings
    media_public_image_output
    media-ingest media_mcp_runner LichtFeld-Studio lfs_visualizer lfs_py)
if(TARGET media_studio_runner)
    add_dependencies(media_contracts media_studio_runner)
endif()

# SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
# SPDX-License-Identifier: GPL-3.0-or-later
option(LFS_BUILD_MEDIA_CLI "Build the optional media-ingest executable" OFF)
get_filename_component(_media_root "${CMAKE_CURRENT_SOURCE_DIR}/../.." ABSOLUTE)

add_library(lfs_media SHARED
    media_ingest.cpp media_backends.cpp hdr_renderer.cpp
    media_probe.cpp frame_sink.cpp file_frame_sink.cpp
    video_frame_extractor.cpp hdr_tonemap.cpp)
target_compile_definitions(lfs_media PRIVATE LFS_MEDIA_EXPORTS)
target_compile_features(lfs_media PUBLIC cxx_std_23)
target_include_directories(lfs_media PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/include"
    PRIVATE "${_media_root}/src")
# Existing Studio adapters consume the provider's public FFmpeg ABI until they
# migrate to opaque media frame contracts; they never link FFmpeg themselves.
target_include_directories(lfs_media SYSTEM PUBLIC ${FFMPEG_INCLUDE_DIRS})
target_link_libraries(lfs_media PUBLIC lfs_error
    PRIVATE lfs_image_codecs lfs_logger)
include("${_media_root}/cmake/FFmpegProvider.cmake")
lfs_link_ffmpeg_provider(lfs_media)
set_target_properties(lfs_media PROPERTIES POSITION_INDEPENDENT_CODE ON
    CXX_VISIBILITY_PRESET hidden VISIBILITY_INLINES_HIDDEN ON
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}"
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}")
if(MSVC)
    target_compile_options(lfs_media PRIVATE /utf-8 /W4)
    target_compile_definitions(lfs_media PUBLIC NOMINMAX)
endif()
if(NOT BUILD_PORTABLE)
    install(DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/include/media" DESTINATION include)
endif()
# The optional CLI is also required by the root media contract tests.
if(LFS_BUILD_MEDIA_CLI OR BUILD_TESTS OR BUILD_FORMAT_TESTS OR BUILD_VISUALIZER_TESTS)
    add_executable(media-ingest cli.cpp)
    set_target_properties(media-ingest PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}")
    target_compile_features(media-ingest PRIVATE cxx_std_23)
    target_link_libraries(media-ingest PRIVATE lfs_media)
    target_include_directories(media-ingest PRIVATE "${_media_root}/src" "${CMAKE_BINARY_DIR}/include")
    add_dependencies(media-ingest lfs_git_version)
    if(MSVC)
        target_compile_options(media-ingest PRIVATE /utf-8 /W4)
    endif()
endif()

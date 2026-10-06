# SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
# SPDX-License-Identifier: GPL-3.0-or-later
option(LFS_BUILD_MEDIA_CLI "Build the optional media-ingest executable" OFF)
get_filename_component(_media_root "${CMAKE_CURRENT_SOURCE_DIR}/../.." ABSOLUTE)

# target_link_libraries interprets vcpkg's optimized/debug qualifiers correctly.
# Passing that list to INTERFACE_LINK_LIBRARIES would reject static macOS builds.
if(DEFINED VCPKG_TARGET_TRIPLET)
    # The I/O module also uses the FFMPEG pkg-config prefix. Its cached bare
    # library names must not short-circuit the vcpkg finder on reconfigure.
    set(FFMPEG_FOUND FALSE)
    set(FFMPEG_LIBRARIES "")
    find_package(FFMPEG REQUIRED)
    set(_media_ffmpeg_include ${FFMPEG_INCLUDE_DIRS})
    set(_media_ffmpeg_libraries ${FFMPEG_LIBRARIES})
else()
    find_path(_media_ffmpeg_include libavformat/avformat.h REQUIRED)
    foreach(_component avformat avcodec avutil swscale)
        find_library(_media_${_component}_library NAMES ${_component} REQUIRED)
        list(APPEND _media_ffmpeg_libraries ${_media_${_component}_library})
    endforeach()
endif()
add_library(lfs_media SHARED
    media_ingest.cpp media_backends.cpp hdr_renderer.cpp
    ../io/media_probe.cpp ../io/frame_sink.cpp ../io/file_frame_sink.cpp
    ../io/video_frame_extractor.cpp ../io/hdr_tonemap.cpp)
target_compile_definitions(lfs_media PRIVATE LFS_MEDIA_EXPORTS)
target_compile_features(lfs_media PUBLIC cxx_std_23)
target_include_directories(lfs_media PUBLIC "${_media_root}/src/io/include"
    PRIVATE "${_media_root}/src" "${_media_root}/src/io" ${_media_ffmpeg_include})
target_link_libraries(lfs_media PUBLIC lfs_error
    PRIVATE lfs_image_codecs lfs_logger ${_media_ffmpeg_libraries})
if(CMAKE_SYSTEM_NAME STREQUAL "Linux")
    # FFmpeg's x86 assembly uses direct relocations to archive-local data.
    # Keep those symbols local when linking static FFmpeg into this shared module,
    # including Release builds without LFS_HIDDEN_SYMBOLS and the CPU test harness.
    target_link_options(lfs_media PRIVATE
        "LINKER:--exclude-libs,libavcodec.a:libavutil.a")
endif()
set_target_properties(lfs_media PROPERTIES POSITION_INDEPENDENT_CODE ON
    CXX_VISIBILITY_PRESET hidden VISIBILITY_INLINES_HIDDEN ON
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}"
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}")
if(MSVC)
    target_compile_options(lfs_media PRIVATE /utf-8 /W4)
    target_compile_definitions(lfs_media PUBLIC NOMINMAX)
endif()
if(LFS_BUILD_MEDIA_CLI)
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

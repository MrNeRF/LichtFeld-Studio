# SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
# SPDX-License-Identifier: GPL-3.0-or-later
cmake_minimum_required(VERSION 3.24)
project(LichtFeldMediaExtractionContracts LANGUAGES CXX)
enable_testing()
find_package(Python3 3.10 REQUIRED COMPONENTS Interpreter)
find_program(LFS_MEDIA_TEST_FFMPEG NAMES ffmpeg REQUIRED)
find_program(LFS_MEDIA_TEST_FFPROBE NAMES ffprobe REQUIRED)
find_package(nlohmann_json CONFIG REQUIRED)
find_package(OpenEXR 3.4.15 CONFIG REQUIRED)
find_package(JPEG REQUIRED)
find_package(PNG REQUIRED)
find_package(ZLIB REQUIRED)
find_package(TIFF REQUIRED)
find_package(WebP CONFIG QUIET)
if(NOT TARGET WebP::webp)
    find_package(PkgConfig REQUIRED)
    pkg_check_modules(MEDIA_WEBP REQUIRED IMPORTED_TARGET libwebp)
    add_library(WebP::webp ALIAS PkgConfig::MEDIA_WEBP)
endif()
if(DEFINED VCPKG_TARGET_TRIPLET)
    find_package(FFMPEG REQUIRED)
    set(MEDIA_FFMPEG_INCLUDE ${FFMPEG_INCLUDE_DIRS})
    set(MEDIA_FFMPEG_LIBRARIES ${FFMPEG_LIBRARIES})
else()
    # Existing dynamic dependency prefixes can be reused without a vcpkg rebuild.
    find_path(MEDIA_FFMPEG_INCLUDE libavformat/avformat.h REQUIRED)
    foreach(component avformat avcodec avutil swscale)
        find_library(MEDIA_${component}_LIBRARY NAMES ${component} REQUIRED)
        list(APPEND MEDIA_FFMPEG_LIBRARIES ${MEDIA_${component}_LIBRARY})
    endforeach()
endif()
get_filename_component(MEDIA_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/../../.." ABSOLUTE)
add_executable(media_extraction_runner runner.cpp support.cpp
    ${MEDIA_ROOT}/src/io/video_frame_extractor.cpp
    ${MEDIA_ROOT}/src/io/hdr_tonemap.cpp
    ${MEDIA_ROOT}/src/core/image_codecs.cpp
    ${MEDIA_ROOT}/src/core/image_exr.cpp)
target_compile_features(media_extraction_runner PRIVATE cxx_std_23)
target_compile_definitions(media_extraction_runner PRIVATE LFS_STATIC_BUILD LFS_HAS_CUDA=0)
target_include_directories(media_extraction_runner PRIVATE
    ${MEDIA_ROOT}/src ${MEDIA_ROOT}/src/core/include ${MEDIA_ROOT}/external
    ${MEDIA_FFMPEG_INCLUDE})
target_include_directories(media_extraction_runner BEFORE PRIVATE ${ZLIB_INCLUDE_DIRS})
target_link_libraries(media_extraction_runner PRIVATE nlohmann_json::nlohmann_json
    OpenEXR::OpenEXRCore JPEG::JPEG PNG::PNG ZLIB::ZLIB TIFF::TIFF WebP::webp
    ${MEDIA_FFMPEG_LIBRARIES})
if(MSVC)
    target_compile_definitions(media_extraction_runner PRIVATE NOMINMAX)
    target_compile_options(media_extraction_runner PRIVATE /utf-8 /W4)
endif()
add_test(NAME MediaExtractionContracts
    COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/../test_extraction_contracts.py
        --runner $<TARGET_FILE:media_extraction_runner>)
set_tests_properties(MediaExtractionContracts PROPERTIES LABELS "media;cpu;offline;integration"
    TIMEOUT 180
    ENVIRONMENT "LFS_MEDIA_TEST_FFMPEG=${LFS_MEDIA_TEST_FFMPEG};LFS_MEDIA_TEST_FFPROBE=${LFS_MEDIA_TEST_FFPROBE}")
if(WIN32)
    if(DEFINED VCPKG_TARGET_TRIPLET)
        set(MEDIA_RUNTIME_DEFAULT "${VCPKG_INSTALLED_DIR}/${VCPKG_TARGET_TRIPLET}/bin")
    else()
        get_filename_component(MEDIA_FFMPEG_PREFIX "${MEDIA_FFMPEG_INCLUDE}" DIRECTORY)
        set(MEDIA_RUNTIME_DEFAULT "${MEDIA_FFMPEG_PREFIX}/bin")
    endif()
    set(MEDIA_RUNTIME_LIBRARY_DIR "${MEDIA_RUNTIME_DEFAULT}" CACHE PATH "Runtime DLL directory for the SDR runner")
    set_tests_properties(MediaExtractionContracts PROPERTIES
        ENVIRONMENT_MODIFICATION "PATH=path_list_prepend:${MEDIA_RUNTIME_LIBRARY_DIR}")
endif()

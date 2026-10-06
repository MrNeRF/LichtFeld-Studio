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
add_executable(media_extraction_runner runner.cpp support.cpp probe_unit_contracts.cpp frame_sink_contracts.cpp
    ${MEDIA_ROOT}/src/media/media_ingest.cpp
    ${MEDIA_ROOT}/src/io/media_probe.cpp
    ${MEDIA_ROOT}/src/io/frame_sink.cpp
    ${MEDIA_ROOT}/src/io/file_frame_sink.cpp
    ${MEDIA_ROOT}/src/io/video_player.cpp
    ${MEDIA_ROOT}/src/io/video_frame_extractor.cpp
    ${MEDIA_ROOT}/src/io/hdr_tonemap.cpp
    ${MEDIA_ROOT}/src/core/memory_domain.cpp
    ${MEDIA_ROOT}/src/core/error.cpp
    ${MEDIA_ROOT}/src/core/failure_report.cpp
    ${MEDIA_ROOT}/src/core/image_codecs.cpp
    ${MEDIA_ROOT}/src/core/image_exr.cpp)
target_compile_features(media_extraction_runner PRIVATE cxx_std_23)
target_compile_definitions(media_extraction_runner PRIVATE LFS_STATIC_BUILD LFS_HAS_CUDA=0)
target_include_directories(media_extraction_runner PRIVATE
    ${MEDIA_ROOT}/src ${MEDIA_ROOT}/src/io/include ${MEDIA_ROOT}/src/core/include ${MEDIA_ROOT}/external
    ${MEDIA_FFMPEG_INCLUDE})
target_include_directories(media_extraction_runner BEFORE PRIVATE ${ZLIB_INCLUDE_DIRS})
target_link_libraries(media_extraction_runner PRIVATE nlohmann_json::nlohmann_json
    OpenEXR::OpenEXRCore JPEG::JPEG PNG::PNG ZLIB::ZLIB TIFF::TIFF WebP::webp
    ${MEDIA_FFMPEG_LIBRARIES})
# failure_report.cpp uses std::stacktrace, which GCC ships in libstdc++exp.
if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU" AND CMAKE_CXX_COMPILER_VERSION VERSION_GREATER_EQUAL 12)
    target_link_libraries(media_extraction_runner PRIVATE stdc++exp)
endif()
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

add_test(NAME MediaProbeUnitContracts COMMAND media_extraction_runner --probe-unit)
set_tests_properties(MediaProbeUnitContracts PROPERTIES LABELS "media;cpu;offline" TIMEOUT 30)
if(WIN32)
    set_tests_properties(MediaProbeUnitContracts PROPERTIES
        ENVIRONMENT_MODIFICATION "PATH=path_list_prepend:${MEDIA_RUNTIME_LIBRARY_DIR}")
endif()

add_test(NAME MediaFrameSinkUnitContracts COMMAND media_extraction_runner --sink-unit)
set_tests_properties(MediaFrameSinkUnitContracts PROPERTIES LABELS "media;cpu;offline" TIMEOUT 30)
if(WIN32)
    set_tests_properties(MediaFrameSinkUnitContracts PROPERTIES
        ENVIRONMENT_MODIFICATION "PATH=path_list_prepend:${MEDIA_RUNTIME_LIBRARY_DIR}")
endif()

# Reuse the application's dependency prefix and existing Release job. This is the
# standalone CPU package, not the main app CMake/CUDA/shader configuration.
add_subdirectory(${MEDIA_ROOT}/media ${CMAKE_CURRENT_BINARY_DIR}/standalone-media)
add_test(NAME MediaIngestCLIContracts
    COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/../test_ingest_cli.py
        --cli $<TARGET_FILE:media-ingest> --runner $<TARGET_FILE:media_extraction_runner>)
add_test(NAME MediaIngestPackageConsumer
    COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/../test_ingest_package.py
        --build ${CMAKE_CURRENT_BINARY_DIR}/standalone-media --cmake ${CMAKE_COMMAND}
        --generator "${CMAKE_GENERATOR}" --compiler "${CMAKE_CXX_COMPILER}"
        --resource-compiler "${CMAKE_RC_COMPILER}" --manifest-tool "${CMAKE_MT}" --toolchain "${CMAKE_TOOLCHAIN_FILE}"
        --triplet "${VCPKG_TARGET_TRIPLET}" --packages "${VCPKG_INSTALLED_DIR}"
        --prefix "${CMAKE_PREFIX_PATH}" --deployment-target "${CMAKE_OSX_DEPLOYMENT_TARGET}"
        --osx-architectures "${CMAKE_OSX_ARCHITECTURES}")
set_tests_properties(MediaIngestCLIContracts MediaIngestPackageConsumer PROPERTIES
    LABELS "media;cpu;offline;integration" TIMEOUT 300
    ENVIRONMENT "LFS_MEDIA_TEST_FFMPEG=${LFS_MEDIA_TEST_FFMPEG};LFS_MEDIA_TEST_FFPROBE=${LFS_MEDIA_TEST_FFPROBE}")
if(WIN32)
    set_tests_properties(MediaIngestCLIContracts MediaIngestPackageConsumer PROPERTIES
        ENVIRONMENT_MODIFICATION "PATH=path_list_prepend:${MEDIA_RUNTIME_LIBRARY_DIR}")
endif()

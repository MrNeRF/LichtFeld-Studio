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
find_package(Threads REQUIRED)
find_package(spdlog CONFIG REQUIRED)
set(LFS_HAS_CUDA OFF)
add_compile_definitions(LFS_HAS_CUDA=0)
add_subdirectory(${MEDIA_ROOT}/src/diagnostics ${CMAKE_CURRENT_BINARY_DIR}/diagnostics)
add_subdirectory(${MEDIA_ROOT}/src/core/logger ${CMAKE_CURRENT_BINARY_DIR}/logger)
add_subdirectory(${MEDIA_ROOT}/src/core/error ${CMAKE_CURRENT_BINARY_DIR}/error)
add_subdirectory(${MEDIA_ROOT}/src/core/image_codecs ${CMAKE_CURRENT_BINARY_DIR}/image-codecs)
set(GIT_SRC_DIR "${MEDIA_ROOT}")
set(GIT_VERSION_HEADER "${CMAKE_BINARY_DIR}/include/git_version.h")
include(${MEDIA_ROOT}/cmake/GenerateGitVersion.cmake)
add_custom_target(lfs_git_version COMMAND ${CMAKE_COMMAND}
    "-DGIT_SRC_DIR=${MEDIA_ROOT}" "-DGIT_VERSION_HEADER=${GIT_VERSION_HEADER}"
    -P "${MEDIA_ROOT}/cmake/GenerateGitVersion.cmake")
set(LFS_BUILD_MEDIA_CLI ON)
add_subdirectory(${MEDIA_ROOT}/src/media ${CMAKE_CURRENT_BINARY_DIR}/media)
add_library(media_shared_core_bridge SHARED shared_core_bridge.cpp)
target_link_libraries(media_shared_core_bridge PRIVATE lfs_error)
set_target_properties(media_shared_core_bridge PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}")
add_executable(media_extraction_runner runner.cpp probe_unit_contracts.cpp frame_sink_contracts.cpp shared_core_contracts.cpp jpeg_backend_contracts.cpp
    ${MEDIA_ROOT}/src/io/video_player.cpp)
target_compile_features(media_extraction_runner PRIVATE cxx_std_23)
target_compile_definitions(media_extraction_runner PRIVATE LFS_HAS_CUDA=0)
target_include_directories(media_extraction_runner PRIVATE
    ${MEDIA_ROOT}/src ${MEDIA_ROOT}/src/io ${MEDIA_FFMPEG_INCLUDE})
target_link_libraries(media_extraction_runner PRIVATE lfs_media lfs_image_codecs lfs_logger lfs_diagnostics media_shared_core_bridge ${MEDIA_FFMPEG_LIBRARIES})
if(MSVC)
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

add_test(NAME MediaSharedCoreContracts COMMAND media_extraction_runner --shared-core-unit)
set_tests_properties(MediaSharedCoreContracts PROPERTIES LABELS "media;cpu;offline" TIMEOUT 30)
if(WIN32)
    set_tests_properties(MediaSharedCoreContracts PROPERTIES
        ENVIRONMENT_MODIFICATION "PATH=path_list_prepend:${MEDIA_RUNTIME_LIBRARY_DIR}")
endif()

add_test(NAME MediaIngestCLIContracts
    COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/../test_ingest_cli.py
        --cli $<TARGET_FILE:media-ingest> --runner $<TARGET_FILE:media_extraction_runner>
        --version-header ${GIT_VERSION_HEADER})
set_tests_properties(MediaIngestCLIContracts PROPERTIES
    LABELS "media;cpu;offline;integration" TIMEOUT 300
    ENVIRONMENT "LFS_MEDIA_TEST_FFMPEG=${LFS_MEDIA_TEST_FFMPEG};LFS_MEDIA_TEST_FFPROBE=${LFS_MEDIA_TEST_FFPROBE}")
if(WIN32)
    set_tests_properties(MediaIngestCLIContracts PROPERTIES
        ENVIRONMENT_MODIFICATION "PATH=path_list_prepend:${MEDIA_RUNTIME_LIBRARY_DIR}")
endif()

# Compile the actual binding group against the same shared library. This stays
# in the existing offline media job, without configuring tensor/GPU targets.
find_package(Python 3.10 REQUIRED COMPONENTS Interpreter Development.Module)
find_package(nanobind CONFIG REQUIRED)
nanobind_add_module(media_test_bindings python_bindings.cpp
    ${MEDIA_ROOT}/src/python/lfs/py_media.cpp)
target_compile_features(media_test_bindings PRIVATE cxx_std_23)
target_include_directories(media_test_bindings PRIVATE ${MEDIA_ROOT}/src)
target_link_libraries(media_test_bindings PRIVATE lfs_media)
if(MSVC)
    target_compile_options(media_test_bindings PRIVATE /utf-8)
endif()
add_test(NAME MediaPythonBindingsContracts
    COMMAND ${Python_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/../test_media_bindings.py
        --module-dir $<TARGET_FILE_DIR:media_test_bindings> --runtime-dir ${CMAKE_BINARY_DIR}
        --dependency-dir "${MEDIA_RUNTIME_LIBRARY_DIR}")
set_tests_properties(MediaPythonBindingsContracts PROPERTIES LABELS "media;cpu;offline;integration"
    TIMEOUT 180
    ENVIRONMENT "LFS_MEDIA_TEST_FFMPEG=${LFS_MEDIA_TEST_FFMPEG};LFS_MEDIA_TEST_FFPROBE=${LFS_MEDIA_TEST_FFPROBE}")

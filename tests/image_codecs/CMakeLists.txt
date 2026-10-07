# SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
# SPDX-License-Identifier: GPL-3.0-or-later

# Also runnable standalone without CUDA, the GUI, or LibTorch:
# cmake -S tests/image_codecs -B build-image-codecs -DCMAKE_PREFIX_PATH=...
cmake_minimum_required(VERSION 3.24)
if(CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
    project(LichtFeldImageCodecTests LANGUAGES CXX)
    enable_testing()
endif()

get_filename_component(LFS_CODEC_ROOT "${CMAKE_CURRENT_LIST_DIR}/../.." ABSOLUTE)
find_package(GTest REQUIRED)
find_package(Threads REQUIRED)
find_package(OpenEXR 3.4.15 CONFIG REQUIRED)
find_package(JPEG REQUIRED)
find_package(PNG REQUIRED)
find_package(ZLIB REQUIRED)
find_package(TIFF REQUIRED)
if(NOT TARGET WebP::webp)
    find_package(WebP CONFIG QUIET)
    if(NOT TARGET WebP::webp)
        find_package(PkgConfig REQUIRED)
        pkg_check_modules(CODEC_WEBP REQUIRED IMPORTED_TARGET libwebp)
        add_library(WebP::webp ALIAS PkgConfig::CODEC_WEBP)
    endif()
endif()

if(NOT TARGET lfs_image_codecs)
    add_subdirectory(${LFS_CODEC_ROOT}/src/core/image_codecs ${CMAKE_CURRENT_BINARY_DIR}/image-codecs)
endif()
add_executable(lichtfeld_image_codec_tests
    ../test_exr_codecs.cpp
    ../test_image_codec_dependencies.cpp)
target_compile_features(lichtfeld_image_codec_tests PRIVATE cxx_std_20)
target_include_directories(lichtfeld_image_codec_tests PRIVATE ${LFS_CODEC_ROOT}/src ${LFS_CODEC_ROOT}/src/core/include)
target_include_directories(lichtfeld_image_codec_tests BEFORE PRIVATE ${ZLIB_INCLUDE_DIRS})
target_include_directories(lichtfeld_image_codec_tests SYSTEM PRIVATE ${LFS_CODEC_ROOT}/external)
target_link_libraries(lichtfeld_image_codec_tests PRIVATE
    lfs_image_codecs
    GTest::gtest_main Threads::Threads OpenEXR::OpenEXRCore OpenEXR::OpenEXR
    JPEG::JPEG PNG::PNG TIFF::TIFF WebP::webp ZLIB::ZLIB)
if(MSVC)
    target_compile_options(lichtfeld_image_codec_tests PRIVATE /utf-8 /W4)
    target_compile_definitions(lichtfeld_image_codec_tests PRIVATE NOMINMAX)
endif()
add_test(NAME lichtfeld_image_codec_tests COMMAND lichtfeld_image_codec_tests)
set_tests_properties(lichtfeld_image_codec_tests PROPERTIES LABELS "fast;image;cpu" TIMEOUT 120)

# Exercise the actual import/export boundary; direct-source codec tests cannot
# detect missing exports from the hidden-visibility shared library.

add_library(lichtfeld_codec_exports_consumer STATIC ../test_image_codec_exports.cpp)
target_include_directories(lichtfeld_codec_exports_consumer PRIVATE ${LFS_CODEC_ROOT}/src)
target_link_libraries(lichtfeld_codec_exports_consumer PRIVATE lfs_image_codecs)
add_executable(lichtfeld_codec_exports_test test_codec_exports_main.cpp)
target_link_libraries(lichtfeld_codec_exports_test PRIVATE lichtfeld_codec_exports_consumer)
add_test(NAME lichtfeld_codec_exports_test COMMAND lichtfeld_codec_exports_test)
set_tests_properties(lichtfeld_codec_exports_test PROPERTIES LABELS "fast;image;cpu" TIMEOUT 30)

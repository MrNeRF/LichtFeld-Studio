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
find_package(TIFF REQUIRED)
if(NOT TARGET WebP::webp)
    find_package(WebP CONFIG QUIET)
    if(NOT TARGET WebP::webp)
        find_package(PkgConfig REQUIRED)
        pkg_check_modules(CODEC_WEBP REQUIRED IMPORTED_TARGET libwebp)
        add_library(WebP::webp ALIAS PkgConfig::CODEC_WEBP)
    endif()
endif()

add_executable(lichtfeld_image_codec_tests
    ../test_exr_codecs.cpp
    ${LFS_CODEC_ROOT}/src/core/image_codecs.cpp
    ${LFS_CODEC_ROOT}/src/core/image_exr.cpp)
target_compile_features(lichtfeld_image_codec_tests PRIVATE cxx_std_20)
target_include_directories(lichtfeld_image_codec_tests PRIVATE ${LFS_CODEC_ROOT}/src)
target_include_directories(lichtfeld_image_codec_tests SYSTEM PRIVATE ${LFS_CODEC_ROOT}/external)
target_link_libraries(lichtfeld_image_codec_tests PRIVATE
    GTest::gtest_main Threads::Threads OpenEXR::OpenEXRCore OpenEXR::OpenEXR
    JPEG::JPEG PNG::PNG TIFF::TIFF WebP::webp)
if(MSVC)
    target_compile_options(lichtfeld_image_codec_tests PRIVATE /utf-8 /W4)
    target_compile_definitions(lichtfeld_image_codec_tests PRIVATE NOMINMAX)
endif()
add_test(NAME lichtfeld_image_codec_tests COMMAND lichtfeld_image_codec_tests)
set_tests_properties(lichtfeld_image_codec_tests PROPERTIES LABELS "fast;image;cpu" TIMEOUT 120)

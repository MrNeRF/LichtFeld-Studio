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
find_package(libheif CONFIG REQUIRED)
find_package(lcms2 CONFIG REQUIRED)
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
    if(NOT TARGET lfs_backend_configuration)
        include("${LFS_CODEC_ROOT}/cmake/GpuBackendRegistry.cmake")
        lfs_generate_backend_registry()
        add_library(lfs_backend_configuration INTERFACE)
        target_include_directories(lfs_backend_configuration INTERFACE "${CMAKE_BINARY_DIR}/include")
    endif()
    # The EXR writer uses the same typed-error provider as the root build.
    find_package(nlohmann_json CONFIG REQUIRED)
    find_package(spdlog CONFIG REQUIRED)
    find_package(xxHash CONFIG REQUIRED)
    if(NOT TARGET lfs_diagnostics)
        add_subdirectory(${LFS_CODEC_ROOT}/src/diagnostics ${CMAKE_CURRENT_BINARY_DIR}/diagnostics)
    endif()
    if(NOT TARGET lfs_logger)
        add_subdirectory(${LFS_CODEC_ROOT}/src/core/logger ${CMAKE_CURRENT_BINARY_DIR}/logger)
    endif()
    if(NOT TARGET lfs_error)
        add_subdirectory(${LFS_CODEC_ROOT}/src/core/error ${CMAKE_CURRENT_BINARY_DIR}/error)
    endif()
    if(NOT TARGET lfs_file_io)
        add_subdirectory(${LFS_CODEC_ROOT}/src/core/file_io ${CMAKE_CURRENT_BINARY_DIR}/file-io)
    endif()
    add_subdirectory(${LFS_CODEC_ROOT}/src/core/image_codecs ${CMAKE_CURRENT_BINARY_DIR}/image-codecs)
endif()
add_executable(lichtfeld_image_codec_tests
    ../test_avif_encode.cpp
    ../test_jxl_codecs.cpp
    ../test_heif_codecs.cpp
    ../test_exr_codecs.cpp
    ../test_exr_memory.cpp
    ../test_image_codec_dependencies.cpp
    ../test_png_memory.cpp
    ../test_tiff_memory.cpp
    ../test_tiff_orientation.cpp
    ../test_stb_paths.cpp
    ${LFS_CODEC_ROOT}/src/io/stb_image_impl.cpp)
target_compile_features(lichtfeld_image_codec_tests PRIVATE cxx_std_20)
target_compile_definitions(lichtfeld_image_codec_tests PRIVATE LFS_HEIF_FIXTURES="${LFS_CODEC_ROOT}/tests/data/heif")
target_compile_definitions(lichtfeld_image_codec_tests PRIVATE LFS_JXL_FIXTURES="${LFS_CODEC_ROOT}/tests/data/jxl")
target_include_directories(lichtfeld_image_codec_tests PRIVATE ${LFS_CODEC_ROOT}/src ${LFS_CODEC_ROOT}/src/core/include)
target_include_directories(lichtfeld_image_codec_tests BEFORE PRIVATE ${ZLIB_INCLUDE_DIRS})
target_include_directories(lichtfeld_image_codec_tests SYSTEM PRIVATE ${LFS_CODEC_ROOT}/external)
target_link_libraries(lichtfeld_image_codec_tests PRIVATE
    lfs_image_codecs heif PkgConfig::JXL lcms2::lcms2
    GTest::gtest_main Threads::Threads OpenEXR::OpenEXRCore OpenEXR::OpenEXR
    JPEG::JPEG PNG::PNG TIFF::TIFF WebP::webp ZLIB::ZLIB)
if(MSVC)
    target_sources(lichtfeld_image_codec_tests PRIVATE ../test_heic_wic_unavailable.cpp ../test_heic_mf_unavailable.cpp)
    target_link_libraries(lichtfeld_image_codec_tests PRIVATE ole32 windowscodecs mfuuid)
    target_compile_options(lichtfeld_image_codec_tests PRIVATE /utf-8 /W4)
    target_compile_definitions(lichtfeld_image_codec_tests PRIVATE NOMINMAX)
endif()
if(WIN32 AND MSVC)
    target_sources(lichtfeld_image_codec_tests PRIVATE ../native_filesystem_test.manifest)
    target_compile_definitions(lichtfeld_image_codec_tests PRIVATE LFS_REQUIRE_LONG_PATHS_DISABLED)
endif()
add_test(NAME lichtfeld_image_codec_tests COMMAND lichtfeld_image_codec_tests
    --gtest_filter=-JxlCodec.EncoderLargeImagesPreserveSamplesAndRejectPixelBudget:JxlCodec.LargeImagesPreserveSamplesAndRecoverFromLimits:ProductionJpeg.MemoryLargeImagesPreserveFileEncodingAndRecoverFromLimits:PngMemory.ParallelStripesMatchFileBytesAndKeepInputUnmodified:PngMemory.LargeImagesPreserveSamplesAndRecoverFromOutputLimits:TiffMemory.LargeImagesPreserveSamplesAndRecoverFromOutputLimits:ExrMemory.LargeImagesPreserveSamplesAndRecoverFromLimits)
# This is the aggregate codec suite, including the EXR compression matrix.
# Allow the full matrix to complete on hosted Windows Debug runners.
set_tests_properties(lichtfeld_image_codec_tests PROPERTIES LABELS "fast;image;cpu" TIMEOUT 300)
add_test(NAME lichtfeld_jpeg_large_memory_test COMMAND lichtfeld_image_codec_tests
    --gtest_filter=ProductionJpeg.MemoryLargeImagesPreserveFileEncodingAndRecoverFromLimits)
set_tests_properties(lichtfeld_jpeg_large_memory_test PROPERTIES LABELS "stress;image;cpu" TIMEOUT 180)
add_test(NAME lichtfeld_png_memory_stress_test COMMAND lichtfeld_image_codec_tests
    --gtest_filter=PngMemory.ParallelStripesMatchFileBytesAndKeepInputUnmodified:PngMemory.LargeImagesPreserveSamplesAndRecoverFromOutputLimits)
set_tests_properties(lichtfeld_png_memory_stress_test PROPERTIES LABELS "stress;image;cpu" TIMEOUT 180)

add_test(NAME lichtfeld_tiff_memory_stress_test COMMAND lichtfeld_image_codec_tests
    --gtest_filter=TiffMemory.LargeImagesPreserveSamplesAndRecoverFromOutputLimits)
set_tests_properties(lichtfeld_tiff_memory_stress_test PROPERTIES LABELS "stress;image;cpu" TIMEOUT 180)

add_test(NAME lichtfeld_exr_memory_stress_test COMMAND lichtfeld_image_codec_tests
    --gtest_filter=ExrMemory.LargeImagesPreserveSamplesAndRecoverFromLimits)
set_tests_properties(lichtfeld_exr_memory_stress_test PROPERTIES LABELS "stress;image;cpu" TIMEOUT 180)

add_test(NAME lichtfeld_jxl_memory_stress_test COMMAND lichtfeld_image_codec_tests
    --gtest_filter=JxlCodec.LargeImagesPreserveSamplesAndRecoverFromLimits:JxlCodec.EncoderLargeImagesPreserveSamplesAndRejectPixelBudget)
set_tests_properties(lichtfeld_jxl_memory_stress_test PROPERTIES LABELS "stress;image;cpu" TIMEOUT 180)

if(TARGET lfs_io)
    add_executable(lichtfeld_image_discovery_tests ../test_image_format_discovery.cpp)
    target_include_directories(lichtfeld_image_discovery_tests PRIVATE ${LFS_CODEC_ROOT}/src)
    target_compile_features(lichtfeld_image_discovery_tests PRIVATE cxx_std_23)
    target_compile_definitions(lichtfeld_image_discovery_tests PRIVATE LFS_HEIF_FIXTURES="${LFS_CODEC_ROOT}/tests/data/heif")
    target_compile_definitions(lichtfeld_image_discovery_tests PRIVATE LFS_JXL_FIXTURES="${LFS_CODEC_ROOT}/tests/data/jxl")
    target_link_libraries(lichtfeld_image_discovery_tests PRIVATE lfs_io lfs_core lfs_image_codecs GTest::gtest_main)
    add_test(NAME lichtfeld_image_discovery_tests COMMAND lichtfeld_image_discovery_tests)
    set_tests_properties(lichtfeld_image_discovery_tests PROPERTIES LABELS "fast;image;cpu" TIMEOUT 60)
    if(WIN32)
        set_tests_properties(lichtfeld_image_discovery_tests PROPERTIES
            ENVIRONMENT_MODIFICATION "PATH=path_list_prepend:$<TARGET_FILE_DIR:lfs_core>")
    endif()
endif()

# Exercise the actual import/export boundary; direct-source codec tests cannot
# detect missing exports from the hidden-visibility shared library.

add_library(lichtfeld_codec_exports_consumer STATIC ../test_image_codec_exports.cpp)
target_include_directories(lichtfeld_codec_exports_consumer PRIVATE ${LFS_CODEC_ROOT}/src)
target_link_libraries(lichtfeld_codec_exports_consumer PRIVATE lfs_image_codecs)
add_executable(lichtfeld_codec_exports_test test_codec_exports_main.cpp)
target_link_libraries(lichtfeld_codec_exports_test PRIVATE lichtfeld_codec_exports_consumer)
add_test(NAME lichtfeld_codec_exports_test COMMAND lichtfeld_codec_exports_test)
set_tests_properties(lichtfeld_codec_exports_test PROPERTIES LABELS "fast;image;cpu" TIMEOUT 30)
if(WIN32)
    # Project DLLs live beside the app, not beside these nested test executables.
    set_tests_properties(lichtfeld_image_codec_tests lichtfeld_jpeg_large_memory_test lichtfeld_png_memory_stress_test lichtfeld_tiff_memory_stress_test lichtfeld_codec_exports_test PROPERTIES
        ENVIRONMENT_MODIFICATION "PATH=path_list_prepend:$<TARGET_FILE_DIR:lfs_image_codecs>")
endif()

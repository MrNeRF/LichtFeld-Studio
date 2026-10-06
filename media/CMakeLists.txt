# SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
# SPDX-License-Identifier: GPL-3.0-or-later
cmake_minimum_required(VERSION 3.24)
project(LichtFeldMedia VERSION 0.1.0 LANGUAGES CXX)
include(GNUInstallDirs)
include(CMakePackageConfigHelpers)
option(LFS_MEDIA_BUILD_CLI "Build media-ingest CPU command-line interface" ON)
get_filename_component(LFS_MEDIA_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/.." ABSOLUTE)
include(cmake/MediaDependencies.cmake)
add_library(lfs_media STATIC
    ${LFS_MEDIA_ROOT}/src/media/media_ingest.cpp
    ${LFS_MEDIA_ROOT}/src/media/host_contract.cpp
    ${LFS_MEDIA_ROOT}/src/media/hdr_cpu_unavailable.cpp
    ${LFS_MEDIA_ROOT}/src/io/media_probe.cpp
    ${LFS_MEDIA_ROOT}/src/io/frame_sink.cpp
    ${LFS_MEDIA_ROOT}/src/io/file_frame_sink.cpp
    ${LFS_MEDIA_ROOT}/src/io/video_frame_extractor.cpp
    ${LFS_MEDIA_ROOT}/src/io/hdr_tonemap.cpp
    ${LFS_MEDIA_ROOT}/src/core/error.cpp
    ${LFS_MEDIA_ROOT}/src/core/memory_domain.cpp
    ${LFS_MEDIA_ROOT}/src/core/image_codecs.cpp
    ${LFS_MEDIA_ROOT}/src/core/image_exr.cpp)
add_library(LichtFeldMedia::media ALIAS lfs_media)
set_target_properties(lfs_media PROPERTIES EXPORT_NAME media POSITION_INDEPENDENT_CODE ON)
target_compile_features(lfs_media PUBLIC cxx_std_23)
target_compile_definitions(lfs_media PUBLIC LFS_STATIC_BUILD PRIVATE LFS_HAS_CUDA=0 LFS_MEDIA_CPU_ONLY=1)
target_include_directories(lfs_media PUBLIC
    $<BUILD_INTERFACE:${LFS_MEDIA_ROOT}/src/io/include>
    $<BUILD_INTERFACE:${LFS_MEDIA_ROOT}/src/core/include>
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
    PRIVATE ${LFS_MEDIA_ROOT}/src ${LFS_MEDIA_ROOT}/src/io ${LFS_MEDIA_ROOT}/external)
target_include_directories(lfs_media BEFORE PRIVATE ${ZLIB_INCLUDE_DIRS})
target_link_libraries(lfs_media PRIVATE LichtFeldMedia::ffmpeg nlohmann_json::nlohmann_json
    OpenEXR::OpenEXRCore JPEG::JPEG PNG::PNG ZLIB::ZLIB TIFF::TIFF WebP::webp Threads::Threads)
if(MSVC)
    target_compile_options(lfs_media PRIVATE /utf-8 /W4)
    target_compile_definitions(lfs_media PUBLIC NOMINMAX)
endif()
if(LFS_MEDIA_BUILD_CLI)
    add_executable(media-ingest ${LFS_MEDIA_ROOT}/src/media/cli.cpp)
    target_link_libraries(media-ingest PRIVATE lfs_media nlohmann_json::nlohmann_json LichtFeldMedia::ffmpeg)
    target_include_directories(media-ingest PRIVATE ${LFS_MEDIA_ROOT}/src)
    if(MSVC)
        target_compile_options(media-ingest PRIVATE /utf-8 /W4)
    endif()
    install(TARGETS media-ingest RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})
endif()
install(TARGETS lfs_media EXPORT LichtFeldMediaTargets ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR})
install(DIRECTORY ${LFS_MEDIA_ROOT}/src/io/include/io/media DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/io)
foreach(_header error.hpp error_codes.hpp assert.hpp export.hpp source_site.hpp failure_report.hpp)
    install(FILES ${LFS_MEDIA_ROOT}/src/core/include/core/${_header} DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/core)
endforeach()
configure_package_config_file(cmake/LichtFeldMediaConfig.cmake.in
    ${CMAKE_CURRENT_BINARY_DIR}/LichtFeldMediaConfig.cmake
    INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/LichtFeldMedia)
write_basic_package_version_file(${CMAKE_CURRENT_BINARY_DIR}/LichtFeldMediaConfigVersion.cmake
    VERSION ${PROJECT_VERSION} COMPATIBILITY SameMajorVersion)
install(EXPORT LichtFeldMediaTargets NAMESPACE LichtFeldMedia:: DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/LichtFeldMedia)
install(FILES ${CMAKE_CURRENT_BINARY_DIR}/LichtFeldMediaConfig.cmake
              ${CMAKE_CURRENT_BINARY_DIR}/LichtFeldMediaConfigVersion.cmake
              cmake/MediaDependencies.cmake DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/LichtFeldMedia)
install(FILES ${LFS_MEDIA_ROOT}/LICENSE DESTINATION ${CMAKE_INSTALL_DATADIR}/lichtfeld-media)

# Preserve notices from the dependency prefix actually reused by this profile.
# Runtime DLL redistribution/source obligations are a separate release step.
if(DEFINED VCPKG_INSTALLED_DIR AND DEFINED VCPKG_TARGET_TRIPLET)
    foreach(_port ffmpeg libjpeg-turbo libpng zlib tiff libwebp nlohmann-json openexr
                  imath libdeflate bzip2 liblzma zstd x264 openh264)
        set(_notice "${VCPKG_INSTALLED_DIR}/${VCPKG_TARGET_TRIPLET}/share/${_port}/copyright")
        if(EXISTS "${_notice}")
            install(FILES "${_notice}" DESTINATION ${CMAKE_INSTALL_DATADIR}/lichtfeld-media/licenses
                RENAME ${_port}-copyright.txt)
        endif()
    endforeach()
endif()

install(FILES stb-image-license.txt DESTINATION ${CMAKE_INSTALL_DATADIR}/lichtfeld-media/licenses)

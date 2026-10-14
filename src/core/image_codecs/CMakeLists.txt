# SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
# SPDX-License-Identifier: GPL-3.0-or-later
find_package(libheif CONFIG REQUIRED)
find_package(lcms2 CONFIG REQUIRED)
add_library(lfs_image_codecs SHARED ../image_codecs.cpp ../image_heif.cpp ../image_heic_system.cpp ../image_exr.cpp ../image_exr_writer.cpp)
target_compile_definitions(lfs_image_codecs PRIVATE LFS_IMAGE_CODECS_EXPORTS)
target_include_directories(lfs_image_codecs PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/../include"
    PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/../.." "${CMAKE_CURRENT_SOURCE_DIR}/.."
    "${CMAKE_CURRENT_SOURCE_DIR}/../../../external")
target_include_directories(lfs_image_codecs BEFORE PRIVATE ${ZLIB_INCLUDE_DIRS})
target_link_libraries(lfs_image_codecs PUBLIC lfs_error lfs_file_io PRIVATE PNG::PNG JPEG::JPEG ZLIB::ZLIB
    heif lcms2::lcms2 TIFF::TIFF WebP::webp OpenEXR::OpenEXRCore Threads::Threads nlohmann_json::nlohmann_json)
set_target_properties(lfs_image_codecs PROPERTIES
    POSITION_INDEPENDENT_CODE ON CXX_STANDARD 23 CXX_STANDARD_REQUIRED ON
    CXX_VISIBILITY_PRESET hidden VISIBILITY_INLINES_HIDDEN ON
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}"
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}")
target_compile_features(lfs_image_codecs PUBLIC cxx_std_23)
if(WIN32)
    target_link_libraries(lfs_image_codecs PRIVATE windowscodecs ole32 mfplat mfuuid)
elseif(APPLE)
    target_link_libraries(lfs_image_codecs PRIVATE "-framework ImageIO" "-framework CoreGraphics" "-framework CoreFoundation")
endif()
if(MSVC)
    target_compile_options(lfs_image_codecs PRIVATE /utf-8 /W4)
    target_compile_definitions(lfs_image_codecs PUBLIC NOMINMAX)
endif()

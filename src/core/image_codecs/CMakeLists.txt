# SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
# SPDX-License-Identifier: GPL-3.0-or-later
add_library(lfs_image_codecs SHARED ../image_codecs.cpp ../image_exr.cpp)
target_compile_definitions(lfs_image_codecs PRIVATE LFS_IMAGE_CODECS_EXPORTS)
target_include_directories(lfs_image_codecs PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/../include"
    PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/../.." "${CMAKE_CURRENT_SOURCE_DIR}/.."
    "${CMAKE_CURRENT_SOURCE_DIR}/../../../external")
target_include_directories(lfs_image_codecs BEFORE PRIVATE ${ZLIB_INCLUDE_DIRS})
target_link_libraries(lfs_image_codecs PRIVATE PNG::PNG JPEG::JPEG ZLIB::ZLIB
    TIFF::TIFF WebP::webp OpenEXR::OpenEXRCore Threads::Threads)
set_target_properties(lfs_image_codecs PROPERTIES
    POSITION_INDEPENDENT_CODE ON CXX_STANDARD 23 CXX_STANDARD_REQUIRED ON
    CXX_VISIBILITY_PRESET hidden VISIBILITY_INLINES_HIDDEN ON
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}"
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}")
target_compile_features(lfs_image_codecs PUBLIC cxx_std_23)
if(MSVC)
    target_compile_options(lfs_image_codecs PRIVATE /utf-8 /W4)
    target_compile_definitions(lfs_image_codecs PUBLIC NOMINMAX)
endif()

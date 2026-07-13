# SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
#
# SPDX-License-Identifier: GPL-3.0-or-later

find_package(CURL REQUIRED)
find_package(OpenSSL REQUIRED)

if(NOT LFS_ONNXRUNTIME_TARGET OR NOT TARGET "${LFS_ONNXRUNTIME_TARGET}")
    message(FATAL_ERROR "ONNX Runtime target was not configured; include cmake/SetupOnnxRuntime.cmake before src/preprocessing")
endif()
if(NOT LFS_ONNXRUNTIME_INCLUDE_DIR)
    message(FATAL_ERROR "ONNX Runtime include directory was not configured")
endif()

add_library(lfs_preprocessing STATIC
    preprocess.cpp
    include/preprocessing/preprocess.hpp
)

target_include_directories(lfs_preprocessing
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>
    PRIVATE
        ${CMAKE_SOURCE_DIR}/src
        ${CMAKE_SOURCE_DIR}/src/core/include
        ${CMAKE_SOURCE_DIR}/external
        ${LFS_ONNXRUNTIME_INCLUDE_DIR}
)

target_link_libraries(lfs_preprocessing
    PUBLIC
        lfs_core
    PRIVATE
        lfs_io                  # Scene import (Loader) for depth-anchor precompute
        lfs_depth_anchor        # Shared depth-anchor fit + sidecar cache
        ${LFS_ONNXRUNTIME_TARGET}
        CURL::libcurl
        OpenSSL::Crypto
        OpenImageIO::OpenImageIO
)

target_compile_features(lfs_preprocessing PUBLIC cxx_std_23)

set_target_properties(lfs_preprocessing PROPERTIES
    CXX_STANDARD 23
    CXX_STANDARD_REQUIRED ON
    POSITION_INDEPENDENT_CODE ON
)

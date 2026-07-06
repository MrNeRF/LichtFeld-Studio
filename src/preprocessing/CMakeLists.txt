# SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
#
# SPDX-License-Identifier: GPL-3.0-or-later

find_package(CURL REQUIRED)
find_package(OpenSSL REQUIRED)
find_package(onnxruntime CONFIG REQUIRED)
find_path(LFS_ONNXRUNTIME_INCLUDE_DIR
    NAMES onnxruntime_cxx_api.h
    PATH_SUFFIXES onnxruntime onnxruntime/core/session
    REQUIRED
)

if(TARGET onnxruntime::onnxruntime)
    set(LFS_ONNXRUNTIME_TARGET onnxruntime::onnxruntime)
elseif(TARGET onnxruntime)
    set(LFS_ONNXRUNTIME_TARGET onnxruntime)
else()
    find_library(LFS_ONNXRUNTIME_LIBRARY NAMES onnxruntime REQUIRED)
    find_path(LFS_ONNXRUNTIME_INCLUDE_DIR onnxruntime_cxx_api.h REQUIRED)
    add_library(lfs_onnxruntime_external UNKNOWN IMPORTED)
    set_target_properties(lfs_onnxruntime_external PROPERTIES
        IMPORTED_LOCATION "${LFS_ONNXRUNTIME_LIBRARY}"
        INTERFACE_INCLUDE_DIRECTORIES "${LFS_ONNXRUNTIME_INCLUDE_DIR}")
    set(LFS_ONNXRUNTIME_TARGET lfs_onnxruntime_external)
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

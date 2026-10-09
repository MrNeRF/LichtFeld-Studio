# SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
#
# SPDX-License-Identifier: GPL-3.0-or-later

find_package(CURL REQUIRED)
find_package(OpenSSL REQUIRED)

add_library(lfs_preprocessing STATIC
    preprocess.cpp
    lens_priors.cpp
    lens_views.cpp
    lens_views.hpp
    include/preprocessing/preprocess.hpp
    include/preprocessing/lens_priors.hpp
)

target_include_directories(lfs_preprocessing
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>
    PRIVATE
        ${CMAKE_SOURCE_DIR}/src
        ${CMAKE_SOURCE_DIR}/src/core/include
        ${CMAKE_SOURCE_DIR}/external
)

target_link_libraries(lfs_preprocessing
    PUBLIC
        lfs_core
    PRIVATE
        lfs_io                  # Scene import (Loader) for depth-anchor precompute
        $<$<BOOL:${LFS_BUILD_TRAINER}>:lfs_depth_anchor>
        CURL::libcurl
        OpenSSL::Crypto
)

target_compile_features(lfs_preprocessing PUBLIC cxx_std_23)

set_target_properties(lfs_preprocessing PROPERTIES
    CXX_STANDARD 23
    CXX_STANDARD_REQUIRED ON
    POSITION_INDEPENDENT_CODE ON
    CUDA_STANDARD 20
    CUDA_STANDARD_REQUIRED ON
    CUDA_SEPARABLE_COMPILATION OFF
)

lfs_add_gpu_program(lfs_preprocessing lens_views_program SOURCE lens_views.slang COMPUTE setup rays sampleFace projectFace blend
    DEPENDS "${CMAKE_SOURCE_DIR}/src/training/vulkan/shaders/gsplat_common.slangh"
            "${CMAKE_SOURCE_DIR}/src/core/tensor/backend/vulkan/shaders/sh_decode.slangh")

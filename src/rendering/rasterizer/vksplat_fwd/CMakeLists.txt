# SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
#
# SPDX-License-Identifier: GPL-3.0-or-later

add_library(lfs_vksplat_fwd STATIC
        src/buffer.cpp
        src/gs_pipeline.cpp
        src/gs_renderer.cpp
        src/perf_timer.cpp
)

target_include_directories(lfs_vksplat_fwd
        PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/src
        ${CMAKE_CURRENT_SOURCE_DIR}
)

target_link_libraries(lfs_vksplat_fwd
        PUBLIC
        Vulkan::Vulkan
)

target_compile_features(lfs_vksplat_fwd PUBLIC cxx_std_23)
set_target_properties(lfs_vksplat_fwd PROPERTIES
        POSITION_INDEPENDENT_CODE ON
)

target_compile_definitions(lfs_vksplat_fwd PRIVATE
        $<$<PLATFORM_ID:Windows>:NOMINMAX>
)

set(LFS_VKSPLAT_SPV_DIR "${CMAKE_CURRENT_SOURCE_DIR}/shader")
target_compile_definitions(lfs_vksplat_fwd PUBLIC
        LFS_VKSPLAT_SPV_DIR="${LFS_VKSPLAT_SPV_DIR}/"
)

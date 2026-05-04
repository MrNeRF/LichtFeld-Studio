# SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
#
# SPDX-License-Identifier: GPL-3.0-or-later

add_subdirectory(rasterizer)

set(RENDERING_TENSOR_SOURCES
        cuda_vulkan_interop.cpp
        cuda_vulkan_interop.cu
        gs_rasterizer_tensor.cpp
        raster_rendering_engine.cpp
        cuda_interaction_kernels.cu
        point_cloud_raster.cu
        mesh2splat_cpu.cpp
        screen_overlay_renderer.cpp
)

add_library(lfs_rendering_tensor STATIC ${RENDERING_TENSOR_SOURCES})

target_link_libraries(lfs_rendering_tensor
        PUBLIC
        lfs_core
        PRIVATE
        lfs_rasterizer
        CUDA::cudart
        glm::glm
        OpenImageIO::OpenImageIO
        spdlog::spdlog
        TBB::tbb
)

target_include_directories(lfs_rendering_tensor
        PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
        $<INSTALL_INTERFACE:include>
        PRIVATE
        ${CMAKE_SOURCE_DIR}/src
)

target_compile_definitions(lfs_rendering_tensor PRIVATE
        $<$<PLATFORM_ID:Windows>:NOMINMAX>
)

add_library(lfs_rendering INTERFACE)
target_link_libraries(lfs_rendering INTERFACE lfs_rendering_tensor)

if(NOT BUILD_PORTABLE)
    install(FILES ${CMAKE_CURRENT_SOURCE_DIR}/include/rendering/rendering.hpp
                  ${CMAKE_CURRENT_SOURCE_DIR}/include/rendering/frame_contract.hpp
                  ${CMAKE_CURRENT_SOURCE_DIR}/include/rendering/mesh2splat.hpp
            DESTINATION include/rendering)
endif()

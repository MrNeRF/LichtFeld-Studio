# SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
#
# SPDX-License-Identifier: GPL-3.0-or-later

if(LFS_USE_VULKAN)
    add_library(lfs_vulkan_runtime STATIC vulkan_wait.cpp)
    target_include_directories(lfs_vulkan_runtime PUBLIC
        "${CMAKE_SOURCE_DIR}/src" "${CMAKE_SOURCE_DIR}/src/core/include")
    target_compile_features(lfs_vulkan_runtime PUBLIC cxx_std_23)
    set_target_properties(lfs_vulkan_runtime PROPERTIES POSITION_INDEPENDENT_CODE ON)
    target_link_libraries(lfs_vulkan_runtime PUBLIC Vulkan::Vulkan lfs_core)
    # Mesh2Splat compiles GLSL to SPIR-V at runtime on the Vulkan path.
    find_package(glslang CONFIG REQUIRED)
endif()
add_subdirectory(rasterizer)

set(RENDERING_TENSOR_SOURCES
        mesh2splat_shared.cpp
        mesh2splat_tensor.cpp
        selection_group_count.cpp
        selection_screen_ops.cpp
        selection_mask_ops.cpp
        raster_rendering_engine.cpp
        export_post_process.cpp
        screen_overlay_renderer.cpp
)
# The Vulkan converter needs geometry shaders, which MoltenVK lacks; Apple
# builds use the tensor converter.
if(LFS_USE_VULKAN AND NOT APPLE)
    list(APPEND RENDERING_TENSOR_SOURCES mesh2splat.cpp)
endif()
if(LFS_HAS_CUDA)
    list(APPEND RENDERING_TENSOR_SOURCES rasterizer/cuda/point_cloud_raster.cu)
endif()

add_library(lfs_rendering_tensor STATIC ${RENDERING_TENSOR_SOURCES})
lfs_add_gpu_program(lfs_rendering_tensor mesh2splat_program SOURCE mesh2splat.slang
    VERTEX mesh2splatVertex FRAGMENT mesh2splatFragment)

target_link_libraries(lfs_rendering_tensor
        PUBLIC
        lfs_core
        PRIVATE
        $<$<BOOL:${LFS_HAS_CUDA}>:CUDA::cudart>
        glm::glm
        spdlog::spdlog
        TBB::tbb
)
if(LFS_USE_VULKAN)
    target_link_libraries(lfs_rendering_tensor PRIVATE
        Vulkan::Vulkan
        glslang::glslang
        glslang::SPIRV
        glslang::glslang-default-resource-limits
        lfs_vulkan_runtime)
endif()

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
        $<$<OR:$<NOT:$<BOOL:${LFS_USE_VULKAN}>>,$<PLATFORM_ID:Darwin>>:LFS_MESH2SPLAT_TENSOR_DEFAULT>
)

add_library(lfs_rendering INTERFACE)
target_link_libraries(lfs_rendering INTERFACE lfs_rendering_tensor)

if(NOT BUILD_PORTABLE)
    install(FILES ${CMAKE_CURRENT_SOURCE_DIR}/include/rendering/rendering.hpp
                  ${CMAKE_CURRENT_SOURCE_DIR}/include/rendering/frame_contract.hpp
                  ${CMAKE_CURRENT_SOURCE_DIR}/include/rendering/mesh2splat.hpp
            DESTINATION include/rendering)
endif()

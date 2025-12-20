# SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
#
# SPDX-License-Identifier: GPL-3.0-or-later

add_subdirectory(rasterizer)

set(RENDERING_SOURCES
        rendering_engine.cpp
        rendering_engine_impl.cpp
        rendering_pipeline.cpp
        gs_rasterizer_tensor.cpp
        point_cloud_renderer.cpp
        grid_renderer.cpp
        bbox_renderer.cpp
        axes_renderer.cpp
        pivot_renderer.cpp
        text_renderer.cpp
        viewport_gizmo.cpp
        framebuffer_factory.cpp
        shader_manager.cpp
        screen_renderer.cpp
        split_view_renderer.cpp
        camera_frustum_renderer.cpp
)

# Conditionally add CUDA GL interop
if(CUDA_GL_INTEROP_ENABLED)
    list(APPEND RENDERING_SOURCES
        cuda_gl_interop.cpp
        cuda_gl_kernels.cu)
endif()

add_library(lfs_rendering STATIC ${RENDERING_SOURCES})

target_link_libraries(lfs_rendering
        PUBLIC
        lfs_core                # Link core module
        lfs_geometry            # LibTorch-free geometry module
        # gs_kernels              # Temporarily disabled - requires LibTorch
        PRIVATE
        lfs_rasterizer
        lfs_training            # For gsplat rasterizer in GUT mode
        glm::glm
        glad::glad
        glfw
        ${OPENGL_LIBRARIES}
        CUDA::cudart
        spdlog::spdlog
        Freetype::Freetype
)

# Since we're linking to lfs_core PUBLIC, we inherit its include directories
# which include: ${CMAKE_BINARY_DIR}/include (for config.h)
#                ${CMAKE_SOURCE_DIR}/gsplat (for Common.h)
target_include_directories(lfs_rendering
        PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>
        PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${CMAKE_SOURCE_DIR}/src
        ${OPENGL_INCLUDE_DIRS}
)

# Handle shaders
set(RENDERING_BUILD_RESOURCE_DIR "${CMAKE_CURRENT_BINARY_DIR}/resources")
set(RENDERING_SOURCE_RESOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/resources")

file(MAKE_DIRECTORY "${RENDERING_BUILD_RESOURCE_DIR}/shaders")

file(GLOB SHADER_FILES "${RENDERING_SOURCE_RESOURCE_DIR}/shaders/*")
foreach(SHADER_FILE ${SHADER_FILES})
    get_filename_component(SHADER_NAME ${SHADER_FILE} NAME)
    configure_file(${SHADER_FILE} "${RENDERING_BUILD_RESOURCE_DIR}/shaders/${SHADER_NAME}" COPYONLY)
endforeach()

target_compile_definitions(lfs_rendering PRIVATE
        SHADER_PATH="${RENDERING_BUILD_RESOURCE_DIR}/shaders"
        RENDERING_SOURCE_SHADER_PATH="${RENDERING_SOURCE_RESOURCE_DIR}/shaders"
        $<$<BOOL:${CUDA_GL_INTEROP_ENABLED}>:CUDA_GL_INTEROP_ENABLED>
        $<$<PLATFORM_ID:Windows>:NOMINMAX>
)

# Export only the public header
install(FILES ${CMAKE_CURRENT_SOURCE_DIR}/include/rendering/rendering.hpp
        DESTINATION include/rendering
)

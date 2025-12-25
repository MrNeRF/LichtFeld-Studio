# SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
#
# SPDX-License-Identifier: GPL-3.0-or-later

add_library(lfs_visualizer STATIC
        # Core
        visualizer.cpp
        visualizer_impl.cpp
        core/main_loop.cpp
        core/data_loading_service.cpp
        core/editor_context.cpp
        core/parameter_manager.cpp

        # Window management
        window/window_manager.cpp

        # Rendering
        rendering/rendering_manager.cpp
        rendering/framerate_controller.cpp

        # Theme system
        theme/theme.cpp

        # GUI system
        gui/gui_manager.cpp
        gui/html_viewer_export.cpp
        gui/ui_widgets.cpp
        gui/panels/main_panel.cpp
        gui/panels/training_panel.cpp
        gui/panels/crop_box_panel.cpp
        gui/panels/transform_panel.cpp
        gui/panels/gizmo_toolbar.cpp
        gui/panels/scene_panel.cpp
        gui/panels/tools_panel.cpp
        gui/windows/file_browser.cpp
        gui/windows/image_preview.cpp
        gui/windows/export_dialog.cpp
        gui/windows/notification_popup.cpp
        gui/windows/save_directory_popup.cpp
        gui/windows/exit_confirmation_popup.cpp

        # Scene management
        scene/scene.cpp
        scene/scene_manager.cpp

        # Input handling
        input/input_controller.cpp
        input/input_bindings.cpp

        # Training integration
        training/training_state.cpp
        training/training_manager.cpp

        # Tools
        tools/brush_tool.cpp
        tools/align_tool.cpp
        tools/selection_tool.cpp

        # Command system (undo/redo)
        command/command_history.cpp
        command/commands/cropbox_command.cpp
        command/commands/crop_command.cpp
        command/commands/selection_command.cpp
        command/commands/saturation_command.cpp
        command/commands/mirror_command.cpp
        command/commands/transform_command.cpp
        gui/panels/menu_bar.cpp
        gui/panels/menu_bar.hpp
        gui/utils/windows_utils.cpp
        gui/utils/drag_drop_native.cpp
)

# Set include directories
target_include_directories(lfs_visualizer
        PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>
        PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${CMAKE_SOURCE_DIR}/src
        ${CMAKE_SOURCE_DIR}/external/httplib
        ${OPENGL_INCLUDE_DIRS}
)

# Link dependencies
target_link_libraries(lfs_visualizer
        PRIVATE
        lfs_core           # Use modular core library (LibTorch-free)
        lfs_geometry       # LibTorch-free geometry module
        lfs_io
        lfs_rendering
        lfs_training
        glad::glad
        glfw
        imgui::imgui
        imguizmo::imguizmo # 3D manipulation gizmo
        glm::glm
        TBB::tbb
        spdlog::spdlog
        nlohmann_json::nlohmann_json
        OpenSSL::SSL
        OpenSSL::Crypto
        CUDA::cudart
        ${OPENGL_LIBRARIES}
)

# Set C++ standard
target_compile_features(lfs_visualizer PUBLIC cxx_std_23)

# Compiler options
target_compile_options(lfs_visualizer PRIVATE
        $<$<CXX_COMPILER_FRONTEND_VARIANT:GNU>:-Wall -Wextra -Wpedantic>
        $<$<CXX_COMPILER_FRONTEND_VARIANT:MSVC>:/W4>
        $<$<AND:$<CONFIG:Debug>,$<CXX_COMPILER_ID:GNU,Clang>>:-O0 -g>
        $<$<AND:$<CONFIG:Release>,$<CXX_COMPILER_ID:GNU,Clang>>:-O3>
)

# Handle resources (shaders and assets)
set(VISUALIZER_BUILD_RESOURCE_DIR "${CMAKE_CURRENT_BINARY_DIR}/resources")
set(VISUALIZER_SOURCE_RESOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/resources")

# Create resource directories in build folder
file(MAKE_DIRECTORY "${VISUALIZER_BUILD_RESOURCE_DIR}/shaders")
file(MAKE_DIRECTORY "${VISUALIZER_BUILD_RESOURCE_DIR}/assets")

# Copy shaders to build directory
file(GLOB SHADER_FILES "${VISUALIZER_SOURCE_RESOURCE_DIR}/shaders/*")
foreach (SHADER_FILE ${SHADER_FILES})
    get_filename_component(SHADER_NAME ${SHADER_FILE} NAME)
    configure_file(${SHADER_FILE} "${VISUALIZER_BUILD_RESOURCE_DIR}/shaders/${SHADER_NAME}" COPYONLY)
endforeach ()

# Copy assets from resources/assets if they exist
file(GLOB ASSET_FILES "${VISUALIZER_SOURCE_RESOURCE_DIR}/assets/*")
foreach (ASSET_FILE ${ASSET_FILES})
    get_filename_component(ASSET_NAME ${ASSET_FILE} NAME)
    configure_file(${ASSET_FILE} "${VISUALIZER_BUILD_RESOURCE_DIR}/assets/${ASSET_NAME}" COPYONLY)
endforeach ()

set(GUI_ASSET_FILES
        "${CMAKE_CURRENT_SOURCE_DIR}/gui/assets/JetBrainsMono-Regular.ttf"
        "${CMAKE_CURRENT_SOURCE_DIR}/gui/assets/lichtfeld-icon.png"
)

foreach (GUI_ASSET_FILE ${GUI_ASSET_FILES})
    if (EXISTS ${GUI_ASSET_FILE} AND NOT IS_DIRECTORY ${GUI_ASSET_FILE})
        get_filename_component(ASSET_NAME ${GUI_ASSET_FILE} NAME)
        configure_file(${GUI_ASSET_FILE} "${VISUALIZER_BUILD_RESOURCE_DIR}/assets/${ASSET_NAME}" COPYONLY)
    endif ()
endforeach ()

# Copy icon files
file(MAKE_DIRECTORY "${VISUALIZER_BUILD_RESOURCE_DIR}/assets/icon")
file(GLOB ICON_FILES "${CMAKE_CURRENT_SOURCE_DIR}/gui/assets/icon/*.png")
foreach (ICON_FILE ${ICON_FILES})
    get_filename_component(ICON_NAME ${ICON_FILE} NAME)
    configure_file(${ICON_FILE} "${VISUALIZER_BUILD_RESOURCE_DIR}/assets/icon/${ICON_NAME}" COPYONLY)
endforeach ()

# Compile definitions
target_compile_definitions(lfs_visualizer PRIVATE
        $<$<BOOL:${CUDA_GL_INTEROP_ENABLED}>:CUDA_GL_INTEROP_ENABLED>
        PROJECT_ROOT_PATH="${PROJECT_SOURCE_DIR}"
        VISUALIZER_SHADER_PATH="${VISUALIZER_BUILD_RESOURCE_DIR}/shaders"
        VISUALIZER_ASSET_PATH="${VISUALIZER_BUILD_RESOURCE_DIR}/assets"
        VISUALIZER_SOURCE_SHADER_PATH="${VISUALIZER_SOURCE_RESOURCE_DIR}/shaders"
        VISUALIZER_SOURCE_ASSET_PATH="${VISUALIZER_SOURCE_RESOURCE_DIR}/assets"
        $<$<CONFIG:Debug>:DEBUG_BUILD>
        $<$<CONFIG:Release>:RELEASE_BUILD>
)

# Platform-specific settings
if (WIN32)
    target_compile_definitions(lfs_visualizer PRIVATE
            NOMINMAX
            _USE_MATH_DEFINES
    )
    target_link_libraries(lfs_visualizer PRIVATE
            Ole32       # OLE drag-drop (IDropTarget)
    )
elseif (UNIX AND NOT APPLE)
    target_link_libraries(lfs_visualizer PRIVATE
            GL
            GLU
            X11
    )
endif ()

# Export only the public header
install(FILES ${CMAKE_CURRENT_SOURCE_DIR}/include/visualizer/visualizer.hpp
        DESTINATION include/visualizer
)

# Optionally install resources
install(DIRECTORY ${VISUALIZER_SOURCE_RESOURCE_DIR}/
        DESTINATION share/LichtFeld-Studio/visualizer/resources
        FILES_MATCHING
        PATTERN "*.vert"
        PATTERN "*.frag"
        PATTERN "*.ttf"
)

# Add custom target for resource files (makes them show up in IDEs)
file(GLOB_RECURSE VISUALIZER_RESOURCE_FILES
        "${VISUALIZER_SOURCE_RESOURCE_DIR}/shaders/*.vert"
        "${VISUALIZER_SOURCE_RESOURCE_DIR}/shaders/*.frag"
        "${VISUALIZER_SOURCE_RESOURCE_DIR}/assets/*.ttf"
)
if (VISUALIZER_RESOURCE_FILES)
    add_custom_target(visualizer_resources SOURCES ${VISUALIZER_RESOURCE_FILES})
endif ()

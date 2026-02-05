# SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
#
# SPDX-License-Identifier: GPL-3.0-or-later

# Fetch libvterm for terminal emulation
include(${CMAKE_SOURCE_DIR}/cmake/FetchLibvterm.cmake)

# SHARED on Windows to avoid singleton duplication across .exe/.pyd boundary
if(WIN32)
    set(LFS_VIS_LIB_TYPE SHARED)
else()
    set(LFS_VIS_LIB_TYPE STATIC)
endif()

add_library(lfs_visualizer ${LFS_VIS_LIB_TYPE}
        # Core
        visualizer.cpp
        visualizer_impl.cpp
        core/main_loop.cpp
        core/data_loading_service.cpp
        core/editor_context.cpp
        core/parameter_manager.cpp
        core/services.cpp

        # Window management
        window/window_manager.cpp

        # Rendering
        rendering/rendering_manager.cpp
        rendering/framerate_controller.cpp

        # Theme system
        theme/theme.cpp

        # GUI system
        gui/gui_manager.cpp
        gui/gizmo_manager.cpp
        gui/panel_layout.cpp
        gui/layout_state.cpp
        gui/async_task_manager.cpp
        gui/startup_overlay.cpp
        gui/sequencer_ui_manager.cpp
        gui/gizmo_transform.cpp
        gui/html_viewer_export.cpp
        gui/ui_widgets.cpp
        gui/panels/windows_console_utils.cpp
        gui/panels/tools_panel.cpp
        gui/panels/python_console_panel.cpp
        gui/panels/python_scripts_panel.cpp
        gui/editor/python_language_def.cpp
        gui/editor/editor_theme.cpp
        gui/editor/autocomplete/static_providers.cpp
        gui/editor/autocomplete/autocomplete_manager.cpp
        gui/editor/autocomplete/python_introspector.cpp
        gui/editor/python_editor.cpp
        gui/editor/console_output.cpp

        # Terminal emulator
        gui/terminal/pty_process.cpp
        gui/terminal/terminal_widget.cpp
        gui/windows/file_browser.cpp
        gui/windows/disk_space_error_dialog.cpp
        gui/windows/video_extractor_dialog.cpp

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
        tools/selection_tool.cpp
        tools/brush_tool.cpp
        tools/align_tool.cpp
        tools/unified_tool_registry.cpp
        tools/builtin_tools.cpp

        # Icon cache
        gui/icon_cache.cpp

        # Selection system
        selection/selection_service.cpp

        # Sequencer (camera animation)
        sequencer/sequencer_controller.cpp
        sequencer/sequencer_panel.cpp

        # Operator system (Blender-style)
        operator/operator_context.cpp
        operator/operator_id.cpp
        operator/operator_registry.cpp
        operator/property_schema.cpp
        operator/ops/align_ops.cpp
        operator/ops/brush_ops.cpp
        operator/ops/edit_ops.cpp
        operator/ops/selection_ops.cpp
        operator/ops/transform_ops.cpp

        # Compositional operations system
        operation/undo_entry.cpp
        operation/undo_history.cpp
        operation/operation.cpp
        operation/pipeline.cpp
        operation/ops/select_ops.cpp
        operation/ops/edit_ops.cpp
        operation/ops/transform_ops.cpp

        # Python hook invoker (DLL boundary singleton)
        ${CMAKE_SOURCE_DIR}/src/python/ui_hooks.cpp

        # IPC for MCP integration
        ipc/selection_server.cpp
        ipc/view_context.cpp

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

target_link_libraries(lfs_visualizer
        PRIVATE
        lfs_core
        lfs_geometry
        lfs_io
        lfs_rendering
        lfs_training
        lfs_python_runtime
        lfs_python_utils
        lfs_sequencer
        glad::glad
        glfw
        imgui::imgui
        implot::implot
        imguizmo::imguizmo
        imgui_textedit
        glm::glm
        TBB::tbb
        spdlog::spdlog
        nlohmann_json::nlohmann_json
        OpenSSL::SSL
        OpenSSL::Crypto
        CUDA::cudart
        ${OPENGL_LIBRARIES}
        vterm
)

# Set C++ standard
target_compile_features(lfs_visualizer PUBLIC cxx_std_23)

if(WIN32)
    set_target_properties(lfs_visualizer PROPERTIES
        POSITION_INDEPENDENT_CODE ON
        CXX_VISIBILITY_PRESET hidden
        VISIBILITY_INLINES_HIDDEN ON
        LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}
        RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}
        RUNTIME_OUTPUT_DIRECTORY_DEBUG ${CMAKE_BINARY_DIR}
        RUNTIME_OUTPUT_DIRECTORY_RELEASE ${CMAKE_BINARY_DIR}
        RUNTIME_OUTPUT_DIRECTORY_RELWITHDEBINFO ${CMAKE_BINARY_DIR}
        RUNTIME_OUTPUT_DIRECTORY_MINSIZEREL ${CMAKE_BINARY_DIR}
    )
endif()

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

# Copy locale files
file(MAKE_DIRECTORY "${VISUALIZER_BUILD_RESOURCE_DIR}/locales")
file(GLOB LOCALE_FILES "${CMAKE_CURRENT_SOURCE_DIR}/gui/resources/locales/*.json")
foreach (LOCALE_FILE ${LOCALE_FILES})
    get_filename_component(LOCALE_NAME ${LOCALE_FILE} NAME)
    configure_file(${LOCALE_FILE} "${VISUALIZER_BUILD_RESOURCE_DIR}/locales/${LOCALE_NAME}" COPYONLY)
endforeach ()

# Compile definitions
target_compile_definitions(lfs_visualizer PRIVATE
        LFS_VIS_EXPORTS
        $<$<BOOL:${CUDA_GL_INTEROP_ENABLED}>:CUDA_GL_INTEROP_ENABLED>
        PROJECT_ROOT_PATH="${PROJECT_SOURCE_DIR}"
        VISUALIZER_SHADER_PATH="${VISUALIZER_BUILD_RESOURCE_DIR}/shaders"
        VISUALIZER_ASSET_PATH="${VISUALIZER_BUILD_RESOURCE_DIR}/assets"
        VISUALIZER_LOCALE_PATH="${VISUALIZER_BUILD_RESOURCE_DIR}/locales"
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
            util  # PTY support (forkpty)
    )
endif ()

if(NOT BUILD_PORTABLE)
    install(FILES ${CMAKE_CURRENT_SOURCE_DIR}/include/visualizer/visualizer.hpp
            DESTINATION include/visualizer)
endif()

# Optionally install resources
install(DIRECTORY ${VISUALIZER_SOURCE_RESOURCE_DIR}/
        DESTINATION share/LichtFeld-Studio/visualizer/resources
        FILES_MATCHING
        PATTERN "*.vert"
        PATTERN "*.frag"
        PATTERN "*.ttf"
)

# Install locale files
install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/gui/resources/locales/
        DESTINATION share/LichtFeld-Studio/locales
        FILES_MATCHING
        PATTERN "*.json"
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

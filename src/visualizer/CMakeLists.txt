# SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
#
# SPDX-License-Identifier: GPL-3.0-or-later

# Fetch libvterm for terminal emulation
include(${CMAKE_SOURCE_DIR}/cmake/FetchLibvterm.cmake)
include(${CMAKE_SOURCE_DIR}/cmake/CompileShaders.cmake)

# SHARED on Windows to avoid singleton duplication across .exe/.pyd boundary
if(WIN32)
    set(LFS_VIS_LIB_TYPE SHARED)
else()
    set(LFS_VIS_LIB_TYPE STATIC)
endif()

option(LFS_DEV_IMPORT_SOURCE_RESOURCES
        "Load RmlUI resources and locales from source in non-portable builds"
        ON)

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
        window/vulkan_context.cpp
        window/vulkan_image_barrier_tracker.cpp
        window/vulkan_loader_probe.cpp

        # Rendering
        rendering/camera_interaction_service.cpp
        rendering/rendering_manager.cpp
        rendering/rendering_manager_events.cpp
        rendering/rendering_manager_frame.cpp
        rendering/rendering_manager_vulkan.cpp
        rendering/rendering_manager_viewport.cpp
        rendering/split_view_composition.cpp
        rendering/split_view_service.cpp
        rendering/viewport_artifact_service.cpp
        rendering/viewport_frame_lifecycle_service.cpp
        rendering/viewport_appearance_correction.cpp
        rendering/viewport_request_builder.cpp
        rendering/passes/vulkan_scene_image_uploader.cpp
        rendering/passes/vulkan_viewport_pass.cpp
        rendering/passes/vulkan_mesh_pass.cpp
        rendering/passes/vulkan_environment_pass.cpp
        rendering/passes/vulkan_depth_blit_pass.cpp
        rendering/passes/vulkan_split_view_pass.cpp
        rendering/framerate_controller.cpp

        # Theme system
        theme/theme.cpp

        # GUI system
        gui/gui_manager.cpp
        gui/vulkan_ui_texture.cpp
        gui/global_context_menu.cpp
        gui/gizmo_manager.cpp
        gui/pie_menu.cpp
        gui/panel_layout.cpp
        gui/panel_registry.cpp
        gui/native_panels.cpp
        gui/layout_state.cpp
        gui/async_task_manager.cpp
        gui/bounds_gizmo.cpp
        gui/rotation_gizmo.cpp
        gui/scale_gizmo.cpp
        gui/translation_gizmo.cpp
        gui/video_export_utils.cpp
        gui/startup_overlay.cpp
        gui/rml_menu_bar.cpp
        gui/rml_right_panel.cpp
        gui/rml_shell_frame.cpp
        gui/rml_viewport_overlay.cpp
        gui/rml_status_bar.cpp
        gui/gpu_memory_query.cpp
        gui/line_renderer.cpp
        gui/rml_overlay_context.cpp
        gui/film_strip_renderer.cpp
        gui/sequencer_ui_manager.cpp
        gui/rml_modal_overlay.cpp
        gui/rml_sequencer_overlay.cpp
        gui/keyframe_scene_sync.cpp
        gui/gizmo_transform.cpp
        gui/html_viewer_export.cpp
        gui/scene_panel_native.cpp
        gui/ui_widgets.cpp
        gui_capabilities.cpp
        gui/panels/windows_console_utils.cpp
        gui/panels/tools_panel.cpp
        gui/panels/python_console_panel.cpp
        gui/panels/python_scripts_panel.cpp
        gui/editor/python_lsp_client.cpp
        gui/editor/python_editor.cpp
        gui/editor/zep_rml_display.cpp
        gui/editor/stdio_process.cpp

        # Terminal emulator
        gui/terminal/pty_process.cpp
        gui/terminal/terminal_widget.cpp
        gui/windows/video_extractor_dialog.cpp
        gui/video_widget_factory.cpp

        # Scene management
        scene/scene_manager.cpp
        scene/selection_state.cpp

        # Input handling
        input/input_controller.cpp
        input/input_router.cpp
        input/input_bindings.cpp
        input/sdl_key_mapping.cpp

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
        sequencer/rml_sequencer_panel.cpp
        sequencer/rml_sequencer_panel_timeline.cpp
        sequencer/rml_sequencer_panel_timeline_input.cpp

        # Operator system (Blender-style)
        operator/operator_context.cpp
        operator/operator_id.cpp
        operator/operator_registry.cpp
        operator/property_schema.cpp
        operator/ops/align_ops.cpp
        operator/ops/brush_ops.cpp
        operator/ops/edit_ops.cpp
        operator/ops/scene_ops.cpp
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

        # View/render bridge for plugin capabilities
        ipc/view_context.cpp

        gui/panels/menu_bar.cpp
        gui/panels/menu_bar.hpp
        gui/utils/native_file_dialog.cpp
        gui/utils/drag_drop_native.cpp
        gui/utils/file_association.cpp

        # RmlUI integration
        gui/rmlui/rmlui_system_interface.cpp
        gui/rmlui/rml_text_input_handler.cpp
        gui/rmlui/rmlui_manager.cpp
        gui/rmlui/rml_document_utils.cpp
        gui/rmlui/rml_panel_host.cpp
        gui/rmlui/rml_theme.cpp
        gui/rmlui/elements/chromaticity_element.cpp
        gui/rmlui/elements/color_picker_element.cpp
        gui/rmlui/elements/crf_curve_element.cpp
        gui/rmlui/elements/loss_graph_element.cpp
        gui/rmlui/elements/python_editor_element.cpp
        gui/rmlui/elements/scene_graph_element.cpp
        gui/rmlui/elements/terminal_element.cpp
)

# MSVC 14.44 ICEs (C1001 in msc1.cpp:1589) on the sequencer panel TUs under
# /std:c++latest in both Debug (/Od /RTC1) and Release (/O2 /Ob2). Pin them to
# /std:c++23 (not "latest") and add /bigobj for the heavy tensor + RmlUI + fmt instantiations.
if(MSVC)
    set_source_files_properties(
            ${CMAKE_CURRENT_SOURCE_DIR}/sequencer/rml_sequencer_panel.cpp
            ${CMAKE_CURRENT_SOURCE_DIR}/sequencer/rml_sequencer_panel_timeline.cpp
            PROPERTIES COMPILE_OPTIONS "/std:c++23;/bigobj"
    )
endif()

# Set include directories
target_include_directories(lfs_visualizer
        PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>
        PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${CMAKE_SOURCE_DIR}/src
        ${CMAKE_SOURCE_DIR}/external/httplib
)

target_link_libraries(lfs_visualizer
        PRIVATE
        lfs_core
        lfs_video
        lfs_geometry
        lfs_io
        lfs_rendering_tensor
        lfs_training
        lfs_python_runtime
        lfs_python_utils
        lfs_sequencer
        SDL3::SDL3
        imgui::imgui
        implot::implot
        Zep::Zep
        glm::glm
        TBB::tbb
        spdlog::spdlog
        nlohmann_json::nlohmann_json
        OpenSSL::SSL
        OpenSSL::Crypto
        CUDA::cudart
        nfd::nfd
        vterm
)

if(ENABLE_VULKAN_VIEWER)
    target_sources(lfs_visualizer
            PRIVATE
            gui/rmlui/rmlui_vk_backend.cpp
            rendering/vksplat_input_packer.cpp
            rendering/vksplat_viewport_renderer.cpp)
    set(glslang_DIR "${CMAKE_BINARY_DIR}/vcpkg_installed/${VCPKG_TARGET_TRIPLET}/share/glslang"
        CACHE PATH "Path to vcpkg-installed glslang config" FORCE)
    find_package(glslang CONFIG REQUIRED)

    add_executable(lfs_shader_compiler EXCLUDE_FROM_ALL
            tools/shader_compiler.cpp)
    target_compile_features(lfs_shader_compiler PRIVATE cxx_std_23)
    target_link_libraries(lfs_shader_compiler
            PRIVATE
            glslang::glslang
            glslang::SPIRV
            glslang::glslang-default-resource-limits)

    set(LFS_VIEWPORT_SHADER_SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/rendering/resources/viewport")
    set(LFS_VIEWPORT_SHADER_BINARY_DIR "${CMAKE_CURRENT_BINARY_DIR}/generated/shaders/viewport")
    target_include_directories(lfs_visualizer PRIVATE "${CMAKE_CURRENT_BINARY_DIR}/generated/shaders")
    compile_shader(lfs_visualizer "${LFS_VIEWPORT_SHADER_SOURCE_DIR}/screen_quad.vert" "${LFS_VIEWPORT_SHADER_BINARY_DIR}/screen_quad.vert.spv.h" kScreenQuadVertSpv)
    compile_shader(lfs_visualizer "${LFS_VIEWPORT_SHADER_SOURCE_DIR}/scene.frag" "${LFS_VIEWPORT_SHADER_BINARY_DIR}/scene.frag.spv.h" kSceneFragSpv)
    compile_shader(lfs_visualizer "${LFS_VIEWPORT_SHADER_SOURCE_DIR}/vignette.frag" "${LFS_VIEWPORT_SHADER_BINARY_DIR}/vignette.frag.spv.h" kVignetteFragSpv)
    compile_shader(lfs_visualizer "${LFS_VIEWPORT_SHADER_SOURCE_DIR}/grid.vert" "${LFS_VIEWPORT_SHADER_BINARY_DIR}/grid.vert.spv.h" kGridVertSpv)
    compile_shader(lfs_visualizer "${LFS_VIEWPORT_SHADER_SOURCE_DIR}/grid.frag" "${LFS_VIEWPORT_SHADER_BINARY_DIR}/grid.frag.spv.h" kGridFragSpv)
    compile_shader(lfs_visualizer "${LFS_VIEWPORT_SHADER_SOURCE_DIR}/overlay.vert" "${LFS_VIEWPORT_SHADER_BINARY_DIR}/overlay.vert.spv.h" kOverlayVertSpv)
    compile_shader(lfs_visualizer "${LFS_VIEWPORT_SHADER_SOURCE_DIR}/overlay.frag" "${LFS_VIEWPORT_SHADER_BINARY_DIR}/overlay.frag.spv.h" kOverlayFragSpv)
    compile_shader(lfs_visualizer "${LFS_VIEWPORT_SHADER_SOURCE_DIR}/shape_overlay.vert" "${LFS_VIEWPORT_SHADER_BINARY_DIR}/shape_overlay.vert.spv.h" kShapeOverlayVertSpv)
    compile_shader(lfs_visualizer "${LFS_VIEWPORT_SHADER_SOURCE_DIR}/shape_overlay.frag" "${LFS_VIEWPORT_SHADER_BINARY_DIR}/shape_overlay.frag.spv.h" kShapeOverlayFragSpv)
    compile_shader(lfs_visualizer "${LFS_VIEWPORT_SHADER_SOURCE_DIR}/textured_overlay.vert" "${LFS_VIEWPORT_SHADER_BINARY_DIR}/textured_overlay.vert.spv.h" kTexturedOverlayVertSpv)
    compile_shader(lfs_visualizer "${LFS_VIEWPORT_SHADER_SOURCE_DIR}/textured_overlay.frag" "${LFS_VIEWPORT_SHADER_BINARY_DIR}/textured_overlay.frag.spv.h" kTexturedOverlayFragSpv)
    compile_shader(lfs_visualizer "${LFS_VIEWPORT_SHADER_SOURCE_DIR}/pivot.vert" "${LFS_VIEWPORT_SHADER_BINARY_DIR}/pivot.vert.spv.h" kPivotVertSpv)
    compile_shader(lfs_visualizer "${LFS_VIEWPORT_SHADER_SOURCE_DIR}/pivot.frag" "${LFS_VIEWPORT_SHADER_BINARY_DIR}/pivot.frag.spv.h" kPivotFragSpv)
    compile_shader(lfs_visualizer "${LFS_VIEWPORT_SHADER_SOURCE_DIR}/mesh.vert" "${LFS_VIEWPORT_SHADER_BINARY_DIR}/mesh.vert.spv.h" kMeshVertSpv)
    compile_shader(lfs_visualizer "${LFS_VIEWPORT_SHADER_SOURCE_DIR}/mesh.frag" "${LFS_VIEWPORT_SHADER_BINARY_DIR}/mesh.frag.spv.h" kMeshFragSpv)
    compile_shader(lfs_visualizer "${LFS_VIEWPORT_SHADER_SOURCE_DIR}/mesh_shadow.vert" "${LFS_VIEWPORT_SHADER_BINARY_DIR}/mesh_shadow.vert.spv.h" kMeshShadowVertSpv)
    compile_shader(lfs_visualizer "${LFS_VIEWPORT_SHADER_SOURCE_DIR}/mesh_shadow.frag" "${LFS_VIEWPORT_SHADER_BINARY_DIR}/mesh_shadow.frag.spv.h" kMeshShadowFragSpv)
    compile_shader(lfs_visualizer "${LFS_VIEWPORT_SHADER_SOURCE_DIR}/mesh_wireframe.vert" "${LFS_VIEWPORT_SHADER_BINARY_DIR}/mesh_wireframe.vert.spv.h" kMeshWireframeVertSpv)
    compile_shader(lfs_visualizer "${LFS_VIEWPORT_SHADER_SOURCE_DIR}/mesh_wireframe.frag" "${LFS_VIEWPORT_SHADER_BINARY_DIR}/mesh_wireframe.frag.spv.h" kMeshWireframeFragSpv)
    compile_shader(lfs_visualizer "${LFS_VIEWPORT_SHADER_SOURCE_DIR}/environment.frag" "${LFS_VIEWPORT_SHADER_BINARY_DIR}/environment.frag.spv.h" kEnvironmentFragSpv)
    compile_shader(lfs_visualizer "${LFS_VIEWPORT_SHADER_SOURCE_DIR}/depth_blit.frag" "${LFS_VIEWPORT_SHADER_BINARY_DIR}/depth_blit.frag.spv.h" kDepthBlitFragSpv)
    compile_shader(lfs_visualizer "${LFS_VIEWPORT_SHADER_SOURCE_DIR}/split_view.frag" "${LFS_VIEWPORT_SHADER_BINARY_DIR}/split_view.frag.spv.h" kSplitViewFragSpv)
    compile_shader(lfs_visualizer "${LFS_VIEWPORT_SHADER_SOURCE_DIR}/vksplat_compose.comp" "${LFS_VIEWPORT_SHADER_BINARY_DIR}/vksplat_compose.comp.spv.h" kVkSplatComposeCompSpv)

    target_link_libraries(lfs_visualizer
            PRIVATE
            Vulkan::Vulkan
            GPUOpen::VulkanMemoryAllocator
            volk::volk_headers)
    target_link_libraries(lfs_visualizer PRIVATE lfs_vksplat_fwd)
    if(WIN32)
        target_compile_definitions(lfs_visualizer PRIVATE VK_USE_PLATFORM_WIN32_KHR)
    endif()
endif()

# RmlUI: on Linux, lfs_visualizer is STATIC and gets linked into both the
# exe and the Python .so, which would duplicate librmlui.a globals and cause
# crashes.  Wrap the static libraries in a small shared library so there is
# exactly one copy of RmlUI in the process.
find_package(RmlUi CONFIG REQUIRED)

if(NOT WIN32)
    file(WRITE "${CMAKE_CURRENT_BINARY_DIR}/lfs_rmlui_stub.cpp" "")
    add_library(lfs_rmlui SHARED "${CMAKE_CURRENT_BINARY_DIR}/lfs_rmlui_stub.cpp")

    find_package(lunasvg CONFIG REQUIRED)
    target_link_libraries(lfs_rmlui PRIVATE
        -Wl,--whole-archive
        $<TARGET_FILE:RmlUi::Core>
        $<TARGET_FILE:RmlUi::Debugger>
        -Wl,--no-whole-archive
        Freetype::Freetype
        lunasvg::lunasvg
    )

    get_target_property(_rmlui_includes RmlUi::Core INTERFACE_INCLUDE_DIRECTORIES)
    if(_rmlui_includes)
        target_include_directories(lfs_rmlui PUBLIC ${_rmlui_includes})
    endif()
    target_compile_definitions(lfs_rmlui PUBLIC RMLUI_STATIC_LIB)

    if(TARGET Freetype::Freetype)
        get_target_property(_ft_includes Freetype::Freetype INTERFACE_INCLUDE_DIRECTORIES)
        if(_ft_includes)
            target_include_directories(lfs_rmlui PUBLIC ${_ft_includes})
        endif()
    endif()

    set_target_properties(lfs_rmlui PROPERTIES
        POSITION_INDEPENDENT_CODE ON
        LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}
    )

    target_link_libraries(lfs_visualizer PUBLIC lfs_rmlui)
else()
    target_link_libraries(lfs_visualizer PUBLIC RmlUi::RmlUi)
endif()

# Set C++ standard
target_compile_features(lfs_visualizer PUBLIC cxx_std_23)

set_target_properties(lfs_visualizer PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}
)
if(WIN32)
    set_target_properties(lfs_visualizer PROPERTIES
        POSITION_INDEPENDENT_CODE ON
        CXX_VISIBILITY_PRESET hidden
        VISIBILITY_INLINES_HIDDEN ON
    )
endif()
get_property(_IS_MULTI_CONFIG GLOBAL PROPERTY GENERATOR_IS_MULTI_CONFIG)
if(WIN32 AND _IS_MULTI_CONFIG)
    set_target_properties(lfs_visualizer PROPERTIES
        LIBRARY_OUTPUT_DIRECTORY              ${CMAKE_BINARY_DIR}/$<CONFIG>
        RUNTIME_OUTPUT_DIRECTORY              ${CMAKE_BINARY_DIR}/$<CONFIG>
        RUNTIME_OUTPUT_DIRECTORY_DEBUG        ${CMAKE_BINARY_DIR}/Debug
        RUNTIME_OUTPUT_DIRECTORY_RELEASE      ${CMAKE_BINARY_DIR}/Release
        RUNTIME_OUTPUT_DIRECTORY_RELWITHDEBINFO ${CMAKE_BINARY_DIR}/RelWithDebInfo
        RUNTIME_OUTPUT_DIRECTORY_MINSIZEREL   ${CMAKE_BINARY_DIR}/MinSizeRel
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
set(RENDERING_SOURCE_RESOURCE_DIR "${PROJECT_SOURCE_DIR}/src/rendering/resources")

# Create resource directories in build folder
file(MAKE_DIRECTORY "${VISUALIZER_BUILD_RESOURCE_DIR}/assets")
file(MAKE_DIRECTORY "${VISUALIZER_BUILD_RESOURCE_DIR}/assets/fonts")

# Copy assets from resources/assets if they exist
file(GLOB ASSET_FILES CONFIGURE_DEPENDS "${VISUALIZER_SOURCE_RESOURCE_DIR}/assets/*")
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

set(RENDERING_FONT_FILES
        "${RENDERING_SOURCE_RESOURCE_DIR}/assets/JetBrainsMono-Regular.ttf"
)

foreach (RENDERING_FONT_FILE ${RENDERING_FONT_FILES})
    if (EXISTS ${RENDERING_FONT_FILE} AND NOT IS_DIRECTORY ${RENDERING_FONT_FILE})
        get_filename_component(FONT_NAME ${RENDERING_FONT_FILE} NAME)
        configure_file(${RENDERING_FONT_FILE} "${VISUALIZER_BUILD_RESOURCE_DIR}/assets/fonts/${FONT_NAME}" COPYONLY)
    endif ()
endforeach ()

# Copy icon files
file(MAKE_DIRECTORY "${VISUALIZER_BUILD_RESOURCE_DIR}/assets/icon")
file(GLOB ICON_FILES CONFIGURE_DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/gui/assets/icon/*.png")
foreach (ICON_FILE ${ICON_FILES})
    get_filename_component(ICON_NAME ${ICON_FILE} NAME)
    configure_file(${ICON_FILE} "${VISUALIZER_BUILD_RESOURCE_DIR}/assets/icon/${ICON_NAME}" COPYONLY)
endforeach ()

# Copy scene icon files (used by RmlUI scene panel)
file(MAKE_DIRECTORY "${VISUALIZER_BUILD_RESOURCE_DIR}/assets/icon/scene")
file(GLOB SCENE_ICON_FILES CONFIGURE_DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/gui/assets/icon/scene/*.png")
foreach (ICON_FILE ${SCENE_ICON_FILES})
    get_filename_component(ICON_NAME ${ICON_FILE} NAME)
    configure_file(${ICON_FILE} "${VISUALIZER_BUILD_RESOURCE_DIR}/assets/icon/scene/${ICON_NAME}" COPYONLY)
endforeach ()

# Copy sequencer icon files (used by RmlUI sequencer panel)
file(MAKE_DIRECTORY "${VISUALIZER_BUILD_RESOURCE_DIR}/assets/icon/sequencer")
file(GLOB SEQUENCER_ICON_FILES CONFIGURE_DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/gui/assets/icon/sequencer/*.png")
foreach (ICON_FILE ${SEQUENCER_ICON_FILES})
    get_filename_component(ICON_NAME ${ICON_FILE} NAME)
    configure_file(${ICON_FILE} "${VISUALIZER_BUILD_RESOURCE_DIR}/assets/icon/sequencer/${ICON_NAME}" COPYONLY)
endforeach ()

# Copy bundled environment maps
file(MAKE_DIRECTORY "${VISUALIZER_BUILD_RESOURCE_DIR}/assets/environments")
file(GLOB ENVIRONMENT_ASSET_FILES CONFIGURE_DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/gui/assets/environments/*")
foreach (ENVIRONMENT_ASSET_FILE ${ENVIRONMENT_ASSET_FILES})
    if (EXISTS ${ENVIRONMENT_ASSET_FILE} AND NOT IS_DIRECTORY ${ENVIRONMENT_ASSET_FILE})
        get_filename_component(ENVIRONMENT_ASSET_NAME ${ENVIRONMENT_ASSET_FILE} NAME)
        configure_file(${ENVIRONMENT_ASSET_FILE} "${VISUALIZER_BUILD_RESOURCE_DIR}/assets/environments/${ENVIRONMENT_ASSET_NAME}" COPYONLY)
    endif ()
endforeach ()

# Copy RmlUI resources (re-copy on every build so edits are picked up)
file(MAKE_DIRECTORY "${VISUALIZER_BUILD_RESOURCE_DIR}/assets/rmlui")
file(GLOB RMLUI_RESOURCE_FILES CONFIGURE_DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/gui/rmlui/resources/*")
foreach (RMLUI_FILE ${RMLUI_RESOURCE_FILES})
    get_filename_component(RMLUI_NAME ${RMLUI_FILE} NAME)
    configure_file(${RMLUI_FILE} "${VISUALIZER_BUILD_RESOURCE_DIR}/assets/rmlui/${RMLUI_NAME}" COPYONLY)
endforeach ()
add_custom_target(copy_rmlui_resources ALL
    COMMAND ${CMAKE_COMMAND} -E rm -rf "${VISUALIZER_BUILD_RESOURCE_DIR}/assets/rmlui"
    COMMAND ${CMAKE_COMMAND} -E make_directory "${VISUALIZER_BUILD_RESOURCE_DIR}/assets/rmlui"
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
        ${RMLUI_RESOURCE_FILES}
        "${VISUALIZER_BUILD_RESOURCE_DIR}/assets/rmlui/"
    COMMENT "Syncing RmlUI resources"
)
add_dependencies(lfs_visualizer copy_rmlui_resources)

# Copy locale files
file(MAKE_DIRECTORY "${VISUALIZER_BUILD_RESOURCE_DIR}/locales")
file(GLOB LOCALE_FILES CONFIGURE_DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/gui/resources/locales/*.json")
foreach (LOCALE_FILE ${LOCALE_FILES})
    get_filename_component(LOCALE_NAME ${LOCALE_FILE} NAME)
    configure_file(${LOCALE_FILE} "${VISUALIZER_BUILD_RESOURCE_DIR}/locales/${LOCALE_NAME}" COPYONLY)
endforeach ()

set(LFS_VISUALIZER_DEV_RESOURCE_DEFINITIONS)
if(LFS_DEV_IMPORT_SOURCE_RESOURCES AND NOT BUILD_PORTABLE)
    list(APPEND LFS_VISUALIZER_DEV_RESOURCE_DEFINITIONS
            LFS_DEV_RMLUI_SOURCE_DIR="${CMAKE_CURRENT_SOURCE_DIR}/gui/rmlui/resources"
            LFS_DEV_LOCALE_SOURCE_DIR="${CMAKE_CURRENT_SOURCE_DIR}/gui/resources/locales"
    )
    message(STATUS "Resource dev mode: RmlUI imports from ${CMAKE_CURRENT_SOURCE_DIR}/gui/rmlui/resources")
    message(STATUS "Resource dev mode: locales import from ${CMAKE_CURRENT_SOURCE_DIR}/gui/resources/locales")
endif()

# Compile definitions
target_compile_definitions(lfs_visualizer PRIVATE
        LFS_VIS_EXPORTS
        PROJECT_ROOT_PATH="${PROJECT_SOURCE_DIR}"
        VISUALIZER_ASSET_PATH="${VISUALIZER_BUILD_RESOURCE_DIR}/assets"
        VISUALIZER_LOCALE_PATH="${VISUALIZER_BUILD_RESOURCE_DIR}/locales"
        VISUALIZER_SOURCE_ASSET_PATH="${VISUALIZER_SOURCE_RESOURCE_DIR}/assets"
        ${LFS_VISUALIZER_DEV_RESOURCE_DEFINITIONS}
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
            Shell32     # SHChangeNotify (file association)
            dxgi        # DXGI per-process VRAM query
    )
elseif (CMAKE_SYSTEM_NAME STREQUAL "Linux")
    target_link_libraries(lfs_visualizer PRIVATE
            X11
            util  # PTY support (forkpty)
    )
endif ()

if(NOT BUILD_PORTABLE)
    install(FILES ${CMAKE_CURRENT_SOURCE_DIR}/include/visualizer/visualizer.hpp
            DESTINATION include/visualizer)
endif()

# Optionally install resources
install(FILES ${RENDERING_SOURCE_RESOURCE_DIR}/assets/JetBrainsMono-Regular.ttf
        DESTINATION share/LichtFeld-Studio/visualizer/resources/assets/fonts)

install(DIRECTORY ${VISUALIZER_SOURCE_RESOURCE_DIR}/
        DESTINATION share/LichtFeld-Studio/visualizer/resources
        FILES_MATCHING
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
        "${VISUALIZER_SOURCE_RESOURCE_DIR}/assets/*.ttf"
)
if (VISUALIZER_RESOURCE_FILES)
    add_custom_target(visualizer_resources SOURCES ${VISUALIZER_RESOURCE_FILES})
endif ()

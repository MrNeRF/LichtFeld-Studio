# ImGui Inventory

This inventory tracks remaining ImGui usage while the application moves new UI work to RmlUi.
Do not add new ImGui surfaces or backend dependencies; new user-facing panels should be RmlUi.

## Dependency State

- `vcpkg.json` still includes `imgui` with `docking-experimental`, `freetype`, and `sdl3-binding`.
- No `imgui[opengl3-binding]` dependency remains.
- No owned `imgui_impl_opengl*` include remains.
- `implot` remains only because legacy ImGui surfaces still link against it.

## Remaining ImGui Surfaces

| Area | Files | Current role | Migration hint |
| --- | --- | --- | --- |
| Frame/input bootstrap | `src/visualizer/gui/gui_manager.*`, `src/visualizer/window/window_manager.cpp`, `src/visualizer/input/input_controller.cpp` | Owns ImGui context, SDL3 platform backend, frame begin/end, focus capture, cursor mapping, and `.ini` persistence. | Move input capture/cursor/text-input ownership to RmlUi and keep a small compatibility bridge only while legacy panels exist. |
| Panel shell compatibility | `src/visualizer/gui/panel_registry.cpp`, `src/visualizer/gui/panel_layout.cpp`, `src/visualizer/gui/rmlui/rml_panel_host.cpp` | Uses ImGui docking/layout/draw-list coordinates to host native and Rml panels. | Keep `PanelRegistry` as data, replace the layout host with the Rml shell/frame and remove ImGui dummy/clip/dock dependencies. |
| Legacy native widgets | `src/visualizer/gui/ui_widgets.*`, `src/visualizer/gui/panels/windows_console_utils.cpp`, `src/visualizer/gui/windows/video_extractor_dialog.cpp` | Shared immediate widgets and one remaining dialog implementation. | Replace with Rml modal/dialog components and shared RCSS controls. |
| Viewport overlays and tools | `src/visualizer/tools/{align_tool,brush_tool,selection_tool}.cpp`, `src/visualizer/gui/{gizmo_transform.hpp,pie_menu.hpp,startup_overlay.cpp,gui_manager.cpp}` | Draw-list overlays, tool hints, selection rectangles, cursor labels, pie menu, and transitional viewport chrome. | World/viewport geometry belongs in `VulkanViewportPass`; command UI and tool affordances belong in Rml overlay documents. |
| Theme bridge | `src/visualizer/theme/theme.*` | Theme model still uses `ImVec2`, `ImVec4`, and `ImU32` and applies an ImGui style. | Split theme tokens into renderer-neutral types, then provide separate ImGui compatibility and Rml RCSS adapters. |
| Sequencer bridge | `src/visualizer/sequencer/rml_sequencer_panel.cpp` | Sequencer UI is RmlUi, but the panel still touches ImGui for host integration details. | Remove once panel hosting no longer depends on ImGui geometry/frame state. |
| Python plugin UI compatibility | `src/python/lfs/py_ui*.cpp`, `src/python/lfs/py_uilist.cpp`, `src/python/lfs/*panel_adapter*` | Maintains the existing immediate-mode Python UI API and adapters, including Rml-backed adapters that still sample ImGui input state. | Freeze the ImGui-shaped API, route new plugin UI through Rml documents/data binding, and avoid exposing new ImGui primitives. |

## Already RmlUi-First

- Main menu/status/right shell: `src/visualizer/gui/rml_menu_bar.cpp`, `rml_status_bar.cpp`, `rml_right_panel.cpp`, `rml_shell_frame.cpp`.
- Python console/editor panel: `src/visualizer/gui/panels/python_console_panel.cpp`, `src/visualizer/gui/rmlui/elements/python_editor_element.cpp`.
- Sequencer panel and timeline: `src/visualizer/sequencer/rml_sequencer_panel*.cpp`.
- Modal overlay and global context menu: `src/visualizer/gui/rml_modal_overlay.cpp`, `global_context_menu.cpp`.
- Vulkan UI textures and viewport textured overlays: `src/visualizer/gui/vulkan_ui_texture.cpp`, `src/visualizer/rendering/passes/vulkan_viewport_pass.cpp`.

## Guardrails

- Do not add `imgui_impl_opengl*`, `glad`, or OpenGL-backed texture paths.
- Do not add new ImGui panels for profiler or Vulkan tooling; use RmlUi.
- Keep compatibility code scoped and removable: new state should live outside ImGui-specific types unless it is an adapter boundary.

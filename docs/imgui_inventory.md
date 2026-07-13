# ImGui Inventory

This inventory tracks production ImGui usage that still blocks removing Dear ImGui
from LichtFeld Studio. New user-facing UI should be RmlUi-first, and new code
should not add ImGui, ImPlot, or backend dependencies.

## Dependency State

- `vcpkg.json` still includes `imgui` with `docking-experimental`, `freetype`, and
  `sdl3-binding`; it also still includes `implot`.
- The root `CMakeLists.txt` still calls `find_package(imgui REQUIRED)` and
  `find_package(implot REQUIRED)`.
- `src/visualizer/CMakeLists.txt`, `src/python/CMakeLists.txt`, and
  `tests/CMakeLists.txt` still link `imgui::imgui` and `implot::implot`.
- No `imgui[opengl3-binding]` dependency remains.
- No owned `imgui_impl_opengl*` include remains.
- `implot` remains only because legacy ImGui/Python surfaces still create and share
  an ImPlot context.

## Current Hit Count Snapshot

Approximate production source hit counts from an audit of `src/`:

| Area | Approx. hits | Primary files | Current role |
| --- | ---: | --- | --- |
| Python immediate-mode UI API | ~430 | `src/python/lfs/py_ui.cpp`, `py_ui.hpp`, `py_ui_menus.cpp`, `py_uilist.cpp`, `*panel_adapter*.cpp`, `python_runtime.*`, `module.cpp` | Exposes `PyUILayout`, plugin panels, menus, popups, modals, key enums, image widgets, drag/drop, and ImGui/ImPlot context sharing. |
| Shared native widget helpers | ~321 | `src/visualizer/gui/ui_widgets.*` | Shared immediate widgets used by remaining native/legacy UI code. |
| Theme bridge | ~230 | `src/visualizer/theme/theme.*`, `src/visualizer/gui/rmlui/rml_theme.cpp` | Theme data still uses `ImVec2`, `ImVec4`, `ImU32`, and applies ImGui style state. |
| GUI frame/input/bootstrap | ~160 | `src/visualizer/gui/gui_manager.*`, `src/visualizer/window/window_manager.cpp`, `src/visualizer/input/input_controller.cpp`, `src/visualizer/visualizer_impl.cpp` | Creates ImGui/ImPlot contexts, initializes the SDL3 ImGui backend, owns frame begin/end, focus capture, cursor mapping, text input, and `.ini` persistence. |
| Panel host/layout compatibility | ~37 | `src/visualizer/gui/panel_registry.cpp`, `panel_layout.cpp`, `rmlui/rml_panel_host.cpp` | Uses ImGui viewport/draw-list geometry for floating panels, panel caching, and Rml/native panel hosting. |
| Viewport overlays and tools | ~55 | `src/visualizer/gui/gizmo_manager.*`, `gizmo_transform.hpp`, `pie_menu.*`, `startup_overlay.cpp`, `sequencer_ui_manager.cpp`, `src/visualizer/tools/selection_tool.cpp`, `rml_status_bar.cpp`, `rmlui/rmlui_manager.hpp` | Uses ImGui cursor/draw-list/input state for overlay affordances, pie menu rendering, selection cursor reads, and transitional chrome. |

Re-run the snapshot with:

```sh
for f in $(rg -l "#include <imgui|#include \"imgui|ImGui::|ImPlot|ImVec|ImU32|ImTextureID|ImDraw|ImGui[A-Za-z_]*|imgui_impl" src | sort); do
  c=$(rg -n "#include <imgui|#include \"imgui|ImGui::|ImPlot|ImVec|ImU32|ImTextureID|ImDraw|ImGui[A-Za-z_]*|imgui_impl" "$f" | wc -l)
  printf "%4d %s\n" "$c" "$f"
done
```

## Remaining ImGui-Enabled Surfaces

| Surface | Files | Current role | Migration hint |
| --- | --- | --- | --- |
| Python plugin UI compatibility | `src/python/lfs/py_ui*.cpp`, `src/python/lfs/py_uilist.cpp`, `src/python/lfs/python_panel_adapter.hpp`, `src/python/lfs/*panel_adapter*` | Maintains the existing immediate-mode Python UI API. Rml-backed adapters exist, but the raw API and fallback panel adapter still expose ImGui-shaped behavior. | Freeze the legacy API, route plugin panels through retained Rml immediate-mode or Rml documents/data binding, then remove raw ImGui/ImPlot bindings and context sharing. |
| Frame/input bootstrap | `src/visualizer/gui/gui_manager.*`, `src/visualizer/window/window_manager.cpp`, `src/visualizer/input/input_controller.cpp` | Owns ImGui context, SDL3 platform backend, frame begin/end, focus capture, cursor mapping, text input, and `.ini` persistence. | Move capture, text-input, cursor, timing, and settings ownership to RmlUi/SDL-native state. |
| Panel shell compatibility | `src/visualizer/gui/panel_registry.cpp`, `src/visualizer/gui/panel_layout.cpp`, `src/visualizer/gui/rmlui/rml_panel_host.cpp` | Uses ImGui viewport/draw-list coordinates to host native and Rml panels. | Keep `PanelRegistry` as data; replace ImGui viewport/draw-list dependencies with the Rml shell/frame and renderer-native overlay coordinates. |
| Legacy native widgets | `src/visualizer/gui/ui_widgets.*`, `src/visualizer/gui/panels/windows_console_utils.cpp` | Shared immediate widgets and console helpers. | Replace remaining users with Rml components or renderer-native controls, then delete the helper layer. |
| Viewport overlays and tools | `src/visualizer/gui/gizmo_manager.*`, `src/visualizer/gui/gizmo_transform.hpp`, `src/visualizer/gui/pie_menu.*`, `src/visualizer/gui/startup_overlay.cpp`, `src/visualizer/tools/selection_tool.cpp` | Draw-list overlays, tool hints, selection cursor reads, pie menu, startup overlay, and transitional viewport chrome. | World/viewport geometry belongs in renderer-native overlay passes; command UI and tool affordances belong in Rml overlay documents. |
| Theme bridge | `src/visualizer/theme/theme.*`, `src/visualizer/gui/rmlui/rml_theme.cpp` | Theme model still stores ImGui vector/color/integer types and exposes ImGui color helpers. | Split theme tokens into renderer-neutral types, then keep only Rml RCSS and renderer-native adapters. |

## Native Panel Registry Blockers

These registered native panels still travel through the compatibility panel shell or
ImGui-era overlay plumbing even when their visible UI is partly RmlUi:

- `native.video_extractor`: floating video extractor wrapper.
- `native.selection_overlay`: viewport selection overlay.
- `native.node_transform_gizmo`, `native.cropbox_gizmo`, `native.ellipsoid_gizmo`,
  `native.viewport_gizmo`: transform, crop, ellipsoid, and orientation gizmo surfaces.
- `native.viewport_decorations`: viewport chrome/decorations.
- `native.pie_menu`: pie menu rendering and input.
- `native.startup_overlay`: startup/drag-hover overlay.
- `native.sequencer`: Rml sequencer document hosted through the native/direct panel path.
- `native.python_overlay`: Python viewport draw handlers via ImGui-shaped `PyUILayout`.
- `lfs.scene`: scene header panel hosted as a native panel.

## Already RmlUi-First

- Main menu, status bar, right shell, and shell frame:
  `src/visualizer/gui/rml_menu_bar.cpp`, `rml_status_bar.cpp`,
  `rml_right_panel.cpp`, `rml_shell_frame.cpp`.
- Modal overlay and global context menu:
  `src/visualizer/gui/rml_modal_overlay.cpp`, `global_context_menu.cpp`.
- Python console/editor panel:
  `src/visualizer/gui/panels/python_console_panel.cpp`,
  `src/visualizer/gui/rmlui/elements/python_editor_element.cpp`.
- Sequencer document and timeline:
  `src/visualizer/sequencer/rml_sequencer_panel*.cpp`.
- Most built-in Python app panels, including export/import/rendering/histogram/scripts,
  mesh2splat, asset manager, plugin marketplace, selection groups, selection controls,
  transform controls, input settings, and training panels.
- Vulkan UI textures and viewport textured overlays:
  `src/visualizer/gui/vulkan_ui_texture.cpp`,
  `src/visualizer/rendering/passes/vulkan_viewport_pass.cpp`.

## Recommended Removal Order

1. Port or freeze the Python immediate-mode UI API and remove raw ImGui/ImPlot
   bindings from Python-facing code.
2. Convert theme tokens away from `ImVec2`, `ImVec4`, and `ImU32`.
3. Replace GUI bootstrap/input/cursor/text-input ownership with RmlUi and SDL-native
   state.
4. Replace panel host/layout ImGui viewport and draw-list usage.
5. Port native viewport overlays, gizmos, pie menu, startup overlay, video extractor,
   and scene header surfaces to RmlUi or renderer-native overlay passes.
6. Remove `imgui`/`implot` from `vcpkg.json`, CMake targets, tests, and third-party
   license/docs references that are no longer true.

## Guardrails

- Do not add `imgui_impl_opengl*`, `glad`, or OpenGL-backed texture paths.
- Do not add new ImGui panels for profiler or Vulkan tooling; use RmlUi.
- Keep compatibility code scoped and removable: new state should live outside
  ImGui-specific types unless it is an adapter boundary.

# Reactive UI Migration: Remaining Work

Current branch: `reactive_ui`

Last completed commits:
- `681177eae Gate menu bar label sync by source version`
- `a62db2085 Remove remaining dirty panel intervals`
- `40f15f521 Remove redundant dirty panel intervals`
- `161d18287 Make asset manager panel dirty-driven`
- `d2d494d90 Make histogram panel dirty-driven`
- `65043c70b Make image preview panel dirty-driven`

Current verified state:
- Merge from `origin/master` is complete.
- First-party Python plugin panels no longer define `update_interval_ms`.
- Python panel updates are dirty/reactive driven through `NativeAppStore` subscriptions, explicit model update requests, or targeted deferred timers.
- Menu bar label localization/copying is gated by menu source version plus `language_generation`.
- Last broad Python panel slice passed: 191 tests.
- Last C++ build passed: `cmake --build build -j16`.

## Remaining Goal

Drive idle GUI CPU toward zero and keep p99 CPU UI work under 2 ms by removing remaining unconditional native/RmlUi panel work and tightening frame routing around explicit demand.

## Next Batch 1: Native Panel Preload Gating

Focus files:
- `src/visualizer/gui/panel_registry.{hpp,cpp}`
- `src/visualizer/gui/panel_layout.cpp`
- `src/visualizer/gui/scene_panel_native.{hpp,cpp}`

Tasks:
- Audit `preload_panels_direct`, `preload_single_panel_direct`, and `preload_child_panels_direct`.
- Avoid running expensive native/Rml preload paths when a panel has a valid cached direct draw layer and no relevant dirty state.
- Preserve layout measurement correctness. Do not skip first-frame or size-change measurement.
- Add a small explicit dirty/layout hint API if needed; avoid a broad "damage bus" abstraction unless it removes real complexity.

Acceptance:
- Cached right-panel branch does not call full scene-panel sync on idle frames.
- Scene tree, history, logging, and search still update on scene/selection/language/history/log changes.
- Existing scene panel behavior and input remain unchanged.

## Next Batch 2: Scene Panel Reactive Invalidations

Focus files:
- `src/visualizer/gui/scene_panel_native.cpp`
- `src/visualizer/gui/rmlui/elements/scene_graph_element.cpp`
- `src/visualizer/include/visualizer/app_store.hpp`

Tasks:
- Replace remaining passive scene-panel polling with explicit invalidation sources:
  - `scene_generation`
  - `selection_generation`
  - `language_generation`
  - undo-history generation
  - log-buffer generation
- Keep local input-driven updates immediate for search/filter/rename/context menu.
- Mark host content dirty exactly where data/model/DOM changes happen.

Acceptance:
- Idle cached frames do not rebuild scene rows.
- Selection changes update the scene panel without polling.
- Language changes relocalize the panel exactly once per generation.

## Next Batch 3: Bottom Dock And Sequencer

Focus files:
- `src/visualizer/gui/panel_layout.cpp`
- `src/visualizer/gui/native_panels.{hpp,cpp}`
- `src/visualizer/gui/sequencer_ui_manager.{hpp,cpp}`
- `src/visualizer/gui/rml_sequencer_overlay.{hpp,cpp}`

Tasks:
- Add cached bottom-dock rendering parallel to right-panel cached rendering.
- Gate sequencer preload/draw by visibility, input activity, animation state, and dirty state.
- Keep resizing and floating panel behavior live while the user interacts.

Acceptance:
- Idle frames hit cached bottom-dock/sequencer paths.
- Scrubbing, resizing, and keyframe edits remain live.

## Next Batch 4: Modal And Startup Overlay Demand

Focus files:
- `src/visualizer/gui/rml_modal_overlay.{hpp,cpp}`
- `src/visualizer/gui/startup_overlay.{hpp,cpp}`
- `src/visualizer/gui/gui_manager.cpp`

Tasks:
- Add cheap `needsRender()` style queries for modal/startup overlays.
- Only process/render full overlay paths when open, newly queued, input-active, theme/language changed, or first-frame settling is needed.
- Keep modal text input and IME handling intact.

Acceptance:
- Closed modal overlay does not enter full render path on idle frames.
- Startup overlay remains responsive and correctly localized/themed.

## Next Batch 5: Frame Router Cleanup

Focus files:
- `src/visualizer/gui/gui_manager.cpp`
- `src/visualizer/visualizer_impl.cpp`

Tasks:
- Tighten `GuiManager::render()` into explicit demand checks:
  - menu bar
  - right panel
  - bottom dock
  - viewport overlay
  - status bar
  - modal/context menu
  - startup overlay
- Keep `FrameDemand::store_dirty` in the top-level skip gate.
- Add or preserve `LOG_PERF` evidence around each branch.

Acceptance:
- Idle `loop_idle skip_gui_render=true` fires consistently after training stops.
- Store-only updates redraw only the affected RmlUi host/context.

## Final Verification

Run focused checks after each batch:

```bash
cmake --build build -j16
build/vcpkg_installed/x64-linux/tools/python3/python3.12 -m pytest \
  tests/python/test_rmlui_image_sources.py \
  tests/python/test_histogram_panel.py \
  tests/python/test_asset_manager_panel.py \
  tests/python/test_scripts_panel.py \
  tests/python/test_selection_groups_panel.py \
  tests/python/test_mesh2splat_panel.py \
  tests/python/test_plugin_marketplace_panel.py \
  tests/python/test_export_panel.py \
  tests/python/test_import_dialog_panels.py \
  tests/python/test_input_settings_panel.py \
  tests/python/test_rendering_panel_regressions.py \
  tests/python/test_training_panel_regressions.py
```

End-to-end smoke:

```bash
./build/LichtFeld-Studio -d data/bicycle --output-path output --images images_4 \
  --strategy mcmc --max-cap 1500000 --log-level debug --train -i 7000 --start
```

Hard targets:
- Idle loaded scene with training paused: `loop_idle skip_gui_render=true` on every wake.
- Main-thread GUI idle CPU under 0.5%.
- `gui_render.cpu_ui_before_vulkan_begin` p99 under 2 ms over a 60-second steady-state run.
- A 1 Hz `fps` store update should only dirty/redraw status bar paths; viewport overlay and right panel should use cached branches.


codex resume 019e686d-2ad4-7b81-9ce0-bcef32d1d113

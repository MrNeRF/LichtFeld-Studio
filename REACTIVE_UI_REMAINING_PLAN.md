# Reactive UI Migration: Remaining Work

Current branch: `reactive_ui`

Last completed commits:
- `00c2a5864 Cache idle bottom dock sequencer rendering`
- `131da29ec Publish scene panel invalidations through app store`
- `a0034df40 Gate native scene panel sync by visible sources`
- `a1a860dc3 Document remaining reactive UI migration work`
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
- Native scene panel sync is source-stamped and driven by store-published scene/selection generations.
- Bottom dock has a cached render branch; the sequencer panel queues cached RmlUi textures on idle frames.
- Last broad Python panel slice passed: 191 tests.
- Last focused sequencer C++ slice passed: `./build/tests/lichtfeld_tests --gtest_filter='*Sequencer*'`.
- Last C++ build passed: `cmake --build build -j16`.

## Remaining Goal

Drive idle GUI CPU toward zero and keep p99 CPU UI work under 2 ms by removing remaining unconditional native/RmlUi panel work and tightening frame routing around explicit demand.

## Completed Batches

- Native scene panel preload gating: `a0034df40`.
- Scene panel reactive invalidation publication: `131da29ec`.
- Bottom dock and sequencer cached rendering: `00c2a5864`.

## Next Batch 1: Modal And Startup Overlay Demand

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

## Next Batch 2: Frame Router Cleanup

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

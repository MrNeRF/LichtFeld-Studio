# Reactive UI Migration: Remaining Work

Current branch: `reactive_ui`

Last completed commits:
- `ed519dac5 Gate idle panel registry branches`
- `f9107672c Cache idle right panel shell rendering`
- `22582d69d Cache idle viewport overlay rendering`
- `a7a76ec5b Cache idle status bar rendering`
- `4f7477613 Gate idle legacy Python popup draws`
- `f5f2dec65 Cache idle shell frame rendering`
- `099ea616e Cache idle menu bar rendering`
- `0130e5ce1 Cache idle context menu rendering`
- `fcb5b0649 Cache modal and startup overlay rendering`
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
- Modal requests wake the event loop, closed modal overlays skip input/render work, and modal/startup overlays reuse cached RmlUi textures once stable.
- Global context menus are gated behind pending/open state and reuse cached RmlUi textures while open and idle.
- The Rml menu bar keeps existing input/menu semantics but reuses cached foreground textures when labels, theme, size, and open dropdown state are stable.
- The shell frame no longer rewrites region properties or queues a raw RmlUi render on stable frames; it reuses a cached background texture keyed by layout/theme.
- The status bar keeps live rendering for active/dirty frames but uses a cached texture from `renderCached()` on stable frames.
- The viewport overlay preserves document-hook, tooltip, and input invalidation but uses cached foreground texture blits for stable frames.
- The right-panel Rml shell now reuses a cached texture when tab/layout/input flags are stable; dirty tab, resize, splitter, or hover paths still refresh.
- Legacy Python popup drawing is guarded by an explicit hook-presence predicate, so idle frames skip Python UI preparation/GIL work when no popup callback is registered.
- `GuiManager::render()` now computes explicit panel-space demand flags and skips side-panel preload, floating-panel hit testing/draw, status-bar plugin draw, and screen-overlay renderer work when those spaces have no enabled panels.
- Last broad Python panel slice passed: 191 tests.
- Last frame-router slice passed: `./build/tests/lichtfeld_tests --gtest_filter='*Shell*:*Rml*:*Menu*:*Modal*:*Startup*'` and the 191-test Python panel suite.
- Last focused Python popup/plugin slice passed: 74 tests from plugin system, plugin API surface, and import dialog panels.
- Last focused status/Rml C++ slice passed: `./build/tests/lichtfeld_tests --gtest_filter='*Status*:*Rml*:VisualizerPostWorkTest.*'`.
- Last focused training/rendering Python slice passed: 51 tests.
- Last focused viewport overlay-adjacent slice passed: 23 Rml/modal/startup C++ tests and 28 Python rendering/selection/input tests.
- Last focused right-panel shell slice passed: 12 Rml/menu C++ tests and 57 Python rendering/training/selection tests.
- Last focused menu Python slices passed: 226 plugin/menu tests plus 14 menubar/menu schema/API tests.
- Last focused context-menu-adjacent Python slice passed: 41 tests from plugin marketplace, input settings, and import dialogs.
- Last focused modal/post-work C++ slice passed: `./build/tests/lichtfeld_tests --gtest_filter='PyModalRegistryRegression.*:VisualizerPostWorkTest.*'`.
- Last focused modal/startup C++ slice passed: `./build/tests/lichtfeld_tests --gtest_filter='*Modal*:*Startup*'`.
- Last focused sequencer C++ slice passed: `./build/tests/lichtfeld_tests --gtest_filter='*Sequencer*'`.
- Last focused shell/Rml C++ slice passed: `./build/tests/lichtfeld_tests --gtest_filter='*Shell*:*Rml*'`.
- Last C++ build passed: `cmake --build build -j16`.

## Remaining Goal

Drive idle GUI CPU toward zero and keep p99 CPU UI work under 2 ms by removing remaining unconditional native/RmlUi panel work and tightening frame routing around explicit demand.

## Completed Batches

- Native scene panel preload gating: `a0034df40`.
- Scene panel reactive invalidation publication: `131da29ec`.
- Bottom dock and sequencer cached rendering: `00c2a5864`.
- Modal and startup overlay demand/caching: `fcb5b0649`.
- Context menu demand/caching: `0130e5ce1`.
- Menu bar cached rendering: `099ea616e`.
- Shell frame cached rendering: `f5f2dec65`.
- Legacy Python popup demand gating: `4f7477613`.
- Status bar cached rendering: `a7a76ec5b`.
- Viewport overlay cached rendering: `22582d69d`.
- Right-panel shell cached rendering: `f9107672c`.
- Idle panel registry branch gating: `ed519dac5`.

## Next Batch 1: Frame Router Cleanup

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

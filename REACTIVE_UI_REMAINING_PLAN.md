# Reactive UI Migration: Remaining Work

Current branch: `reactive_ui`

Last completed commits:
- `781779335 Skip stable gizmo tool-state updates`
- `6dbdbfe60 Skip idle menu bar hover scans`
- `54f94aeae Skip idle viewport overlay input hit tests`
- `8b351724d Skip idle right panel input hit tests`
- `ac6daaa15 Skip stable editor context refreshes`
- `96a8ebb0d Gate idle panel registry branches`
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
- Merge from `origin/master` is complete, including the GT camera-metrics crash fix and Vulkan mip-toggle fix.
- `AppStore.fps` store publications now drive the status-bar FPS value after the first reactive FPS update instead of only dirtying a repaint that still read `RenderingManager::getAverageFPS()`.
- Reactive store subscriber exceptions are logged per callback and no longer abort the rest of the GUI-thread drain.
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
- `EditorContext::update()` is now source-stamped at the router level and is skipped on unrelated store-only redraws while still refreshing on scene generation, selection generation, trainer state, structural scene changes, or input activity.
- `RmlRightPanel::processInput()` keeps existing drag/focus/layout semantics but skips the RmlUi hover hit-test on frames with no pointer, keyboard, text, drag, focus-blur, or layout work.
- `RmlViewportOverlay::processInput()` now preserves existing hover/focus claims but skips the RmlUi element hit-test on unchanged, activity-free frames with no VRAM HUD drag capture.
- `RmlMenuBar::processInput()` now reuses the previous hovered menu label on unchanged, pointer-idle frames with stable context size and no pending render work.
- `GizmoManager::updateToolState()` is source-stamped and returns early when selected-node presence, active tool, gizmo type, selection submode, UI visibility, and tool/renderer ownership are stable.
- Last gizmo/tool-state slice passed: `cmake --build build -j16`, 38 focused Python toolbar/tool/rendering/selection/input tests, and 14 non-GPU C++ input/Rml/post-work tests.
- Last end-to-end smoke passed via MCP/runtime inspection: `./build/LichtFeld-Studio -d data/bicycle --output-path output --images images_4 --strategy mcmc --max-cap 1500000 --log-level perf --log-file /tmp/reactive-ui-perf.log --train -i 7000 --no-splash`.
- Fresh smoke metrics: `gui_render.cpu_ui_before_vulkan_begin` across 768 rendered frames had median 0.31 ms, p95 0.57 ms, p99 0.99 ms, max 42.34 ms from startup/preload. After training finished, idle wakes logged `loop_idle skip_gui_render=true` with zero GUI render work.
- Surgical `fps` store update passed via MCP `editor.run`: three `NativeAppStore.fps.value` updates produced only cached right-panel and viewport-overlay branches plus status-bar cache refreshes; no live `renderRightPanel` calls. Post-update CPU UI across 6 rendered frames had median 0.98 ms, p99 1.21 ms, max 1.55 ms, then idle skips resumed.
- Last broad Python panel slice passed: 204 tests.
- Last frame-router slice passed: `./build/tests/lichtfeld_tests --gtest_filter='*Shell*:*Rml*:*Menu*:*Modal*:*Startup*'` and the 191-test Python panel suite.
- Last editor-context router slice passed: `./build/tests/lichtfeld_tests --gtest_filter='*Selection*:*Rml*:*Menu*:*Modal*:*Startup*:VisualizerPostWorkTest.*'` and 67 Python toolbar/rendering/training/selection tests.
- Last right-panel input slice passed: `./build/tests/lichtfeld_tests --gtest_filter='*Rml*:*Menu*:*Modal*:*Startup*:VisualizerPostWorkTest.*'` and 68 Python rendering/training/selection/input tests.
- Last viewport-overlay input slice passed: `./build/tests/lichtfeld_tests --gtest_filter='*Rml*:*Modal*:*Startup*:*Selection*:VisualizerPostWorkTest.*:-PythonIntegrationTest.*:RotatedShCorrectnessTest.*'` and 41 Python toolbar/viewport/rendering/selection/input tests. A broader exploratory C++ filter also ran and exposed unrelated pre-existing failures in `PythonIntegrationTest.RenderViewMatchesViewportVerticalOrientation` and `RotatedShCorrectnessTest.ViewportParityWithExportUnderRotationAndNonUniformScale`.
- Last menu-bar input slice passed: `./build/tests/lichtfeld_tests --gtest_filter='*Menu*:*Rml*:VisualizerPostWorkTest.*'` and 13 Python menubar/menu-schema/toolbar tests. A broader plugin-heavy Python command hit unrelated local setup failures (`uv` unavailable and `lichtfeld.log` missing in that test harness).
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
- Last post-merge focused C++ slice passed: `./build/tests/lichtfeld_tests --gtest_filter='ReactiveStoreTest.*:CameraImageLoadTest.*:SplitViewServiceTest.GtRenderCameraUsesVisualizerCameraAxesAndNormalizedSceneRotation:SplitViewServiceTest.SharedCameraPoseHelperNormalizesSceneRotationAndAppliesVisualizerAxes:SplitViewServiceTest.GtComparisonPlanPreservesGtTextureOrigin:RenderSettingsBackendNormalization.*:RenderSettingsProxy.*'`.
- Last post-merge Python panel slice passed: `build/vcpkg_installed/x64-linux/tools/python3/python3.12 -m pytest tests/python/test_rmlui_image_sources.py tests/python/test_histogram_panel.py tests/python/test_asset_manager_panel.py tests/python/test_scripts_panel.py tests/python/test_selection_groups_panel.py tests/python/test_mesh2splat_panel.py tests/python/test_plugin_marketplace_panel.py tests/python/test_export_panel.py tests/python/test_import_dialog_panels.py tests/python/test_input_settings_panel.py tests/python/test_rendering_panel_regressions.py tests/python/test_training_panel_regressions.py tests/python/test_store.py tests/python/test_viewport_overlay_hooks.py`.

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
- Idle panel registry branch gating: `96a8ebb0d`.
- Stable editor context refresh gating: `ac6daaa15`.
- Idle right panel input hit-test skip: `8b351724d`.
- Idle viewport overlay input hit-test skip: `54f94aeae`.
- Idle menu bar hover scan skip: `6dbdbfe60`.
- Stable gizmo tool-state update skip: `781779335`.

## Remaining Follow-Ups

The frame-router/cache migration has passed the recorded idle and p99 targets on this branch. Remaining work is release hygiene rather than another required hot-path batch:

- Run OS-level main-thread/process idle CPU measurement on the release validation machine.
- Keep the legacy Python UI hook API during the planned deprecation window, then remove it in the scheduled EOL release.
- Revisit native GT metrics / VRAM HUD DOM updates only if future profiling shows churn there; the current path is event-driven and cached.
- Re-run the smoke/perf command after any additional router or panel invalidation change.

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
  --strategy mcmc --max-cap 1500000 --log-level perf \
  --log-file /tmp/reactive-ui-perf.log --train -i 7000 --no-splash
```

Hard targets:
- Idle loaded scene with training paused: `loop_idle skip_gui_render=true` on every wake.
- Main-thread GUI idle CPU under 0.5%.
- `gui_render.cpu_ui_before_vulkan_begin` p99 under 2 ms over a 60-second steady-state run.
- A 1 Hz `fps` store update should only dirty/redraw status bar paths; viewport overlay and right panel should use cached branches.

Verified on 2026-05-28:
- End-to-end smoke reached iteration 7000 and finished via MCP runtime state.
- `gui_render.cpu_ui_before_vulkan_begin` p99 was 0.99 ms across the smoke run's rendered frames.
- Post-training idle wakes consistently logged `loop_idle skip_gui_render=true`.
- Three fps-only store updates kept right panel and viewport overlay cached while refreshing status bar.

Residual follow-ups:
- Measure OS-level main-thread/process idle CPU externally if required by release validation.
- Keep the old Python UI hook API only for the planned deprecation window; removal remains a future release step.


codex resume 019e686d-2ad4-7b81-9ce0-bcef32d1d113

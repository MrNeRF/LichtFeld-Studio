# Reactive UI Migration Handoff

Branch: `reactive_ui`

## Current State

The branch implements the reactive RmlUi migration and has passed the recorded idle and p99 UI CPU targets on local smoke runs.

Implemented:

- Native `AppStore` and `core::reactive::Store` for typed, dirty-driven UI state.
- Python `lfs_plugins.ui.store.AppStore` facade and `lichtfeld.ui.store` bridge.
- Store-driven updates for the first-party Python panels that previously relied on periodic polling.
- Cached render paths for menu bar, status bar, shell frame, modal/startup overlays, viewport overlay, right panel, bottom dock, and plugin panel spaces.
- Frame router demand gating for side-panel preload, panel input hit tests, editor-context refresh, gizmo state sync, document hooks, and RmlUi cached texture submission.
- Frame-pacing timers separated from CPU UI timers so swapchain waits no longer appear as `gui_render` work.
- Thresholded micro-timers for hot UI subscopes to avoid performance logging becoming the measured overhead.

The old Python UI hook API remains available for the deprecation window. New or migrated code should publish through `AppStore` and invalidate panel data models or cached panel hosts directly.

## Verification Snapshot

Last full build:

```bash
cmake --build build -j16
```

Last focused C++ slice:

```bash
./build/tests/lichtfeld_tests \
  --gtest_filter='LoggerTest.*:VisualizerPostWorkTest.*:PanelRegistryAnimationDemandTest.*:PanelLayoutRenderDemandTest.*'
```

Previously recorded smoke results on this branch:

- `gui_render.cpu_ui_before_vulkan_begin` p99 below 2 ms during steady-state runs.
- Loaded-scene idle wakes resumed as `loop_idle skip_gui_render=true`.
- FPS-only store updates refreshed the status bar while right panel and viewport overlay stayed on cached paths.

Use this smoke command after any router, panel invalidation, or performance-instrumentation change:

```bash
./build/LichtFeld-Studio -d data/bicycle --output-path output --images images_4 \
  --strategy mcmc --max-cap 1500000 --log-level perf \
  --log-file /tmp/reactive-ui-perf.log --train -i 7000 --no-splash
```

## Remaining Release Gates

- Run the OS-level main-thread/process idle CPU check on the release validation machine.
- Keep the legacy Python UI hook API only for the planned deprecation window, then remove it in the scheduled EOL release.
- Re-run the smoke/perf command after any further router or panel invalidation change.
- If sub-1 ms interaction frames are required while scrubbing toolbar flyouts, profile the flyout/hover RmlUi layout path specifically; idle frame routing is no longer the main bottleneck.

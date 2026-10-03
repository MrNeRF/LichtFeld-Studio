# Node Editor V7 validation

Validated on Apple M5 Max, macOS, Retina 2× UI density, using the isolated app
on MCP port 45695 and `garden.ply` (1,000,000 stored splats). The viewer uses
Vulkan/MoltenVK. Tensor tests separately cover CPU, Metal and Vulkan. Changes
are uncommitted on top of `cb375ae49`.

## Design decisions

- Arrange defaults to the selected nodes; an empty selection arranges the
  complete graph. Selection layout preserves the original bounding-box centre,
  leaves other locations and the current view unchanged, and records one undo
  entry. Header, context-menu and Shift+L actions share this behaviour.
  `nodes_editor_arrange` accepts `selection_only: false` to force whole-graph
  layout, or an explicit `nodes` list. An explicit empty list is a no-op.
- Wires retain a direct cubic curve when unobstructed. Otherwise they use
  clearance lanes around cards, with rounded corners and separate offsets for
  fan-in/fan-out. A coordinate-compressed rectilinear search handles multiple
  turns and narrow corridors. Backward links travel around cards. Rendering,
  selection and knife intersection use the same cached paths. Node bounds,
  socket positions, links and UI density invalidate routes; pan and selection
  do not. Zoom only changes routes when readable-title sizing changes bounds.
- Colour inputs use one swatch and the existing picker, without stacked RGB
  fields. Signed grading inputs with descriptor bounds `[-1, 1]` also get an
  inspector colour-offset wheel. Its opponent-colour plane preserves the mean
  while dragging, subject to the descriptor bounds; double-click resets all
  offsets to zero. Neutral offsets display as neutral grey, not black. The
  standard picker maps these signed values to/from its `[0, 1]` colour space.
  Both controls share the existing single-gesture undo path.
- Add, Frame, Arrange, Open and Save now use existing icon assets and localised
  tooltips. Category title colours blend the active theme's semantic colours
  into the card surface at 32%; titles and Add categories share category icons.
  Existing theme text/icon colours preserve dark/light contrast.

## Real-app checks

The ten-node graph includes Group Input/Output, Remove Floaters, Scale Clamp,
Colour Correct, Recolour, HSV Range, Join Geometry, Splats to Points and Math.
Join has three inputs; the two processing branches plus the direct source
produce 2,992,498 splats. The last two nodes are unconnected helper examples.

The graph and editor were manipulated through MCP. `ui.pointer` supplied
40-step node, wire, pan, wheel-control and picker gestures. Screenshots were
captured at step 20 while the button was still held, decoded from
`render_capture_window`, and visually inspected. Assertions checked:

- Selection-only Arrange moves only the requested nodes; header Arrange also
  records exactly one undo entry, and undo restores every location exactly.
- Cards, wires and the colour-wheel dot update before pointer release.
- A 40-step grading-wheel drag yields one undo entry and non-zero signed
  offsets; undo restores the original values. The existing colour picker also
  changes the values during its drag and produces one undo entry.
- A deliberately backward Join → Group Output link routes around the right,
  bottom and left of intervening cards, rather than through them.
- Dark/light title text and icons remain legible against the stronger tints.
  Add categories use matching icons, and the inspector uses compact swatches
  with wheels only for signed grading inputs.

## Canvas timings

Existing `lf.nodes.performance()` counters were reset immediately before each
gesture. Times include measured canvas CPU work, not whole-viewer frame time or
GPU execution. Each drag has 41 recorded frames; zoom has 20 wheel events.

| Gesture | Mean ms | p95 ms | Maximum ms | Evaluation requests / evaluations |
| --- | ---: | ---: | ---: | ---: |
| Node drag | 0.242 | 0.481 | 0.776 | 0 / 0 |
| Wire drag | 0.402 | 1.034 | 1.188 | 0 / 0 |
| Middle-drag pan | 0.407 | 1.044 | 1.219 | 0 / 0 |
| Wheel zoom | 1.771 | 4.597 | 4.597 | 0 / 0 |

The requested ≤2 ms drag budget was met for these captures. Zoom still has
outliers above 2 ms; this is not a claim that every canvas interaction meets a
strict 2 ms worst-frame budget. No before/after speedup is claimed.

The 50-node/58-link routing regression test repeatedly moves an obstacle and
recomputes paths: mean **0.156009 ms**, maximum **0.17775 ms** in the final Metal
test invocation. Pan/selection/unchanged-layout caching is separately asserted.
Raw gesture measurements are in `build/node-v7-frame-times.json` and
`build/node-v7-gestures.log`.

## Screenshots

All files below are in `/private/tmp/node-editor-visual/`:

- `v7-initial-dark.png`, `v7-overview-dark.png`, `v7-arranged-dark.png`
- `v7-compact-colour-dark.png`, `v7-compact-colour-light.png`,
  `v7-arranged-light.png`
- `v7-node-mid-drag.png`, `v7-wire-mid-drag.png`, `v7-pan-mid-drag.png`
- `v7-selection-before.png`, `v7-selection-arranged.png`,
  `v7-header-arranged-selection.png`
- `v7-backward-link-dark.png`
- `v7-grading-dark.png`, `v7-colour-wheel-mid-drag.png`,
  `v7-colour-wheel-edited.png`
- `v7-colour-picker.png`, `v7-colour-picker-mid-drag.png`,
  `v7-colour-picker-edited.png`
- `v7-add-dark.png`, `v7-add-light.png`, `v7-add-categories-dark.png`,
  `v7-add-categories-light.png`

## Build, tests and gates

Build with `SDKROOT=/Library/Developer/CommandLineTools/SDKs/MacOSX26.5.sdk`:

```sh
ninja -C build lfs_core lichtfeld_tests LichtFeld-Studio lfs_py lichtfeld_visualizer_tests
```

All targets succeeded. The existing Python static-library deployment-version
linker warnings remain. C++ changes were clang-formatted using the repository
configuration; `git diff --check` is clean.

Metal and Vulkan each used:

```sh
build/tests/lichtfeld_tests --tensor-backend=metal --gtest_filter='*Nodes*:NodeCanvas*:McpNodeToolsTest.*:*RadiusNeighbor*:*radius_neighbor*'
build/tests/lichtfeld_tests --tensor-backend=vulkan --gtest_filter='*Nodes*:NodeCanvas*:McpNodeToolsTest.*:*RadiusNeighbor*:*radius_neighbor*'
```

The leading wildcard includes the parameterised `Backends/NodesCore` suites,
which `Nodes*` alone does not match. Exact summary tails:

```text
# build/node-v7-metal-tests.log
[==========] 178 tests from 8 test suites ran. (2741 ms total)
[  PASSED  ] 175 tests.
[  SKIPPED ] 3 tests, listed below:
[  SKIPPED ] Backends/NodesCore.CpuMeshUploadsOnceAndJoinsGpuSplats/CPU
[  SKIPPED ] Backends/NodesCoreScale.MillionSplatsColourSelectionCleanupAndTorus/CPU
[  SKIPPED ] Backends/NodesCoreScale.SpatialOperationsStayInteractive/CPU

# build/node-v7-vulkan-tests.log
[==========] 178 tests from 8 test suites ran. (2669 ms total)
[  PASSED  ] 175 tests.
[  SKIPPED ] 3 tests, listed below:
[  SKIPPED ] Backends/NodesCore.CpuMeshUploadsOnceAndJoinsGpuSplats/CPU
[  SKIPPED ] Backends/NodesCoreScale.MillionSplatsColourSelectionCleanupAndTorus/CPU
[  SKIPPED ] Backends/NodesCoreScale.SpatialOperationsStayInteractive/CPU
```

Those skips intentionally exclude GPU-only tests from the CPU parameterisation.
An additional explicit CPU-parameter run used `--gtest_filter='*Nodes*/CPU'`
with Metal selected for runtime initialisation (the executable has no `cpu`
backend CLI option):

```text
# build/node-v7-cpu-tests.log
[==========] 45 tests from 2 test suites ran. (986 ms total)
[  PASSED  ] 42 tests.
[  SKIPPED ] 3 tests, listed below:
```

All visualizer tests, including 19 node-canvas widget tests:

```text
# build/tests/lichtfeld_visualizer_tests; build/node-v7-visualizer-tests.log
[==========] 185 tests from 23 test suites ran. (1562 ms total)
[  PASSED  ] 183 tests.
[  SKIPPED ] 2 tests, listed below:
[  SKIPPED ] ViewerGpuMemory.VulkanStatusDoesNotRequireCudaOrReportDeviceUsageAsProcessUsage
[  SKIPPED ] ViewerGpuMemory.CudaStatusKeepsDeviceUsageAndItsOptionalUtilizationReading
```

These two GPU-memory tests intentionally skip their non-Apple paths here.
New coverage includes subset layout, retained card identity/no evaluation,
MCP subset validation, obstacle/backward/parallel routes, narrow and multi-turn
corridors, route caching, knife hit testing, colour-wheel reset/undo and real
RmlUi pointer sequences for wheel and picker dragging.

```sh
LFS_TEST_BUILD_DIR=build /private/tmp/claude-501/-Users-mrnerf-projects-LichtFeld-Studio/6947805d-f038-4b64-9def-572e7045027a/scratchpad/venv/bin/python -m pytest tests/python/test_nodes.py tests/python/test_localization_contracts.py -q
python3 tools/check_backend_neutrality.py
python3 tools/error_debt_census.py --baseline tools/error_debt_baseline.json
python3 tools/check_ui_hardcoded.py
```

```text
============================== 35 passed in 2.13s ==============================
Backend neutrality: 0 violation(s).
no new error-debt violations
No likely hardcoded UI strings found.
```

Logs are `build/node-v7-{pytest,backend-gate,debt-gate,ui-gate}.log`.

## Files changed

- `src/app/mcp_node_editor_tools.cpp`
- `src/visualizer/CMakeLists.txt`
- `src/visualizer/gui/area_editors.cpp`
- `src/visualizer/gui/node_canvas_interaction.{hpp,cpp}`,
  `node_canvas_layout.cpp`, new `node_canvas_routing.cpp`
- `src/visualizer/gui/rmlui/elements/node_canvas_element.{hpp,cpp}`,
  `node_canvas_fields.cpp`, `node_canvas_menu.cpp`, `node_canvas_widgets.{hpp,cpp}`,
  new `colour_offset_element.{hpp,cpp}`
- `src/visualizer/gui/rmlui/rmlui_manager.cpp`
- `src/visualizer/gui/rmlui/resources/node_editor{,.theme}.rcss`
- `src/visualizer/gui/resources/locales/{en,de,es,fr,it,ja,ko,nl,pl,zh}.json`
- `tests/test_node_canvas_interaction.cpp`, `tests/test_node_canvas_widgets.cpp`,
  `tests/test_mcp_node_tools.cpp`
- `docs/docs/development/mcp/node-editor.md`,
  `docs/docs/development/node-graph/design.md`, this report

## Remaining limits

- If overlapping cards completely cover a socket, no unobstructed path can
  reach that socket. The link is retained; moving the cards or Arrange makes
  routing possible. Non-overlapping arranged cards are covered by route tests.
- The zoom timing tail noted above exceeds 2 ms; the measured drags do not.
- These are local macOS/MoltenVK visual checks, not native Windows/Linux or
  CUDA rendering checks. This round does not modify tensor/backend code.

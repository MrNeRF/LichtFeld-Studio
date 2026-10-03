# V8: visible node bodies and sidebar text

## Root cause and fix

The V7 overview was framed at approximately **0.504 zoom**. `layoutCard()` set
`lod-labels` below 0.65; its stylesheet made the entire `.socket-rows` subtree
transparent and the footer hidden. That also hid the settings expander. The
elements were present, and the category title markup and theme colours were
not the cause. V7's visual verification missed the inappropriate LOD rule.

The initial new regression test failed on both themes at 0.5 and 0.6, reporting
ancestor opacity **0** and hidden footer visibility. Its failing output is in
`build/node-v8-regression-before.log`.

The fix removes body-hiding LOD entirely. Only inline value editors are hidden
below 0.75, with socket labels retained as their fallback. Body/caption fonts
retain the 11dp minimum at zoom ≥0.75, then scale below that to fit their rows;
titles retain their readable minimum. This avoids both clipped footer text and
enlarging cards into neighbouring cards when zooming out. Card bounds include
their borders, and full footer text is available in its tooltip when elided.

Modifier names now use a `text-overflow: ellipsis` label while unfocused and the
existing text input while editing. Their full name is the tooltip. Inspector
status text uses non-breaking spaces inside items, leaving wrapping at the
` · ` separators: `15 ms` stays together, as does `11% selected`.

## Regression coverage

`tests/test_node_canvas_widgets.cpp` now runs the actual node-editor base and
theme styles through RmlUi layout/render passes in both LichtFeld dark and light:

- At zoom 1, 0.75, 0.6, 0.5, 0.3, 1.4 and back to 1, socket labels, the settings
  expander and footer have non-zero dimensions and non-empty contents.
- Their ancestors are displayed, visible and non-transparent; text/background
  contrast is at least 3:1. Text remains ≥11dp at zoom ≥0.75.
- Their bounds remain inside the card; arranged cards stay non-overlapping
  across the tested zoom range.
- The modifier label has ellipsis styling. A real RmlUi pointer sequence hits
  the rename input, resolves its full-name tooltip and focuses it for editing;
  rename still commits correctly.
- Inspector status text contains non-breaking value/unit spaces.

The fixture also clears stale process-wide scene callbacks before constructing
its scene, so the new first test is isolated from earlier visualizer fixtures.

## Full-resolution visual checks

An isolated instance on port 45695 loaded `garden.ply` (1M splats), with the same
ten-node graph as V7. MCP set the view, selection and theme; `ui.pointer` toggled
the sidebar. Every saved capture was opened at original resolution and compared
against `/private/tmp/node-editor-visual/v5-stepper.png`, including replacement
captures made during iteration. The final images are 2×-density window captures:

- `/private/tmp/node-editor-visual/v8-overview-dark.png`
- `/private/tmp/node-editor-visual/v8-overview-light.png`
- `/private/tmp/node-editor-visual/v8-selected-node.png`
- `/private/tmp/node-editor-visual/v8-selected-node-light.png`
- `/private/tmp/node-editor-visual/v8-zoom-0.6.png`
- `/private/tmp/node-editor-visual/v8-zoom-1.4.png`

The overviews show labels, expanders and footers on the formerly empty cards,
including Scale Clamp, Recolour, Join Geometry and Group Output. The selected
views show complete statistics at zoom 1, an ellipsised long modifier name and
the inspector's duration wrapping as a whole item. The 0.6 capture hides the
sidebar to show the entire graph; the other views retain it. Nodes without
settings do not have an expander, and unevaluated helpers display `—`.

## Build and test tails

All requested targets plus the visualizer tests built successfully using
`SDKROOT=/Library/Developer/CommandLineTools/SDKs/MacOSX26.5.sdk`:

```sh
ninja -C build lfs_core lichtfeld_tests LichtFeld-Studio lfs_py lichtfeld_visualizer_tests
```

The existing Python static-library deployment-version linker warnings remain.
Changed C++ files are clang-formatted; `git diff --check` is clean.

Metal and Vulkan each ran the broadened filter
`*Nodes*:NodeCanvas*:McpNodeToolsTest.*:*RadiusNeighbor*:*radius_neighbor*`
to include the parameterised CPU/Metal/Vulkan node suites:

```text
# build/node-v8-metal-tests.log
[==========] 178 tests from 8 test suites ran. (2887 ms total)
[  PASSED  ] 175 tests.
[  SKIPPED ] 3 tests, listed below:
[  SKIPPED ] Backends/NodesCore.CpuMeshUploadsOnceAndJoinsGpuSplats/CPU
[  SKIPPED ] Backends/NodesCoreScale.MillionSplatsColourSelectionCleanupAndTorus/CPU
[  SKIPPED ] Backends/NodesCoreScale.SpatialOperationsStayInteractive/CPU

# build/node-v8-vulkan-tests.log
[==========] 178 tests from 8 test suites ran. (2813 ms total)
[  PASSED  ] 175 tests.
[  SKIPPED ] 3 tests, listed below:
[  SKIPPED ] Backends/NodesCore.CpuMeshUploadsOnceAndJoinsGpuSplats/CPU
[  SKIPPED ] Backends/NodesCoreScale.MillionSplatsColourSelectionCleanupAndTorus/CPU
[  SKIPPED ] Backends/NodesCoreScale.SpatialOperationsStayInteractive/CPU

# Explicit *Nodes*/CPU filter; build/node-v8-cpu-tests.log
[==========] 45 tests from 2 test suites ran. (1061 ms total)
[  PASSED  ] 42 tests.
[  SKIPPED ] 3 tests, listed below:

# build/tests/lichtfeld_visualizer_tests; build/node-v8-visualizer-tests.log
[==========] 188 tests from 23 test suites ran. (1649 ms total)
[  PASSED  ] 186 tests.
[  SKIPPED ] 2 tests, listed below:
[  SKIPPED ] ViewerGpuMemory.VulkanStatusDoesNotRequireCudaOrReportDeviceUsageAsProcessUsage
[  SKIPPED ] ViewerGpuMemory.CudaStatusKeepsDeviceUsageAndItsOptionalUtilizationReading
```

The three CPU skips are GPU-only tests. The two visualizer skips concern
non-Apple GPU-memory paths. CPU parameter tests use `--tensor-backend=metal`
for runtime initialisation; the executable does not accept `cpu` as that CLI
option. The full visualizer run includes 22 passing `NodeCanvasWidgets` tests.

Python used the specified scratchpad venv with `LFS_TEST_BUILD_DIR=build` and
`tests/python/test_nodes.py tests/python/test_localization_contracts.py -q`:

```text
============================== 35 passed in 1.70s ==============================
Backend neutrality: 0 violation(s).
no new error-debt violations
No likely hardcoded UI strings found.
```

Logs: `build/node-v8-{pytest,backend-gate,debt-gate,ui-gate}.log`. All three
repository gate commands used the original requested arguments and baseline.
No locale keys were added or removed.

## Files changed in this follow-up

- `src/visualizer/gui/rmlui/elements/node_canvas_element.cpp`
- `src/visualizer/gui/rmlui/elements/node_canvas_widgets.{hpp,cpp}`
- `src/visualizer/gui/rmlui/elements/node_canvas_status.cpp`
- `src/visualizer/gui/rmlui/resources/node_editor.rcss`
- `tests/test_node_canvas_widgets.cpp`
- `tests/test_rml_static_style_boundaries.cpp`
- `docs/docs/development/node-graph/design.md`
- This report

V7 changes remain intact and unstaged. No commits were created. Below editing
zoom, text is deliberately smaller; long footers can still elide horizontally,
but neither labels nor footer contents are hidden by LOD.

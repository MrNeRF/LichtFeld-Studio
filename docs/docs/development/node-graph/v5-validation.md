# Node Editor V5 validation

## Editor fixes and verification

Validated on Apple M5 Max, macOS, with the million-splat `garden.ply`, using an
isolated home and MCP port 45695. Only that instance was stopped afterwards;
the existing instance on port 45698 was left running. No commits or staging.

### Sidebar

The reported word-per-line collapse did not reproduce on the freshly launched,
merged HEAD. The preceding V4 fix already protects RmlUi's private scrollbar
children during retained sidebar patching: replacing those children was the
root cause of the selection-dependent narrow/corrupt inspector. This round
also removes the sidebar's dependence on intrinsic/auto section widths:

- Sections and modifier rows explicitly fill their parent, with border-box sizing.
- The modifier name gets the remaining flex width, with icons aligned at the end.
- Add Modifier is a non-wrapping, border-box, full-width button. The baseline's
  button could extend beyond its section padding; that overflow is fixed.
- Inspector controls have a 120dp minimum width and retain the existing
  Properties-style label/control rows.

`SidebarFillsColumnAtOneAndTwoDpAcrossResizeAndSelection` runs with dp ratios
1 and 2, three editor heights, and repeated Colour Correct/Value selection.
It asserts section width >=260dp, control and editable-name widths >=120dp,
Add button width >=230dp and containment within section padding, and last-run
text height <=30dp. The private-scrollbar regression remains covered too.

Both real-app DPI captures were inspected alongside Properties. The modifier
name, Add button and last-run text fit normally; numeric controls fill their
column. The 1x capture used a temporary forced-1x `applyUiScale` diagnostic build,
since native Retina density is 2. That override was removed, the app rebuilt,
and the 2x capture taken again; `gui_manager.cpp` has no diff.

### Navigation and number fields

Canvas pan consults `InputController` and the same current-tool/held-key input
binding lookup used by the viewport. The default viewport pan is right-drag;
an unmoved right-click still opens the canvas menu. Middle-drag remains an
additional canvas pan gesture. The existing shared trackpad navigation mapping
and preference speeds are retained.

Live `ui_pointer` checks on garden:

- Trackpad-mode wheel `(dx=10, dy=8)` changed pan from `(38.420,325.948)` to
  `(-12.986,367.072)`.
- Middle drag `(800,650)` -> `(960,750)`, 40 steps, changed pan by `(160,100)`.
- Rebinding CAMERA_PAN to Alt+left changed pan by `(120,80)` for that injected
  drag. The binding was restored afterwards. A pure interaction regression
  also exercises the rebound binding.

Numeric fields share the existing scrub/typed-edit implementation. New end
buttons use the descriptor step, Shift x10 and Alt x0.1. They appear on canvas
hover and remain visible in the inspector; vector components use the same
widget. Holding repeats after 400ms, then every 70ms. One press/hold/scrub/typed
edit records one undo entry. Enter starts typing; Escape restores the value.
Widget tests cover modifiers, repeat, undo and keyboard editing. In the app,
an Exposure + click changed 0 -> 0.1; a 60-pixel scrub at 2x changed 0.1 -> 3.1.

### Names and MCP

All shared MCP graph/target resolution accepts UUID first, otherwise an exact
unique name. Ambiguous names fail without fuzzy matching. Graph/stack/editor
responses include identities and names. Resource paths accept the same lookup.
Graph create, rename and import choose unused numeric suffixes, including the
standalone Python library; stored modifier references stay canonical UUIDs.
The existing actionable error messages are unchanged.

The live graph workflow used `tree="Autumn Lawn"` and `target="garden"`.
Repeated creation returned `Autumn Lawn 2` and `Autumn Lawn 3`; export by name
resolved to the expected UUID. Tests cover rename/import suffixing and undo.

## Node and tensor design decisions

- Mesh to Splats is now a core tensor implementation, independent of geometry
  shaders. Density times surface area determines count, capped by Max Count.
  Area-weighted face draws and uniform barycentrics generate surface samples;
  tensor-built orthonormal frames produce wxyz rotations. Vertex normals and
  colours interpolate when present. Material textures are uploaded once per
  distinct image per evaluation, sampled with repeated UVs and no second V flip,
  and multiplied by material base colour. Material colour and grey are fallbacks.
  The renderer's Mesh2Splat panel/library is unchanged.
- Points to Splats uses half the mean three-nearest-neighbour distance. Device
  distance/sort scratch is chunked; radii clamp around their median. Above 300K,
  a deterministic random 4096-point sample supplies one radius. This requested
  large-cloud path is an approximation, not exact per-point kNN. The exact
  <=300K path remains quadratic work, with bounded scratch.
- `radius_neighbor_counts` shares the existing spatial hash seam on CPU, CUDA,
  Metal and Vulkan. It excludes the point itself, handles hash collisions
  without double-counting, returns Int32, and saturates at the requested limit.
  An optional query mask lets relative-radius octave buckets query all reference
  points without doing unnecessary queries. Floaters saturates at its threshold;
  Neighbour Count uses the Int32 limit. The coarse dense-grid shortcut is removed.
- Sampling gets per-call seeds through the public tensor API. It does not reset
  or advance the application's global RNG; regression tests verify isolation.
- Floater preview filters to candidates without changing their attributes.
  Scale Clamp uses the log min/max midpoint and half-log aspect range, blended
  by selection. Posterize quantizes clamped RGB and fades selected higher SH.
- Python declarations now require explicit `inputs`, `outputs`, `properties`
  lists and `execute(self, ctx)`. Attribute declarations and `evaluate` fallback
  are rejected. Registered callback classes release at Python exit before
  nanobind teardown, so declaration objects do not leak.
- Conversion tests cover sphere surface/normal/count/seed behavior, vertex,
  submesh material and texture colours, grid spacing, and >300K sampling. Radius
  tests compare against brute force with masks, coincident/non-finite points,
  collisions and saturation. Random scale tests verify the aspect guarantee.
  The million-splat fixture now contains guaranteed inside/outside clusters:
  exact neighbour counting correctly removed almost all of its old sparse
  random fixture before the torus assertion.

CUDA mirrors the existing radius-neighbour kernel/stream path but was **not
compiled or run** on this Mac. Vulkan tests use MoltenVK, not native Linux/Windows
Vulkan. Physical trackpad feel was not tested; injected events and navigation
unit tests cover the routing. No new localization keys were needed.

## Screenshots

All files are under `/private/tmp/node-editor-visual/`, decoded from
`render_capture_window` and visually inspected:

```text
v5-before.png
v5-sidebar-1x.png
v5-sidebar-2x.png
v5-stepper.png
v5-scrub.png
v5-swipe-pan.png
v5-middle-pan.png
v5-rebound-pan.png
```

## Test and gate output

All commands exited 0. Build uses
`SDKROOT=/Library/Developer/CommandLineTools/SDKs/MacOSX26.5.sdk` and
`ninja -C build lfs_core lichtfeld_tests LichtFeld-Studio lfs_py`.
`lichtfeld_visualizer_tests` was also built. The full rebuild retains existing
static-Python-library SDK warnings (objects built for macOS 26.6, link target
26.0). The final incremental build tail, in `build/node-v5-build.log`:

```text
[4/23] Staging Vulkan rasterizer shaders
[5/23] Copying lfs_plugins Python module
[6/23] Syncing visualizer runtime resources
[7/23] Refreshing git_version.h
[8/18] Refreshing lfs_core ABI stamp
```

Required `Nodes*:*RadiusNeighbor*:*radius_neighbor*` filter, Metal and Vulkan
respectively (`build/node-v5-metal.log`, `build/node-v5-vulkan.log`):

```text
[==========] 20 tests from 5 test suites ran. (817 ms total)
[  PASSED  ] 20 tests.
[==========] 20 tests from 5 test suites ran. (149 ms total)
[  PASSED  ] 20 tests.
```

The required prefix does not match parameterized `Backends/NodesCore` tests
other than the radius test, so the broader filter was also run with both backend
flags: `*Nodes*:*RadiusNeighbor*:*radius_neighbor*:NodeCanvas*:McpNodeToolsTest.*`.
It includes CPU, Metal and Vulkan parameterizations. Exact summary tails:

```text
# build/node-v5-all.log
[==========] 162 tests from 8 test suites ran. (2092 ms total)
[  PASSED  ] 161 tests.
[  SKIPPED ] 1 test, listed below:
[  SKIPPED ] Backends/NodesCoreScale.MillionSplatsColourSelectionCleanupAndTorus/CPU
# build/node-v5-all-vulkan.log
[==========] 162 tests from 8 test suites ran. (1895 ms total)
[  PASSED  ] 161 tests.
[  SKIPPED ] 1 test, listed below:
[  SKIPPED ] Backends/NodesCoreScale.MillionSplatsColourSelectionCleanupAndTorus/CPU
```

The only skip is explicitly GPU-only. Expected invalid-input tests log a
contract failure and pass. The visualizer filter
`NodeCanvasWidgets.*:Screen*.*:Trackpad*.*:PreferencesMigration.*:RmlStaticStyleBoundaries.*`
includes 16 canvas-widget tests (`build/node-v5-ui-regressions.log`):

```text
[==========] 97 tests from 9 test suites ran. (874 ms total)
[  PASSED  ] 97 tests.
```

Required Python command, with `LFS_TEST_BUILD_DIR=build` and the supplied venv,
`-m pytest tests/python/test_nodes.py tests/python/test_localization_contracts.py -q`
(`build/node-v5-python-tests.log`):

```text
tests/python/test_nodes.py .........                                     [ 25%]
tests/python/test_localization_contracts.py ..........................   [100%]

============================== 35 passed in 1.59s ==============================
```

Exact gate outputs (`build/node-v5-{neutrality,error-debt,hardcoded}.log`):

```text
Backend neutrality: 0 violation(s).
no new error-debt violations
No likely hardcoded UI strings found.
```

Changed C++/CUDA/Objective-C++ files pass clang-format; `git diff --check` is
clean. No new UI text bypasses localization. `build-examples/` was not touched.

## Files changed

```text
docs/docs/development/mcp/node-editor.md
docs/docs/development/node-graph/design.md
docs/docs/development/node-graph/python-nodes.md
docs/docs/development/node-graph/v5-validation.md
docs/plugins/examples/node_posterize.py
src/app/mcp_node_editor_tools.cpp
src/app/mcp_node_resources.cpp
src/app/mcp_node_tools.cpp
src/core/include/core/detail/gpu_backend_ops.hpp
src/core/include/core/detail/tensor_impl.hpp
src/core/include/core/tensor_spatial.hpp
src/core/nodes/builtin_cleanup.cpp
src/core/nodes/builtin_common.hpp
src/core/nodes/builtin_conversion.cpp
src/core/nodes/builtin_selection.cpp
src/core/nodes/builtin_splat.cpp
src/core/tensor/backend/cuda/cuda_ops_spatial.cpp
src/core/tensor/backend/cuda/kernels/tensor_spatial.cu
src/core/tensor/backend/cuda/kernels/tensor_spatial.hpp
src/core/tensor/backend/facade_entries.def
src/core/tensor/backend/metal/kernels.metal
src/core/tensor/backend/metal/metal_backend_ops.hpp
src/core/tensor/backend/metal/metal_ops.mm
src/core/tensor/backend/vulkan/shaders/radius_neighbors.slang
src/core/tensor/backend/vulkan/vk_backend_ops.hpp
src/core/tensor/backend/vulkan/vk_ops_spatial.cpp
src/core/tensor/internal/point_spatial.hpp
src/core/tensor/tensor_factory.cpp
src/core/tensor/tensor_spatial.cpp
src/core/tensor/tensor_unified_ops.cpp
src/python/lfs/py_nodes.cpp
src/python/lfs_plugins/node_posterize.py
src/python/stubs/lichtfeld/nodes.pyi
src/visualizer/gui/node_canvas_interaction.cpp
src/visualizer/gui/node_canvas_interaction.hpp
src/visualizer/gui/rmlui/elements/node_canvas_element.cpp
src/visualizer/gui/rmlui/elements/node_canvas_element.hpp
src/visualizer/gui/rmlui/elements/node_canvas_fields.cpp
src/visualizer/gui/rmlui/elements/node_canvas_widgets.cpp
src/visualizer/gui/rmlui/resources/node_editor.rcss
src/visualizer/gui/rmlui/resources/node_editor.theme.rcss
src/visualizer/include/visualizer/nodes/modifier_manager.hpp
src/visualizer/input/input_controller.hpp
src/visualizer/nodes/modifier_manager.cpp
tests/python/test_nodes.py
tests/test_mcp_node_tools.cpp
tests/test_node_canvas_interaction.cpp
tests/test_node_canvas_widgets.cpp
tests/test_nodes_core.cpp
```

Also updated the existing local
`docs/docs/development/node-graph/examples/ex20_floater_preview.py` description
and candidate-count check for the new preview semantics. These example scripts
are currently untracked and ignored by the repository's `*.py` rule; this local
edit therefore does not appear in the tracked diff. The other examples' API
calls remain compatible and their intent was left unchanged.

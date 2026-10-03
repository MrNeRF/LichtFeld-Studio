# Self-explaining nodes: V9 validation

## Scope and design

54 built-in node types now have a one-sentence description and 2–4 lines of
guidance. All 214 input, output and property declarations have descriptions.
The English catalogue is the source of truth; CMake embeds its node section
for headless descriptors. German, Spanish, French, Italian, Japanese, Korean,
Dutch, Polish and Chinese have translated catalogues with identical keys.

Registry presentation copies are cached by language generation. Evaluation
continues to use the original descriptors: changing language neither edits
graphs nor requests evaluation. Python plugins keep their own strings;
the bundled Posterize plugin uses the built-in catalogue.

The supplied Isolation Radius wording is retained verbatim. It is 139
characters, so its test explicitly exempts it from the otherwise enforced
120-character control-text limit. Node descriptions are at most 110 characters
and exclude the requested implementation terms.

Help appears in the Add preview (hover and keyboard search highlight), delayed
canvas tooltips, and the inspector. Inspector rows reveal muted helper text on
focus. How to use starts expanded and remembers its state per type within the
editor. Selection-producing nodes enable selection preview unless the user has
explicitly disabled it. Learn more opens the checked-out Markdown reference in
development builds, or its repository page in packaged builds, through the OS
URL/document opener.

`tools/generate_node_reference.py` generated 54 node pages and an index from
the application's English `lf.nodes.node_types()` export. Tables include stable
identifiers, defaults, ranges/steps, choices and descriptions. Example links
come from exact quoted type identifiers in the example sources. The existing
23 scripts and their shared helper were hidden by the blanket `*.py` ignore
rule; a scoped exception makes them available to clean checkouts, unchanged.

## Changed files

- Core: `registry.hpp/.cpp`, built-in registration files, new
  `builtin_text.cpp` / `builtin_text.hpp.in`, and core CMake embedding.
- Python: `py_nodes.cpp`, `nodes.pyi`, bundled/example Posterize declarations.
- MCP: node descriptor serialization, bounded `ui.pointer` hover injection,
  the existing injected-pointer polling seam, and MCP documentation.
- GUI: node canvas element/menu/status/widgets, new `node_canvas_help.cpp`,
  node editor RCSS/theme, and visualizer CMake.
- Localisation: all ten locale JSON files.
- Reference: generator, 55 generated Markdown pages, design/Python-node docs,
  `.gitignore` documentation-example exception.
- Tests: core metadata, Python metadata/locales/reference freshness, MCP
  descriptors/host-only reference, injected hover lifetime, and RmlUi help
  layout/search/focus/language/preview behaviour.

## Commands and output tails

With `SDKROOT=/Library/Developer/CommandLineTools/SDKs/MacOSX26.5.sdk`:

```text
ninja -C build lfs_core lichtfeld_tests LichtFeld-Studio lfs_py lichtfeld_visualizer_tests
exit 0
```

The linker emits existing macOS deployment-version warnings for the Python
static library (built for 26.6; application deployment target 26.0).

For both `--tensor-backend=metal` and `--tensor-backend=vulkan`:

```text
build/tests/lichtfeld_tests --tensor-backend=<backend> \
  --gtest_filter='*Nodes*:NodeCanvas*:McpNodeToolsTest.*:McpInjectedPointer.*:*RadiusNeighbor*:*radius_neighbor*'
[==========] 182 tests from 9 test suites ran.
[  PASSED  ] 179 tests.
[  SKIPPED ] 3 tests, listed below:
[  SKIPPED ] Backends/NodesCore.CpuMeshUploadsOnceAndJoinsGpuSplats/CPU
[  SKIPPED ] Backends/NodesCoreScale.MillionSplatsColourSelectionCleanupAndTorus/CPU
[  SKIPPED ] Backends/NodesCoreScale.SpatialOperationsStayInteractive/CPU
```

CPU parameterisations are included in these runs. The command-line backend
selector accepts only CUDA, Vulkan or Metal; `--tensor-backend=cpu` is not a
supported selector. CUDA was not run on this Mac.

```text
build/tests/lichtfeld_visualizer_tests
[==========] 191 tests from 23 test suites ran.
[  PASSED  ] 189 tests.
[  SKIPPED ] 2 tests, listed below:
[  SKIPPED ] ViewerGpuMemory.VulkanStatusDoesNotRequireCudaOrReportDeviceUsageAsProcessUsage
[  SKIPPED ] ViewerGpuMemory.CudaStatusKeepsDeviceUsageAndItsOptionalUtilizationReading
```

This includes all 25 `NodeCanvasWidgets` tests, including the previous
card-body/contrast, sidebar-width, retained-DOM and no-evaluation regressions.

```text
LFS_TEST_BUILD_DIR=build <scratchpad-venv>/bin/python -m pytest \
  tests/python/test_nodes.py tests/python/test_localization_contracts.py -q
39 passed

python3 tools/generate_node_reference.py --descriptors build/node-v9-descriptors.json --check
55 node reference pages checked

python3 tools/check_backend_neutrality.py
Backend neutrality: 0 violation(s).

python3 tools/error_debt_census.py --baseline tools/error_debt_baseline.json
no new error-debt violations

python3 tools/check_ui_hardcoded.py
No likely hardcoded UI strings found.
```

Logs: `build/node-v9-{build,metal,vulkan,visualizer,python,reference,neutrality,error-debt,hardcoded}.log`.
Changed/new C++ files were clang-formatted; `git diff --check` passes.

### Additional check, not green

The optional repository-wide `check_python_stubs` target reports pre-existing
generated/committed drift in `__init__.pyi`, `keymap.pyi`, `mesh.pyi`, and the
handwritten `nodes.pyi`. Examples include the frame-callback lifetime API,
`TOGGLE_NODE_EDITOR`, C++ template whitespace and handwritten versus generated
node class declarations. The handwritten node stub was updated with this
round's descriptions/help; it was not replaced with the less descriptive
generated class stub. No unrelated stub files were changed.

## Visual checks

Isolated garden.ply instance on port 45695, 2560×1440 capture at 2× density.
The graph has ten nodes and includes Colour Correct, HSV Range, Remove Floaters,
Scale Clamp, Recolour and multi-input Join Geometry. MCP discovery preceded
mutations; the graph was built with node tools. All captures below were opened
at original resolution and inspected:

- `/private/tmp/node-editor-visual/v9-add-preview.png`: category submenu and
  adjacent description/help card with category icon; no clipped help text.
- `/private/tmp/node-editor-visual/v9-socket-tooltip.png`: Geometry input label
  tooltip after the 500 ms delay, using `ui.pointer` hover.
- `/private/tmp/node-editor-visual/v9-inspector-how-to.png`: full description,
  expanded instructions, controls and Learn more in the dark theme.
- `/private/tmp/node-editor-visual/v9-inspector-focused-help.png`: Max Aspect
  helper text appears under the focused numeric row.
- `/private/tmp/node-editor-visual/v9-inspector-light.png`: the same inspector
  and readable node bodies in the light theme.
- `/private/tmp/node-editor-visual/v9-generated-reference.png`: generated
  Scale Clamp Markdown displayed, not executed, in the app's text editor.

Node labels, settings expanders and result footers remain visible. Help uses
the existing UI font and theme colours. Keyboard search/highlight and remembered
help expansion are additionally covered by synthetic RmlUi widget tests.

The MCP hover action retains the existing pointer override for a bounded
duration (default 1000 ms, maximum 10000 ms); it does not introduce another
input path. A subsequent pointer action or expiry returns normal input.

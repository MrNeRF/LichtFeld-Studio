# Node Editor V4 validation

## Scope and fixes

Validated on Apple M5 Max, macOS, 3024 × 1768 framebuffer at 2× UI density,
with the one-million-splat `garden.ply`. The renderer is Vulkan/MoltenVK;
tensor evaluation was exercised with Metal and Vulkan, and unit tests also
exercise CPU tensors. No native Linux/Windows Vulkan or CUDA device was available.

- Canvas DOM patching no longer happens during RmlUi's render traversal.
  Evaluation results update status, not the layout model. Retained cards,
  controls, links and live gesture positions survive submit → busy → install.
- RmlUi `drag` events now drive the same live interaction state as `mousemove`.
  Injected pointer events retain their last position across separate MCP calls,
  and native events cannot consume an injected event's state. This matters on
  macOS, where polled cursor state otherwise disagrees with queued SDL events.
- Reopening invalidates the retained view, resolves the current modifier and
  graph, and defers framing until the canvas has its actual dimensions.
  Deleting a graph removes its modifier instances in the same undo entry;
  Python replacement no longer leaves an invalid old modifier ahead of the new one.
- Running/queued states and header progress appear after 80 ms. Completed nodes
  retain their stats. The currently running node has an animated title accent.
- Arrange uses dependency depths, helper placement, six barycentre sweeps and
  actual card dimensions. It is available in the header, context menu and
  Shift+L; one undo reverses it. First-show overlapping graphs auto-arrange.
- Input edits use small value-delta undo entries, coalesced over a contiguous
  burst of the same input with a 500 ms idle boundary. They no longer snapshot
  and diff the complete graph for every Python call.
- Canvas wheel/pinch/swipe routing shares the viewport navigation helpers and
  preferences. Mouse wheel zooms; trackpad swipe pans; Ctrl/Cmd swipe and pinch
  zoom; middle drag pans. `ui_pointer` also injects pinch events.
- Cards show field/link sockets by default, with persistent inline-settings
  expanders. Header preview/sidebar controls use icon toggles. The Add popup is
  cursor-positioned, focused-search-first, cascading and height-limited.
- Each card owns its socket mesh. Selected/moved cards come forward, hiding
  sockets behind them. Python errors show exception type/message; full
  tracebacks go to WARN logs.
- Retained sidebar patching now excludes RmlUi's private non-DOM children.
  Previously, appending a section could replace the internal scrollbar element,
  producing a narrow or corrupt inspector after selection changes.
- Twenty-eight MCP tools, descriptor/graph/stack/editor resources, the
  `nodes.evaluate` job and evaluation events follow the existing MCP conventions.
  See [the MCP guide](../mcp/node-editor.md) for exact arguments and playbook.

## Gesture and edit measurements

An 11-node graph contains Colour Correct, HSV Range, Remove Floaters, Simplify,
Transform, Scale Clamp, Join Geometry, Recolour, Group Input/Output, and an
unconnected Inside Mesh. Ten nodes participate in evaluation. Simplify ratio
is 0.98. Results are approximately 894K splats, depending on the scrubbed radius.

The following are complete canvas CPU work, in milliseconds, **mean / p95 / max**
(nearest-rank p95). Drags use 50 moves with 16 ms between moves, with captures
and state reads before release; a separate 120-step convenience drag runs
across heavy evaluation and installation. Measurements include MCP observation
overhead and are not GPU timestamps.

| Interaction | Metal tensor backend | Vulkan tensor backend |
| --- | ---: | ---: |
| Node drag | 0.301 / 0.559 / 0.839 | 0.345 / 0.722 / 0.887 |
| Middle-drag pan | 0.369 / 0.722 / 1.040 | 0.460 / 0.805 / 0.946 |
| Box select | covered by regression test | 0.437 / 0.811 / 1.443 |
| Wire drag + connect | 0.226 / 0.438 / 1.403 | 0.296 / 0.525 / 1.396 |
| Re-route | 0.220 / 0.437 / 1.407 | 0.313 / 0.563 / 1.790 |
| Unmodified wheel zoom | 0.562 / 0.719 / 0.760 | 0.602 / 0.767 / 0.854 |
| Rapid value edits | 0.953 / 1.454 / 1.891 | 0.807 / 1.221 / 1.317 |

Pan, zoom, select, box select and node move issue zero evaluation requests.
Wire drags issue none until drop, followed by exactly one request. The final
Vulkan connect observation window, including result installation, peaked at
1.396 ms. An earlier V4 settling run had a 3.882 ms canvas outlier: these samples
do **not** establish a hard 2 ms upper bound for every frame or larger graphs.

The historical V3 baseline remains in `performance-validation.md`, with its
different measurement scope stated explicitly; no identical-scope speedup ratio
is inferred from it. Old captures that did not actually deliver a held pointer
are not used as evidence of live dragging.

Pure MCP input edits achieved **120.37 calls/s** (60 consecutive calls).
A real Python loop of 120 `set_input` calls achieved **116.63 calls/s** on the
heavy graph in the final run, exceeding the requested 60 calls/s. One contiguous
burst produces one undo entry, covered by a manager regression test.

In the rapid value benchmark, only Recolour and Group Output ran; HSV Range,
Colour Correct and cleanup nodes remained cached. Final request-to-install
latency was 17.29 ms with Metal and 36.06 ms with Vulkan. These deliberately
rapid 100-edit bursts superseded intermediate work and installed only the final
result; they are not evidence of 100 displayed updates. The worker remains one
in-flight evaluation plus one latest queued request.

## Vulkan trace and remaining viewport limit

Viewport timings force real rendering by alternating nearby camera positions
with `render_reconstruction_sample_frame`. They include viewer render/submit/
present wall time, but exclude idle event-loop sleep.

| Tensor backend/run | Idle mean / p95 / max ms | Busy mean / p95 / max ms | Request → install |
| --- | ---: | ---: | ---: |
| Metal | 8.278 / 8.519 / 8.747 (94 frames) | 8.250 / 8.458 / 8.640 (77) | 650 ms |
| Vulkan before retirement fix | 8.282 / 8.475 / 8.562 (94) | 9.808 / 17.962 / 44.513 (183) | 1826 ms |
| Vulkan after retirement fix | 8.293 / 8.382 / 8.415 (94) | 9.855 / 18.231 / 42.420 (175) | 1743 ms |

Four-second, 1 ms sampling traces plus existing performance logs identify:

1. Before the fix, `ModifierManager::installReady` could release an old tensor,
   enter the Vulkan allocator's retired-storage collector and wait in
   `vkQueueWaitIdle` on the viewer thread. Old published results, preview storage
   and stale ready results now retire on the evaluation worker, outside its
   mutex. The after trace no longer contains this install-time wait. A regression
   test checks that the old payload deleter runs off the viewer thread.
2. The dominant remaining viewer waits are swapchain image fences and drawable
   acquisition. A logged `RasterizeForward` GPU interval reached 43.06 ms alongside
   a 43.45 ms image-fence wait. That is actual render/GPU queue contention under
   MoltenVK, not synchronous node evaluation in a pointer callback.
3. The existing renderer's scratch-buffer retirement can still enter the Vulkan
   allocator's consumer-queue-idle safety path. MoltenVK's shared allocation
   residency rules require that protection before freeing storage; simply
   bypassing it would be unsafe. This renderer/allocator path is not eliminated
   by the node-result retirement fix. Native Vulkan hardware would be needed to
   distinguish the remaining driver-specific behaviour from general GPU pressure.

Thus the viewer-thread node-result release bug is fixed, but **Vulkan viewport
tail latency remains a known limit**; the 42 ms maximum must not be described as
meeting an 8 ms viewport budget. Metal tensor evaluation on this machine does
not show the corresponding busy-frame tail. No device-wide synchronization was
added to the viewer's node-evaluation path. Work still uses the public tensor
queue/completion APIs, and only completed results install on the viewer thread.

Trace artifacts (local, not committed):

- `build/node-editor-v4-vulkan-sample.txt`
- `build/node-editor-v4-vulkan-after-sample.txt`
- `build/node-editor-v4-vulkan-runtime-perf.log`
- `build/node-editor-v4-vulkan-after-runtime.log`
- `build/node-editor-v4-{metal,vulkan,vulkan-after}-performance.json`
- `build/node-editor-v4-metal-gestures.json`, `build/node-editor-v4-gestures.json`

## Visual and end-to-end verification

Captures were decoded and inspected against the adjacent Properties panel at
2× density. All paths below are under `/private/tmp/node-editor-visual/`:

- `v4-mid-processing-drag.png`, `v4-mid-processing-drag-later.png`,
  `v4-after-processing-drag.png`: the same 11-node graph during a 120-step node
  drag, across running → installed; retained card identity/count and uncommitted
  location checked before release. Running/queued title/footer states are visible.
- `v4-mid-heavy-node-drag.png`, `v4-mid-heavy-pan.png`,
  `v4-mid-heavy-box-select.png`, `v4-mid-heavy-wire-connect.png`,
  `v4-mid-heavy-reroute.png`: actual held-pointer states, not post-release captures.
- `v4-wire-cut.png`, `v4-overlap.png`, `v4-compact-inspector.png`: knife removal
  and undo, front-card socket occlusion, compact cards with full inspector.
- `v4-add-popup.png`, `v4-add-search.png`: cascading categories and focused,
  typed HSV filtering.
- `v4-python-replaced.png`, `v4-reopened.png`: Python graph replacement and
  close/open retain the correct six-node graph; Frame works without a manual
  editor rebind. At fit-all zoom the title-only LOD is intentional.
- `v4-python-error-muted.png`: muted card and concise Python exception inspector.
- `v4-mcp-six-node.png`, `v4-mcp-scrubbed.png`, `v4-mcp-applied.png`,
  `v4-mcp-undone.png`: a six-node graph built, arranged, scrubbed, evaluated,
  applied and undone **using MCP tools only, without `editor_run`**.

`ui_pointer` exercised mouse wheel, explicit trackpad swipe, Cmd+swipe and pinch.
Automatic device classification and preference speed scaling have unit coverage.
No physical trackpad was available; synthetic pinch exercises the SDL event and
normal per-frame panel-input path, not a parallel input channel.

## Tests and gates

All modified/new C++ files were clang-formatted. Existing staged changes were
preserved; this round was not committed. `build-examples/` was left alone.

```text
LichtFeld-Studio, lfs_py, lichtfeld_tests, lichtfeld_visualizer_tests: built
Required Nodes* filter (Metal): 17 passed
Broad node/canvas/MCP suite (Metal): 142 passed, 1 skipped
Broad node/canvas/MCP suite (Vulkan): 142 passed, 1 skipped
CPU node parameterization: 33 passed, 1 skipped
Visualizer canvas/screen/keymap/trackpad suite: 78 passed
Python nodes/localization/keymap: 41 passed
Backend neutrality: 0 violation(s).
no new error-debt violations
No likely hardcoded UI strings found.
git diff --check: clean
```

The single skip is the explicitly GPU-only million-splat scale test on CPU.
The broader suite includes every new MCP tool and resource on a splat scene,
metadata/errors, shared undo, worker jobs, apply/capture, editor reopen,
retained DOM, burst coalescing, stale-result rejection and viewer-thread install.
The headless widget fixture reports missing sprite warnings because it does not
load the application's sprite sheet; the real-app captures render those icons.
Existing Python-library SDK-version linker warnings remain.

Full output tails/logs are in `build/node-editor-v4-{nodes-required,all-tests,
vulkan-tests,cpu-tests,widgets,pytest}.log`, `build/node-editor-v4-final-build.log`
and `build/node-editor-v4-{backend,error,ui}-gate.log`.

Native file-dialog Open/Save and physical trackpad feel were not automated.
The editor currently supports one selected link at a time; the MCP schema
documents that limit. Very large textured meshes and 50+ node V4 layouts were
not benchmarked in this follow-up.

## Files changed in this round

This inventory is relative to the pre-existing index (plus new source/docs files),
not a list of all files staged by earlier rounds.

```text
AGENTS.md
CMakeLists.txt
docs/docs/development/mcp/index.md
docs/docs/development/mcp/node-editor.md
docs/docs/development/node-graph/performance-validation.md
docs/docs/development/node-graph/v4-validation.md
src/app/include/app/mcp_event_handlers.hpp
src/app/include/app/mcp_node_tools.hpp
src/app/mcp_gui_tools.cpp
src/app/mcp_node_editor_tools.cpp
src/app/mcp_node_resources.cpp
src/app/mcp_node_tools.cpp
src/app/mcp_node_utils.hpp
src/app/mcp_pointer_tool.cpp
src/app/mcp_runtime_tools.cpp
src/core/include/core/nodes/evaluator.hpp
src/core/include/core/nodes/events.hpp
src/core/include/core/nodes/tree.hpp
src/core/nodes/evaluator.cpp
src/core/nodes/tree.cpp
src/python/lfs/py_nodes.cpp
src/visualizer/CMakeLists.txt
src/visualizer/gui/area_editors.cpp
src/visualizer/gui/area_editors.hpp
src/visualizer/gui/gui_input.hpp
src/visualizer/gui/gui_manager.cpp
src/visualizer/gui/node_canvas_interaction.cpp
src/visualizer/gui/node_canvas_interaction.hpp
src/visualizer/gui/node_canvas_layout.cpp
src/visualizer/gui/panel_input_utils.hpp
src/visualizer/gui/panel_registry.cpp
src/visualizer/gui/resources/locales/de.json
src/visualizer/gui/resources/locales/en.json
src/visualizer/gui/resources/locales/es.json
src/visualizer/gui/resources/locales/fr.json
src/visualizer/gui/resources/locales/it.json
src/visualizer/gui/resources/locales/ja.json
src/visualizer/gui/resources/locales/ko.json
src/visualizer/gui/resources/locales/nl.json
src/visualizer/gui/resources/locales/pl.json
src/visualizer/gui/resources/locales/zh.json
src/visualizer/gui/rmlui/elements/node_canvas_dom.cpp
src/visualizer/gui/rmlui/elements/node_canvas_element.cpp
src/visualizer/gui/rmlui/elements/node_canvas_element.hpp
src/visualizer/gui/rmlui/elements/node_canvas_menu.cpp
src/visualizer/gui/rmlui/elements/node_canvas_status.cpp
src/visualizer/gui/rmlui/elements/node_canvas_widgets.cpp
src/visualizer/gui/rmlui/elements/node_canvas_widgets.hpp
src/visualizer/gui/rmlui/resources/node_editor.rcss
src/visualizer/gui/rmlui/resources/node_editor.theme.rcss
src/visualizer/gui/rmlui/rml_panel_host.cpp
src/visualizer/gui/rmlui/rmlui_manager.cpp
src/visualizer/gui/rmlui/rmlui_manager.hpp
src/visualizer/gui/screen_host.cpp
src/visualizer/gui/screen_host.hpp
src/visualizer/include/visualizer/nodes/modifier_manager.hpp
src/visualizer/input/frame_input_buffer.hpp
src/visualizer/input/injected_pointer.cpp
src/visualizer/input/injected_pointer.hpp
src/visualizer/input/input_controller.cpp
src/visualizer/input/input_controller.hpp
src/visualizer/input/navigation_gestures.hpp
src/visualizer/internal/viewport.hpp
src/visualizer/nodes/modifier_evaluation_worker.cpp
src/visualizer/nodes/modifier_evaluation_worker.hpp
src/visualizer/nodes/modifier_manager.cpp
src/visualizer/nodes/modifier_manager_evaluation.cpp
src/visualizer/window/window_manager.cpp
tests/CMakeLists.txt
tests/python/test_nodes.py
tests/test_mcp_node_tools.cpp
tests/test_mcp_screen_tools.cpp
tests/test_node_canvas_interaction.cpp
tests/test_node_canvas_widgets.cpp
tests/test_nodes_modifier_manager.cpp
```

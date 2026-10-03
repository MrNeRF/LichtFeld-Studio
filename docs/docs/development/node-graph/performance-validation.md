# Node Editor performance validation

The current follow-up results, regression coverage and trace findings are in
[V4 validation](v4-validation.md). The measurements below are the historical
V3 baseline, not a claim about the current pointer injection or overlapping-card
behaviour.

Validated on Apple M5 Max, macOS, 2560 × 1440 framebuffer, 2× UI density.
The viewport renderer is Vulkan. Tensor evaluation was tested separately with
Metal and Vulkan (MoltenVK on this machine), using `garden.ply` with 1,000,000
stored splats. No CUDA device or native Linux/Windows Vulkan device was available.

## Reproduction

Launch the isolated MCP instance on port 45695, initialize MCP, and dismiss any
startup recovery modal before measuring. Use `editor_run` with
`show_console: false`. Create this graph:

```text
Group Input → Colour Correct → Remove Floaters → Simplify → Transform Geometry
            → Scale Clamp → Join Geometry → Recolour → Group Output
HSV Range.Selection → Colour Correct.Selection
```

There are 11 nodes, including an unconnected Inside Mesh node; 10 participate
in the evaluated graph. Remove Floaters uses its defaults, including relative
isolation radius 3.0; Simplify ratio is 0.98. The result has 894,237 splats.
The wire gesture connects HSV Range.Selection to Recolour.Selection.

Use `lf.nodes.performance(reset=True)` immediately before a gesture and read it
afterwards. Node/wire/pan drags use `ui.pointer`, 100 interpolated steps. Zoom
uses Ctrl+wheel around the cursor. Scrubbing changes Recolour.Weight at zoom 1.
For viewport samples, repeatedly call `render_reconstruction_sample_frame`
with alternating nearby camera positions, first idle and then after changing
Remove Floaters.Isolation Radius to 3.01. Do not call synchronous
`lf.nodes.evaluate` inside measurement windows.

## Canvas measurements

Milliseconds, mean / p95 / maximum. The baseline captured input/layout only;
the after measurements additionally include deferred RmlUi rendering. They
are deliberately **not** presented as identical-scope speedup ratios.

| Interaction | Before, Metal, input/layout only | After, Metal, complete canvas CPU work | After, Vulkan, complete canvas CPU work |
| --- | ---: | ---: | ---: |
| Node drag | 0.333 / 0.273 / 21.283 | 0.119 / 0.479 / 0.727 | 0.125 / 0.500 / 0.817 |
| Pan | 0.120 / 0.231 / 0.732 | 0.126 / 0.494 / 0.963 | 0.141 / 0.491 / 0.783 |
| Zoom tick | 0.254 / 0.118 / 1.534 | 0.408 / 0.194 / 1.369 | 0.374 / 0.199 / 1.233 |
| Wire drag + connect | 0.117 / 0.225 / 3.072 | 0.230 / 0.436 / 2.096 | 0.236 / 0.455 / 2.321 |
| Value scrub | 0.397 / 1.045 / 3.219 | 0.658 / 1.101 / 1.449 | 0.363 / 0.624 / 1.053 |

Before: node drag triggered 3 evaluations, zoom 1, connect 2, scrub 2.
After: node drag, pan and zoom triggered **zero requests and zero evaluations**.
Wire dragging triggered none until release, followed by exactly one request.
Selection and box selection have the same zero-evaluation contract in C++ tests.

The strict 2.0 ms worst-frame goal is not fully achieved: connect/drop peaks
were 2.10 ms on Metal and 2.32 ms on Vulkan. Steady gesture p95s were below
0.5 ms. A separate 50-node Metal drag (39 additional unconnected Recolour
nodes) measured 0.461 / 2.421 / 3.330 ms, with zero evaluations.

Tooltips are suppressed during button-held drags: opening one mid-drag had
previously forced an extra layout pass. Undo byte estimates are cached.
Connected socket widgets retain their element identity and are hidden instead
of being destroyed.

## Viewport during expensive evaluation

These are viewer CPU/render-submission/present wall times, excluding idle
event-loop sleep, **not GPU timestamps**.

| Tensor backend | Idle frames; mean / p95 / max ms | Busy frames; mean / p95 / max ms | Request → installed result |
| --- | ---: | ---: | ---: |
| Metal | 122; 8.405 / 9.164 / 19.793 | 80; 8.282 / 9.171 / 9.558 | 674 ms |
| Vulkan / MoltenVK | 125; 8.532 / 9.063 / 43.894 | 141; 9.067 / 12.903 / 30.405 | 1,295 ms |

The Vulkan run still has viewport tail-latency spikes, including while idle.
Moving evaluation off the viewer thread is not a guarantee against GPU or
presentation contention. The original baseline did not include comparable
viewport-frame instrumentation.

The relative-radius isolation optimization queries only the current size
level while retaining all points as potential neighbours. It checks the
central spatial-hash cell first and excludes the same index without excluding
distinct coincident points. Bounded GPU dispatches and request-specific queue
markers prevent a single large isolation dispatch from monopolizing the
timeline; no device-wide viewer-thread synchronization is added.

## Live scrub and cache reuse

Metal: 76 requests, 75 evaluations, 70 installed updates during the gesture;
request-to-install mean 15.65 ms, p95 18.92 ms, maximum 21.04 ms.

Vulkan: 76 requests, 76 evaluations, 14 installed updates; mean 42.50 ms,
p95 46.56 ms, maximum 48.91 ms. Sixty-one superseded evaluations were discarded.
One worker plus one latest queued request bounds queued work, but a slower
backend necessarily installs fewer intermediate updates.

Only Recolour and Group Output ran while scrubbing. HSV Range, Colour Correct,
Remove Floaters, Simplify and other upstream nodes stayed cached. Latency
measures request-to-viewer installation, not physical display scanout.

## Visual checks and artifacts

PNG captures were decoded and inspected at 2× density, including comparison
with the adjacent Properties panel. Final screenshots are under
`/private/tmp/node-editor-visual/`:

- `v3-initial.png`: six-node graph, muted Colour Correct, failing Python node,
  Join Geometry with two inputs, full error inspector.
- `v3-zoom-0.5.png`, `v3-zoom-1.5.png`: readable title LOD and zoomed controls.
- `v3-light.png`: light-theme cards, dots, controls and error treatment.
- `v3-evaluating.png`: heavy graph with updating header and current-node status.
- `v3-footer-stats.png`, `v3-header-status.png`: counts, selected share, cached
  flags, inspector and result header; the latter also shows selection preview.
- `v3-add-popup.png`, `v3-added-node.png`: category popup and pointer-added node.
- `v3-connected.png`, `v3-rerouted.png`, `v3-cut.png`, `v3-moved.png`:
  actual injected pointer gestures.

Raw summary artifacts are `build/node-editor-v3-benchmarks.json` (baseline)
and `build/node-editor-v3-final-benchmarks.json` (after).
The isolated instance was stopped, and its backend preference restored to auto.

## Tests and gates

- Required `Nodes*` filter: 14 passed.
- Broader `*Nodes*:*NodeCanvas*` filter: 125 passed; one intentional CPU
  million-splat scale-test skip. CPU, Metal and Vulkan parameterizations ran.
- Visualizer node-canvas/screen/keymap tests: 58 passed.
- Python nodes/localization/keymap tests: 41 passed.
- Backend neutrality: 0 violations.
- Error-debt census: no new violations.
- Hardcoded UI audit: no candidates.
- Build targets: LichtFeld-Studio, lfs_py, lichtfeld_tests and
  lichtfeld_visualizer_tests succeeded. Existing Python-library SDK-version
  linker warnings remain.

C++ coverage includes retained cards/controls, view-only gestures, one request
on wire drop/value edit, upstream caching, stale-result rejection, viewer-thread
installation, queued mesh-metadata snapshots and worker/published-storage isolation. Native file-dialog Open
and Save flows were not automated. Real hardware trackpad gestures were not
available for physical testing; wheel/pinch routing was reviewed, and wheel
and pointer paths were exercised through MCP.

Other limits: mesh captures copy CPU material/texture metadata; very large
textured meshes were not benchmarked. Socket overlays remain above card content
when cards overlap. Per-node times measure callback work (including blocking
worker readbacks), not independent GPU timestamps for lazy field construction.

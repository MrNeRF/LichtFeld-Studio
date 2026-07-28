---
sidebar_position: 5
---

# RmlUI Context Model

LichtFeld Studio currently uses more than one RmlUI context. The normal panel
path is one `RmlPanelHost` context per panel, rendered into a panel-sized
texture and then composited into the window. The fixed UI also has independent
contexts for the shell frame, menu bar, status bar, right panel, toast overlay,
modal overlay, startup overlay, viewport overlay, sequencer, sequencer overlay,
and global context menu.

This document records the current implementation and the investigation needed
to simplify it. The desired direction is to let RmlUI own as much of stacking,
hit testing, focus, popup, and tooltip behavior as it can. The current
multi-context model is a performance-oriented workaround, not proof that a
global or shared RmlUI context is inherently too expensive.

A panel that did not change can reuse its cached texture instead of recording
its RmlUI document again. The available `performance.log` baseline for this
model reports cached right-panel layout at roughly 0.08--0.11 ms, RmlUI
recording at roughly 0.08--0.40 ms, and CPU UI work before Vulkan frame
recording at roughly 0.6--0.9 ms. These measurements establish a baseline to
preserve or improve; they do not decide the architecture.

## Context inventory

The fixed contexts are intentionally named at creation time. They do not share
one document because they have different ownership, draw geometry, and input
rules.

| Context | Owner | Role |
| --- | --- | --- |
| `shell_frame` | `RmlShellFrame` | window chrome and shell layout |
| `menu_bar` | `RmlMenuBar` | application menus |
| `status_bar` | `RmlStatusBar` | bottom status and progress |
| `right_panel` | `RmlRightPanel` | fixed right-side layout |
| `toast_overlay` | `RmlToastOverlay` | non-modal notifications |
| `modal_overlay` | `RmlModalOverlay` | modal dialogs and their underlay |
| `startup_overlay` | `StartupOverlay` | startup and import transition |
| `viewport_overlay` | `RmlViewportOverlay` | viewport controls and HUD state |
| `sequencer` | `RmlSequencerPanel` | timeline panel |
| `sequencer_overlay` | `RmlSequencerOverlay` | timeline overlay content |
| `global_context_menu` | `GlobalContextMenu` | window-space context menu |

Every dynamic `RmlPanelHost` adds one more context. Its document is laid out
in panel-local coordinates, rendered to a panel-sized texture, and composited
at the panel's window-space rectangle. This is why a local RmlUI popup cannot
be assumed to participate in window-wide hit testing. A normal session with at
least nine hosted panels has twenty or more live RmlUI contexts. The exact
total depends on which dynamic panels and plugins are alive in that session.

### Dynamic context producers

Dynamic contexts do not have one closed runtime list because Python panels and
plugins can create them. Their producers and naming rules are nevertheless
enumerable:

| Producer | Context name |
| --- | --- |
| `NativeScenePanel` | `scene_panel_native` |
| Python console panel | `python_console_panel` |
| Video extractor dialog | `video_extractor` |
| `RmlImModePanelAdapter` | `im_mode_<counter>` |
| `RmlPythonPanelAdapter` | caller-provided `context_name_` |

The last row is the extension point: its names and count depend on installed
or loaded Python panels. All of these paths create the same `RmlPanelHost` and
therefore have the same local rendering, input, tooltip, and popup behavior.

## Consequences

RmlUI only performs hit testing, focus management, and stacking inside one
context. A panel-local dropdown can extend beyond the panel texture, but its
context cannot see another panel or a fixed overlay. The application therefore
owns the cross-context policy:

- `GuiManager::hitTestPointer()` defines the top-level priority order for
  modals, menus, export locks, overlays, floating panels, sequencer, viewport
  overlays, and resize edges.
- `PanelRegistry` owns floating-panel stack order through
  `float_stack_order`, `alloc_float_stack_order_locked()`, and
  `bring_floating_panel_to_front_locked()`.
- `RmlPanelHost` maps window coordinates into its local context and handles an
  open select box that extends outside the panel shape. It also forwards pointer,
  text, and keyboard input to its context, maintains pointer capture, and owns
  its document-local tooltip controller. The open-select path is implemented by
  `openDropdownBounds()`, `openDropdownContainsPoint()`,
  `openDropdownOptionAtPoint()`, and `setManualDropdownHover()`.
- `RmlUIManager` tracks the position and draw order of contexts for active
  overlays, cursor updates, occlusion checks, and pending tooltip wake-ups.

This duplication is the trade-off of the cache boundary, not an indication that
one of these responsibilities was accidentally omitted. In particular, a
tooltip is currently a `frame-tooltip` element appended to the document that
owns the hovered element; it cannot be rendered by another context without an
explicit handoff of its text and window-space anchor.

The current top-level pointer order is, from highest to lowest: modal/startup
or global context menu; menu bar; viewport export lock; active RmlUI overlay;
floating panel or panel resize; sequencer or viewport overlay; then the right
panel resize edge. `GuiManager::hitTestPointer()` is the source of truth for
that order. A component must not infer its precedence from render order alone.

New UI surfaces must fit this policy deliberately. A surface that blocks input
needs an explicit position in the top-level priority order. A floating surface
needs an owner for stack order, coordinate conversion, pointer capture, and
keyboard focus. Do not add a second ad-hoc routing path beside these owners.

## Core contributor guide

Read the code in this order when changing the RmlUI architecture:

1. `RmlUIManager` creates, destroys, tracks, queues, and caches contexts.
2. `RmlPanelHost` owns the per-panel document, local coordinate conversion,
   input forwarding, tooltip state, and panel texture cache.
3. `PanelRegistry` owns panel registration and floating-panel interaction and
   stack order.
4. `GuiManager::hitTestPointer()` assigns cross-surface pointer precedence.
5. The relevant fixed-surface owner supplies its document layout and any
   surface-specific input behavior.

Use the existing owner instead of adding a parallel path:

| Change | Owner to modify | Invariant |
| --- | --- | --- |
| New fixed RmlUI surface | its component and `GuiManager` if it blocks input | define its pointer precedence explicitly |
| New ordinary or plugin panel | `PanelRegistry` / `RmlPanelHost` integration | preserve the panel-local cache boundary |
| Floating-panel interaction | `PanelRegistry` | use the existing stack-order and capture state |
| Dropdown or tooltip behavior | `RmlPanelHost` and `RmlUIManager` | keep coordinates, capture, focus, and occlusion in one owner |
| Context render/cache behavior | `RmlUIManager` queueing and cache code | do not refresh unchanged panel textures unnecessarily |

Before merging a cross-context change, exercise the precedence order with a
modal, menu, floating panel, viewport overlay, and an open dropdown. A change
that affects rendering or cache invalidation must also compare the existing
`gui_render.panel_layout.*`, `gui_render.rmlui_record`, and
`gui_render.cpu_ui_before_vulkan_begin` measurements.

## Panel rendering orchestration

`PanelRegistry` is the owner of panel selection and floating-panel interaction;
the layout code decides where a selected panel is placed. This separation is
important because the registry must serve the same panel through several
rendering paths without giving the panel a second lifecycle.

There are three semantic selection targets:

- all eligible top-level panels in one `PanelSpace`;
- one panel identified by id; and
- the direct children of one panel.

Each target can use the ordinary panel draw path or a direct path. The direct
path has three phases with distinct contracts:

| Phase | Purpose | Required behavior |
| --- | --- | --- |
| Preload | prepare content before a later direct draw | must not emit the panel's visible draw |
| Direct draw | render a panel at layout-provided bounds | must report or preserve the consumed height |
| Cached direct draw | reuse a previously recorded panel texture | a cache hit must avoid a full redraw; a miss must fall back to direct draw |

The current public registry API represents the cross-product of those targets
and phases with separate methods. That naming is an implementation detail, not
a contract for new panels or plugins. A future consolidation must preserve the
three target scopes, the phase ordering, and the cached-draw fallback before
removing individual entry points.

Direct rendering may also require input, a vertical input clip, panel space,
and a layout-forced height. Treat these as properties of one render invocation.
They must not leak into the next invocation of the same panel, including when
the panel changes layout location or its cached texture is reused.

The pre-scene path is especially sensitive: `GuiManager` can preload an active
tab and its child panels before scene rendering, while `PanelLayoutManager`
performs the later placement and direct draw. Do not reorder or merge those
calls merely because they address the same panel; doing so can move layout work
onto a latency-sensitive point in the frame.

### Floating-panel state

Floating interaction is not generic panel registration metadata. Its stable
responsibilities are position, auto-centering, stack order, drag and resize
state, user-selected height, and the last window-space bounds used for hit
testing. Only a panel in `PanelSpace::FLOATING` needs this state.

When changing registry storage, preserve these lifecycle rules:

- replacing a registered floating panel with the same id retains its placement
  and stack order where the current registry does so;
- entering or leaving the floating space creates or discards its interaction
  state deliberately;
- UI-scale changes and explicit size resets cancel transient drag/resize state;
  and
- stack promotion and pointer hit testing read the same last-known bounds.

This keeps registration data (identity, label, parent, space, ordering and
options) independent from live mouse interaction, while retaining the present
input and z-order semantics.

## Regression coverage

The context model is covered by focused tests at more than one level. C++ tests
exercise executable registry, layout, and input behavior. Python regressions
inspect resource and integration contracts that are difficult to construct in a
headless RmlUI session. Neither group is a substitute for the manual
cross-context checks described below.

| Area | Tests | What they protect |
| --- | --- | --- |
| Panel selection and direct rendering | `test_panel_registry_render_paths.cpp` | space, single-panel, and child selection; preload; cache hit and miss fallback; invocation-local input and clip state; consumed height |
| Panel visibility and floating stack | `test_panel_registry_animation_demand.cpp` | visible animation demand by panel space; floating-panel stack promotion; disabled floating-panel behavior |
| Right-panel layout policy | `test_panel_layout_render_demand.cpp` | scene-header and active-tab live/cached combinations, including the required preload before a live direct draw |
| Shared-context framework contract | `test_rml_shared_context_contract.cpp` | two documents in one real RmlUI context; native ordering, same-bounds click delivery, window-space document bounds and hit testing, and unloading one document while another remains live |
| RmlUI text and key translation | `test_rml_text_input_handler.cpp`, `test_sdl_rml_key_mapping.cpp` | text selection shortcuts, SDL-to-RmlUI keys and modifiers, and frame input filtering by window |
| RmlUI document/resource boundaries | `test_rml_path_utils.cpp`, `test_rml_static_style_boundaries.cpp`, `tests/python/test_rmlui_image_sources.py` | file/image source encoding and rewriting, stylesheet ownership, dirty-update policies, and update requests from dynamic image content |
| Fixed-surface routing and tooltips | `tests/python/test_menubar_resources.py` | menu popup layering and retained submenu bounds; menu pointer/keyboard blocking; tooltip wake-up demand; viewport-overlay positioning |
| Plugin panel contract | `tests/python/test_plugin_api_surface.py`, `tests/python/test_plugin_docs.py`, `tests/python/test_layout_composition.py` | public panel and layout API surface, typed floating panel spaces, documented plugin lifecycle, and nested layout behavior |

The Python resource tests are source-level regressions: they confirm that the
required RML, RCSS, and integration calls remain present, but do not execute a
live multi-context window. Treat their failures as a contract change requiring
review, and do not interpret their success as proof of end-to-end pointer or
rendering behavior.

### Choosing a test target

For panel-registry, layout, and context-model changes, enable
`BUILD_GUI_TESTS`. It builds `lichtfeld_gui_tests`, which contains the focused
PanelRegistry and right-panel layout tests and does not require LibTorch.
`BUILD_TESTS` remains the full historical suite: many tensor tests use LibTorch
as a correctness oracle, so that option intentionally requires the external
Torch SDK. Use the lightweight target for UI work unless the change also
affects the tensor implementation.

### What remains manual

No automated test currently drives a full window containing modal, menu,
floating panel, viewport overlay, and panel-local dropdown simultaneously.
Before merging a change to cross-context routing, cache invalidation, or popup
behavior, run that interaction matrix manually and compare the documented
`gui_render` timings. This is also the required validation for an input-order
change that cannot be represented by the focused registry tests.

## Architectural investigation

The current model should not be treated as the target architecture. Its
panel-sized caches solve a measured problem, but they also require the
application to reproduce behavior that RmlUI normally provides inside one
context. The investigation is therefore: can a correct use of one global
context, or a smaller number of shared contexts, retain the useful cache and
frame-time properties while deleting or materially reducing that application
code?

The repository history provides two important facts, and one important gap:

| Evidence | What it establishes | What it does not establish |
| --- | --- | --- |
| The first RmlUI implementation in commit `2a79f343` already created separate fixed and panel contexts. | Multi-context ownership predates the current cache work. | It does not contain a directly comparable global-context implementation. |
| Commit `e281a8e9` added cached direct draws, passive-mouse-move suppression, per-context frame tracking, and per-context `rmlui_record` timers. | The present performance work is deliberately granular and can be measured per context. | It does not prove that the same work could not be expressed efficiently with shared RmlUI documents. |
| The current tree contains the cache and routing implementation described above. | It identifies the code that a successful simplification should remove or minimize. | It does not explain a historical global-context slowdown. |

Do not assign a cause to a previous global-context performance problem until a
reproducible revision, workload, and trace are available. Possible causes to
test include invalidating too much of one document, rendering all documents
every frame, update scheduling caused by hover or animation state, excessive
full-window texture work, and backend cache behavior. They are hypotheses, not
findings.

### Code to minimize

Success is not measured by reducing the raw number of contexts alone. It is
measured by allowing RmlUI to own behavior that currently crosses an
application-defined context boundary. The primary candidates are:

| Current code | RmlUI behavior being reconstructed | Question for a shared model |
| --- | --- | --- |
| `GuiManager::hitTestPointer()` | window-wide surface precedence | Can normal document ordering and native RmlUI hit testing replace part of this chain? |
| `PanelRegistry` floating stack order and capture state | window stacking and pointer targeting | Can ordinary document z-order replace it for RML-backed floating surfaces? |
| `RmlPanelHost` dropdown bounds, manual hover, coordinate conversion, and forwarding | popup placement and input routing | Can a native popup remain in the same context as its owner without an application handoff? |
| `RmlPanelHost` tooltip and focus forwarding | hover, focus, capture, and keyboard ownership | Can document-local RmlUI behavior remove the per-host coordination? |
| `RmlUIManager` tracked frames, overlay occlusion, cursor, and tooltip wake-ups | cross-context ordering and invalidation | Which parts disappear when the relevant surfaces share a context, and which are still required by application policy? |

Some policy will remain application-owned even in the best outcome. For
example, a modal that deliberately blocks viewport tools still needs an
application-level decision. The target is to remove duplicated UI mechanics,
not to make RmlUI responsible for unrelated editor policy.

### Experiments and decision criteria

Work in small, reversible experiments. Do not migrate all UI surfaces at once.
Each experiment must keep the current path available for an A/B comparison and
must use the same scene, window size, visible panels, idle time, and scripted
interaction sequence.

1. **Baseline the current path.** Capture context count, per-context
   `gui_render.rmlui_record.*`, `gui_render.panel_layout.*`,
   `gui_render.cpu_ui_before_vulkan_begin`, frame time, and GPU UI cost for an
   idle session and an interaction-heavy session.
2. **Trace invalidation.** Identify which context or document causes `Update()`
   and `Render()` in each workload, and whether hover, tooltip, animation,
   resize, theme, or model updates caused it.
3. **Create one bounded shared-context prototype.** Start with one family of
   ordinary RML-backed panels or one popup path. Do not begin with modal,
   startup, or viewport-tool ownership. Preserve its behavior through the
   existing path so results are comparable and reversible.
4. **Compare native behavior and cost.** Verify z-order, hit testing, wheel,
   capture, focus, keyboard dismissal, and tooltip/popup behavior. Compare the
   metrics in step 1, including GPU cost; lower C++ routing complexity is not a
   win if it regresses frame time or invalidates the whole UI on ordinary edits.
5. **Choose the next scope from evidence.** Expand the shared model only if it
   removes a meaningful custom path without a material regression. Otherwise
   retain the cache boundary and record the measured reason.

### Existing instrumentation and first prototype

The existing RmlUI timing is CPU timing. `RmlUIManager` emits a per-context
`gui_render.rmlui_record.*` timer for a direct render, a `.cache_refresh` timer
when it records a layer into a texture, and a `.cached_context.*` timer when it
only blits that texture. `PanelLayoutManager` and `GuiManager` add the
`gui_render.panel_layout.*` and `gui_render.cpu_ui_before_vulkan_begin`
measurements. This is enough to locate CPU invalidation and distinguish cache
refresh from cache reuse.

The RmlUI Vulkan path does not currently expose equivalent GPU timestamps. A
GPU comparison therefore needs an external Vulkan/GPU profiler, or a separate
and validated timestamp-query addition. Do not infer GPU cost from the CPU
timers alone.

The project pins RmlUI 6.2. That version explicitly supports on-demand
rendering: after an update, `GetNextUpdateDelay()` reports zero for an
immediate frame, a finite delay for a scheduled internal update, or infinity
when no redraw is needed. The current integration uses that API in panel,
status-bar, and viewport-overlay owners only to detect the zero-delay case and
set an `animation_active_` flag. The application wait loop separately tracks
tooltip reveal deadlines, but does not currently aggregate finite RmlUI update
delays across contexts.

This is a candidate integration gap to measure, not a diagnosed performance
bug. A shared-context experiment must record the next-update delay after each
`Update()` and compare two policies: the current immediate-animation behavior
and a scheduler that wakes at the earliest finite RmlUI delay. If the latter
changes idle CPU use or perceived responsiveness, it may explain part of a
historical global-context result without requiring a context-count change.

For a runtime trace, set `LFS_RML_CONTEXT_DEMAND_TRACE=1`. The integration
then emits `gui_render.rmlui_context_demand` only when a queued context changes
between immediate, delayed, or idle update demand. It is diagnostic-only and
off by default; use it to identify the context that keeps the render loop
awake without adding one log line per frame.

### Initial baseline observation

An initial 1280x720 performance-log capture on 2026-07-28 produced 5,225 GUI
frames over about 44 seconds. Its two log files were consecutive rotation
segments of one run, not independent samples. The trace is useful for locating
work, but it is not a clean performance result: verbose performance logging
itself writes enough data to rotate the log, and the run includes application
and Python redraw activity.

The meaningful observation is distribution, not the largest wall-clock sample:
the scene header's `scene_panel_native` context refreshed its cached Vulkan
layer 386 times, with a median refresh time of about 6.6 ms. The rendering-tab
context refreshed 41 times with a median of about 0.4 ms. The trace also
contains an input-free interval in which `gui_anim=true` kept frames running at
roughly 20 Hz. At the time, it did not identify which RmlUI context requested
the immediate update, so it could not be attributed to context count or to the
shared-context question.

### Context-demand follow-up

A second capture on the same machine on 2026-07-28 enabled
`LFS_RML_CONTEXT_DEMAND_TRACE=1`. Its three rotated files are consecutive
segments of one approximately 18-second run (13:48:15--13:48:33), with 7,264
`loop_end` records. The trace reports only state changes, rather than adding a
line for every frame.

At startup, `scene_panel_native`, `lfs.rendering`, and `startup_overlay`
briefly requested an immediate update. The scene and rendering contexts became
idle within about 0.2 seconds; the startup overlay became idle after its
approximately three-second transition. The shell frame, right panel, viewport
overlay, status bar, and menu bar initially reported `idle`. Later state
changes were short interaction-sized bursts in `scene_panel_native`,
`lfs.rendering`, `menu_bar`, and `modal_overlay`; every completed burst resolved
to `idle` or, twice, to a finite delay of 0.578 seconds. The final menu-bar
transition occurred about one second before the application stopped, so it does
not establish a sustained state.

Most importantly, from 13:48:19.653 until 13:48:26.108 none of the queued
contexts changed from `idle` to immediate demand. Nevertheless, the loop
recorded 2,965 frames from 13:48:20 through 13:48:25; 2,960 of those had
`py_redraw=true`. This rules out the queued RmlUI contexts as the explanation
for that dense period. The transition into that period follows dismissal of the
startup overlay. The now-visible Python empty-state overlay called
`lf.ui.request_redraw()` unconditionally from its draw hook, so every draw
requested the next frame. That call was removed: the overlay is static, and
its visibility and content changes already enter through ordinary input or
state invalidation. This was a render-on-demand feedback loop, not an RmlUI
context cost. A follow-up trace after the removal closed the startup overlay
at 13:58:30.726, observed one `py_redraw=true` frame at 13:58:30.809, and then
returned to `py_redraw=false` while idle. Its later dense sections carried
`input_event=true` and mouse-move records, which is expected interaction work
rather than an idle redraw loop.

The finding still does **not** prove that a shared context is cheap. It does
establish that the historical global-context concern cannot be validated by
attributing this capture's high frame count to the current RmlUI contexts.

The finite-delay observations also make the scheduler experiment concrete:
measure whether aggregating the earliest `GetNextUpdateDelay()` improves
responsiveness or idle wake-ups before changing the context topology. The
trace is a diagnostic run with performance logging enabled, not a benchmark;
repeatable A/B timings and a small shared-context spike remain required.

The first shared-context prototype should be deliberately small, but it cannot
reuse two current `RmlPanelHost` instances unchanged. A host currently owns and
destroys its context, resizes that context to its panel bounds (including
multi-pass content-height measurement), owns a per-host layer cache, and
forwards local input and capture. Sharing its `Rml::Context` would therefore
make one panel's size or lifecycle change affect the other before it tests any
useful native behavior.

Use two stages instead:

1. **Framework spike outside the current host path.** This now exists as
    `test_rml_shared_context_contract.cpp`. It creates one real RmlUI context
    with two controlled documents and verifies native document ordering, hit
    testing, click delivery for same-bounds documents, window-space document
    bounds, and independent document unloading. It is deliberately headless:
    it establishes framework semantics, not rendering cost or full-window
    application behavior.
2. **Narrow application prototype after an ownership split.** Separate context
   ownership, document bounds, input dispatch, and cache ownership in
   `RmlPanelHost` or a successor. Only then place the right-panel chrome and
   one known RML-backed active tab in a full-window shared context, leaving
   every other panel type on the existing path and retaining a runtime fallback.

The second stage is still a measurement prototype, not a proposed migration.
`Context::Render()` renders the visible documents of that context, whereas the
current path can capture and reuse each panel's layer independently. It must
therefore answer whether a change in one tab invalidates or records unrelated
right-panel content, and whether the removed manual routing compensates for
any broader render work.

The framework spike also establishes a current limitation. With two documents
given distinct window-space bounds, `Context::GetElementAtPoint()` identifies
the document under each point and follows a document move after `Update()`.
However, sending those same points through `Context::ProcessMouse...()`
delivered both clicks to the document loaded first, including when the listener
was attached directly to each hit-test target. Therefore a shared context cannot
yet be assumed to remove application-level document input filtering. Any
dynamic-host spike must retain the present explicit document ownership check
while investigating whether a different RmlUI document structure or input entry
point can make the dispatch native.

### Opt-in shell/right-panel prototype

The first application-level spike is enabled only by setting
`LFS_RML_SHARED_RIGHT_PANEL_SPIKE=1` before starting the application. It makes
`RmlRightPanel` borrow the full-window `shell_frame` context instead of
creating `right_panel`. The right-panel document receives explicit
window-space bounds; its input is sent in window coordinates and restricted to
elements owned by that document. The shell frame records the shared context
once, after the right-panel document has updated. Without the environment
variable, the two components retain their separate contexts and existing cache
paths.

This is deliberately a pessimistic cache experiment: changing right-panel
chrome refreshes the shell's full-window cache. That makes any cost visible and
avoids claiming an artificial cache win. It also means this prototype is not a
candidate final architecture. It does not include a dynamic `RmlPanelHost`,
does not remove `GuiManager`'s editor-level precedence policy, and does not
yet remove dropdown or tooltip bridging code.

Validate the spike manually before interpreting timings:

1. open, change, close, and scroll tabs; drag the panel-width edge and scene
   splitter; resize the application window;
2. move the pointer from the right panel into the viewport and back, checking
   cursor and hover clearing;
3. open a floating panel above the right panel and verify it still blocks
   underlying interaction; and
4. capture the same idle and interaction workloads with and without the
   variable, using the existing CPU timers and an external GPU profiler when
   available.

#### First diagnostic capture

Two local performance-log captures on 2026-07-28 verified the topology change.
The baseline emitted a `right_panel` context-demand record; the spike emitted
its startup marker and no `right_panel` context record, while `shell_frame`
remained the shared context. Neither capture contained an RmlUI or right-panel
error. The one common unrelated error was a failed MCP HTTP-server start.

The captures are **not** a performance comparison: the baseline ran about
44.7 seconds and logged 710 UI frames, while the spike ran about 39.5 seconds
and logged 1,154 UI frames with a different interaction sequence. They do,
however, confirm the intended invalidation trade-off. The baseline refreshed
the shell cache once; the spike refreshed it 63 times as the right-panel
document changed. Those shared-cache refreshes had a median CPU time of about
0.46 ms (maximum 1.90 ms in this run). This is expected for the deliberately
full-window cache and is evidence to improve cache scope, not a reason to
claim a performance win or loss.

Before either stage loads multiple documents, verify that their RmlUI data-model
names and element ids do not collide. The current documents generally use named
data models such as `asset_manager`, `rendering`, and `training`, but that is a
precondition to check, not a namespace guarantee for plugins.

The ownership split must preserve these explicit contracts:

| Responsibility today | Required future boundary |
| --- | --- |
| `RmlPanelHost` creates and destroys `context_name_` | a shared-context owner alone creates and destroys the context; documents release only themselves |
| `RmlPanelHost::updateContextLayout()` calls `SetDimensions()` | shared contexts have window dimensions; each document receives its own bounds and local layout contract |
| per-host `direct_cache_` captures one context layer | cache scope is selected deliberately: whole shared context, safe document region, or no cache for the spike |
| `RmlPanelHost::forwardInput()` translates/captures per panel | one shared owner sends window-space input to RmlUI once, with application policy only at external boundaries |
| per-host tooltip/dropdown helpers bridge local coordinates | use native document behavior where possible; retain only an explicit bridge for behavior RmlUI cannot express |

RmlUI's own context documentation confirms the constraint being tested:
`Update()` processes elements in the context's documents and `Render()` renders
all visible elements in those documents. The application controls when those
calls happen, so invalidation scheduling and cache capture remain part of the
integration design. See the [RmlUI context manual](https://mikke89.github.io/RmlUiDoc/pages/cpp_manual/contexts.html).

Record at least these workloads for both paths:

| Workload | Purpose |
| --- | --- |
| Idle after caches settle | establishes cache reuse and wake-up cost |
| Pointer movement across chrome and active tab | exercises hover, cursor, and hit testing |
| Text edit, selection, and keyboard dismissal in the tab | exercises focus and keyboard ownership |
| Open dropdown or tooltip crossing the tab boundary | exercises native stacking, popup placement, and input capture |
| Tab/model update and right-panel resize | exposes document invalidation and full-window cache cost |

`GlobalContextMenu` is a useful existing full-window context, but it is not
proof that arbitrary panel RML can simply be moved there. A shared popup host
would need an explicit window-space anchor and a defined focus, capture,
dismissal, and callback contract. It is one experiment to evaluate, not the
only acceptable result.

### Test and benchmark roles

The focused GUI tests protect the current panel orchestration while experiments
are developed. In particular, `test_panel_registry_render_paths.cpp` guards
preload, direct-draw, and cached-draw fallback contracts, and the layout and
animation-demand tests guard related registry behavior. They are useful
regressions for a migration because a changed rendering architecture must not
silently break those contracts.

`test_rml_shared_context_contract.cpp` is a deliberately initialized, live
RmlUI-context test; unlike the registry tests, it does not use a mock context.
It proves only the framework behavior listed above. It does **not** compare
context counts, exercise application popup policy, render through the Vulkan
backend, or measure CPU/GPU performance. Passing it is a necessary safety
signal for a future shared owner, not evidence that a global or shared context
is viable. The experiment protocol above therefore still requires focused
tests and reproducible benchmark traces.

If the framework spike becomes an application prototype, extend tests in
layers:

1. extend the existing deterministic RmlUI integration test with focus
   transfer and input outside both documents;
2. add focused tests for the ownership boundary, proving that an owned context
   is released exactly once; and
3. the existing `BUILD_GUI_TESTS` regressions for panel selection, cache
   fallback, and layout orchestration.

`RmlContextOwner` now makes the first ownership boundary explicit. The current
`RmlPanelHost` uses its owned form, so behavior is unchanged. Its borrowed form
is covered by the same RmlUI harness: destroying that wrapper leaves the shared
context live and able to load a document. This is intentionally only the
lifetime boundary; it does not make current panel-local layout, input, or cache
state safe to share yet. Performance remains a benchmark rather than a
unit-test assertion, because it depends on scene, GPU, window size, and
interaction workload.

## Current boundary

Until an experiment produces evidence, retain the current routing and cache
code rather than replacing it with an incomplete abstraction. This document
does not select one context for all UI, retain one context per panel forever,
or prescribe a shared popup context. It records the current behavior, the
custom code to minimize, and the evidence required for a later implementation
decision.

## Relevant code

- `src/visualizer/gui/rmlui/rml_panel_host.cpp`
- `src/visualizer/gui/rmlui/rmlui_manager.cpp`
- `src/visualizer/gui/gui_manager.cpp`
- `src/visualizer/gui/panel_registry.cpp`

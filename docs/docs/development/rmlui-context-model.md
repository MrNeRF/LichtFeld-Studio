---
sidebar_position: 5
---

# RmlUI Context Model

LichtFeld Studio intentionally uses more than one RmlUI context. The normal
panel path is one `RmlPanelHost` context per panel, rendered into a panel-sized
texture and then composited into the window. The fixed UI also has independent
contexts for the shell frame, menu bar, status bar, right panel, toast overlay,
modal overlay, startup overlay, viewport overlay, sequencer, sequencer overlay,
and global context menu.

This is not an accidental duplication. A panel that did not change can reuse
its cached texture instead of recording its RmlUI document again. The
`performance.log` baseline for this model reports cached right-panel layout at
roughly 0.08--0.11 ms, RmlUI recording at roughly 0.08--0.40 ms, and CPU UI
work before Vulkan frame recording at roughly 0.6--0.9 ms. Keeping that
property matters more than reducing the context count.

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

## Decision

Keep the one-context-per-panel model. Do not replace it with one global RmlUI
context, and do not introduce a shared transient popup context as part of this
work.

The alternatives considered are:

1. **One context for all UI.** Rejected: it would discard the panel-sized cache
   boundary that makes unchanged panels inexpensive to render.
2. **Keep the current model.** Accepted: it preserves the measured cache path
   and contains the present routing behavior in its existing owners.
3. **One shared transient popup context.** Technically feasible, but deferred:
   `GlobalContextMenu` already demonstrates a full-window, independently
   cached context. Moving a panel dropdown or tooltip is not a reparenting
   operation, however. Native select state and document-local tooltip elements
   belong to their source context. A shared context would need a popup model,
   window-space anchor, callbacks for selection/dismissal, and an explicit
   focus and capture handoff. It needs that design and a cache-cost measurement
   before it is safer than the current code.

## Popup direction

The current model remains the default: panels stay independent and cacheable.
This document does not introduce a shared popup context or change existing
input handling.

If manual popup code becomes the limiting factor, the next step is a scoped
proposal for one shared transient overlay context. It should host only
cross-panel popup primitives such as dropdowns, context menus, and tooltips;
it must not absorb ordinary panels or modal ownership. Before that change can
land, the proposal must specify:

1. how an anchor in a panel-local context becomes a window-space rectangle;
2. where the transient overlay sits relative to modal and menu contexts;
3. pointer capture, dismissal, wheel routing, and keyboard focus rules; and
4. cache invalidation and rendering cost compared with the current panel cache.

The proposed shared context would render a new popup document from data owned
by the source panel. It would not move the source panel's existing RmlUI
elements between contexts. `GlobalContextMenu` is the closest existing example
of the required full-window rendering and input shape, but it is not a generic
popup host yet.

Until those rules and measurements exist, retain the existing manual routing
instead of replacing it with a partial overlay abstraction.

Reopen the shared-overlay option only when at least one of these conditions is
true:

- a new popup must cross a panel boundary and would otherwise duplicate the
  current dropdown routing;
- two existing popup implementations need the same window-space anchor,
  capture, dismissal, and focus behavior; or
- profiling shows that the manual routing or popup redraw, rather than the
  panel cache, is a material UI cost.

The follow-up must preserve the current panel cache and compare the relevant
`gui_render.panel_layout.*` and `gui_render.rmlui_record` timings before and
after the change. It should also cover the following interaction cases:

- a popup extending over another panel;
- modal and menu precedence over a popup;
- click, release, and wheel input after leaving the popup bounds; and
- keyboard focus and dismissal while a popup is open.

## Scope boundary

This document is the source of truth for the context model and its ownership
rules. It does not introduce a popup migration. Any shared transient overlay
implementation is a separate behavior-changing change with its own design and
validation plan.

## Relevant code

- `src/visualizer/gui/rmlui/rml_panel_host.cpp`
- `src/visualizer/gui/rmlui/rmlui_manager.cpp`
- `src/visualizer/gui/gui_manager.cpp`
- `src/visualizer/gui/panel_registry.cpp`

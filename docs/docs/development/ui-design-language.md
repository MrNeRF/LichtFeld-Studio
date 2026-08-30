---
sidebar_position: 6
title: UI design language and window patterns
---

# UI design language and window patterns

This document is the working contract for LichtFeld Studio's RmlUi windows and
panels. It describes the visual and structural language already established by
the Asset Manager, Rendering, Training, Preferences, frame-extraction, and
shared dialog surfaces.

The examples in this document are implementation references, not a requirement
to reproduce every existing detail. In particular, older resources may still
contain hard-coded colors or inline layout declarations. New work should follow
the themed/shared patterns below.

## The short version

1. Choose a window class from the taxonomy before writing markup.
2. Reuse the shell (`docked-panel` or `floating-window`) and shared components.
3. Keep RML semantic and data-bound; put layout in `.rcss` and palette values
   in `.theme.rcss`.
4. Use the shared tokens and semantic state classes for color, focus, status,
   and destructive actions.
5. Make the content fit the available width and height, including a narrow
   dock, a resized floating window, long translations, and light themes.
6. Use a real modal for a blocking decision; use an anchored popup for a local
   choice; use a context menu for a command list.

## Visual language

### Palette roles

The theme catalog defines roles, not per-panel colors. A panel should express
hierarchy by choosing a role and an elevation, not by inventing another blue or
gray. Use `@{...}` tokens in `.theme.rcss` and keep semantic colors consistent
across all themes.

| Role | Use it for | Typical token/classes |
| --- | --- | --- |
| `background` | Application-level and chart/backdrop depth | `@{background}` |
| `surface` | The panel body and ordinary controls | `@{surface}` |
| `surface_bright` | Raised controls, hover surfaces, selected containers | `@{surface_bright}` |
| `border` | Dividers, input outlines, card edges, resize affordances | `@{border}` |
| `text` | Primary labels, values, titles | `@{text}`, `.text-default` |
| `text_dim` | Hints, metadata, disabled labels, secondary values | `@{text_dim}`, `.text-muted`, `.text-disabled` |
| `primary` | Selection, active tabs, focus, links, the main action | `@{primary}`, `.btn--primary` |
| `success` | Completed/enabled/positive actions | `.status-success`, `.btn--success` |
| `warning` | Caution, pending work, overwrite or destructive confirmation context | `.status-warning`, `.btn--warning` |
| `error` | Failed work and irreversible/destructive actions | `.status-error`, `.btn--error` |
| `info` | Informational status and non-critical guidance | `.status-info` |

Use alpha/blend helpers for hierarchy (`@{alpha(...)}`, `@{blend(...)}`) rather
than opaque hard-coded colors. Selection should be legible through both an
accent edge and a restrained surface tint. Hover should be discoverable but
must not look like a committed selection. Disabled controls use the shared
opacity treatment and remain readable enough to explain why they are present.

### Type, density, and geometry

- Inter is the default UI font. The shared component stylesheet uses a compact
  `12dp` body/control scale; use it for ordinary labels and controls.
- Reserve larger type for hierarchy: approximately `14dp` for a subsection
  emphasis and `16–17dp` for a prominent metric or empty-state title. Do not
  turn every label into a heading or uppercase label.
- Use JetBrains Mono for values that benefit from column alignment or exact
  reading, such as numeric telemetry, ranges, paths, and diagnostic details.
- Author dimensions in `dp`. Reserve `px`/inline geometry for values calculated
  at runtime (for example a chart bar height or a popup position).
- Prefer the theme's spacing, rounding, and scrollbar tokens. The built-in
  presets expose window/frame/popup rounding, frame padding, item spacing,
  inner spacing, scrollbar size, and minimum grab size. A panel-specific value
  is justified only when it describes content geometry (for example a preview
  aspect ratio or a timeline height).
- Use borders and small surface changes to separate regions. A card does not
  need a shadow or a large radius just because it is a card. Floating windows
  receive the shared elevation; nested content should stay visually quiet.

### Interaction states

Every interactive element needs an intentional default, hover, active/pressed,
focus-visible, disabled, and (where relevant) selected state. The shared
components already provide these states for buttons, fields, selects, sections,
checkboxes, radios, sliders, context-menu items, and icon buttons.

Keyboard focus must remain visible and navigation order must follow the visual
reading order. A tooltip can explain an icon-only action, but it does not
replace a meaningful accessible label. Localize visible text and tooltip keys;
do not use an icon or color as the only explanation of state.

## Window taxonomy

Choose one of these surfaces before deciding on markup. A panel may move between
docked and floating space, but its content should not become a second unrelated
design when it does.

| Surface | Use it for | Shell and placement | Reference |
| --- | --- | --- | --- |
| Docked panel | Persistent controls or status that belong beside the viewport | `docked_panel.rml`; plugin `space` is usually `MAIN_PANEL_TAB`, `SIDE_PANEL`, or a built-in parent | Rendering, Training, Asset Manager |
| Floating panel/window | A utility with independent width/height, a larger workflow, or a temporary tool | `floating_window.rml`; `space = FLOATING`; provide a sensible `size` | Preferences |
| Native/custom floating window | A media preview or specialized interaction that needs direct/native rendering | Keep the shared title/content/overlay contract even when the panel owns its host | Video/frame extraction |
| Anchored popup | A short local choice, such as a color picker or compact menu | Position next to its trigger, clamp to the host, close on completion/outside click | Rendering and Training color pickers |
| Modal dialog | A decision or input that must block the underlying task | `modal_overlay.rml` through the modal API; use an explicit semantic style | Confirm, overwrite, conflict, and error dialogs |
| Context menu | A command list attached to an item or viewport location | Global context-menu resource or `begin_context_menu()` | Asset actions and viewport commands |
| Viewport overlay | Contextual, low-chrome information or manipulation directly over the 3D view | `VIEWPORT_OVERLAY`; do not use it as a substitute for a settings window | Selection/gizmo overlays |

### Docked panels

A docked panel is part of the application's continuous workspace. It should not
draw a fake title bar or an extra outer window border. Start from the docked
template and let `#content-wrap`/`#content` fill the host. Keep the panel's
primary scroll region explicit; fixed headers, telemetry, and action rows should
not jump when a long settings list scrolls.

Rendering and Training demonstrate the canonical settings pattern:

```rml
<body template="docked-panel" data-model="my_panel">
  <div class="section-header text-accent" data-event-click="toggle_section('camera')">
    <span class="section-arrow text-accent">&#x25B6;</span>
    <span>{{camera_label}}</span>
  </div>
  <div class="section-content">
    <div class="setting-row setting-row--aligned">
      <span class="setting-row__label-col">{{exposure_label}}</span>
      <div class="setting-row__control-col setting-row__control-col--slider">
        <input type="range" class="setting-slider" data-value="exposure" />
        <span class="slider-value">{{exposure_text}}</span>
      </div>
    </div>
  </div>
</body>
```

The shared `setting-row__label-col` gives Rendering and Training the same label
column, while the control column can fill the remaining width. Use
`setting-row__control-col--fill`, `--slider`, `--checkbox`, and `--color` for
the intended control geometry instead of creating another row layout per panel.

### Floating windows

Use the floating template for an independent window. It supplies the shared
title bar, close affordance, content padding, scroll behavior, border, radius,
and theme-driven shadow. A retained Python panel normally looks like this:

```python
from pathlib import Path
import lichtfeld as lf


class InspectorPanel(lf.ui.Panel):
    id = "example.inspector"
    label = "Inspector"
    space = lf.ui.PanelSpace.FLOATING
    template = str(Path(__file__).with_name("inspector.rml"))
    size = (520, 420)
    height_mode = lf.ui.PanelHeightMode.FILL
    update_policy = "dirty"
```

The RML chooses the shell explicitly:

```rml
<head>
  <link type="text/template" href="floating_window.rml"/>
  <link type="text/rcss" href="inspector.rcss"/>
</head>
<body template="floating-window" data-model="inspector">
  <!-- semantic content only -->
</body>
```

The floating window must remain useful when resized. Give large content a
single clear scroll region, use flexible columns, and let labels ellipsize or
wrap rather than letting controls paint outside the frame. The Preferences
window is the reference for a two-column utility: navigation is stable on the
left, content scrolls independently, and the footer action row stays visible.

### Specialized windows

The frame-extraction window is intentionally richer than an ordinary settings
panel: preview, transport controls, timeline, trim range, input/output paths,
format settings, progress, and completion/error states form one task sequence.
That sequence is the useful pattern:

1. Put the primary preview or result first.
2. Keep transport/timeline actions close to that preview.
3. Group settings by task with clear section rules.
4. Keep the main action and progress/status at the end of the flow.
5. Put overwrite confirmation on an input-blocking overlay.

Specialized rendering may justify its own chart/timeline classes, but it still
uses the shared title bar, button variants, typography roles, spacing rhythm,
and modal behavior. Avoid turning a specialized window into a new global shell.

## Constructing a panel

### Separate the four layers

RmlUi resources have four clear owners:

| Layer | Owns | Should not own |
| --- | --- | --- |
| `.rml` | DOM structure, semantic classes, stable ids, localization bindings, events | Palette literals, runtime layout calculations, business logic |
| `.rcss` | Flex/grid layout, dimensions, spacing, overflow, typography scale, transitions | Theme-specific palette choices |
| `.theme.rcss` | Colors, image tints, themed decorators, shadows, themed radii | Content structure or business behavior |
| Python/C++ | Runtime state, model values, event handlers, visibility, live geometry, data-driven colors | Broad static selector styling |

Every hosted RML document receives the shared component stylesheet and themed
component styles. A sibling stylesheet is discovered automatically when a
retained panel has `panel.rml`, `panel.rcss`, and optionally
`panel.theme.rcss`. The effective order is shared components, linked panel
styles, sibling base styles, host theme styles, then sibling theme styles.

Use the panel's data model for state and event bindings:

```python
def on_bind_model(self, ctx):
    model = ctx.create_data_model("inspector")
    if model is None:
        return
    model.bind_func("title", lambda: self._title)
    model.bind_event("reset", self._reset)
    self._handle = model.get_handle()


def on_mount(self, doc):
    self._doc = doc


def on_unmount(self, doc):
    self._doc = None
    if doc:
        doc.remove_data_model("inspector")
```

For data panels, prefer `update_policy = "dirty"` and invalidate the model
when state changes. Use interval updates only for animation-like work. Clean up
subscriptions and document listeners in `on_unmount`; otherwise a closed
window can continue to receive events.

### Shared component recipes

#### Toolbars

Use a compact flex row with one clear primary action, optional search/filter,
and icon buttons for frequent utility actions. Keep destructive actions away
from the primary action and use the semantic button variant. A close button is
part of floating chrome; a docked panel normally does not need one.

The Asset Manager is a useful management-toolbar composition: search, import,
view mode, sort, refresh, and scan status are separate affordances, followed by
the content region. Preserve that hierarchy even if the toolbar is simplified
for a narrower panel.

#### Sections and settings

Use `.section-header` with `.text-accent`, a `.section-arrow`, and a matching
`.section-content` for collapsible groups. Keep the expanded state in model or
panel state so it survives redraws. Use one row per setting, a stable label
column, and a control whose value is visible without opening another dialog.

Preferences extends this pattern with navigation tabs, descriptions, and a
persistent footer. Descriptions explain scope or side effects; they should not
replace a label or become a wall of text.

#### Buttons and status

Use `.btn` plus one variant:

- `.btn--primary`: the next/main action;
- `.btn--secondary`: neutral navigation, browse, cancel, or reset;
- `.btn--success`: start, resume, apply, or completed positive state;
- `.btn--warning`: caution or pending/overwrite context;
- `.btn--error`: delete, uninstall, clear, or other irreversible action.

Use `.status-success`, `.status-warning`, `.status-error`, `.status-info`,
`.text-muted`, and `.text-disabled` for status text. Pair color with a word or
icon so status is not color-only. A progress bar should have a nearby label or
percentage and should occupy the same region as the operation it describes.

Training is the reference for state-specific controls: ready, running,
paused, completed, stopped, stopping, and error each expose only the actions
valid for that state. Do not leave an action enabled and rely on a later error
message to explain that it is unavailable.

#### Lists, cards, and details

Use a list when comparison and scanning matter; use cards when a preview and a
small action set matter. Both need a stable selected state, a subtle hover
state, truncated long text, and explicit empty/loading/error states.

The Asset Manager combines a toolbar, filter/sidebar, list-or-gallery switch,
virtualized result region, and selected-item details. This is the preferred
information architecture for a dense catalog. Keep the details region useful
for no selection, multiple selection, missing files, and a selected item; do not
leave a blank rectangle when the data is unavailable.

#### Popups, dialogs, and menus

Choose the smallest blocking scope:

- Use an anchored, non-modal popup for a color picker or a compact local
  editor. Position it from runtime geometry, clamp it to the host, and close it
  after the choice or when focus leaves it.
- Use `lf.ui.confirm_dialog`, `input_dialog`, or `message_dialog` for a global
  decision, input, or message. These use `modal_overlay.rml`, semantic info /
  warning / error styles, a themed surface, and a backdrop that blocks input.
- Use the global context-menu resource or `begin_context_menu()` for commands
  attached to an item. Do not repurpose a modal for a context menu.

The modal backdrop is an input boundary, not merely a dimming effect. While a
modal is visible, pointer and keyboard input must not reach the viewport or a
panel underneath it. Confirmations should name the consequence and provide a
neutral cancel action plus a clearly styled destructive/affirmative action.

## Do and don't

| Do | Don't |
| --- | --- |
| Start with `docked-panel` or `floating-window`. | Create a third title-bar/window shell for an ordinary panel. |
| Reuse `.btn`, `.setting-row`, `.section-header`, `.progress__text`, and status classes. | Copy a neighboring panel's private button or row styles into a new namespace. |
| Put palette choices in `.theme.rcss` with tokens. | Hard-code dark-theme colors in RML, base RCSS, inline style, or SVG attributes. |
| Use model-driven classes for selected, expanded, disabled, loading, and error states. | Mutate broad static style properties from every update tick. |
| Use `dp` for authored dimensions and flexible flex children. | Use fixed widths that only fit the default English/dark window size. |
| Keep one source of truth for docked/floating content. | Duplicate markup and let docked and floating versions drift. |
| Use `lf.ui.confirm_dialog` for global confirmations. | Hand-roll a local modal backdrop for each destructive action. |
| Give icon-only actions a tooltip/accessible title and a focus state. | Make an unlabeled icon the only way to discover an action. |
| Make destructive actions explicit and use `.btn--error`. | Use the primary accent for delete/uninstall/clear. |
| Check empty, loading, missing, disabled, error, and success states. | Treat the happy path as the whole design. |
| Test all registered themes, narrow docks, resizing, keyboard focus, and long text. | Validate only a fixed-size dark screenshot. |

## Review checklist

Before calling a window or panel complete, verify:

- [ ] The surface has the correct `PanelSpace` and shell for its lifetime and
      interaction scope.
- [ ] RML contains semantic structure and bindings; static layout is in base
      RCSS; themed values are in the sibling theme RCSS.
- [ ] No new hard-coded palette literals, inline static styles, or broad
      per-frame `SetProperty` styling were introduced.
- [ ] Shared focus, hover, active, selected, disabled, and status states are
      visible and keyboard reachable.
- [ ] The panel works docked and floating where that transition is supported.
- [ ] Headers/actions remain visible while only the intended content region
      scrolls.
- [ ] Narrow width, resized height, long localized strings, and all manifest
      themes have been checked.
- [ ] Empty, loading, missing-data, success, error, and destructive-confirm
      states are explicit.
- [ ] Modal overlays block underlying viewport/panel input and can be closed or
      answered without leaving stale listeners behind.
- [ ] Dynamic lists preserve stable identity, focus, and expansion state across
      model updates.
- [ ] Runtime subscriptions, listeners, textures, and data models are released
      in `on_unmount`/the native equivalent.

## Reference resources

Use these files when implementing or reviewing a surface:

- `src/visualizer/gui/rmlui/resources/components.rcss` and
  `components.theme.rcss`: shared controls, rows, buttons, status, sections,
  scrollbars, and theme tokens.
- `src/visualizer/gui/rmlui/resources/docked_panel.rml` and
  `floating_window.rml`: standard panel shells.
- `src/visualizer/gui/rmlui/resources/modal_overlay.rml` and
  `modal_overlay.theme.rcss`: global modal structure and semantic dialog styles.
- `src/visualizer/gui/rmlui/resources/global_context_menu.rml`: context-menu
  structure and themed menu surface.
- `src/visualizer/gui/rmlui/resources/asset_manager.{rml,rcss,theme.rcss}`:
  dense catalog information architecture and selection/details states.
- `src/visualizer/gui/rmlui/resources/{rendering,training,preferences,
  video_extractor}.{rml,rcss,theme.rcss}`: settings, stateful controls,
  preferences navigation, and specialized media workflow patterns. (The
  frame-extraction resource currently has no sibling theme file; its next
  modernization should move palette literals into themed styles.)
- `docs/docs/development/rmlui-styling.md`: stylesheet ownership, load order,
  and the boundary between static styling and runtime geometry.
- `docs/plugins/getting-started.md`: panel metadata, retained shells, panel
  spaces, lifecycle hooks, layout choices, and popup APIs.

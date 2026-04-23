---
sidebar_position: 5
---
# RmlUI Styling

LichtFeld Studio keeps RmlUI structure, styling, theme values, and runtime state in separate layers.

## Ownership

- `.rml` owns document structure, bindings, stable ids, and semantic classes.
- `.rcss` owns static layout and non-theme styling: spacing, flex rules, dimensions, overflow, typography scale, transitions, and intentionally semantic non-palette colors.
- `.theme.rcss` owns palette-dependent styling. Use it for `color`, `image-color`, `background-color`, `border-color`, themed decorators, shadows, and themed radii.
- C++ owns runtime state only: dynamic geometry, visibility, model-driven attributes, live data values, data-driven colors, and RmlUI behavior properties such as drag or navigation.

Do not generate broad selector styles in C++. If a selector can be edited as a resource, it belongs in `.rcss` or `.theme.rcss`.

## Load Order

Every RmlUI document receives shared component styles first:

1. `components.rcss`
2. `components.theme.rcss`
3. linked `text/rcss` files from the document, in document order
4. the document's sibling `.rcss` if it was not already linked
5. the document's sibling `.theme.rcss`

Documents hosted by `RmlPanelHost` also receive `panel_host.theme.rcss` before the sibling theme file. The host file is intentionally small and should stay generic; panel-specific selectors belong beside the panel.

For a panel named `rendering.rml`, keep the files together:

```text
rendering.rml
rendering.rcss
rendering.theme.rcss
```

## Theme Tokens

Theme files are templates. Use `@{...}` tokens instead of hardcoded palette values:

```css
.panel-title {
    color: @{text};
    border-top-color: @{panel.primary_border_soft};
}

.panel-row:hover {
    background-color: @{alpha(primary,0.12)};
}
```

Supported token forms include palette names such as `@{text}`, `@{surface}`, `@{primary}`, `@{warning}`, `@{error}`, numeric values such as `@{num(size.frame_rounding)}`, and color helpers such as `@{alpha(border,0.4)}` or `@{blend(surface,primary,0.18)}`.

## Theme Catalog

Theme presets are listed in `src/visualizer/gui/assets/themes/manifest.json`. The manifest owns catalog metadata:

- `id`: stable preset id used by preferences and `lf.ui.set_theme(id)`
- `file`: theme value file under the same directory
- `fallback`: built-in base theme used for missing values
- `label_key`: translation key used by UI menus
- `mode`: `dark` or `light`
- `order`: menu order

Individual theme JSON files own values only: `name`, `palette`, `sizes`, `fonts`, and other theme sections. Do not duplicate catalog fields such as `id`, `label_key`, `mode`, or `order` inside the theme value file.

Python UI code must read `lf.ui.themes()` instead of hardcoding the theme list. `lf.ui.get_theme()` returns the stable id of the active preset.

## Hot Reload

RmlUI hot reload watches `.rml`, `.rcss`, and `.theme.rcss`. Editing a resource should change the live UI without requiring a C++ rebuild.

If a theme edit appears to do nothing, check for one of these issues:

- The selector is still styled in C++ with `SetProperty`.
- The selector is in a shared theme file that loads after the file you edited.
- The element is using a data-driven inline style, such as chart coordinates or timeline positions.
- The file is not the document's sibling `.theme.rcss` and is not imported by that document.

## C++ Rules

Allowed `SetProperty` examples:

- popup `left` / `top`
- viewport or document `width` / `height`
- virtual-list row `top`
- `display` for runtime visibility
- progress width or model-driven chart geometry
- data-driven colors, for example per-keyframe color chips
- explicit user/API sizing, for example an immediate-mode `button(size=...)`
- RmlUI behavior properties such as `drag`, `nav-left`, and `nav-right`

Avoid `SetProperty` for static styling:

- default `position`, `left`, `right`, `width`, `height`
- default text colors
- default backgrounds and borders
- icon colors
- fixed padding, margins, radius, and shadows

The Python immediate-mode bridge follows the same rule. Generated elements should receive semantic classes such as `.im-control--fill` or `.im-label--centered`; only data colors, caller-supplied sizes, tooltip coordinates, and runtime state should remain inline.

When in doubt, create a class in `.rml` or C++, put static layout in `.rcss`, and put palette-dependent values in `.theme.rcss`.

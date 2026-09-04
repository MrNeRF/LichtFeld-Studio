---
sidebar_position: 6
---
# Theme format v2

Theme schema v2 groups related dark and light variants in one family file. The built-in catalog is `src/visualizer/gui/assets/themes/manifest.json`:

```json
{
  "schema_version": 2,
  "families": [
    { "file": "example.json", "order": 50 }
  ]
}
```

The family file contains one or both semantic variants:

```json
{
  "schema_version": 2,
  "id": "example",
  "name": "Example",
  "shared": {
    "sizes": { "window_rounding": 6.0 }
  },
  "variants": {
    "dark": {
      "id": "example_night",
      "name": "Night",
      "label_key": "menu.view.theme.dark",
      "fallback": "dark",
      "palette": {
        "background": [0.08, 0.09, 0.11, 1.0],
        "surface": [0.13, 0.14, 0.17, 1.0],
        "text": [0.96, 0.97, 0.99, 1.0]
      }
    },
    "light": {
      "id": "example_day",
      "name": "Day",
      "label_key": "menu.view.theme.light",
      "fallback": "light"
    }
  }
}
```

[Download the complete v2 family template](/examples/theme-family-v2.json). It is valid JSON and contains every property currently read by the runtime. Keys beginning with `_` are descriptive metadata for humans; the loader ignores unknown keys.

## Family and variant fields

- `schema_version` must be `2`.
- `id` is the stable lowercase family id. Use underscores rather than spaces or hyphens.
- `name` is the family name shown in selectors.
- `shared` is optional. Supported theme properties placed here are applied before the selected variant.
- `variants` must contain a `dark` variant, a `light` variant, or both.
- Each variant requires a unique stable `id`. `name` is the short user-visible variant name and does not have to be `Dark` or `Light`.
- `fallback` selects the built-in `dark` or `light` defaults used for missing properties. An unknown fallback is replaced by the semantic variant default.
- `label_key` and variant `order` are optional. A custom visible name does not require a localization key.

Supported value sections are the sections represented by the runtime theme structure: `palette`, `sizes`, `fonts`, `menu`, `context_menu`, `viewport`, `shadows`, `vignette`, `button`, `overlay`, and `gradients`. Existing font fields are preserved; themes do not need to define them.

Colors use `[red, green, blue, alpha]` arrays with values from `0.0` to `1.0`. Missing properties inherit from `fallback`; malformed family files are ignored without preventing the remaining catalog from loading.

## Property reference

All sections may appear in `shared` or in an individual variant. Shared values are applied first and variant values override them.

| Section | Runtime properties |
| --- | --- |
| `palette` | `background`, `surface`, `surface_bright`, `primary`, `primary_dim`, `secondary`, `text`, `text_dim`, `border`, `success`, `warning`, `error`, `info`, `row_even`, `row_odd` |
| `sizes` | `window_rounding`, `frame_rounding`, `popup_rounding`, `scrollbar_rounding`, `grab_rounding`, `tab_rounding`, `border_size`, `child_border_size`, `popup_border_size`, `window_padding`, `frame_padding`, `item_spacing`, `item_inner_spacing`, `indent_spacing`, `scrollbar_size`, `grab_min_size`, `toolbar_button_size`, `toolbar_padding`, `toolbar_spacing` |
| `fonts` | `regular_path`, `bold_path`, `base_size`, `small_size`, `large_size`, `heading_size`, `section_size` |
| `menu` | `bg_lighten`, `hover_lighten`, `active_alpha`, `popup_lighten`, `popup_rounding`, `popup_border_size`, `border_alpha`, `bottom_border_darken`, `frame_padding`, `item_spacing`, `popup_padding` |
| `context_menu` | `rounding`, `header_alpha`, `header_hover_alpha`, `header_active_alpha`, `padding`, `item_spacing` |
| `viewport` | `corner_radius`, `border_size`, `border_alpha`, `border_darken` |
| `shadows` | `enabled`, `offset`, `blur`, `alpha` |
| `vignette` | `enabled`, `intensity`, `radius`, `softness` |
| `button` | `tint_normal`, `tint_hover`, `tint_active` |
| `overlay` | `background`, `text`, `text_dim`, `border`, `icon`, `highlight`, `selection`, `selection_flash` |

Two-component dimensions such as padding and shadow offset use `[x, y]`. Theme font properties are retained for format compatibility and for consumers that use them; defining a font section is optional.

## Optional gradients

Every gradient is optional and contains `start` and `end` colors. Missing or invalid gradients use the palette-derived default, so existing themes do not need to be changed merely to support gradients.

Supported gradient keys and their current consumers are:

| Gradient | Current use |
| --- | --- |
| `window_body` | Floating-window body and shared window frame |
| `panel_body` | Hosted and docked panel bodies, sequencer surfaces, application toolbar/status chrome, and the theme-derived tint for viewport toolbars, contextual tool strips, and orientation-gizmo buttons |
| `window_title` | Floating-window title bars and top menu chrome |
| `section_header` | Collapsible section headers, sequencer transport header, and right-panel tab chrome |
| `section_header_hover` | Hovered and expanded section headers |
| `progress` | Progress fills, selected controls, active viewport tools, active gizmo buttons, and right-panel accent edges/separators |
| `scrubber_track` | Numeric scrub-field tracks |
| `scrubber_fill` | Numeric scrub-field fills |
| `histogram_header` | Histogram hero/header surface |
| `histogram_fill` | Histogram and comparison bars |
| `histogram_selection` | Selected histogram and comparison bars |

Solid, translucent, and frosted viewport-control modes are a user preference,
not theme-manifest properties. They reuse `panel_body`, `primary`, `secondary`,
`border`, and text roles. Consequently custom themes need no additional glass
fields, and omitting gradients still produces a palette-derived result.

The presence of `panel_body` also enables the enhanced hosted-panel treatment, including compact styled right-sidebar tabs. This capability is driven by the property, not by a hardcoded family name.

For example:

```json
"gradients": {
  "panel_body": {
    "start": [0.15, 0.17, 0.22, 1.0],
    "end": [0.10, 0.11, 0.15, 1.0]
  }
}
```

## Compatibility and fallback

- v1 manifests and standalone theme JSON files remain loadable.
- A missing family file, invalid schema, invalid family id, duplicate variant id, or family without a dark/light variant causes that family to be skipped.
- If the manifest cannot provide any valid themes, LichtFeld falls back to its built-in dark theme.
- If a requested variant is absent, LichtFeld uses the other variant in that family when available. If the saved family no longer exists, it falls back to the `lichtfeld` dark theme.
- A family with only one variant is selected directly and does not expose a mode selector.
- `auto` is valid only for a family with both variants and when system theme detection is available at runtime. Otherwise the saved choice falls back to dark.

Each missing gradient receives a palette-derived fallback. An invalid individual gradient is ignored while the rest of the family remains usable. A flat v1 or v2 theme therefore does not need to add gradients unless it wants the richer treatment.

## Development installation and hot reload

The current release loads its theme catalog from packaged application assets; there is no end-user import/delete UI or user theme directory yet. For development:

1. Copy the [complete template](/examples/theme-family-v2.json) into `src/visualizer/gui/assets/themes/` and rename it.
2. Give the family and variants unique stable ids.
3. Add `{ "file": "your_theme.json", "order": 50 }` to `src/visualizer/gui/assets/themes/manifest.json`.
4. Edit the family while the development resource source is active; the manifest and theme files participate in hot reload.

Removing a registered development theme from the manifest hot-reloads the catalog; if the active preset no longer resolves, the application activates the built-in dark fallback. Deleting only the family file is handled when the catalog is next loaded (for example after restart or a manifest reload): the missing family is skipped and the same fallback applies. File deletion by itself is not a managed uninstall operation. A user-facing importer will require its own storage, validation, and deletion lifecycle.

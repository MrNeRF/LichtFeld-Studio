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

## Family and variant fields

- `schema_version` must be `2`.
- `id` is the stable lowercase family id. Use underscores rather than spaces or hyphens.
- `name` is the family name shown in selectors.
- `shared` is optional. Supported theme properties placed here are applied before the selected variant.
- `variants` must contain a `dark` variant, a `light` variant, or both.
- Each variant requires a unique stable `id`. `name` is the short user-visible variant name and does not have to be `Dark` or `Light`.
- `fallback` selects the built-in `dark` or `light` defaults used for missing properties. An unknown fallback is replaced by the semantic variant default.
- `label_key` and variant `order` are optional.

Supported value sections are the sections represented by the runtime theme structure: `palette`, `sizes`, `fonts`, `menu`, `context_menu`, `viewport`, `shadows`, `vignette`, `button`, `overlay`, and `gradients`. Existing font fields are preserved; themes do not need to define them.

Colors use `[red, green, blue, alpha]` arrays with values from `0.0` to `1.0`. Missing properties inherit from `fallback`; malformed family files are ignored without preventing the remaining catalog from loading.

## Optional gradients

Every gradient is optional and contains `start` and `end` colors. Missing or invalid gradients use the palette-derived default, so existing themes do not need to be changed merely to support gradients.

Supported gradient keys are:

- `window_body`, `panel_body`, and `window_title`
- `section_header` and `section_header_hover`
- `progress`
- `scrubber_track` and `scrubber_fill`
- `histogram_header`, `histogram_fill`, and `histogram_selection`

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

The catalog and theme files participate in theme hot reload during development.

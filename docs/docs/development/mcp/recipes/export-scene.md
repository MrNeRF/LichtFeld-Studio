---
sidebar_position: 5
---

# Export Scene

Use this flow when you need to export one or more scene nodes to `PLY`, `SOG`, `SPZ`, `USD`, or the standalone HTML viewer.

## Sequence

1. Read `lichtfeld://scene/nodes` or call `scene_list_nodes`.
2. Choose either a single `node` or a list of `nodes`.
3. Call one of the `scene_export_*` tools.
4. Treat the export as synchronous in the current GUI implementation.

## Export To PLY

```json
{
  "tool": "scene_export_ply",
  "arguments": {
    "path": "/tmp/export.ply",
    "node": "training_model",
    "sh_degree": 3
  }
}
```

Other export entry points:

- `scene_export_sog`
- `scene_export_spz`
- `scene_export_usd`
- `scene_export_html`

## Status And Cancellation

These tools document the current execution model:

```json
{
  "tool": "scene_export_status",
  "arguments": {}
}
```
```json
{
  "tool": "scene_export_cancel",
  "arguments": {}
}
```

In the current GUI implementation:

- exports complete synchronously
- `scene_export_status` reports idle state
- `scene_export_cancel` returns an error because there is nothing cancellable once export starts

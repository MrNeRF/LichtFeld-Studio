---
sidebar_position: 5
---

# Export Scene

Use this flow when you need to export one or more scene nodes to `PLY`, `SOG`, `SPZ`, or the standalone HTML viewer.

## Sequence

1. Read `lichtfeld://scene/nodes` or call `scene.list_nodes`.
2. Choose either a single `node` or a list of `nodes`.
3. Call one of the `scene.export_*` tools.
4. Treat the export as synchronous in the current GUI implementation.

## Export To PLY

```json
{
  "tool": "scene.export_ply",
  "arguments": {
    "path": "/tmp/export.ply",
    "node": "training_model",
    "sh_degree": 3
  }
}
```

Other export entry points:

- `scene.export_sog`
- `scene.export_spz`
- `scene.export_html`

## Status And Cancellation

These tools document the current execution model:

```json
{
  "tool": "scene.export_status",
  "arguments": {}
}
```
```json
{
  "tool": "scene.export_cancel",
  "arguments": {}
}
```

In the current GUI implementation:

- exports complete synchronously
- `scene.export_status` reports idle state
- `scene.export_cancel` returns an error because there is nothing cancellable once export starts

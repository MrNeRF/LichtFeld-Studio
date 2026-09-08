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
- `scene_export_streamed_sog` (directory with `lod-meta.json`)
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

## Streamed SOG

`scene_export_streamed_sog` writes a PlayCanvas multi-LOD directory. `path` is
required and names the output directory. Optional settings are `lod_levels` (4,
1–8), `lod_ratio` (0.5, 0.1–0.9), `chunk_count_k` (512), `chunk_extent` (16),
`chunk_min_k` (8), and `kmeans_iterations` (10). Node selection accepts `node`,
`nodes`, `uuid`, `uuids`, or session-local integer `node_ids`, with the same
selection fallback as SOG. `sh_degree` defaults to 3 and `include_provenance`
to true. The call completes synchronously; `scene_export_status` includes
`streamed_sog` in `supported_formats` and reports idle after completion.

Import either the directory or its `lod-meta.json` with `lf.io.load(path)` in Python.

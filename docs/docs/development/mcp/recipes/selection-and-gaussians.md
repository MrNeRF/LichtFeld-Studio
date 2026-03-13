---
sidebar_position: 2
---

# Selection And Gaussians

Use this flow when the task is to select visible Gaussians, confirm the result, and then inspect or edit the selected data.

## Sequence

1. Read `lichtfeld://selection/current`.
2. Use one of the `selection.*` tools.
3. Confirm with `selection.get` or `lichtfeld://selection/current`.
4. Use `gaussians.read`, `gaussians.write`, `transform.*`, or `operator.invoke` as the next step.

## Rectangle Selection

```json
{
  "tool": "selection.rect",
  "arguments": {
    "x0": 100,
    "y0": 120,
    "x1": 420,
    "y1": 360,
    "mode": "replace",
    "camera_index": 0
  }
}
```

Other supported selection entry points:

- `selection.click`
- `selection.brush`
- `selection.lasso`
- `selection.polygon`
- `selection.ring`
- `selection.by_description`

## Confirm The Result

```json
{
  "tool": "selection.get",
  "arguments": {
    "max_indices": 1024
  }
}
```

## Read Gaussian Fields

```json
{
  "tool": "gaussians.read",
  "arguments": {
    "fields": ["means", "opacities"],
    "limit": 16
  }
}
```

## Write Gaussian Fields

Use this only after confirming the target node, indices, and field width:

```json
{
  "tool": "gaussians.write",
  "arguments": {
    "field": "opacities",
    "indices": [0, 1],
    "values": [0.5, 0.75]
  }
}
```

## Notes

- `mode` is `replace`, `add`, or `remove`
- `selection.by_description` is higher-latency because it uses an LLM vision round-trip
- after writes, re-read `lichtfeld://selection/current` or `gaussians.read` instead of assuming the mutation landed

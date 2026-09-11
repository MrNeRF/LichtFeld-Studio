---
sidebar_position: 1
---

# Load Dataset And Train

Use this flow when the goal is to load a COLMAP dataset or checkpoint and then monitor training through normalized runtime state.

## Sequence

1. Read `lichtfeld://runtime/catalog` to confirm the runtime job ids.
2. Call `scene_load_dataset` or `scene_load_checkpoint`.
3. Read `lichtfeld://scene/state` to confirm the training snapshot.
4. Call `training_start`.
5. Observe `training.main` through `runtime_job_wait`, `runtime_job_describe`, or `runtime_events_tail`.

## Load A Dataset

```json
{
  "tool": "scene_load_dataset",
  "arguments": {
    "path": "/data/colmap/room",
    "images_folder": "images",
    "max_iterations": 30000,
    "strategy": "mcmc"
  }
}
```

Notes:

- `path` is required.
- `images_folder`, `max_iterations`, and `strategy` are optional.
- Dataset import also shows up under the normalized runtime job id `import.dataset`.

## Start Training

```json
{
  "tool": "training_start",
  "arguments": {}
}
```

## Monitor Progress

Wait for a change in the normalized training job:

```json
{
  "tool": "runtime_job_wait",
  "arguments": {
    "job_id": "training.main",
    "until": "changed",
    "timeout_ms": 5000
  }
}
```

Read recent training events without creating a subscription:

```json
{
  "tool": "runtime_events_tail",
  "arguments": {
    "types": [
      "training.progress",
      "training.completed",
      "training.stopped"
    ],
    "max_events": 20
  }
}
```

## State Checks

- `lichtfeld://scene/state` gives the normalized training snapshot
- `training_get_loss_history` or `lichtfeld://training/loss_curve` gives the loss curve
- `lichtfeld://runtime/state` gives a broader runtime summary

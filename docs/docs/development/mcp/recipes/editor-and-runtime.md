---
sidebar_position: 4
---

# Editor And Runtime

Use this flow when the task is to execute Python inside the integrated editor and then observe output through normalized runtime state.

## Sequence

1. Optionally call `editor_set_code`.
2. Call `editor_run`.
3. If the script is still running, watch `editor.python` through `runtime_job_wait`.
4. Read output with `editor_get_output` or `lichtfeld://editor/output`.

## Set Code

```json
{
  "tool": "editor_set_code",
  "arguments": {
    "code": "print('hello from MCP')",
    "show_console": true
  }
}
```

## Run The Script

```json
{
  "tool": "editor_run",
  "arguments": {
    "wait_for_completion": true,
    "wait_for_output": true,
    "timeout_ms": 2000
  }
}
```

Or set and run in one call:

```json
{
  "tool": "editor_run",
  "arguments": {
    "code": "print('hello from MCP')",
    "wait_for_completion": true,
    "wait_for_output": true,
    "timeout_ms": 2000
  }
}
```

## Wait On The Normalized Runtime Job

```json
{
  "tool": "runtime_job_wait",
  "arguments": {
    "job_id": "editor.python",
    "until": "inactive",
    "timeout_ms": 5000
  }
}
```

## Read Output

```json
{
  "tool": "editor_get_output",
  "arguments": {
    "max_chars": 20000,
    "tail": true
  }
}
```

## Notes

- `editor_run` can place code into the editor and execute it in one step
- use `editor_is_running` for a cheap status check
- use `editor_interrupt` if the script must be stopped

## AprilTag calibration

`transform_detect_apriltags` and `transform_scale_by_apriltags` start the
`apriltag.scan` runtime job and return immediately after capturing camera inputs.
Use `runtime_job_describe` or `runtime_job_wait` with `job_id: "apriltag.scan"`
to read progress and the completed `result`. A successful start does not mean
calibration succeeded: check the job's `outcome` and `result.success`.

Cancel with `runtime_job_control` and
`{"job_id": "apriltag.scan", "action": "cancel"}`. Cancellation is cooperative:
in-flight image decoding/detection or solving can finish, but cancellation is
checked again before any scale is applied. Only one AprilTag job runs at a time;
starting another after completion replaces its stored result.

Image detection and solving run in a worker. Scene reads for the snapshot and
final application run on the viewer thread. Results are rejected if scene
identity or transforms changed during scanning. Without `targets`, scaling
covers the complete scene regardless of selection; pass explicit node names
only when partial scaling is intended.

Python's `detect_scene_apriltags` and `scale_scene_by_apriltags` remain synchronous
for background editor scripts. Calls from viewer-thread UI callbacks are
rejected; use the MCP job for an asynchronous scan.

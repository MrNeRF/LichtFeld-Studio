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

`transform_detect_apriltags` and `transform_scale_by_apriltags` start `apriltag.scan`.
Use `runtime_job_describe` or `runtime_job_wait` with `job_id: "apriltag.scan"`
for progress and results; check both `outcome` and `result.success`.
Cancel with `runtime_job_control` and `{"job_id": "apriltag.scan", "action": "cancel"}`.
Cancellation waits for in-flight decoding/detection or solving and prevents application.
Python and MCP share this job. Only one scan runs at a time: a second start returns
a busy error without canceling or queuing behind the active scan. Restarting replaces
the runtime job's stored result; a Python caller keeps its own completion result.

Scaling uses `{tag_a, tag_b, distance}` constraints and covers the complete scene
regardless of selection. Pass `targets` only for partial scaling, or
`apply_to_scene: false` to compute without applying. Scene changes invalidate a scan.
Python's synchronous `detect_scene_apriltags` and `scale_scene_by_apriltags` must
run in a background editor script; viewer-thread UI callbacks are rejected. Python
releases the GIL while waiting for its request to finish. Failure, cancellation,
scene replacement, and shutdown all settle that wait. Canceling the editor script
does not automatically cancel the native scan.

If a requested manifest cannot be written, the scan fails and leaves the scene
unchanged for both Python and MCP. Cancellation can arrive after a manifest has
been written; it prevents scene application but does not remove that file.

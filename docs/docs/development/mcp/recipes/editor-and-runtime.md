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

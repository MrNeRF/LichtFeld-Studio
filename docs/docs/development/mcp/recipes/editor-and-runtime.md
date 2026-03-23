---
sidebar_position: 4
---

# Editor And Runtime

Use this flow when the task is to execute Python inside the integrated editor and then observe output through normalized runtime state.

## Sequence

1. Optionally call `editor.set_code`.
2. Call `editor.run`.
3. If the script is still running, watch `editor.python` through `runtime.job.wait`.
4. Read output with `editor.get_output` or `lichtfeld://editor/output`.

## Set Code

```json
{
  "tool": "editor.set_code",
  "arguments": {
    "code": "print('hello from MCP')",
    "show_console": true
  }
}
```

## Run The Script

```json
{
  "tool": "editor.run",
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
  "tool": "editor.run",
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
  "tool": "runtime.job.wait",
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
  "tool": "editor.get_output",
  "arguments": {
    "max_chars": 20000,
    "tail": true
  }
}
```

## Notes

- `editor.run` can place code into the editor and execute it in one step
- use `editor.is_running` for a cheap status check
- use `editor.interrupt` if the script must be stopped

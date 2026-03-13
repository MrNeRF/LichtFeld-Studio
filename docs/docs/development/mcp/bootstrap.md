---
sidebar_position: 2
---

# Bootstrap

Use this when you want the shortest reliable path from connection to action.

## Minute-One Flow

1. Connect using `.mcp.json` and call `initialize`.
2. Call `resources/list` and `tools/list` once.
3. Read `lichtfeld://runtime/catalog`.
4. Read `lichtfeld://runtime/state`.
5. Read `lichtfeld://ui/state` if the task touches the interactive GUI.
6. Read `lichtfeld://scene/state` and `lichtfeld://selection/current` if the task touches training or Gaussian selection.
7. Only then choose a tool namespace and act.

## Why Start With Resources

Resources are cheaper to reason about than source code:

- they normalize current app state
- they expose stable ids and resource links
- they remove guesswork around active jobs, current tools, and selected objects

For most tasks, resources answer the first two questions an agent needs:

1. What exists right now?
2. Which exact id should I use?

## How To Pick The Next Namespace

- Need dataset import, training, or export progress: start with `runtime.*`
- Need toolbar, menu, or panel actions: start with `ui.*`
- Need operator ids or schemas: start with `operator.*` or `lichtfeld://operators/registry`
- Need visible Gaussians or scene nodes: start with `selection.*`, `gaussians.*`, or `scene.*`
- Need Python execution inside the app: start with `editor.*`

## Minimal Mutation Pattern

Use the same pattern for most tasks:

1. Read the relevant resource.
2. Invoke the tool with explicit arguments.
3. Read back the narrowest state resource.
4. If the action is long-running, switch to `runtime.job.wait` or `runtime.events.tail`.

## Long-Running Jobs

The runtime catalog currently advertises these normalized job ids:

- `training.main`
- `editor.python`
- `import.dataset`
- `export.scene`
- `export.video`
- `operator.modal`

Example wait loop:

```json
{
  "tool": "runtime.job.wait",
  "arguments": {
    "job_id": "training.main",
    "until": "changed",
    "timeout_ms": 5000
  }
}
```

## When To Read Source

Read source only when one of these is true:

- the relevant tool metadata or resource shape is still ambiguous
- you need backend behavior that is intentionally not normalized into MCP
- you are changing the MCP implementation itself

If you do need source, start at the registration site for the namespace you already identified instead of reading the repo top-down.

# Agent Guide

This repository exposes a large local MCP surface for LichtFeld Studio. When the task is to drive the app, inspect runtime state, manipulate the GUI, run training, edit selections, or export data, prefer MCP discovery before reading C++.

## Fast Start

- MCP server config lives in `.mcp.json`.
- Default endpoint: `http://localhost:45677/mcp`.
- Initialize first, then use `tools/list` and `resources/list`.
- Read these resources before guessing tool names or argument shapes:
  1. `lichtfeld://runtime/catalog`
  2. `lichtfeld://runtime/state`
  3. `lichtfeld://ui/state`
  4. `lichtfeld://scene/state`
  5. `lichtfeld://selection/current`

Pull narrower resources only when needed:

- `lichtfeld://ui/tools`
- `lichtfeld://ui/menus`
- `lichtfeld://ui/panels`
- `lichtfeld://operators/registry`
- `lichtfeld://operators/modal_state`
- `lichtfeld://scene/nodes`
- `lichtfeld://scene/selected_nodes`
- `lichtfeld://runtime/jobs/<job_id>`
- `lichtfeld://runtime/events/<event_type>`

## Operating Rules

- Prefer resources for discovery and current state; use tools for mutations.
- Check tool metadata before invoking: `category`, `kind`, `runtime`, `thread_affinity`, `destructive`, `long_running`, `user_visible`.
- For long-running operations, pair the mutating call with `runtime.job.describe`, `runtime.job.wait`, or `runtime.events.tail` instead of sleeping.
- Do not guess operator ids, menu paths, panel ids, or UI tool ids. Read `operators/registry`, `ui/tools`, or `ui/menus` first.
- After selection writes, confirm with `selection.get` or `lichtfeld://selection/current`.
- Scene exports are synchronous in the current GUI implementation. `scene.export_status` reports idle semantics and `scene.export_cancel` cannot stop an export once started.
- `editor.run` can wait inline. For longer scripts, use `editor.is_running`, `editor.wait`, and `editor.get_output`.

## Important Runtime Job IDs

- `training.main`
- `editor.python`
- `import.dataset`
- `export.scene`
- `export.video`
- `operator.modal`

## Default Playbooks

### Load Dataset And Train

- `scene.load_dataset` or `scene.load_checkpoint`
- read `lichtfeld://scene/state`
- `training.start`
- watch `training.main` through `runtime.job.*` or `runtime.events.tail`

### Manipulate UI Or Operators

- read `lichtfeld://ui/state`
- inspect `lichtfeld://ui/tools`, `lichtfeld://ui/menus`, or `lichtfeld://operators/registry`
- call `ui.*` or `operator.*`
- if the tool goes modal, watch `operator.modal` and `lichtfeld://operators/modal_state`

### Work On Gaussian Selection

- use `selection.rect`, `selection.click`, `selection.brush`, `selection.lasso`, or `selection.by_description`
- verify via `selection.get` or `lichtfeld://selection/current`
- then call `gaussians.read`, `gaussians.write`, `transform.*`, `scene.*`, or `operator.invoke`

## Docs

See `docs/docs/development/mcp/` for the repo-local MCP guide and workflow recipes.

# Agent Guide

This repository exposes a large local MCP surface for LichtFeld Studio. When the task is to drive the app, inspect runtime state, manipulate the GUI, run training, edit selections, or export data, prefer MCP discovery before reading C++.

## Fast Start

- MCP server config lives in `.mcp.json`.
- Repo MCP config launches `scripts/lichtfeld_mcp_bridge.py`, which starts LichtFeld Studio and then proxies MCP over the local HTTP endpoint.
- Default HTTP endpoint after startup: `http://localhost:45677/mcp`.
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

- Prefer the bridge-backed `.mcp.json` entry when you want Codex to start LichtFeld for you. Use the raw HTTP endpoint only if LichtFeld is already running.
- Prefer resources for discovery and current state; use tools for mutations.
- Check tool metadata before invoking. `annotations` contains only the standard MCP hints `readOnlyHint`, `destructiveHint`, and `idempotentHint`. LichtFeld metadata is under `_meta`, using keys such as `app.lichtfeld/category`, `app.lichtfeld/kind`, `app.lichtfeld/runtime`, `app.lichtfeld/thread_affinity`, `app.lichtfeld/long_running`, and `app.lichtfeld/user_visible`.
- For long-running operations, pair the mutating call with `runtime_job_describe`, `runtime_job_wait`, or `runtime_events_tail` instead of sleeping.
- Do not guess operator ids, menu paths, panel ids, or UI tool ids. Read `operators/registry`, `ui/tools`, or `ui/menus` first.
- After selection writes, confirm with `selection_get` or `lichtfeld://selection/current`.
- Scene exports are synchronous in the current GUI implementation. `scene_export_status` reports idle semantics and `scene_export_cancel` cannot stop an export once started.
- `editor_run` can wait inline. For longer scripts, use `editor_is_running`, `editor_wait`, and `editor_get_output`.

## Important Runtime Job IDs

- `training.main`
- `editor.python`
- `import.dataset`
- `export.scene`
- `export.video`
- `operator.modal`

## Default Playbooks

### Load Dataset And Train

- `scene_load_dataset` or `scene_load_checkpoint`
- read `lichtfeld://scene/state`
- `training_start`
- watch `training.main` through `runtime_job_*` or `runtime_events_tail`

### Manipulate UI Or Operators

- read `lichtfeld://ui/state`
- inspect `lichtfeld://ui/tools`, `lichtfeld://ui/menus`, or `lichtfeld://operators/registry`
- call `ui_*` or `operator_*`
- if the tool goes modal, watch `operator.modal` and `lichtfeld://operators/modal_state`

### Work On Gaussian Selection

- use `selection_rect`, `selection_click`, `selection_brush`, `selection_lasso`, or `selection_by_description`
- verify via `selection_get` or `lichtfeld://selection/current`
- then call `gaussians_read`, `gaussians_write`, `transform_*`, `scene_*`, or `operator_invoke`

## Docs

See `docs/docs/development/mcp/` for the repo-local MCP guide and workflow recipes.

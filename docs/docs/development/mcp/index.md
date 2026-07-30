---
sidebar_position: 1
---

# MCP Guide

This branch exposes a broad MCP surface for LichtFeld Studio. The goal of this section is not to repeat every tool signature. It is to make the surface fast to navigate so an agent can discover state, pick the right namespace, and execute the shortest safe sequence for a task.

## Working Contract

- Use MCP resources for discovery and current state.
- Use MCP tools for mutations and long-running actions.
- Read state before acting.
- Prefer targeted resources over code reading.
- Fall back to source only when the resource and tool metadata still leave ambiguity.

## Tool Metadata

In `tools/list`, `annotations` contains only the standard MCP hints `readOnlyHint`, `destructiveHint`, and `idempotentHint`. LichtFeld-specific metadata lives under `_meta` with namespaced keys: `app.lichtfeld/category`, `app.lichtfeld/kind`, `app.lichtfeld/runtime`, `app.lichtfeld/thread_affinity`, `app.lichtfeld/long_running`, and `app.lichtfeld/user_visible`.

## First Resources To Read

Read these first in most sessions:

1. `lichtfeld://runtime/catalog`
2. `lichtfeld://runtime/state`
3. `lichtfeld://ui/state`
4. `lichtfeld://scene/state`
5. `lichtfeld://selection/current`
6. `lichtfeld://history/state`

Then narrow further:

- `lichtfeld://ui/tools` for tool ids, active state, and availability
- `lichtfeld://ui/menus` for menu trees and invokable actions
- `lichtfeld://ui/panels` for panel ids and registry state
- `lichtfeld://operators/registry` for operator ids, flags, and input schemas
- `lichtfeld://scene/nodes` and `lichtfeld://scene/selected_nodes` for scene manipulation
- `lichtfeld://history/state` and `lichtfeld://history/stack` for undo/redo inspection
- `lichtfeld://runtime/jobs/<job_id>` and `lichtfeld://runtime/events/<event_type>` for long-running work

## Main Namespaces

| Namespace | Use |
| --- | --- |
| `scene_*` | Load datasets, checkpoints, scene nodes, exports |
| `training_*` | Training state, loss, and training control |
| `runtime_*` | Normalized job and event tracking |
| `ui_tool_*`, `ui_menu_*`, `ui_panel_*`, `ui_operator_*` | Drive the registered GUI surface |
| `operator_*` | Introspect and invoke registered GUI operators, including modal flows |
| `selection_*` | Screen-space Gaussian selection |
| `transform_*` | Node transform inspection and edits |
| `gaussians_*` | Raw Gaussian tensor reads and writes |
| `history_*` | Shared undo/redo inspection, playback, and grouped transactions |
| `editor_*` | Integrated Python console execution |
| `camera_*`, `render_*`, `sequencer_*` | View state, captures, and timeline workflows |

## Runtime Model

`lichtfeld://runtime/catalog` is the bootstrap resource for long-running work. It advertises normalized job ids and event types, including:

- `training.main`
- `editor.python`
- `import.dataset`
- `export.scene`
- `export.video`
- `operator.modal`

When a tool can block or keep running, use the runtime APIs instead of sleeping:

- `runtime_job_list`
- `runtime_job_describe`
- `runtime_job_wait`
- `runtime_job_control`
- `runtime_events_tail`

## Recommended Reading Order

- [Connecting MCP Clients](connecting-clients.md) for client setup (Claude Desktop, Claude Code, in-repo agents)
- [Bootstrap](bootstrap.md) for the discovery-first workflow
- [Recipes](recipes/) for concrete task sequences

## What Not To Do

- Do not guess operator ids or modal event shapes.
- Do not scan the entire source tree before checking `resources/list`, `tools/list`, and the bootstrap resources.
- Do not poll blindly when `runtime_job_wait` or `runtime_events_tail` can tell you what changed.
- Do not assume exports are asynchronous in the current GUI implementation.

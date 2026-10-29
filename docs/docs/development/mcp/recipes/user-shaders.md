---
sidebar_position: 8
---

# User shaders

Initialize MCP, discover `tools/list`, then read `lichtfeld://shaders/catalog`,
`lichtfeld://shaders/state` and `lichtfeld://scene/nodes`. Registration names are
`shader.list`, `shader.get`, `shader.set_source`, `shader.bind`, `shader.unbind`,
`shader.set_param`, and `shader.remove`. MCP transport exposes these as
`shader_list`, `shader_get`, `shader_set_source`, `shader_bind`, `shader_unbind`,
`shader_set_param`, and `shader_remove`; use the names returned by discovery.

## Create, bind and observe

```json
{"tool":"shader_set_source","arguments":{"name":"Identity","source":"import lfs; LfsElement effect(LfsElement e, LfsContext ctx) { return e; }"}}
```

Use the returned `shader_id` with `shader_bind` and `{"target":{"kind":"global"}}`.
For a node, use `{"kind":"node","node_uuid":"<splat node UUID>"}`. Global and node
effects run in that order.

Read `lichtfeld://shaders/assets/<shader_id>` or call `shader_get` to observe source,
bindings and diagnostics. Shader linking is driven by visible draws, so an asset
can remain pending with no scene or no applicable draw. There is no `shader.compile`
or `shader.cancel` tool. Discover the `shader.compile` **runtime job** through the
catalog and wait with `runtime_job_wait`; a job becoming inactive does not itself
prove that the requested revision was displayed. Check the asset status.

For the parameterized example in the [user guide](../../../features/user-shaders.md),
call `shader_set_param` with `shader_id`, `name:"strength"`, `value:0.75` after
successful reflection. Supply `expected_revision` when an edit must reject a stale
source revision. Python offers `lf.shaders.list/get/set_source/bind/unbind/set_param/remove/status/wait`;
`wait` also accepts `source_revision` and `timeout_ms` (0–30000).

## Editor documents

Create a tab with `editor_new_document` and `{"mode":"shader","code":"..."}`.
Use its `document_id` for `editor_set_target`, `editor_set_code`, `editor_run` and
`editor_wait`. A Shader Run updates an asset and its target; it never runs as Python.
`editor_interrupt` releases remembered Run targets only while they still hold the
same asset UUID. A later rebind to that same asset is released; a different asset is kept.

`editor_wait` reports outcomes such as `applied`, `failed`, `unbound`, `timeout`,
`superseded` and `document_closed`. Inspect the outcome and requested revision;
a completed response can describe a failed compile. An active Python tab uses
`editor.python`; use `editor_wait` with document identity for a Shader tab.

## Output

`ctx.time` follows scene time and export frames, not elapsed wall time. Effects
apply only to splats. Rendered captures/video include effects; geometry exports
do not bake them.

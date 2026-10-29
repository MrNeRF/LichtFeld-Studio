---
sidebar_position: 12
---

# User shaders

User shaders change how splats look in the viewer and rendered images or video.
They do not modify the stored Gaussian tensors, training, meshes, or point clouds.
Open the editor, choose **Shader**, enter Slang source, choose a target, and Run.
Stop unbinds the document's Run targets while they still hold its asset UUID,
including a later rebind to the same asset; bindings to a different asset are kept.
A minimal shader leaves each splat unchanged:

```slang
import lfs;
LfsElement effect(LfsElement e, LfsContext ctx) {
    return e;
}
```

## Parameters and targets

Expose editable parameters with `LfsParam(min, max, initial)`:

```slang
import lfs;
struct Params {
    [LfsParam(0, 1, 0.5)] float strength;
};
LfsElement effect(LfsElement e, LfsContext ctx, Params p) {
    e.color *= p.strength;
    return e;
}
```

The Python editor uses the same assets and parameters:

```python
import lichtfeld as lf
source = """import lfs;
struct Params { [LfsParam(0, 1, 0.5)] float strength; };
LfsElement effect(LfsElement e, LfsContext ctx, Params p) {
    e.color *= p.strength; return e;
}"""
asset = lf.shaders.set_source(source, name="Dimmer")["shader_id"]
lf.shaders.bind(asset, {"kind": "global"})
state = lf.shaders.wait(asset, timeout_ms=30000)
# Compilation needs a visible splat draw. A timeout is not a successful apply.
if state["outcome"] == "applied":
    lf.shaders.set_param(asset, "strength", 0.75)
```

Execution order is **global → node → Custom reveal**. A node target is
`{"kind": "node", "node_uuid": node_uuid}`; use a splat node UUID from the scene.
A Custom reveal uses the sequencer clip:

```python
lf.ui.set_reveal_clip({"start": 0.0, "end": 3.0, "softness": 0.2,
                       "effect": 12, "shader_id": asset})
```

The reveal target in the Shader editor controls this same clip. Removing or
unbinding its shader removes the Custom clip through the normal undoable action.
Unbind other targets with `lf.shaders.unbind(target)` before removing an asset
with `lf.shaders.remove(asset)`. At most eight distinct assets can be bound at once.

## Time, compilation and recovery

`ctx.time` is scene timeline time in seconds. It advances during playback and
follows scrubbing and export frame time; it is not a wall clock and does not
keep an idle viewport rendering. `ctx.progress` is clamped reveal progress,
or 1 without a reveal; `ctx.softness` is the reveal edge width, or 0 without one.

Compilation follows draw demand. Saving an unbound asset does not compile it.
Read `lf.shaders.status(asset)` for diagnostics, pending hosts, and the displayed
revision. A failed edit keeps the last successfully displayed program in the
current session. A fresh project load has no cached program to fall back to:
an unavailable shader contributes no effect. A Custom clip stays in the timeline
when its source fails to compile or its language is unknown.

Projects preserve assets, bindings, named parameter overrides, Shader tabs and
drafts. Source matching an associated asset is stored once. Unapplied draft text
is stored separately; reopening does not restore Run ownership. Overrides survive
saving before reflection and migrate by matching parameter name and component
count. Damaged entries produce a warning; a missing or structurally rejected
Custom asset removes only that clip, leaving other tracks intact.

## Saving and sharing

Shader edits participate in project Save, Save As, autosave, Discard and undo.
A project with a shader library opens read-only in builds without shader support.
A Custom clip requires a build with shader support even to open. Removing the
library, bindings and Custom clip removes those requirements on the next save.
Shader drafts without assets use the existing editor-field preservation contract.

Rendered images and video include supported shader effects. Video keeps the
composition selected for that export while you edit the live scene. Geometry
exports do not bake these effects into splat tensors. Portable and gallery
publication omit shader assets, bindings, Shader drafts and Custom reveal clips.
Direct publication shows a completion notice; gallery review shows a warning.
Keep the original project to retain the editable shaders and their source.

See the [MCP recipe](../development/mcp/recipes/user-shaders.md) for automation.

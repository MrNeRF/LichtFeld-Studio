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

Execution order is **global → node**. A node target is
`{"kind": "node", "node_uuid": node_uuid}`; use a splat node UUID from the scene.
Unbind targets with `lf.shaders.unbind(target)` before removing an asset
with `lf.shaders.remove(asset)`. At most eight distinct assets can be bound at once.

## Time, compilation and recovery

`ctx.time` is scene timeline time in seconds. It advances during playback and
follows scrubbing and export frame time; it is not a wall clock and does not
keep an idle viewport rendering. Global and node effects read `ctx.progress`
as 1 and `ctx.softness` as 0.

Compilation follows draw demand. Saving an unbound asset does not compile it.
Read `lf.shaders.status(asset)` for diagnostics, pending hosts, and the displayed
revision. A failed edit keeps the last successfully displayed program in the
current session.

## Rendered output

Shader edits participate in undo. Rendered images and video include supported
shader effects. Video keeps the composition selected for that export while you
edit the live scene. Geometry exports do not bake these effects into splat tensors.

See the [MCP recipe](../development/mcp/recipes/user-shaders.md) for automation.

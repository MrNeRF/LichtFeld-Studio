# Python nodes

Modifier callbacks run on the evaluation worker with the Python GIL and a
dedicated tensor work queue. Treat input geometry as immutable: return replaced
components instead of modifying tensors in place. Keep callbacks computational;
scene/UI mutations belong on the viewer thread, outside node evaluation.
The viewport keeps its previous result while a request is running. Scrubbing
coalesces queued requests and retains upstream node caches.

`lf.nodes.evaluate(node_name)` explicitly waits for dirty work. Reading
`lf.nodes.evaluated(node_name)` returns the last installed result without starting
an evaluation. For profiling, `lf.nodes.performance(reset=True)` begins a sample
window; `lf.nodes.performance()` reads counters, per-node runs and frame/latency
samples without resetting them.

Python plug-ins can register node types through `lichtfeld.nodes`.
Node classes declare `inputs = [...]`, `outputs = [...]`, and `properties = [...]`
lists and implement `execute(self, ctx)`. Each declaration needs an explicit
identifier and type. Attribute-style declarations and `evaluate` callbacks are
not supported: using the same attribute for an input and output silently loses
one declaration in Python. Lists allow both to be named `Geometry` safely.
Exceptions are attached to the failing node and do
not stop evaluation of unrelated modifier stacks.

```python
import lichtfeld as lf

class PassThrough(lf.nodes.Node):
    id = "example.pass_through"
    label = "Pass Through"
    category = "Utilities"
    description = "Passes your geometry through without changing it."
    help = "Use while testing a graph connection.\nConnect Geometry at both ends."
    inputs = [lf.nodes.Input("Geometry", "geometry", description="Geometry to pass through; connect a source.")]
    outputs = [lf.nodes.Output("Geometry", "geometry", description="The unchanged input geometry.")]

    def execute(self, ctx):
        return {"Geometry": ctx.input("Geometry")}

lf.nodes.register_node(PassThrough)
```

`Input`, `Output` and `Property` accept an optional `description` keyword.
Write what the control does, its units/range and what its default means.
The editor shows these strings on hover and when an inspector control has focus.
`Node.description` is a short sentence and `Node.help` is optional multiline
plain text, shown in the Add preview and the inspector's How to use block.
Plug-ins provide their own strings (or use their own translation catalogue);
their identifiers are never translated.

The [node reference](nodes/index.md) is generated from `lf.nodes.node_types()`.
Run `tools/generate_node_reference.py` in the application for the complete
catalogue, including Object Info. An English JSON export can also be passed
with `--descriptors`; `--check` detects stale pages without writing them.

Use `new_tree`, `NodeTree.add_node`, and `NodeTree.link` to construct a node graph.
The default graph name is "Node Graph"; duplicate names gain a numeric suffix
("Node Graph 2", "Node Graph 3") on creation, rename, or import. Open the Node Editor with
`lf.ui.screen.open_editor("node_editor")`.
`evaluate_tree(tree, geometry)` evaluates without a running application, which
is useful for plug-in tests. In the application, `add_modifier` attaches the
graph to a scene node; `evaluated` reads the derived result and
`apply_modifier` bakes it into stored geometry as one undo operation.

See `docs/plugins/examples/node_posterize.py` for a complete plug-in lifecycle.

## Animated inputs

Unlinked number, vector and colour inputs use the sequencer's animation tracks.
Click ◇ beside a value, Ctrl+click the value, or hover it and press `I` to insert
a keyframe at the playhead. ◆ marks an exact keyframe; ◇ marks an interpolated
value between keys. Editing an animated value updates its key at the playhead.
Tracks appear beneath the camera track in the sequencer; clicking a diamond
seeks to that key's time. Times are in seconds.

```python
radius = tree.add_node("lfs.compare", "Reveal radius")
radius.keyframe_insert("B", time=0.0, value=0.2, easing=3)
radius.keyframe_insert("B", time=2.0, value=8.0)
# easing: 0 linear, 1 ease in, 2 ease out, 3 ease in/out (sequencer cubic easing)
radius.keyframe_remove("B", time=2.0)
```

Omitting time uses the playhead; omitting value captures the displayed value.
The MCP equivalents are `nodes.keyframe_set` and `nodes.keyframe_remove`, with
`tree`, `node`, `input`, and optional `time`/`value`. Use the existing sequencer
tools to scrub or play. Assign `node.name` to rename a node without losing its
tracks. Rename, copy/paste, grouping and ungrouping remap the stable targets
`nodes/<tree uuid>/<node name>/<input identifier>`.

Scene Time provides Seconds and Frame without keyframes. Only time-dependent
modifier stacks and their Object Info consumers are submitted on a scrub;
queued work is latest-wins. Video export waits for each frame's evaluation
before capturing geometry. Project files keep tracks in the existing sequencer
chapter, while graph JSON and the clipboard include the corresponding tracks.
For headless tests, use `evaluate_tree(tree, geometry, time=0.5, device="cpu")`
or `device="gpu"`; leaving time out uses the playhead (initially zero headless).
Posterize quantizes clamped base RGB, not the raw SH DC coefficient. Its Selection
field blends the quantized colour and fades selected view-dependent SH coefficients.

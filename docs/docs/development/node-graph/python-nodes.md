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
Node classes declare inputs, outputs, and properties as class attributes and
implement `evaluate(ctx)`. Exceptions are attached to the failing node and do
not stop evaluation of unrelated modifier stacks.

```python
import lichtfeld as lf

class PassThrough(lf.nodes.Node):
    id = "example.pass_through"
    label = "Pass Through"
    category = "Utilities"
    inputs = [lf.nodes.Input("Geometry", "geometry")]
    outputs = [lf.nodes.Output("Geometry", "geometry")]

    def evaluate(self, ctx):
        return {"Geometry": ctx.input("Geometry")}

lf.nodes.register_node(PassThrough)
```

Use `new_tree`, `NodeTree.add_node`, and `NodeTree.link` to construct a node graph.
The default graph name is "Node Graph". Open the Node Editor with
`lf.ui.screen.open_editor("node_editor")`.
`evaluate_tree(tree, geometry)` evaluates without a running application, which
is useful for plug-in tests. In the application, `add_modifier` attaches the
graph to a scene node; `evaluated` reads the derived result and
`apply_modifier` bakes it into stored geometry as one undo operation.

See `docs/plugins/examples/node_posterize.py` for a complete plug-in lifecycle.

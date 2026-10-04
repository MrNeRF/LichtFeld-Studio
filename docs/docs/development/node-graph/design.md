# Node Modifiers — Design (v1)

Status: implemented. Later tree types (for example a
pipeline tree: camera path → video diffusion → train) must fit without
rewriting anything here.

## 1. Goals and non-goals

Goals
- Non-destructive, node-based editing of scene geometry: Gaussian splats,
  point clouds and meshes.
- Modifier model: a **modifier stack per scene node**; each modifier
  references a **node group** (a node graph stored once per project and
  shareable between modifiers). The viewport shows the **evaluated** result;
  the stored payload is never touched until the user **applies** a modifier.
- Masks/selections are **fields**: deferred per-element functions evaluated
  against the geometry of the node that consumes them (no mask/geometry
  length mismatch is possible).
- Backend neutral: every built-in node uses only the public `lfs` tensor API
  (CUDA, Metal, Vulkan). `python3 tools/check_backend_neutrality.py` and
  `python3 tools/error_debt_census.py --baseline tools/error_debt_baseline.json`
  must stay clean.
- Extensible: node types, socket types and tree types live in registries.
  Built-in nodes are C++; plugins add node types in Python through
  `lichtfeld.nodes`, against `lf.Tensor` only.
- Edit mode only (scene content type `SplatFiles`). While a dataset/training
  scene is loaded the editor shows a notice and nothing evaluates.

Non-goals for v1
- Pipeline trees, nested node groups (group
  nodes inside trees), node frames/reroutes, anonymous attribute outputs,
  connected-components clean-up, interface editing UI (interface inputs are
  editable from Python only).

## 2. Layering

```
src/core/nodes/            (part of lfs_core, namespace lfs::nodes)
  types       Geometry, components, Value, Field, SocketType ids
  registry    TreeTypeRegistry, SocketTypeRegistry, NodeTypeRegistry
  tree        NodeTree (nodes, links, interface), JSON (de)serialisation,
              validation, implicit conversions
  evaluator   lazy topological evaluation, per-modifier output cache
  builtin     built-in node implementations (pure tensor code)
src/visualizer/nodes/      ModifierManager (stacks per scene node, scene
                           binding, undo, persistence glue, Object Info /
                           Mesh-to-Splats nodes that need the scene or the
                           renderer), node editor UI
src/python/lfs/py_nodes.*  lichtfeld.nodes bindings
src/python/lfs_plugins/    built-in Python node(s), example
```

Nothing in `src/core/nodes` knows about the scene, the GUI or Python.
Host-dependent nodes (Object Info, Mesh to Splats) are registered by the
visualizer through the same public registry a plugin uses.

## 3. Data model

### 3.1 Geometry

```cpp
struct SplatsComponent {          // raw 3DGS values, float32, device tensors
    Tensor means;                 // [N,3]
    Tensor sh0;                   // [N,3]   DC term
    Tensor shN;                   // [N,K,3] canonical layout, K = (d+1)^2-1
    Tensor scaling;               // [N,3]   log scale
    Tensor rotation;              // [N,4]   quaternion w x y z
    Tensor opacity;               // [N]     logit
    int sh_degree = 0;
    float scene_scale = 1.f;
    AttributeMap attributes;      // name -> Tensor [N] or [N,C]
};
struct PointsComponent { Tensor positions; /*[N,3]*/ Tensor colors; /*[N,3] f32 0..1*/ AttributeMap attributes; };
struct MeshComponent   { std::shared_ptr<const MeshData> mesh; };
struct Geometry { std::optional<SplatsComponent> splats; std::optional<PointsComponent> points; std::optional<MeshComponent> mesh; };
```

- Tensors are reference counted; copying a component is shallow. **Nodes
  never write into input tensors**; they build new tensors and share every
  untouched one (copy-on-write per field).
- Conversion from `SplatData` drops soft-deleted Gaussians and reads `shN`
  through `shN_canonical()`. Conversion back uses the `SplatData`
  constructor plus `shN_set_from_canonical()`.
- Every structural operation (delete, separate, join) carries
  `attributes` along with the built-in arrays.
- Splat log-scales may be finite or `-inf` (an exactly collapsed, zero-size
  axis), never NaN or `+inf`. Scale edits preserve collapsed axes unless Set
  Scale explicitly replaces them with positive sizes at full selection.
  Simplify preserves zero covariance eigenvalues; merging different positions
  or orientations may create genuine extent. Instance on Points accepts signed
  multipliers: magnitudes set size and signs mirror positions, orientation and
  SH colour. A singular instance transform retains its input SH colour.

### 3.2 Domains and fields

Domain of a component: `Splat`, `Point`, `Vertex` (mesh vertices).

A field is an immutable expression DAG (`std::shared_ptr<const FieldNode>`).
`evaluate(FieldContext)` returns a tensor of length N of the context
component (`[N]` for float/int/bool, `[N,3]` for vector/colour). Constants are
fields with no context dependency and are constant-folded. Field results are
memoised per evaluation (key: field node identity + context identity), so a
field used by two consumers on the same geometry is computed once.

Context reads:

| Field | Splat | Point | Vertex |
|---|---|---|---|
| Position | means | positions | vertices |
| Colour | 0.5 + C0·sh0 (unclamped) | colors | vertex colours rgb, else 1 |
| Opacity | sigmoid(opacity) | 1 | vertex colour alpha, else 1 |
| Scale | exp(scaling) [N,3] | 0 | 0 |
| Index | arange(N) | arange(N) | arange(N) |
| Named Attribute | attributes[name] | attributes[name] | — |

### 3.3 Sockets and values

Socket type ids (namespaced): `lfs.geometry`, `lfs.float`, `lfs.int`,
`lfs.bool`, `lfs.vector`, `lfs.colour`, `lfs.string`. Value sockets carry
fields. An input socket declares a default value, optional min/max/step,
whether it accepts fields (`field: true`) or needs a single value, and
`multi_input` (Join Geometry). Implicit conversions:
float↔int (truncate), float/int→bool (`!= 0`), bool→float (0/1),
float→vector/colour (broadcast), vector↔colour (identity),
vector/colour→float (mean). Geometry never converts.

**Selection convention**: selection inputs are float fields ("Selection",
default 1.0, bool links convert). Attribute writers blend by
`w = clamp(selection, 0, 1)`: `new = old·(1-w) + value·w`. Structural nodes
treat `selection >= 0.5` as selected.

### 3.4 Node types

```
NodeTypeInfo {
  id            "lfs.set_colour" (built-ins), "<plugin>.<name>" (plugins)
  label, category, description, help, version
  tree_types    {"lfs.geometry"}
  inputs/outputs  SocketDecl{identifier, label, type, default, min, max,
                             step, field, multi_input, hide_value, description}
  properties    PropertyDecl{identifier, label, kind (enum|string|int|float|bool|data),
                             default, items, min, max, description}
  evaluate      fn(NodeContext&) -> void       (C++ or Python)
  upgrade       optional fn(json params, int from_version) -> json
}
```

The editor UI and the property panels are generated from these
descriptors; no per-node UI code exists. `lichtfeld.nodes.node_types()`
returns them as JSON.

Built-in English text lives in the `nodes` catalogue in `locales/en.json`,
embedded into core at build time for headless use. Presentation uses cached
localised descriptor copies, invalidated by the language generation; evaluation
retains untranslated descriptors and exact identifiers. Plug-ins supply their
own strings. `description` is one scene-focused sentence; `help` is optional
plain text with short usage, starting-value and caveat lines.

### 3.5 Trees, groups, modifiers

- `NodeTree { uuid, name, tree_type, nodes[], links[], interface{inputs[], outputs[]} }`.
  A geometry tree always has exactly one Group Input and one Group Output
  node; interface input 0 and output 0 are `Geometry`.
- `Node { name (unique in tree), type_id, location, muted, input_values{identifier: value},
  properties{identifier: value} }`. Unknown keys are dropped and missing
  keys defaulted on load. A node whose type is not registered loads as a
  **missing-type placeholder**: it keeps its data and links, shows an error,
  and fails evaluation of its branch with that error (it never crashes and
  never loses data).
- Links: `{from_node, from_socket, to_node, to_socket}`. Validation rejects
  cycles, type-incompatible links, and a second link into a
  non-multi-input socket (the new link replaces the old one).
- Muted node: outputs pass through the first input of the same type, else
  the output's default.
- Project library: node graphs are stored once per project; any number of
  modifiers can reference one tree.
- `Modifier { name, tree_uuid, enabled, show_viewport, input_overrides{identifier: value},
  stored_selections{node_name: bitmask over the host's stored elements} }`.
- `ModifierStack` per scene node UUID: modifiers evaluate in order, each
  feeding the next.

### 3.6 Evaluation

- Lazy, topological from Group Output; only nodes reachable from it run.
- Errors are per node (`NodeError` with a user-facing message). A failing
  node makes the stack's result "failed": the viewport then shows the
  **stored** payload (unmodified) and the editor marks the node red.
- Cache: per modifier, per node output, keyed by a hash of
  (type id, version, properties, input values, upstream keys, input
  geometry generation). Editing one node re-runs only that node and its
  downstream nodes. Cached outputs share tensors, so cost is per changed
  field.
- Evaluation runs on a background worker with its own tensor work queue:
  the newest request wins, stale results are discarded, and the viewer
  thread installs only fence-complete results (see "Evaluation and interaction scheduling" below). View-only edits
  (pan, zoom, select, move) never evaluate. Python nodes take the GIL on the
  worker. Each node records its last evaluation time.

### 3.7 Scene binding (evaluated payloads)

- `SceneNode` keeps its stored payload (`model`, `point_cloud`, `mesh`)
  untouched. It gains a derived, non-persisted **evaluated payload**
  (`std::shared_ptr<SplatData> evaluated_model`,
  `std::shared_ptr<PointCloud> evaluated_point_cloud`,
  `std::shared_ptr<MeshData> evaluated_mesh`), set and cleared by the
  ModifierManager through Scene API (`setNodeEvaluatedPayload`,
  `clearNodeEvaluatedPayload`), which bumps the render generation.
- Every Scene accessor that feeds **display, picking, selection index space
  and export** uses the effective payload (evaluated if present, else
  stored): combined model (single-node alias and concatenation, sync and
  worker paths), visible splat node slots, `snapshotVisibleSplats`,
  `getVisibleMeshes`, selection capacity per node, visible SH degrees,
  point cloud collection. Nodes with an evaluated payload are excluded from
  consolidation.
- Undo snapshots and project persistence keep using the stored payload,
  so saving, loading and undo need no special cases.
- Type rule in v1: a SPLAT node displays the result's splats, a MESH node
  the mesh, a POINTCLOUD node the points. Converting between kinds happens
  inside a graph (Object Info + Mesh to Splats, Points to Splats, …).
- Selection-index edits (delete selected, cut, mirror selection, …) on a
  node that currently shows an evaluated payload are refused with a
  message ("Apply or hide the node modifiers to edit the stored splats").
  Edits that change the stored payload by other means (crop, transform
  bake) trigger re-evaluation.
- Stored Selection: the user hides the modifiers (show_viewport off),
  selects on the stored data, and captures the selection into a Stored
  Selection node. The capture is persisted with the modifier.
- Apply: bakes the result of the stack up to and including that modifier
  into the stored payload (one undo entry) and removes those modifiers.

### 3.8 Persistence, undo

- Project chapter `NODE` (singleton JSON):
  `{schema_version:1, trees:[…], stacks:{<node uuid>:[modifier…]}}`.
- Every graph or stack edit (from UI or Python) is one undo entry storing
  before/after JSON; consecutive edits of the same value merge.

## 4. Built-in node catalogue (v1)

Categories: Input, Output, Geometry, Splat, Colour, Selection, Utilities,
Conversion, Clean-up. All selection inputs follow §3.3.

Input / output
- **Group Input**, **Group Output**.
- **Object Info** (visualizer): property `object` (scene node name),
  `transform_space` Original|Relative. Output Geometry: the target's
  effective geometry; Relative maps it into the host node's local space.
  Reference cycles between stacks are an error.
- **Value** (float), **Integer**, **Boolean**, **Vector**, **Colour**.
- Field inputs: **Position**, **Colour Attribute** (named "Colour"),
  **Opacity**, **Scale**, **Index**, **Named Attribute** (name, type),
  **Random Value** (min, max, seed; bit-identical across backends for a given
  element index and seed), **Stored Selection** (captured bitmask; bool field;
  invert property).

Utilities (field operations)
- **Math** (add, subtract, multiply, divide, power, minimum, maximum,
  absolute, sqrt, floor, fraction, sine, cosine, greater than, less than,
  clamp), **Vector Math** (add, subtract, multiply, scale, length, distance,
  dot, normalise), **Compare** (float: <, <=, >, >=, ==, != with epsilon),
  **Boolean Math** (and, or, not, xor), **Map Range** (linear, clamp
  option), **Separate XYZ**, **Combine XYZ**, **Separate Colour** /
  **Combine Colour** (RGB or HSV), **Mix Colour** (mix, multiply, add,
  subtract; factor field).

Selection (field producers)
- **Box Selection** (centre, size, rotation in degrees XYZ, falloff),
  **Ellipsoid Selection** (centre, radii, rotation, falloff),
  **Colour Key** (colour, tolerance, softness),
  **HSV Range** (hue, hue range, hue softness, saturation min/max,
  value min/max, softness),
  **Inside Mesh** (Geometry input carrying a mesh; ray-parity test; tensor
  ops chunked over points × triangles with a bounding-box pre-filter),
  **Neighbour Count** (exact Euclidean radius counts, excluding the point itself;
  a sparse spatial hash keeps scratch independent of scene extent).

Geometry
- **Transform Geometry** (translation, rotation degrees XYZ, uniform scale;
  splats through core `transform`, which rotates SH; points and meshes
  transform positions/normals).
- **Set Position** (selection, position, offset).
- **Delete Geometry** (selection; splats/points remove elements; meshes
  remove every face that uses a selected vertex).
- **Separate Geometry** (selection → Selection, Inverted).
- **Join Geometry** (multi-input; splats with different SH degrees are
  padded to the highest degree with zeros; meshes merge with index and
  material offsets).

Splat
- **Set Colour** (selection, colour; property "clear view-dependent colour"
  default on → shN scaled by 1-w).
- **Set Opacity** (selection, opacity 0..1).
- **Set Scale** (selection, scale vector, activated units).
- **Set SH Degree** (0..3; truncates or zero-pads canonical shN).
- **Sharpen** (selection, amount 0..0.95, keep coverage).
- **Scale Clamp** (selection, max aspect = largest/middle scale axis, default
  16; `include_flat` property off by default). In log space it shortens only the
  largest axis to the middle axis plus the aspect limit, then blends by
  selection; the middle and smallest axes stay unchanged. This removes needles
  without thickening the flat discs used for trained surfaces. Include Flat
  restores the largest/smallest midpoint clamp for pancake-shaped splats.
  Both modes compare only nonzero axes; collapsed axes remain collapsed.

Colour
- **Colour Correct** (selection, exposure, black point, white point, midpoint,
  contrast, saturation, hue shift, temperature, tint, shadows, midtones,
  highlights, gamma; auto-range property). The affine part `c' = A·c + b`
  (exposure, black/white levels, temperature/tint, contrast about 0.5,
  saturation about Rec.709 luma, luminance-preserving hue rotation) applies to
  sh0 through the base colour and to **every shN coefficient as `A·shN`** (SH
  bands are linear in colour); midpoint, three-way grading and gamma are
  non-linear and only affect the base colour. Blended by selection.
- **Recolour** (selection, colour, weight; keep shading and fade
  view-dependent properties). Mixes base colour toward a target, optionally
  retaining per-splat luminance and fading shN by the recolour weight.
- **Invert Colour** (selection; `A = -I, b = 1`, so shN is negated).

Clean-up
- **Remove Floaters** (selection limits where removal may happen; min
  opacity, max size (0 = off), isolation radius (0 = off), min neighbours;
  preview: show only removal candidates, with all attributes unchanged).
  Isolation uses `radius_neighbor_counts`, saturated at min neighbours. Relative
  mode groups query radii into up to eight octave buckets, querying each bucket
  against all reference points. Zero min neighbours disables isolation removal.
- **Simplify** (ratio; wraps `simplify_splats`, host side).
- **Decimate** (selection, keep fraction; keeps the most important selected
  splats by activated opacity times maximum activated scale, while unselected
  splats are always retained).

Conversion
- **Points to Splats** (radius, 0 = half the mean three-nearest-neighbour distance;
  per-point radii clamp to 0.25–4 times the median). A device spatial hash searches
  27 neighbouring cells, expanding once to 125 cells if fewer than three neighbours
  were found. Each cell examines at most 128 hash entries, bounding dense-region
  work; this is a local spacing estimate, not an exact global k-NN search. A bounded
  sample determines the cell width; no pairwise distance matrix is built.
  **Splats to Points**, **Mesh to Points** (vertices).
- **Mesh to Splats** (core; density per surface unit², max count default 2,000,000,
  opacity default 0.95, seed). Area-weighted face sampling and uniform barycentrics
  create flat degree-zero Gaussians oriented to interpolated vertex/face normals.
  Colours prefer vertex colours, then albedo texture times material base colour,
  then material base colour, then mid-grey. Textures use the loader's flipped UVs
  without a second V flip. The Mesh2Splat panel remains a separate renderer tool.

## 5. Python API (`lichtfeld.nodes`)

```python
import lichtfeld as lf
N = lf.nodes
N.node_types()                         # list[dict] descriptors
t = N.new_tree("Grade")                # Group Input -> Group Output
cc = t.add_node("lfs.colour_correct", location=(200, 0))
t.link(t.input_node, "Geometry", cc, "Geometry")
t.link(cc, "Geometry", t.output_node, "Geometry")
cc.set_input("Exposure", 0.5)
cc.set_property("mode", "...")
m = N.add_modifier("garden", t)        # on scene node "garden"
m.enabled = True; m.show_viewport = True
N.evaluate("garden")                   # {"ok":bool, "errors":{node:msg}, "time_ms":float}
N.evaluated("garden")                  # Geometry with lf.Tensor components
m.capture_selection("Stored Selection")
N.apply_modifier("garden", m.name)
t.to_json(); N.load_tree(json_str)

class Posterize(N.Node):               # plugin node
    id = "example.posterize"; label = "Posterize"; category = "Colour"
    inputs = [N.Input("Geometry", "lfs.geometry"),
              N.Input("Selection", "lfs.float", default=1.0, field=True),
              N.Input("Levels", "lfs.int", default=4, min=2, max=64)]
    outputs = [N.Output("Geometry", "lfs.geometry")]
    def execute(self, ctx):
        geo = ctx.input("Geometry")
        s = geo.splats
        w = ctx.field("Selection", s)              # lf.Tensor [N]
        ...
        return {"Geometry": geo.replace(splats=s.replace(sh0=new_sh0))}
N.register_node(Posterize)
```

## 6. Node editor UI

- New editor type `node_editor` (single instance), placement ActiveView /
  Bottom / 0.38. Opened from the editor-type menu, a View3D header toggle,
  and a keymap action (default Shift+F3).
- Shows the stack of the **active scene node** (the selected node in the
  scene panel).
- Canvas: grid, node boxes (title bar coloured by category, sockets
  coloured by type, unconnected input values shown), bezier wires coloured
  by type, invalid links red, failing nodes red, muted nodes dimmed.
- Interactions: pan (middle drag, two-finger scroll), zoom (wheel with
  modifier / pinch), select (click), move (drag), connect (drag
  output→input), disconnect (drag a connected input away), delete (X or
  Delete), mute (M), duplicate (Shift+D), add (Shift+A or right click →
  categories → node types, including plugin nodes), frame all (Home).
- Sidebar: target node; modifier list (add with a fresh tree or an existing
  tree, remove, reorder, enable, show in viewport, apply); selected node's
  inputs and properties with widgets generated from descriptors; node error
  and evaluation time.
- All strings localised.

### Layout and colour controls

Arrange (header, Shift+L, context menu, or MCP) lays out the selected subgraph
around its existing bounding-box centre. Other nodes and the current view stay
unchanged. With no selection it lays out and frames the entire graph. Both are
single undo steps; MCP can also specify an explicit node list or force all nodes.

Wires share cached graph-space paths between rendering, hit-testing, knife cuts
and splice gestures. Clear forward curves stay cubic; obstructed/backward links
use clearance lanes around cards, with rounded bends and a rectilinear visibility
search when a single lane cannot connect them. Pan, selection and zoom without
card-size changes reuse the paths. Moving any obstructing card invalidates them.
Coincident cards can conceal a socket itself; Arrange resolves that physically
unroutable case.

Colour inputs show one swatch using the standard colour picker, without RGB
number rows. Signed colour descriptors with bounds -1..1 additionally show an
88dp opponent-colour wheel in the inspector. Its centre is neutral; dragging
changes chroma while retaining the achromatic mean until a channel is clamped,
and double-click resets all offsets to zero. Signed swatches/picker channels map
-1..1 to 0..1 so neutral displays grey. Each wheel/picker gesture is one undo step.
Cards show swatches only when inline settings are expanded; connected sockets
remain visible. Titles and Add entries share category icons; title tints blend
the active theme's semantic accents into its surface colour.

Overview LOD hides value editors only, never the socket labels, settings
expander or result footer. Body text retains its 11dp minimum at zoom ≥0.75;
below that it scales with zoom to fit the existing rows without clipping or
enlarging cards into their neighbours. Titles retain their readable minimum.
Narrow footers use an ellipsis with the complete result in a tooltip.
Modifier names similarly elide when not being edited, with their full name as a
tooltip. Inspector status text wraps between bullet-separated items, not inside
values such as `12 ms` or `11% selected`.

### Evaluation and interaction scheduling

The canvas reads the manager's last result; it never evaluates a graph. Pan,
zoom, selection and layout edits do not invalidate evaluation. Layout still
participates in undo. Cards are retained by node identity; selection and result
updates patch classes, footer text and inspector status in place. Grid, static
wires and the live wire have separate geometry caches.

`ModifierManager::tick()` captures at most one request per frame. A single
worker owns a dedicated `TensorWorkQueue`, with one in-flight request and one
replaceable queued request. Each request carries a stack generation, JSON
graphs and read-only stored-payload tensor references. New generations cancel
old evaluations between nodes; stale completed results are discarded. Python
node callbacks acquire the GIL on the worker. The explicit Python `evaluate`
and Apply APIs may wait; canvas edits never do.

The worker waits for the captured producer completion, creates its own GPU
source copies once per source generation, and publishes separate payload
storage after its own fence completes. This separation matters on Metal:
logical queues share one in-order timeline, and storage readiness tracks GPU
readers as well as writers. A worker cache must not reuse displayed storage as
its scratch/input storage. Saturating neighbour-count queries check the current
cell first and exit immediately at the requested count. Cells are one radius
wide, limiting unnecessary candidates in the exact 27-cell search. Their hash build and
query dispatches are queue-ordered, without host waits between query batches.
Boolean radius queries retain their bounded-batch scheduling. No host copies or
device-wide waits are performed by the canvas or result installation.

Evaluation uses the GPU whenever available, including CPU-origin mesh and point
payloads. The worker retains uploaded source geometry; mesh attributes and used
albedo textures are cached by `MeshData::id()` and generation. Node outputs are
normalized to the evaluation device, and Join Geometry also aligns operands to
its first input. An explicit CPU evaluation override supports offline execution
and backend-contract tests. Simplify remains a host-library operation; its result
is transferred back before downstream nodes run.

Only the viewer thread installs fence-complete results. Pending or failed
requests leave the previous successful payload visible. The worker preserves
per-node caches, evaluated selection previews, element counts, selection shares
and timings. A cached no-op reuses the existing published payload.

`lf.nodes.performance(reset=True)` starts a bounded diagnostic capture.
Subsequent reads return request/evaluation/install/discard counters, per-node
run counts, canvas CPU work (input/layout plus deferred RmlUi drawing), viewer
frame CPU/present time, idle/busy viewport-frame samples, and request-to-install
latency. These are CPU wall-clock measurements, not GPU timestamp queries.

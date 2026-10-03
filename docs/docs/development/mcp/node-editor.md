# Node graphs and Node Editor

Initialize MCP and read `tools/list` before calling these tools. Node graphs
and scene targets accept a UUID or an exact, unique name; modifiers use
UUIDs and nodes/sockets use exact descriptor identifiers. Ambiguous names are
rejected, never normalised or fuzzy-matched. Responses return UUIDs and names.
Create, rename and import uniquify graph names with numeric suffixes, such as
`Autumn Lawn 2`.
All mutations run through the viewer work queue and ModifierManager. Graph and
stack edits use shared history. Selection and pan/zoom are view state only.
Responses contain the resulting graph, stack or editor state.
Deleting a graph also removes modifiers using that graph; a single undo restores
the graph and its modifier instances together.

## Discovery

- `lichtfeld://nodes/types`: descriptors, including Python plugins, socket types,
  defaults, field/multi-input flags, ranges/steps, properties and enum items.
- `lichtfeld://nodes/trees`: project graph library; append `/<uuid>` for full JSON,
  including locations and per-node UI settings.
- `lichtfeld://nodes/stacks`: target stacks; append `/<scene node uuid>` for
  modifiers and the last evaluation (errors, counts, selected share, timing/cache).
- `lichtfeld://nodes/editor`: open state, target, shown modifier/graph, selection,
  pan/zoom, preview flag, worker progress, and node/socket screen coordinates.
  Coordinates match `render_capture_window` and `ui_pointer`.

## Graph and modifier tools

| Tools | Arguments |
| --- | --- |
| `nodes_tree_create` | optional `name` (default `Node Graph`) |
| `nodes_tree_delete`, `nodes_tree_export_json` | `tree` UUID or unique name |
| `nodes_tree_rename` | `tree`, `name` |
| `nodes_tree_import_json` | `json` object; assigns a fresh UUID |
| `nodes_node_add` | `tree`, `type_id`; optional unique `name`, `location: [x,y]` |
| `nodes_node_remove` | `tree`, `node`; Group Input/Output are protected |
| `nodes_node_set_input` | `tree`, `node`, `input`, `value` |
| `nodes_node_set_property` | `tree`, `node`, `property`, `value` |
| `nodes_node_mute` | `tree`, `node`, `muted` |
| `nodes_node_move` | `tree`, `node`, `location: [x,y]` |
| `nodes_link`, `nodes_unlink` | `tree`, `from_node`, `from_socket`, `to_node`, `to_socket` |
| `nodes_modifier_add` | `target` scene UUID or unique name, `tree`; optional `name` |
| `nodes_modifier_remove`, `nodes_modifier_apply` | `target`, `modifier` UUID |
| `nodes_modifier_move` | `target`, `modifier`, zero-based `index` |
| `nodes_modifier_set` | `target`, `modifier`; optional `name`, `enabled`, `show_viewport`, `input_overrides` |
| `nodes_modifier_capture_selection` | `target`, `modifier`, Stored Selection `node` |
| `nodes_evaluate` | `target`; optional `wait`, `timeout_ms` (default 30000) |

Values are plain JSON scalars, vectors `[x,y,z]`, or colours `[r,g,b,a]` (alpha
defaults to one). `input_overrides` maps graph interface identifiers to values;
null removes an override. Imports/exports transfer JSON, not filesystem paths.
Rapid edits of one input coalesce into one undo burst (500 ms idle boundary).
The worker keeps one in-flight evaluation and one latest request. View edits
never request evaluation. Apply is an explicit bake and is undoable.

`nodes_evaluate` submits only dirty work. Without `wait` it returns immediately;
with `wait` it polls off the viewer thread and returns `timed_out` when necessary.
The runtime job is `nodes.evaluate`. Use `runtime_job_describe` / `runtime_job_wait`
and `runtime_events_tail` with `nodes.evaluation.started`, `.progress`,
`.completed` or `.failed`. Results are published on the viewer thread only after
their worker fence completes. Previous successful geometry stays visible while
work runs or fails.

## Editor tools

- `nodes_editor_open`, `nodes_editor_close`
- `nodes_editor_show`: `target`, plus `modifier` or a `tree` used by that stack
- `nodes_editor_select`: `nodes` (exact names), optional `links` (zero or one
  link object with the four socket endpoint fields)
- `nodes_editor_arrange`: dependency layout plus frame, one undo step
- `nodes_editor_frame`: frame all shown nodes
- `nodes_editor_view`: optional `pan: [x,y]` in screen pixels, `zoom` (0.3–2.5)
- `nodes_editor_preview_selection`: `enabled`

For a six-node graph: create a graph; add Colour Correct, HSV Range, Recolour,
and Join Geometry; replace the initial direct link with the desired geometry
chain; connect HSV Range.Selection to a Selection input. Add a modifier to the
scene target, open/show/arrange the editor, and evaluate. Inspect the returned
stack before applying. Use `history_undo` to restore the unapplied stack.

To verify live gestures, read socket/card coordinates, send `ui_pointer` down
and several move actions, capture *before* up, then finish the gesture. Do not
push SDL events directly: the polled-pointer override is needed on macOS.

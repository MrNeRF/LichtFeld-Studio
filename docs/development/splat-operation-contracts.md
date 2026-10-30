# Non-destructive splat crops and operation contracts

Crop volumes provide editable filtering for individual splats. Boxes,
ellipsoids and polygons can coexist as independent scene nodes, each with its
own parameters, transform and enabled state. The scene tree organizes them in
a virtual Crop container with a collective enable switch.

Finishing crop editing retains the volume and leaves the source geometry intact.
Applying a crop to geometry is a separate, explicit action. This document covers
the user workflow, operation descriptors, scene hierarchy, rendering,
persistence and validation contracts.

## User workflow

```text
Create:
  Select splat or one of its crops
    -> Crop toolbar command
    -> new independent box, ellipsoid or polygon
    -> edit
    -> Finish retains the editable volume
    -> project Save persists it to disk

Revisit:
  Single-click saved crop -> selection and outline only
    -> double-click or context-menu Edit
    -> edit the same instance with its shape fixed
    -> Finish

Apply permanently:
  Crop or Crop-container context menu
    -> Apply to geometry...
    -> confirm
    -> bake geometry and remove the applied volume(s)
```

A new crop starts with shape switching available. Reopening an existing crop
locks its saved type: the current type stays selected and the other type buttons
use the disabled style. Transform operations, fitting, reset and Finish remain
available. Starting Crop again creates another instance even when the previously
finished crop is still selected.

The viewport toolbar provides crop creation and parameter editing. The scene
tree provides selection, effect switches and context-menu actions.
Checkmark, Enter and the crop pie menu's Finish retain the crop and its enabled
state without modifying the source geometry. Selecting another item leaves
crop editing.

## Scene hierarchy and the virtual container

The scene tree presents:

```text
Splat
└─ Crop (N)                    virtual, collapsible row
   ├─ Crop Box                independent effect switch
   ├─ Ellipsoid               independent effect switch
   ├─ Crop Polygon            independent effect switch
   └─ Another Crop Box        independent effect switch
```

The actual scene graph is:

```text
Splat
├─ Crop Box
├─ Ellipsoid
├─ Crop Polygon
└─ Another Crop Box
```

The container is not a Scene GROUP. It adds no parent node, UUID, transform or
coordinate system. Existing crop UUIDs, real parents and local transforms remain
unchanged. Its expansion is session state; it starts collapsed. It is excluded
from scene-node counts, selection ranges, Select All, transform tools and
reparent destinations. Double click only folds or unfolds it.

The container's right-click actions are:

| Action | Effect | Undo |
| --- | --- | --- |
| Enable/disable | Changes the owner's collective crop gate, preserving every child's enabled flag | One entry |
| Apply active crops to geometry | Applies the enabled crops' intersection and removes those volumes; disabled crops remain editable | One transaction |
| Delete all crops | Removes every crop volume, preserving splat geometry and unrelated children; resets the empty container gate | One transaction |

An individual crop's context menu provides Edit, Duplicate, Apply to geometry
and Delete. Duplicate copies that crop's settings into a new instance. Applying
one crop affects only that volume; other crop instances remain editable.

## Enabled state and composition

Each crop keeps its own payload `enabled` flag. The splat's
`crop_effects_enabled` flag is a separate collective gate:

| Collective gate | Individual crop | Filtering contribution |
| --- | --- | --- |
| On | Enabled | Contributes its crop constraint |
| On | Disabled | No constraint from that crop |
| Off | Either state | No crop constraint; the individual flag is preserved |

Enabled crop volumes combine by **intersection**. A Gaussian must pass every
active constraint on its owning splat. Inverting a volume excludes its region.
Disabling one volume removes only its constraint. Crop order does not define an
ordered modifier evaluation, union or arbitrary Boolean expression.

The row switch controls filtering, not scene visibility. Helper outlines follow
selection transiently. The bulk visibility eye is hidden for selections that
include these crop rows, avoiding a second ambiguous control. Ordinary scene
objects retain their visibility eyes. Other element types use their own
visibility semantics.

## Operation catalog and tool categories

`core/splat_operations.hpp` defines the runtime descriptors used by the scene
tree. Each descriptor contains its kind, scene-node type, category, setter
operator and duplicate operator. An empty operator ID means that action is not
exposed by the descriptor.

| Scene element | Category | Setter operator | Duplicate operator |
| --- | --- | --- | --- |
| Crop Box | Crop volume | `crop_box.set` | `crop_volume.duplicate` |
| Ellipsoid | Crop volume | `ellipsoid.set` | `crop_volume.duplicate` |
| Crop Polygon | Crop volume | `crop_polygon.set` | `crop_volume.duplicate` |
| Collision result | Derived data | None | None |

The collision entry represents generated data rather than a retained modifier
recipe. It has no independent effect-enabled value in the detached reader.

The descriptor operator IDs reference the operator registry. Executable controls
use its poll, validation and undo paths. Fit and Reset are toolbar actions. Box
and ellipsoid use their registered Fit/Reset operators; polygon fitting and
reset use their toolbar paths.

Other editing tools are classified by their role:

| Tool category | Functions | Representation |
| --- | --- | --- |
| Transform | Translate, rotate, scale, set/bake transform, align, scale by distance | Commands and tool state |
| Selection | Submodes, all/none/invert, grow/shrink, opacity/scale queries, selection groups | Selection state and commands |
| Selection filters | Depth and crop selection filters | Filtering state; scene crop filtering respects all enabled owner crops and the collective gate |
| Direct edit | Mirror, Gaussian delete/copy/cut/paste, field writes | Geometry-edit commands |
| Reduction | Splat simplification | Produces another splat; no persistent editable recipe inferred |
| Organization | Duplicate/reparent splat, group selection | Scene commands |

The retained non-destructive volume workflow applies to the three crop shapes.
Other tools operate as commands, generated results or shared tool state.

## Identity, validation and history

`readSplatOperation(owner, element)` returns
`lfs::Result<SplatOperationSnapshot>`. The reader validates:

- a SPLAT owner and a supported element type;
- distinct valid node IDs and non-nil, distinct UUIDs;
- reciprocal direct parent/child ownership;
- the payload required by the element type.

The snapshot copies parameters, local transform and local visibility/lock state.
It contains no Gaussian tensors, voxel buffers or borrowed pointers, performs
no hydration and creates no identities. Its optional enabled value is the
crop's local flag, not its effective state after the collective gate.

Errors use `lfs::Error`, the Core domain and standard error codes. The reader
has no selection, name or ancestor fallback. Callers read live scene state on
the scene-owning thread or under scene read protection and re-resolve UUIDs
before later mutations.

Set/Fit/Reset operators accept an optional explicit `node_uuid`, which takes
precedence over `node` and selection. An invalid, deleted or wrong-type explicit
UUID cancels the command without falling back. Callers that omit it use the
name/selection target-resolution path.

Scene-tree actions revalidate owner/element identities. Bake and collective
removal confirmations recheck their captured identities and scene state before
execution. Ancestor locks, disabled editing, training and modal constraints gate
mutations. Collective bake requires resident splat data, an enabled container
and valid unlocked active volumes; delete-all requires all affected volumes to
be unlocked. Display names are RML-escaped.

Finishing an edit never bakes. An explicit single bake groups parameter commit,
geometry change and volume removal into one undo transaction. Collective bake
and delete also use one transaction each. Undo restores geometry, crop payloads,
enabled states and the original child order; replay does not append a restored
crop in a different sibling position.

## Rendering and persistence

Crop filters use a dynamically sized GPU storage-buffer table. A sorted owner
index points to contiguous box, ellipsoid and polygon ranges; lookup finds the
owner before evaluating its filters. There is no application-imposed fixed
crop-count cap. Counts, offsets, GPU descriptor range and allocation limits are
validated explicitly instead of silently dropping filters.

The fixed overlay prefix contains point-cloud/single-volume filters,
selection, view and reveal controls. Custom shader parameters retain their fixed
address after that prefix; the dynamic crop directory follows the custom tail.
Buffer upload/cache handling follows the actual payload, including shrink and
removal. Source crop node IDs stay on the CPU: a gizmo preview replaces its exact
source filter without overwriting a sibling or reactivating a disabled crop.

Viewport, scene crop-selection filtering and video export honor the same crop
composition and collective gate. Video snapshots retain all ellipsoid instances
as well as all boxes and polygons. Video export captures shader-aware scene
state for each frame.

Crop instances are UUID-bearing scene nodes with per-instance payloads.
The owner gate is stored in the optional SCNG boolean `crop_effects_enabled`: missing
means true, the writer stores false only, and invalid field types are rejected.
Full capture and incremental upsert/sparse updates preserve the same contract.
Splat duplication, node clipboard and undo retain repeated crop children and
the collective gate. A duplicated crop gets a fresh UUID and independent data
without cloning the splat's Gaussian tensors.

## Scope and compatibility

The multi-crop workflow supports individual splats, including splats
inside a normal scene group. Selecting a normal GROUP, or a mixed selection
containing one, disables the Crop command. The virtual Crop container does not
provide group-owned modifiers.

Point clouds use their own crop-helper visibility paths. Repeated crop volumes
and the virtual Crop container are supported for splats only. Training-region
bounds and loss use their training-specific contract; retained scene crops do
not define multiple training regions.

Crop volumes are spatial filters rather than an ordered general-purpose modifier
stack. Camera, sequencing, postprocess and Mesh-to-Splat workflows have separate
contracts. Retained crop editing leaves source geometry intact; explicit geometry
actions such as bake modify it.

Crop controls, confirmation text and the Finish Crop Editing action are localized
in all ten shipped languages. Locale validation checks matching keys and
compatible placeholders.

## Verification

The native test targets are:

| Target | Coverage |
| --- | --- |
| `lichtfeld_splat_operation_tests` | Descriptor and detached reader contracts |
| `lichtfeld_splat_crop_workflow_tests` | Registered operators, retained scene-tree events, repeated crops, collective state, packing and clipboard |
| `lichtfeld_visualizer_tests` | GUI integration, crop lifecycle, bake transactions, locks and persistence |
| `lichtfeld_rmlui_tests` / `lichtfeld_sidebar_input_tests` | Retained UI and input contracts |

Enable `BUILD_SPLAT_WORKFLOW_TESTS=ON` for the focused workflow target and
`BUILD_VISUALIZER_TESTS=ON` for GUI integration targets. The focused suite runs
without a window or GPU renderer; its clipboard hydration regression opts into
CUDA with `LFS_TEST_CROP_CLIPBOARD_GPU=1`.

Windows validation uses MSVC, sccache and FAST_COMPILE. Coverage includes
creation and explicit editing, locked shape on reopen, single and collective
bake/delete, undo/redo with exact sibling order, group-selection rejection and
save/reopen of crop identities, flags and geometry. GPU image comparisons use
120 crops, 40 of each shape, in 3dgs and 3dgut, including owner isolation,
suspension/restoration and buffer shrink/regrowth. Custom shader parameter
coexistence and per-frame video export are also covered.

Localization validation uses:

```sh
python tools/check_locale_completeness.py
python tests/python/test_localization_contracts.py
python tools/check_ui_hardcoded.py --fail-on-candidates
```

Formatting, whitespace, I/O discipline and error-debt checks accompany the
runtime tests.

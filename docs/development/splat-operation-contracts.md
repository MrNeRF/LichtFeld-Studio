# Splat operation contracts

The read-only catalog in `core/splat_operations.hpp` classifies existing native
splat editing tools. It is a foundation for an object inspector, not a second
operator registry, execution engine or persistent modifier stack. Catalog IDs
are classification IDs; they must not be passed to the operator dispatcher.
`tool_identifier`, when present, names the existing toolbar entry.

## Coverage

| Category | Existing paths classified |
| --- | --- |
| Crop volume | Crop Box, ellipsoid, Crop Polygon |
| Derived data | Collision generation and exported result |
| Geometry reduction | Splat simplification producing another splat |
| Transform | Translate, rotate, scale, set transform, bake transform, align, scale by distance |
| Selection | Selection tool, all/none/invert, grow/shrink, by opacity/scale, selection groups |
| Selection filter | Depth and crop filters |
| Direct edit | Mirror, Gaussian deletion/copy/cut/paste, Gaussian field writes |
| Scene organization | Duplicate splat, reparent splat, group selection |

The selection toolbar entry includes its existing Centers, Rectangle, Polygon,
Lasso, Rings, Color, Box and Sphere submodes. Local/World, pivot and mirror axes
remain settings of the existing tools, not independent modifier instances.
The test compares toolbar bindings with the actual native Python definitions.

Navigation, rendering/postprocess settings, import/export workflows, training,
temporal editing and Mesh-to-Splat are outside this splat-editing catalog.
Group-only commands such as ungroup and merge are also outside this first
single-splat contract. A catalog entry does not change the legacy command's
multi-selection support or advertise a group-owned operation.

## Descriptors and permissions

All descriptors describe a single splat context. None enables group owners or
reordering. Capabilities describe implemented paths, not whether an action is
currently allowed: live training, locks, hydration, selection and active jobs
still require the existing application poll/validation services.

Crop volumes expose editable parameters, local enable state and helper display
paths. Collisions expose generation/export, but no persistent editable recipe
or independent operation enable switch. Simplification is command-only: its
parameters live in the current request, not in a recoverable operation on every
resulting splat. Selection, transforms and editing remain commands rather than
being mislabeled as nondestructive modifiers.

The existing crop tool can convert or replace other crop volume shapes. This
catalog does not promise that arbitrary crop shapes coexist or form an ordered
stack, nor does it change those replacement rules.

## Reading existing scene data

`readSplatOperation(owner, element)` returns a detached snapshot of a direct
crop-volume or collision child of a splat. Pass live nodes from the same scene
on the scene-owning thread, or under the caller's existing scene read protection.
The adapter checks owner type, distinct non-nil identities and reciprocal direct
parent/child links. It never falls back to selection, names or ancestors.

The snapshot copies existing parameters and the child's local transform. It
contains no Gaussian tensors or voxel words and does not hydrate unloaded
geometry. `enabled` is absent for collision results. `node_visible` and
`node_locked` are local node flags; ancestor state and global renderer switches
must not be inferred from them. The snapshot survives payload removal, but its
UUIDs must be resolved and revalidated before any later edit.

The adapter is an inspection path, not parameter validation. It preserves
existing payload values verbatim and reports missing payloads rather than
manufacturing defaults. It cannot recover a recipe from an ordinary splat
produced by simplification, and does not invent operation UUIDs.

## Authoritative data and mutation paths

| Read data | Authoritative source | Existing mutation path to reuse |
| --- | --- | --- |
| Crop Box | `SceneNode::cropbox` and local transform | `CropBoxSetOperator` / `cap::updateCropBox`, existing gizmo and undo paths |
| Ellipsoid | `SceneNode::ellipsoid` and local transform | `EllipsoidSetOperator` / `cap::updateEllipsoid`, existing gizmo and undo paths |
| Crop Polygon | `SceneNode::crop_polygon` and local transform | `CropPolygonSetOperator` and existing contour editing commands |
| Collision parameters | Completed `CollisionVolume::settings` | `generate_collision` / `AsyncTaskManager::startCollision`; export through the existing collision job |
| Simplification request | Current `SplatSimplifyOptions` request | `simplify_splats` / existing async simplification service |
| Node visibility and lock | Existing observable node properties | SceneManager and established undo-aware scene commands |

The adapter deliberately adds no setter and does not bypass undo or job capture
checks. Future UI/API consumers must resolve targets explicitly and reuse these
paths; they must not write cached snapshot parameters back directly.

## Persistence and compatibility

No fields are added to SceneNode, the project chapters or existing payloads.
No migrations, serializer changes, generated IDs, renderer gates or evaluation
changes are introduced. The original node payload remains the sole source of
truth. New recipes, source/result separation and group aggregation require
subsequent changes with their own persistence and runtime contracts.

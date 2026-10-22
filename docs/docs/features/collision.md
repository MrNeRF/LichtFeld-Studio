---
title: Collision meshes and walk testing
---

Open **Tools > Collision** to generate native collision geometry from visible,
loaded splat nodes. No external `splat-transform` executable is required.
Deleted splats are excluded. Stop training before taking the source snapshot.

Choose **Indoor**, **Outdoor** (default), or **Object**, then **Apply preset**.
All pipeline controls remain independently editable:

| Stage | Controls and defaults |
| --- | --- |
| Cluster filter (optional) | Resolution `1.0`, opacity `0.999`, minimum Gaussian contribution `0.1` |
| Voxelize | World-space voxel size `0.05` (5 cm), accumulated opacity `0.1` |
| External fill (Indoor) | Size `1.6`; warns and returns the original volume if the seed is exterior or consumed by sealing |
| Floor fill (Outdoor) | Size `1.6`; patches only surrounded columns within twice this distance |
| Capsule carve (Indoor) | Height `1.6`, radius `0.2`; height `0` disables it |
| Mesh | Smooth marching cubes with coplanar simplification, or exact exposed voxel faces |

The shared seed is entered in **data-world XYZ** and is initially taken from the
camera. **Use camera position** refreshes it. Clustering chooses the nearest
occupied coarse component; exterior fill checks that it is enclosed; carving
requires a capsule centered at the seed to fit. Floor filling and carving honor
the project's navigation up-axis.

Generation runs on a cancellable worker. The source snapshot owns its data, so
editing cannot race the worker. A scene change or project reload discards stale
results rather than replacing a node with outdated data. Failures leave existing
collision geometry intact. Generation considers every visible source Gaussian at
the requested voxel size; it never crops sources or silently coarsens voxels.

## Sparse storage and resource limits

Generation streams mixed occupancy through 8-by-8-by-8 alpha-evaluation tiles into
an adaptive octree. Uniform empty and solid regions stay compressed; the mostly
empty world bounding volume is not allocated. A source BVH conservatively proves
which isolated weak supports cannot reach the opacity threshold before computing
bounds. Weak Gaussians that can combine with other supports remain in the evaluator.
Rotated/scaled Gaussian contributions retain accumulated-opacity semantics.

Exterior sealing and capsule carving operate on adaptive regions and connected
free-space leaves. Floor fill collects only active XZ columns, preserves the
four-direction surrounded-gap rule, builds a Morton-ordered height quadtree, and
constructs the union directly rather than mutating one column at a time. Smooth
and exposed-face meshes use bounded 32-cubed boundary chunks instead of scanning
the entire world grid. Smooth chunks have coplanar simplification and exact
half-grid seam welding; voxel preview uploads occupancy without uploading the
physical mesh.

Limits apply to actual source, allocated tree, active-column and output-mesh work:
10 million source Gaussians, 512,000,001 CPU tree words, 8 million floor columns,
32,000,001 floor-height nodes, 64 million mesh vertices, and 384 million indices.
Generation, validation and project restore share the same tree-word limit
(approximately 1.91 GiB of words per volume); it remains below the GPU's reserved
high-bit pointer range. Mesh limits permit approximately 2.15 GiB of positions
and indices before normals, colors and staging.
Tree operations and individual postprocessing stages also have finite work
budgets. Grid axes are limited to 2^21 cells for coordinate/precision safety.
GPU occupancy must fit the device's storage-buffer range. Allocator capacity,
source snapshots, staging copies and the mesh contribute additional peak memory.
Errors name the exhausted budget; no partial collider replaces the previous node.
The legacy dense payload reader retains its 16-million-cell safety limit; this is
not a limit on newly generated sparse world bounds.

Version-2 project payloads store compact octree topology and settings. Version-1
dense run-length payloads remain readable. Sparse generation follows the scalable
uniform-region/brick approach inspected in
[splat-transform's voxel sources](https://github.com/playcanvas/splat-transform/tree/4f90012be983078010d631fa0572baa99d5c1965/src/lib/voxel),
including `sparse-voxel-grid.ts`, `voxelize.ts`, `fill-exterior.ts`,
`fill-floor.ts`, `carve.ts`, and `flood-fill.ts`. The native representation and
algorithms are independently implemented, not byte-identical upstream output.

## Scene graph and editing

The generated **Collision** node is a native mesh node with an additional immutable
voxel/settings payload. It supports normal transforms, parenting, visibility,
duplication, deletion and deletion undo. Both mesh geometry and collision settings/
occupancy persist in `.licht` projects, including deferred payload hydration.

**Voxel boxes** ray-marches the occupancy to the first surface per pixel and
composites an axis-shaded gray box overlay over the splats: 40% face opacity,
80% dark-edge opacity, and approximately 1.5-pixel edges. Hidden/internal edges do
not accumulate and splat colors are not modified. Like SuperSplat's debug view,
the overlay deliberately ignores splat depth. Inside solid occupancy it shows the
exit surface in orange; selected nodes receive a blue tint when viewed from free
space. Node visibility and transforms apply, including after project reload.
**Collision mesh** restores the chosen smooth or exposed-face physical surface.
Neither view changes the occupancy volume used by walk collision.

Each collision node remains an independently selectable, transformable volume
with its own settings and occupancy, not millions of scene nodes. Individual
voxel/face editing is not implemented. Overlapping collision nodes currently
composite independently; they are not merged into a single nearest-hit query.
Consequently, intersecting overlays can include farther surfaces and their blended
appearance depends on scene draw order.
The GPU uses the sparse tree with mixed leaves compressed into 4-cubed occupancy
bitmasks. Traversal skips empty/uniform regions instead of stepping through the
entire bounding grid.

The visual behavior follows the primary
[SuperSplat viewer implementation](https://github.com/playcanvas/supersplat-viewer/blob/ece8201c6a0dbea2dcccc418ec63c4217c7b2956/src/voxel-debug-overlay.ts)
(PlayCanvas, MIT). The native GLSL implementation is independently written; no
upstream tool is executed.

**Edit source splats** exits walk, hides the collision overlay and selects visible
source splats. Use the existing native selection, delete, crop and transform tools
with their normal undo support. Then **Generate / Regenerate** rebuilds the
remembered collision target from the current visible sources and makes it visible
again. Regeneration realigns the node to the current navigation frame; it is not a
deformation of the previous mesh. **Generate new node** keeps existing nodes.

## Full-scene Bicycle validation

The separate `bicycle-collision-full-scene.licht` validation project contains the
entire visible 1,000,000-Gaussian trained Model and one Collision node, with no
regional source copy. Generation used 5 cm voxels and opacity `0.1`, **clustering
OFF, floor fill OFF, exterior fill OFF, and carve OFF**. The pristine trained
`project.licht` and `splat_30000.ply` remain separate and unchanged. Earlier ROI
projects/screenshots are partial diagnostics, not full-scene results.

The GUI run generated 112,350,948 occupied voxels in 15.649 seconds using
35,975,081 CPU tree words (137.2 MiB). Smooth meshing took another 26.403 seconds,
producing 28,191,123 vertices and 55,748,034 triangles. This is a large,
high-resolution proxy, not a compact game-ready mesh: positions and indices alone
are approximately 960.6 MiB before normals, allocator capacity and save staging.
The separate project is approximately 1.27 GiB. A standalone optimized run on the
same million-source dump measured 18.99 seconds generation, 19.97 seconds meshing,
18.6 MiB GPU occupancy and 1.67 GiB peak RSS; compiler floating-point differences
slightly change boundary occupancy and coplanar simplification counts.

Save/reload restores the full Model, mesh and sparse preview. Walk entered from
`(-3, 3, -3)`, grounded at eye Y approximately `0.85081`, and remained stable;
hiding the collider rejected Walk entry and restoring visibility allowed it.
Native regressions additionally cover archive hydration, duplication/deletion
undo, transformed capsules, distant sparse components, boundary seams, all six
inside-solid bounding exits, and order-dependent independent overlays.

The separate `bicycle-collision-full-scene-outdoor.licht` project also completes
the full default **Outdoor** path: all 1,000,000 visible Gaussians, 5 cm voxels,
opacity `0.1`, **floor fill ON at `1.6`, clustering OFF, exterior fill OFF,
and carve OFF**. No source copy, crop or resolution change is used. The former
128-million-word tree and 32-million-vertex mesh caps were conservative
implementation limits, not machine exhaustion; the bounded capacities above
allow the actual canonical result.

Native GUI generation produced 2,031,784,505 occupied cells and 133,030,713 CPU
tree words (507.5 MiB) in 18.381 seconds. Smooth meshing and normals took another
104.631 seconds, producing 62,855,233 vertices and 125,356,766 triangles.
Positions and indices alone occupy approximately 2.10 GiB; this remains a very
large proxy despite chunk-local coplanar simplification, not an optimized
game-ready mesh. The compacted saved project is approximately 1.63 GiB.
Native generation/save/reload peaked at approximately 19.04 GiB RSS on a 62 GiB
host; the much smaller occupancy buffer does not imply that the complete
authoring pipeline is suitable for small-memory hosts.

The standalone full-source Outdoor benchmark measured 22.300 seconds generation,
97.251 seconds meshing and 3.77 GiB peak RSS, with 133,033,817 CPU words,
18,036,857 packed GPU words (68.8 MiB), 59,499,132 vertices and 118,644,536 triangles.
It used 2,215,493 active floor columns and 2,399,449 height nodes rather than
allocating the 3337-by-1793-by-4095 world grid. Small native/compiler arithmetic
differences at voxel boundaries can cause larger downstream differences in
coplanar simplification counts.

Reopening the Outdoor project hydrated both the complete Model and Collision
payload, retaining the mesh counts and voxel preview. Walk entered from
`(0, 4, -3)` and grounded stably at eye Y approximately `0.824`. Hiding the only
collider rejected Walk entry; restoring it allowed entry again. Floor extrusion
can create tall filled columns beneath canopy-only source supports; those are
real generated occupancy, not omitted source regions or a second display mode.
Synthetic regressions additionally check dense/sparse floor parity, uneven
heights, surrounded holes, padding, persistence and cancellation.

## Walk testing

Use **Enter walk** in the panel or the **Walk Camera (collision)** toolbar button.
At least one visible, loaded collision node is required, and the player capsule
must fit at the current camera. Defaults are a `1.8` head height, `0.2` radius,
`2.5` world units/second and `0.25` maximum step elevation.

- **WASD** moves horizontally; **Shift** doubles speed.
- Use the normal camera look drag to turn.
- Gravity grounds the capsule; walls slide and low obstacles can be stepped over.
- **Escape** or **Exit walk** returns to fly mode.

Orbit/fly remain unchanged outside walk. Pan, zoom and pivot glides are disabled
while walking; home, focus, camera browsing and external teleports leave walk.
Hidden collision nodes do not collide. Rigid/uniform node transforms preserve
capsule radius; anisotropic scale/shear uses conservative clearance.

Walk uses voxel occupancy, so its clearance can be slightly more conservative
than the smooth visual mesh.

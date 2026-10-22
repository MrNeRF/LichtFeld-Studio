---
title: Collision meshes and walk testing
---

Open **Tools > Collision** with your splat scene loaded. Generation uses all
visible, loaded splats and excludes deleted splats. Stop training first.
No external `splat-transform` executable is required.

## Quick setup

1. Choose **Indoor**, **Outdoor**, or **Object**. Changing the type applies its preset.
2. Set **Scene units / metre**. Reconstruction coordinates have no inherent metric
   scale: if a known two-metre object spans five scene units, enter `2.5`.
   This calibrates collision and player dimensions without resizing the splats.
3. Drag **Cell size (metres)**. Start at `0.05` (5 cm); larger cells are faster and
   less detailed. The readout also shows the actual cell size in scene units.
4. Place the camera where walking should start, press **Use camera position**,
   then **Generate / Regenerate**.
5. Inspect the boxes, then **Enter walk**. Use **Overlay strength** and
   **Cutaway height** to reveal surfaces inside the scene.

The normal generation path builds full-resolution voxel collision and its preview.
It does **not** build a triangle surface: neither the preview nor Walk needs one.
A progress bar shows completed stages and voxelization regions, the current task,
and elapsed time. Stages have different costs; the bar is not a time estimate.
**Cancel** leaves the existing collider intact. Source snapshots own their data;
source edits or a project change invalidate an in-flight result.

The initial seed comes from the camera, in data-world XYZ. Use camera position
refreshes it. Keep it at standing eye height above the desired floor. Clustering
chooses the nearest coarse component; exterior filling needs an enclosed seed;
capsule carving needs enough clearance there. The navigation up-axis is respected.

## Advanced controls

Advanced settings retain independent access to the pipeline parameters:

| Stage | Defaults at one scene unit per metre |
| --- | --- |
| Cluster filter (optional) | Resolution `1`, opacity `0.999`, minimum contribution `0.1` |
| Voxelization | Cell size `0.05`, accumulated opacity `0.1` |
| Indoor | Exterior fill size `1.6`, capsule carve height `1.6`, radius `0.2` |
| Outdoor | Floor fill size `1.6`; exterior fill and carving off |
| Object | Exterior fill, floor fill and carving off |
| Surface | Smooth or exposed voxel faces; separate surface resolution multiplier |

Dimensional preset values and the Walk capsule scale with **Scene units / metre**.
Changing it rescales these settings, not existing generated nodes. Regenerate to
apply changes. The exact voxel size remains editable in scene units under Advanced.

New Outdoor authoring enables **Avoid overhead floor columns**. Only supports at
least 1.6 metres below the seed can start downward floor extrusion. Overhead
foliage stays occupied but no longer automatically becomes a tall floor column.
This is an LFS addition, not an upstream parameter. Disable it for unrestricted
floor fill. Existing projects retain their original setting. The guard depends on
placing the seed at an appropriate standing height; it is not ground recognition.
Floor patches are limited to surrounded columns within the configured reach.

The defaults and scene types follow the
[SuperSplat collision guide](https://developer.playcanvas.com/user-manual/supersplat/studio/collision/)
and [splat-transform parameters](https://developer.playcanvas.com/user-manual/splat-transform/collision/).
Their metre-based dimensions require calibration for an arbitrarily scaled scene.
The native algorithms are independently implemented and do not promise
byte-identical upstream output.

## Preview and optional surfaces

The box preview ray-marches occupancy to the first surface per pixel, with
axis-shaded gray translucent faces and dark screen-space edges. It follows the
[SuperSplat viewer overlay](https://github.com/playcanvas/supersplat-viewer/blob/ece8201c6a0dbea2dcccc418ec63c4217c7b2956/src/voxel-debug-overlay.ts).
Hidden/internal edges do not accumulate. Inside solid occupancy the exit surface
is orange; selected nodes have a blue tint. Splat colors are unchanged. Like the
reference, the overlay ignores splat depth.

**Overlay strength** and **Cutaway height** update display uniforms immediately.
They neither rebuild nor copy occupancy, and do not alter Walk collision.
Cutaway clips the upper part of the volume along its local up-axis. These controls
and the selected display mode persist per node. Intersecting nodes composite
independently, so their blended appearance remains draw-order dependent.

Use **Build / show surface** under Advanced only when a triangle surface is needed.
The default `4×` surface multiplier conservatively combines occupied cells into a
coarser grid before meshing. All components remain represented, but fine gaps can
close. `1×` retains the original voxel detail and can produce a very large mesh.
Changing the multiplier or shape rebuilds the optional surface on request;
switching back to an already-built surface reuses it. **Voxel boxes** restores the
original-resolution preview. Walk always uses the original occupancy.

Each Collision is an independent scene node with transforms, parenting,
visibility, duplication, deletion and deletion undo. The fast voxel-only node has
a tiny bounds carrier, not an extracted physical surface. Build a surface before
using its mesh geometry. `.licht` projects retain occupancy, settings, display
controls and any explicitly built surface, including deferred hydration. Legacy
dense and sparse collision projects remain readable.
Individual voxel/face editing is not implemented. **Edit source splats** exits
Walk, hides the collider and selects visible source splats. Use the normal native
selection, delete, crop and transform tools, then regenerate the same node.
**Generate new node** retains existing colliders.

## Performance and limits

Collision snapshots copy only means, raw scale/rotation/opacity and deletion
state, without spherical harmonics or optimizer state. Up to eight independent
spatial regions voxelize concurrently. Conservative opacity bounds skip provably
empty/solid regions; active bricks evaluate only relevant Gaussian support and
stop accumulating a cell once it is occupied. Sparse postprocessing takes
ownership of its input tree instead of copying it. Archive snapshots share the
immutable payload. Optional surface normals use a direct indexed pass without
constructing OpenMesh adjacency or per-vertex uniform colors.

On the same full 1,000,000-Gaussian Bicycle source dump, with 5 cm cells, opacity
`0.1` and unrestricted Outdoor floor fill `1.6`, the optimized standalone run
produced the same **2,031,785,065 occupied cells** in **10.45 s**, versus the
previous **22.30 s** standalone generation. The new default path also skips the
previous **97.25 s** standalone surface extraction. This comparison preserves
source coverage and voxel resolution; it is not a cropped preview.

With the optional overhead guard and seed `(0,3,-3)`, the same full source took
**9.55 s** and **298 MiB** CPU occupancy. An explicitly requested `4×` smooth
surface took another **2.76 s**, producing **4.76 million triangles**. This result
uses different floor behavior, so it is not an identical-occupancy speed comparison.
A coarse surface can still be large; these timings are machine-specific CPU
measurements and exclude GUI upload/save costs.

The native GUI full-source run with the guard completed generation in **8.88 s**
with **298.2 MiB** occupancy, and its optional `4×` surface in **2.88 s**
(4,952,794 triangles). Separate `bicycle-collision-fast-preview.licht` and
`bicycle-collision-fast-surface.licht` validation files retain the complete Model;
the pristine training project and PLY are unchanged. Regeneration, cancellation,
live display scrubs, project replacement and rehydrated Walk were exercised.
Render snapshots retain payload ownership until consumed, including when the
collider is replaced or deleted during a frame.

Storage uses an adaptive occupancy tree and 8-cubed evaluation bricks, not a dense
world allocation. GPU traversal uses uniform regions and packed 4-cubed leaf
masks. Finite limits apply to actual allocated work: 10 million source Gaussians,
512,000,001 CPU tree words (about 1.91 GiB), 8 million floor columns,
32,000,001 floor-height nodes, 64 million surface vertices and 384 million indices.
Grid axes are limited to 2^21 cells. GPU buffers must fit device limits.
Snapshots, allocator capacity, uploads and optional surfaces add peak memory.
Failures name the exceeded budget and do not replace an existing collider.
The legacy dense reader retains its 16-million-cell limit.

## Walk testing

Use **Enter walk** in the panel or the **Walk Camera (collision)** toolbar button.
A visible, loaded collision node and free player capsule are required. The panel
applies calibrated defaults of 1.8 m eye height, 0.2 m radius, 2.5 m/s speed and
0.25 m step height; these remain editable in Advanced.

- **WASD** moves horizontally; **Shift** runs.
- Use the normal camera look drag to turn.
- Gravity grounds the capsule; walls slide and low obstacles can be stepped over.
- **Escape** or **Exit walk** returns to fly mode.

Pan, zoom and pivot glides are disabled while walking. Home, focus, camera browsing
and external teleports leave Walk. Hidden colliders do not collide. Anisotropic
scale/shear uses conservative capsule clearance. Walk tests the voxel volume,
so it can be more conservative than the optional smooth surface. Capsule queries
skip sparse empty regions and use exact segment/box distance, rather than scanning
all cells with iterative per-cell distance searches.

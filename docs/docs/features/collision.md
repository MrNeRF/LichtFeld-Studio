---
title: Collision meshes and walk testing
---

Open **Tools > Collision** with your splat scene loaded. Generation uses all
visible, loaded splats and excludes deleted splats. Stop training first.
No external `splat-transform` executable is required.

## Quick setup

The same surface-first setup works for rooms, outdoor scans and objects:

1. Set **Scene units / metre**. Reconstruction coordinates have no inherent metric
   scale: if a known two-metre object spans five scene units, enter `2.5`.
   This calibrates collision and player dimensions without resizing the splats.
2. Drag **Cell size (metres)**. Start at `0.05` (5 cm); larger cells are faster and
   less detailed. The readout also shows the actual cell size in scene units.
3. Place the camera near the desired region and press **Use camera position**.
   Optionally enable **Keep connected region near seed** to exclude disconnected
   source geometry. Leave it off to include all visible splats.
4. Press **Generate / Regenerate**, inspect the boxes, then **Enter walk**.
   Use **Overlay strength** and **Cutaway height** to reveal surfaces inside the scene.

**Next generation** shows the current source-filter, filling and carving settings. To restore the
shared surface-only setup, choose **Reset to surface collision** under Advanced.
This keeps the scale, cell size and seed; regenerate to apply it. No Indoor/Outdoor
choice or automatic scene classification is required.

The normal generation path builds full-resolution voxel collision and its preview.
It does **not** build a triangle surface: neither the preview nor Walk needs one.
A progress bar shows completed stages and voxelization regions, the current task,
and elapsed time. Stages have different costs; the bar is not a time estimate.
**Cancel** leaves the existing collider intact. Source snapshots own their data;
source edits or a project change invalidate an in-flight result.

The popup groups controls under collapsible **Generate collision**, **Inspect
collision**, **Walk and edit**, and **Advanced** headers. Progress and errors stay
visible when settings are collapsed. Advanced filtering, voxelization, filling,
seed and triangle-surface controls each have their own header.

Progress names the current CPU or GPU task and shows the work-stage count,
percentage, elapsed time and time since the last progress update. Completed,
cancelled and failed operations retain their final status. A failure includes
recovery advice and expandable **Error details** with the failed task, source
count, relevant settings and exact diagnostic. **Copy details** copies that report.

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
| Common setup | Surface shell; all filling and carving off |
| Optional room sealing | Exterior fill size `1.6`; carve height `1.6`, radius `0.2` when explicitly enabled |
| Optional solid ground | Floor fill size `1.6` |
| Surface | Smooth or exposed voxel faces; separate surface resolution multiplier |

Dimensional settings and the Walk capsule scale with **Scene units / metre**.
Changing it rescales these settings, not existing generated nodes. Regenerate to
apply changes. The exact voxel size remains editable in scene units under Advanced.

New authoring leaves ground fill **off**. Voxelization follows Gaussian
surface support; filling below the first occupied support in each XZ column can
turn foliage and other elevated surfaces into solid pillars. This is the main
cause of a dense, enclosing result, not the sparse tree's compression. Enable
**Fill solid ground (optional)** explicitly when that volume is wanted. Its
Advanced size starts at `1.6` metres; patches are limited to surrounded columns.
**Avoid overhead floor columns** limits eligible supports to at least 1.6 metres
below the seed, but cannot distinguish low vegetation from actual ground.
Existing saved settings and occupancy are unchanged; uncheck ground fill and
regenerate an older Outdoor collider to get the surface shell.

Voxelization samples the closest point in each voxel box to the Gaussian center,
transforms that point into the rotated/scaled Gaussian frame, and sums extinction
density before converting it to opacity (`1 - exp(-density)`). This follows the
[upstream voxel sampler](https://github.com/playcanvas/splat-transform/blob/4f90012be983078010d631fa0572baa99d5c1965/src/lib/gpu/gpu-voxelization.ts).
Center-only sampling could miss sub-voxel sheets; the previous over-alpha
accumulation also differed from upstream. Native sparse bounds and early exits
use the same density threshold as the cell evaluator.

Metric dimensions and fill options are based on the
[SuperSplat collision guide](https://developer.playcanvas.com/user-manual/supersplat/studio/collision/)
and [splat-transform parameters](https://developer.playcanvas.com/user-manual/splat-transform/collision/).
Their metre-based dimensions require calibration for an arbitrarily scaled scene.
LFS deliberately starts without filling to preserve visible surface shape.
Room sealing and ground fill change different regions of the volume and are
not automatically combined. Existing projects retain their precise options;
legacy Indoor/Outdoor/Object preset identifiers remain readable in archives.
The native algorithms are independently implemented and do not promise
byte-identical upstream output.

## Connected-region filtering

The optional filter coarse-voxelizes the source, finds the occupied component
nearest the seed, and retains splats associated with that component. Flooding is
six-connected: touching only at an edge or corner does not join components.
The retained count and any seed snap distance are shown after generation. A snap
to a distant component is a reason to move the seed, not proof of a good selection.

Following [upstream cluster selection](https://github.com/playcanvas/splat-transform/blob/4f90012be983078010d631fa0572baa99d5c1965/src/lib/voxel/filter-cluster.ts),
splats centered inside the selected region are retained immediately, including
thin and low-opacity detail. Other splats must contribute at occupied coarse-cell
centers. If a splat's three-sigma AABB is wider than two coarse cells on any axis,
it needs at least `ceil(10% * AABB volume / cell volume)` qualifying hits instead
of one. This rejects large outliers whose tails barely touch the selected region.
Native queries skip empty tree regions and count uniformly qualifying occupied
regions without scanning every cell of a large AABB. Source ordering is retained.
The coarse pass, connected flood and splat selection report progress separately.
The input Model is unchanged: filtering affects the generated collider only.

The previous filter omitted center retention and the large-splat overlap rule.
On the full 1M Bicycle dump with coarse resolution `1`, opacity `0.999`, minimum
contribution `0.1` and seed `(0,3,-3)`, it kept just **548 splats**. The corrected
filter retains **855,274**. Filtering itself takes about **0.05 s** in both runs;
the coarse CPU pass dominates at roughly **1.25 s**. Corrected full generation
takes **5.36 s**, versus **1.48 s** for the severely over-filtered old result.
This is a correctness comparison, not an identical-output speedup. Upstream uses
GPU coarse voxelization; these measurements are for our native CPU path and do
not constitute a benchmark against the upstream executable.

The corresponding native GUI run retains **855,727 splats**, generates in **5.25 s**,
and uses **33.3 MiB** CPU occupancy (6,692,430 occupied cells). Native source
activation/rounding can differ slightly from the standalone source dump. The
separate validation archive is `bicycle-collision-clustered.licht`; the original
1M Model and pristine training outputs remain intact. Walk grounds at eye Y
approximately `0.867` from `(0,3,-3)`.

Native grid alignment and Euclidean nearest-cell search can differ from
upstream block alignment and shell-order seed search. The native filter also
avoids upstream's block-count-only "all in one cluster" bypass: separate occupied
components can share a block. Tests compare hierarchical queries to an independent
cell-by-cell evaluator and cover center retention, outliers and corner contact.

## Preview and optional surfaces

The box preview ray-marches occupancy to the first surface per pixel, with
axis-shaded gray translucent faces and dark screen-space edges. It follows the
[SuperSplat viewer overlay](https://github.com/playcanvas/supersplat-viewer/blob/ece8201c6a0dbea2dcccc418ec63c4217c7b2956/src/voxel-debug-overlay.ts).
Hidden/internal edges do not accumulate. Edges use the actual hit voxel, including
after a traversal skip through a uniform solid region. Inside solid occupancy the exit surface
is orange; selected nodes have a blue tint. Splat colors are unchanged. Like the
reference, the overlay ignores splat depth.

**Overlay strength** and **Cutaway height** update display uniforms immediately.
They neither rebuild nor copy occupancy, and do not alter Walk collision.
Cutaway clips the upper part of the volume along its local up-axis. These controls
and the selected display mode persist per node. Intersecting nodes composite
independently, so their blended appearance remains draw-order dependent.

**Sharp box preview** under Advanced is on by default and traces every pixel.
Turning it off requests hardware 2×2 fragment shading where supported, with a
full-resolution fallback elsewhere. This softens edges without changing any
collision cells or Walk queries. Opacity, cutaway and sharpness are live controls.

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
state, without spherical harmonics or optimizer state. The GUI uses native CUDA
for both coarse cluster voxelization and final voxelization. Conservative bounds
and the Gaussian BVH are prepared on CPU; weak-support bounds use up to eight
threads. CUDA expands only potentially occupied regions and evaluates 4³ bricks
with 64 lanes, stopping density accumulation once the requested threshold is met.
Prepared Gaussian/BVH records upload once per pass; only nonempty brick masks
return to CPU for canonical Walk/archive topology. There is no dense world-sized
alpha allocation or intermediate triangle surface. Connected filtering, optional
filling/carving and canonical-tree assembly remain on CPU. This is GPU
voxelization, not an entirely GPU-resident authoring pipeline. The CPU evaluator
remains available as a reference. Sparse postprocessing takes
ownership of its input tree instead of copying it. Archive snapshots share the
immutable payload. Optional surface normals use a direct indexed pass without
constructing OpenMesh adjacency or per-vertex uniform colors.

When all Gaussian supports already fit the sparse coordinate limit, preparation
keeps their conservative bounds directly. It skips the optional per-splat overlap
proofs that otherwise cost hundreds of millions of CPU BVH visits on large aerial
scans. Empty padding still uses sparse storage. Supports outside the coordinate
limit retain the original opacity-bound pruning path; all contributions inside
the resulting grid are still evaluated.

The earlier CPU baseline on a full 1,000,000-Gaussian Bicycle source dump, with
5 cm cells and opacity `0.1`, produces **122,325,185 occupied cells** in
**12.25 s**, using **147.4 MiB** CPU
occupancy (standalone peak RSS **578 MiB**). With the same revised sampler and
optional guarded ground fill, it produces **850,702,239 cells**, uses **299.4 MiB**
occupancy, and takes **13.84 s**. Ground fill changes collision semantics; these
are not identical-output speed comparisons. Both include all source Gaussians
without cropping, clustering or coarsening occupancy. Timings are machine-specific
and exclude GUI upload/save costs.

The earlier CPU native GUI surface-only run took **10.35 s**, with **122,325,186 cells** and
**147.4 MiB** CPU occupancy. The separate full-scene validation project is
`bicycle-collision-surface-aligned.licht`; the complete 1M Model remains visible.
Save/reload preserves the collider identity and surface-only settings; rehydrated
Walk grounds stably at eye Y approximately `0.868` from `(0,3,-3)`. The pristine
training project and PLY are unchanged. Matched-camera before/after captures show
the removed floor-fill curtains beneath vegetation. Raw source floaters and
axis-aligned voxel stair steps remain represented.

The default preview path skips optional triangle extraction entirely. Earlier
unrestricted Outdoor generation required another 97.25 s for a 118.6-million-
triangle surface. Coarsened optional surfaces can still be large. Render snapshots
retain payload ownership until consumed, including when the collider is replaced
or deleted during a frame.

Storage uses an adaptive occupancy tree, not a dense world allocation. The CPU
reference evaluates 8³ bricks; CUDA evaluates 4³ bricks. GPU traversal uses uniform regions and packed 4-cubed leaf
masks. Finite limits apply to actual allocated work: 10 million source Gaussians,
512,000,001 CPU tree words (about 1.91 GiB), 8 million floor columns,
32,000,001 floor-height nodes, 64 million surface vertices and 384 million indices.
Grid axes are limited to 2^21 cells. CUDA worklists are bounded to 32 million
active regions. GPU buffers must fit device limits.
Snapshots, allocator capacity, uploads and optional surfaces add peak memory.
Failures name the exceeded budget and do not replace an existing collider.
The legacy dense reader retains its 16-million-cell limit.

### Measured CUDA and rendering improvements

On the RTX 4090 validation machine, CUDA generation takes **1.86 s** unfiltered
and **1.10 s** with the corrected connected filter, versus the prior **12.25 s**
and **5.36 s** CPU baselines. These include CPU preparation and tree assembly,
but exclude GUI snapshot, upload and save costs. Unfiltered CUDA occupancy is
122,325,196 cells / 147.4 MiB; clustered occupancy is 6,692,435 cells / 33.3 MiB.
Floating-point evaluation near a threshold can change a few occupied cells;
coarse connectivity can amplify such boundary differences. No byte-identical
CPU/GPU output is promised. Small deterministic fixtures compare every cell.

The GUI generated all visible 1M Bicycle splats with filtering in **1.21 s**,
retaining 855,274 splats and 6,692,434 occupied cells. The separate archive is
`bicycle-collision-gpu.licht`; the pristine training project and PLY are unchanged.

The severe preview slowdown was a numerical traversal bug: rounding a
non-crossed coordinate could move it backwards at an edge, creating a two-cell
cycle. In the unfiltered overview, 536 rays exhausted the 9,227-step safety bound.
Every axis now advances monotonically along its ray. Ancestor paths and packed
brick masks are cached; uniform regions jump directly to their exit. Camera
occupancy is queried once per draw, storage uploads to device-preferred memory,
and projected collider bounds limit the scissor or reject offscreen draws.
The sharp overview improved from **40.16 ms to 1.26 ms** GPU time.

The following comparison runs the unchanged
[SuperSplat WGSL](https://github.com/playcanvas/supersplat-viewer/blob/34c137ab1c010b99a1c199b68c36210990331262/src/voxel-debug-overlay.ts)
in Node Dawn 0.6.2/Vulkan on the same RTX 4090, driver 615.78.08, with the exact
same occupancy and camera. Both use 1920×1080, full pixel shading, ten warmups,
30 GPU timestamp samples and RGBA32F/depth32F render attachments. Upstream time
includes compute and its fullscreen composite; native includes its rendering
pass. Upload, compilation, CPU readback, splats and UI are excluded. Upstream
retains its original RGBA8 intermediate and its own edge-width formula.

| Bicycle view | LFS sharp median | SuperSplat median | Speed ratio |
| --- | ---: | ---: | ---: |
| Full surface overview | 1.263 ms | 1.665 ms | 1.32× |
| Full surface, Walk height | 1.355 ms | 1.697 ms | 1.25× |
| Clustered overview | 1.057 ms | 1.345 ms | 1.27× |
| Clustered close-up | 0.711 ms | 0.986 ms | 1.39× |
| Legacy floor-filled overview | 0.252 ms | 0.413 ms | 1.64× |

Overview hit coverage differs at only 3 of 2,073,600 pixels, consistent with the
different boundary epsilon. This is a kernel/pass comparison on one machine,
not an end-to-end benchmark of either application or a universal speed guarantee.
Hardware 2×2 preview is deliberately excluded from this quality comparison.

Native vertical capsule intersection takes about **1 µs/query** on 1,000 seeded
near-scene positions. Axis-aligned capsules use exact interval distance; rotated
capsules retain exact piecewise-quadratic segment/box distance after a cheap
conservative rejection. Upstream's iterative penetration resolver is a different
operation and is not presented as an equivalent timing.

See `tests/benchmarks/collision/README.md` for reproducible render and query
benchmarks. Neither benchmark retrains or modifies a project.

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

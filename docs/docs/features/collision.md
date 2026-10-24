---
title: Collision meshes and walk testing
---

Open **Tools > Collision** with your splat scene loaded. Generation uses all
visible, loaded splats and excludes deleted splats. Stop training first.
No external `splat-transform` executable is required.

## Quick setup

The same surface-first setup works for rooms, outdoor scans and objects:

1. Leave **Auto** enabled for a practical starting resolution based on the visible
   Gaussian surface footprint and source count. Large aerial scans need larger
   cells than small object scans. The resolved size is reported after generation.
2. Use the single **Cell size (metres)** field or its adjacent scrubber for an exact
   custom value. Editing either turns Auto off. Custom resolution is never retried
   at a larger size silently. Calibrate **Scene units / metre** under Advanced if
   the reconstruction is not already metric.
3. Start with **Surface cleanup**. It removes exceptional volume and footprint-area
   outliers without selecting a central crop or dropping disconnected components.
   **Keep all splats** disables this statistical cleanup.
4. Press **Generate**, inspect the boxes, then expand **Walk and edit** and choose
   **Enter walk**. **Overlay strength** and **Cutaway height** reveal inner surfaces.
   Select an existing collider to regenerate it, or choose **New collider** for a
   separate node.

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

The popup follows the Logging utility window: one compact settings toolbar and
one generation action bar, with a scrollable body. Cell size, opacity and source
filtering appear once. **Inspect collision**, **Walk and edit**, and **Advanced**
are collapsible. Scale, filter tuning, filling, seed and triangle-surface options
live under Advanced. Progress and error reports are outside those sections.

Progress names the current CPU or GPU task and shows the work-stage count,
percentage, elapsed time and time since the last progress update. Completed,
cancelled and failed operations retain their final status. A failure includes
recovery advice and expandable **Error details** with the failed task, source
count, relevant settings and exact diagnostic. **Copy details** copies that report.

The initial seed comes from the camera, in data-world XYZ. **Use camera position**
under Advanced filtering refreshes it. Keep it at standing eye height above the desired floor. Strict clustering
chooses the nearest coarse component; exterior filling needs an enclosed seed;
capsule carving needs enough clearance there. The navigation up-axis is respected.

## Saving and exporting

Collision nodes have a distinct subdivided-cube icon in the scene graph. Right-click
one and choose **Export…** to open the separate **Export collision** window, also available
from **File → Export → Collision…**. The File menu uses the selected
collision node, or automatically chooses the scene's only collider; select one
explicitly when the scene contains several. Exporting uses
an immutable payload snapshot on a worker, with progress, cancellation and error
details. It does not regenerate occupancy or change the scene.

- **Save your `.licht` project** to embed the collision occupancy, generation
  settings, preview controls and node transform. No external sidecar is required
  when reopening a project; deferred hydration preserves the collider too.
- **Export for SuperSplat…** writes a version 1.1 `.voxel.json` / `.voxel.bin`
  pair directly from occupancy. It does not build triangles or round-trip through
  GPU buffers. In SuperSplat Studio, open **Assets → Collision → Upload**, select
  both files, and upload the pair alongside your splat.
- **Export surface (.glb)…** writes a standard triangle GLB. If a surface is already
  built, that surface is exported. Otherwise the exporter builds the optional
  surface using the node's saved surface settings (4× cell size by default).
  To choose another detail or shape, build the surface under **Advanced → Triangle
  surface** before exporting.
  No normals, GPU upload or scene-node replacement is needed for that export.

Both exports bake the complete node/parent world transform and the PlayCanvas PLY
import rotation (`Rz(180°)`). Export your splats with the same world transforms so
the pair aligns. Scene units are preserved; SuperSplat recommends one unit per
metre for walking. GLB accepts arbitrary affine transforms and corrects reflected
triangle winding. The voxel format has no transform field: export preserves exact
cells for axis permutations/reflections, translation and uniform scale. A rotated,
sheared or nonuniformly scaled grid fails explicitly, with guidance to align the
scene and regenerate or use GLB; it is never silently resampled.

The format's 24-bit node and leaf indices are checked before writing. If the export
exceeds them, increase collision cell size and regenerate. Temporary files are
removed on failure/cancellation. Replacing a pair requires confirmation for both
companions, publishes metadata last, and restores prior files on a publication
failure. As in SuperSplat, navigation outside the exported grid is treated as solid
by its viewer; LFS Walk retains its own existing out-of-grid behavior.

References: [SuperSplat collision uploads](https://developer.playcanvas.com/user-manual/supersplat/studio/collision/)
and the [voxel format specification](https://developer.playcanvas.com/user-manual/splat-transform/voxel-format/).

## Advanced controls

Advanced settings retain independent access to the pipeline parameters:

| Stage | Defaults at one scene unit per metre |
| --- | --- |
| Source filtering | Surface cleanup; optional connected source region at resolution `1` |
| Strict opacity region (optional) | Resolution `1`, opacity `0.999`, minimum contribution `0.1` |
| Voxelization | Auto cell size (minimum `0.05` metres), accumulated opacity `0.1` |
| Common setup | Surface shell; all filling and carving off |
| Optional room sealing | Exterior fill size `1.6`; carve height `1.6`, radius `0.2` when explicitly enabled |
| Optional solid ground | Floor fill size `1.6` |
| Surface | Smooth or exposed voxel faces; separate surface resolution multiplier |

Dimensional settings and the Walk capsule scale with **Scene units / metre**.
Changing it rescales these settings, not existing generated nodes. Regenerate to
apply changes. Exact cell size is edited in metres in the toolbar; its scene-unit
equivalent is shown in the summary. Existing saved colliders retain exact resolution
and strict filtering semantics unless Auto or the new filtering mode is selected.

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
The `0.1` default thresholds the accumulated voxel opacity, not each individual
splat's opacity. Several faint splats can jointly form a solid voxel. The optional
strict cluster stage separately uses `0.999` coarse opacity and `0.1` minimum
contribution, matching the published upstream defaults.
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

**Surface cleanup** uses far-out Tukey fences (Q3 + 3 IQR) in log Gaussian
thickness and equivalent-volume radius, with at least an eightfold scale margin.
A very thin sheet (thickness below one quarter of Q1) is exempt from the volume
criterion; a long axis alone is not enough to reject it. Fewer than 128 usable
splats bypass the volume/thickness test.

A separate footprint test catches very broad, thin splats that volume alone
misses. The footprint estimate is the largest of the three transformed axis-pair
three-sigma ellipse areas:
`9 * pi * max(length(cross(axis_i, axis_j)))`. With at least 16 usable splats,
the automatic cutoff is Q3 plus the larger of three IQRs or a 64-fold area margin
in log space. Statistics are invariant to uniform scene scale and rotation.
**Advanced > Source filtering > Max splat area (m²)** sets an explicit cutoff; `0`
restores automatic detection. A custom cutoff also works on smaller sources.
The limit applies to **Surface cleanup** and **Connected region**. Generation
reports the oversized count and actual cutoff, and the native selection retains
the excluded source indices. Source positions, opacity and visibility are untouched;
this ignores oversized splats for collision rather than moving them out of view.
If everything is excluded, the error suggests raising the limit or choosing
**Keep all splats**, and the existing collider is preserved.

This is a heuristic: unusual legitimate multi-scale
geometry can be rejected, so the removed count is reported and **Keep all splats**
is always available. Source splats are never modified.

**Connected region** additionally builds a sparse graph of source-centre cells.
All 26 immediate neighbours connect, including diagonal surface observations;
Gaussian support boxes do not bridge the graph. This avoids requiring a sparse
surface to reach 99.9% opacity before it can be structural. A seed inside an
observed cell respects that component. Otherwise, the most populated component
within two connectivity cells of the nearest observed support is selected, so a
nearby isolated floater does not automatically beat a substantial surface.
This is an explicit region filter: disconnected objects can be excluded. It does
not dilate or fill the final collision cells.

**Strict opacity region** preserves the earlier SuperSplat-style algorithm:
coarse opacity voxelization, nearest occupied seed and six-connected flood fill.
Its opacity and minimum-contribution controls are shown only in that mode.
A scan with weak, thin or fragmented support may select a tiny component at the
strict 0.999 default; inspect the retained count or choose Connected region.

In Strict opacity region, following [upstream cluster selection](https://github.com/playcanvas/splat-transform/blob/4f90012be983078010d631fa0572baa99d5c1965/src/lib/voxel/filter-cluster.ts),
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

Storage uses an adaptive occupancy tree, not a dense world allocation.
Sparse archives now use version-3 preorder ternary topology with short run tokens;
it writes directly from the tree and decodes directly into native child blocks.
Versions 1 (dense RLE) and 2 (sparse hex tags) remain readable. This avoids the
old two-nodes-per-hex-character expansion that exceeded the JSON chapter limit
on large scans. Cleanup notes and resolved automatic settings survive reload. The CPU
reference evaluates 8³ bricks; CUDA evaluates 4³ bricks. GPU traversal uses uniform regions and packed 4-cubed leaf
masks. Finite limits apply to actual allocated work: 10 million source Gaussians,
512,000,001 CPU tree words (about 1.91 GiB), 8 million floor columns,
32,000,001 floor-height nodes, 64 million surface vertices and 384 million indices.
Grid axes are limited to 2^21 cells. CUDA uses at most 1,048,576 child regions per
GPU batch, with parent queues retained per depth. Compressed occupied output is
bounded to 64 million records. GPU buffers must fit device limits.
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

Step height is a maximum climb, not a fixed upward jump. Walk searches for the
minimum lift that clears the lower capsule, sweeps the full body upward and
forward, then settles onto support. Ground following snaps down only to support
within the step budget; larger ledges use gravity. Partial wall travel and vertical
contacts are refined to keep the capsule clear of walls and thin ceilings.

The displayed eye follows grounded height changes with a critically damped offset,
independently of the physical collision body. Horizontal movement and airborne
falling continue to follow the body. Camera settling keeps redraw active after
movement keys are released, and an upper-cap clearance sweep limits the eye offset
under overhangs. These behaviours respect navigation up and Scene units / metre;
they do not add hover clearance or manufacture a floor over missing collision.

In a native 60 FPS regression with a 5 cm raised floor patch, the previous 25 cm
step setting caused a 24.73 cm single-frame camera rise. Minimum stepping limits
the physical peak to the actual 5 cm patch (maximum frame rise 1.25 cm), while the
displayed eye's maximum frame rise is 4.93 mm. This is a synthetic motion regression,
not a full-scene performance or surface reconstruction measurement.

Pan, zoom and pivot glides are disabled while walking. Home, focus, camera browsing
and external teleports leave Walk. Hidden colliders do not collide. Anisotropic
scale/shear uses conservative capsule clearance. Walk tests the voxel volume,
so it can be more conservative than the optional smooth surface. Capsule queries
skip sparse empty regions and use exact segment/box distance, rather than scanning
all cells with iterative per-cell distance searches.

## Robust generation measurements (Forest 001709)

The full aligned five-million-splat `images_4` training snapshot was measured on
an RTX 4090. These are native generation timings, excluding file parsing,
training, GUI upload and optional triangle meshing:

- Auto + Surface cleanup: **5.68 s**, resolved **0.28 m**, **161,762,050** occupied
  cells, **669,886,756 bytes** canonical occupancy; **2.28 GiB** peak process RSS.
- Explicit 0.20 m + Surface cleanup: **12.67 s** before the final linear validation
  optimization; **404,009,408** cells, **1,426,784,516 bytes** occupancy.
- Cleanup retained **4,997,972 / 5,000,000** source splats. No crop or ROI was used.
- Connected source selection retained **4,824,405** splats near the road seed;
  this is an optional region selection, not full-scene coverage.
- Explicit 0.05 m exceeds the current 64M compressed-output-record budget on this
  entire aerial scan. It fails explicitly; custom detail is never coarsened.

Those initial cleanup measurements predate footprint-area rejection. With the
additional area test, the same full source completes Auto generation in
**5.192 s** at **0.28 m**, producing **153,886,846** occupied cells and
**631,785,700 bytes** occupancy, with **2.28 GiB** peak process RSS. It retains
**4,997,159 / 5,000,000** splats: **2,838** oversized-area and **3** additional
volume outliers are ignored. The automatic three-sigma area cutoff is
**533.5 m²**. This changes the input selection and is not an identical-output
speed comparison.

The corresponding native GUI check completes generation in **4.649 s** and
snapshot/preview adoption in **6.8 s**, with **153,886,796** occupied cells and
**602.5 MiB** canonical occupancy. The existing collider UUID is retained; the
Model stays visible at 5M splats and all 1,874 camera world transforms are unchanged.
An intentionally tiny custom area limit also exercises the real UI failure path:
all sources are excluded, the previous collider remains, and the panel offers
specific recovery advice. The validation copy is then restored to automatic area
detection; pristine training and alignment handoff files are untouched.

GPU traversal uses bounded depth-first batches instead of a scene-wide frontier.
Parent queues stay on-device; only occupancy records and small counters return.
Conservative lower-density proofs emit uniform occupied regions before leaf-cell
sampling; uncertain boundaries still use the exact original sampler. Sparse
validation checks canonical sibling-block ownership and depth with one byte per
eight-node block, reducing validation scratch memory eightfold. Auto estimates
resolution from the median two-sigma projected Gaussian footprint and source count,
with a target of 32M surface samples. It is a starting estimate, not a guarantee
that every source or optional fill/surface operation fits its resource budget.

Reproduce native timings with the `bench_collision_generation` opt-in CMake target
and a native source snapshot (13 float32 values per row: centre xyz, three
column-major one-sigma axes, alpha). The benchmark accepts `none`, `cleanup`,
`connected`, `strict`, and `auto`, logs stage timestamps, and reports actual size,
source count, occupancy and memory payload separately from parsing and meshing.

For an equal-resolution comparison, the full 1M Bicycle snapshot was run three
alternating trials per version at 5 cm with filtering/filling off, matching
`-O3 -mavx2 -mfma` and CUDA fast-math settings. Median generation fell from
**1.517 s to 1.286 s (15.2% less time)**. Every run produced the same
**122,325,186 occupied cells** and **154,557,764-byte** canonical tree. Whole-process
peak RSS was similar (~672 MiB before, ~676 MiB after); the eightfold scratch
reduction is specific to validation, not an eightfold total-memory claim. The GPU voxelization stage (including transfers)
was slightly slower on this small source; CPU validation savings
produced the end-to-end improvement. The forest Auto result above uses a different
resolution and must not be presented as an equal-detail speedup.

The actual GUI full-forest run completed native generation in **5.263 s**
(**7.5 s** through snapshot/preview adoption), resolved 0.28 m, with
**161,762,005** occupied cells and **638.8 MiB** occupancy. Original Model=5M
remained visible. All **1,874 camera world transforms** matched their pre-generation
fingerprint. The small occupancy difference from the standalone PLY benchmark
comes from native snapshot/PLY reconstruction and grid-frame floating point;
this is not the equal-input Bicycle comparison.

A separate `forest-collision-adaptive-validation.licht` saved and reopened through
actual archive hydration with both Model=5M and the collider restored, all 1,874
cameras present, Auto still enabled at 0.28 m, and the cleanup count preserved.
The prior hex encoding produced an 87,110,676-byte JSON chapter that exceeded the
64 MiB reader limit; the version-3 encoding reopens successfully under that same
limit. The pristine training project, trained PLY and aligned handoff are unchanged.

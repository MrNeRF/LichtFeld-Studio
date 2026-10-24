---
title: Collision meshes and walk testing
---

## Generate and inspect

Open **Tools > Collision** with your splat scene loaded and training stopped.
Generation uses all visible, loaded splats, excluding deleted splats. No external
`splat-transform` process is required.

1. Leave **Auto** enabled to estimate cell size from source count and Gaussian
   footprint. To choose an exact resolution, scrub or type **Cell size (metres)**;
   editing disables Auto. **Opacity** defaults to `0.1`.
2. Start with **Surface cleanup**, then press **Generate**. Select an existing
   collider to regenerate it, or choose **New collider** to keep both.
3. Inspect using **Rendering > Collision > Voxel boxes**. Opacity and cutaway
   update immediately without rebuilding or changing collision accuracy.
4. Choose **Walk Camera** in the viewport toolbar, or enable Walk under
   **Rendering > Collision > Walk**. Any loaded collision box or collision surface
   enables Walk, including hidden nodes.

The progress bar names the current CPU/GPU stage; percentages are not time
estimates. **Cancel** preserves the existing collider. Failures include recovery
advice and expandable **Error details**; **Copy details** copies the diagnostic.
Source edits or project changes invalidate in-flight results.

The default generates occupancy and a preview, without extracting triangles.
Colliders support transforms, parenting, visibility, duplication, deletion and undo.
To edit geometry, leave Walk, hide the collider and edit source splats with the
normal selection/delete/crop/transform tools, then regenerate. Individual voxel or
face editing is not implemented.

## Advanced settings

The same surface-first setup works indoors and outdoors. Filling is optional:
extruding vegetation downwards can create dense pillars rather than surface-shaped
collision. **Reset to surface collision** disables clustering, filling and carving
while retaining scale, cell size and seed. Existing saved occupancy is unchanged
until regeneration.

| Setting | Behavior |
| --- | --- |
| Scene units / metre | Calibrates arbitrary reconstruction scale. Rescales dimensional generation and Walk settings, not the source scene. |
| Auto resolution | Estimates practical detail; exact custom resolutions are never silently coarsened. |
| Opacity | Thresholds accumulated voxel opacity, not each splat's opacity. Faint overlapping splats can jointly occupy a cell. |
| Surface cleanup | Ignores exceptional volume and footprint-area outliers without cropping or discarding disconnected components. |
| Connected region | Adds source-centre connectivity filtering near the seed. Disconnected objects can be excluded. |
| Strict opacity region | Uses upstream-style coarse opacity clustering: resolution `1`, opacity `0.999`, minimum contribution `0.1`. |
| Keep all splats | Disables source cleanup/filtering. |
| Max splat area (m²) | Explicit three-sigma footprint cutoff for cleanup/connected modes; `0` uses a scale-relative statistical cutoff. |
| Exterior fill | Seals gaps and fills boundary-connected exterior space. Skips with a warning if the seed is outside or consumed by sealing. Default size `1.6` metres. |
| Fill solid ground | Extrudes supports downwards and patches surrounded columns; off by default. Patch size starts at `1.6` metres. |
| Avoid overhead floor columns | Restricts extrusion to supports at least `1.6` metres below the seed. This does not identify ground beneath vegetation. |
| Capsule carving | Retains space reachable by the capsule from the seed. Height `0` disables it; radius defaults to `0.2` metres. |

**Use camera position** refreshes the seed in data-world XYZ, respecting navigation
up. Place it near the desired surface; for carving it denotes the capsule centre.
A sparse scan may form a tiny region at strict `0.999` opacity: use Connected region
or Surface cleanup instead. Filtering never changes source splats; excluded counts
and resolved settings are recorded in the result.

Cleanup uses far-out Tukey fences in log thickness/equivalent-volume radius with
an eightfold scale margin; thin sheets are exempt from volume rejection. The area
test uses `9π max(|axis_i × axis_j|)` and at least a 64-fold margin. Small samples
bypass automatic tests. These are heuristics: use Keep all splats if legitimate
multi-scale geometry is excluded.

Connected region connects all 26 neighbouring source-centre cells without using
large Gaussian support boxes as bridges. An occupied seed retains its component;
otherwise nearby support population breaks ties with isolated floaters. Strict
mode instead floods six-connected coarse occupied cells, retains splats centred
inside, and tests contribution/overlap for the others.

## Preview and triangle surfaces

The preview ray-marches sparse occupancy to the first surface. It uses translucent,
axis-shaded gray faces and dark screen-space edges; selected nodes tint blue, and
inside-solid exits tint orange. Hidden edges do not accumulate. It overlays splats
without recoloring them or testing splat depth, following the
[SuperSplat viewer overlay](https://github.com/playcanvas/supersplat-viewer/blob/ece8201c6a0dbea2dcccc418ec63c4217c7b2956/src/voxel-debug-overlay.ts).
Intersecting independent overlays remain draw-order dependent.

**Advanced > Triangle surface > Build surface node** creates a separate mesh with
its own visibility and transform. Smooth uses marching cubes with coplanar
simplification; Faces follows exposed voxel faces. The default `4×` cell multiplier
conservatively coarsens only the surface and may close fine gaps; `1×` retains exact
voxel detail and can produce very large meshes. Both node types retain the original
occupancy for Walk, even after the voxel source node is deleted.

## Walk

**Rendering > Collision > Walk** contains live head-height, radius, speed and step
scrubs. Unsafe capsule edits are rejected with an error. **WASD** moves, **Shift**
runs, and the normal look drag turns. **Escape** or another camera mode exits Walk.
Viewport wheel zoom, pan and pivot glides are disabled while walking.

Step height is a maximum climb. Walk finds the minimum required lift, sweeps upward
and forward while preserving wall/ceiling clearance, then settles onto support.
Small drops follow ground; larger ledges use gravity. A separate smoothed display
camera reduces voxel-step bumps without moving the physical capsule through walls
or ceilings. Sparse capsule queries use segment/box distance. Nonuniform transforms
use conservative clearance. Walk tests occupancy, so a smooth triangle surface may
appear less conservative than the actual collision.

## Saving and exporting

Saving `.licht` embeds occupancy, settings, display controls, transforms and any
built surface, including deferred hydration. No sidecar is needed to reopen it.

Right-click a collider and choose **Export…**, or use **File > Export > Collision…**.
Select a node when several colliders exist. Exports run asynchronously with
progress, cancellation and errors:

- **Export for SuperSplat** writes a version 1.1 `.voxel.json` / `.voxel.bin` pair.
  Upload both under **SuperSplat Studio > Assets > Collision > Upload**.
- **Export surface (.glb)** uses an existing surface or builds one at the node's
  saved detail/shape. It does not replace the node or change Walk occupancy.

Both bake the complete node world transform and PlayCanvas PLY import rotation
(`Rz(180°)`). Export splats with matching transforms. Units are preserved; calibrate
metric scale for walking. GLB supports affine transforms and reflected winding.
The voxel format supports exact axis permutations/reflections, uniform scale and
translation. Other transforms fail explicitly rather than silently resampling;
align/regenerate or use GLB. Its 24-bit indices are checked before publication.
Replacing exports requires confirmation; failed/cancelled writes clean temporary
files and restore previous companions. SuperSplat treats space outside its grid as
solid; LFS Walk retains its own out-of-grid behavior.

## Implementation and limits

Snapshots copy geometry/deletion state, excluding SH/color/optimizer data. CUDA
voxelizes coarse/final fields with conservative bounds, a Gaussian BVH and packed
4³ bricks. Accumulated extinction converts to opacity as `1 - exp(-density)` using
the closest voxel-box point to each Gaussian centre. CPU preparation, filtering,
optional filling/carving and canonical-tree assembly remain; the entire pipeline
is not GPU-resident. CPU evaluation is retained as a correctness reference.

Sparse storage allocates occupied structure rather than the dense world grid.
Archives write version-3 compressed preorder topology and retain readers for dense
RLE/version-2 sparse projects. GPU traversal skips uniform regions and uses packed
leaf masks. Explicit budgets preserve cancellation and actionable failures:
32M source Gaussians, 512,000,001 CPU tree words, 64M compressed CUDA output records,
8M floor columns, 32,000,001 floor-height nodes, 64M surface vertices, 384M indices,
and 2²¹ cells per axis. GPU buffers must fit device limits. Snapshot/allocator/
surface allocations add peak memory; the legacy dense reader is limited to 16M cells.

### Measured performance

Garden was trained at `images_4`, SH 0 and MRNF defaults for 30,000 iterations,
reaching 20M Gaussians in 921.126 seconds. A separate collision project reopened
and hydrated the model and all 185 cameras with original scene records unchanged.

On an i9-14900K / RTX 4090 (driver 615.78.08), three alternating baseline/tuned
trials used identical frozen input, Auto + Surface cleanup, opacity `0.1`, no
clustering/filling/carving. Auto resolved `0.05` scene units, without asserting
metric scale. Cleanup excluded 13 area outliers; no crop or resolution change.

| Measurement | Baseline | Tuned |
| --- | ---: | ---: |
| Source extraction median | 0.590 s | 0.124 s |
| Generation median | 8.592 s | 4.103 s |
| Generation peak RSS | 3.748 GiB | 3.747 GiB |
| 1080p box overlay, three views | 0.746–0.958 ms | 0.676–0.861 ms |

Generation excludes parsing/extraction/serialization; baseline/tuned occupancy is
byte-identical (9,872,885 cells). Extraction saves 672 MiB of output capacity, not
whole-process RSS. Rendering measures only the warmed overlay GPU pass; first-frame
wall time remains 121–125 ms. These are not whole-viewport FPS. Coverage differs
at one of 6,220,800 pixels across the views from grid-boundary rounding. See
[the native benchmark guide](../../../tests/benchmarks/collision/README.md)
for repeatable inputs and timing boundaries.

Upstream references: [collision recommendations](https://developer.playcanvas.com/user-manual/supersplat/studio/collision/),
[parameters](https://developer.playcanvas.com/user-manual/splat-transform/collision/),
[voxel format](https://developer.playcanvas.com/user-manual/splat-transform/voxel-format/).

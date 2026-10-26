# Collision

Tools > Collision (below Splat Simplify) generates a voxel child from one selected loaded splat,
including hidden sources. Hover the settings for details. Source geometry and appearance are
unchanged; the collider follows later source transforms and has its own editable transform.

Cell sizes use scene units. Auto reports its resolved size after preparation and invalidates it when
inputs change. Generation covers the complete filtered source, excluding storage padding;
opacity/area/cleanup affect occupancy without hidden ROI or coarsening.

All visible colliders preview independently of selection as first-hit overlays over the scene.
Axis-shaded gray faces have dark edges; the default 40% face opacity gives 80% edge opacity, 0% is
invisible and 100% fully opaque. Inside solid space, the first empty voxel marks an orange exit.
Perspective, orthographic and panorama views honor node transforms. Lazy preview failures are logged
without disabling other rendering.

Rendering > Collision contains preview opacity and live Walk settings. Enter Walk from either camera
toolbar; hidden loaded colliders still participate. Unsupported spawns are rejected and movement
stops at unsupported ledges.

File > Export > Collision and collider context-menu export share a popup that captures its target
when opened. Export opens the standard LFS save dialog; picker cancellation writes nothing.
Generation/export expose job progress, errors and cancellation, and consume immutable snapshots.

SuperSplat pair export supports signed-axis rotations/reflections and uniform scale; other grid
transforms require source alignment and regeneration. Export storage bounds expand by at most three
empty cells per axis to align 4-cell blocks; native bounds and `sceneBounds` retain the logical
extent. Pair replacement rolls back ordinary failures/cancellation, but is neither crash-atomic nor
concurrent-reader-atomic.

Generation allows 64 million sparse output records and checks available GPU memory before growth,
compaction and sorting. Limit errors suggest a larger cell size; explicit resolution is never
silently changed. Packed address limits and SuperSplat's 24-bit node indices also apply, with
provably impossible export topology rejected before sorting. Projects store shared collision
occupancy once in binary payloads.

Deferred: Faces/Smooth, GLB, fill/carve, connected components, cutaway, coarsening and in-place
regeneration. Unsupported command options fail explicitly. See [Command
line](../cli.md#voxel-collision) for CLI and Python APIs.

# Collision

Tools > Collision generates a voxel child from one selected loaded splat, even if hidden.
The source stays unchanged; the collider follows its source and has an editable transform.
Cell size uses scene units. Auto reports its size and targets 8M surface samples, leaving
headroom without guaranteeing exportability. Opacity and cleanup/area filters control occupancy.

Visible colliders preview regardless of selection. Rendering > Collision controls preview
opacity (40% by default) and live Walk settings. Enter Walk from either camera toolbar;
hidden loaded colliders still participate. Spawns need clearance and supporting ground.
Movement stops at unsupported ledges.

File > Export > Collision or the collider context menu exports a SuperSplat v1.1 pair:
`.voxel.json` metadata and `.voxel.bin` occupancy. Keep both files together.
Replacing an existing pair requires approval; cancelling the picker writes nothing.
Export supports uniform scale and signed-axis rotations/reflections, not arbitrary
rotation, shear or nonuniform scale: align the source and regenerate first.

Generation allows 64 million sparse records; GPU memory and export's 24-bit indices
can impose lower limits. Increase cell size after a limit error; explicit sizes are never
silently changed. See [Command line](../cli.md#voxel-collision) for CLI and Python APIs.

---
title: Train a region of interest
---

# Train a region of interest

A crop box on the training model makes it a region of interest (ROI): training concentrates on the splats and pixels inside the box, and `--crop` can remove everything outside it at the end. Use it to train one part of a scene, or a large scene piece by piece.

```sh
LichtFeld-Studio -d ./dataset -o ./output --headless --cropbox -0.6,-0.2,-0.1,0.55,0.3,1.9
LichtFeld-Studio -d ./dataset -o ./output --headless --cropbox -0.6,-0.2,-0.1,0.55,0.3,1.9 \
  --cropbox-loss-weight 0 --crop --export ply
```

## The box

- `--cropbox x0,y0,z0,x1,y1,z1` adds an enabled crop box under the training model with that minimum and maximum corner, in the dataset's coordinates. Each minimum must be below its maximum.
- `--cropbox-rotation rx,ry,rz` turns that box about its centre, see [Rotated boxes](#rotated-boxes).
- The box is a normal crop box node. The saved project keeps it, and the GUI shows and edits it like one drawn by hand: add it with the crop tool, then move, rotate or scale it with the gizmo or the numeric fields.
- With `--eval`, `training_config.json` records the box as `cropbox` (and `cropbox_rotation`). `--eval-mask cropbox` scores only the pixels the box covers, see [Evaluation](evaluation.md).
- `--cropbox` and `--cropbox-rotation` set up the box when a headless run starts a fresh model. With `--resume` the box comes from the project; in the GUI, draw it with the crop tool.

### Rotated boxes

`--cropbox-rotation rx,ry,rz` takes three angles in degrees about the dataset's X, Y and Z axes, the same frame as the `--cropbox` corners. The box keeps its size and centre from `--cropbox` and is turned about the centre by `Rx(rx) · Ry(ry) · Rz(rz)`: first about Z, then Y, then X. Any orientation can be expressed, so the box can follow a rotated tile grid or a tilted scene. For example, `0,0,30` turns the box's X edge toward `(cos 30°, sin 30°, 0)`. Without the flag the box stays aligned to the axes.

Everything uses the turned box: the loss and learning-rate weighting outside it, `--cropbox-growth-inside`, `--crop`, `--eval-mask cropbox` and the saved project, which opens as a normal rotated crop box.

The GUI's transform panel shows the box in the viewer's axes, where Y and Z point the other way. Its Rotation fields therefore show the same turn as `rx + 180, ry, rz` (wrapped to ±180°, while `|ry|` is below 90°), and its Position as `(cx, −cy, −cz)`. To copy angles from the panel to the command line, subtract 180 from X. A box without rotation shows Rotation `±180, 0, 0`.

Use `=` when a value starts with a minus sign, for example `--cropbox-rotation=-15,0,40`.

## Training options

| Flag | GUI | Default | Meaning |
|---|---|---|---|
| `--cropbox-loss-weight` | Outside ROI loss weight | `0.1` | Weight of the pixel losses for camera rays that miss the box. `0` ignores those pixels completely; raise it when the surrounding scene should still shape the result. |
| `--cropbox-lr-scale` | Rejected splat LR scale | `0.1` | How fast splats away from the box still learn. Splats inside the box, and those in front of or behind it in a training view, always learn at full rate, so no floaters form in the box. Strategy noise, decay and resets stay active. |
| `--cropbox-growth-inside` | ROI growth inside | `0` | Share of new splats placed inside the box. `0` is automatic: growth follows the error, which already favours the box. `1` puts all growth into the box; lower values leave some for the scene around it. |
| `--crop` | Crop to ROI at end | off | Remove the splats outside the box when training completes, and leave them out of every export. See below. |

The options only matter with an enabled crop box on the training model. In the GUI they are in the viewport toolbar's **Crop ROI Settings** while a crop box is selected.

With a crop box, the loss shown in the progress bar, the GUI's loss chart and to Python measures the image inside the box only: the training image loss over the pixels that see the box, plus the usual regularizers. Mask opacity and alpha-consistency penalties retain their training weights, including `--cropbox-loss-weight`. The image pixels outside the box still train with that weight, but their photometric loss does not change the number, and a view that does not see the box is left out of it.

## Cropping with `--crop`

- **Training completes:** every splat outside the box is removed from the model (with an inverted box, the splats inside it). The final project save and every export hold only the ROI. The project can still be resumed, from the cropped model.
- **Training is stopped or fails:** nothing is removed. This covers the GUI's Stop, Ctrl+C or SIGTERM in a headless run and any other stop request, even one during the last iteration. The saved project keeps every splat, so a resumed run continues with all of them.
- **Exporting:** with `--crop` on and an active box, `--export`, File > Export, Python `lf.export_scene` and MCP `scene.export_*` / `scene.save_ply` write only the splats inside the box. They crop a copy: the model in memory, the scene and the project stay as they are, so a stopped run can be exported and still resumed. Only the training model is cropped; other nodes in the same export are written whole. With `--exclude-export`, both filters apply.
- In the GUI, exports follow **Crop to ROI at end** as currently set, so switching it off after a stop exports the whole model. Headless exports use the run's settings.
- If no splat lies inside the box, the export fails instead of writing an empty file.
- `--crop` needs a crop box: `--cropbox`, or a resumed project that has one. Without an active box it does nothing. Gallery publishing, `convert` and project saves are not cropped.

## Large scenes in tiles

### Automatic tiling

To train a whole dataset, request a total tile count such as `--auto-tile 4`. LichtFeld Studio makes adaptive cells from sparse geometry and SfM observations; `--auto-tile 2x2` retains a fixed grid. For adaptive cells it compares candidate cuts using sampled point density, camera support across the boundary, spatial coverage by multiple cameras, and image resolution. These are planning estimates, not measured training time or VRAM. It measures robust scene bounds, aligns them with 3D PCA by default, then first trains one global coarse prior and initializes every local tile from it. Each tile runs in a separate process, so its GPU resources are released before the next tile starts. The generated core boxes, overlapping training contexts, coarse-prior settings, depth-mask paths and per-tile camera counts are recorded in `output/tiles.json`; completed tile jobs record their duration. Successful runs are exported to `output/tiles/tile_i_j/tile_i_j.ply`; after all cells finish, `output/merged.ply` contains the combined model. The merge checks that every binary PLY has the same vertex properties, then copies the vertex data in bounded chunks. It does not load the tile models into RAM or VRAM. The merged file still needs disk space for all tile data, and its PLY header is rebuilt with a fresh minimal provenance stamp.

```sh
LichtFeld-Studio -d ./dataset -o ./output --auto-tile 4 --iter 30000
```

The default orientation uses 3D PCA on sparse points. Choose `--tile-align planar_yaw`, `planar_pitch`, or `planar_roll` to align only in the XY, XZ, or YZ plane, respectively. `--tile-align axis_aligned` disables PCA. Bounds and orientation use sparse points when available, falling back to camera centres if there are too few points. The advanced CLI option `--tile-source points|both|cameras` changes only the positions used for bounds and orientation, not the evidence used for adaptive splitting or view selection. Camera-only bounds can exclude scene geometry when cameras occupy a small part of the reconstruction. `--tile-quantile 0.02` controls the outlier trim.

Use `--auto-tile 3x2` for an explicit grid, or `--tile-size 30` to derive the grid from a target width in dataset units. `--tile-overlap 0.15` is the default; accepted values are 0 through 0.3. It expands only the **training context**: the exported ownership cores remain disjoint, with a deterministic half-open rule at interior seams. `--no-tile-pca` aligns the grid to the dataset axes, and `--no-tile-merge` leaves just the individual exports. The coarse pass defaults to 5000 iterations at resize factor 4; tune it with `--tile-coarse-iterations` and `--tile-coarse-resize`. Auto-tiling uses fresh dataset training with `--centralize off`; it cannot be combined with a manual `--cropbox`, resumed training, `--init`, `--random` or added splats. If a tile fails or the user interrupts training, merging stops and the completed tile exports remain available.

Both partition modes use robust scene bounds. Adaptive partitioning repeatedly splits the region with the highest estimated workload. It evaluates midpoint and point/camera quantile cuts on both local axes, favors candidates supported by shared cameras, and compares estimated workload, spatial coverage, and sparse image-space coverage until the requested count is reached. Image-space coverage counts the 16x16 image cells containing in-tile SfM observations in selected views, using native SfM pixel coordinates when available. During a tile job with zero outside-ROI loss, the trainer first uses a conservative 32x32 z-buffer proxy from the coarse Gaussian centres: it downweights a pixel only when a 3x3 neighbourhood is consistently occupied by a nearer Gaussian owned by another core. Empty or ambiguous pixels retain their ROI weight. If this generated file is unavailable, it falls back to the existing conservative 64x64 SfM image-space support mask. These masks are combined with the crop-box ROI weight and existing user masks. This is not a dense rendered-depth loss and is not HUG's learned segmentation. Adaptive partitioning reports an error if the sparse geometry cannot support that many distinct cells. Boundary tiles extend only their **context** X/Y faces by at least 10% of a cell toward a wider percentile of the selected points/cameras (one quarter of the grid's trim quantile, or 0.5% with the default 2% trim). Each extension is capped at half a cell. The local Z bounds are unchanged; with full 3D alignment, local Z need not be vertical. This helps retain finite geometry near the scene edges but does not solve an unbounded sky or background. Preview boxes show the ownership cores; edited plans may additionally supply `context_min` and `context_max`.

Each tile is cropped after training by default. Its initial coarse model contains core Gaussians and all coarse centres in the overlapping context; sparse initialization additionally retains selected-camera observations in that context. Only the core survives export. Tile export also conservatively clips oversized Gaussian extents at the owner boundary, avoiding a long splat from painting its neighbour. `--no-tile-crop` keeps the splats outside its crop box only for unmerged inspection exports; a merged auto-tile run requires the strict core crop. The default `--tile-outside-loss-weight 0` ignores pixels outside the crop box; raise it toward `1` if surrounding scene content should also be trained. `--no-tile-merge` skips the combined PLY but always leaves the individual exports available.

Camera selection scores each view by SfM tracks inside the tile and their spread across a 16x16 image grid. Views with enough track and image evidence are kept. Independently, a pinhole view is kept when the tile's projected 3D volume covers at least 25% of its image, including empty airspace that surface tracks may miss. Additional views cover the tile's observed 4x4x2 spatial cells twice where possible. When SfM evidence is sparse, a 2% projected-volume threshold and a nearest-camera floor of eight provide a fallback. The adaptive planner calls the same selector for each proposed tile, using bounded SfM samples for speed; the final camera allowlist is recalculated from full tracks, so estimated and final counts may still differ. A positive outside-ROI loss weight uses the selected cameras rather than adding every dataset camera. The selected UID and image name pairs are saved beside each tile and checked when the child process loads the dataset; support and coverage counts plus fallback status are recorded in `tiles.json`. With zero outside-ROI loss, each job initializes from points in its box plus sparse points observed by its selected cameras; very sparse tiles retain a small nearest-point floor. Camera selection is recalculated for edited boxes.

Before the global coarse pass, auto-tiling estimates one global brightness correction from shared SfM tracks. It samples up to 2048 observations per image at reduced resolution, compares the same 3D point across images, and solves robust relative gains on the largest connected camera group. Gains are limited to 0.5–2 and saved with camera identities in `output/tile_exposure.json`. The coarse pass and every tile job apply this same gain to their training images in memory; source photos are untouched. Cameras with fewer than eight reliable shared observations, outside the connected group, or with float HDR images keep gain 1. This handles multiplicative brightness differences in encoded RGB, not white balance, vignetting, changing illumination, or arbitrary camera response curves. If learned exposure correction is enabled, it refines the residual from this common baseline; it must not establish an independent baseline per tile. Evaluation against the original photos is not exposure-normalized.

To rerun edited boxes from the GUI, pass `--tile-plan output/auto_tile_plan.json` together with the same `--auto-tile` count or grid. The plan supplies each box's bounds and rotation, so it overrides automatic geometry.

In the GUI, open a dataset and use **Automatic tiling** in the Training panel. Choose an adaptive count, fixed grid, or target tile width; all public tiling controls (overlap, orientation, bounds source, trim, outside-ROI loss, crop/merge, and coarse-pass iterations/resize) are available there. Preview the ownership boxes before choosing **Train All Tiles**. You can move, rotate or resize preview boxes before starting; the edited cores and their preserved context margins are saved to `auto_tile_plan.json` and used for training. The preview inspector uses the same SfM selector as the batch, so it lists the exact camera allowlist for each tile and can temporarily show only that box and those camera frusta. The batch writes its progress to `auto_tiles.log` in the output directory. The GUI launches a separate training process and remains available while it runs. It passes the selected strategy, iteration count, SH degree, maximum Gaussian count, image resize factor, image width limit, and the Exposure Correction toggle to that process; other training options use their CLI defaults.

Each cell uses the existing ROI training controls. The merged model has one mean-owned export per internal boundary instead of retaining two overlap copies. The generated bounds come from sparse geometry and cameras, so they may exclude distant backgrounds or sky. For outdoor scenes, inspect the preview boxes and enlarge their outer edges or vertical extent before training. Disabling crop alone will not reconstruct regions whose pixels have zero outside loss weight.

### Manual tiling

Train each tile as its own run with its own box, ignore the pixels outside it, and crop the result:

```sh
LichtFeld-Studio -d ./dataset -o ./tiles --output-name tile_a --headless \
  --cropbox -10,-5,-10,0,5,0 --cropbox-loss-weight 0 --crop --export ply
LichtFeld-Studio -d ./dataset -o ./tiles --output-name tile_b --headless \
  --cropbox 0,-5,-10,10,5,0 --cropbox-loss-weight 0 --crop --export ply
```

A tile run that is stopped keeps all its splats and can be continued with `--resume`; it is cropped when it completes.

When the tile grid is turned relative to the dataset axes, give each tile its unturned extent around its centre and the grid's angle:

```sh
LichtFeld-Studio -d ./dataset -o ./tiles --output-name tile_a --headless \
  --cropbox=-10,-5,-10,0,5,0 --cropbox-rotation=0,25,0 --cropbox-loss-weight 0 --crop --export ply
```

The box is turned about its own centre, so a turned grid needs each tile's centre placed on the turned grid as well.

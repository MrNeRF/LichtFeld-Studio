---
title: Command line
---

# Command line

## Voxel collision

```sh
LichtFeld-Studio voxelize scene.ply -o scene.voxel.json --voxel-size 0.05
LichtFeld-Studio voxelize scene.sog scene.voxel.json --auto --overwrite
```

Writes a SuperSplat v1.1 `.voxel.json` / `.voxel.bin` pair.

`--voxel-opacity` sets accumulated opacity (default 0.1, range [0,1)). `--filter cleanup` rejects
outliers; `--filter none` disables cleanup. `--max-gaussian-area` is a three-sigma projected area in
squared scene units (0 disables the manual cap). Cleanup may still apply its automatic limit.

See [Collision](features/collision.md) for Auto resolution, full-source coverage, memory/format limits
and cancellation guarantees. Overwrite approval covers both outputs once.

Python: `lichtfeld.io.generate_voxels(path, settings)` and `lichtfeld.io.save_voxels(volume, path,
overwrite=False, transform=[])`. `VoxelSettings` exposes these controls; a progress callback
returning `False` cancels. Optional transforms are 16 column-major data-world values.

## Embed a dataset in a project

`LichtFeld-Studio licht <project.licht> --embed` embeds the dataset currently referenced by a project. To select a
different dataset, provide its path after `--embed`:

```sh
LichtFeld-Studio licht scene.licht --embed
LichtFeld-Studio licht scene.licht --embed ./dataset
```

Relative dataset paths are resolved from the current working directory. COLMAP datasets with a supported sparse or
flat layout and Transforms datasets are accepted. The project is updated in place; the source dataset is left on disk.

Run `LichtFeld-Studio licht --help` for command usage.

## Sharpen training images

`--sharpen-amount` sharpens the training images as they are loaded. [Sharpen training images](features/sharpening.md)
explains the kernels, the other `--sharpen-*` options and how sharpening affects evaluation.

## Reduce color variance

`--color-variance-weight` adds the SteadySplats color variance loss, which lowers the noise of a model in stochastic
renderers. [Color variance loss](features/color-variance.md) explains the loss, `--color-variance-warmup-iters` and the
settings it cannot be combined with.

## Train a region of interest

`--cropbox` limits training to a box, `--cropbox-rotation` turns it, and `--crop` removes everything outside it when
training completes and from exports. [Train a region of interest](features/region-of-interest.md) describes the box and
its rotation, the ROI training options and tiled training.

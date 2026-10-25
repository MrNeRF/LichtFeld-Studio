---
title: Command line
---

# Command line

## Voxel collision

```sh
LichtFeld-Studio voxelize scene.ply -o scene.voxel.json --voxel-size 0.05
LichtFeld-Studio voxelize scene.sog scene.voxel.json --auto --overwrite
```

Writes a SuperSplat v1.1 `.voxel.json` / `.voxel.bin` pair. Cell size uses scene units;
`--auto` chooses a scale-relative size from median transformed splat footprint and source count,
without a fixed metre floor; the resolved cell size is reported. `--voxel-opacity` sets accumulated opacity
(default 0.1, range [0,1)). `--filter cleanup` (default) rejects source outliers;
`--filter none` disables cleanup. `--max-gaussian-area` is in squared scene units,
with zero selecting automatic rejection. Bounds still cover the entire retained source.
There is no hidden region of interest or resolution coarsening.

All shared Gaussian splat formats, including SOG `meta.json` and SSOG folders, are accepted.
Original LOD leaves are used; partially resident out-of-core models fail explicitly.
Datasets, point clouds and meshes are not Gaussian inputs.
Connected filtering, seed/cluster flags, fill/carve, surfaces and GLB export are deferred;
unsupported options are errors. Overwrite approval covers both files once.
Cancellation and ordinary publication failures restore the previous pair; this is not
crash-atomic or atomic for concurrent readers. Export requires a uniformly scaled,
signed-axis-aligned affine grid and reflects cells relative to logical dimensions.

Python uses `lichtfeld.io.generate_voxels(path, settings)` and
`lichtfeld.io.save_voxels(volume, path, overwrite=False, transform=[])`.
`VoxelSettings` exposes the generation controls above. A progress callback returning
`False` cancels. The optional transform is 16 column-major data-world matrix values.

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

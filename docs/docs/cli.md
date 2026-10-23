---
title: Command line
---

# Command line

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

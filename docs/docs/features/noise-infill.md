---
title: Noise infill
---

# Noise infill

Splats trained on a solid object often come out partly transparent. Noise infill removes this false transparency with
noise guided splatting ([arXiv 2510.15736](https://arxiv.org/abs/2510.15736)). Opaque noise splats fill the inside of
the object during training and change color on every step, so wherever the surface is see-through the render shows noise
and the photometric loss makes the surface opaque.

Pass a closed mesh of the object, such as its photogrammetry or scanner mesh, in the dataset's coordinates:

```sh
LichtFeld-Studio -d ./dataset -o ./output --init ./init.ply --noise-infill-mesh ./scan.obj --train
```

In the Training panel, choose the mesh under Initialization.

## What happens

When training starts, the mesh is voxelized and one opaque noise splat is placed in every cell inside it that is at least
two cells clear of its surface. Open boundary loops, such as the underside of an object scanned on a table, are capped
first. The cell size follows from the mesh's volume, for about 200,000 cells inside.

The noise is frozen: it is never moved, densified or pruned. It stays invisible for the first 30% of the training steps,
so the surface forms undisturbed. Then the position learning rate restarts, and every step gives each noise splat a new
color: red, green, blue, cyan, magenta or yellow.

The noise does not count toward `--max-cap` or `--popspa-target-count` and no export carries it. Project files keep it so a run can resume.
Evaluation hides it while scoring. `--noise-infill-mesh` needs `--freeze-lr-scale` at 0.

## Evaluation

[`--eval-green-infill`](evaluation.md#green-infill) shows how solid a surface is, against the same mesh:

```sh
LichtFeld-Studio -d ./dataset -o ./output --init ./init.ply --noise-infill-mesh ./scan.obj --eval-all --eval-green-infill --train
```

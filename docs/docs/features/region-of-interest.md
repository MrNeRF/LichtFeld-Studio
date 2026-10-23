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
- The box is a normal crop box node. The saved project keeps it, and the GUI shows and edits it like one drawn by hand: add it with the crop tool, then move, rotate or scale it with the gizmo or the numeric fields.
- With `--eval`, `training_config.json` records the box as `cropbox`. `--eval-mask cropbox` scores only the pixels the box covers, see [Evaluation](evaluation.md).

## Training options

| Flag | GUI | Default | Meaning |
|---|---|---|---|
| `--cropbox-loss-weight` | Outside ROI loss weight | `0.1` | Weight of the pixel losses for camera rays that miss the box. `0` ignores those pixels completely; raise it when the surrounding scene should still shape the result. |
| `--cropbox-lr-scale` | Rejected splat LR scale | `0.1` | How fast splats away from the box still learn. Splats inside the box, and those in front of or behind it in a training view, always learn at full rate, so no floaters form in the box. Strategy noise, decay and resets stay active. |
| `--cropbox-growth-inside` | ROI growth inside | `0` | Share of new splats placed inside the box. `0` is automatic: growth follows the error, which already favours the box. `1` puts all growth into the box; lower values leave some for the scene around it. |
| `--crop` | Crop to ROI at end | off | Remove the splats outside the box when training completes, and leave them out of every export. See below. |

The options only matter with an enabled crop box on the training model. In the GUI they are in the viewport toolbar's **Crop ROI Settings** while a crop box is selected.

## Cropping with `--crop`

- **Training completes:** every splat outside the box is removed from the model (with an inverted box, the splats inside it). The final project save and every export hold only the ROI. The project can still be resumed, from the cropped model.
- **Training is stopped or fails:** nothing is removed. This covers the GUI's Stop, Ctrl+C or SIGTERM in a headless run and any other stop request, even one during the last iteration. The saved project keeps every splat, so a resumed run continues with all of them.
- **Exporting:** with `--crop` on and an active box, `--export`, File > Export, Python `lf.export_scene` and MCP `scene.export_*` / `scene.save_ply` write only the splats inside the box. They crop a copy: the model in memory, the scene and the project stay as they are, so a stopped run can be exported and still resumed. Only the training model is cropped; other nodes in the same export are written whole. With `--exclude-export`, both filters apply.
- In the GUI, exports follow **Crop to ROI at end** as currently set, so switching it off after a stop exports the whole model. Headless exports use the run's settings.
- If no splat lies inside the box, the export fails instead of writing an empty file.
- `--crop` needs a crop box: `--cropbox`, or a resumed project that has one. Without an active box it does nothing. Gallery publishing, `convert` and project saves are not cropped.

## Large scenes in tiles

Train each tile as its own run with its own box, ignore the pixels outside it, and crop the result:

```sh
LichtFeld-Studio -d ./dataset -o ./tiles --output-name tile_a --headless \
  --cropbox -10,-5,-10,0,5,0 --cropbox-loss-weight 0 --crop --export ply
LichtFeld-Studio -d ./dataset -o ./tiles --output-name tile_b --headless \
  --cropbox 0,-5,-10,10,5,0 --cropbox-loss-weight 0 --crop --export ply
```

A tile run that is stopped keeps all its splats and can be continued with `--resume`; it is cropped when it completes.

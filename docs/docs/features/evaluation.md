# Evaluation

LichtFeld Studio can score a reconstruction against its input images while it trains, or later from a saved project. It reports PSNR, SSIM, LPIPS and optionally FLIP for every evaluated image and for the whole set.

## Running an evaluation

| Flag | Meaning |
|---|---|
| `--eval` | Hold out every 8th image in image-name order (`--test-every N` changes the interval) and evaluate those images. |
| `--eval-all` | Train on every image and evaluate all of them. The scores then show fit, not generalization. |
| `--eval-steps 7000,30000` | Iterations to evaluate at. The last iteration is always evaluated. |
| `--eval-flip` | Also compute FLIP and save its error maps. |
| `--no-save-eval-images` | Skip the comparison images. |

Fresh datasets use case-sensitive lexicographic image-name order, starting with the first image. A `.licht` project keeps its saved holdout when resumed or evaluated; explicitly passing `--test-every` rebuilds it in image-name order. Legacy `.resume` checkpoints retain record-order splitting.

In the GUI the same options sit in the **Dataset** section of the training panel once **Evaluate** is enabled.

## Evaluating a finished model

```bash
LichtFeld-Studio eval output/project.licht -o output/eval
LichtFeld-Studio eval model.ply -d data/scene --images images_4 --enable-mip -o output/eval
```

`eval` scores a model without training it and writes only the files below to `-o`; the model file is not changed. A `.licht` project or `.resume` checkpoint is scored at its last checkpoint with the dataset and settings it was trained with. A splat file (`.ply`, `.spz`, `.sog`, `.ssog`, `.usd`) is scored as saved, as step 0, so pass the dataset and the settings it was trained with: `--images`, `--undistort`, `--max-width`, `--enable-mip`, `--gut`, `--bg-color`, `--add-splat`, `--ppisp-sidecar`. Every option on this page applies and replaces the stored value.

A model trained without `--eval` has seen the held-out images, so its scores show fit; `--eval-all` scores every image.

## Output files

Everything is written to the output directory:

```
<output_directory>
├── metrics.csv              one row per evaluated iteration
├── metrics_report.txt       best and final scores in plain text
├── per_image_metrics.json   every image at every evaluated iteration
└── eval_step_30000
    ├── IMG_0001.png               ground truth | render, unscored pixels black; with a mask three rows:
    │                              unmasked, mask applied, inverted mask applied
    ├── ...
    └── flip
        ├── IMG_0001.png           FLIP error map, with --eval-flip
        └── ...
```

Images are named after their input image. When two input images share a name, the later ones get their evaluation index appended.

A caption band above each row names its two images, for example `Ground Truth, Mask` and `Render, Mask`; the image pixels themselves are not drawn over.

`metrics.csv` has the columns `iteration, psnr, ssim, lpips, time_per_image, num_gaussians, normal_angle_deg, depth_absrel, bias_r, bias_g, bias_b, bias_corr_r, bias_corr_g, bias_corr_b, flip`. Columns without a value are left empty.

`per_image_metrics.json` maps each image name to its size and a list of evaluations:

```json
{
  "IMG_0001.JPG": {
    "width": 1600,
    "height": 1066,
    "evaluations": [
      {
        "step": 30000,
        "split": "test",
        "psnr": 25.41,
        "ssim": 0.812,
        "lpips": 0.184,
        "flip": 0.093,
        "masked": true,
        "bit_depth": 8,
        "evaluated_pixel_fraction": 0.62,
        "validity_mask_applied": false
      }
    ]
  }
}
```

`split` is `test` for held-out images and `train` with `--eval-all`. An image that could not be scored keeps its entry with a `skipped_reason`.

With `--eval-mask-report`, every listed pixel set has its own object in the evaluation, named `masked`, `unmasked` and `inverted`, each with that set's `psnr`, `ssim`, `lpips`, `flip` and `evaluated_pixel_fraction` (see [Pixel sets and the headline set](#pixel-sets-and-the-headline-set)). Only `step`, `split`, `bit_depth` and `validity_mask_applied` stay outside them, and the `masked` true/false entry is left out because the object names the set. An image that could not be scored has its `skipped_reason` beside them and no set objects. `metrics.csv` holds the headline set's scores.

## Metrics

- **PSNR** and **SSIM** measure pixel and structural agreement. Higher is better.
- **LPIPS** compares deep features and follows perceived similarity more closely. Lower is better. The weights are downloaded on first use; `--no-download` disables that.
- **FLIP** estimates how different the two images look when flipped between on a display, per pixel, in [0, 1]. Lower is better. It uses the published LDR-FLIP model for a 0.7 m wide 3840 pixel display viewed from 0.7 m (67 pixels per degree) and treats pixel values as sRGB. The error map uses the magma colour scale: black is no visible difference.

### Color variance training

MCMC, MRNF and IGS+ support the optional SteadySplats color variance loss through the FastGS training renderer. Set **Color Variance Weight** and **Color Variance Warm-up** in the training panel's loss controls, or use `color_variance_weight` and `color_variance_warmup_iters` in an optimization config. Weight `0` disables it by default. The warm-up is the number of completed training iterations before the loss starts, and follows training step scaling. Both settings are saved in checkpoints.

This implements the ground-truth color objective in [SteadySplats, Appendix 13](https://arxiv.org/html/2610.05576v2#A13): sum the RGB squared error of each visible splat, weighted by its alpha-compositing contribution, then average over image pixels. It uses evaluated fragment colors, including the renderer's color clamps. It does not normalize by accumulated opacity or add a background penalty; the paper's variance interpretation assumes full coverage.

The first version requires unmasked FastGS training with appearance correction and mip filtering off, and no crop box loss weighting or POPSpa phase. MRNF enables exposure correction by default; turn it off to use this loss. A weight of `1` after `15000` warm-up iterations in a `30000`-iteration run is a starting experiment based on the paper, not a replacement for a strategy's own settings.

Compare runs from the same warm-up checkpoint with the same image-exposure budget. Check sorted PSNR, SSIM and LPIPS as well as noise in a stochastic renderer: reducing color variance can reduce detail and peak reconstruction quality. LFS's existing 3DGS and 3DGUT viewport choices do not show the stochastic sampling noise this loss targets.

## Evaluation masks

A mask restricts all metrics to a part of each image, for example the object without its background. Pixels outside it are left out of PSNR, FLIP and the colour bias; SSIM uses only windows that lie fully inside the mask; LPIPS averages each feature layer's distances over the mask, with the outside black in both images.

| `--eval-mask` | Selects |
|---|---|
| `mesh:<file>` | Pixels covered by a mesh (OBJ, PLY, ...) in the dataset's coordinates. |
| `bbox:x0,y0,z0,x1,y1,z1` | Pixels covered by an axis-aligned box given by its minimum and maximum corner, in the dataset's coordinates. |
| `cropbox` | Pixels covered by the training model's crop box, as placed when training starts. |
| `masks:<folder>` | One mask image per evaluated image, found by the image's name, see [Mask folders](#mask-folders). White pixels are scored. |
| `depth:near,far` | Rendered pixels that are solid and whose depth lies between `near` and `far`. |
| `points` or `points:radius,close` | Pixels around the initial point cloud: every point is drawn as a disk of `radius` pixels (default 2) and gaps up to twice `close` pixels (default 3) are closed. |
| `points:<file>` | Pixels around the points of a splat or point cloud PLY in the dataset's coordinates, drawn like `points` with the default radius and closing. Use it when the initial point cloud is not the subject, for example when the subject is added with `--add-splat`. |
| `splat:<file>` | Pixels a splat PLY covers when rendered with its positions, sizes, rotations and opacities: pixels whose rendered opacity reaches `--eval-mask-opacity` (default 0.85; 0.5 adds a thin rim of background, values near 1 trim the subject's edges). |
| `none` | No mask; clears a mask stored in a resumed project. |

To score the pixels outside the mask, list `inverted` first in `--eval-mask-report`, see below. Mesh, box, crop box and point masks follow each camera's lens, including distortion when evaluating distorted images.

### Pixel sets and the headline set

`--eval-mask-report` lists the pixel sets every view is scored against, all in the same evaluation. It needs `--eval-mask`.

| Set | Pixels | Same scores as |
|---|---|---|
| `mask` | The mask, as above. | the evaluation with the mask alone |
| `unmasked` | Every valid pixel of the view. | `--eval-mask none` |
| `inverted` | The valid pixels outside the mask. | a mask that covers exactly those pixels |

Only the listed sets are scored. The first one is the headline set: its scores go to `metrics.csv`, to the score lines of the log and of `metrics_report.txt`, to the GUI and to the first table of `compare_report.txt`. The order of the other sets makes no difference. Without `--eval-mask-report`, the mask alone is scored.

| `--eval-mask-report=` | Scored | Headline |
|---|---|---|
| (not given) | mask | mask |
| `mask,unmasked,inverted` | all three | mask |
| `inverted,mask,unmasked` | all three | inverted |
| `inverted` | inverted | inverted |
| `unmasked,mask` | unmasked and mask | unmasked |

Valid pixels are all pixels of the view, except when distorted images are evaluated with `--undistort`: the pixels the undistorted render cannot reach are then left out of every set. `inverted` is the exact complement of the thresholded mask inside the valid pixels; for a mask folder with grey edges it can differ along those edges from masks whose grey values were inverted before thresholding.

Each view is rendered and its reference decoded once for all sets, and SSIM reuses its full-image map. LPIPS and FLIP run once per set, because they blank the pixels outside the set before comparing, so each listed set adds one LPIPS pass (and one FLIP pass with `--eval-flip`) per view. The FLIP error maps show the headline set.

The scores of every listed set are written to `per_image_metrics.json` (example below) and `compare.json`, each set under its own name. The other sets also get their own lines in the log and in `metrics_report.txt`, and their own tables in `compare_report.txt`. In the GUI, **Headline Set** picks the headline set, and **Also Score Mask**, **Also Score Unmasked** and **Also Score Inverted** add the others; they appear once an evaluation mask is set. Without an applied mask the sets are not scored, every pixel is, and a warning says so.

```json
{
  "step": 30000,
  "split": "test",
  "bit_depth": 8,
  "validity_mask_applied": false,
  "masked": {"psnr": 25.41, "ssim": 0.812, "lpips": 0.184, "evaluated_pixel_fraction": 0.62},
  "unmasked": {"psnr": 23.87, "ssim": 0.781, "lpips": 0.203, "evaluated_pixel_fraction": 1.0},
  "inverted": {"psnr": 22.95, "ssim": 0.764, "lpips": 0.215, "evaluated_pixel_fraction": 0.38}
}
```

`points` uses the points the model was initialized from, either the dataset's sparse points or the file given with `--init`. A resumed project reads them again from there, so that file or dataset must still be in place.

In the GUI, **Browse** picks a mesh, **Mask Folder** a mask folder and **Use Crop Box** the crop box; the text field accepts any of the specifications above.

### Mask folders

`masks:<folder>` is a normal path to the folder that holds the masks; it does not have to be inside the dataset. An absolute path is used as given. A relative path is resolved from the directory the command is started in, not from the dataset, and is stored as an absolute path, so a resumed project still finds it. A folder that does not exist stops the run before anything is evaluated.

Each mask is found from the name the image has in its dataset:

| Images come from | Image name | Example |
|---|---|---|
| A COLMAP dataset | The name stored in the model, which is the path below the images folder | `cam_a/frame_0001.exr` |
| A `transforms.json` dataset | The image's file name | `frame_0001.png` |
| `compare` with a reference folder | The path below the reference folder | `cam_a/frame_0001.exr` |
| `compare` with a reference dataset or `--colmap` | The name stored in the COLMAP model | `cam_a/frame_0001.exr` |

The mask folder is searched including its subfolders:

1. The image's path inside the folder comes first, so a folder that mirrors the image layout always works: `eval_masks/cam_a/frame_0001.png`.
2. Otherwise the file name alone is looked up anywhere in the folder, so a flat folder works too: `eval_masks/frame_0001.png`. If that file name exists in more than one subfolder, the match is ambiguous and the image is reported as having no mask. Mirror the image layout whenever file names repeat across folders.

At each place the mask may be named like the image including its extension (`frame_0001.exr`), with the extension replaced by `.png`, `.jpg`, `.jpeg` or `.mask.png` (`frame_0001.png`), or with one of those appended (`frame_0001.exr.png`). An EXR image can therefore use a PNG mask. Names are matched ignoring case, and `/` and `\` both separate folders.

A mask has the resolution of its image file and is scaled together with the image. White pixels are scored; `--eval-mask-report=inverted` scores the others. Every evaluated image needs a mask: a missing one stops the evaluation and names the image.

For example, a COLMAP dataset whose two cameras both have a `frame_0001`:

```
dataset/
├── images/
│   ├── cam_a/frame_0001.exr
│   └── cam_b/frame_0001.exr
└── sparse/0/
eval_masks/
├── cam_a/frame_0001.png
└── cam_b/frame_0001.png
```

```bash
LichtFeld-Studio --headless -d dataset -o output --eval --eval-mask masks:eval_masks
LichtFeld-Studio compare dataset/ renders/ --eval-mask masks:/data/eval_masks
```

The masks mirror the layout because `frame_0001.png` alone would match both cameras.

## Distorted images

With `--undistort`, training uses undistorted images. `--eval-space` chooses what evaluation compares against:

- `distorted` (default): the original images, with the render warped into the original lens. Pixels outside the undistorted view are not scored and are black in the saved images.
- `undistorted`: the undistorted training images.

Without `--undistort` the images are evaluated as they are; with `--gut` a distorted camera is rendered in its own lens.

## Bit depth

Renders are rounded to the precision of the reference before scoring, so a perfect reconstruction scores perfectly. `--eval-bit-depth` chooses that precision:

| Value | Render and reference compared at |
|---|---|
| `auto` (default) | The reference file's own encoding: 8-bit, 16-bit, or full float for floating point files such as EXR. |
| `8` | 8 bits. |
| `16` | 16 bits. 16-bit references keep their full precision. |
| `float` | Full precision, values clamped to [0, 1]. |

The bit depth used for each image is recorded as `bit_depth` in `per_image_metrics.json` (32 for float).

## Comparing image sets

`compare` scores images made elsewhere, for example renders from another tool, against reference images with the same metrics and masks as evaluation:

```bash
LichtFeld-Studio compare plates/ renders/ -o scores
LichtFeld-Studio compare gt_dataset/ render_dataset/ --eval-mask mesh:hero.obj --eval-flip --save-images
LichtFeld-Studio compare gt_dataset/ render_dataset/ --eval-mask splat:init.ply --eval-space undistorted
LichtFeld-Studio compare plates/ renders/ --colmap sparse/0 --eval-mask masks:mattes/ --undistort
```

The reference and the test are each an image, a folder of images, or a COLMAP dataset folder with `images/` and `sparse/0` (or `sparse/`). Images are paired by file name without extension, ignoring case, so `shot_0101.png` matches `shot_0101.exr`; inside datasets the path below `images/` counts too. Files without a counterpart are listed and skipped. A test set at another resolution with the same aspect ratio is compared at the smaller of the two resolutions, both downscaled with the Lanczos filter training uses; any other size mismatch skips the pair.

The reference cameras come from the reference dataset, from `--colmap`, or, for a reference folder without cameras, from the test dataset. Every mask except `masks:<folder>` needs them. When both sets are datasets, the test cameras tell how the test images were made:

- The same lens as the reference (at any resolution): both images are compared as they are with `--eval-space distorted` (default), or both are undistorted the same way with `--eval-space undistorted`.
- A pinhole camera while the reference has lens distortion: the test was rendered undistorted. `distorted` warps it into the reference lens and leaves pixels it does not cover unscored; `undistorted` undistorts the reference onto the test's pinhole grid.

Any other lens difference stops the comparison. Test cameras placed elsewhere than their reference cameras are reported; masks follow the reference cameras. For a test folder without cameras, `--undistort` says it was rendered with the reference cameras undistorted by LichtFeld Studio. When no camera has lens distortion, both spaces are the same.

| Flag | Meaning |
|---|---|
| `-o <folder>` | Where the reports go; default `<test>_compare` next to the test images. |
| `--eval-mask`, `--eval-mask-opacity`, `--eval-mask-report` | `masks:<folder>`, `mesh:<file>`, `bbox:...`, `points:<file>` or `splat:<file>`, as in training. The crop box, depth and initial point masks need a trained model. |
| `--colmap <sparse>` | COLMAP model with the cameras of a reference folder that is not a dataset, matched by image name. |
| `--eval-space` | `distorted` or `undistorted`, see above. |
| `--undistort` | The test folder was rendered with the undistorted reference cameras. |
| `--eval-bit-depth`, `--eval-flip` | As in training. |
| `--save-images` | Save reference and test side by side to `compare_images/` (captioned `Ground Truth` and `Test`, three rows with a mask), and FLIP error maps to `compare_images/flip/` with `--eval-flip`. |

The scores go to `compare.json` (the settings, the mean scores and each image's PSNR, SSIM, LPIPS, FLIP, size, bit depth and scored pixel fraction; with `--eval-mask-report` the scores of each pixel set in its own `masked`, `unmasked` or `inverted` object, as in `per_image_metrics.json`) and `compare_report.txt` (the table the command prints); a rerun into the same folder replaces them and `compare_images/`. Both image sets are read with the same decoder, so identical files score PSNR 100 and SSIM 1. Values are compared as stored: a linear EXR and an sRGB PNG of the same image do not match.

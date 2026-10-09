# Evaluation

LichtFeld Studio can score a reconstruction against its input images while it trains, or later from a saved project. It reports PSNR, SSIM, LPIPS and optionally FLIP for every evaluated image and for the whole set.

## Running an evaluation

| Flag | Meaning |
|---|---|
| `--eval` | Hold out every 8th image (`--test-every N` changes the interval) and evaluate those images. |
| `--eval-all` | Train on every image and evaluate all of them. The scores then show fit, not generalization. |
| `--eval-steps 7000,30000` | Iterations to evaluate at. The last iteration is always evaluated. |
| `--eval-flip` | Also compute FLIP and save its error maps. |
| `--no-save-eval-images` | Skip the comparison images. |

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
    ├── IMG_0001.png               reference | render, unscored pixels black; with a mask three rows:
    │                              as rendered, mask applied, inverted mask applied
    ├── IMG_0001_flip.png          FLIP error map, with --eval-flip
    └── ...
```

Images are named after their input image. When two input images share a name, the later ones get their evaluation index appended.

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

## Metrics

- **PSNR** and **SSIM** measure pixel and structural agreement. Higher is better.
- **LPIPS** compares deep features and follows perceived similarity more closely. Lower is better. The weights are downloaded on first use; `--no-download` disables that.
- **FLIP** estimates how different the two images look when flipped between on a display, per pixel, in [0, 1]. Lower is better. It uses the published LDR-FLIP model for a 0.7 m wide 3840 pixel display viewed from 0.7 m (67 pixels per degree) and treats pixel values as sRGB. The error map uses the magma colour scale: black is no visible difference.

## Evaluation masks

A mask restricts all metrics to a part of each image, for example the object without its background. Pixels outside it are left out of PSNR, FLIP and the colour bias; SSIM uses only windows that lie fully inside the mask; LPIPS averages each feature layer's distances over the mask, with the outside black in both images.

| `--eval-mask` | Selects |
|---|---|
| `mesh:<file>` | Pixels covered by a mesh (OBJ, PLY, ...) in the dataset's coordinates. |
| `bbox:x0,y0,z0,x1,y1,z1` | Pixels covered by an axis-aligned box given by its minimum and maximum corner, in the dataset's coordinates. |
| `cropbox` | Pixels covered by the training model's crop box, as placed when training starts. |
| `masks:<folder>` | One mask image per input image, matched by file name. White pixels are scored. |
| `depth:near,far` | Rendered pixels that are solid and whose depth lies between `near` and `far`. |
| `points` or `points:radius,close` | Pixels around the initial point cloud: every point is drawn as a disk of `radius` pixels (default 2) and gaps up to twice `close` pixels (default 3) are closed. |
| `points:<file>` | Pixels around the points of a splat or point cloud PLY in the dataset's coordinates, drawn like `points` with the default radius and closing. Use it when the initial point cloud is not the subject, for example when the subject is added with `--add-splat`. |
| `splat:<file>` | Pixels a splat PLY covers when rendered with its positions, sizes, rotations and opacities: pixels whose rendered opacity reaches `--eval-mask-opacity` (default 0.85; 0.5 adds a thin rim of background, values near 1 trim the subject's edges). |
| `none` | No mask; clears a mask stored in a resumed project. |

`--eval-mask-invert` scores the pixels outside the mask instead. Mesh, box, crop box and point masks follow each camera's lens, including distortion when evaluating distorted images.

`points` uses the points the model was initialized from, either the dataset's sparse points or the file given with `--init`. A resumed project reads them again from there, so that file or dataset must still be in place.

In the GUI, **Browse** picks a mesh, **Mask Folder** a mask folder and **Use Crop Box** the crop box; the text field accepts any of the specifications above.

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
| `--eval-mask`, `--eval-mask-invert`, `--eval-mask-opacity` | `masks:<folder>`, `mesh:<file>`, `bbox:...`, `points:<file>` or `splat:<file>`, as in training. The crop box, depth and initial point masks need a trained model. |
| `--colmap <sparse>` | COLMAP model with the cameras of a reference folder that is not a dataset, matched by image name. |
| `--eval-space` | `distorted` or `undistorted`, see above. |
| `--undistort` | The test folder was rendered with the undistorted reference cameras. |
| `--eval-bit-depth`, `--eval-flip` | As in training. |
| `--save-images` | Save reference and test side by side to `compare_images/` (three rows with a mask), and FLIP error maps with `--eval-flip`. |

The scores go to `compare.json` (the settings, the mean scores and each image's PSNR, SSIM, LPIPS, FLIP, size, bit depth and scored pixel fraction) and `compare_report.txt` (the table the command prints); a rerun into the same folder replaces them and `compare_images/`. Both image sets are read with the same decoder, so identical files score PSNR 100 and SSIM 1. Values are compared as stored: a linear EXR and an sRGB PNG of the same image do not match.

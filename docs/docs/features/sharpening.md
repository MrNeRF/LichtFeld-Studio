---
title: Sharpen training images
---

# Sharpen training images

LichtFeld Studio can sharpen the training images while it loads them. Use it when the photos are slightly soft (focus, motion blur, lens softness or downscaling) and the reconstruction should come out crisper. Sharpening is off by default.

```sh
LichtFeld-Studio -d ./dataset -o ./output --train --sharpen-amount 0.5
LichtFeld-Studio -d ./dataset -o ./output --train --sharpen-amount 0.5 --sharpen-threshold 0.01
LichtFeld-Studio -d ./dataset -o ./output --train --sharpen-kernel laplacian4 --sharpen-amount 0.2
```

## Options

| Flag | Default | Range | Meaning |
|---|---|---|---|
| `--sharpen-amount` | `0` | 0 to 5 | Strength. `0` turns sharpening off; any other value turns it on. |
| `--sharpen-kernel` | `usm` | `usm`, `laplacian4`, `laplacian8` | The filter, see below. |
| `--sharpen-radius` | `1` | 0.3 to 10 | Blur radius (Gaussian sigma) of `usm` in pixels of the trained image, after `--resize_factor` and `--max-width`. The Laplacian kernels ignore it. |
| `--sharpen-threshold` | `0` | 0 to 1 | Brightness differences below this value stay unsharpened. `0` sharpens everything. |

### Kernels

- `usm` is unsharp masking with a Gaussian blur, the filter behind "Unsharp Mask" in image editors. Small radii (0.5 to 1) sharpen fine texture; larger radii (2 to 5) raise contrast along broader edges.
- `laplacian4` compares each pixel with its four side neighbours. At amount 1 it is the textbook kernel `0 −1 0 / −1 5 −1 / 0 −1 0`.
- `laplacian8` compares each pixel with all eight neighbours. At amount 1 it is `−1 −1 −1 / −1 9 −1 / −1 −1 −1`.

The Laplacian kernels are much stronger than `usm` at the same amount, because amount 1 is already the full textbook kernel. Start at 0.1 to 0.25 with them.

### The filter

All three kernels sharpen luma only. Luma is computed from the stored values (gamma-encoded, as in the image file) with the ITU-R BT.709 weights:

```
Y = 0.2126 R + 0.7152 G + 0.0722 B
```

Each kernel computes a low-pass version `L` of `Y`, and the detail is `d = Y − L`. Where `|d|` is smaller than `--sharpen-threshold`, `d` is set to 0 (a hard threshold, like the threshold of an image editor's Unsharp Mask). Every colour channel `c` then becomes

```
c' = clamp(c + k · amount · d, 0, 1)
```

rounded back to the 8-bit or 16-bit grid of the image. Adding the same `k · amount · d` to R, G and B keeps the hue, so no colour fringes appear.

| Kernel | Low-pass `L` | `k` |
|---|---|---|
| `usm` | Gaussian blur of `Y` with σ = `--sharpen-radius`: weights `exp(−x² / 2σ²)` truncated at ±⌈3σ⌉ pixels (at most 30) and normalized to sum 1, applied as a horizontal and then a vertical pass | 1 |
| `laplacian4` | Mean of the 4 side neighbours | 4 |
| `laplacian8` | Mean of the 8 neighbours | 8 |

Pixels outside the image repeat the nearest edge pixel.

- `usm` is unsharp masking in the textbook form `g = f + k · (f − f_blur)`. `--sharpen-amount` is `k`: amount 1 is classic unsharp masking, and amounts above 1 are high-boost filtering.
- For `laplacian4`, `4 · (Y − mean of the 4 neighbours) = −∇²Y` with the 4-neighbour Laplacian, so the filter is the classic Laplacian sharpening `c' = c − amount · ∇²Y`. `laplacian8` is the same with the 8-neighbour Laplacian.

### References

- W. F. Schreiber, "Wirephoto quality improvement by unsharp masking," *Pattern Recognition* 2(2), 117–120, 1970. [doi:10.1016/0031-3203(70)90007-5](https://doi.org/10.1016/0031-3203(70)90007-5). The original unsharp masking paper.
- R. C. Gonzalez and R. E. Woods, *Digital Image Processing*, 4th ed., Pearson, 2018, Section 3.6. Laplacian sharpening, unsharp masking and high-boost filtering, including the two 3×3 kernels above.
- A. Polesel, G. Ramponi and V. J. Mathews, "Image enhancement via adaptive unsharp masking," *IEEE Transactions on Image Processing* 9(3), 505–510, 2000. Shows how plain unsharp masking amplifies noise in flat areas, which `--sharpen-threshold` counters.
- ITU-R Recommendation BT.709, *Parameter values for the HDTV standards for production and international programme exchange*. The luma weights.

### Choosing values

- Start with `--sharpen-amount 0.25` to `0.5` and the default `usm` radius. Above 1, edges get halos and noise turns into speckle, and the model learns both.
- For noisy photos (high ISO, dark scenes), add `--sharpen-threshold 0.01` to `0.02` so flat areas and noise stay untouched.
- The radius is measured in the trained image: after `-r 4`, a radius of 1 spans 4 pixels of the original photo.

## What gets sharpened

- Only brightness. The same change is added to red, green and blue, so sharpening adds no colour fringes and does not boost colour noise.
- Only training images. Evaluation (`--eval`) and the ground-truth view in the viewer use the original photos, so PSNR, SSIM and LPIPS are measured against the unsharpened images. PSNR usually drops a little as the amount grows, because the model learns the sharpened images.
- Each image is sharpened once, on the GPU, after resizing and undistortion, when it is first loaded. The image cache keeps the sharpened copy, so later epochs cost nothing extra. A 4K image takes well under a millisecond.
- `--use-16bit` images are sharpened at 16-bit precision.

## In the GUI and in projects

The same options are in the **Dataset** section of the training panel: **Sharpen** (the kernel), **Amount**, **Radius** (unsharp mask only) and **Threshold**. They can be changed before training starts and are locked while it runs.

The settings are saved in the project and in its checkpoints, so `--resume` keeps sharpening the same way. A sharpen flag passed together with `--resume` replaces the stored value. Projects saved before this option existed load with sharpening off.

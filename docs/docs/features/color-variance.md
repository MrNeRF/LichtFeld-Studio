---
title: Color variance loss
---

# Color variance loss

The color variance loss makes the splats that cover a pixel agree on its color, instead of reaching the photo by blending very different colors. A model trained this way shows less noise in a stochastic renderer, which draws only some of the splats per sample. LichtFeld Studio's viewport blends every splat by default, so it shows that noise only in its [stochastic modes](stochastic-rendering.md); the loss is meant for models that will be viewed with stochastic rendering. It is off by default.

```sh
LichtFeld-Studio -d ./dataset -o ./output --headless --no-exposure-correction \
  --color-variance-weight 1 --color-variance-warmup-iters 15000
```

## Options

| Flag | GUI | Default | Meaning |
|---|---|---|---|
| `--color-variance-weight` | Color Variance Weight | `0` | Weight of the loss, from `0` to `8`. `0` turns it off. |
| `--color-variance-warmup-iters` | Color Variance Warm-up | `0` | Completed training iterations before the loss starts. It must be below the iteration count and scales with `--steps-scaler`. |

In the GUI both are in the training panel's loss controls. Config files and Python use `color_variance_weight` and `color_variance_warmup_iters`, and checkpoints and projects keep both.

A weight of `1` from iteration 15000 of a 30000-iteration run follows the paper, which trains a normal model first and then refines it with the loss. Higher weights reduce the variance further but blur fine detail.

## Requirements

The loss works with MCMC, MRNF and IGS+ on the FastGS training renderer. Training refuses to start with it when any of these is on:

- 3DGUT (`--gut`)
- the mip filter
- appearance correction: exposure correction, the bilateral grid or PPISP. MRNF turns exposure correction on by default, so add `--no-exposure-correction`.
- training masks
- POPSpa sparsification

With a crop box region of interest, the loss weights the pixels the same way as the photometric loss, so `--cropbox-loss-weight` applies to both. See [Train a region of interest](region-of-interest.md).

## The loss

For each pixel, the loss adds up the squared RGB distance between every visible splat's color `c_i` and the training image's color `C_gt`, weighted by the splat's blending weight `P(i) = alpha_i * T_i` (its opacity at the pixel times the transmittance in front of it), and averages over the pixels ([SteadySplats](https://arxiv.org/html/2610.05576v2), Eq. 30):

```
L_var = 1 / (H * W) * sum over pixels (x, y) of sum over splats i of P_xy(i) * |c_i - C_gt(x, y)|^2
```

Training adds `weight * L_var` to the photometric loss.

The colors are the splats' evaluated colors, including the renderer's clamps. The loss does not divide by the accumulated opacity and leaves the background out. With a region of interest, the plain mean over the pixels becomes the weighted mean with the region weights.

## Cost

The loss trades some reconstruction quality for lower variance. On the Mip-NeRF 360 garden scene (MRNF, 30000 iterations, exposure correction off), a weight of `1` from iteration 15000 lowered PSNR from 27.77 to 27.55 dB and SSIM from 0.878 to 0.874. To compare settings, train from the same warm-up checkpoint and check the metrics together with the noise in a stochastic renderer.

## References

- F. Windisch, T. Köhler, L. Radl, C. Wyman, G. Kopanas, B. Kerbl, M. Steinberger. "SteadySplats: Resampling of Low-Variance Gaussians for High-Fidelity Stochastic Rendering." arXiv:2610.05576, 2026. Section 13 derives the gradients.

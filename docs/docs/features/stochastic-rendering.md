---
title: Stochastic rendering
---

# Stochastic rendering

The viewport can draw 3DGS scenes without sorting the splats. **Rendering → Quality & Performance → Advanced → Transparency** selects the mode:

| Mode | Behavior |
| --- | --- |
| Alpha blend | Sorts the splats and blends them. The default. |
| Stochastic | Each pixel shows one splat per frame, picked at random by its opacity. Frames are averaged while the view rests. |
| SteadySplats | Stochastic samples, combined with the samples of neighboring pixels and previous frames, so there is less noise, also in motion. |

Each frame is shown at once. **Refinement frames** (16 by default, up to 1024) is the number of frames rendered after the view or the scene changes; then the viewport idles until the next change. More frames give a cleaner still image.

Python uses `lf.get_render_settings().transparency_mode` (`"alpha_blend"`, `"stochastic"`, `"steady_splats"`) and `stochastic_max_frames`. The MCP tools `render_settings_get` and `render_settings_set` take the same fields. The setting is saved with the session.

How noisy the image is depends on the model: one trained with the [color variance loss](color-variance.md) shows far less noise.

## Limits

The stochastic modes apply to 3DGS without LOD. The viewport keeps alpha blending with 3DGUT, in depth views, while the scene contains a mesh, camera frustums are shown or the align tool picks surfaces, with selection outlines, rings or center markers, and for previews and exports. The grid is depth-tested against the latest sample only, so it looks noisy behind semi-transparent splats.

## SteadySplats

The implementation follows [SteadySplats](https://arxiv.org/html/2610.05576v2), Algorithms 1 to 3:

- A splat survives at a pixel with the probability of its opacity there. The nearest survivor by radial distance is the sample; with no survivor the pixel shows the background.
- The four direct neighbors share their samples. The weights use each splat's opacity at both pixels and the occlusion by up to four distinct splats the pixel sampled recently.
- The previous image is reprojected with the sample's depth and read with a Catmull-Rom filter. Its weight of 0.7 shrinks when it falls outside the color range of the 3 x 3 neighborhood in YCoCg space, a range that narrows under motion. Within two pixels of the border the current frame is used alone.
- The visibility thresholds come from a 128 x 128 x 64 spatiotemporal blue-noise volume that the build generates, offset per splat. Stochastic mode uses white noise.
- While the view rests, the frames are also averaged in a separate buffer that gradually takes over, as in the authors' [web viewer](https://derthomy.github.io/SteadySplats/viewer.html).

## Memory

While a stochastic mode is on, each view needs 32 bytes per pixel (Stochastic) or 96 bytes per pixel (SteadySplats), about 66 MB or 200 MB at 1920 x 1080, plus 4 bytes per splat and a 1 MiB noise volume. Switching back to alpha blend frees them. As in the authors' viewer, SteadySplats stores each of the four recent occluders in 8 bytes: the splat, a 16-bit depth key and its occlusion toward the four neighbors in steps of 1/15.

## References

- F. Windisch, T. Köhler, L. Radl, C. Wyman, G. Kopanas, B. Kerbl, M. Steinberger. "SteadySplats: Resampling of Low-Variance Gaussians for High-Fidelity Stochastic Rendering." arXiv:2610.05576, 2026.
- A. Wolfe, N. Morrical, T. Akenine-Möller, R. Ramamoorthi. "Spatiotemporal Blue Noise Masks." EGSR 2022.

# 3DGUT

3DGUT (3D Gaussian Unscented Transform) is an alternative method of rendering proposed by NVIDIA Research that uses raytracing instead of rasterization. Most significantly, it allows rendering and training with nonlinear projections, like camera models with distortion.

## When to Use
Use 3DGUT when your COLMAP camera model is not PINHOLE or SIMPLE_PINHOLE.

:::warning
RealityScan (formerly RealityCapture) may export datasets whose images are already undistorted while the COLMAP metadata still reports a distorted camera model. LFS automatically normalizes zero-distortion pinhole-family models during import, but stale non-pinhole metadata can still produce incorrect 3DGUT results.
:::

## How to Use
To enable 3DGUT, use the `--gut` flag.

## Supported Camera Models
- SIMPLE_PINHOLE
- PINHOLE
- SIMPLE_RADIAL
- RADIAL
- OPENCV
- FULL_OPENCV
- OPENCV_FISHEYE
- RADIAL_FISHEYE
- SIMPLE_RADIAL_FISHEYE

## Screen-size controls with MRNF

With GUT and MRNF, `max_screen_share` measures the fraction of the image covered by a Gaussian's opacity-tightened projected bounding rectangle, clipped to the image. It includes the camera projection and distortion. The rectangle is a size bound, not an exact count of contributing pixels. A value of `0.1` means ten percent of the image area; `0` or `1` disables the controls.

MRNF preserves coarse coverage while adding detail:

- During growth, `oversize_split_fraction` prioritizes over-limit Gaussians within the existing growth budget. Zero disables that priority.
- Once growth has ended and refinement has begun, refinement clips oversized scales and `screen_share_penalty` applies a soft scale penalty on Adam steps. Zero penalty disables the soft penalty independently of clipping and splitting.

The handoff follows the configured growth schedule, including fill pacing when enabled. Measurements keep their maximum across training views until the next refinement and stop with the refinement window. These controls encourage smaller splats; they do not impose an immediate hard image-area bound or a total intersection-memory limit.

FastGS retains its angular size statistic, so equal numeric limits need not select the same Gaussians across renderers. Switching renderers clears the measurement window to avoid mixing those units.

## References
- [3DGUT: Enabling Distorted Cameras and Secondary Rays in Gaussian Splatting](https://research.nvidia.com/labs/toronto-ai/3DGUT/) - Original paper and project page
- [3dgrut Repository](https://github.com/nv-tlabs/3dgrut) - Reference implementation
- [gsplat 3DGUT pull request](https://github.com/nerfstudio-project/gsplat/pull/667) - Implementation in gsplat which LFS's 3DGUT support is based on

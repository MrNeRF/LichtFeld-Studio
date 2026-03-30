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

## References
- [3DGUT: Enabling Distorted Cameras and Secondary Rays in Gaussian Splatting](https://research.nvidia.com/labs/toronto-ai/3DGUT/) - Original paper and project page
- [3dgrut Repository](https://github.com/nv-tlabs/3dgrut) - Reference implementation
- [gsplat 3DGUT pull request](https://github.com/nerfstudio-project/gsplat/pull/667) - Implementation in gsplat which LFS's 3DGUT support is based on

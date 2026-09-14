/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#pragma once

#include "Common.h"

namespace gsplat_lfs {
    // Geometry channels use camera Z for pinhole cameras and radial distance for
    // fisheye/panoramic cameras. Normals are camera-space, facing the camera.
    void geometry_features_fwd(const float* means, const float* quats, const float* scales,
                               const float* viewmats, const float* rgb, float* features,
                               uint32_t N, uint32_t C, uint32_t channels,
                               CameraModelType model, cudaStream_t stream);
    void geometry_features_bwd(const float* means, const float* quats, const float* scales,
                               const float* viewmats, const float* grad_features, float* grad_rgb,
                               float* grad_means, float* grad_quats,
                               uint32_t N, uint32_t C, uint32_t channels,
                               CameraModelType model, cudaStream_t stream);
    void flatten_scale_grad(const float* scales, float* grad_scales, uint32_t N,
                            float weight, cudaStream_t stream);
    void geometry_camera_rays(float* rays, uint32_t width, uint32_t height,
                              const float* K, CameraModelType model, const float* radial,
                              const float* tangential, const float* thin_prism, cudaStream_t stream);
    void expected_depth_bwd(const float* depth_accum, const float* alpha,
                            float* grad_depth, float* grad_alpha, uint32_t pixels, cudaStream_t stream);
} // namespace gsplat_lfs

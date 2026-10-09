/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include "lfs/training/ops/geometry_types.hpp"

#include <cuda_runtime.h>
#include <vector>

namespace lfs::training::kernels {

    // Projects the anchor cloud into the prior and
    // returns the raw (prior value, camera-space depth) sample pairs. Empty when
    // too few samples land in view. Synchronizes the stream; startup use only.

    [[nodiscard]] std::vector<lfs::gpu_ops::AnchorSample> collect_depth_anchor_samples(
        const float* points_xyz, // [N,3] CUDA
        size_t num_points,
        const float* w2c, // [16] CUDA row-major world-to-camera
        float fx,
        float fy,
        float cx,
        float cy,
        const float* prior, // [H,W] CUDA
        int width,
        int height,
        float near_plane,
        const float aabb_lo[3],
        const float aabb_hi[3],
        cudaStream_t stream = nullptr,
        const DepthCameraProjection& projection = {});

    // Scale-and-shift-invariant depth supervision on alpha-normalized expected
    // depth in inverse-depth space using a fixed per-camera anchor alignment.
    // The loss is alpha-weighted Geman-McClure plus a gradient-alignment term.
    // Emits gradients w.r.t. both the accumulated depth map and the alpha map.
    // anchor: fixed per-camera alignment; invalid or null anchors disable the
    // loss for the image.
    // prior_quantization_step: quantization step of the prior in target units
    // (1/255 for 8-bit priors, 1/65535 for 16-bit, 0 for float). Residuals and
    // gradient-alignment differences inside the quantizer's half-step corridor
    // carry no loss and no gradient; without this, sign gradients drag smooth
    // surfaces onto the prior's staircase (terracing on coarse 8-bit priors).
    void launch_depth_loss(
        const float* rendered_depth_accum,
        const float* rendered_alpha_accum,
        const float* target_depth,
        float* grad_depth,
        float* grad_alpha,
        float* loss_out,
        float* partial_sums,
        int width,
        int height,
        float weight,
        float gradient_term_weight,
        float prior_quantization_step = 0.0f,
        const DepthAnchor* anchor = nullptr,
        cudaStream_t stream = nullptr,
        const float* pixel_weight = nullptr);

} // namespace lfs::training::kernels

/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include "lfs/training/ops/geometry_types.hpp"

#include <cuda_runtime.h>

namespace lfs::training::kernels {

    // Alpha-weighted cosine supervision of the accumulated camera-space normal
    // map against a decoded prior:
    //   L = weight * sum(alpha * (1 - cos(n, t))) / max(sum(alpha), 1)
    // over valid pixels. Emits gradients w.r.t. the accumulated normal map
    // only; the FastGS rasterizer backward also propagates that normal-map
    // gradient through blend weights, so it can affect opacity/position/conic
    // as well as per-Gaussian orientation.
    // Silently disables (loss 0, grads 0) when too few pixels carry a valid
    // prior and coverage — check the kValid slot.
    void launch_normal_loss(
        const float* rendered_normal, // [3*H*W] CHW accumulated normals
        const float* rendered_alpha,  // [H*W]
        const float* target_normal,   // [3*H*W] CHW prior in [-1,1]
        float* grad_normal,           // [3*H*W] CHW, written
        float* loss_out,              // [1], written
        float* partial_sums,          // [normal_loss_partial_count(num_pixels)]
        int width,
        int height,
        float weight,
        cudaStream_t stream = nullptr,
        const float* pixel_weight = nullptr);

} // namespace lfs::training::kernels

/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include <cstddef>
#include <cuda_runtime.h>

namespace lfs::training::kernels {

    // Final/diagnostic slots at the front of the partials buffer. kSlotCount
    // must stay even: the double-precision block partials start right after
    // the float slots and need 8-byte alignment.
    namespace normal_loss_slots {
        constexpr int kValid = 0;
        constexpr int kSumAlpha = 1;
        constexpr int kCount = 2;
        constexpr int kMeanCos = 3;
        constexpr int kInvNorm = 4;
        constexpr int kSlotCount = 6;
    } // namespace normal_loss_slots

    constexpr float kNormalLossMinAlpha = 1.0e-3f;
    // Priors encode "no prediction" (sky, matting holes) as ~(0.5,0.5,0.5),
    // which decodes to a near-zero vector; a real unit normal decodes to ~1.
    constexpr float kNormalLossMinPriorNorm = 0.5f;
    // Skip pixels whose blended normal nearly cancels (opposing primitives at
    // silhouettes): the direction is ambiguous there and 1/|n| in the cosine
    // gradient would amplify it into the dominant term of the whole image.
    constexpr float kNormalLossMinRenderNorm = 0.1f;
    constexpr float kNormalLossMinValidCount = 64.0f;
    constexpr float kNormalLossMinValidWeight = 16.0f;

    [[nodiscard]] size_t normal_loss_partial_count(size_t num_pixels);

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
        cudaStream_t stream = nullptr);

} // namespace lfs::training::kernels

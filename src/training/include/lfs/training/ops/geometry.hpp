/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include "lfs/training/ops/geometry_types.hpp"
#include "lfs/training/ops/types.hpp"

#include <array>
#include <vector>

namespace lfs::gpu_ops {
    struct DepthParams {
        float weight, gradient_weight, prior_quantization_step;
        const training::kernels::DepthAnchor* anchor;
    };
    struct AnchorParams {
        Intrinsics intrinsics;
        float near_plane;
        std::array<float, 3> aabb_lo, aabb_hi;
    };
    // Scratch retains the diagnostic float prefix followed by aligned double
    // statistics: depth 10+6B slots, normal/consistency 6+6B, B=min(ceil(HW/256),1024).
    struct GeometryLossOps {
        void (*depth)(In depth, In alpha, In target, In pixel_weight,
                      Out grad_depth, Out grad_alpha, Out loss, Out partials, const DepthParams&);
        void (*normal)(In normal, In alpha, In target, In pixel_weight,
                       Out grad_normal, Out loss, Out partials, float weight);
        void (*consistency)(In normal, In depth, In alpha, In pixel_weight,
                            Out grad_normal, Out grad_depth, Out grad_alpha,
                            Out loss, Out partials, Intrinsics, float weight);
        void (*prior_depth)(In prior_normal, In depth, In alpha, In pixel_weight,
                            Out grad_depth, Out grad_alpha, Out loss, Out partials,
                            Intrinsics, float weight);
        std::vector<AnchorSample> (*collect_anchor_samples)(In points, In view, In prior, const AnchorParams&);
    };
} // namespace lfs::gpu_ops

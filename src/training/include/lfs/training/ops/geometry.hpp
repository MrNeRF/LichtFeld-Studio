/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include "lfs/training/ops/geometry_types.hpp"
#include "lfs/training/ops/types.hpp"

#include <array>
#include <vector>

namespace lfs::gpu_ops {
    struct GeometryCamera {
        float fx = 0, fy = 0, cx = 0, cy = 0;
        core::Tensor rays;
        bool wrap_horizontal = false;
        GeometryCamera() = default;
        GeometryCamera(float fx, float fy, float cx, float cy, core::Tensor rays = {}, bool wrap = false)
            : fx(fx), fy(fy), cx(cx), cy(cy), rays(std::move(rays)), wrap_horizontal(wrap) {}
        GeometryCamera(Intrinsics k) : GeometryCamera(k.fx, k.fy, k.cx, k.cy) {}
    };
    struct DepthParams {
        float weight, gradient_weight, prior_quantization_step;
        const training::kernels::DepthAnchor* anchor;
        const training::kernels::DepthRayGrid* ray_grid = nullptr;
    };
    struct AnchorParams {
        Intrinsics intrinsics;
        float near_plane;
        std::array<float, 3> aabb_lo, aabb_hi;
        training::kernels::DepthCameraProjection projection;
    };
    // Scratch retains the diagnostic float prefix followed by aligned double
    // statistics: depth 10+6B slots, normal/consistency 6+6B, B=min(ceil(HW/256),1024).
    struct GeometryLossOps {
        void (*depth_impl)(In depth, In alpha, In target, In pixel_weight,
                           Out grad_depth, Out grad_alpha, Out loss, Out partials, const DepthParams&);
        void depth(In depth, In alpha, In target, In pixel_weight, Out grad_depth, Out grad_alpha,
                   Out loss, Out partials, const DepthParams& params) const {
            if (!params.ray_grid) {
                depth_impl(depth, alpha, target, pixel_weight, grad_depth, grad_alpha, loss, partials, params);
                return;
            }
            const auto& grid = *params.ray_grid;
            const auto height = static_cast<int>(depth.shape()[0]);
            const auto width = static_cast<int>(depth.shape()[1]);
            const auto x = (Tensor::arange(0.f, static_cast<float>(width), 1.f, depth.device()) +
                            (0.5f - grid.cx)) /
                           grid.fx;
            const auto y = (Tensor::arange(0.f, static_cast<float>(height), 1.f, depth.device()) +
                            (0.5f - grid.cy)) /
                           grid.fy;
            const auto length = (x.square().reshape({1, width}) + y.square().reshape({height, 1}) + 1.f).sqrt();
            depth_impl(depth * length, alpha, target, pixel_weight, grad_depth, grad_alpha, loss, partials, params);
            grad_depth.mul_(length);
        }

        void (*normal)(In normal, In alpha, In target, In pixel_weight,
                       Out grad_normal, Out loss, Out partials, float weight);
        void (*consistency)(In normal, In depth, In alpha, In pixel_weight,
                            Out grad_normal, Out grad_depth, Out grad_alpha,
                            Out loss, Out partials, GeometryCamera, float weight);
        void (*prior_depth)(In prior_normal, In depth, In alpha, In pixel_weight,
                            Out grad_depth, Out grad_alpha, Out loss, Out partials,
                            GeometryCamera, float weight);
        std::vector<AnchorSample> (*collect_anchor_samples)(In points, In view, In prior, const AnchorParams&);
    };
} // namespace lfs::gpu_ops

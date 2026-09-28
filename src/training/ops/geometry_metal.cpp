/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

// Metal GeometryLossOps; see ops/geometry_cuda.cpp for the reference behaviour.

#include "metal_families.hpp"
#include "metal_kernels.hpp"

#include "core/assert.hpp"

#include <algorithm>
#include <cstring>
#include <format>
#include <string_view>

namespace lfs::training {
    namespace {
        using namespace lfs::gpu_ops;
        namespace k = kernels;
        namespace mk = metal;

        constexpr uint32_t kThreads = k::kGeometryLossThreads;
        constexpr size_t kMaxAnchorSamples = 262144;

        struct GeomParams {
            uint64_t normal, depth, alpha, target, weight;
            uint64_t grad_normal, grad_depth, grad_alpha, loss, finals, blocks;
            int32_t width, height;
            uint32_t num_blocks, has_weight;
            float fx, fy, cx, cy;
            float loss_weight, lambda_grad, quantization_step, floor_override;
            int32_t anchor_model;
            float anchor_scale, anchor_shift, anchor_floor;
            float min_count, min_weight;
            uint32_t prior;
        };

        // The partials buffer: slot_count float finals, then the block statistics.
        GeomParams base_params(In alpha, In weight, Out loss, Out partials, const int width, const int height,
                               const size_t slot_count, const size_t required, const std::string_view name) {
            const size_t pixels = static_cast<size_t>(width) * static_cast<size_t>(height);
            LFS_ASSERT_MSG(pixels > 0 && pixels <= UINT32_MAX,
                           std::format("{} needs 1..2^32-1 pixels (width={}, height={})", name, width, height));
            LFS_ASSERT_MSG(alpha.numel() == pixels,
                           std::format("{} alpha must be [H,W] (alpha={}, H={}, W={})", name, alpha.shape().str(), height, width));
            LFS_ASSERT_MSG(!weight.is_valid() || weight.numel() == pixels,
                           std::format("{} pixel weight must be [H,W] (weight={}, H={}, W={})", name, weight.shape().str(),
                                       height, width));
            LFS_ASSERT_MSG(partials.is_valid() && partials.dtype() == core::DataType::Float32 && partials.numel() >= required,
                           std::format("{} partials need {} Float32 values (have={})", name, required, partials.numel()));
            const uint64_t finals = mk::address(partials);
            GeomParams params{};
            params.alpha = mk::address(alpha);
            params.weight = mk::address(weight);
            params.has_weight = weight.is_valid() ? 1u : 0u;
            params.loss = mk::address(loss);
            params.finals = finals;
            params.blocks = finals + slot_count * sizeof(float);
            params.width = width;
            params.height = height;
            params.num_blocks = static_cast<uint32_t>(k::geometry_loss_block_count(pixels));
            return params;
        }

        void set_intrinsics(GeomParams& params, const Intrinsics& intrinsics) {
            params.fx = intrinsics.fx;
            params.fy = intrinsics.fy;
            params.cx = intrinsics.cx;
            params.cy = intrinsics.cy;
        }

        void depth(In depth, In alpha, In target, In pixel_weight, Out grad_depth, Out grad_alpha, Out loss,
                   Out partials, const DepthParams& p) {
            const int width = static_cast<int>(depth.shape()[1]), height = static_cast<int>(depth.shape()[0]);
            const size_t pixels = static_cast<size_t>(width) * height;
            GeomParams params = base_params(alpha, pixel_weight, loss, partials, width, height,
                                            k::depth_loss_slots::kSlotCount, k::depth_loss_partial_count(pixels),
                                            "depth loss");
            const bool anchor = p.anchor != nullptr && p.anchor->valid;
            params.depth = mk::address(depth);
            params.target = mk::address(target);
            params.grad_depth = mk::address(grad_depth);
            params.grad_alpha = mk::address(grad_alpha);
            params.loss_weight = p.weight;
            params.lambda_grad = p.gradient_weight;
            params.quantization_step = std::max(p.prior_quantization_step, 0.0f);
            params.floor_override = anchor ? p.anchor->floor : 0.0f;
            params.anchor_model = anchor ? p.anchor->model : 0;
            params.anchor_scale = anchor ? p.anchor->scale : 0.0f;
            params.anchor_shift = anchor ? p.anchor->shift : 0.0f;
            params.anchor_floor = anchor ? p.anchor->floor : 0.0f;
            // Block statistics over num_blocks groups, then a one-group finalize.
            const auto run = [&](const std::string_view function, const uint32_t groups) {
                mk::launch(function, params, {&depth, &alpha, &target, &pixel_weight, &grad_depth, &grad_alpha, &loss, &partials},
                           groups, kThreads);
            };
            run("geom_depth_primary", params.num_blocks);
            run("geom_depth_finalize_primary", 1);
            run("geom_depth_inverse", params.num_blocks);
            run("geom_depth_finalize_alignment", 1);
            run("geom_depth_grad", params.num_blocks);
            run("geom_depth_finalize_loss", 1);
        }

        void normal(In normal, In alpha, In target, In pixel_weight, Out grad_normal, Out loss, Out partials,
                    float weight) {
            const int width = static_cast<int>(normal.shape()[2]), height = static_cast<int>(normal.shape()[1]);
            const size_t pixels = static_cast<size_t>(width) * height;
            GeomParams params = base_params(alpha, pixel_weight, loss, partials, width, height,
                                            k::normal_loss_slots::kSlotCount, k::normal_loss_partial_count(pixels),
                                            "normal loss");
            params.normal = mk::address(normal);
            params.target = mk::address(target);
            params.grad_normal = mk::address(grad_normal);
            params.loss_weight = weight;
            params.min_count = k::kNormalLossMinValidCount;
            params.min_weight = k::kNormalLossMinValidWeight;
            const auto run = [&](const std::string_view function, const uint32_t groups) {
                mk::launch(function, params, {&normal, &alpha, &target, &pixel_weight, &grad_normal, &loss, &partials}, groups,
                           kThreads);
            };
            run("geom_normal_stats", params.num_blocks);
            run("geom_normal_finalize_stats", 1);
            run("geom_normal_grad", params.num_blocks);
            run("geom_normal_finalize_loss", 1);
        }

        // Consistency (prior = false) compares the depth normal with the rendered
        // normal and also writes grad_normal; prior depth compares it with the
        // normal prior. Both add into the caller's gradients.
        void depth_normal(In normal, In depth, In alpha, In pixel_weight, Out grad_normal, Out grad_depth,
                          Out grad_alpha, Out loss, Out partials, const Intrinsics& intrinsics, const float weight,
                          const bool prior) {
            const int width = static_cast<int>(depth.shape()[1]), height = static_cast<int>(depth.shape()[0]);
            const size_t pixels = static_cast<size_t>(width) * height;
            GeomParams params = base_params(alpha, pixel_weight, loss, partials, width, height,
                                            k::normal_consistency_slots::kSlotCount,
                                            k::normal_consistency_partial_count(pixels),
                                            prior ? "normal prior depth loss" : "normal consistency loss");
            params.normal = mk::address(normal);
            params.depth = mk::address(depth);
            params.grad_normal = mk::address(grad_normal);
            params.grad_depth = mk::address(grad_depth);
            params.grad_alpha = mk::address(grad_alpha);
            params.loss_weight = weight;
            params.min_count = k::kNormalConsistencyMinValidCount;
            params.min_weight = k::kNormalConsistencyMinValidWeight;
            params.prior = prior ? 1u : 0u;
            set_intrinsics(params, intrinsics);
            const auto run = [&](const std::string_view function, const uint32_t groups) {
                mk::launch(function, params,
                           {&normal, &depth, &alpha, &pixel_weight, &grad_normal, &grad_depth, &grad_alpha, &loss, &partials},
                           groups, kThreads);
            };
            run("geom_depth_normal_stats", params.num_blocks);
            run("geom_normal_finalize_stats", 1);
            run("geom_depth_normal_grad", params.num_blocks);
            run("geom_normal_finalize_loss", 1);
        }

        void consistency(In normal, In depth, In alpha, In pixel_weight, Out grad_normal, Out grad_depth,
                         Out grad_alpha, Out loss, Out partials, Intrinsics intrinsics, float weight) {
            depth_normal(normal, depth, alpha, pixel_weight, grad_normal, grad_depth, grad_alpha, loss, partials,
                         intrinsics, weight, false);
        }

        void prior_depth(In prior_normal, In depth, In alpha, In pixel_weight, Out grad_depth, Out grad_alpha,
                         Out loss, Out partials, Intrinsics intrinsics, float weight) {
            Tensor no_normal_gradient;
            depth_normal(prior_normal, depth, alpha, pixel_weight, no_normal_gradient, grad_depth, grad_alpha, loss,
                         partials, intrinsics, weight, true);
        }

        std::vector<AnchorSample> collect_anchor_samples(In points, In view, In prior, const AnchorParams& p) {
            const size_t num_points = points.shape()[0];
            if (num_points == 0)
                return {};
            LFS_ASSERT_MSG(num_points <= UINT32_MAX / 3,
                           std::format("anchor sampling supports at most 2^32/3 points (points={})", num_points));
            const size_t stride = std::max<size_t>(1, num_points / kMaxAnchorSamples);
            const size_t samples = (num_points + stride - 1) / stride;
            Tensor pairs = Tensor::empty({kMaxAnchorSamples, 2}, core::Device::GPU);
            Tensor count = Tensor::zeros({1}, core::Device::GPU, core::DataType::Int32);
            struct {
                uint64_t points, w2c, prior, pairs, count;
                mk::Float4 aabb_lo, aabb_hi;
                uint32_t samples, stride;
                int32_t width, height, capacity;
                float fx, fy, cx, cy, near_plane;
            } const params{mk::address(points),
                           mk::address(view),
                           mk::address(prior),
                           mk::address(pairs),
                           mk::address(count),
                           {p.aabb_lo[0], p.aabb_lo[1], p.aabb_lo[2], 0.f},
                           {p.aabb_hi[0], p.aabb_hi[1], p.aabb_hi[2], 0.f},
                           static_cast<uint32_t>(samples),
                           static_cast<uint32_t>(stride),
                           static_cast<int32_t>(prior.shape()[1]),
                           static_cast<int32_t>(prior.shape()[0]),
                           static_cast<int32_t>(kMaxAnchorSamples),
                           p.intrinsics.fx,
                           p.intrinsics.fy,
                           p.intrinsics.cx,
                           p.intrinsics.cy,
                           p.near_plane};
            mk::launch_items("geom_anchor_collect", params, {&points, &view, &prior, &pairs, &count}, samples);
            // Startup-only readback, as the CUDA path synchronizes here too.
            const int found = std::min(count.item<int>(), static_cast<int>(kMaxAnchorSamples));
            if (found < k::kMinAnchorSamples)
                return {};
            const Tensor host = pairs.slice(0, 0, static_cast<size_t>(found)).cpu().contiguous();
            std::vector<AnchorSample> result(static_cast<size_t>(found));
            std::memcpy(result.data(), host.data_ptr(), result.size() * sizeof(AnchorSample));
            return result;
        }
    } // namespace

    const lfs::gpu_ops::GeometryLossOps& metal_geometry_ops() {
        static const lfs::gpu_ops::GeometryLossOps ops{depth, normal, consistency, prior_depth, collect_anchor_samples};
        return ops;
    }
} // namespace lfs::training

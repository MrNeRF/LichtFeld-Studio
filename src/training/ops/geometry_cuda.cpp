/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "lfs/training/ops/geometry_cuda.hpp"

#include "../kernels/depth_loss.hpp"
#include "../kernels/normal_consistency_loss.hpp"
#include "../kernels/normal_loss.hpp"
#include "core/tensor_cuda_interop.hpp"

namespace lfs::training {
    namespace {
        using namespace lfs::gpu_ops;
        static_assert(sizeof(AnchorSample) == sizeof(float2));
        static_assert(alignof(AnchorSample) == alignof(float2));
        const float* weight_ptr(In weight) {
            return weight.is_valid() ? weight.ptr<float>() : nullptr;
        }
        void depth(In depth, In alpha, In target, In pixel_weight,
                   Out grad_depth, Out grad_alpha, Out loss, Out partials, const DepthParams& p) {
            kernels::launch_depth_loss(depth.ptr<float>(), alpha.ptr<float>(), target.ptr<float>(),
                                       grad_depth.ptr<float>(), grad_alpha.ptr<float>(), loss.ptr<float>(), partials.ptr<float>(),
                                       depth.shape()[1], depth.shape()[0], p.weight, p.gradient_weight, p.prior_quantization_step,
                                       p.anchor, core::getCurrentCUDAStream(), weight_ptr(pixel_weight));
        }
        void normal(In normal, In alpha, In target, In pixel_weight,
                    Out grad_normal, Out loss, Out partials, float weight) {
            kernels::launch_normal_loss(normal.ptr<float>(), alpha.ptr<float>(), target.ptr<float>(),
                                        grad_normal.ptr<float>(), loss.ptr<float>(), partials.ptr<float>(),
                                        normal.shape()[2], normal.shape()[1], weight, core::getCurrentCUDAStream(), weight_ptr(pixel_weight));
        }
        void consistency(In normal, In depth, In alpha, In pixel_weight,
                         Out grad_normal, Out grad_depth, Out grad_alpha, Out loss, Out partials,
                         Intrinsics k, float weight) {
            kernels::launch_normal_consistency_loss(normal.ptr<float>(), depth.ptr<float>(), alpha.ptr<float>(),
                                                    grad_normal.ptr<float>(), grad_depth.ptr<float>(), grad_alpha.ptr<float>(), loss.ptr<float>(), partials.ptr<float>(),
                                                    depth.shape()[1], depth.shape()[0], k.fx, k.fy, k.cx, k.cy, weight,
                                                    core::getCurrentCUDAStream(), weight_ptr(pixel_weight));
        }
        void prior_depth(In normal, In depth, In alpha, In pixel_weight,
                         Out grad_depth, Out grad_alpha, Out loss, Out partials, Intrinsics k, float weight) {
            kernels::launch_normal_prior_depth_loss(normal.ptr<float>(), depth.ptr<float>(), alpha.ptr<float>(),
                                                    grad_depth.ptr<float>(), grad_alpha.ptr<float>(), loss.ptr<float>(), partials.ptr<float>(),
                                                    depth.shape()[1], depth.shape()[0], k.fx, k.fy, k.cx, k.cy, weight,
                                                    core::getCurrentCUDAStream(), weight_ptr(pixel_weight));
        }
        std::vector<AnchorSample> collect_anchor_samples(In points, In view, In prior, const AnchorParams& p) {
            return kernels::collect_depth_anchor_samples(points.ptr<float>(), points.shape()[0], view.ptr<float>(),
                                                         p.intrinsics.fx, p.intrinsics.fy, p.intrinsics.cx, p.intrinsics.cy, prior.ptr<float>(),
                                                         prior.shape()[1], prior.shape()[0], p.near_plane, p.aabb_lo.data(), p.aabb_hi.data(), core::getCurrentCUDAStream());
        }
        const GeometryLossOps kCudaGeometryOps{depth, normal, consistency, prior_depth, collect_anchor_samples};
    } // namespace
    const gpu_ops::GeometryLossOps& cuda_geometry_ops() { return kCudaGeometryOps; }
} // namespace lfs::training

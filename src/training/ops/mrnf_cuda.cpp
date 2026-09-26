/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "lfs/training/ops/mrnf_cuda.hpp"

#include "core/tensor_cuda_interop.hpp"
#include "kernels/mrnf_kernels.hpp"
#include "lfs/training/refine_scratch.hpp"

#include <array>

namespace lfs::training {
    namespace {

        using lfs::gpu_ops::Bounds;
        using lfs::gpu_ops::DecayParams;
        using lfs::gpu_ops::GumbelParams;
        using lfs::gpu_ops::MrnfNoiseParams;
        using lfs::gpu_ops::ProjectParams;
        using lfs::gpu_ops::ScalarValidity;
        using lfs::gpu_ops::Tensor;

        template <typename T>
        [[nodiscard]] const T* optional_ptr(const Tensor& tensor, size_t& count) {
            if (!tensor.is_valid()) {
                count = 0;
                return nullptr;
            }
            count = tensor.numel();
            return tensor.ptr<T>();
        }

        void noise(
            Tensor& means, const Tensor& raw_opacity, const Tensor& visibility, const Tensor& frozen,
            const MrnfNoiseParams& params) {
            size_t frozen_n = 0;
            const bool* frozen_ptr = optional_ptr<bool>(frozen, frozen_n);
            mrnf_strategy::launch_mrnf_noise_injection(
                means.ptr<float>(),
                raw_opacity.ptr<float>(),
                visibility.ptr<float>(),
                frozen_ptr,
                frozen_n,
                params.lr_mean,
                params.noise_weight,
                params.median_scale,
                means.shape()[0],
                params.seed,
                lfs::core::getCurrentCUDAStream());
        }

        void decay(
            Tensor& raw_opacity, Tensor& log_scales, const Tensor& frozen, const Tensor& far_mask,
            const DecayParams& params) {
            size_t frozen_n = 0;
            size_t far_n = 0;
            const bool* frozen_ptr = optional_ptr<bool>(frozen, frozen_n);
            const bool* far_ptr = optional_ptr<bool>(far_mask, far_n);
            mrnf_strategy::launch_mrnf_decay(
                raw_opacity.ptr<float>(),
                log_scales.ptr<float>(),
                frozen_ptr,
                frozen_n,
                far_ptr,
                far_n,
                params.opacity_decay,
                params.scale_decay,
                params.far_decay_scale,
                params.train_t,
                log_scales.shape()[0],
                lfs::core::getCurrentCUDAStream());
        }

        Bounds percentile_bounds(const Tensor& means, const float percentile) {
            mrnf_strategy::MRNFBounds raw{};
            mrnf_strategy::launch_percentile_bounds(
                means.ptr<float>(),
                means.shape()[0],
                percentile,
                &raw,
                lfs::core::getCurrentCUDAStream());
            Bounds bounds{};
            for (int axis = 0; axis < 3; ++axis) {
                bounds.center[axis] = raw.center[axis];
                bounds.extent[axis] = raw.extent[axis];
            }
            bounds.median_size = raw.median_size;
            bounds.max_extent = raw.max_extent;
            return bounds;
        }

        ScalarValidity median_extent(const Tensor& raw_scales) {
            ScalarValidity result;
            mrnf_strategy::launch_median_geomean_extent(
                raw_scales.ptr<float>(),
                raw_scales.shape()[0],
                &result.value,
                &result.valid,
                lfs::core::getCurrentCUDAStream());
            return result;
        }

        void gumbel(
            GumbelTopKScratch* scratch, const Tensor& weights, Tensor& indices,
            const GumbelParams& params) {
            mrnf_strategy::launch_gumbel_topk(
                weights.ptr<float>(),
                weights.numel(),
                indices.numel(),
                params.seed,
                indices.ptr<int64_t>(),
                lfs::core::getCurrentCUDAStream(),
                params.compact_sparse,
                scratch,
                params.known_nnz);
        }

        void fold(
            Tensor& visibility, Tensor& weight_max, Tensor& densification, Tensor& ratio_max,
            const float ratio_power) {
            mrnf_strategy::launch_fold_densification_and_zero(
                visibility.ptr<float>(),
                weight_max.ptr<float>(),
                densification.ptr<float>(),
                visibility.numel(),
                lfs::core::getCurrentCUDAStream(),
                2,
                ratio_max.is_valid() ? ratio_max.ptr<float>() : nullptr,
                ratio_power);
        }

        void fold_error(Tensor& weight_max, Tensor& densification) {
            mrnf_strategy::launch_fold_densification_error_and_zero(
                weight_max.ptr<float>(),
                densification.ptr<float>(),
                weight_max.numel(),
                lfs::core::getCurrentCUDAStream());
        }

        void project_centers(
            const Tensor& means, const Tensor& view, Tensor& means2d, Tensor& radii,
            const ProjectParams& params) {
            mrnf_strategy::launch_project_visible_centers(
                means.ptr<float>(),
                view.ptr<float>(),
                params.intrinsics.fx,
                params.intrinsics.fy,
                params.intrinsics.cx,
                params.intrinsics.cy,
                params.image.w,
                params.image.h,
                params.near_plane,
                means2d.ptr<float>(),
                radii.ptr<float>(),
                means.shape()[0],
                lfs::core::getCurrentCUDAStream());
        }

        void gather_center_error(
            const Tensor& means2d, const Tensor& radii, const Tensor& error, Tensor& scores) {
            mrnf_strategy::launch_gather_center_error(
                means2d.ptr<float>(),
                radii.ptr<float>(),
                error.ptr<float>(),
                static_cast<int>(error.shape()[1]),
                static_cast<int>(error.shape()[0]),
                scores.ptr<float>(),
                means2d.shape()[0],
                lfs::core::getCurrentCUDAStream());
        }

        void far_mask(
            const Tensor& means, Tensor& mask, const std::array<float, 3> center, const float radius) {
            mrnf_strategy::launch_far_field_mask(
                means.ptr<float>(),
                center[0],
                center[1],
                center[2],
                radius,
                mask.ptr<bool>(),
                means.shape()[0],
                lfs::core::getCurrentCUDAStream());
        }

        void mean_abs_error(const Tensor& predicted, const Tensor& target, Tensor& error) {
            mrnf_strategy::launch_mean_abs_error_hw(
                predicted.ptr<float>(),
                target.ptr<float>(),
                static_cast<int>(predicted.shape()[0]),
                static_cast<int>(predicted.shape()[1]),
                static_cast<int>(predicted.shape()[2]),
                error.ptr<float>(),
                lfs::core::getCurrentCUDAStream());
        }

        void seed_weights(const Tensor& error, const Tensor& alpha, Tensor& weights) {
            mrnf_strategy::launch_seed_weights_from_error_alpha(
                error.ptr<float>(),
                alpha.ptr<float>(),
                weights.ptr<float>(),
                weights.numel(),
                lfs::core::getCurrentCUDAStream());
        }

        void gather_seeds(
            const Tensor& indices, const Tensor& target, const Tensor& alpha, const Tensor& depth,
            Tensor& rgb, Tensor& sampled_alpha, Tensor& sampled_depth) {
            mrnf_strategy::launch_gather_seed_payloads(
                indices.ptr<int64_t>(),
                indices.numel(),
                alpha.numel(),
                target.ptr<float>(),
                static_cast<int>(target.shape()[0]),
                alpha.ptr<float>(),
                depth.is_valid() ? depth.ptr<float>() : nullptr,
                rgb.ptr<float>(),
                sampled_alpha.ptr<float>(),
                sampled_depth.ptr<float>(),
                lfs::core::getCurrentCUDAStream());
        }

        float sorted_median(const Tensor& values) {
            return mrnf_strategy::launch_sorted_median(values.ptr<float>(), values.numel(),
                                                       lfs::core::getCurrentCUDAStream());
        }

        void starvation_weights(Tensor& weights, const Tensor& visibility, const float median) {
            mrnf_strategy::launch_apply_explore_starvation_weights(
                weights.ptr<float>(),
                visibility.ptr<float>(),
                weights.numel(),
                median,
                lfs::core::getCurrentCUDAStream());
        }

        const lfs::gpu_ops::MrnfOps kCudaMrnfOps{
            .noise = noise,
            .decay = decay,
            .percentile_bounds = percentile_bounds,
            .median_extent = median_extent,
            .gumbel = gumbel,
            .fold = fold,
            .fold_error = fold_error,
            .project_centers = project_centers,
            .gather_center_error = gather_center_error,
            .far_mask = far_mask,
            .mean_abs_error = mean_abs_error,
            .seed_weights = seed_weights,
            .gather_seeds = gather_seeds,
            .sorted_median = sorted_median,
            .starvation_weights = starvation_weights,
        };

    } // namespace

    const lfs::gpu_ops::MrnfOps& cuda_mrnf_ops() {
        return kCudaMrnfOps;
    }

} // namespace lfs::training

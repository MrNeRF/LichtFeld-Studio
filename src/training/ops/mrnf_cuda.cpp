/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "lfs/training/ops/mrnf_cuda.hpp"

#include "core/tensor/backend/cuda/kernels/tensor_ops.hpp"
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
            Tensor& raw_opacity, Tensor& log_scales, const Tensor& frozen,
            const DecayParams& params) {
            size_t frozen_n = 0;
            const bool* frozen_ptr = optional_ptr<bool>(frozen, frozen_n);
            mrnf_strategy::launch_mrnf_decay(
                raw_opacity.ptr<float>(),
                log_scales.ptr<float>(),
                frozen_ptr,
                frozen_n,
                params.opacity_decay,
                params.scale_decay,
                params.train_t,
                log_scales.shape()[0],
                lfs::core::getCurrentCUDAStream(),
                params.rendered_count.is_valid() ? params.rendered_count.ptr<float>() : nullptr,
                params.reach.is_valid() ? params.reach.ptr<float>() : nullptr);
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

        void fold_error(Tensor& weight_max, Tensor& densification) {
            mrnf_strategy::launch_fold_densification_error_and_zero(
                weight_max.ptr<float>(),
                densification.ptr<float>(),
                weight_max.numel(),
                lfs::core::getCurrentCUDAStream());
        }

        size_t compact_bool_indices(const Tensor& mask, Tensor& indices, size_t count) {
            return core::tensor_ops::launch_nonzero_bool(mask.ptr<unsigned char>(), indices.ptr<int64_t>(),
                                                         mask.numel(), count, mask.stream());
        }
        void prune_bounds(const Tensor& means, const Tensor& scale_max, Tensor& mask,
                          std::array<float, 3> center, float maximum, float log_maximum) {
            const auto origin = Tensor::from_vector(std::vector<float>{center[0], center[1], center[2]}, {1, 3}, means.device());
            const auto distance = (means - origin).abs().max(1);
            mask.copy_from(mask.logical_or((scale_max > log_maximum).logical_or(means.isfinite().all(1).logical_and(distance > maximum))));
        }
        void replace_parent_weights(const Tensor& opacity, const Tensor& visibility, const Tensor& active,
                                    const Tensor& trainable, const Tensor& edge, Tensor& weights) {
            mrnf_strategy::launch_replace_parent_weights(opacity.ptr<float>(), visibility.ptr<float>(),
                                                         active.is_valid() ? active.ptr<bool>() : nullptr, trainable.is_valid() ? trainable.ptr<bool>() : nullptr,
                                                         edge.is_valid() ? edge.ptr<float>() : nullptr, weights.ptr<float>(), opacity.numel());
        }
        float kth_largest(const Tensor& values, const Tensor& include, const size_t k) {
            const auto mask = include.is_valid() && include.dtype() != core::DataType::Bool
                                  ? include.to(core::DataType::Bool)
                                  : include;
            return mrnf_strategy::launch_kth_largest(values.ptr<float>(), mask.is_valid() ? mask.ptr<bool>() : nullptr,
                                                     values.numel(), k, lfs::core::getCurrentCUDAStream());
        }
        void rank_exchange_rows(const Tensor& primary, const Tensor& secondary, const Tensor& eligible, Tensor& rows) {
            mrnf_strategy::launch_rank_exchange_rows(
                primary.ptr<float>(), secondary.is_valid() ? secondary.ptr<float>() : nullptr, eligible.ptr<bool>(),
                primary.numel(), rows.numel(), rows.ptr<int64_t>(), lfs::core::getCurrentCUDAStream());
        }
        using gpu_ops::In;
        using gpu_ops::Out;
        void fold_view(Out error, Out info, Out views) {
            mrnf_strategy::launch_fold_densification_error_view_sum_and_zero(error.ptr<float>(), info.ptr<float>(), error.numel(), views.is_valid() ? views.ptr<float>() : nullptr, core::getCurrentCUDAStream());
        }
        void mean_view(Out error, In info) {
            mrnf_strategy::launch_refine_error_view_mean(error.ptr<float>(), info.ptr<float>(), error.numel(), core::getCurrentCUDAStream());
        }
        void fold_need(Out error, float scale, float maximum, float power) {
            mrnf_strategy::launch_fold_need(error.ptr<float>(), scale, maximum, power, error.numel(), core::getCurrentCUDAStream());
        }
        void blend_peak(Out step, Out peak, Out renders, float decay) {
            mrnf_strategy::launch_blend_peak_update(step.ptr<float>(), peak.ptr<float>(), renders.ptr<float>(), decay, peak.numel(), core::getCurrentCUDAStream());
        }
        void covered_factor(In peak, In renders, In opacity, In scales, float grace, float keep, float floor, Out factor) {
            mrnf_strategy::launch_mrnf_covered_factor(peak.ptr<float>(), renders.ptr<float>(), opacity.ptr<float>(), scales.ptr<float>(), grace, keep, floor, peak.numel(), factor.ptr<float>(), core::getCurrentCUDAStream());
        }
        void retire_key(In peak, In opacity, float dead_peak, Out key) {
            mrnf_strategy::launch_mrnf_retire_key(peak.ptr<float>(), opacity.ptr<float>(), dead_peak, peak.numel(), key.ptr<float>(), core::getCurrentCUDAStream());
        }
        void lowest_rows(In score, In eligible, Out rows) {
            mrnf_strategy::launch_lowest_eligible_rows(score.ptr<float>(), eligible.ptr<bool>(), score.numel(), rows.numel(), rows.ptr<int64_t>(), core::getCurrentCUDAStream());
        }
        const gpu_ops::MrnfGrowthOps growth_ops{
            .fold_view = fold_view,
            .mean_view = mean_view,
            .fold_need = fold_need,
            .blend_peak = blend_peak,
            .covered_factor = covered_factor,
            .retire_key = retire_key,
            .lowest_rows = lowest_rows,
        };
        const lfs::gpu_ops::MrnfOps kCudaMrnfOps{
            .noise = noise,
            .decay = decay,
            .percentile_bounds = percentile_bounds,
            .gumbel = gumbel,
            .fold_error = fold_error,
            .compact_bool_indices = compact_bool_indices,
            .prune_bounds = prune_bounds,
            .replace_parent_weights = replace_parent_weights,
            .kth_largest = kth_largest,
            .rank_exchange_rows = rank_exchange_rows,
            .growth = &growth_ops,
        };

    } // namespace

    const lfs::gpu_ops::MrnfOps& cuda_mrnf_ops() {
        return kCudaMrnfOps;
    }

} // namespace lfs::training

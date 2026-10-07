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
                params.rendered_count.is_valid() ? params.rendered_count.ptr<float>() : nullptr);
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
        const lfs::gpu_ops::MrnfOps kCudaMrnfOps{
            .noise = noise,
            .decay = decay,
            .percentile_bounds = percentile_bounds,
            .gumbel = gumbel,
            .fold_error = fold_error,
            .compact_bool_indices = compact_bool_indices,
            .prune_bounds = prune_bounds,
            .replace_parent_weights = replace_parent_weights,
        };

    } // namespace

    const lfs::gpu_ops::MrnfOps& cuda_mrnf_ops() {
        return kCudaMrnfOps;
    }

} // namespace lfs::training

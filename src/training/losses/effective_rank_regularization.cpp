/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */
#include "effective_rank_regularization.hpp"
#include "core/tensor_backend.hpp"
#include "lfs/training/ops/registry.hpp"
#include <cmath>
#include <format>
namespace lfs::training::losses {
    lfs::Result<core::Tensor> EffectiveRankRegularization::forward(
        const core::Tensor& scales, core::Tensor& grad, const Params& params,
        const core::Tensor& active, const core::Tensor& frozen, size_t active_count) {
        try {
            LFS_ASSERT_MSG(scales.is_valid() && scales.ndim() == 2 && scales.size(1) == 3 &&
                               scales.device() == core::Device::GPU && scales.dtype() == core::DataType::Float32 && scales.is_contiguous(),
                           std::format("Effective-rank scales must be contiguous GPU float32 [N,3] (shape={}, dtype={}, device={})",
                                       scales.shape().str(), static_cast<int>(scales.dtype()), static_cast<int>(scales.device())));
            LFS_ASSERT_MSG(grad.is_valid() && grad.shape() == scales.shape() && grad.device() == scales.device() &&
                               grad.dtype() == scales.dtype() && grad.is_contiguous(),
                           std::format("Effective-rank gradient must match scales (gradient={}, scales={})", grad.shape().str(), scales.shape().str()));
            LFS_ASSERT_MSG(std::isfinite(params.erank_weight) && params.erank_weight >= 0 &&
                               std::isfinite(params.thin_scale_weight) && params.thin_scale_weight >= 0 &&
                               std::isfinite(params.epsilon) && params.epsilon > 0,
                           std::format("Effective-rank weights must be finite/nonnegative and epsilon positive (rank={}, thin={}, epsilon={})",
                                       params.erank_weight, params.thin_scale_weight, params.epsilon));
            const size_t n = scales.size(0);
            for (const auto* mask : {&active, &frozen}) {
                LFS_ASSERT_MSG(!mask->is_valid() || (mask->device() == core::Device::GPU && mask->dtype() == core::DataType::Bool &&
                                                     mask->ndim() == 1 && mask->numel() == n && mask->is_contiguous()),
                               std::format("Effective-rank mask must be GPU bool [N] (shape={}, N={})", mask->shape().str(), n));
            }
            if (!active.is_valid())
                active_count = n;
            LFS_ASSERT_MSG(active_count <= n && (!n || active_count > 0),
                           std::format("Effective-rank active count must be in [1,N] (active={}, N={})", active_count, n));
            const auto backend = core::default_gpu_backend();
            const auto* ops = training_ops(backend).popspa;
            LFS_ASSERT_MSG(ops != nullptr, unavailable_training_family(backend, Family::Popspa).value_or("POPSpa training ops are unavailable"));
            auto loss = core::Tensor::zeros({1}, core::Device::GPU);
            if (n > 0 && active_count > 0)
                ops->effective_rank(scales, grad, active, frozen, loss, active_count, params.erank_weight,
                                    params.thin_scale_weight, params.epsilon);
            return loss;
        } catch (const std::exception& e) {
            return lfs::make_error({.code = lfs::ErrorCode::InvalidArgument, .domain = lfs::ErrorDomain::Training, .user_message = "POPSpa shape regularization failed.", .detail = e.what(), .detection = LFS_SOURCE_SITE_CURRENT()});
        }
    }
} // namespace lfs::training::losses

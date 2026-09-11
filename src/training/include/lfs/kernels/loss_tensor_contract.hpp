/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include "core/assert.hpp"
#include "core/cuda_safe_format.hpp"
#include "core/tensor.hpp"

#include <climits>
#include <cmath>
#include <cstddef>
#include <string>
#include <string_view>
#include <utility>

namespace lfs::training::kernels {

    struct PreparedLossImages {
        lfs::core::Tensor prediction;
        lfs::core::Tensor target;
    };

    namespace loss_contract_detail {

        [[nodiscard]] std::string format_loss_weight(float weight);

        [[nodiscard]] inline lfs::core::Tensor prepare_image(
            const lfs::core::Tensor& input,
            const bool target,
            const std::string_view name) {
            LFS_ASSERT_MSG(input.is_valid(), lfs::core::detail::format_cuda_safe("{} must be a valid tensor", name));
            LFS_ASSERT_MSG(input.device() == lfs::core::Device::CUDA,
                           lfs::core::detail::format_cuda_safe("{} must be a CUDA tensor", name));
            LFS_ASSERT_MSG(input.ndim() == 3 || input.ndim() == 4,
                           lfs::core::detail::format_cuda_safe("{} must have shape [C,H,W] or [N,C,H,W] (shape={})",
                                                               name, input.shape().str()));
            const bool valid_dtype = target
                                         ? input.dtype() == lfs::core::DataType::Float32 ||
                                               input.dtype() == lfs::core::DataType::UInt8
                                         : input.dtype() == lfs::core::DataType::Float32;
            LFS_ASSERT_MSG(valid_dtype,
                           lfs::core::detail::format_cuda_safe("{} has an unsupported dtype ({})", name,
                                                               static_cast<int>(input.dtype())));

            size_t indexed_elements = 1;
            for (size_t dim = 0; dim < input.shape().rank(); ++dim) {
                const size_t extent = input.shape()[dim];
                LFS_ASSERT_MSG(extent > 0,
                               lfs::core::detail::format_cuda_safe("{} dimensions must be positive (shape={})",
                                                                   name, input.shape().str()));
                LFS_ASSERT_MSG(extent <= static_cast<size_t>(INT_MAX) &&
                                   indexed_elements <= static_cast<size_t>(INT_MAX) / extent,
                               lfs::core::detail::format_cuda_safe("{} exceeds the signed kernel-index budget (shape={})",
                                                                   name, input.shape().str()));
                indexed_elements *= extent;
            }

            auto prepared = input.contiguous();
            if (prepared.ndim() == 3)
                prepared = prepared.unsqueeze(0);
            return prepared;
        }

    } // namespace loss_contract_detail

    [[nodiscard]] inline PreparedLossImages prepare_loss_images(
        const lfs::core::Tensor& prediction,
        const lfs::core::Tensor& target) {
        auto prepared_prediction = loss_contract_detail::prepare_image(prediction, false, "Loss prediction");
        auto prepared_target = loss_contract_detail::prepare_image(target, true, "Loss target");
        LFS_ASSERT_MSG(prepared_prediction.shape() == prepared_target.shape(),
                       lfs::core::detail::format_cuda_safe("Loss prediction and target shapes must match (prediction={}, target={})",
                                                           prepared_prediction.shape().str(), prepared_target.shape().str()));
        return {
            .prediction = std::move(prepared_prediction),
            .target = std::move(prepared_target),
        };
    }

    [[nodiscard]] inline lfs::core::Tensor prepare_loss_prediction(
        const lfs::core::Tensor& prediction,
        const std::string_view name) {
        return loss_contract_detail::prepare_image(prediction, false, name);
    }

    [[nodiscard]] inline lfs::core::Tensor prepare_loss_target(
        const lfs::core::Tensor& target,
        const std::string_view name) {
        return loss_contract_detail::prepare_image(target, true, name);
    }

    [[nodiscard]] inline lfs::core::Tensor prepare_loss_mask(
        const lfs::core::Tensor& input,
        const lfs::core::Tensor& prepared_image) {
        LFS_ASSERT_MSG(input.is_valid(), "Loss mask must be a valid tensor");
        LFS_ASSERT_MSG(input.device() == lfs::core::Device::CUDA, "Loss mask must be a CUDA tensor");
        LFS_ASSERT_MSG(input.dtype() == lfs::core::DataType::Float32 ||
                           input.dtype() == lfs::core::DataType::UInt8 ||
                           input.dtype() == lfs::core::DataType::Bool,
                       lfs::core::detail::format_cuda_safe("Loss mask has an unsupported dtype ({})",
                                                           static_cast<int>(input.dtype())));
        LFS_ASSERT_MSG(input.ndim() == 2 ||
                           (input.ndim() == 3 && input.shape()[0] == 1),
                       lfs::core::detail::format_cuda_safe("Loss mask must have shape [H,W] or [1,H,W] (shape={})",
                                                           input.shape().str()));

        auto mask = input.contiguous();
        if (mask.ndim() == 3)
            mask = mask.squeeze(0);
        LFS_ASSERT_MSG(mask.shape()[0] == prepared_image.shape()[2] &&
                           mask.shape()[1] == prepared_image.shape()[3],
                       lfs::core::detail::format_cuda_safe("Loss mask dimensions must match the image (mask={}, image={})",
                                                           mask.shape().str(), prepared_image.shape().str()));
        return mask;
    }

    inline void validate_loss_context_images(
        const lfs::core::Tensor& prediction,
        const lfs::core::Tensor& target,
        const int height,
        const int width) {
        LFS_ASSERT_MSG(prediction.is_contiguous() && target.is_contiguous(),
                       "Loss backward context images must be contiguous");
        const auto prepared = prepare_loss_images(prediction, target);
        LFS_ASSERT_MSG(height > 0 && width > 0 &&
                           prepared.prediction.shape()[2] == static_cast<size_t>(height) &&
                           prepared.prediction.shape()[3] == static_cast<size_t>(width),
                       lfs::core::detail::format_cuda_safe("Loss backward dimensions must match the context image "
                                                           "(height={}, width={}, image={})",
                                                           height, width, prepared.prediction.shape().str()));
    }

    inline void validate_loss_weight(const float weight) {
        LFS_ASSERT_MSG(std::isfinite(weight) && weight >= 0.0f && weight <= 1.0f,
                       loss_contract_detail::format_loss_weight(weight));
    }

} // namespace lfs::training::kernels

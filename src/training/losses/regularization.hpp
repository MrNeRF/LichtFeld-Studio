/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include "core/tensor.hpp"
#include <expected>
#include <string>

namespace lfs::training::losses {

    /**
     * @brief L1 regularization on exp(scaling_raw) with fused CUDA kernel
     *
     * Forward:  scaling = exp(scaling_raw)
     * Loss:     L = weight * mean(scaling)
     * Gradient: ∂L/∂scaling_raw = (weight / N) * exp(scaling_raw)
     *
     * NOTE: This loss writes gradients directly to scaling_raw_grad in-place
     */
    struct ScaleRegularization {
        struct Params {
            float weight; ///< Regularization weight
        };

        /**
         * @brief Compute scale regularization loss and accumulate gradients
         * @param scaling_raw [N, 3] raw scaling parameters
         * @param scaling_raw_grad [N, 3] gradient tensor (will be accumulated to)
         * @param params Loss parameters
         * @return loss_tensor (GPU) or error - loss stays on GPU!
         * @note Accumulates gradients directly to scaling_raw_grad
         */
        static std::expected<lfs::core::Tensor, std::string> forward(
            const lfs::core::Tensor& scaling_raw,
            lfs::core::Tensor& scaling_raw_grad,
            const Params& params);

        static std::expected<lfs::core::Tensor, std::string> forward_loss_only(
            const lfs::core::Tensor& scaling_raw,
            const Params& params);
    };

    /**
     * @brief L1 regularization on sigmoid(opacity_raw) with fused CUDA kernel
     *
     * Forward:  opacity = sigmoid(opacity_raw)
     * Loss:     L = weight * mean(opacity)
     * Gradient: ∂L/∂opacity_raw = (weight / N) * sigmoid(x) * (1 - sigmoid(x))
     *
     * NOTE: This loss writes gradients directly to opacity_raw_grad in-place
     */
    struct OpacityRegularization {
        struct Params {
            float weight; ///< Regularization weight
        };

        /**
         * @brief Compute opacity regularization loss and accumulate gradients
         * @param opacity_raw [N, 1] raw opacity parameters
         * @param opacity_raw_grad [N, 1] gradient tensor (will be accumulated to)
         * @param params Loss parameters
         * @return loss_tensor (GPU) or error - loss stays on GPU!
         * @note Accumulates gradients directly to opacity_raw_grad
         */
        static std::expected<lfs::core::Tensor, std::string> forward(
            const lfs::core::Tensor& opacity_raw,
            lfs::core::Tensor& opacity_raw_grad,
            const Params& params);

        static std::expected<lfs::core::Tensor, std::string> forward_loss_only(
            const lfs::core::Tensor& opacity_raw,
            const Params& params);
    };

} // namespace lfs::training::losses

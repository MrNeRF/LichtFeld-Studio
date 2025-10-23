/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once
#include <torch/torch.h>

namespace gs {
    namespace regularization {

        /**
         * Compute L1 regularization on exp(scaling_raw) with chain rule gradients.
         *
         * Forward:  scaling = exp(scaling_raw)
         * Loss:     L = weight * mean(scaling)
         * Gradient: ∂L/∂scaling_raw = (weight / N) * exp(scaling_raw)
         *
         * This function:
         * 1. Computes loss = weight * mean(exp(scaling_raw))
         * 2. Accumulates gradients: scaling_raw.grad += (weight / N) * exp(scaling_raw)
         *
         * @param scaling_raw Raw scaling parameters [N, 3] (3 scales per Gaussian)
         * @param weight Regularization weight
         * @return Loss value (scalar)
         */
        float compute_exp_l1_regularization_with_grad_cuda(
            torch::Tensor& scaling_raw,
            float weight);

        /**
         * Compute L1 regularization on sigmoid(opacity_raw) with chain rule gradients.
         *
         * Forward:  opacity = sigmoid(opacity_raw)
         * Loss:     L = weight * mean(opacity)
         * Gradient: ∂L/∂opacity_raw = (weight / N) * sigmoid(x) * (1 - sigmoid(x))
         *
         * This function:
         * 1. Computes loss = weight * mean(sigmoid(opacity_raw))
         * 2. Accumulates gradients: opacity_raw.grad += (weight / N) * σ(x) * (1 - σ(x))
         *
         * @param opacity_raw Raw opacity parameters [N, 1]
         * @param weight Regularization weight
         * @return Loss value (scalar)
         */
        float compute_sigmoid_l1_regularization_with_grad_cuda(
            torch::Tensor& opacity_raw,
            float weight);

    } // namespace regularization
} // namespace gs

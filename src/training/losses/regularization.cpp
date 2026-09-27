/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "regularization.hpp"
#include "lfs/training/ops/registry.hpp"
#include <format>

namespace lfs::training::losses {

    std::expected<lfs::core::Tensor, std::string> ScaleRegularization::forward(
        const lfs::core::Tensor& scaling_raw,
        lfs::core::Tensor& scaling_raw_grad,
        const Params& params) {
        try {
            if (params.weight <= 0.0f) {
                return lfs::core::Tensor::zeros({1}, lfs::core::Device::GPU);
            }

            // Validate inputs
            if (scaling_raw.device() != lfs::core::Device::GPU) {
                return std::unexpected("scaling_raw must be on CUDA device");
            }
            if (scaling_raw_grad.device() != lfs::core::Device::GPU) {
                return std::unexpected("scaling_raw_grad must be on CUDA device");
            }
            if (scaling_raw.shape() != scaling_raw_grad.shape()) {
                return std::unexpected("scaling_raw and scaling_raw_grad must have same shape");
            }

            size_t n = scaling_raw.numel();
            if (n == 0) {
                return lfs::core::Tensor::zeros({1}, lfs::core::Device::GPU);
            }

            // Allocate temporary buffers
            size_t num_blocks = std::min((n + 255) / 256, size_t(1024));
            auto temp_buffer = lfs::core::Tensor::empty({num_blocks}, lfs::core::Device::GPU);
            auto loss_tensor = lfs::core::Tensor::empty({1}, lfs::core::Device::GPU);

            // Launch LibTorch-free fused kernel with warp reductions
            training_ops(core::default_gpu_backend()).extra_loss->regularize(scaling_raw, scaling_raw_grad, loss_tensor, temp_buffer, lfs::gpu_ops::Regularizer::Scale, params.weight);

            // NO .item<float>() - keep on GPU!
            return loss_tensor;

        } catch (const std::exception& e) {
            return std::unexpected(std::format("Error in ScaleRegularization::forward: {}", e.what()));
        }
    }

    std::expected<lfs::core::Tensor, std::string> ScaleRegularization::forward_loss_only(
        const lfs::core::Tensor& scaling_raw,
        const Params& params) {
        try {
            if (params.weight <= 0.0f) {
                return lfs::core::Tensor::zeros({1}, lfs::core::Device::GPU);
            }
            if (scaling_raw.device() != lfs::core::Device::GPU) {
                return std::unexpected("scaling_raw must be on CUDA device");
            }

            size_t n = scaling_raw.numel();
            if (n == 0) {
                return lfs::core::Tensor::zeros({1}, lfs::core::Device::GPU);
            }

            size_t num_blocks = std::min((n + 255) / 256, size_t(1024));
            auto temp_buffer = lfs::core::Tensor::empty({num_blocks}, lfs::core::Device::GPU);
            auto loss_tensor = lfs::core::Tensor::empty({1}, lfs::core::Device::GPU);

            lfs::core::Tensor no_gradient;
            training_ops(core::default_gpu_backend()).extra_loss->regularize(scaling_raw, no_gradient, loss_tensor, temp_buffer, lfs::gpu_ops::Regularizer::Scale, params.weight);

            return loss_tensor;
        } catch (const std::exception& e) {
            return std::unexpected(std::format("Error in ScaleRegularization::forward_loss_only: {}", e.what()));
        }
    }

    std::expected<lfs::core::Tensor, std::string> OpacityRegularization::forward(
        const lfs::core::Tensor& opacity_raw,
        lfs::core::Tensor& opacity_raw_grad,
        const Params& params) {
        try {
            if (params.weight <= 0.0f) {
                return lfs::core::Tensor::zeros({1}, lfs::core::Device::GPU);
            }

            // Validate inputs
            if (opacity_raw.device() != lfs::core::Device::GPU) {
                return std::unexpected("opacity_raw must be on CUDA device");
            }
            if (opacity_raw_grad.device() != lfs::core::Device::GPU) {
                return std::unexpected("opacity_raw_grad must be on CUDA device");
            }
            if (opacity_raw.shape() != opacity_raw_grad.shape()) {
                return std::unexpected("opacity_raw and opacity_raw_grad must have same shape");
            }

            size_t n = opacity_raw.numel();
            if (n == 0) {
                return lfs::core::Tensor::zeros({1}, lfs::core::Device::GPU);
            }

            // Allocate temporary buffers
            size_t num_blocks = std::min((n + 255) / 256, size_t(1024));
            auto temp_buffer = lfs::core::Tensor::empty({num_blocks}, lfs::core::Device::GPU);
            auto loss_tensor = lfs::core::Tensor::empty({1}, lfs::core::Device::GPU);

            // Launch LibTorch-free fused kernel with warp reductions
            training_ops(core::default_gpu_backend()).extra_loss->regularize(opacity_raw, opacity_raw_grad, loss_tensor, temp_buffer, lfs::gpu_ops::Regularizer::Opacity, params.weight);

            // NO .item<float>() - keep on GPU!
            return loss_tensor;

        } catch (const std::exception& e) {
            return std::unexpected(std::format("Error in OpacityRegularization::forward: {}", e.what()));
        }
    }

    std::expected<lfs::core::Tensor, std::string> OpacityRegularization::forward_loss_only(
        const lfs::core::Tensor& opacity_raw,
        const Params& params) {
        try {
            if (params.weight <= 0.0f) {
                return lfs::core::Tensor::zeros({1}, lfs::core::Device::GPU);
            }
            if (opacity_raw.device() != lfs::core::Device::GPU) {
                return std::unexpected("opacity_raw must be on CUDA device");
            }

            size_t n = opacity_raw.numel();
            if (n == 0) {
                return lfs::core::Tensor::zeros({1}, lfs::core::Device::GPU);
            }

            size_t num_blocks = std::min((n + 255) / 256, size_t(1024));
            auto temp_buffer = lfs::core::Tensor::empty({num_blocks}, lfs::core::Device::GPU);
            auto loss_tensor = lfs::core::Tensor::empty({1}, lfs::core::Device::GPU);

            lfs::core::Tensor no_gradient;
            training_ops(core::default_gpu_backend()).extra_loss->regularize(opacity_raw, no_gradient, loss_tensor, temp_buffer, lfs::gpu_ops::Regularizer::Opacity, params.weight);

            return loss_tensor;
        } catch (const std::exception& e) {
            return std::unexpected(std::format("Error in OpacityRegularization::forward_loss_only: {}", e.what()));
        }
    }

} // namespace lfs::training::losses

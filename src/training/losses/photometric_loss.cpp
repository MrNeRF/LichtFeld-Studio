/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "photometric_loss.hpp"
#include "lfs/kernels/l1_loss.cuh"
#include "lfs/kernels/ssim.cuh"
#include <format>

namespace lfs::training::losses {

    void PhotometricLoss::ensure_buffers(const std::vector<size_t>& shape, size_t num_blocks) {
        // Only reallocate if shape or num_blocks changed
        if (allocated_shape_ != shape || allocated_num_blocks_ != num_blocks) {
            lfs::core::TensorShape tshape(shape);
            grad_buffer_ = lfs::core::Tensor::empty(tshape, lfs::core::Device::CUDA);
            loss_scalar_ = lfs::core::Tensor::zeros({1}, lfs::core::Device::CUDA);

            if (num_blocks > 0) {
                l1_reduction_buffer_ = lfs::core::Tensor::empty({num_blocks}, lfs::core::Device::CUDA);
            }

            allocated_shape_ = shape;
            allocated_num_blocks_ = num_blocks;
        }
    }

    std::expected<std::pair<lfs::core::Tensor, PhotometricLoss::Context>, std::string>
    PhotometricLoss::forward(
        const lfs::core::Tensor& rendered,
        const lfs::core::Tensor& gt_image,
        const Params& params) {
        try {
            // Ensure 4D shape [N, C, H, W] by adding batch dimension if needed
            auto rendered_4d = rendered.ndim() == 3 ? rendered.unsqueeze(0) : rendered;
            auto gt_4d = gt_image.ndim() == 3 ? gt_image.unsqueeze(0) : gt_image;

            // Validate shapes
            if (rendered_4d.shape() != gt_4d.shape()) {
                return std::unexpected("Shape mismatch: rendered and gt_image must have same shape");
            }

            lfs::core::Tensor grad_combined;
            lfs::core::Tensor loss_tensor_gpu;

            // Optimize: only compute what's needed based on lambda_dssim
            if (params.lambda_dssim == 0.0f) {
                // Pure L1 loss
                size_t N = rendered_4d.numel();
                size_t num_blocks = std::min((N + 255) / 256, size_t(1024));

                // Ensure buffers are sized correctly
                ensure_buffers(rendered_4d.shape().dims(), num_blocks);

                lfs::training::kernels::launch_fused_l1_loss(
                    rendered_4d.ptr<float>(),
                    gt_4d.ptr<float>(),
                    grad_buffer_.ptr<float>(),
                    loss_scalar_.ptr<float>(),
                    l1_reduction_buffer_.ptr<float>(),
                    N,
                    nullptr);

                grad_combined = grad_buffer_;
                loss_tensor_gpu = loss_scalar_;

            } else if (params.lambda_dssim == 1.0f) {
                // Pure SSIM loss
                auto [ssim_value_tensor, ssim_ctx] = lfs::training::kernels::ssim_forward(
                    rendered_4d, gt_4d, ssim_workspace_, /*apply_valid_padding=*/true);

                // loss = 1 - ssim
                loss_tensor_gpu = lfs::core::Tensor::full({1}, 1.0f, lfs::core::Device::CUDA) - ssim_value_tensor;

                // Backward: d(loss)/d(ssim) = -1 (since loss = 1 - ssim)
                grad_combined = lfs::training::kernels::ssim_backward(ssim_ctx, ssim_workspace_, -1.0f);

            } else {
                // Combined L1+SSIM loss (fused kernel)
                auto [loss_tensor, fused_ctx] = lfs::training::kernels::fused_l1_ssim_forward(
                    rendered_4d, gt_4d, params.lambda_dssim, fused_workspace_, /*apply_valid_padding=*/true);

                grad_combined = lfs::training::kernels::fused_l1_ssim_backward(fused_ctx, fused_workspace_);
                loss_tensor_gpu = loss_tensor;
            }

            // Remove batch dimension if input was 3D
            if (rendered.ndim() == 3) {
                grad_combined = grad_combined.squeeze(0);
            }

            Context ctx{
                .loss_tensor = loss_tensor_gpu,
                .grad_image = grad_combined};
            return std::make_pair(loss_tensor_gpu, ctx);
        } catch (const std::exception& e) {
            return std::unexpected(std::format("Error computing photometric loss with gradient: {}", e.what()));
        }
    }

} // namespace lfs::training::losses

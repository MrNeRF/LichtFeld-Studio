/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include "core/tensor.hpp"
#include "lfs/kernels/ssim.cuh"
#include <expected>
#include <string>

namespace lfs::training::losses {

    /**
     * @brief Photometric loss combining L1 and SSIM with manual gradient computation
     *
     * Loss = (1 - lambda_dssim) * L1 + lambda_dssim * (1 - SSIM)
     *
     * This is a libtorch-free implementation that wraps the existing CUDA SSIM kernels.
     * Pre-allocates SSIM workspace to avoid per-frame allocation overhead.
     */
    struct PhotometricLoss {
        struct Params {
            float lambda_dssim; ///< Weight for D-SSIM term (0.0 = pure L1, 1.0 = pure SSIM)
        };

        struct Context {
            lfs::core::Tensor loss_tensor; ///< [1] scalar loss on GPU (avoid sync!)
            lfs::core::Tensor grad_image;  ///< [H, W, C] gradient w.r.t. rendered image
        };

        /**
         * @brief Compute photometric loss and gradient
         * @param rendered [H, W, C] rendered image
         * @param gt_image [H, W, C] ground truth image
         * @param params Loss parameters
         * @return (loss_tensor, context) or error - loss stays on GPU!
         */
        std::expected<std::pair<lfs::core::Tensor, Context>, std::string> forward(
            const lfs::core::Tensor& rendered,
            const lfs::core::Tensor& gt_image,
            const Params& params);

        const kernels::FusedL1SSIMWorkspace& fused_workspace() const { return fused_workspace_; }
        const kernels::SSIMWorkspace& ssim_workspace() const { return ssim_workspace_; }

    private:
        // Pre-allocated SSIM workspace
        lfs::training::kernels::SSIMWorkspace ssim_workspace_;

        // Pre-allocated fused L1+SSIM workspace
        lfs::training::kernels::FusedL1SSIMWorkspace fused_workspace_;

        // Pre-allocated buffers for loss computation (eliminates ~35GB allocation churn)
        lfs::core::Tensor grad_buffer_;         // Reusable gradient buffer [N, C, H, W]
        lfs::core::Tensor loss_scalar_;         // Reusable scalar loss [1]
        lfs::core::Tensor l1_reduction_buffer_; // Reusable L1 reduction buffer [num_blocks]
        std::vector<size_t> allocated_shape_;   // Track allocated shape
        size_t allocated_num_blocks_ = 0;       // Track allocated reduction buffer size

        // Ensure buffers are sized correctly (only reallocates if shape/size changed)
        void ensure_buffers(const std::vector<size_t>& shape, size_t num_blocks);
    };

} // namespace lfs::training::losses

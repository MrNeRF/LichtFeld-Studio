/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once
#include "core/tensor.hpp"
#include <tuple>
#include <vector>

namespace lfs::training::kernels {

    inline constexpr float SSIM_EPSILON = 1e-8f;

    // Pre-allocated workspace for SSIM computation
    struct SSIMWorkspace {
        // Forward pass buffers
        lfs::core::Tensor ssim_map;      // [N, C, H, W]
        lfs::core::Tensor dm_dmu1;       // [N, C, H, W]
        lfs::core::Tensor dm_dsigma1_sq; // [N, C, H, W]
        lfs::core::Tensor dm_dsigma12;   // [N, C, H, W]

        // Backward pass buffers
        lfs::core::Tensor dL_dmap;  // [N, C, H, W]
        lfs::core::Tensor dL_dimg1; // [N, C, H, W]

        // Cropped buffer for efficient mean computation (avoids .contiguous() allocation)
        lfs::core::Tensor ssim_map_cropped; // [N, C, H-10, W-10] contiguous buffer

        // Track allocated size
        std::vector<size_t> allocated_shape;

        // Resize workspace if needed (only reallocates if shape changed)
        void ensure_size(const std::vector<size_t>& shape) {
            if (allocated_shape != shape) {
                lfs::core::TensorShape tshape(shape);
                ssim_map = lfs::core::Tensor::empty(tshape, lfs::core::Device::CUDA);
                dm_dmu1 = lfs::core::Tensor::empty(tshape, lfs::core::Device::CUDA);
                dm_dsigma1_sq = lfs::core::Tensor::empty(tshape, lfs::core::Device::CUDA);
                dm_dsigma12 = lfs::core::Tensor::empty(tshape, lfs::core::Device::CUDA);
                dL_dmap = lfs::core::Tensor::empty(tshape, lfs::core::Device::CUDA);
                dL_dimg1 = lfs::core::Tensor::empty(tshape, lfs::core::Device::CUDA);

                // Allocate cropped buffer (10 pixels smaller in H and W for valid padding)
                if (shape.size() == 4 && shape[2] > 10 && shape[3] > 10) {
                    std::vector<size_t> cropped_shape = {shape[0], shape[1], shape[2] - 10, shape[3] - 10};
                    ssim_map_cropped = lfs::core::Tensor::empty(lfs::core::TensorShape(cropped_shape), lfs::core::Device::CUDA);
                }

                allocated_shape = shape;
            }
        }
    };

    // Context for manual SSIM forward/backward (like RasterizeContext)
    struct SSIMContext {
        lfs::core::Tensor img1;
        lfs::core::Tensor img2;
        lfs::core::Tensor dm_dmu1;
        lfs::core::Tensor dm_dsigma1_sq;
        lfs::core::Tensor dm_dsigma12;
        int original_h;
        int original_w;
        bool apply_valid_padding;
    };

    std::pair<lfs::core::Tensor, SSIMContext> ssim_forward(
        const lfs::core::Tensor& img1,
        const lfs::core::Tensor& img2,
        bool apply_valid_padding = true);

    // Version with pre-allocated workspace
    std::pair<lfs::core::Tensor, SSIMContext> ssim_forward(
        const lfs::core::Tensor& img1,
        const lfs::core::Tensor& img2,
        SSIMWorkspace& workspace,
        bool apply_valid_padding = true);

    // Per-pixel SSIM map result for masked loss computation
    struct SSIMMapResult {
        lfs::core::Tensor ssim_map;   // [N, C, H, W]
        lfs::core::Tensor ssim_value; // Mean SSIM scalar
        SSIMContext ctx;
    };

    // Returns per-pixel SSIM map (same padding when apply_valid_padding=false)
    SSIMMapResult ssim_forward_map(
        const lfs::core::Tensor& img1,
        const lfs::core::Tensor& img2,
        bool apply_valid_padding = false);

    // Manual SSIM backward (no autograd) - computes gradient w.r.t. img1
    lfs::core::Tensor ssim_backward(
        const SSIMContext& ctx,
        float grad_loss); // Gradient of loss w.r.t. SSIM value (scalar)

    // Optimized version with pre-allocated workspace
    lfs::core::Tensor ssim_backward(
        const SSIMContext& ctx,
        SSIMWorkspace& workspace,
        float grad_loss);

    // Per-pixel gradient version for masked SSIM (d(loss)/d(ssim_map) per pixel)
    lfs::core::Tensor ssim_backward_with_grad_map(
        const SSIMContext& ctx,
        const lfs::core::Tensor& dL_dmap); // [N, C, H, W] per-pixel gradient

    // ============================================================================
    // Fused L1+SSIM Loss
    // ============================================================================

    // Workspace for fused L1+SSIM (extends SSIMWorkspace)
    struct FusedL1SSIMWorkspace {
        // Forward pass buffers
        lfs::core::Tensor loss_map;      // [N, C, H, W] per-pixel combined loss
        lfs::core::Tensor dm_dmu1;       // [N, C, H, W] SSIM partial derivative
        lfs::core::Tensor dm_dsigma1_sq; // [N, C, H, W] SSIM partial derivative
        lfs::core::Tensor dm_dsigma12;   // [N, C, H, W] SSIM partial derivative

        // Backward pass buffer
        lfs::core::Tensor grad_img; // [N, C, H, W] combined gradient

        // Cropped buffer for mean computation
        lfs::core::Tensor loss_map_cropped; // [N, C, H-10, W-10]

        // Track allocated size
        std::vector<size_t> allocated_shape;

        void ensure_size(const std::vector<size_t>& shape) {
            if (allocated_shape != shape) {
                lfs::core::TensorShape tshape(shape);
                loss_map = lfs::core::Tensor::empty(tshape, lfs::core::Device::CUDA);
                dm_dmu1 = lfs::core::Tensor::empty(tshape, lfs::core::Device::CUDA);
                dm_dsigma1_sq = lfs::core::Tensor::empty(tshape, lfs::core::Device::CUDA);
                dm_dsigma12 = lfs::core::Tensor::empty(tshape, lfs::core::Device::CUDA);
                grad_img = lfs::core::Tensor::empty(tshape, lfs::core::Device::CUDA);

                if (shape.size() == 4 && shape[2] > 10 && shape[3] > 10) {
                    std::vector<size_t> cropped = {shape[0], shape[1], shape[2] - 10, shape[3] - 10};
                    loss_map_cropped = lfs::core::Tensor::empty(lfs::core::TensorShape(cropped), lfs::core::Device::CUDA);
                }
                allocated_shape = shape;
            }
        }
    };

    // Context for fused L1+SSIM backward pass
    struct FusedL1SSIMContext {
        lfs::core::Tensor img1;
        lfs::core::Tensor img2;
        lfs::core::Tensor dm_dmu1;
        lfs::core::Tensor dm_dsigma1_sq;
        lfs::core::Tensor dm_dsigma12;
        float ssim_weight;
        int H, W;
        bool apply_valid_padding;
    };

    // Fused L1+SSIM forward: loss = (1-w)*L1 + w*(1-SSIM)
    std::pair<lfs::core::Tensor, FusedL1SSIMContext> fused_l1_ssim_forward(
        const lfs::core::Tensor& img1,
        const lfs::core::Tensor& img2,
        float ssim_weight,
        FusedL1SSIMWorkspace& workspace,
        bool apply_valid_padding = true);

    // Fused L1+SSIM backward
    lfs::core::Tensor fused_l1_ssim_backward(
        const FusedL1SSIMContext& ctx,
        FusedL1SSIMWorkspace& workspace);

    // ============================================================================
    // Fused Masked L1+SSIM Loss (for segmentation/ignore mask modes)
    // ============================================================================

    struct MaskedFusedL1SSIMWorkspace {
        lfs::core::Tensor loss_map;      // [N, C, H, W]
        lfs::core::Tensor dm_dmu1;       // [N, C, H, W]
        lfs::core::Tensor dm_dsigma1_sq; // [N, C, H, W]
        lfs::core::Tensor dm_dsigma12;   // [N, C, H, W]
        lfs::core::Tensor grad_img;      // [N, C, H, W]
        lfs::core::Tensor masked_loss;   // [1] scalar
        lfs::core::Tensor mask_sum;      // [1] scalar

        std::vector<size_t> allocated_shape;

        void ensure_size(const std::vector<size_t>& shape) {
            if (allocated_shape != shape) {
                lfs::core::TensorShape tshape(shape);
                loss_map = lfs::core::Tensor::empty(tshape, lfs::core::Device::CUDA);
                dm_dmu1 = lfs::core::Tensor::empty(tshape, lfs::core::Device::CUDA);
                dm_dsigma1_sq = lfs::core::Tensor::empty(tshape, lfs::core::Device::CUDA);
                dm_dsigma12 = lfs::core::Tensor::empty(tshape, lfs::core::Device::CUDA);
                grad_img = lfs::core::Tensor::empty(tshape, lfs::core::Device::CUDA);
                masked_loss = lfs::core::Tensor::empty({1}, lfs::core::Device::CUDA);
                mask_sum = lfs::core::Tensor::empty({1}, lfs::core::Device::CUDA);
                allocated_shape = shape;
            }
        }
    };

    struct MaskedFusedL1SSIMContext {
        lfs::core::Tensor img1;
        lfs::core::Tensor img2;
        lfs::core::Tensor mask;
        lfs::core::Tensor dm_dmu1;
        lfs::core::Tensor dm_dsigma1_sq;
        lfs::core::Tensor dm_dsigma12;
        float ssim_weight;
        float mask_sum_value;
        int H, W;
    };

    // Fused masked L1+SSIM forward
    std::pair<lfs::core::Tensor, MaskedFusedL1SSIMContext> masked_fused_l1_ssim_forward(
        const lfs::core::Tensor& img1,
        const lfs::core::Tensor& img2,
        const lfs::core::Tensor& mask,
        float ssim_weight,
        MaskedFusedL1SSIMWorkspace& workspace);

    // Fused masked L1+SSIM backward
    lfs::core::Tensor masked_fused_l1_ssim_backward(
        const MaskedFusedL1SSIMContext& ctx,
        MaskedFusedL1SSIMWorkspace& workspace);

} // namespace lfs::training::kernels

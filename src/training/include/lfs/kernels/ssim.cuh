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

        // Tiny reduction buffers for valid-padding mean computation
        lfs::core::Tensor reduction_temp;   // [<=1024]
        lfs::core::Tensor reduction_result; // [1]

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
                reduction_temp = lfs::core::Tensor::empty({1024}, lfs::core::Device::CUDA);
                reduction_result = lfs::core::Tensor::empty({1}, lfs::core::Device::CUDA);

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

    // Lightweight workspace when only the per-pixel SSIM map is needed.
    struct SSIMMapWorkspace {
        lfs::core::Tensor ssim_map; // [N, C, H, W]
        std::vector<size_t> allocated_shape;

        void ensure_size(const std::vector<size_t>& shape) {
            if (allocated_shape != shape) {
                ssim_map = lfs::core::Tensor::empty(lfs::core::TensorShape(shape), lfs::core::Device::CUDA);
                allocated_shape = shape;
            }
        }
    };

    // Returns per-pixel SSIM map (same padding when apply_valid_padding=false)
    SSIMMapResult ssim_forward_map(
        const lfs::core::Tensor& img1,
        const lfs::core::Tensor& img2,
        bool apply_valid_padding = false);

    // Computes a full-resolution error map [H, W] from SSIM without allocating backward buffers.
    void ssim_error_map_forward(
        const lfs::core::Tensor& img1,
        const lfs::core::Tensor& img2,
        SSIMMapWorkspace& workspace,
        lfs::core::Tensor& error_map);

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
        lfs::core::Tensor ssim_map;      // [N, 1, H, W] per-pixel channel-mean SSIM values
        lfs::core::Tensor dm_dmu1;       // [N, C, H, W] SSIM partial derivative
        lfs::core::Tensor dm_dsigma1_sq; // [N, C, H, W] SSIM partial derivative
        lfs::core::Tensor dm_dsigma12;   // [N, C, H, W] SSIM partial derivative

        // Backward pass buffer
        lfs::core::Tensor grad_img; // [N, C, H, W] combined gradient

        // Tiny reduction buffers for valid-padding mean computation
        lfs::core::Tensor reduction_temp;   // [<=1024]
        lfs::core::Tensor reduction_result; // [1]

        // Track allocated size
        std::vector<size_t> allocated_shape;

        void ensure_size(const std::vector<size_t>& shape) {
            if (allocated_shape != shape) {
                lfs::core::TensorShape tshape(shape);
                std::vector<size_t> map_shape = shape;
                map_shape[1] = 1;
                ssim_map = lfs::core::Tensor::empty(lfs::core::TensorShape(map_shape), lfs::core::Device::CUDA);
                dm_dmu1 = lfs::core::Tensor::empty(tshape, lfs::core::Device::CUDA);
                dm_dsigma1_sq = lfs::core::Tensor::empty(tshape, lfs::core::Device::CUDA);
                dm_dsigma12 = lfs::core::Tensor::empty(tshape, lfs::core::Device::CUDA);
                grad_img = lfs::core::Tensor::empty(tshape, lfs::core::Device::CUDA);
                reduction_temp = lfs::core::Tensor::empty({1024}, lfs::core::Device::CUDA);
                reduction_result = lfs::core::Tensor::empty({1}, lfs::core::Device::CUDA);
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
    // Decoupled L1+SSIM Loss for appearance modeling
    // ============================================================================

    struct DecoupledGradients {
        lfs::core::Tensor grad_corrected; // Gradient through the appearance-corrected image
        lfs::core::Tensor grad_raw;       // Direct gradient to the raw render (contrast/structure only)
    };

    struct DecoupledFusedL1SSIMWorkspace {
        lfs::core::Tensor ssim_map;          // [N, 1, H, W] channel-mean decoupled SSIM map
        lfs::core::Tensor app_dm_dmu1;       // [N, C, H, W] d(ssim_map)/d mu(corrected)
        lfs::core::Tensor raw_dm_dmu1;       // [N, C, H, W] indirect mu(raw) contribution via sigma terms
        lfs::core::Tensor raw_dm_dsigma1_sq; // [N, C, H, W] lambda-scaled d(ssim_map)/d sigma^2(raw)
        lfs::core::Tensor raw_dm_dsigma12;   // [N, C, H, W] lambda-scaled d(ssim_map)/d sigma12(raw)
        lfs::core::Tensor zero_terms;        // [N, C, H, W] reusable zeros for unused backward terms
        lfs::core::Tensor grad_corrected;    // [N, C, H, W]
        lfs::core::Tensor grad_raw;          // [N, C, H, W]
        lfs::core::Tensor reduction_temp;    // [<=1024]
        lfs::core::Tensor reduction_result;  // [1]

        std::vector<size_t> allocated_shape;

        void ensure_size(const std::vector<size_t>& shape) {
            if (allocated_shape != shape) {
                lfs::core::TensorShape tshape(shape);
                std::vector<size_t> map_shape = shape;
                map_shape[1] = 1;
                ssim_map = lfs::core::Tensor::empty(lfs::core::TensorShape(map_shape), lfs::core::Device::CUDA);
                app_dm_dmu1 = lfs::core::Tensor::empty(tshape, lfs::core::Device::CUDA);
                raw_dm_dmu1 = lfs::core::Tensor::empty(tshape, lfs::core::Device::CUDA);
                raw_dm_dsigma1_sq = lfs::core::Tensor::empty(tshape, lfs::core::Device::CUDA);
                raw_dm_dsigma12 = lfs::core::Tensor::empty(tshape, lfs::core::Device::CUDA);
                zero_terms = lfs::core::Tensor::zeros(tshape, lfs::core::Device::CUDA);
                grad_corrected = lfs::core::Tensor::empty(tshape, lfs::core::Device::CUDA);
                grad_raw = lfs::core::Tensor::empty(tshape, lfs::core::Device::CUDA);
                reduction_temp = lfs::core::Tensor::empty({1024}, lfs::core::Device::CUDA);
                reduction_result = lfs::core::Tensor::empty({1}, lfs::core::Device::CUDA);
                allocated_shape = shape;
            }
        }
    };

    struct DecoupledFusedL1SSIMContext {
        lfs::core::Tensor corrected_img;
        lfs::core::Tensor raw_img;
        lfs::core::Tensor gt_img;
        lfs::core::Tensor app_dm_dmu1;
        lfs::core::Tensor raw_dm_dmu1;
        lfs::core::Tensor raw_dm_dsigma1_sq;
        lfs::core::Tensor raw_dm_dsigma12;
        float ssim_weight;
        int H, W;
        bool apply_valid_padding;
    };

    std::pair<lfs::core::Tensor, DecoupledFusedL1SSIMContext> decoupled_fused_l1_ssim_forward(
        const lfs::core::Tensor& corrected_img,
        const lfs::core::Tensor& raw_img,
        const lfs::core::Tensor& gt_img,
        float ssim_weight,
        DecoupledFusedL1SSIMWorkspace& workspace,
        bool apply_valid_padding = true);

    DecoupledGradients decoupled_fused_l1_ssim_backward(
        const DecoupledFusedL1SSIMContext& ctx,
        DecoupledFusedL1SSIMWorkspace& workspace);

    // ============================================================================
    // Fused Masked L1+SSIM Loss (for segmentation/ignore mask modes)
    // ============================================================================

    struct MaskedFusedL1SSIMWorkspace {
        lfs::core::Tensor ssim_map;       // [N, 1, H, W] per-pixel channel-mean SSIM values
        lfs::core::Tensor dm_dmu1;        // [N, C, H, W]
        lfs::core::Tensor dm_dsigma1_sq;  // [N, C, H, W]
        lfs::core::Tensor dm_dsigma12;    // [N, C, H, W]
        lfs::core::Tensor grad_img;       // [N, C, H, W]
        lfs::core::Tensor reduction_temp; // [<=2048], split into loss and mask partial sums
        lfs::core::Tensor masked_loss;    // [1] scalar
        lfs::core::Tensor mask_sum;       // [1] scalar

        std::vector<size_t> allocated_shape;

        void ensure_size(const std::vector<size_t>& shape) {
            if (allocated_shape != shape) {
                lfs::core::TensorShape tshape(shape);
                std::vector<size_t> map_shape = shape;
                map_shape[1] = 1;
                ssim_map = lfs::core::Tensor::empty(lfs::core::TensorShape(map_shape), lfs::core::Device::CUDA);
                dm_dmu1 = lfs::core::Tensor::empty(tshape, lfs::core::Device::CUDA);
                dm_dsigma1_sq = lfs::core::Tensor::empty(tshape, lfs::core::Device::CUDA);
                dm_dsigma12 = lfs::core::Tensor::empty(tshape, lfs::core::Device::CUDA);
                grad_img = lfs::core::Tensor::empty(tshape, lfs::core::Device::CUDA);
                reduction_temp = lfs::core::Tensor::empty({2048}, lfs::core::Device::CUDA);
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

    struct MaskedDecoupledFusedL1SSIMWorkspace {
        lfs::core::Tensor ssim_map;    // [N, 1, H, W]
        lfs::core::Tensor app_dm_dmu1; // [N, C, H, W]
        lfs::core::Tensor raw_dm_dmu1;
        lfs::core::Tensor raw_dm_dsigma1_sq;
        lfs::core::Tensor raw_dm_dsigma12;
        lfs::core::Tensor zero_terms;
        lfs::core::Tensor grad_corrected; // [N, C, H, W]
        lfs::core::Tensor grad_raw;       // [N, C, H, W]
        lfs::core::Tensor reduction_temp; // [<=2048]
        lfs::core::Tensor masked_loss;    // [1]
        lfs::core::Tensor mask_sum;       // [1]

        std::vector<size_t> allocated_shape;

        void ensure_size(const std::vector<size_t>& shape) {
            if (allocated_shape != shape) {
                lfs::core::TensorShape tshape(shape);
                std::vector<size_t> map_shape = shape;
                map_shape[1] = 1;
                ssim_map = lfs::core::Tensor::empty(lfs::core::TensorShape(map_shape), lfs::core::Device::CUDA);
                app_dm_dmu1 = lfs::core::Tensor::empty(tshape, lfs::core::Device::CUDA);
                raw_dm_dmu1 = lfs::core::Tensor::empty(tshape, lfs::core::Device::CUDA);
                raw_dm_dsigma1_sq = lfs::core::Tensor::empty(tshape, lfs::core::Device::CUDA);
                raw_dm_dsigma12 = lfs::core::Tensor::empty(tshape, lfs::core::Device::CUDA);
                zero_terms = lfs::core::Tensor::zeros(tshape, lfs::core::Device::CUDA);
                grad_corrected = lfs::core::Tensor::empty(tshape, lfs::core::Device::CUDA);
                grad_raw = lfs::core::Tensor::empty(tshape, lfs::core::Device::CUDA);
                reduction_temp = lfs::core::Tensor::empty({2048}, lfs::core::Device::CUDA);
                masked_loss = lfs::core::Tensor::empty({1}, lfs::core::Device::CUDA);
                mask_sum = lfs::core::Tensor::empty({1}, lfs::core::Device::CUDA);
                allocated_shape = shape;
            }
        }
    };

    struct MaskedDecoupledFusedL1SSIMContext {
        lfs::core::Tensor corrected_img;
        lfs::core::Tensor raw_img;
        lfs::core::Tensor gt_img;
        lfs::core::Tensor mask;
        lfs::core::Tensor app_dm_dmu1;
        lfs::core::Tensor raw_dm_dmu1;
        lfs::core::Tensor raw_dm_dsigma1_sq;
        lfs::core::Tensor raw_dm_dsigma12;
        float ssim_weight;
        float mask_sum_value;
        int H, W;
    };

    std::pair<lfs::core::Tensor, MaskedDecoupledFusedL1SSIMContext> masked_decoupled_fused_l1_ssim_forward(
        const lfs::core::Tensor& corrected_img,
        const lfs::core::Tensor& raw_img,
        const lfs::core::Tensor& gt_img,
        const lfs::core::Tensor& mask,
        float ssim_weight,
        MaskedDecoupledFusedL1SSIMWorkspace& workspace);

    DecoupledGradients masked_decoupled_fused_l1_ssim_backward(
        const MaskedDecoupledFusedL1SSIMContext& ctx,
        MaskedDecoupledFusedL1SSIMWorkspace& workspace);

    // Fused SSIM map → error map: error_map[i] = max(0, 1 - mean_c(ssim_map[c, i]))
    // Replaces .neg().add(1).mean({1}).squeeze(0).clamp_min(0).contiguous() chain
    void launch_ssim_to_error_map(
        const lfs::core::Tensor& ssim_map,
        lfs::core::Tensor& error_map);

} // namespace lfs::training::kernels

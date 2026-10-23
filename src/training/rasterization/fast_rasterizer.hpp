/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include "core/tensor.hpp"
#include "optimizer/adam_optimizer.hpp"

namespace lfs::training {

    struct FastGSFusedExtraGradients {
        float scale_reg_weight = 0.0f;
        bool scale_reg_log = false;
        float scale_reg_normalizer = 1.0f;
        float erank_reg_weight = 0.0f;
        float dc_reg_weight = 0.0f;
        float sh_rest_reg_weight = 0.0f;
        float flatten_reg_weight = 0.0f;
        float opacity_reg_weight = 0.0f;
        float* rendered_count = nullptr;
        // Optional persistent device scalars (caller zeros each step). Accumulated in
        // preprocess_backward so loss-only reg kernels can be skipped on the FastGS path.
        float* scale_reg_loss_out = nullptr;
        float* opacity_reg_loss_out = nullptr;
        float* erank_reg_loss_out = nullptr;
        float* sh_rest_reg_loss_out = nullptr;
        float* dc_reg_loss_out = nullptr;
        // Optional edge guidance folded into the main blend backward. The map
        // is a median-normalized row-major float32 [H,W], and the destination
        // is a zeroed float32 [N] per-view scratch vector.
        const float* edge_weight_map = nullptr;
        float* edge_score_out = nullptr;
        float* impact_sum_out = nullptr;
        float* impact_peak_out = nullptr;
    };

    struct FastGSColorVarianceLoss {
        lfs::core::Tensor target;
        lfs::core::Tensor loss; // Optional caller-zeroed CUDA float32 scalar.
        float weight = 0.0f;
        // Optional CUDA Float32 [H, W] pixel weights, such as the crop box ROI. The loss
        // then takes their weighted mean over the pixels, like the photometric loss.
        lfs::core::Tensor pixel_weight;
    };

} // namespace lfs::training

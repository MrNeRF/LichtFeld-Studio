/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include "optimizer/adam_optimizer.hpp"
#include <rasterization_api.h>

namespace lfs::training {

    struct FastGSFusedExtraGradients {
        float scale_reg_weight = 0.0f;
        float flatten_reg_weight = 0.0f;
        float opacity_reg_weight = 0.0f;
        // Optional persistent device scalars (caller zeros each step). Accumulated in
        // preprocess_backward so loss-only reg kernels can be skipped on the FastGS path.
        float* scale_reg_loss_out = nullptr;
        float* opacity_reg_loss_out = nullptr;
        const float* sparsity_opa_sigmoid = nullptr;
        const float* sparsity_z = nullptr;
        const float* sparsity_u = nullptr;
        int sparsity_n = 0;
        float sparsity_rho = 0.0f;
        float sparsity_grad_loss = 0.0f;
        // Optional edge guidance folded into the main blend backward. The map
        // is a median-normalized row-major float32 [H,W], and the destination
        // is a zeroed float32 [N] per-view scratch vector.
        const float* edge_weight_map = nullptr;
        float* edge_score_out = nullptr;
    };

    [[nodiscard]] fast_lfs::rasterization::FusedAdamSettings fast_adam_settings(
        const lfs::gpu_ops::BackwardAdam& adam);

} // namespace lfs::training

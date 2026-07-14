/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include <cstdint>

namespace fast_lfs::rasterization {

    // Optimizer moments are quantised: m (first) as signed int8 around zero-point 128,
    // v (second) as quantised sqrt(v); both with per-primitive fp32 scales. param stays fp32.
    struct FusedAdamParam {
        float* param = nullptr;
        std::uint8_t* exp_avg_q = nullptr;
        std::uint8_t* exp_avg_sq_q = nullptr;
        float* exp_avg_scale = nullptr;
        float* exp_avg_sq_scale = nullptr;
        const bool* frozen_mask = nullptr;
        int frozen_mask_size = 0;
        int n_elements = 0;
        int n_attributes = 0;
        float step_size = 0.0f;
        float bias_correction2_sqrt_rcp = 1.0f;
        bool enabled = false;
    };

    struct FusedAdamSettings {
        bool enabled = false;
        float beta1 = 0.9f;
        float beta2 = 0.999f;
        float eps = 1e-15f;
        float scale_reg_weight = 0.0f;
        float flatten_reg_weight = 0.0f;
        float opacity_reg_weight = 0.0f;
        const float* sparsity_opa_sigmoid = nullptr;
        const float* sparsity_z = nullptr;
        const float* sparsity_u = nullptr;
        int sparsity_n = 0;
        float sparsity_rho = 0.0f;
        float sparsity_grad_loss = 0.0f;

        FusedAdamParam means;
        FusedAdamParam scaling;
        FusedAdamParam rotation;
        FusedAdamParam opacity;
        FusedAdamParam sh0;
        FusedAdamParam shN;
    };

} // namespace fast_lfs::rasterization

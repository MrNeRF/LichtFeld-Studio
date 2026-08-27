/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include <cstdint>

namespace fast_lfs::rasterization {

    // Optimizer moments use joint (u, log_s) packing plus float4 bounds per
    // 256-splat block. joint_bits == 0 disables updates; 8 or 16 selects the codec.
    // SH value quant: when sh_value_bits==16, `param` is uint16 codes
    // (pad-dropped cell-linear swizzle) and sh_value_bounds is float2 per 256-prim block.
    struct FusedAdamParam {
        float* param = nullptr; // float params OR bitcast uint16* when sh_value_bits==16
        // Joint codec: packed (u,log_s) bytes + float4 bounds[n_bounds]
        std::uint8_t* joint_packed = nullptr;
        float* joint_bounds = nullptr; // float4 as 4 floats per bound
        int joint_bits = 0;            // 0=disabled, 8=SH, 16=non-SH
        // SH value quant — only meaningful on fused.shN
        float* sh_value_bounds = nullptr; // float2 as 2 floats per 256-prim block
        int sh_value_bits = 0;            // 0=fp32 param, 16=uint16 codes
        int sh_value_n_cells = 0;         // cells per prim (45 for SH3 pad-dropped)
        int n_primitives = 0;             // live splat count (bounds index = prim/256)
        const bool* frozen_mask = nullptr;
        int frozen_mask_size = 0;
        float frozen_lr_scale = 0.0f;
        const bool* crop_damping_mask = nullptr;
        int crop_damping_mask_size = 0;
        float cropbox_lr_scale = 1.0f;
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
        // Optional persistent device scalars (caller zeros). Filled by preprocess_backward
        // via block-reduce + atomicAdd so the trainer can skip loss-only reg kernels.
        float* scale_reg_loss_out = nullptr;
        float* opacity_reg_loss_out = nullptr;
        const float* sparsity_opa_sigmoid = nullptr;
        const float* sparsity_z = nullptr;
        const float* sparsity_u = nullptr;
        int sparsity_n = 0;
        float sparsity_rho = 0.0f;
        float sparsity_grad_loss = 0.0f;
        bool per_splat_mean_step = false;
        float mean_step_median_extent = 0.0f;
        float mean_step_r_min = 1.0f;
        float mean_step_r_max = 300.0f;
        const bool* mean_step_far_mask = nullptr;
        int mean_step_far_mask_n = 0;

        FusedAdamParam means;
        FusedAdamParam scaling;
        FusedAdamParam rotation;
        FusedAdamParam opacity;
        FusedAdamParam sh0;
        FusedAdamParam shN;
    };

} // namespace fast_lfs::rasterization

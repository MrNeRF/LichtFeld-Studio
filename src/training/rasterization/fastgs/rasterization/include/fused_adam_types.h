/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include <cstddef>
#include <cstdint>

namespace fast_lfs::rasterization {

    inline constexpr int kSparseReboundSlots = 7;

    // Arena-backed GPU scratch for expand-only block bounds in sparse backward preprocessing.
    // Slots: 0:means, 1:sh0, 2:scaling, 3:rotation, 4:opacity, 5:shN_joint, 6:shN_value.
    struct SparseReboundState {
        std::uint32_t* counts_pre = nullptr;
        std::uint32_t* counts_post = nullptr;
        std::uint32_t* dirty_bits = nullptr;
        std::uint32_t* rebound_list = nullptr;
        std::uint32_t* needed_u = nullptr; // [7 * n_blocks]: float-as-uint max |u| (slots 0..5) or |v| (slot 6)
        std::uint32_t* needed_s = nullptr; // [6 * n_blocks]: float-as-uint max log_s (slots 0..5)
        std::uint32_t n_blocks = 0u;
        std::uint32_t words_per_slot = 0u;
        std::size_t clear_bytes = 0u;

        static std::size_t align256(const std::size_t n) noexcept {
            return (n + 255u) & ~std::size_t{255};
        }

        static std::size_t required_bytes(const int n_primitives) noexcept {
            if (n_primitives <= 0)
                return 0;
            const std::size_t nb = (static_cast<std::size_t>(n_primitives) + 255u) / 256u;
            const std::size_t words = (nb + 31u) / 32u;
            const std::size_t counts_bytes = align256(2u * kSparseReboundSlots * sizeof(std::uint32_t));
            const std::size_t dirty_bytes = align256(kSparseReboundSlots * words * sizeof(std::uint32_t));
            const std::size_t needed_u_bytes = align256(kSparseReboundSlots * nb * sizeof(std::uint32_t));
            const std::size_t needed_s_bytes = align256((kSparseReboundSlots - 1) * nb * sizeof(std::uint32_t));
            const std::size_t list_bytes = align256(kSparseReboundSlots * nb * sizeof(std::uint32_t));
            return counts_bytes + dirty_bytes + needed_u_bytes + needed_s_bytes + list_bytes;
        }

        static SparseReboundState from_blob(char* blob, const int n_primitives) noexcept {
            SparseReboundState s{};
            if (!blob || n_primitives <= 0)
                return s;
            const std::size_t nb = (static_cast<std::size_t>(n_primitives) + 255u) / 256u;
            const std::size_t words = (nb + 31u) / 32u;
            const std::size_t counts_bytes = align256(2u * kSparseReboundSlots * sizeof(std::uint32_t));
            const std::size_t dirty_bytes = align256(kSparseReboundSlots * words * sizeof(std::uint32_t));
            const std::size_t needed_u_bytes = align256(kSparseReboundSlots * nb * sizeof(std::uint32_t));
            const std::size_t needed_s_bytes = align256((kSparseReboundSlots - 1) * nb * sizeof(std::uint32_t));
            s.n_blocks = static_cast<std::uint32_t>(nb);
            s.words_per_slot = static_cast<std::uint32_t>(words);
            s.counts_pre = reinterpret_cast<std::uint32_t*>(blob);
            s.counts_post = s.counts_pre + kSparseReboundSlots;
            s.dirty_bits = reinterpret_cast<std::uint32_t*>(blob + counts_bytes);
            s.needed_u = reinterpret_cast<std::uint32_t*>(blob + counts_bytes + dirty_bytes);
            s.needed_s = reinterpret_cast<std::uint32_t*>(blob + counts_bytes + dirty_bytes + needed_u_bytes);
            s.rebound_list = reinterpret_cast<std::uint32_t*>(blob + counts_bytes + dirty_bytes + needed_u_bytes + needed_s_bytes);
            s.clear_bytes = counts_bytes + dirty_bytes + needed_u_bytes + needed_s_bytes;
            return s;
        }
    };

    // Optimizer moments use joint (u, log_s) packing plus float4 bounds per
    // 256-splat block. joint_bits == 0 disables updates; 8 or 16 selects the codec.
    // SH value quant: when sh_value_bits==16, `param` is uint16 codes
    // (pad-dropped cell-linear swizzle) and sh_value_bounds is float2 per 256-prim block.
    struct FusedAdamParam {
        // Optional raw gradient destination. When present, capture instead of updating Adam.
        float* gradient_out = nullptr;
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
        const float* screen_share_max = nullptr;
        int screen_share_n = 0;
        float screen_share_limit = 0.0f;
        float screen_share_penalty = 0.0f;
    };

    // Optional per-step MRNF visibility/error/peak fold fused into sparse backward.
    struct MrnfFoldArgs {
        float* refine_weight = nullptr;
        float* densification_info = nullptr;
        float* coverage_views = nullptr;
        const bool* frozen_mask = nullptr;
        std::size_t frozen_n = 0;
        float* step_peak = nullptr;
        float* peak = nullptr;
        float* renders = nullptr;
        float inv_peak_scale = 1.0f;
        std::size_t n_primitives = 0;
    };

    // Optional post-Adam MRNF low-opacity mean noise injection fused across sparse
    // geometry (contributing rows) and invisible decay (non-contributing rows).
    struct MrnfNoiseArgs {
        const float* vis_count = nullptr;
        const bool* frozen_mask = nullptr;
        std::size_t frozen_mask_size = 0;
        float lr_mean_noise_weight = 0.0f;
        float median_scale = 0.0f;
        float max_logit = -1.0e30f;
        std::uint64_t seed = 0;
        bool enabled = false;
    };

    struct FusedAdamSettings {
        bool enabled = false;
        std::uint32_t step_seed = 0u;
        std::uint16_t* lazy_last_step = nullptr;
        int lazy_last_step_n = 0;
        std::uint16_t* geom_last_step = nullptr;
        int geom_last_step_n = 0;
        float beta1 = 0.9f;
        float beta2 = 0.999f;
        float eps = 1e-15f;
        float scale_reg_weight = 0.0f;
        bool scale_reg_log = false;
        float scale_reg_normalizer = 1.0f;
        float erank_reg_weight = 0.0f;
        float dc_reg_weight = 0.0f;
        float sh_rest_reg_weight = 0.0f;
        float flatten_reg_weight = 0.0f;
        float opacity_reg_weight = 0.0f;
        float* rendered_count = nullptr;
        // Optional [N] row raised to each primitive's largest blend weight alpha * T over the pixels of the view
        // (blend backward, atomic max).
        float* blend_peak = nullptr;
        // The scale, rank and flatten regularizers skip splats that did not contribute to the view.
        bool reg_rendered_only = false;
        // Optional persistent device scalars (caller zeros). Filled by preprocess_backward
        // via block-reduce + atomicAdd so the trainer can skip loss-only reg kernels.
        float* scale_reg_loss_out = nullptr;
        float* opacity_reg_loss_out = nullptr;
        float* erank_reg_loss_out = nullptr;
        float* sh_rest_reg_loss_out = nullptr;
        float* dc_reg_loss_out = nullptr;
        bool per_splat_mean_step = false;
        float mean_step_median_extent = 0.0f;
        const bool* mean_step_far_mask = nullptr;
        int mean_step_far_mask_n = 0;
        // Ray-depth rows (lfs/training/ray_depth_step.cuh) stepped after the geometry rows; null: off.
        float* ray_depth_state = nullptr;
        int ray_depth_n = 0;
        float ray_depth_lr = 0.0f;
        float ray_depth_mean_lr = 0.0f;   // position learning rate, world units
        float ray_depth_far_limit = 0.0f; // the rasterizer's far plane, world units

        FusedAdamParam means;
        FusedAdamParam scaling;
        FusedAdamParam rotation;
        FusedAdamParam opacity;
        FusedAdamParam sh0;
        FusedAdamParam shN;
        MrnfFoldArgs mrnf_fold{};
        MrnfNoiseArgs mrnf_noise{};
    };

} // namespace fast_lfs::rasterization

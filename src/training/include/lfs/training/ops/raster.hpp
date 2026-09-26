/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include "lfs/training/ops/domain_types.hpp"
#include "lfs/training/ops/types.hpp"

#include <array>
#include <cstdint>
#include <string_view>

namespace lfs::gpu_ops {

    enum class ShStorage { Float32,
                           IeeeFloat16,
                           Q16 };

    struct ShParams {
        ShStorage storage = ShStorage::Float32;
        uint32_t active_bases = 1;
        uint32_t layout_bases = 1;
    };

    struct SplatInputs {
        In means, raw_scales, raw_rotations, raw_opacities;
        In sh0, shN, sh_value_bounds;
    };

    struct RenderOutputs {
        Out image, alpha, depth, normal;
    };
    struct RenderGradients {
        In image, alpha, depth, normal;
    };

    // Tile size is the rendered viewport. Clipping stays 0.01 / 1e10 inside
    // the implementation.
    struct FastParams {
        HW full_image;
        Intrinsics intrinsics;
        int tile_x = 0;
        int tile_y = 0;
        int tile_w = 0;
        int tile_h = 0;
        ShParams sh;
        bool mip_filter = false;
        bool render_normal = false;
        bool render_depth = true;
    };

    // One enabled group is one parameter updated exactly once.
    // Order: means, scaling, rotation, opacity, sh0, shN.
    enum class AdamSlot : std::size_t {
        Means = 0,
        Scaling = 1,
        Rotation = 2,
        Opacity = 3,
        Sh0 = 4,
        ShN = 5,
    };

    struct BackwardAdamParam {
        Out parameter, packed_moments, joint_bounds, sh_value_bounds;
        In frozen_mask, crop_damping_mask, screen_share;

        int joint_bits = 0;
        int value_bits = 0;
        int value_cells = 0;
        int primitives = 0;
        int elements = 0;
        int attributes = 0;
        float step_size = 0.f;
        float bc2_sqrt_rcp = 1.f;
        float frozen_lr_scale = 0.f;
        float cropbox_lr_scale = 1.f;
        float screen_share_limit = 0.f;
        float screen_share_penalty = 0.f;
        bool enabled = false;
    };

    // A successful backward schedules exactly one update of every enabled Adam
    // binding, including its moment and bound updates and the requested
    // regularization and sparsity contributions. Those writes are ordered
    // before later dependent work. The trainer does its existing bookkeeping
    // once and does not run another parameter update. CUDA performs the update
    // inside backward with the fused kernels. Another backend may compute
    // gradients and apply Adam inside this same call. Disabled groups stay
    // untouched. Success means the work was submitted; it does not synchronize.
    struct BackwardAdam {
        std::array<BackwardAdamParam, 6> groups;
        Out scale_reg_loss, opacity_reg_loss;
        In sparsity_sigmoid, sparsity_z, sparsity_u, far_mask;

        float beta1 = 0.9f;
        float beta2 = 0.999f;
        float eps = 1e-15f;
        float scale_reg_weight = 0.f;
        float flatten_reg_weight = 0.f;
        float opacity_reg_weight = 0.f;
        float sparsity_rho = 0.f;
        float sparsity_grad_loss = 0.f;
        float median_extent = 0.f;
        float r_min = 1.f;
        float r_max = 300.f;
        bool per_splat_mean_step = false;
    };

    struct RasterResult {
        enum class Code { Success,
                          ResourceExhausted,
                          InstanceOverflow,
                          Failed };
        Code code = Code::Failed;
        bool has_work = false;
        std::string_view message;
    };

    struct FastSaved {
        State backend;
    };

    struct FastRasterOps {
        State (*create)();

        RasterResult (*forward)(
            FastSaved&, const SplatInputs&, In view, In camera_position,
            In bg_color, In bg_image, const FastParams&,
            const RenderOutputs&, Out max_screen_share);

        void (*backward)(
            FastSaved&, const RenderGradients&,
            Out densification, In error_map, In edge_map, Out edge_scores,
            const BackwardAdam&, DensificationType);

        void (*release)(FastSaved&) noexcept;
        void (*warmup)();
    };

} // namespace lfs::gpu_ops

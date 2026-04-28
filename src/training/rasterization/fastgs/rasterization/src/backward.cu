/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "backward.h"
#include "buffer_utils.h"
#include "helper_math.h"
#include "kernels_backward.cuh"
#include "rasterization_config.h"
#include "utils.h"
#include <functional>

void fast_lfs::rasterization::backward(
    const float* densification_error_map,
    const float* grad_image,
    const float* grad_alpha,
    const float* image,
    const float* alpha,
    const float3* means,
    const float3* scales_raw,
    const float4* rotations_raw,
    const float* raw_opacities,
    const float3* sh_coefficients_rest,
    const float4* w2c,
    const float3* cam_position,
    char* per_primitive_buffers_blob,
    char* per_tile_buffers_blob,
    const uint* sorted_primitive_indices,
    float* grad_opacity_helper,
    float3* grad_color_helper,
    float2* grad_mean2d_helper,
    float* grad_conic_helper,
    float4* grad_w2c,
    float* densification_info,
    const int n_primitives,
    const int n_instances,
    const int active_sh_bases,
    const int total_bases_sh_rest,
    const int width,
    const int height,
    const float fx,
    const float fy,
    const float cx,
    const float cy,
    bool mip_filter,
    DensificationType densification_type,
    FusedAdamSettings fused_adam) {
    const dim3 grid(div_round_up(width, config::tile_width), div_round_up(height, config::tile_height), 1);
    const int n_tiles = grid.x * grid.y;

    // These blobs are from the arena and are guaranteed to be valid
    PerPrimitiveBuffers per_primitive_buffers = PerPrimitiveBuffers::from_blob(per_primitive_buffers_blob, n_primitives);
    PerTileBuffers per_tile_buffers = PerTileBuffers::from_blob(per_tile_buffers_blob, n_tiles);

    if (n_instances != 0) {
        // Backward blend (template dispatch eliminates densification branch from inner loop)
        auto launch_blend_backward = [&]<DensificationType DENS_TYPE>() {
            kernels::backward::blend_backward_cu<DENS_TYPE><<<n_tiles, config::block_size_blend_backward>>>(
                per_tile_buffers.instance_ranges,
                sorted_primitive_indices,
                per_primitive_buffers.mean2d,
                per_primitive_buffers.conic_opacity,
                per_primitive_buffers.color,
                grad_image,
                grad_alpha,
                image,
                alpha,
                per_tile_buffers.n_contributions,
                per_tile_buffers.final_transmittance,
                grad_mean2d_helper,
                grad_conic_helper,
                grad_opacity_helper,
                grad_color_helper,
                densification_info,
                densification_error_map,
                n_primitives,
                width,
                height,
                grid.x);
        };
        if (densification_type == DensificationType::MRNF && densification_info != nullptr) {
            launch_blend_backward.template operator()<DensificationType::MRNF>();
        } else if (densification_info != nullptr && densification_error_map != nullptr) {
            launch_blend_backward.template operator()<DensificationType::MCMC>();
        } else {
            launch_blend_backward.template operator()<DensificationType::None>();
        }
        CHECK_CUDA(config::debug, "blend_backward")

        // Backward preprocess
        auto launch_preprocess_backward = [&]<bool MIP_FILTER, int ACTIVE_SH_BASES>() {
            kernels::backward::preprocess_backward_cu<MIP_FILTER, ACTIVE_SH_BASES><<<div_round_up(n_primitives, config::block_size_preprocess_backward), config::block_size_preprocess_backward>>>(
                means,
                scales_raw,
                rotations_raw,
                sh_coefficients_rest,
                w2c,
                cam_position,
                raw_opacities,
                per_primitive_buffers.n_touched_tiles,
                grad_mean2d_helper,
                grad_conic_helper,
                grad_opacity_helper,
                grad_color_helper,
                grad_w2c,
                (densification_error_map == nullptr && densification_type == DensificationType::None) ? densification_info : nullptr,
                n_primitives,
                total_bases_sh_rest,
                static_cast<float>(width),
                static_cast<float>(height),
                fx,
                fy,
                cx,
                cy,
                fused_adam);
        };
        auto launch_preprocess_backward_for_mip = [&]<int ACTIVE_SH_BASES>() {
            if (mip_filter) {
                launch_preprocess_backward.template operator()<true, ACTIVE_SH_BASES>();
            } else {
                launch_preprocess_backward.template operator()<false, ACTIVE_SH_BASES>();
            }
        };
        if (active_sh_bases <= 1) {
            launch_preprocess_backward_for_mip.template operator()<1>();
        } else if (active_sh_bases <= 4) {
            launch_preprocess_backward_for_mip.template operator()<4>();
        } else if (active_sh_bases <= 9) {
            launch_preprocess_backward_for_mip.template operator()<9>();
        } else {
            launch_preprocess_backward_for_mip.template operator()<16>();
        }
        CHECK_CUDA(config::debug, "preprocess_backward")
    }

    auto launch_invisible = [&](const FusedAdamParam& param, const char* name, const int extra_grad_kind = 0) {
        if (!param.enabled || param.n_elements <= 0 || param.n_attributes <= 0)
            return;
        kernels::backward::adam_step_invisible<<<div_round_up(param.n_elements, config::block_size_adam_step_invisible), config::block_size_adam_step_invisible>>>(
            per_primitive_buffers.n_touched_tiles,
            param,
            fused_adam,
            extra_grad_kind,
            fused_adam.beta1,
            fused_adam.beta2,
            fused_adam.eps);
        CHECK_CUDA(config::debug, name)
    };

    launch_invisible(fused_adam.means, "adam_step_invisible (means)");
    launch_invisible(fused_adam.scaling, "adam_step_invisible (scaling)", 1);
    launch_invisible(fused_adam.rotation, "adam_step_invisible (rotation)");
    launch_invisible(fused_adam.opacity, "adam_step_invisible (opacity)", 2);
    launch_invisible(fused_adam.sh0, "adam_step_invisible (sh0)");
    launch_invisible(fused_adam.shN, "adam_step_invisible (shN)");
}

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
    const float* grad_depth,
    const float* grad_normal,
    const float* image,
    const float* alpha,
    const float3* means,
    const float3* scales_raw,
    const float4* rotations_raw,
    const float* raw_opacities,
    const float4* sh_coefficients_rest,
    const float4* w2c,
    const float3* cam_position,
    const float3* primitive_normals,
    char* per_primitive_buffers_blob,
    char* per_tile_buffers_blob,
    const uint* sorted_primitive_indices,
    float* grad_opacity_helper,
    float3* grad_color_helper,
    float2* grad_mean2d_helper,
    float* grad_conic_helper,
    float* grad_depth_helper,
    float3* grad_normal_helper,
    float4* grad_w2c,
    float* densification_info,
    const int n_primitives,
    const int n_instances,
    const int active_sh_bases,
    const int sh_layout_bases,
    const int width,
    const int height,
    const float fx,
    const float fy,
    const float cx,
    const float cy,
    bool mip_filter,
    DensificationType densification_type,
    FusedAdamSettings fused_adam,
    cudaStream_t stream) {
    const dim3 grid(div_round_up(width, config::tile_width), div_round_up(height, config::tile_height), 1);
    const uint64_t n_tiles_u64 = static_cast<uint64_t>(grid.x) * static_cast<uint64_t>(grid.y);
    const int n_tiles = checked_to_int(n_tiles_u64, "n_tiles exceeds int range");
    const uint sh_layout_slots = kernels::shSlotsForBases(static_cast<uint>(sh_layout_bases));

    // These blobs are from the arena and are guaranteed to be valid
    PerPrimitiveBuffers per_primitive_buffers = PerPrimitiveBuffers::from_blob(per_primitive_buffers_blob, n_primitives);
    PerTileBuffers per_tile_buffers = PerTileBuffers::from_blob(per_tile_buffers_blob, n_tiles);
    auto* fastgs_status = per_primitive_buffers.forward_status;

    if (n_instances > 0) {
        // Backward blend (template dispatch eliminates densification branch from inner loop)
        auto launch_blend_backward_typed = [&]<DensificationType DENS_TYPE, bool NORMAL_CHANNEL>() {
            kernels::backward::blend_backward_cu<DENS_TYPE, NORMAL_CHANNEL><<<n_tiles, config::block_size_blend_backward, 0, stream>>>(
                per_tile_buffers.instance_ranges,
                sorted_primitive_indices,
                per_primitive_buffers.mean2d,
                per_primitive_buffers.conic_opacity,
                per_primitive_buffers.color,
                per_primitive_buffers.depths,
                primitive_normals,
                grad_image,
                grad_alpha,
                grad_depth,
                grad_normal,
                image,
                alpha,
                per_tile_buffers.n_contributions,
                per_tile_buffers.final_transmittance,
                grad_mean2d_helper,
                grad_conic_helper,
                grad_depth_helper,
                grad_normal_helper,
                grad_opacity_helper,
                grad_color_helper,
                densification_info,
                densification_error_map,
                fastgs_status,
                static_cast<uint>(n_instances),
                n_primitives,
                width,
                height,
                grid.x);
            LFS_CUDA_LAUNCH_CHECK(stream, "fastgs.backward.blend_backward");
        };
        auto launch_blend_backward = [&]<DensificationType DENS_TYPE>() {
            if (grad_normal != nullptr && grad_normal_helper != nullptr) {
                launch_blend_backward_typed.template operator()<DENS_TYPE, true>();
            } else {
                launch_blend_backward_typed.template operator()<DENS_TYPE, false>();
            }
        };
        if (densification_type == DensificationType::MRNF && densification_info != nullptr) {
            launch_blend_backward.template operator()<DensificationType::MRNF>();
        } else if (densification_info != nullptr && densification_error_map != nullptr) {
            launch_blend_backward.template operator()<DensificationType::MCMC>();
        } else {
            launch_blend_backward.template operator()<DensificationType::None>();
        }
        check_cuda_with_fastgs_status(cudaGetLastError(), "blend_backward", fastgs_status, "blend_backward", static_cast<uint64_t>(n_primitives), n_tiles_u64);
        sync_fastgs_phase_if_requested("blend_backward", fastgs_status, "blend_backward", static_cast<uint64_t>(n_primitives), n_tiles_u64);
    }

    // Backward preprocess — runs UNCONDITIONALLY now (handles both visible primitives'
    // backward + Adam, and invisible primitives' Adam-only momentum decay via the
    // vksplat-style fold). Replaces the previous adam_step_invisible launches.
    if (n_primitives > 0) {
        auto launch_preprocess_backward = [&]<bool MIP_FILTER, int ACTIVE_SH_BASES>() {
            kernels::backward::preprocess_backward_cu<MIP_FILTER, ACTIVE_SH_BASES><<<div_round_up(n_primitives, config::block_size_preprocess_backward), config::block_size_preprocess_backward, 0, stream>>>(
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
                grad_depth_helper,
                (grad_normal != nullptr) ? grad_normal_helper : nullptr,
                grad_opacity_helper,
                grad_color_helper,
                grad_w2c,
                (densification_error_map == nullptr && densification_type == DensificationType::None) ? densification_info : nullptr,
                n_primitives,
                static_cast<float>(width),
                static_cast<float>(height),
                fx,
                fy,
                cx,
                cy,
                sh_layout_slots,
                fused_adam);
            LFS_CUDA_LAUNCH_CHECK(stream, "fastgs.backward.preprocess_backward");
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
        check_cuda_with_fastgs_status(cudaGetLastError(), "preprocess_backward", fastgs_status, "preprocess_backward", static_cast<uint64_t>(n_primitives), n_tiles_u64);
        sync_fastgs_phase_if_requested("preprocess_backward", fastgs_status, "preprocess_backward", static_cast<uint64_t>(n_primitives), n_tiles_u64);
    }
}

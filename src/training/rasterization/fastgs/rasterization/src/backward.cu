/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "backward.h"
#include "buffer_utils.h"
#include "forward.h"
#include "helper_math.h"
#include "kernels_backward.cuh"
#include "rasterization_config.h"
#include "utils.h"
#include <atomic>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <mutex>

namespace {
    std::atomic_bool g_force_generic_preprocess_for_testing{false};
} // namespace

void fast_lfs::rasterization::set_force_generic_preprocess_for_testing(const bool force) noexcept {
    g_force_generic_preprocess_for_testing.store(force, std::memory_order_relaxed);
}

bool fast_lfs::rasterization::force_generic_preprocess_for_testing() noexcept {
    return g_force_generic_preprocess_for_testing.load(std::memory_order_relaxed);
}

bool fast_lfs::rasterization::backward(
    const float* densification_error_map,
    const float* grad_image,
    const BackgroundAlphaGradient& background_grad,
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
    const unsigned char* contributor_masks,
    const uint* contributor_positions,
    std::uint8_t* primitive_touched,
    const uint* primitive_work_indices,
    const uint* visible_indices,
    const std::uint8_t* active_block_flag,
    const uint* active_block_ids,
    const int n_active_blocks,
    char* sparse_rebound_blob,
    float* grad_opacity_helper,
    float3* grad_color_helper,
    float3* grad_color_mean_helper,
    float2* grad_mean2d_helper,
    float3* grad_conic_helper,
    float* grad_depth_helper,
    float3* grad_normal_helper,
    float4* grad_w2c,
    float3* grad_cam_position,
    float* densification_weight,
    float* densification_error,
    const int n_primitives,
    const int n_instances,
    const int n_visible,
    const int visible_capacity,
    const int active_sh_bases,
    const int sh_layout_bases,
    const int width,
    const int height,
    const float fx,
    const float fy,
    const float cx,
    const float cy,
    bool mip_filter,
    float dilation_scale,
    DensificationType densification_type,
    FusedAdamSettings fused_adam,
    // model-truth shN-rest decode binds (independent of fused Adam
    // enablement, which is off during SH warmup while the buffer is already q16).
    const float2* shN_value_bounds,
    const uint shN_value_n_cells,
    const uint shN_value_bits,
    const float* edge_weight_map,
    float* edge_score_out,
    cudaStream_t stream,
    const ColorVarianceLoss& color_variance) {
    const dim3 grid(div_round_up(width, config::tile_width), div_round_up(height, config::tile_height), 1);
    const uint64_t n_tiles_u64 = static_cast<uint64_t>(grid.x) * static_cast<uint64_t>(grid.y);
    const int n_tiles = checked_to_int(n_tiles_u64, "n_tiles exceeds int range");
    const uint sh_layout_slots = kernels::shSlotsForBases(static_cast<uint>(sh_layout_bases));

    // These blobs are from the arena and are guaranteed to be valid
    PerPrimitiveBuffers per_primitive_buffers =
        PerPrimitiveBuffers::from_persistent_blob(
            per_primitive_buffers_blob,
            visible_capacity > 0 ? visible_capacity : std::max(n_visible, 1));
    PerTileBuffers per_tile_buffers = PerTileBuffers::from_blob(per_tile_buffers_blob, n_tiles);
    auto* fastgs_status = per_primitive_buffers.forward_status;

    if (n_instances > 0) {
        // The warp cull mode is a test hook (production: on), read at launch so
        // tests can set it after forward returns.
        const int warp_cull_mode = warp_cull_mode_for_testing();

        auto launch_blend_backward_typed =
            [&]<DensificationType DENS_TYPE, bool DEPTH_CHANNEL, bool NORMAL_CHANNEL, bool BLEND_PEAK, bool EDGE_CHANNEL, int WARP_CULL_MODE, bool COLOR_VARIANCE>() {
                kernels::backward::blend_backward_cu<DENS_TYPE, DEPTH_CHANNEL, NORMAL_CHANNEL, BLEND_PEAK, EDGE_CHANNEL, WARP_CULL_MODE, COLOR_VARIANCE>
                    <<<n_tiles, config::block_size_blend_backward, 0, stream>>>(
                        per_tile_buffers.instance_ranges,
                        sorted_primitive_indices,
                        contributor_masks,
                        contributor_positions,
                        primitive_work_indices,
                        per_primitive_buffers.projected,
                        per_primitive_buffers.depths,
                        primitive_normals,
                        grad_image,
                        background_grad,
                        grad_depth,
                        grad_normal,
                        image,
                        per_tile_buffers.n_contributions,
                        per_tile_buffers.final_transmittance,
                        grad_mean2d_helper,
                        grad_conic_helper,
                        grad_depth_helper,
                        grad_normal_helper,
                        grad_opacity_helper,
                        grad_color_helper,
                        densification_weight,
                        densification_error,
                        densification_error_map,
                        edge_weight_map,
                        edge_score_out,
                        fused_adam.blend_peak,
                        primitive_touched,
                        fastgs_status,
                        static_cast<uint>(n_instances),
                        n_primitives,
                        static_cast<uint>(n_visible),
                        width,
                        height,
                        grid.x,
                        color_variance);
                LFS_CUDA_LAUNCH_CHECK(stream, "fastgs.backward.blend_backward");
            };
        auto launch_blend_backward_subtile =
            [&]<DensificationType DENS_TYPE, bool BLEND_PEAK, bool EDGE_CHANNEL>() {
                const bool use_tile_lpt = (n_tiles > 1);
                // Uncompacted forward lists keep the forward's LPT tile order valid.
                const bool reuse_forward_order = contributor_positions == nullptr && n_instances > 0;
                if (use_tile_lpt && !reuse_forward_order) {
                    const uint n_tiles_u32 = static_cast<uint>(n_tiles);
                    kernels::backward::prepare_tile_lpt_keys_bwd_cu<<<div_round_up(n_tiles_u32, 256u), 256u, 0, stream>>>(
                        per_tile_buffers.instance_ranges,
                        per_tile_buffers.tile_counts_in,
                        per_tile_buffers.tile_ids_in,
                        n_tiles_u32);
                    LFS_CUDA_LAUNCH_CHECK(stream, "fastgs.backward.prepare_tile_lpt_keys");
                    const int count_bits = std::max(1, extract_end_bit(static_cast<uint>(n_instances)));
                    size_t cub_bytes = per_tile_buffers.tile_sort_cub_bytes;
                    LFS_CUDA_CHECK_MSG(
                        cub::DeviceRadixSort::SortPairsDescending(
                            per_tile_buffers.tile_sort_cub, cub_bytes,
                            per_tile_buffers.tile_counts_in, per_tile_buffers.tile_counts_out,
                            per_tile_buffers.tile_ids_in, per_tile_buffers.tile_order,
                            n_tiles, 0, count_bits, stream),
                        "FastGS backward tile LPT sort");
                }
                static std::once_flag carveout_once;
                std::call_once(carveout_once, [] {
                    // 50% carveout gives 64 KB SMEM (fits 21 x 3 KB blocks = 42 warps/SM)
                    // while retaining 64 KB L1 cache for projected-splat reuse.
                    LFS_FASTGS_CUDA_CALL(
                        cudaFuncSetAttribute(
                            kernels::backward::blend_backward_subtile_cu<DENS_TYPE, BLEND_PEAK, EDGE_CHANNEL>,
                            cudaFuncAttributePreferredSharedMemoryCarveout,
                            50),
                        "cudaFuncSetAttribute(blend_backward_subtile_cu)");
                });
                kernels::backward::blend_backward_subtile_cu<DENS_TYPE, BLEND_PEAK, EDGE_CHANNEL>
                    <<<static_cast<uint>(n_tiles) * 4u, 64u, 0, stream>>>(
                        per_tile_buffers.instance_ranges,
                        use_tile_lpt ? per_tile_buffers.tile_order : nullptr,
                        sorted_primitive_indices,
                        contributor_masks,
                        contributor_positions,
                        primitive_work_indices,
                        per_primitive_buffers.projected,
                        grad_image,
                        background_grad,
                        per_tile_buffers.n_contributions,
                        per_tile_buffers.final_transmittance,
                        grad_mean2d_helper,
                        grad_conic_helper,
                        grad_opacity_helper,
                        grad_color_helper,
                        densification_weight,
                        densification_error,
                        densification_error_map,
                        edge_weight_map,
                        edge_score_out,
                        fused_adam.blend_peak,
                        primitive_touched,
                        fastgs_status,
                        static_cast<uint>(n_instances),
                        n_primitives,
                        static_cast<uint>(n_visible),
                        width,
                        height,
                        grid.x);
                LFS_CUDA_LAUNCH_CHECK(stream, "fastgs.backward.blend_backward_subtile");
            };
        // Specialize the common RGB-only training path so unused depth/edge channels
        // drop out of register state and warp reductions; keep generic depth+edge on rare modes.
        const bool depth_active = grad_depth != nullptr && grad_depth_helper != nullptr;
        const bool edge_active = edge_weight_map != nullptr && edge_score_out != nullptr;
        auto launch_blend = [&]<DensificationType DENS_TYPE, bool NORMAL_CHANNEL, int WARP_CULL_MODE>() {
            const bool blend_peak = fused_adam.blend_peak != nullptr;
            const bool color_var = color_variance.weight > 0.0f;
            if constexpr (!NORMAL_CHANNEL && WARP_CULL_MODE == 0) {
                if (!depth_active && !color_var) {
                    if (contributor_masks != nullptr) {
                        if (blend_peak && edge_active)
                            launch_blend_backward_subtile.template operator()<DENS_TYPE, true, true>();
                        else if (blend_peak)
                            launch_blend_backward_subtile.template operator()<DENS_TYPE, true, false>();
                        else if (edge_active)
                            launch_blend_backward_subtile.template operator()<DENS_TYPE, false, true>();
                        else
                            launch_blend_backward_subtile.template operator()<DENS_TYPE, false, false>();
                        return;
                    }
                    if (blend_peak && edge_active)
                        launch_blend_backward_typed.template operator()<DENS_TYPE, false, false, true, true, 0, false>();
                    else if (blend_peak)
                        launch_blend_backward_typed.template operator()<DENS_TYPE, false, false, true, false, 0, false>();
                    else if (edge_active)
                        launch_blend_backward_typed.template operator()<DENS_TYPE, false, false, false, true, 0, false>();
                    else
                        launch_blend_backward_typed.template operator()<DENS_TYPE, false, false, false, false, 0, false>();
                    return;
                }
            }
            if (blend_peak && color_var)
                launch_blend_backward_typed.template operator()<DENS_TYPE, true, NORMAL_CHANNEL, true, true, WARP_CULL_MODE, true>();
            else if (blend_peak)
                launch_blend_backward_typed.template operator()<DENS_TYPE, true, NORMAL_CHANNEL, true, true, WARP_CULL_MODE, false>();
            else if (color_var)
                launch_blend_backward_typed.template operator()<DENS_TYPE, true, NORMAL_CHANNEL, false, true, WARP_CULL_MODE, true>();
            else
                launch_blend_backward_typed.template operator()<DENS_TYPE, true, NORMAL_CHANNEL, false, true, WARP_CULL_MODE, false>();
        };
        auto launch_production = [&]<DensificationType DENS_TYPE>() {
            if (grad_normal != nullptr && grad_normal_helper != nullptr)
                launch_blend.template operator()<DENS_TYPE, true, 0>();
            else
                launch_blend.template operator()<DENS_TYPE, false, 0>();
        };
        if (warp_cull_mode == 1) {
            if (densification_weight != nullptr && densification_error != nullptr &&
                (densification_type == DensificationType::MRNF || densification_error_map != nullptr)) {
                launch_blend.template operator()<DensificationType::MRNF, false, 1>();
            } else if (densification_error != nullptr && densification_error_map != nullptr) {
                launch_blend.template operator()<DensificationType::MCMC, false, 1>();
            } else {
                launch_blend.template operator()<DensificationType::None, false, 1>();
            }
        } else if (warp_cull_mode == 2) {
            launch_blend.template operator()<DensificationType::None, false, 2>();
        } else if (densification_weight != nullptr && densification_error != nullptr &&
                   (densification_type == DensificationType::MRNF || densification_error_map != nullptr)) {
            // Both rows: the MRNF variant sums the weight and the (map-weighted) error.
            launch_production.template operator()<DensificationType::MRNF>();
        } else if (densification_error != nullptr && densification_error_map != nullptr) {
            // The error row alone.
            launch_production.template operator()<DensificationType::MCMC>();
        } else {
            launch_production.template operator()<DensificationType::None>();
        }
        check_cuda_with_fastgs_status(cudaGetLastError(), "blend_backward", fastgs_status, "blend_backward", static_cast<uint64_t>(n_primitives), n_tiles_u64);
        sync_fastgs_phase_if_requested("blend_backward", fastgs_status, "blend_backward", static_cast<uint64_t>(n_primitives), n_tiles_u64);
    }

    // Backward preprocess over all primitives: visible rows get their backward and
    // Adam step, invisible rows the Adam-only momentum decay. The sparse path splits
    // this into SH-rest, visible geometry and invisible geometry launches; otherwise
    // one full-N preprocess_backward_cu handles both.
    const float w_f = static_cast<float>(width);
    const float h_f = static_cast<float>(height);
    float clip_left, clip_right, clip_top, clip_bottom;
    ewa_clip_bounds(w_f, h_f, fx, fy, cx, cy, clip_left, clip_right, clip_top, clip_bottom);
    if (n_primitives > 0) {
        auto ok_geom16 = [](const FusedAdamParam& p) {
            return !p.enabled || (p.joint_bits == 16 && p.joint_packed != nullptr && p.joint_bounds != nullptr);
        };
        const bool can_use_sparse =
            visible_indices != nullptr &&
            sparse_rebound_blob != nullptr &&
            grad_color_mean_helper != nullptr &&
            n_visible > 0 &&
            fused_adam.enabled &&
            fused_adam.reg_rendered_only &&
            fused_adam.opacity_reg_weight <= 0.0f &&
            fused_adam.means.gradient_out == nullptr &&
            fused_adam.scaling.gradient_out == nullptr &&
            fused_adam.rotation.gradient_out == nullptr &&
            fused_adam.opacity.gradient_out == nullptr &&
            fused_adam.sh0.gradient_out == nullptr &&
            fused_adam.shN.gradient_out == nullptr &&
            ok_geom16(fused_adam.means) &&
            ok_geom16(fused_adam.sh0) &&
            ok_geom16(fused_adam.scaling) &&
            ok_geom16(fused_adam.rotation) &&
            ok_geom16(fused_adam.opacity) &&
            grad_w2c == nullptr &&
            !force_generic_preprocess_for_testing();

        if (can_use_sparse) {
            const SparseReboundState rebound = SparseReboundState::from_blob(sparse_rebound_blob, n_primitives);
            LFS_FASTGS_CUDA_CALL(cudaMemsetAsync(sparse_rebound_blob, 0, rebound.clear_bytes, stream),
                                 "cudaMemsetAsync(sparse_rebound_header)");

            const uint n_blocks = rebound.n_blocks;
            const bool can_update_shN =
                fused_adam.shN.enabled && sh_layout_slots > 0u && fused_adam.shN.joint_bits == 8 &&
                fused_adam.shN.joint_packed != nullptr && fused_adam.shN.joint_bounds != nullptr;
            if (can_update_shN && fused_adam.step_seed > 0u && (fused_adam.step_seed & 127u) == 0u) {
                kernels::backward::tighten_shN_joint_bounds_cu<<<n_blocks, 256u, 0, stream>>>(
                    fused_adam.shN, static_cast<uint>(n_primitives), sh_layout_slots);
                LFS_CUDA_LAUNCH_CHECK(stream, "fastgs.backward.tighten_shN_joint_bounds");
            }

            kernels::backward::prepare_sparse_bounds_cu<<<div_round_up(n_blocks, 256u), 256u, 0, stream>>>(
                fused_adam, rebound, sh_layout_slots);
            LFS_CUDA_LAUNCH_CHECK(stream, "fastgs.backward.prepare_sparse_bounds");

            const dim3 rebound_grid(std::min<uint>(n_blocks, 512u), static_cast<uint>(kSparseReboundSlots), 1u);
            kernels::backward::rebound_sparse_blocks_cu<false><<<rebound_grid, 256u, 0, stream>>>(
                fused_adam, rebound, static_cast<uint>(n_primitives), sh_layout_slots);
            LFS_CUDA_LAUNCH_CHECK(stream, "fastgs.backward.rebound_sparse_pre");

            const uint full_grid = div_round_up(static_cast<uint>(n_primitives),
                                                static_cast<uint>(config::block_size_preprocess_backward));
            // Compacted active-block launch: inactive blocks never get scheduled.
            const uint sparse_grid = (active_block_ids != nullptr)
                                         ? static_cast<uint>(std::max(n_active_blocks, 0))
                                         : full_grid;

            // Step 1: sparse SH-rest view-dir backward + lazy 8-bit joint Adam (warp-coalesced).
            if (active_sh_bases > 1 && sparse_grid > 0u) {
                kernels::backward::SparseShAdamArgs sh_adam{};
                sh_adam.shN = fused_adam.shN;
                sh_adam.lazy_last_step = fused_adam.lazy_last_step;
                sh_adam.lazy_last_step_n = fused_adam.lazy_last_step_n;
                sh_adam.current_step = fused_adam.step_seed;
                sh_adam.beta1 = fused_adam.beta1;
                sh_adam.beta2 = fused_adam.beta2;
                sh_adam.eps = fused_adam.eps;
                sh_adam.sh_rest_reg_weight = fused_adam.sh_rest_reg_weight;
                sh_adam.sh_rest_reg_loss_out = fused_adam.sh_rest_reg_loss_out;
                sh_adam.has_sh0 = fused_adam.sh0.param != nullptr;

                const bool shN_is_q16 =
                    sh_adam.shN.sh_value_bits == 16 &&
                    sh_adam.shN.sh_value_bounds != nullptr && sh_adam.shN.sh_value_n_cells > 0;
                const bool report_sh_loss = fused_adam.sh_rest_reg_loss_out != nullptr;

                auto launch_sparse_sh_impl = [&]<int ACTIVE_SH_BASES, bool UPDATE_SHN, bool VALUE_Q16>() {
                    if (report_sh_loss) {
                        kernels::backward::preprocess_backward_sparse_sh_cu<ACTIVE_SH_BASES, UPDATE_SHN, VALUE_Q16, true>
                            <<<sparse_grid, config::block_size_preprocess_backward, 0, stream>>>(
                                means, sh_coefficients_rest, cam_position, active_block_flag, active_block_ids,
                                primitive_work_indices, primitive_touched, per_primitive_buffers.projected,
                                grad_color_helper, grad_color_mean_helper,
                                static_cast<uint>(n_primitives),
                                sh_layout_slots, sh_adam, rebound,
                                shN_value_bounds, shN_value_n_cells, shN_value_bits);
                        LFS_CUDA_LAUNCH_CHECK(stream, "fastgs.backward.preprocess_backward_sparse_sh");
                    } else {
                        kernels::backward::preprocess_backward_sparse_sh_cu<ACTIVE_SH_BASES, UPDATE_SHN, VALUE_Q16, false>
                            <<<sparse_grid, config::block_size_preprocess_backward, 0, stream>>>(
                                means, sh_coefficients_rest, cam_position, active_block_flag, active_block_ids,
                                primitive_work_indices, primitive_touched, per_primitive_buffers.projected,
                                grad_color_helper, grad_color_mean_helper,
                                static_cast<uint>(n_primitives),
                                sh_layout_slots, sh_adam, rebound,
                                shN_value_bounds, shN_value_n_cells, shN_value_bits);
                        LFS_CUDA_LAUNCH_CHECK(stream, "fastgs.backward.preprocess_backward_sparse_sh");
                    }
                };
                auto launch_sparse_sh = [&]<int ACTIVE_SH_BASES>() {
                    if (!can_update_shN)
                        launch_sparse_sh_impl.template operator()<ACTIVE_SH_BASES, false, false>();
                    else if (shN_is_q16)
                        launch_sparse_sh_impl.template operator()<ACTIVE_SH_BASES, true, true>();
                    else
                        launch_sparse_sh_impl.template operator()<ACTIVE_SH_BASES, true, false>();
                };
                if (active_sh_bases <= 4)
                    launch_sparse_sh.template operator()<4>();
                else if (active_sh_bases <= 9)
                    launch_sparse_sh.template operator()<9>();
                else
                    launch_sparse_sh.template operator()<16>();
            }

            const bool report_geom_loss =
                fused_adam.dc_reg_loss_out != nullptr ||
                fused_adam.scale_reg_loss_out != nullptr ||
                fused_adam.erank_reg_loss_out != nullptr;

            // Step 2: visible geometry + sh0 backward & lazy 16-bit joint Adam (warp-coalesced).
            {
                kernels::backward::SparseGeomAdamArgs geom_adam{};
                geom_adam.means = fused_adam.means;
                geom_adam.sh0 = fused_adam.sh0;
                geom_adam.scaling = fused_adam.scaling;
                geom_adam.rotation = fused_adam.rotation;
                geom_adam.opacity = fused_adam.opacity;
                geom_adam.geom_last_step = fused_adam.geom_last_step;
                geom_adam.geom_last_step_n = fused_adam.geom_last_step_n;
                geom_adam.current_step = fused_adam.step_seed;
                geom_adam.beta1 = fused_adam.beta1;
                geom_adam.beta2 = fused_adam.beta2;
                geom_adam.eps = fused_adam.eps;
                geom_adam.dc_reg_weight = fused_adam.dc_reg_weight;
                geom_adam.scale_reg_weight = fused_adam.scale_reg_weight;
                geom_adam.scale_reg_log = fused_adam.scale_reg_log;
                geom_adam.scale_reg_normalizer = fused_adam.scale_reg_normalizer;
                geom_adam.erank_reg_weight = fused_adam.erank_reg_weight;
                geom_adam.flatten_reg_weight = fused_adam.flatten_reg_weight;
                geom_adam.rendered_count = fused_adam.rendered_count;
                geom_adam.dc_reg_loss_out = fused_adam.dc_reg_loss_out;
                geom_adam.scale_reg_loss_out = fused_adam.scale_reg_loss_out;
                geom_adam.erank_reg_loss_out = fused_adam.erank_reg_loss_out;
                geom_adam.per_splat_mean_step = fused_adam.per_splat_mean_step;
                geom_adam.mean_step_median_extent = fused_adam.mean_step_median_extent;
                geom_adam.mean_step_far_mask = fused_adam.mean_step_far_mask;
                geom_adam.mean_step_far_mask_n = fused_adam.mean_step_far_mask_n;
                geom_adam.ray_depth_state = fused_adam.ray_depth_state;
                geom_adam.ray_depth_n = fused_adam.ray_depth_n;
                geom_adam.ray_depth_lr = fused_adam.ray_depth_lr;
                geom_adam.ray_depth_mean_lr = fused_adam.ray_depth_mean_lr;
                geom_adam.ray_depth_far_limit = fused_adam.ray_depth_far_limit;
                geom_adam.mrnf_fold = fused_adam.mrnf_fold;
                geom_adam.mrnf_noise = fused_adam.mrnf_noise;

                auto launch_sparse_geom = [&]<bool MIP_FILTER, bool HAS_COLOR_MEAN_GRAD>() {
                    if (report_geom_loss) {
                        kernels::backward::preprocess_backward_sparse_geom_cu<MIP_FILTER, HAS_COLOR_MEAN_GRAD, true>
                            <<<sparse_grid, config::block_size_preprocess_backward, 0, stream>>>(
                                means, scales_raw, rotations_raw, w2c, cam_position, raw_opacities,
                                active_block_flag, active_block_ids, primitive_work_indices, primitive_touched, per_primitive_buffers.projected,
                                grad_mean2d_helper, grad_conic_helper, grad_depth_helper,
                                (grad_normal != nullptr) ? grad_normal_helper : nullptr,
                                grad_opacity_helper, grad_color_helper, grad_color_mean_helper,
                                (densification_error_map == nullptr && densification_type == DensificationType::None) ? densification_weight : nullptr,
                                (densification_error_map == nullptr && densification_type == DensificationType::None) ? densification_error : nullptr,
                                static_cast<uint>(n_primitives),
                                w_f, h_f, fx, fy, clip_left, clip_right, clip_top, clip_bottom,
                                dilation_scale, geom_adam, rebound);
                        LFS_CUDA_LAUNCH_CHECK(stream, "fastgs.backward.preprocess_backward_sparse_geom");
                    } else {
                        kernels::backward::preprocess_backward_sparse_geom_cu<MIP_FILTER, HAS_COLOR_MEAN_GRAD, false>
                            <<<sparse_grid, config::block_size_preprocess_backward, 0, stream>>>(
                                means, scales_raw, rotations_raw, w2c, cam_position, raw_opacities,
                                active_block_flag, active_block_ids, primitive_work_indices, primitive_touched, per_primitive_buffers.projected,
                                grad_mean2d_helper, grad_conic_helper, grad_depth_helper,
                                (grad_normal != nullptr) ? grad_normal_helper : nullptr,
                                grad_opacity_helper, grad_color_helper, grad_color_mean_helper,
                                (densification_error_map == nullptr && densification_type == DensificationType::None) ? densification_weight : nullptr,
                                (densification_error_map == nullptr && densification_type == DensificationType::None) ? densification_error : nullptr,
                                static_cast<uint>(n_primitives),
                                w_f, h_f, fx, fy, clip_left, clip_right, clip_top, clip_bottom,
                                dilation_scale, geom_adam, rebound);
                        LFS_CUDA_LAUNCH_CHECK(stream, "fastgs.backward.preprocess_backward_sparse_geom");
                    }
                };
                if (sparse_grid == 0u) {
                } else if (mip_filter) {
                    if (active_sh_bases > 1)
                        launch_sparse_geom.template operator()<true, true>();
                    else
                        launch_sparse_geom.template operator()<true, false>();
                } else {
                    if (active_sh_bases > 1)
                        launch_sparse_geom.template operator()<false, true>();
                    else
                        launch_sparse_geom.template operator()<false, false>();
                }
            }

            // Step 3: exact invisible geometry + sh0 momentum decay (skips active_block_flag==0
            // work-index reads and uses exact log-space sqrt(g2) cancellation for normal cells).
            {
                kernels::backward::SparseInvisGeomArgs invis_adam{};
                invis_adam.means = fused_adam.means;
                invis_adam.sh0 = fused_adam.sh0;
                invis_adam.scaling = fused_adam.scaling;
                invis_adam.rotation = fused_adam.rotation;
                invis_adam.opacity = fused_adam.opacity;
                invis_adam.step_seed = fused_adam.step_seed;
                invis_adam.geom_last_step = fused_adam.geom_last_step;
                invis_adam.geom_last_step_n = fused_adam.geom_last_step_n;
                invis_adam.beta1 = fused_adam.beta1;
                invis_adam.beta2 = fused_adam.beta2;
                invis_adam.eps = fused_adam.eps;
                invis_adam.dc_reg_weight = fused_adam.dc_reg_weight;
                invis_adam.scale_reg_weight = fused_adam.scale_reg_weight;
                invis_adam.scale_reg_log = fused_adam.scale_reg_log;
                invis_adam.scale_reg_normalizer = fused_adam.scale_reg_normalizer;
                invis_adam.erank_reg_weight = fused_adam.erank_reg_weight;
                invis_adam.dc_reg_loss_out = fused_adam.dc_reg_loss_out;
                invis_adam.scale_reg_loss_out = fused_adam.scale_reg_loss_out;
                invis_adam.erank_reg_loss_out = fused_adam.erank_reg_loss_out;
                invis_adam.per_splat_mean_step = fused_adam.per_splat_mean_step;
                invis_adam.mean_step_median_extent = fused_adam.mean_step_median_extent;
                invis_adam.mean_step_far_mask = fused_adam.mean_step_far_mask;
                invis_adam.mean_step_far_mask_n = fused_adam.mean_step_far_mask_n;
                invis_adam.mrnf_noise = fused_adam.mrnf_noise;

                auto launch_invis = [&]<int ACTIVE_SH_BASES>() {
                    if (report_geom_loss) {
                        kernels::backward::adam_decay_invisible_geom_cu<ACTIVE_SH_BASES, true>
                            <<<full_grid, config::block_size_preprocess_backward, 0, stream>>>(
                                means, sh_coefficients_rest, cam_position, active_block_flag,
                                primitive_work_indices, primitive_touched, per_primitive_buffers.projected,
                                static_cast<uint>(n_primitives), sh_layout_slots,
                                invis_adam, rebound,
                                shN_value_bounds, shN_value_n_cells, shN_value_bits);
                        LFS_CUDA_LAUNCH_CHECK(stream, "fastgs.backward.adam_decay_invisible_geom");
                    } else {
                        kernels::backward::adam_decay_invisible_geom_cu<ACTIVE_SH_BASES, false>
                            <<<full_grid, config::block_size_preprocess_backward, 0, stream>>>(
                                means, sh_coefficients_rest, cam_position, active_block_flag,
                                primitive_work_indices, primitive_touched, per_primitive_buffers.projected,
                                static_cast<uint>(n_primitives), sh_layout_slots,
                                invis_adam, rebound,
                                shN_value_bounds, shN_value_n_cells, shN_value_bits);
                        LFS_CUDA_LAUNCH_CHECK(stream, "fastgs.backward.adam_decay_invisible_geom");
                    }
                };
                if (active_sh_bases <= 1)
                    launch_invis.template operator()<1>();
                else if (active_sh_bases <= 4)
                    launch_invis.template operator()<4>();
                else if (active_sh_bases <= 9)
                    launch_invis.template operator()<9>();
                else
                    launch_invis.template operator()<16>();
            }

            kernels::backward::rebound_sparse_blocks_cu<true><<<rebound_grid, 256u, 0, stream>>>(
                fused_adam, rebound, static_cast<uint>(n_primitives), sh_layout_slots);
            LFS_CUDA_LAUNCH_CHECK(stream, "fastgs.backward.rebound_sparse_post");

            check_cuda_with_fastgs_status(cudaGetLastError(), "preprocess_backward_sparse", fastgs_status, "preprocess_backward", static_cast<uint64_t>(n_primitives), n_tiles_u64);
            sync_fastgs_phase_if_requested("preprocess_backward_sparse", fastgs_status, "preprocess_backward", static_cast<uint64_t>(n_primitives), n_tiles_u64);
            return true;
        }

        auto launch_preprocess_backward = [&]<bool MIP_FILTER, int ACTIVE_SH_BASES, bool DISABLED_FEATURES, bool CAMERA_GRAD>() {
            kernels::backward::preprocess_backward_cu<MIP_FILTER, ACTIVE_SH_BASES, DISABLED_FEATURES, CAMERA_GRAD><<<div_round_up(n_primitives, config::block_size_preprocess_backward), config::block_size_preprocess_backward, 0, stream>>>(
                means,
                scales_raw,
                rotations_raw,
                sh_coefficients_rest,
                w2c,
                cam_position,
                raw_opacities,
                active_block_flag,
                primitive_work_indices,
                primitive_touched,
                per_primitive_buffers.projected,
                grad_mean2d_helper,
                grad_conic_helper,
                grad_depth_helper,
                (grad_normal != nullptr) ? grad_normal_helper : nullptr,
                grad_opacity_helper,
                grad_color_helper,
                grad_w2c,
                grad_cam_position,
                (densification_error_map == nullptr && densification_type == DensificationType::None) ? densification_weight : nullptr,
                (densification_error_map == nullptr && densification_type == DensificationType::None) ? densification_error : nullptr,
                n_primitives,
                w_f,
                h_f,
                fx,
                fy,
                cx,
                cy,
                clip_left,
                clip_right,
                clip_top,
                clip_bottom,
                dilation_scale,
                sh_layout_slots,
                fused_adam,
                shN_value_bounds,
                shN_value_n_cells,
                shN_value_bits);
            LFS_CUDA_LAUNCH_CHECK(stream, "fastgs.backward.preprocess_backward");
        };
        const bool disabled_features = fused_adam.erank_reg_weight <= 0.0f &&
                                       fused_adam.dc_reg_weight <= 0.0f &&
                                       fused_adam.sh_rest_reg_weight <= 0.0f &&
                                       fused_adam.flatten_reg_weight <= 0.0f &&
                                       !fused_adam.scale_reg_log &&
                                       !fused_adam.per_splat_mean_step &&
                                       fused_adam.ray_depth_state == nullptr &&
                                       fused_adam.rendered_count == nullptr &&
                                       !fused_adam.reg_rendered_only;
        auto launch_preprocess_backward_for_mip = [&]<int ACTIVE_SH_BASES, bool DISABLED_FEATURES, bool CAMERA_GRAD>() {
            if (mip_filter) {
                launch_preprocess_backward.template operator()<true, ACTIVE_SH_BASES, DISABLED_FEATURES, CAMERA_GRAD>();
            } else {
                launch_preprocess_backward.template operator()<false, ACTIVE_SH_BASES, DISABLED_FEATURES, CAMERA_GRAD>();
            }
        };
        auto launch_specialized = [&]<int ACTIVE_SH_BASES>() {
            if (grad_w2c != nullptr) {
                // The camera gradient is only instantiated in the runtime-generic kernel.
                launch_preprocess_backward_for_mip.template operator()<ACTIVE_SH_BASES, false, true>();
            } else if (disabled_features && !force_generic_preprocess_for_testing()) {
                launch_preprocess_backward_for_mip.template operator()<ACTIVE_SH_BASES, true, false>();
            } else {
                launch_preprocess_backward_for_mip.template operator()<ACTIVE_SH_BASES, false, false>();
            }
        };
        if (active_sh_bases <= 1) {
            launch_specialized.template operator()<1>();
        } else if (active_sh_bases <= 4) {
            launch_specialized.template operator()<4>();
        } else if (active_sh_bases <= 9) {
            launch_specialized.template operator()<9>();
        } else {
            launch_specialized.template operator()<16>();
        }
        check_cuda_with_fastgs_status(cudaGetLastError(), "preprocess_backward", fastgs_status, "preprocess_backward", static_cast<uint64_t>(n_primitives), n_tiles_u64);
        sync_fastgs_phase_if_requested("preprocess_backward", fastgs_status, "preprocess_backward", static_cast<uint64_t>(n_primitives), n_tiles_u64);
    }
    return false;
}

void fast_lfs::rasterization::flush_lazy_adam(
    const FusedAdamSettings& fused_adam,
    const int n_primitives,
    const int active_sh_bases,
    const int sh_layout_bases,
    cudaStream_t stream) {
    if (n_primitives <= 0 || !fused_adam.enabled || fused_adam.step_seed == 0u) {
        return;
    }
    const uint full_grid = div_round_up(static_cast<uint>(n_primitives),
                                        static_cast<uint>(config::block_size_preprocess_backward));
    if (fused_adam.geom_last_step != nullptr && fused_adam.geom_last_step_n >= n_primitives &&
        fused_adam.rotation.enabled && fused_adam.rotation.joint_bits == 16 &&
        fused_adam.sh0.enabled && fused_adam.sh0.joint_bits == 16) {
        kernels::backward::SparseInvisGeomArgs geom{};
        geom.sh0 = fused_adam.sh0;
        geom.rotation = fused_adam.rotation;
        geom.geom_last_step = fused_adam.geom_last_step;
        geom.geom_last_step_n = fused_adam.geom_last_step_n;
        geom.step_seed = fused_adam.step_seed;
        geom.beta1 = fused_adam.beta1;
        geom.beta2 = fused_adam.beta2;
        geom.eps = fused_adam.eps;
        geom.dc_reg_weight = fused_adam.dc_reg_weight;
        kernels::backward::flush_lazy_geom_cu<<<full_grid, config::block_size_preprocess_backward, 0, stream>>>(
            static_cast<uint>(n_primitives), geom);
        LFS_CUDA_LAUNCH_CHECK(stream, "fastgs.backward.flush_lazy_geom");
    }
    if (fused_adam.lazy_last_step == nullptr || fused_adam.lazy_last_step_n <= 0) {
        return;
    }
    const uint sh_layout_slots = kernels::shSlotsForBases(static_cast<uint>(sh_layout_bases));
    const bool flush_shN =
        active_sh_bases > 1 && fused_adam.shN.enabled && sh_layout_slots > 0u &&
        fused_adam.shN.joint_bits == 8 &&
        fused_adam.shN.joint_packed != nullptr && fused_adam.shN.joint_bounds != nullptr;

    if (flush_shN) {
        kernels::backward::SparseShAdamArgs sh_adam{};
        sh_adam.shN = fused_adam.shN;
        sh_adam.lazy_last_step = fused_adam.lazy_last_step;
        sh_adam.lazy_last_step_n = fused_adam.lazy_last_step_n;
        sh_adam.current_step = fused_adam.step_seed;
        sh_adam.beta1 = fused_adam.beta1;
        sh_adam.beta2 = fused_adam.beta2;
        sh_adam.eps = fused_adam.eps;
        sh_adam.sh_rest_reg_weight = fused_adam.sh_rest_reg_weight;

        const bool shN_is_q16 =
            sh_adam.shN.sh_value_bits == 16 &&
            sh_adam.shN.sh_value_bounds != nullptr && sh_adam.shN.sh_value_n_cells > 0;
        auto launch_flush_sh = [&]<int ACTIVE_SH_BASES>() {
            if (shN_is_q16) {
                kernels::backward::flush_lazy_shN_cu<ACTIVE_SH_BASES, true>
                    <<<full_grid, config::block_size_preprocess_backward, 0, stream>>>(
                        static_cast<uint>(n_primitives), sh_layout_slots, sh_adam);
                LFS_CUDA_LAUNCH_CHECK(stream, "fastgs.backward.flush_lazy_shN");
            } else {
                kernels::backward::flush_lazy_shN_cu<ACTIVE_SH_BASES, false>
                    <<<full_grid, config::block_size_preprocess_backward, 0, stream>>>(
                        static_cast<uint>(n_primitives), sh_layout_slots, sh_adam);
                LFS_CUDA_LAUNCH_CHECK(stream, "fastgs.backward.flush_lazy_shN");
            }
        };
        if (active_sh_bases <= 4)
            launch_flush_sh.template operator()<4>();
        else if (active_sh_bases <= 9)
            launch_flush_sh.template operator()<9>();
        else
            launch_flush_sh.template operator()<16>();
    }
}

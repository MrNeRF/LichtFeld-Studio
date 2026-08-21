/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include "buffer_utils.h"
#include "helper_math.h"
#include "kernel_utils.cuh"
#include "lfs/core/warp_reduce.cuh"
#include "rasterization_config.h"
#include "utils.h"
#include <cooperative_groups.h>
#include <cstdint>
namespace cg = cooperative_groups;

namespace fast_lfs::rasterization::kernels::backward {

    // Gradient clamping to prevent NaN from exploding gradients
    constexpr float GRAD_CLAMP_MAX = 1e4f;

    __device__ inline float clamp_grad(const float g) {
        return fminf(fmaxf(g, -GRAD_CLAMP_MAX), GRAD_CLAMP_MAX);
    }

    __device__ inline float3 clamp_grad3(const float3 g) {
        return make_float3(clamp_grad(g.x), clamp_grad(g.y), clamp_grad(g.z));
    }

    __device__ inline float4 clamp_grad4(const float4 g) {
        return make_float4(clamp_grad(g.x), clamp_grad(g.y), clamp_grad(g.z), clamp_grad(g.w));
    }

    // minBlocks=3 budgets registers so shN Adam no longer lives across
    // the geometry backward (sweep 2..4; 3 is the measured default).
    template <bool MIP_FILTER, int ACTIVE_SH_BASES>
    __global__ void __launch_bounds__(config::block_size_preprocess_backward, 3) preprocess_backward_cu(
        // These four alias fused_adam.{means,scaling,rotation,opacity}.param (Adam writes).
        const float3* means,
        const float3* raw_scales,
        const float4* raw_rotations,
        const float4* __restrict__ sh_coefficients_rest, // compact float4-packed swizzled layout
        const float4* __restrict__ w2c,
        const float3* __restrict__ cam_position,
        const float* raw_opacities,
        const std::uint64_t* __restrict__ primitive_n_touched_tiles,
        const float2* __restrict__ grad_mean2d,
        const float3* __restrict__ grad_conic,
        const float* __restrict__ grad_depth,
        const float3* __restrict__ grad_normal,
        float* __restrict__ grad_opacity_helper,
        float3* __restrict__ grad_color_helper,
        float4* __restrict__ grad_w2c,
        float* __restrict__ densification_info,
        const uint n_primitives,
        const float w,
        const float h,
        const float fx,
        const float fy,
        const float cx,
        const float cy,
        const float clip_left,
        const float clip_right,
        const float clip_top,
        const float clip_bottom,
        const uint sh_layout_slots,
        FusedAdamSettings fused_adam,
        // model-truth shN-rest decode binds. fused_adam.shN.sh_value_*
        // is enablement-gated (null during SH warmup) and gates only the UPDATE
        // path; reading sh_coefficients_rest must always use these.
        const float2* __restrict__ shN_value_bounds,
        const uint shN_value_n_cells,
        const uint shN_value_bits) {
        (void)cx;
        (void)cy;
        auto primitive_idx = cg::this_grid().thread_rank();
        const bool in_range = primitive_idx < n_primitives;

        // Fold scale/opacity regularizer *loss scalars* into this kernel (grads were
        // already fused). Each thread contributes its primitive's share of
        // weight * mean(·); block-reduce + atomicAdd into caller-owned device scalars
        // so the trainer can drop the loss-only full-N reduction kernels + their
        // per-call empty({num_blocks})/empty({1}) allocs.
        float scale_loss_local = 0.0f;
        float opacity_loss_local = 0.0f;
        if (in_range) {
            if (fused_adam.scale_reg_loss_out != nullptr &&
                fused_adam.scale_reg_weight > 0.0f &&
                fused_adam.scaling.param != nullptr &&
                fused_adam.scaling.n_elements > 0) {
                const uint scale_base = static_cast<uint>(primitive_idx) * 3u;
                const float inv_n = 1.0f / static_cast<float>(fused_adam.scaling.n_elements);
                scale_loss_local = fused_adam.scale_reg_weight * inv_n *
                                   (expf(fused_adam.scaling.param[scale_base]) +
                                    expf(fused_adam.scaling.param[scale_base + 1]) +
                                    expf(fused_adam.scaling.param[scale_base + 2]));
            }
            if (fused_adam.opacity_reg_loss_out != nullptr &&
                fused_adam.opacity_reg_weight > 0.0f &&
                fused_adam.opacity.param != nullptr &&
                fused_adam.opacity.n_elements > 0) {
                const float opa = sigmoid(fused_adam.opacity.param[primitive_idx]);
                opacity_loss_local = fused_adam.opacity_reg_weight * opa /
                                     static_cast<float>(fused_adam.opacity.n_elements);
            }
        }
        if (fused_adam.scale_reg_loss_out != nullptr) {
            const float block_sum = lfs::core::warp_ops::block_reduce_sum(scale_loss_local);
            if (threadIdx.x == 0 && block_sum != 0.0f) {
                atomicAdd(fused_adam.scale_reg_loss_out, block_sum);
            }
        }
        if (fused_adam.opacity_reg_loss_out != nullptr) {
            const float block_sum = lfs::core::warp_ops::block_reduce_sum(opacity_loss_local);
            if (threadIdx.x == 0 && block_sum != 0.0f) {
                atomicAdd(fused_adam.opacity_reg_loss_out, block_sum);
            }
        }

        // Grad buffers. Joint block-bounds needs all threads to hit the same
        // adam_step_row sequence — no early returns before Adam sections.
        // sh0/shN Adam runs immediately after SH-dir backward so the SH
        // streaming temporaries do not live across covariance/EWA.
        float mean_grads[3] = {0.0f, 0.0f, 0.0f};
        float rotation_grads[4] = {0.0f, 0.0f, 0.0f, 0.0f};
        float sh0_grads[3] = {0.0f, 0.0f, 0.0f};
        float scale_grads[3] = {0.0f, 0.0f, 0.0f};
        float opacity_grads[1] = {0.0f};
        float3 dL_dmean3d_from_color = make_float3(0.0f, 0.0f, 0.0f);

        const bool invisible = in_range && primitive_n_touched_tiles[primitive_idx] == 0;
        const bool visible = in_range && !invisible;

        // Compute SH backward gradients before entering the geometry path.
        if (invisible) {
            // Reg-only scale/opacity; SH/means/rot get pure momentum decay (zeros).
            const uint scale_base = static_cast<uint>(primitive_idx) * 3u;
            scale_grads[0] = scale_regularization_grad(fused_adam, fused_adam.scaling, scale_base);
            scale_grads[1] = scale_regularization_grad(fused_adam, fused_adam.scaling, scale_base + 1);
            scale_grads[2] = scale_regularization_grad(fused_adam, fused_adam.scaling, scale_base + 2);
            add_flatten_regularization_grads(fused_adam, fused_adam.scaling, scale_base, scale_grads);
            opacity_grads[0] = opacity_extra_grad(fused_adam, fused_adam.opacity, static_cast<uint>(primitive_idx));
        } else if (visible) {
            const float3 mean3d = means[primitive_idx];
            dL_dmean3d_from_color = convert_sh_to_color_backward_grads<ACTIVE_SH_BASES>(
                sh_coefficients_rest, grad_color_helper,
                mean3d, cam_position[0],
                primitive_idx,
                sh_layout_slots,
                sh0_grads,
                // q16: bits==16 + bounds. IEEE f16: bits==16 + null bounds.
                // decode from the model bind, NOT fused_adam.shN
                // the optimizer copy is null during SH warmup while the buffer
                // is already q16 (first degree-up ≤ warmup misread → MMU fault).
                shN_value_bounds,
                shN_value_bounds != nullptr ? shN_value_n_cells : 0u,
                shN_value_bits == 16u ? 16u : 0u);
        } // close visible — SH Adam next kills shN live range before geometry

        // Apply SH Adam for all threads so joint block-bounds reduction remains valid.
        // Also the single re-encode site for SH value quant.
        adam_step_row(sh0_grads, fused_adam.sh0, primitive_idx, 3, fused_adam.beta1, fused_adam.beta2, fused_adam.eps);
        if constexpr (ACTIVE_SH_BASES > 1) {
            const float3 sh_mean = visible ? means[primitive_idx] : make_float3(0.0f, 0.0f, 0.0f);
            const float3 sh_cam = visible ? cam_position[0] : make_float3(0.0f, 0.0f, 0.0f);
            const float3 sh_gc = visible ? grad_color_helper[primitive_idx] : make_float3(0.0f, 0.0f, 0.0f);
            apply_shN_grads_packed<ACTIVE_SH_BASES>(
                fused_adam, primitive_idx, sh_layout_slots,
                sh_mean, sh_cam, sh_gc, visible);
        }
        // sh0 / streamed shN Adam state is dead from here on (compiler can free it).

        // Re-enter the visible-only path for covariance, EWA, and geometry gradients.
        if (visible) {
            const float3 mean3d = means[primitive_idx];

            const float4 w2c_r3 = w2c[2];
            const float depth = w2c_r3.x * mean3d.x + w2c_r3.y * mean3d.y + w2c_r3.z * mean3d.z + w2c_r3.w;
            const float depth_safe = fmaxf(depth, 1e-4f);
            const float inv_depth = 1.0f / depth_safe;
            const float4 w2c_r1 = w2c[0];
            const float x = (w2c_r1.x * mean3d.x + w2c_r1.y * mean3d.y + w2c_r1.z * mean3d.z + w2c_r1.w) * inv_depth;
            const float4 w2c_r2 = w2c[1];
            const float y = (w2c_r2.x * mean3d.x + w2c_r2.y * mean3d.y + w2c_r2.z * mean3d.z + w2c_r2.w) * inv_depth;

            // compute 3d covariance from raw scale and rotation
            const float3 raw_scale = raw_scales[primitive_idx];
            const float3 clamped_scale = make_float3(
                fminf(raw_scale.x, config::max_raw_scale),
                fminf(raw_scale.y, config::max_raw_scale),
                fminf(raw_scale.z, config::max_raw_scale));
            const float3 variance = make_float3(expf(2.0f * clamped_scale.x), expf(2.0f * clamped_scale.y), expf(2.0f * clamped_scale.z));
            const float4 raw_rotation = raw_rotations[primitive_idx];
            const float qr = raw_rotation.x;
            const float qx = raw_rotation.y;
            const float qy = raw_rotation.z;
            const float qz = raw_rotation.w;
            const float qrr_raw = qr * qr, qxx_raw = qx * qx, qyy_raw = qy * qy, qzz_raw = qz * qz;
            const float q_norm_sq = qrr_raw + qxx_raw + qyy_raw + qzz_raw;
            const float q_norm_sq_safe = fmaxf(q_norm_sq, 1e-8f);
            const float inv_q_norm_sq = 2.0f / q_norm_sq_safe;
            const float qxx = qxx_raw * inv_q_norm_sq, qyy = qyy_raw * inv_q_norm_sq, qzz = qzz_raw * inv_q_norm_sq;
            const float qxy = qx * qy * inv_q_norm_sq, qxz = qx * qz * inv_q_norm_sq, qyz = qy * qz * inv_q_norm_sq;
            const float qrx = qr * qx * inv_q_norm_sq, qry = qr * qy * inv_q_norm_sq, qrz = qr * qz * inv_q_norm_sq;
            const mat3x3 rotation = {
                1.0f - (qyy + qzz), qxy - qrz, qry + qxz,
                qrz + qxy, 1.0f - (qxx + qzz), qyz - qrx,
                qxz - qry, qrx + qyz, 1.0f - (qxx + qyy)};
            const mat3x3 rotation_scaled = {
                rotation.m11 * variance.x, rotation.m12 * variance.y, rotation.m13 * variance.z,
                rotation.m21 * variance.x, rotation.m22 * variance.y, rotation.m23 * variance.z,
                rotation.m31 * variance.x, rotation.m32 * variance.y, rotation.m33 * variance.z};
            const mat3x3_triu cov3d{
                rotation_scaled.m11 * rotation.m11 + rotation_scaled.m12 * rotation.m12 + rotation_scaled.m13 * rotation.m13,
                rotation_scaled.m11 * rotation.m21 + rotation_scaled.m12 * rotation.m22 + rotation_scaled.m13 * rotation.m23,
                rotation_scaled.m11 * rotation.m31 + rotation_scaled.m12 * rotation.m32 + rotation_scaled.m13 * rotation.m33,
                rotation_scaled.m21 * rotation.m21 + rotation_scaled.m22 * rotation.m22 + rotation_scaled.m23 * rotation.m23,
                rotation_scaled.m21 * rotation.m31 + rotation_scaled.m22 * rotation.m32 + rotation_scaled.m23 * rotation.m33,
                rotation_scaled.m31 * rotation.m31 + rotation_scaled.m32 * rotation.m32 + rotation_scaled.m33 * rotation.m33,
            };

            // ewa splatting gradient helpers (clip box is grid-uniform; computed once on the host)
            const float tx = clamp(x, clip_left, clip_right);
            const float ty = clamp(y, clip_top, clip_bottom);
            const float j11 = fx * inv_depth;
            const float j13 = -j11 * tx;
            const float j22 = fy * inv_depth;
            const float j23 = -j22 * ty;
            const float3 jw_r1 = make_float3(
                j11 * w2c_r1.x + j13 * w2c_r3.x,
                j11 * w2c_r1.y + j13 * w2c_r3.y,
                j11 * w2c_r1.z + j13 * w2c_r3.z);
            const float3 jw_r2 = make_float3(
                j22 * w2c_r2.x + j23 * w2c_r3.x,
                j22 * w2c_r2.y + j23 * w2c_r3.y,
                j22 * w2c_r2.z + j23 * w2c_r3.z);
            const float3 jwc_r1 = make_float3(
                jw_r1.x * cov3d.m11 + jw_r1.y * cov3d.m12 + jw_r1.z * cov3d.m13,
                jw_r1.x * cov3d.m12 + jw_r1.y * cov3d.m22 + jw_r1.z * cov3d.m23,
                jw_r1.x * cov3d.m13 + jw_r1.y * cov3d.m23 + jw_r1.z * cov3d.m33);
            const float3 jwc_r2 = make_float3(
                jw_r2.x * cov3d.m11 + jw_r2.y * cov3d.m12 + jw_r2.z * cov3d.m13,
                jw_r2.x * cov3d.m12 + jw_r2.y * cov3d.m22 + jw_r2.z * cov3d.m23,
                jw_r2.x * cov3d.m13 + jw_r2.y * cov3d.m23 + jw_r2.z * cov3d.m33);

            // 2d covariance gradient (use same dilation as forward pass)
            constexpr float kernel_size = MIP_FILTER ? config::dilation_mip_filter : config::dilation;
            const float raw_a = dot(jwc_r1, jw_r1);
            const float raw_b = dot(jwc_r1, jw_r2);
            const float raw_c = dot(jwc_r2, jw_r2);
            const float a = raw_a + kernel_size, b = raw_b, c = raw_c + kernel_size;
            const float aa = a * a, bb = b * b, cc = c * c;
            const float ac = a * c, ab = a * b, bc = b * c;
            const float determinant = ac - bb;
            const float determinant_safe = fmaxf(determinant, config::min_cov2d_determinant);
            const float determinant_rcp = 1.0f / determinant_safe;
            const float determinant_rcp_sq = determinant_rcp * determinant_rcp;
            const float3 dL_dconic = grad_conic[primitive_idx];
            float3 dL_dcov2d = determinant_rcp_sq * make_float3(
                                                        2.0f * bc * dL_dconic.y - cc * dL_dconic.x - bb * dL_dconic.z,
                                                        bc * dL_dconic.x - (ac + bb) * dL_dconic.y + ab * dL_dconic.z,
                                                        2.0f * ab * dL_dconic.y - bb * dL_dconic.x - aa * dL_dconic.z);

            const float original_opacity = 1.0f / (1.0f + expf(-raw_opacities[primitive_idx]));
            const float grad_compensated_opacity = grad_opacity_helper[primitive_idx];
            float opacity_compensation = 1.0f;
            if constexpr (MIP_FILTER) {
                const float det_raw = raw_a * raw_c - raw_b * raw_b;
                if (det_raw > config::min_cov2d_determinant && determinant > config::min_cov2d_determinant) {
                    opacity_compensation = sqrtf(det_raw * determinant_rcp);
                } else {
                    opacity_compensation = 0.0f;
                }
            }

            const float sigmoid_derivative = original_opacity * (1.0f - original_opacity);
            opacity_grads[0] = grad_compensated_opacity * opacity_compensation * sigmoid_derivative +
                               opacity_extra_grad(fused_adam, fused_adam.opacity, primitive_idx);

            // 3d covariance gradient
            const mat3x3_triu dL_dcov3d = {
                (jw_r1.x * jw_r1.x) * dL_dcov2d.x + 2.0f * (jw_r1.x * jw_r2.x) * dL_dcov2d.y + (jw_r2.x * jw_r2.x) * dL_dcov2d.z,
                (jw_r1.x * jw_r1.y) * dL_dcov2d.x + (jw_r1.x * jw_r2.y + jw_r1.y * jw_r2.x) * dL_dcov2d.y + (jw_r2.x * jw_r2.y) * dL_dcov2d.z,
                (jw_r1.x * jw_r1.z) * dL_dcov2d.x + (jw_r1.x * jw_r2.z + jw_r1.z * jw_r2.x) * dL_dcov2d.y + (jw_r2.x * jw_r2.z) * dL_dcov2d.z,
                (jw_r1.y * jw_r1.y) * dL_dcov2d.x + 2.0f * (jw_r1.y * jw_r2.y) * dL_dcov2d.y + (jw_r2.y * jw_r2.y) * dL_dcov2d.z,
                (jw_r1.y * jw_r1.z) * dL_dcov2d.x + (jw_r1.y * jw_r2.z + jw_r1.z * jw_r2.y) * dL_dcov2d.y + (jw_r2.y * jw_r2.z) * dL_dcov2d.z,
                (jw_r1.z * jw_r1.z) * dL_dcov2d.x + 2.0f * (jw_r1.z * jw_r2.z) * dL_dcov2d.y + (jw_r2.z * jw_r2.z) * dL_dcov2d.z,
            };

            // gradient of J * W
            const float3 dL_djw_r1 = 2.0f * make_float3(
                                                jwc_r1.x * dL_dcov2d.x + jwc_r2.x * dL_dcov2d.y,
                                                jwc_r1.y * dL_dcov2d.x + jwc_r2.y * dL_dcov2d.y,
                                                jwc_r1.z * dL_dcov2d.x + jwc_r2.z * dL_dcov2d.y);
            const float3 dL_djw_r2 = 2.0f * make_float3(
                                                jwc_r1.x * dL_dcov2d.y + jwc_r2.x * dL_dcov2d.z,
                                                jwc_r1.y * dL_dcov2d.y + jwc_r2.y * dL_dcov2d.z,
                                                jwc_r1.z * dL_dcov2d.y + jwc_r2.z * dL_dcov2d.z);

            // gradient of non-zero entries in J
            const float dL_dj11 = w2c_r1.x * dL_djw_r1.x + w2c_r1.y * dL_djw_r1.y + w2c_r1.z * dL_djw_r1.z;
            const float dL_dj22 = w2c_r2.x * dL_djw_r2.x + w2c_r2.y * dL_djw_r2.y + w2c_r2.z * dL_djw_r2.z;
            const float dL_dj13 = w2c_r3.x * dL_djw_r1.x + w2c_r3.y * dL_djw_r1.y + w2c_r3.z * dL_djw_r1.z;
            const float dL_dj23 = w2c_r3.x * dL_djw_r2.x + w2c_r3.y * dL_djw_r2.y + w2c_r3.z * dL_djw_r2.z;

            // mean3d camera space gradient from J and mean2d (accounts for tx/ty clipping)
            const float2 dL_dmean2d = grad_mean2d[primitive_idx];
            const float dL_ddepth = grad_depth ? grad_depth[primitive_idx] : 0.0f;
            float3 dL_dmean3d_cam = make_float3(
                j11 * dL_dmean2d.x,
                j22 * dL_dmean2d.y,
                -j11 * x * dL_dmean2d.x - j22 * y * dL_dmean2d.y + dL_ddepth);
            const bool valid_x = x >= clip_left && x <= clip_right;
            const bool valid_y = y >= clip_top && y <= clip_bottom;
            if (valid_x)
                dL_dmean3d_cam.x -= j11 * dL_dj13 * inv_depth;
            if (valid_y)
                dL_dmean3d_cam.y -= j22 * dL_dj23 * inv_depth;
            const float factor_x = 1.0f + static_cast<float>(valid_x);
            const float factor_y = 1.0f + static_cast<float>(valid_y);
            dL_dmean3d_cam.z += (j11 * (factor_x * tx * dL_dj13 - dL_dj11) + j22 * (factor_y * ty * dL_dj23 - dL_dj22)) * inv_depth;

            if (grad_w2c != nullptr) {
                atomicAdd(&grad_w2c[0].w, dL_dmean3d_cam.x);
                atomicAdd(&grad_w2c[1].w, dL_dmean3d_cam.y);
                atomicAdd(&grad_w2c[2].w, dL_dmean3d_cam.z);
                atomicAdd(&grad_w2c[0].x, dL_dmean3d_cam.x * mean3d.x);
                atomicAdd(&grad_w2c[0].y, dL_dmean3d_cam.x * mean3d.y);
                atomicAdd(&grad_w2c[0].z, dL_dmean3d_cam.x * mean3d.z);
                atomicAdd(&grad_w2c[1].x, dL_dmean3d_cam.y * mean3d.x);
                atomicAdd(&grad_w2c[1].y, dL_dmean3d_cam.y * mean3d.y);
                atomicAdd(&grad_w2c[1].z, dL_dmean3d_cam.y * mean3d.z);
                atomicAdd(&grad_w2c[2].x, dL_dmean3d_cam.z * mean3d.x);
                atomicAdd(&grad_w2c[2].y, dL_dmean3d_cam.z * mean3d.y);
                atomicAdd(&grad_w2c[2].z, dL_dmean3d_cam.z * mean3d.z);
            }

            // Combine the splatting gradient with the saved SH color gradient.
            const float3 dL_dmean3d_from_splatting = make_float3(
                w2c_r1.x * dL_dmean3d_cam.x + w2c_r2.x * dL_dmean3d_cam.y + w2c_r3.x * dL_dmean3d_cam.z,
                w2c_r1.y * dL_dmean3d_cam.x + w2c_r2.y * dL_dmean3d_cam.y + w2c_r3.y * dL_dmean3d_cam.z,
                w2c_r1.z * dL_dmean3d_cam.x + w2c_r2.z * dL_dmean3d_cam.y + w2c_r3.z * dL_dmean3d_cam.z);

            const float3 dL_dmean3d = dL_dmean3d_from_splatting + dL_dmean3d_from_color;
            const float3 clamped_mean = clamp_grad3(dL_dmean3d);
            mean_grads[0] = clamped_mean.x;
            mean_grads[1] = clamped_mean.y;
            mean_grads[2] = clamped_mean.z;

            // raw scale gradient (zero gradient for clamped scales)
            const float dL_dvariance_x = rotation.m11 * rotation.m11 * dL_dcov3d.m11 + rotation.m21 * rotation.m21 * dL_dcov3d.m22 + rotation.m31 * rotation.m31 * dL_dcov3d.m33 +
                                         2.0f * (rotation.m11 * rotation.m21 * dL_dcov3d.m12 + rotation.m11 * rotation.m31 * dL_dcov3d.m13 + rotation.m21 * rotation.m31 * dL_dcov3d.m23);
            const float dL_dvariance_y = rotation.m12 * rotation.m12 * dL_dcov3d.m11 + rotation.m22 * rotation.m22 * dL_dcov3d.m22 + rotation.m32 * rotation.m32 * dL_dcov3d.m33 +
                                         2.0f * (rotation.m12 * rotation.m22 * dL_dcov3d.m12 + rotation.m12 * rotation.m32 * dL_dcov3d.m13 + rotation.m22 * rotation.m32 * dL_dcov3d.m23);
            const float dL_dvariance_z = rotation.m13 * rotation.m13 * dL_dcov3d.m11 + rotation.m23 * rotation.m23 * dL_dcov3d.m22 + rotation.m33 * rotation.m33 * dL_dcov3d.m33 +
                                         2.0f * (rotation.m13 * rotation.m23 * dL_dcov3d.m12 + rotation.m13 * rotation.m33 * dL_dcov3d.m13 + rotation.m23 * rotation.m33 * dL_dcov3d.m23);
            const float3 dL_draw_scale = make_float3(
                (raw_scale.x < config::max_raw_scale) ? 2.0f * variance.x * dL_dvariance_x : 0.0f,
                (raw_scale.y < config::max_raw_scale) ? 2.0f * variance.y * dL_dvariance_y : 0.0f,
                (raw_scale.z < config::max_raw_scale) ? 2.0f * variance.z * dL_dvariance_z : 0.0f);
            const float3 clamped_scale_grad = clamp_grad3(dL_draw_scale);
            const uint scale_base = primitive_idx * 3;
            scale_grads[0] = clamped_scale_grad.x + scale_regularization_grad(fused_adam, fused_adam.scaling, scale_base);
            scale_grads[1] = clamped_scale_grad.y + scale_regularization_grad(fused_adam, fused_adam.scaling, scale_base + 1);
            scale_grads[2] = clamped_scale_grad.z + scale_regularization_grad(fused_adam, fused_adam.scaling, scale_base + 2);
            add_flatten_regularization_grads(fused_adam, fused_adam.scaling, scale_base, scale_grads);

            // raw rotation gradient
            mat3x3 dL_drotation = {
                2.0f * (rotation_scaled.m11 * dL_dcov3d.m11 + rotation_scaled.m21 * dL_dcov3d.m12 + rotation_scaled.m31 * dL_dcov3d.m13),
                2.0f * (rotation_scaled.m12 * dL_dcov3d.m11 + rotation_scaled.m22 * dL_dcov3d.m12 + rotation_scaled.m32 * dL_dcov3d.m13),
                2.0f * (rotation_scaled.m13 * dL_dcov3d.m11 + rotation_scaled.m23 * dL_dcov3d.m12 + rotation_scaled.m33 * dL_dcov3d.m13),
                2.0f * (rotation_scaled.m11 * dL_dcov3d.m12 + rotation_scaled.m21 * dL_dcov3d.m22 + rotation_scaled.m31 * dL_dcov3d.m23),
                2.0f * (rotation_scaled.m12 * dL_dcov3d.m12 + rotation_scaled.m22 * dL_dcov3d.m22 + rotation_scaled.m32 * dL_dcov3d.m23),
                2.0f * (rotation_scaled.m13 * dL_dcov3d.m12 + rotation_scaled.m23 * dL_dcov3d.m22 + rotation_scaled.m33 * dL_dcov3d.m23),
                2.0f * (rotation_scaled.m11 * dL_dcov3d.m13 + rotation_scaled.m21 * dL_dcov3d.m23 + rotation_scaled.m31 * dL_dcov3d.m33),
                2.0f * (rotation_scaled.m12 * dL_dcov3d.m13 + rotation_scaled.m22 * dL_dcov3d.m23 + rotation_scaled.m32 * dL_dcov3d.m33),
                2.0f * (rotation_scaled.m13 * dL_dcov3d.m13 + rotation_scaled.m23 * dL_dcov3d.m23 + rotation_scaled.m33 * dL_dcov3d.m33)};

            if (grad_normal != nullptr) {
                const float3 g_cam = grad_normal[primitive_idx];
                const float3 g_world = make_float3(
                    w2c_r1.x * g_cam.x + w2c_r2.x * g_cam.y + w2c_r3.x * g_cam.z,
                    w2c_r1.y * g_cam.x + w2c_r2.y * g_cam.y + w2c_r3.y * g_cam.z,
                    w2c_r1.z * g_cam.x + w2c_r2.z * g_cam.y + w2c_r3.z * g_cam.z);
                const float3 view_dir = mean3d - cam_position[0];
                if (variance.x <= variance.y && variance.x <= variance.z) {
                    const float axis_dot = rotation.m11 * view_dir.x + rotation.m21 * view_dir.y + rotation.m31 * view_dir.z;
                    const float sign = axis_dot > 0.0f ? -1.0f : 1.0f;
                    dL_drotation.m11 += sign * g_world.x;
                    dL_drotation.m21 += sign * g_world.y;
                    dL_drotation.m31 += sign * g_world.z;
                } else if (variance.y <= variance.z) {
                    const float axis_dot = rotation.m12 * view_dir.x + rotation.m22 * view_dir.y + rotation.m32 * view_dir.z;
                    const float sign = axis_dot > 0.0f ? -1.0f : 1.0f;
                    dL_drotation.m12 += sign * g_world.x;
                    dL_drotation.m22 += sign * g_world.y;
                    dL_drotation.m32 += sign * g_world.z;
                } else {
                    const float axis_dot = rotation.m13 * view_dir.x + rotation.m23 * view_dir.y + rotation.m33 * view_dir.z;
                    const float sign = axis_dot > 0.0f ? -1.0f : 1.0f;
                    dL_drotation.m13 += sign * g_world.x;
                    dL_drotation.m23 += sign * g_world.y;
                    dL_drotation.m33 += sign * g_world.z;
                }
            }

            const float dL_dqxx = -dL_drotation.m22 - dL_drotation.m33;
            const float dL_dqyy = -dL_drotation.m11 - dL_drotation.m33;
            const float dL_dqzz = -dL_drotation.m11 - dL_drotation.m22;
            const float dL_dqxy = dL_drotation.m12 + dL_drotation.m21;
            const float dL_dqxz = dL_drotation.m13 + dL_drotation.m31;
            const float dL_dqyz = dL_drotation.m23 + dL_drotation.m32;
            const float dL_dqrx = dL_drotation.m32 - dL_drotation.m23;
            const float dL_dqry = dL_drotation.m13 - dL_drotation.m31;
            const float dL_dqrz = dL_drotation.m21 - dL_drotation.m12;
            const float dL_dq_norm_helper = qxx * dL_dqxx + qyy * dL_dqyy + qzz * dL_dqzz + qxy * dL_dqxy + qxz * dL_dqxz + qyz * dL_dqyz + qrx * dL_dqrx + qry * dL_dqry + qrz * dL_dqrz;
            const float4 dL_draw_rotation = make_float4(qx * dL_dqrx + qy * dL_dqry + qz * dL_dqrz - qr * dL_dq_norm_helper, 2.0f * qx * dL_dqxx + qy * dL_dqxy + qz * dL_dqxz + qr * dL_dqrx - qx * dL_dq_norm_helper, 2.0f * qy * dL_dqyy + qx * dL_dqxy + qz * dL_dqyz + qr * dL_dqry - qy * dL_dq_norm_helper, 2.0f * qz * dL_dqzz + qx * dL_dqxz + qy * dL_dqyz + qr * dL_dqrz - qz * dL_dq_norm_helper) * inv_q_norm_sq;
            const float4 clamped_rotation = clamp_grad4(dL_draw_rotation);
            rotation_grads[0] = clamped_rotation.x;
            rotation_grads[1] = clamped_rotation.y;
            rotation_grads[2] = clamped_rotation.z;
            rotation_grads[3] = clamped_rotation.w;

            if (densification_info != nullptr) {
                densification_info[primitive_idx] += 1.0f;
                densification_info[n_primitives + primitive_idx] += length(dL_dmean2d * make_float2(0.5f * w, 0.5f * h));
            }
        } // visible geometry

        // Apply geometry Adam after sh0 and shN have already been updated.
        adam_step_row(mean_grads, fused_adam.means, primitive_idx, 3, fused_adam.beta1, fused_adam.beta2, fused_adam.eps);
        adam_step_row(rotation_grads, fused_adam.rotation, primitive_idx, 4, fused_adam.beta1, fused_adam.beta2, fused_adam.eps);
        adam_step_row(scale_grads, fused_adam.scaling, primitive_idx, 3, fused_adam.beta1, fused_adam.beta2, fused_adam.eps);
        adam_step_row(opacity_grads, fused_adam.opacity, primitive_idx, 1, fused_adam.beta1, fused_adam.beta2, fused_adam.eps);
    }

    struct BlendBackwardAccum {
        float mean_x;
        float mean_y;
        float conic_x;
        float conic_y;
        float conic_z;
        float compensated_opacity;
        float color_x;
        float color_y;
        float color_z;
        float depth;
        float normal_x;
        float normal_y;
        float normal_z;
        float densification_weight;
        float densification_error_weighted;
    };

    __device__ __forceinline__ BlendBackwardAccum make_zero_blend_backward_accum() {
        return {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
    }

    // ---------------------------------------------------------------------------
    // blend_backward_cu — warp-level sub-tile culling reverse walk.
    //
    // Source: Yang, Drettakis, Bernstein, "Warp-Level Culling for Efficient Blending
    // in 3D Gaussian Splatting", ACM CGIT 9(4):54, 2026, doi:10.1145/3820019.
    // Training-side port the paper left open; complements BWD-A T_eff clamp
    // (clamp bounds RANGE; warp-cull skips WITHIN range; warp-scoped sync replaces
    // block fences on the reverse walk).
    //
    // Structure (supersedes BWD-C/D diagonal / lockstep prototypes):
    //   • 16×16 tile → 8×4 sub-tiles, 1 warp / 32 px (pixel-centric)
    //   • reverse walk over [0, T_eff) only (BWD-A clamp kept)
    //   • collaborative batch fetch → block.sync once per batch (NOT per splat)
    //   • ballot + ellipse-exact sub-tile test 32-at-a-time (alphablend_shader.slang
    //     ellipse_box_overlap_test — NOT the paper's AABB-only variant)
    //   • warp-reduce per-splat grads → one atomic per splat per warp
    //
    // Numerical policy: FP reduction order may change → grads within 1e-6 of the
    // uncull reference (NOT bit-exact). WARP_CULL_MODE: 0=on, 1=off (ref), 2=wrong empty.
    // WARP_BWD_WALK_BEGIN / WARP_BWD_WALK_END mark the reverse-walk body: zero
    // The sync-count assertion depends on the block.sync and __syncthreads calls below.
    // ---------------------------------------------------------------------------
    template <DensificationType DENSIFICATION_TYPE, bool NORMAL_CHANNEL, int WARP_CULL_MODE = 0>
    __global__ void __launch_bounds__(config::block_size_blend_backward, 8) blend_backward_cu(
        const uint2* __restrict__ tile_instance_ranges,
        const uint* __restrict__ instance_primitive_indices,
        const PackedMeanBBox* __restrict__ primitive_mean2d,
        const float4* __restrict__ primitive_conic_opacity,
        const float4* __restrict__ primitive_color,
        const float* __restrict__ primitive_depths,
        const float3* __restrict__ primitive_normals,
        const float* __restrict__ grad_image,
        const float* __restrict__ grad_alpha_map,
        const float* __restrict__ grad_depth_map,
        const float* __restrict__ grad_normal_map,
        const float* __restrict__ image,
        const float* __restrict__ alpha_map,
        const uint* __restrict__ tile_n_contributions,
        const float* __restrict__ tile_final_transmittance,
        float2* __restrict__ grad_mean2d,
        float3* __restrict__ grad_conic,
        float* __restrict__ grad_depth,
        float3* __restrict__ grad_normal,
        float* __restrict__ grad_compensated_opacity,
        float3* __restrict__ grad_color,
        float* __restrict__ densification_info,
        const float* __restrict__ densification_error_map,
        FastGSForwardStatus* __restrict__ status,
        const uint n_instances,
        const uint n_primitives,
        const uint width,
        const uint height,
        const uint grid_width,
        const int blend_batch_size_runtime) {
        (void)image;
        (void)alpha_map;
        auto block = cg::this_thread_block();
        const uint tile_idx = block.group_index().x;
        const uint thread_rank = block.thread_rank();
        const uint2 tile_instance_range = tile_instance_ranges[tile_idx];
        if (tile_instance_range.x > tile_instance_range.y || tile_instance_range.y > n_instances) {
            if (thread_rank == 0) {
                report_fastgs_status(
                    status,
                    kFastGSForwardStatusTileInstanceRangeOutOfRange,
                    tile_idx,
                    tile_idx,
                    (static_cast<std::uint64_t>(tile_instance_range.x) << 32) | tile_instance_range.y,
                    make_uint4(tile_instance_range.x, tile_instance_range.y, n_instances, 0),
                    n_instances,
                    tile_instance_range.y);
            }
            return;
        }
        const int tile_n_primitives = static_cast<int>(tile_instance_range.y - tile_instance_range.x);
        if (tile_n_primitives <= 0)
            return;

        static_assert(config::tile_width == 16 && config::tile_height == 16,
                      "warp sub-tile layout assumes 16×16 tiles");
        static_assert(config::warp_subtile_width == 8 && config::warp_subtile_height == 4,
                      "warp sub-tile is 8×4");
        static_assert(config::block_size_blend_backward == 128,
                      "blend_backward is 128 threads (4 warps × 2 pixels)");
        static_assert(config::block_size_blend_backward % 32 == 0,
                      "blend_backward block size must be a multiple of warp size");
        // 8 sub-tiles; 4 warps each own two (base and base+4).
        static_assert(config::block_size_blend / config::block_size_blend_backward == 2,
                      "each thread owns 2 pixels");

        const uint lane_id = thread_rank & 31u;
        const uint warp_id = thread_rank >> 5; // 0..3
        const uint local_x = lane_id & 7u;
        const uint local_y = lane_id >> 3;
        // Two sub-tiles per warp: warp w → subtiles w and w+4 (same 2×2 XY packing as forward).
        const uint st0 = warp_id;     // 0..3
        const uint st1 = warp_id + 4; // 4..7
        const uint st0_ox = (st0 & 1u) * static_cast<uint>(config::warp_subtile_width);
        const uint st0_oy = (st0 >> 1) * static_cast<uint>(config::warp_subtile_height);
        const uint st1_ox = (st1 & 1u) * static_cast<uint>(config::warp_subtile_width);
        const uint st1_oy = (st1 >> 1) * static_cast<uint>(config::warp_subtile_height);
        const uint tlx0 = st0_ox + local_x;
        const uint tly0 = st0_oy + local_y;
        const uint tlx1 = st1_ox + local_x;
        const uint tly1 = st1_oy + local_y;
        const uint pixel_rank0 = tly0 * static_cast<uint>(config::tile_width) + tlx0;
        const uint pixel_rank1 = tly1 * static_cast<uint>(config::tile_width) + tlx1;

        const uint n_pixels = width * height;
        const uint2 tile_coords = {tile_idx % grid_width, tile_idx / grid_width};
        const uint2 start_pixel_coords = {
            tile_coords.x * static_cast<uint>(config::tile_width),
            tile_coords.y * static_cast<uint>(config::tile_height)};
        const uint2 pix0 = {start_pixel_coords.x + tlx0, start_pixel_coords.y + tly0};
        const uint2 pix1 = {start_pixel_coords.x + tlx1, start_pixel_coords.y + tly1};
        const bool inside0 = pix0.x < width && pix0.y < height;
        const bool inside1 = pix1.x < width && pix1.y < height;
        const float2 pixel0 = make_float2(__uint2float_rn(pix0.x), __uint2float_rn(pix0.y)) + 0.5f;
        const float2 pixel1 = make_float2(__uint2float_rn(pix1.x), __uint2float_rn(pix1.y)) + 0.5f;
        const uint pixel_idx0 = inside0 ? width * pix0.y + pix0.x : 0u;
        const uint pixel_idx1 = inside1 ? width * pix1.y + pix1.x : 0u;

        const float sub_w = static_cast<float>(config::warp_subtile_width);
        const float sub_h = static_cast<float>(config::warp_subtile_height);
        const float sub0_x0 = static_cast<float>(start_pixel_coords.x + st0_ox);
        const float sub0_y0 = static_cast<float>(start_pixel_coords.y + st0_oy);
        const float sub1_x0 = static_cast<float>(start_pixel_coords.x + st1_ox);
        const float sub1_y0 = static_cast<float>(start_pixel_coords.y + st1_oy);
        const uint sub0_ix0 = start_pixel_coords.x + st0_ox;
        const uint sub0_iy0 = start_pixel_coords.y + st0_oy;
        const uint sub0_ix1 = sub0_ix0 + static_cast<uint>(config::warp_subtile_width);
        const uint sub0_iy1 = sub0_iy0 + static_cast<uint>(config::warp_subtile_height);
        const uint sub1_ix0 = start_pixel_coords.x + st1_ox;
        const uint sub1_iy0 = start_pixel_coords.y + st1_oy;
        const uint sub1_ix1 = sub1_ix0 + static_cast<uint>(config::warp_subtile_width);
        const uint sub1_iy1 = sub1_iy0 + static_cast<uint>(config::warp_subtile_height);

        // Per-pixel reverse-walk state (two pixels per thread).
        const uint last0 = inside0 ? tile_n_contributions[pixel_idx0] : 0u;
        const uint last1 = inside1 ? tile_n_contributions[pixel_idx1] : 0u;
        float T0 = inside0 ? tile_final_transmittance[tile_idx * config::block_size_blend + pixel_rank0] : 1.0f;
        float T1 = inside1 ? tile_final_transmittance[tile_idx * config::block_size_blend + pixel_rank1] : 1.0f;
        float3 grad_c0 = inside0
                             ? make_float3(grad_image[pixel_idx0],
                                           grad_image[n_pixels + pixel_idx0],
                                           grad_image[2 * n_pixels + pixel_idx0])
                             : make_float3(0.0f);
        float3 grad_c1 = inside1
                             ? make_float3(grad_image[pixel_idx1],
                                           grad_image[n_pixels + pixel_idx1],
                                           grad_image[2 * n_pixels + pixel_idx1])
                             : make_float3(0.0f);
        float grad_T0 = inside0 ? -grad_alpha_map[pixel_idx0] : 0.0f;
        float grad_T1 = inside1 ? -grad_alpha_map[pixel_idx1] : 0.0f;
        const float grad_d0 = (inside0 && grad_depth_map != nullptr) ? grad_depth_map[pixel_idx0] : 0.0f;
        const float grad_d1 = (inside1 && grad_depth_map != nullptr) ? grad_depth_map[pixel_idx1] : 0.0f;
        float3 grad_n0 = make_float3(0.0f);
        float3 grad_n1 = make_float3(0.0f);
        if constexpr (NORMAL_CHANNEL) {
            if (inside0 && grad_normal_map != nullptr) {
                grad_n0 = make_float3(grad_normal_map[pixel_idx0],
                                      grad_normal_map[n_pixels + pixel_idx0],
                                      grad_normal_map[2 * n_pixels + pixel_idx0]);
            }
            if (inside1 && grad_normal_map != nullptr) {
                grad_n1 = make_float3(grad_normal_map[pixel_idx1],
                                      grad_normal_map[n_pixels + pixel_idx1],
                                      grad_normal_map[2 * n_pixels + pixel_idx1]);
            }
        }

        // BWD-A: T_eff = min(tile_n_primitives, max last_contributor) — kept.
        uint local_max_contrib = ::max(last0, last1);
#pragma unroll
        for (int offset = 16; offset > 0; offset /= 2) {
            local_max_contrib = ::max(local_max_contrib,
                                      __shfl_xor_sync(0xffffffffu, local_max_contrib, offset));
        }
        constexpr int kBwdWarps = config::block_size_blend_backward / 32;
        __shared__ uint s_warp_max_contrib[kBwdWarps];
        if (lane_id == 0u) {
            s_warp_max_contrib[warp_id] = local_max_contrib;
        }
        block.sync();
        __shared__ int s_T_eff;
        if (thread_rank == 0) {
            uint max_contrib = 0u;
#pragma unroll
            for (int w = 0; w < kBwdWarps; ++w) {
                max_contrib = ::max(max_contrib, s_warp_max_contrib[w]);
            }
            s_T_eff = min(tile_n_primitives, static_cast<int>(max_contrib));
        }
        block.sync();
        const int T_eff = s_T_eff;
        if (T_eff <= 0) {
            return;
        }

        float pixel_error0 = 1.0f;
        float pixel_error1 = 1.0f;
        if constexpr (DENSIFICATION_TYPE != DensificationType::None) {
            if constexpr (DENSIFICATION_TYPE == DensificationType::MCMC) {
                pixel_error0 = densification_error_map[pixel_idx0];
                pixel_error1 = densification_error_map[pixel_idx1];
            } else if (densification_error_map != nullptr) {
                pixel_error0 = densification_error_map[pixel_idx0];
                pixel_error1 = densification_error_map[pixel_idx1];
            }
        }

        // Fetch batch size: runtime override or config default; shrink for tiny T_eff.
        // Cap at block size so one thread can load one splat (collaborative fetch).
        int batch_size = blend_batch_size_runtime > 0
                             ? blend_batch_size_runtime
                             : config::blend_backward_batch_size;
        batch_size = max(32, min(batch_size, config::block_size_blend_backward));
        batch_size = batch_size & ~31;
        if (T_eff <= 4) {
            batch_size = 32;
        } else if (T_eff <= 16) {
            batch_size = min(batch_size, 64);
        } else if (T_eff <= 36) {
            batch_size = min(batch_size, 96);
        }

        // Shared batch staging (sized to backward block = max collaborative fetch).
        __shared__ uint s_prim_idx[config::block_size_blend_backward];
        __shared__ float2 s_mean2d[config::block_size_blend_backward];
        __shared__ float4 s_conic_opacity[config::block_size_blend_backward];
        __shared__ float4 s_color[config::block_size_blend_backward];
        __shared__ float s_depth[config::block_size_blend_backward];
        __shared__ float3 s_normal[NORMAL_CHANNEL ? config::block_size_blend_backward : 1];
        __shared__ unsigned char s_valid_splat[config::block_size_blend_backward];
        __shared__ ushort4 s_pixel_bbox[config::block_size_blend_backward];
        const bool stage_depth = grad_depth_map != nullptr;

        // Reverse walk: batch_base is reverse-order index into [0, T_eff).
        // tile_primitive_idx = T_eff - reverse_i - 1  (high index first).
        for (int batch_base = 0; batch_base < T_eff; batch_base += batch_size) {
            const int n_batch = min(batch_size, T_eff - batch_base);

            // Collaborative fetch (one block.sync per batch — not per splat).
            if (static_cast<int>(thread_rank) < n_batch) {
                const int reverse_i = batch_base + static_cast<int>(thread_rank);
                const int tile_primitive_idx = T_eff - reverse_i - 1;
                const uint instance_idx =
                    tile_instance_range.x + static_cast<uint>(tile_primitive_idx);
                uint primitive_idx = instance_primitive_indices[instance_idx];
                bool valid = true;
                if (primitive_idx >= n_primitives) {
                    report_fastgs_status(
                        status,
                        kFastGSForwardStatusPrimitiveIndexOutOfRange,
                        instance_idx,
                        tile_idx,
                        primitive_idx,
                        make_uint4(
                            tile_instance_range.x,
                            tile_instance_range.y,
                            static_cast<uint>(tile_primitive_idx),
                            n_instances),
                        n_primitives,
                        primitive_idx);
                    valid = false;
                    primitive_idx = 0;
                }
                s_valid_splat[thread_rank] = valid ? 1u : 0u;
                s_prim_idx[thread_rank] = primitive_idx;
                if (valid) {
                    const PackedMeanBBox geom = primitive_mean2d[primitive_idx];
                    s_mean2d[thread_rank] = geom.mean2d;
                    s_pixel_bbox[thread_rank] = geom.pixel_bbox;
                    s_conic_opacity[thread_rank] = primitive_conic_opacity[primitive_idx];
                    const float3 color_unclamped = make_float3(primitive_color[primitive_idx]);
                    const float3 color_clamped =
                        fminf(fmaxf(color_unclamped, 0.0f), config::max_blend_color);
                    const unsigned factor_bits =
                        (color_unclamped.x <= config::max_blend_color ? 1u : 0u) |
                        (color_unclamped.y <= config::max_blend_color ? 2u : 0u) |
                        (color_unclamped.z <= config::max_blend_color ? 4u : 0u);
                    s_color[thread_rank] = make_float4(
                        color_clamped.x, color_clamped.y, color_clamped.z, __uint_as_float(factor_bits));
                    if (stage_depth)
                        s_depth[thread_rank] = primitive_depths[primitive_idx];
                    if constexpr (NORMAL_CHANNEL) {
                        s_normal[thread_rank] = primitive_normals[primitive_idx];
                    }
                } else {
                    s_mean2d[thread_rank] = make_float2(0.0f, 0.0f);
                    s_pixel_bbox[thread_rank] = make_ushort4(0, 0, 0, 0);
                    s_conic_opacity[thread_rank] = make_float4(0.0f);
                    s_color[thread_rank] = make_float4(0.0f);
                    if (stage_depth)
                        s_depth[thread_rank] = 0.0f;
                    if constexpr (NORMAL_CHANNEL) {
                        s_normal[thread_rank] = make_float3(0.0f);
                    }
                }
            }
            block.sync();

            // WARP_BWD_WALK_BEGIN — no block.sync / __syncthreads inside this region.
            for (int j_base = 0; j_base < n_batch; j_base += 32) {
                const int j_test = j_base + static_cast<int>(lane_id);
                bool hit0 = false, hit1 = false;
                if (j_test < n_batch && s_valid_splat[j_test]) {
                    const ushort4 bb = s_pixel_bbox[j_test];
                    const bool aabb0 = (static_cast<uint>(bb.x) < sub0_ix1) &&
                                       (static_cast<uint>(bb.y) > sub0_ix0) &&
                                       (static_cast<uint>(bb.z) < sub0_iy1) &&
                                       (static_cast<uint>(bb.w) > sub0_iy0);
                    const bool aabb1 = (static_cast<uint>(bb.x) < sub1_ix1) &&
                                       (static_cast<uint>(bb.y) > sub1_ix0) &&
                                       (static_cast<uint>(bb.z) < sub1_iy1) &&
                                       (static_cast<uint>(bb.w) > sub1_iy0);
                    const float4 co = s_conic_opacity[j_test];
                    const float3 conic_t = make_float3(co);
                    if (aabb0) {
                        hit0 = kernels::splat_overlaps_subtile_ellipse(
                            s_mean2d[j_test], conic_t, co.w, sub0_x0, sub0_y0, sub_w, sub_h);
                    }
                    if (aabb1) {
                        hit1 = kernels::splat_overlaps_subtile_ellipse(
                            s_mean2d[j_test], conic_t, co.w, sub1_x0, sub1_y0, sub_w, sub_h);
                    }
                }
                unsigned mask0 = __ballot_sync(0xffffffffu, hit0);
                unsigned mask1 = __ballot_sync(0xffffffffu, hit1);
                if constexpr (WARP_CULL_MODE == 1) {
                    mask0 = mask1 = 0xffffffffu;
                } else if constexpr (WARP_CULL_MODE == 2) {
                    mask0 = mask1 = 0u;
                }

                unsigned live = mask0 | mask1;
                while (live != 0u) {
                    const int k = __ffs(live) - 1;
                    live &= live - 1u;
                    const int j = j_base + k;
                    if (j >= n_batch)
                        break;
                    const bool use0 = ((mask0 >> k) & 1u) != 0u;
                    const bool use1 = ((mask1 >> k) & 1u) != 0u;
                    if constexpr (WARP_CULL_MODE != 0) {
                        if (!s_valid_splat[j])
                            continue;
                    }

                    const int tile_primitive_idx = T_eff - batch_base - j - 1;
                    const uint primitive_idx = s_prim_idx[j];
                    const float2 mean2d = s_mean2d[j];
                    const float4 conic_opacity = s_conic_opacity[j];
                    const float3 conic = make_float3(conic_opacity);
                    const float compensated_opacity = conic_opacity.w;
                    const float4 color_staged = s_color[j];
                    const float3 color = make_float3(color_staged);
                    const unsigned factor_bits = __float_as_uint(color_staged.w);
                    const float3 color_grad_factor = make_float3(
                        (factor_bits & 1u) ? 1.0f : 0.0f,
                        (factor_bits & 2u) ? 1.0f : 0.0f,
                        (factor_bits & 4u) ? 1.0f : 0.0f);
                    const float depth = stage_depth ? s_depth[j] : 0.0f;
                    float3 normal = make_float3(0.0f);
                    if constexpr (NORMAL_CHANNEL) {
                        normal = s_normal[j];
                    }

                    BlendBackwardAccum accum = make_zero_blend_backward_accum();
                    bool has_contribution = false;

                    // Lambda-like: process one pixel's contribution into accum.
                    auto accumulate_pixel = [&](const bool use, const bool inside, const uint last,
                                                const float2 pixel, const float pixel_error,
                                                float& T, float3& grad_c, float& grad_T,
                                                const float grad_d, const float3& grad_n) {
                        if (!use || !inside || static_cast<uint>(tile_primitive_idx) >= last)
                            return;
                        const float2 delta = mean2d - pixel;
                        const float sigma_over_2 =
                            0.5f * (conic.x * delta.x * delta.x + conic.z * delta.y * delta.y) +
                            conic.y * delta.x * delta.y;
                        if (!(sigma_over_2 >= 0.0f))
                            return;
                        const float gaussian = __expf(-sigma_over_2);
                        const float unclamped_alpha = compensated_opacity * gaussian;
                        const float alpha = fminf(unclamped_alpha, config::max_fragment_alpha);
                        if (!(alpha >= config::min_alpha_threshold))
                            return;
                        has_contribution = true;
                        const bool alpha_saturated = unclamped_alpha >= config::max_fragment_alpha;
                        const float one_minus_alpha = 1.0f - alpha;
                        const float one_minus_alpha_safe = fmaxf(one_minus_alpha, 1e-4f);
                        const float transmittance_after = T;
                        const float transmittance_before = transmittance_after / one_minus_alpha_safe;
                        const float blending_weight = transmittance_before * alpha;
                        float normal_dot_grad = 0.0f;
                        if constexpr (NORMAL_CHANNEL) {
                            normal_dot_grad = dot(normal, grad_n);
                        }
                        const float grad_transmittance_after = grad_T;

                        if constexpr (DENSIFICATION_TYPE == DensificationType::MCMC) {
                            accum.densification_weight += blending_weight;
                            accum.densification_error_weighted += blending_weight * pixel_error;
                        }

                        const float3 dL_dcolor = blending_weight * grad_c * color_grad_factor;
                        accum.color_x += dL_dcolor.x;
                        accum.color_y += dL_dcolor.y;
                        accum.color_z += dL_dcolor.z;

                        const float dL_dalpha =
                            dot(transmittance_before * color, grad_c) -
                            grad_transmittance_after * transmittance_before +
                            transmittance_before * depth * grad_d +
                            transmittance_before * normal_dot_grad;
                        accum.compensated_opacity += alpha_saturated ? 0.0f : gaussian * dL_dalpha;
                        accum.depth += blending_weight * grad_d;
                        if constexpr (NORMAL_CHANNEL) {
                            accum.normal_x += blending_weight * grad_n.x;
                            accum.normal_y += blending_weight * grad_n.y;
                            accum.normal_z += blending_weight * grad_n.z;
                        }

                        const float gaussian_grad_helper = alpha_saturated ? 0.0f : -alpha * dL_dalpha;
                        accum.conic_x += 0.5f * gaussian_grad_helper * delta.x * delta.x;
                        accum.conic_y += 0.5f * gaussian_grad_helper * delta.x * delta.y;
                        accum.conic_z += 0.5f * gaussian_grad_helper * delta.y * delta.y;
                        const float2 dL_dmean2d = gaussian_grad_helper * make_float2(
                                                                             conic.x * delta.x + conic.y * delta.y,
                                                                             conic.y * delta.x + conic.z * delta.y);
                        accum.mean_x += dL_dmean2d.x;
                        accum.mean_y += dL_dmean2d.y;

                        if constexpr (DENSIFICATION_TYPE == DensificationType::MRNF) {
                            accum.densification_weight += blending_weight;
                            accum.densification_error_weighted += blending_weight * pixel_error;
                        }

                        T = transmittance_before;
                        grad_T = dot(grad_c, alpha * color) +
                                 alpha * depth * grad_d +
                                 alpha * normal_dot_grad +
                                 grad_transmittance_after * one_minus_alpha;
                    };

                    accumulate_pixel(use0, inside0, last0, pixel0, pixel_error0, T0, grad_c0, grad_T0, grad_d0, grad_n0);
                    accumulate_pixel(use1, inside1, last1, pixel1, pixel_error1, T1, grad_c1, grad_T1, grad_d1, grad_n1);

                    // Warp-reduce → one global atomic per splat per warp.
                    const unsigned contrib_mask = __ballot_sync(0xffffffffu, has_contribution);
                    if (contrib_mask != 0u && primitive_idx < n_primitives) {
                        using lfs::core::warp_ops::warp_reduce_sum;
                        const unsigned n_contrib = __popc(contrib_mask);
                        const int src_lane = __ffs(contrib_mask) - 1;
                        auto reduce_field = [&](float v) -> float {
                            if (n_contrib == 1u)
                                return __shfl_sync(0xffffffffu, v, src_lane);
                            return warp_reduce_sum(v);
                        };
                        const float mean_x = reduce_field(accum.mean_x);
                        const float mean_y = reduce_field(accum.mean_y);
                        const float conic_x = reduce_field(accum.conic_x);
                        const float conic_y = reduce_field(accum.conic_y);
                        const float conic_z = reduce_field(accum.conic_z);
                        const float depth_g = reduce_field(accum.depth);
                        const float opac_g = reduce_field(accum.compensated_opacity);
                        const float color_x = reduce_field(accum.color_x);
                        const float color_y = reduce_field(accum.color_y);
                        const float color_z = reduce_field(accum.color_z);
                        float normal_x = 0.0f, normal_y = 0.0f, normal_z = 0.0f;
                        if constexpr (NORMAL_CHANNEL) {
                            normal_x = reduce_field(accum.normal_x);
                            normal_y = reduce_field(accum.normal_y);
                            normal_z = reduce_field(accum.normal_z);
                        }
                        float dens_w = 0.0f, dens_e = 0.0f;
                        if constexpr (DENSIFICATION_TYPE != DensificationType::None) {
                            dens_w = reduce_field(accum.densification_weight);
                            dens_e = reduce_field(accum.densification_error_weighted);
                        }
                        if (lane_id == 0u) {
                            atomicAdd(&grad_mean2d[primitive_idx].x, clamp_grad(mean_x));
                            atomicAdd(&grad_mean2d[primitive_idx].y, clamp_grad(mean_y));
                            atomicAdd(&grad_conic[primitive_idx].x, clamp_grad(conic_x));
                            atomicAdd(&grad_conic[primitive_idx].y, clamp_grad(conic_y));
                            atomicAdd(&grad_conic[primitive_idx].z, clamp_grad(conic_z));
                            atomicAdd(&grad_depth[primitive_idx], clamp_grad(depth_g));
                            if constexpr (NORMAL_CHANNEL) {
                                atomicAdd(&grad_normal[primitive_idx].x, clamp_grad(normal_x));
                                atomicAdd(&grad_normal[primitive_idx].y, clamp_grad(normal_y));
                                atomicAdd(&grad_normal[primitive_idx].z, clamp_grad(normal_z));
                            }
                            atomicAdd(&grad_compensated_opacity[primitive_idx], clamp_grad(opac_g));
                            atomicAdd(&grad_color[primitive_idx].x, clamp_grad(color_x));
                            atomicAdd(&grad_color[primitive_idx].y, clamp_grad(color_y));
                            atomicAdd(&grad_color[primitive_idx].z, clamp_grad(color_z));
                            if constexpr (DENSIFICATION_TYPE != DensificationType::None) {
                                atomicAdd(&densification_info[primitive_idx], dens_w);
                                atomicAdd(&densification_info[n_primitives + primitive_idx], dens_e);
                            }
                        }
                    }
                }
            }
            // WARP_BWD_WALK_END
            block.sync(); // end of batch: safe re-use of shared staging
        }
    }

} // namespace fast_lfs::rasterization::kernels::backward

/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include "buffer_utils.h"
#include "gaussian_evaluation.cuh"
#include "helper_math.h"
#include "kernel_utils.cuh"
#include "lfs/core/warp_reduce.cuh"
#include "lfs/training/effective_rank.cuh"
#include "lfs/training/mean_step_scale.cuh"
#include "rasterization_config.h"
#include "utils.h"
#include "warp_gradient_reduce.cuh"
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
    template <bool MIP_FILTER, int ACTIVE_SH_BASES, bool DISABLED_FEATURES = false>
    __global__ void __launch_bounds__(config::block_size_preprocess_backward, 3) preprocess_backward_cu(
        // These four alias fused_adam.{means,scaling,rotation,opacity}.param (Adam writes).
        const float3* means,
        const float3* raw_scales,
        const float4* raw_rotations,
        const float4* __restrict__ sh_coefficients_rest, // compact float4-packed swizzled layout
        const float4* __restrict__ w2c,
        const float3* __restrict__ cam_position,
        const float* raw_opacities,
        const uint* __restrict__ primitive_work_indices,
        const uint* __restrict__ primitive_touched,
        const ProjectedRecord* __restrict__ projected,
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
        float erank_loss_local = 0.0f;
        float dc_loss_local = 0.0f;
        float erank_grads[3] = {0.0f, 0.0f, 0.0f};
        if (in_range) {
            if (fused_adam.scale_reg_loss_out != nullptr &&
                fused_adam.scale_reg_weight > 0.0f &&
                fused_adam.scaling.param != nullptr &&
                fused_adam.scaling.n_elements > 0) {
                const uint scale_base = static_cast<uint>(primitive_idx) * 3u;
                const float inv_n = 1.0f / static_cast<float>(fused_adam.scaling.n_elements);
                if constexpr (DISABLED_FEATURES) {
                    scale_loss_local = fused_adam.scale_reg_weight * inv_n *
                                       (expf(fused_adam.scaling.param[scale_base]) +
                                        expf(fused_adam.scaling.param[scale_base + 1]) +
                                        expf(fused_adam.scaling.param[scale_base + 2]));
                } else if (fused_adam.scale_reg_log) {
                    scale_loss_local = fused_adam.scale_reg_weight * 0.01f * inv_n /
                                       fused_adam.scale_reg_normalizer *
                                       (fmaxf(fused_adam.scaling.param[scale_base], -40.0f) +
                                        fmaxf(fused_adam.scaling.param[scale_base + 1], -40.0f) +
                                        fmaxf(fused_adam.scaling.param[scale_base + 2], -40.0f));
                } else {
                    scale_loss_local = fused_adam.scale_reg_weight * inv_n *
                                       (expf(fused_adam.scaling.param[scale_base]) +
                                        expf(fused_adam.scaling.param[scale_base + 1]) +
                                        expf(fused_adam.scaling.param[scale_base + 2]));
                }
            }
            if constexpr (!DISABLED_FEATURES) {
                if (fused_adam.erank_reg_weight > 0.0f && fused_adam.scaling.n_primitives > 0) {
                    const uint scale_base = static_cast<uint>(primitive_idx) * 3u;
                    const auto penalty = lfs::training::effective_rank_penalty(
                        fused_adam.scaling.param[scale_base], fused_adam.scaling.param[scale_base + 1u],
                        fused_adam.scaling.param[scale_base + 2u]);
                    const float coef = fused_adam.erank_reg_weight /
                                       static_cast<float>(fused_adam.scaling.n_primitives);
                    erank_loss_local = coef * penalty.value;
                    for (int axis = 0; axis < 3; ++axis)
                        erank_grads[axis] = coef * penalty.gradient[axis];
                }
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
        if constexpr (!DISABLED_FEATURES)
            if (fused_adam.erank_reg_loss_out != nullptr) {
                const float block_sum = lfs::core::warp_ops::block_reduce_sum(erank_loss_local);
                if (threadIdx.x == 0 && block_sum != 0.0f)
                    atomicAdd(fused_adam.erank_reg_loss_out, block_sum);
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

        const uint work_idx = in_range ? primitive_work_indices[primitive_idx] : 0u;
        const bool invisible = in_range && work_idx == 0xffffffffu;
        const bool visible = in_range && !invisible;
        const bool contributed = visible && (primitive_touched == nullptr || primitive_touched[work_idx] != 0);
        // Visibility counts and all optimizer rows retain their original scope.
        // Only image-derived SH and geometry work is gated by contribution.
        if constexpr (!DISABLED_FEATURES) {
            if (visible && fused_adam.rendered_count != nullptr)
                fused_adam.rendered_count[primitive_idx] += 1.0f;
        }

        // Gate the accumulated image derivative. Negative colour may recover
        // only when its image derivative points brighter under gradient descent.
        uint colour_reg_mask = 7u;
        if (visible) {
            const float3 colour = make_float3(projected[work_idx].color);
            float3 image_grad = grad_color_helper[work_idx];
            if (colour.x < 0.0f && image_grad.x >= 0.0f)
                image_grad.x = 0.0f;
            if (colour.y < 0.0f && image_grad.y >= 0.0f)
                image_grad.y = 0.0f;
            if (colour.z < 0.0f && image_grad.z >= 0.0f)
                image_grad.z = 0.0f;
            if (visible)
                grad_color_helper[work_idx] = image_grad;
            // A regularizer alone cannot take the below-black recovery exception.
            if constexpr (!DISABLED_FEATURES)
                if (fused_adam.sh0.param != nullptr)
                    colour_reg_mask =
                        (colour.x >= 0.0f || image_grad.x < 0.0f ? 1u : 0u) |
                        (colour.y >= 0.0f || image_grad.y < 0.0f ? 2u : 0u) |
                        (colour.z >= 0.0f || image_grad.z < 0.0f ? 4u : 0u);
        }
        if constexpr (!DISABLED_FEATURES) {
            if (!visible && in_range && fused_adam.sh0.param != nullptr &&
                (fused_adam.dc_reg_weight > 0.0f || fused_adam.sh_rest_reg_weight > 0.0f)) {
                const float3 colour = convert_sh_to_color(
                    reinterpret_cast<const float3*>(fused_adam.sh0.param), sh_coefficients_rest,
                    means[primitive_idx], cam_position[0], primitive_idx, ACTIVE_SH_BASES,
                    sh_layout_slots, shN_value_bounds,
                    shN_value_bounds != nullptr ? shN_value_n_cells : 0u,
                    shN_value_bits == 16u ? 16u : 0u);
                colour_reg_mask = (colour.x >= 0.0f ? 1u : 0u) |
                                  (colour.y >= 0.0f ? 2u : 0u) |
                                  (colour.z >= 0.0f ? 4u : 0u);
            }
        }

        // Compute SH backward gradients before entering the geometry path.
        if (in_range && !contributed) {
            // Reg-only scale/opacity; SH/means/rot get pure momentum decay (zeros).
            const uint scale_base = static_cast<uint>(primitive_idx) * 3u;
            scale_grads[0] = scale_regularization_grad<!DISABLED_FEATURES>(fused_adam, fused_adam.scaling, scale_base) + erank_grads[0];
            scale_grads[1] = scale_regularization_grad<!DISABLED_FEATURES>(fused_adam, fused_adam.scaling, scale_base + 1) + erank_grads[1];
            scale_grads[2] = scale_regularization_grad<!DISABLED_FEATURES>(fused_adam, fused_adam.scaling, scale_base + 2) + erank_grads[2];
            if constexpr (!DISABLED_FEATURES)
                add_flatten_regularization_grads(fused_adam, fused_adam.scaling, scale_base, scale_grads);
            opacity_grads[0] = opacity_extra_grad(fused_adam, fused_adam.opacity, static_cast<uint>(primitive_idx));
        } else if (contributed) {
            const float3 mean3d = means[primitive_idx];
            dL_dmean3d_from_color = convert_sh_to_color_backward_grads<ACTIVE_SH_BASES>(
                sh_coefficients_rest, grad_color_helper,
                mean3d, cam_position[0],
                primitive_idx,
                work_idx,
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

        if constexpr (!DISABLED_FEATURES)
            if (in_range && fused_adam.dc_reg_weight > 0.0f && fused_adam.sh0.param != nullptr) {
                constexpr float kSh0 = 0.28209479177387814f;
                const float limit = 0.5f / kSh0;
                const int stride = fused_adam.sh0.n_attributes > 0 ? fused_adam.sh0.n_attributes : 3;
                const float* coeff = fused_adam.sh0.param + static_cast<uint>(primitive_idx) * static_cast<uint>(stride);
                // The gradient stays per splat; the reported loss is the mean over splats like the other regularizers.
                const float reported_loss_weight = fused_adam.dc_reg_weight / (3.0f * static_cast<float>(n_primitives));
                for (int c = 0; c < 3; ++c) {
                    float delta = 0.0f;
                    if (coeff[c] > limit)
                        delta = coeff[c] - limit;
                    else if (coeff[c] < -limit)
                        delta = coeff[c] + limit;
                    if (colour_reg_mask & (1u << c))
                        sh0_grads[c] += (2.0f * fused_adam.dc_reg_weight / 3.0f) * delta;
                    dc_loss_local += reported_loss_weight * delta * delta;
                }
            }
        if constexpr (!DISABLED_FEATURES)
            if (fused_adam.dc_reg_loss_out != nullptr) {
                const float block_sum = lfs::core::warp_ops::block_reduce_sum(dc_loss_local);
                if (threadIdx.x == 0 && block_sum != 0.0f)
                    atomicAdd(fused_adam.dc_reg_loss_out, block_sum);
            }

        // Apply SH Adam for all threads so joint block-bounds reduction remains valid.
        // Also the single re-encode site for SH value quant.
        adam_step_row(sh0_grads, fused_adam.sh0, primitive_idx, 3, fused_adam.beta1, fused_adam.beta2, fused_adam.eps);
        if constexpr (ACTIVE_SH_BASES > 1) {
            const float3 sh_mean = contributed ? means[primitive_idx] : make_float3(0.0f, 0.0f, 0.0f);
            const float3 sh_cam = contributed ? cam_position[0] : make_float3(0.0f, 0.0f, 0.0f);
            const float3 sh_gc = contributed ? grad_color_helper[work_idx] : make_float3(0.0f, 0.0f, 0.0f);
            apply_shN_grads_packed<ACTIVE_SH_BASES, !DISABLED_FEATURES>(
                fused_adam, primitive_idx, sh_layout_slots,
                sh_mean, sh_cam, sh_gc, contributed, colour_reg_mask);
        }
        // sh0 / streamed shN Adam state is dead from here on (compiler can free it).

        // Re-enter the contribution path for covariance, EWA, and geometry gradients.
        if (contributed) {
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
            const float3 dL_dconic = grad_conic[work_idx];
            float3 dL_dcov2d = determinant_rcp_sq * make_float3(
                                                        2.0f * bc * dL_dconic.y - cc * dL_dconic.x - bb * dL_dconic.z,
                                                        bc * dL_dconic.x - (ac + bb) * dL_dconic.y + ab * dL_dconic.z,
                                                        2.0f * ab * dL_dconic.y - bb * dL_dconic.x - aa * dL_dconic.z);

            const float grad_compensated_opacity = grad_opacity_helper[work_idx];
            float opacity_compensation = 1.0f;
            if constexpr (MIP_FILTER) {
                const float det_raw = raw_a * raw_c - raw_b * raw_b;
                if (det_raw > config::min_cov2d_determinant && determinant > config::min_cov2d_determinant) {
                    opacity_compensation = sqrtf(det_raw * determinant_rcp);
                } else {
                    opacity_compensation = 0.0f;
                }
            }

            const float sigmoid_derivative = lfs::core::sigmoid_derivative_from_logit(raw_opacities[primitive_idx]);
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
            const float2 dL_dmean2d = grad_mean2d[work_idx];
            const float dL_ddepth = grad_depth ? grad_depth[work_idx] : 0.0f;
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
            scale_grads[0] = clamped_scale_grad.x + scale_regularization_grad<!DISABLED_FEATURES>(fused_adam, fused_adam.scaling, scale_base) + erank_grads[0];
            scale_grads[1] = clamped_scale_grad.y + scale_regularization_grad<!DISABLED_FEATURES>(fused_adam, fused_adam.scaling, scale_base + 1) + erank_grads[1];
            scale_grads[2] = clamped_scale_grad.z + scale_regularization_grad<!DISABLED_FEATURES>(fused_adam, fused_adam.scaling, scale_base + 2) + erank_grads[2];
            if constexpr (!DISABLED_FEATURES)
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
                const float3 g_cam = grad_normal[work_idx];
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

        } // contribution geometry

        if (visible && densification_info != nullptr) {
            densification_info[primitive_idx] += 1.0f;
            densification_info[n_primitives + primitive_idx] += length(grad_mean2d[work_idx] * make_float2(0.5f * w, 0.5f * h));
        }

        if constexpr (DISABLED_FEATURES) {
            adam_step_row(mean_grads, fused_adam.means, primitive_idx, 3,
                          fused_adam.beta1, fused_adam.beta2, fused_adam.eps);
        } else {
            // Scale the applied mean step, not the raw gradient.
            const bool far_mean_step = fused_adam.per_splat_mean_step &&
                                       fused_adam.mean_step_far_mask != nullptr &&
                                       primitive_idx < static_cast<uint>(fused_adam.mean_step_far_mask_n) &&
                                       fused_adam.mean_step_far_mask[primitive_idx];
            // single call site: helper runs block-wide reductions; divergent duplicates deadlock
            FusedAdamParam means_p = fused_adam.means;
            if (far_mean_step && fused_adam.scaling.param != nullptr) {
                const uint sb = primitive_idx * 3u;
                if (sb + 2u < static_cast<uint>(fused_adam.scaling.n_elements)) {
                    const float* s = fused_adam.scaling.param + sb;
                    means_p.step_size *= lfs::training::per_splat_mean_step_ratio(
                        s[0], s[1], s[2],
                        fused_adam.mean_step_median_extent);
                }
            }
            adam_step_row(mean_grads, means_p, primitive_idx, 3,
                          fused_adam.beta1, fused_adam.beta2, fused_adam.eps);
        }
        adam_step_row(rotation_grads, fused_adam.rotation, primitive_idx, 4, fused_adam.beta1, fused_adam.beta2, fused_adam.eps);
        adam_step_row(scale_grads, fused_adam.scaling, primitive_idx, 3, fused_adam.beta1, fused_adam.beta2, fused_adam.eps);
        adam_step_row(opacity_grads, fused_adam.opacity, primitive_idx, 1, fused_adam.beta1, fused_adam.beta2, fused_adam.eps);
    }

    __device__ __forceinline__ float background_transmittance_grad(
        const BackgroundAlphaGradient& background,
        const float3 grad_color,
        const uint pixel_idx,
        const uint n_pixels) {
        if (background.grad_alpha_map != nullptr) {
            return -background.grad_alpha_map[pixel_idx];
        }
        float3 bg = make_float3(0.0f);
        if (background.bg_image != nullptr) {
            bg = make_float3(background.bg_image[pixel_idx],
                             background.bg_image[n_pixels + pixel_idx],
                             background.bg_image[2 * n_pixels + pixel_idx]);
        } else if (background.bg_color != nullptr) {
            bg = make_float3(background.bg_color[0], background.bg_color[1], background.bg_color[2]);
        }
        float grad_alpha = -(grad_color.x * bg.x + grad_color.y * bg.y + grad_color.z * bg.z);
        if (background.grad_alpha_extra != nullptr) {
            grad_alpha += background.grad_alpha_extra[pixel_idx];
        }
        return -grad_alpha;
    }

    __device__ __forceinline__ float3 background_color_at(
        const BackgroundAlphaGradient& background,
        const uint pixel_idx,
        const uint n_pixels) {
        if (background.bg_image != nullptr) {
            return make_float3(background.bg_image[pixel_idx],
                               background.bg_image[n_pixels + pixel_idx],
                               background.bg_image[2 * n_pixels + pixel_idx]);
        }
        if (background.bg_color != nullptr) {
            return make_float3(background.bg_color[0], background.bg_color[1], background.bg_color[2]);
        }
        return make_float3(0.0f);
    }

    // ---------------------------------------------------------------------------
    // blend_backward_cu: back-to-front gradient walk.
    //
    // Each warp owns one 8x8 quadrant of the 16x16 tile and walks only the splats
    // the forward pass recorded as contributing to it, staged in its own 32-entry
    // slice, so the walk needs no block barrier. Each thread holds two vertically
    // adjacent pixels of one column. Per splat, the dx-free sums are reduced over
    // the column first; the mean and conic terms, which need dx, are then formed
    // once per column and reduced across columns together with the rest
    // (reduce-scatter: every exchange halves the payload, and the totals end in
    // distinct lanes, so they leave in one atomic instruction). Two consecutive
    // splats share the first exchange: the lower half-warp keeps one, the upper
    // half-warp the other.
    // ---------------------------------------------------------------------------
    template <DensificationType DENSIFICATION_TYPE, bool NORMAL_CHANNEL, bool IMPACT_SCORE, int WARP_CULL_MODE = 0, bool COLOR_VARIANCE = false>
    __global__ void __launch_bounds__(config::block_size_blend_backward, 4) blend_backward_cu(
        const uint2* __restrict__ tile_instance_ranges,
        const uint* __restrict__ instance_primitive_indices,
        const unsigned char* __restrict__ contributor_masks,
        const uint* __restrict__ contributor_positions,
        const uint* __restrict__ primitive_work_indices,
        const ProjectedRecord* __restrict__ projected,
        const float* __restrict__ primitive_depths,
        const float3* __restrict__ primitive_normals,
        const float* __restrict__ grad_image,
        const BackgroundAlphaGradient background_grad,
        const float* __restrict__ grad_depth_map,
        const float* __restrict__ grad_normal_map,
        const float* __restrict__ image,
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
        const float* __restrict__ edge_weight_map,
        float* __restrict__ edge_score_out,
        float* __restrict__ impact_sum_out,
        float* __restrict__ impact_peak_out,
        uint* __restrict__ primitive_touched,
        FastGSForwardStatus* __restrict__ status,
        const uint n_instances,
        const uint n_primitives,
        const uint n_visible,
        const uint width,
        const uint height,
        const uint grid_width,
        const ColorVarianceLoss color_variance = {}) {
        static_assert(config::tile_width == 16 && config::tile_height == 16, "quadrant layout assumes 16x16 tiles");
        static_assert(config::block_size_blend_backward == 128, "four warps, one 8x8 quadrant each");
        static_assert(config::warp_subtile_width == 8 && config::warp_subtile_height == 4,
                      "coverage bits address 8x4 regions");

        const uint tile_idx = blockIdx.x;
        const uint thread_rank = threadIdx.x;
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

        const uint lane = thread_rank & 31u;
        const uint warp = thread_rank >> 5;
        const uint row_pair = lane >> 3;
        const uint tile_x = (warp & 1u) * 8u + (lane & 7u);
        const uint tile_y0 = (warp >> 1) * 8u + 2u * row_pair;
        const uint top_region = (warp & 1u) + 4u * (warp >> 1);
        const unsigned warp_regions = (1u << top_region) | (1u << (top_region + 2u));

        const uint n_pixels = width * height;
        const uint pixel_x = (tile_idx % grid_width) * static_cast<uint>(config::tile_width) + tile_x;
        const uint pixel_y0 = (tile_idx / grid_width) * static_cast<uint>(config::tile_height) + tile_y0;
        const float center_x = __uint2float_rn(pixel_x) + 0.5f;
        const float center_y0 = __uint2float_rn(pixel_y0) + 0.5f;

        // Reverse-walk state of the thread's two pixels (rows y0 and y0 + 1).
        bool inside[2];
        uint pixel_idx[2], last[2];
        float T[2], grad_T[2], grad_d[2], pixel_error[2], edge_weight[2], variance_scale[2];
        float3 grad_c[2], grad_n[2], behind[2], target[2];
#pragma unroll
        for (int k = 0; k < 2; ++k) {
            const uint y = pixel_y0 + k;
            inside[k] = pixel_x < width && y < height;
            pixel_idx[k] = inside[k] ? width * y + pixel_x : 0u;
            const uint pixel_rank = (tile_y0 + k) * static_cast<uint>(config::tile_width) + tile_x;
            last[k] = inside[k] ? tile_n_contributions[pixel_idx[k]] : 0u;
            T[k] = inside[k] ? tile_final_transmittance[tile_idx * config::block_size_blend + pixel_rank] : 1.0f;
            grad_c[k] = inside[k] ? make_float3(grad_image[pixel_idx[k]],
                                                grad_image[n_pixels + pixel_idx[k]],
                                                grad_image[2 * n_pixels + pixel_idx[k]])
                                  : make_float3(0.0f);
            grad_T[k] = inside[k] ? background_transmittance_grad(background_grad, grad_c[k], pixel_idx[k], n_pixels) : 0.0f;
            grad_d[k] = (inside[k] && grad_depth_map != nullptr) ? grad_depth_map[pixel_idx[k]] : 0.0f;
            grad_n[k] = make_float3(0.0f);
            if constexpr (NORMAL_CHANNEL) {
                if (inside[k] && grad_normal_map != nullptr) {
                    grad_n[k] = make_float3(grad_normal_map[pixel_idx[k]],
                                            grad_normal_map[n_pixels + pixel_idx[k]],
                                            grad_normal_map[2 * n_pixels + pixel_idx[k]]);
                }
            }
            behind[k] = make_float3(0.0f);
            if constexpr (IMPACT_SCORE)
                behind[k] = inside[k] ? background_color_at(background_grad, pixel_idx[k], n_pixels) : make_float3(0.0f);
            pixel_error[k] = 1.0f;
            if constexpr (DENSIFICATION_TYPE == DensificationType::MCMC) {
                pixel_error[k] = inside[k] ? densification_error_map[pixel_idx[k]] : 1.0f;
            } else if constexpr (DENSIFICATION_TYPE != DensificationType::None) {
                if (densification_error_map != nullptr && inside[k])
                    pixel_error[k] = densification_error_map[pixel_idx[k]];
            }
            edge_weight[k] = (edge_weight_map != nullptr && inside[k]) ? edge_weight_map[pixel_idx[k]] : 0.0f;
            target[k] = make_float3(0.0f);
            variance_scale[k] = 0.0f;
        }
        float variance_loss_local = 0.0f;
        if constexpr (COLOR_VARIANCE) {
            // The target is the expected splat color C = sum_i P(i) c_i / sum_i P(i): the forward
            // color without the background, divided by the coverage sum_i P(i) = 1 - T_final.
            // At this C, dL/dC = -2 sum_i P(i) (c_i - C) = 0, so holding it fixed below is exact.
            // A covered pixel has coverage of at least min_alpha_threshold; an uncovered one has
            // no fragments, so its target does not matter.
            const float pixel_weight_sum =
                color_variance.pixel_weight != nullptr ? *color_variance.pixel_weight_sum : 0.0f;
#pragma unroll
            for (int k = 0; k < 2; ++k) {
                const float coverage = 1.0f - T[k];
                if (inside[k] && coverage >= 0.5f * config::min_alpha_threshold) {
                    const uint idx = pixel_idx[k];
                    const float3 rendered = make_float3(image[idx], image[n_pixels + idx], image[2 * n_pixels + idx]);
                    target[k] = (rendered - T[k] * background_color_at(background_grad, idx, n_pixels)) / coverage;
                }
                if (inside[k]) {
                    if (color_variance.pixel_weight == nullptr)
                        variance_scale[k] = color_variance.weight / static_cast<float>(n_pixels);
                    else if (pixel_weight_sum > 0.0f)
                        variance_scale[k] = color_variance.weight * color_variance.pixel_weight[pixel_idx[k]] / pixel_weight_sum;
                }
            }
        }

        uint warp_last = ::max(last[0], last[1]);
#pragma unroll
        for (int offset = 16; offset > 0; offset /= 2)
            warp_last = ::max(warp_last, __shfl_xor_sync(0xffffffffu, warp_last, offset));
        const int traversal_count = ::min(tile_n_primitives, static_cast<int>(warp_last));
        if (thread_rank == 0u && status)
            status->expected_count = static_cast<unsigned>(traversal_count);

        // Dx-free sums, reduced over the column first. Field order is also the
        // order of the `kU*` indices below.
        constexpr int kUConicZ = 0;
        constexpr int kUColor = 1;
        constexpr int kUDepth = 4;
        constexpr int kUNormal = 5;
        constexpr int kUDensification = kUNormal + (NORMAL_CHANNEL ? 3 : 0);
        constexpr int kUEdge = kUDensification + (DENSIFICATION_TYPE != DensificationType::None ? 2 : 0);
        constexpr int kUImpact = kUEdge + 1;
        constexpr int kU = kUImpact + (IMPACT_SCORE ? 1 : 0);
        // Output field ids of the accumulation targets.
        constexpr int kFieldMeanX = 0, kFieldMeanY = 1, kFieldConicX = 2, kFieldConicY = 3, kFieldConicZ = 4;
        constexpr int kFieldDepth = 5, kFieldOpacity = 6, kFieldColor = 7, kFieldNormal = 10;
        constexpr int kFieldDensification = kFieldNormal + (NORMAL_CHANNEL ? 3 : 0);
        constexpr int kFieldEdge = kFieldDensification + (DENSIFICATION_TYPE != DensificationType::None ? 2 : 0);
        constexpr int kFieldImpact = kFieldEdge + 1;
        auto u_field = [](const int u) {
            if (u == kUConicZ)
                return kFieldConicZ;
            if (u < kUDepth)
                return kFieldColor + u - kUColor;
            if (u == kUDepth)
                return kFieldDepth;
            if (u < kUDensification)
                return kFieldNormal + u - kUNormal;
            if (u < kUEdge)
                return kFieldDensification + u - kUDensification;
            if (u == kUEdge)
                return kFieldEdge;
            return kFieldImpact;
        };
        auto store_field = [&](const int field, const float value, const uint primitive, const uint work) {
            if (field < 0 || value == 0.0f)
                return;
            const size_t row = static_cast<size_t>(work);
            float* target_ptr = nullptr;
            if (field < kFieldConicX)
                target_ptr = reinterpret_cast<float*>(grad_mean2d) + 2 * row + field;
            else if (field < kFieldDepth)
                target_ptr = reinterpret_cast<float*>(grad_conic) + 3 * row + field - kFieldConicX;
            else if (field == kFieldDepth)
                target_ptr = grad_depth + work;
            else if (field == kFieldOpacity)
                target_ptr = grad_compensated_opacity + work;
            else if (field < kFieldNormal)
                target_ptr = reinterpret_cast<float*>(grad_color) + 3 * row + field - kFieldColor;
            else if (field < kFieldDensification)
                target_ptr = reinterpret_cast<float*>(grad_normal) + 3 * row + field - kFieldNormal;
            else if (field < kFieldEdge)
                target_ptr = densification_info + static_cast<size_t>(field - kFieldDensification) * n_primitives + primitive;
            else if (field == kFieldEdge)
                target_ptr = edge_score_out != nullptr ? edge_score_out + primitive : nullptr;
            else
                target_ptr = impact_sum_out + primitive;
            if (target_ptr != nullptr)
                atomicAdd(target_ptr, field < kFieldDensification ? clamp_grad(value) : value);
        };

        __shared__ uint s_position[config::block_size_blend_backward];
        __shared__ uint s_primitive[config::block_size_blend_backward];
        __shared__ uint s_work[config::block_size_blend_backward];
        __shared__ float2 s_mean2d[config::block_size_blend_backward];
        __shared__ float3 s_conic[config::block_size_blend_backward];
        __shared__ float3 s_conic_base2[config::block_size_blend_backward];
        __shared__ float2 s_opacity[config::block_size_blend_backward];
        __shared__ float4 s_color[config::block_size_blend_backward];
        __shared__ float s_depth[config::block_size_blend_backward];
        __shared__ float3 s_normal[NORMAL_CHANNEL ? config::block_size_blend_backward : 1];
        const bool stage_depth = grad_depth_map != nullptr;
        const uint warp_base = warp * 32u;

        struct SplatSums {
            float h;
            float h_dy;
            float u[kU];
            float peak;
            bool contributed;
        };
        // Walks one staged splat over the thread's two pixels, back to front.
        auto walk_splat = [&](const uint slot) {
            const uint position = s_position[slot];
            const float2 mean2d = s_mean2d[slot];
            const float3 conic_base2 = s_conic_base2[slot];
            const float4 color_staged = s_color[slot];
            const float3 color = make_float3(color_staged);
            const unsigned factor_bits = __float_as_uint(color_staged.w);
            const float3 color_grad_factor = make_float3((factor_bits & 1u) ? 1.0f : 0.0f,
                                                         (factor_bits & 2u) ? 1.0f : 0.0f,
                                                         (factor_bits & 4u) ? 1.0f : 0.0f);
            const float depth = stage_depth ? s_depth[slot] : 0.0f;
            float3 normal = make_float3(0.0f);
            if constexpr (NORMAL_CHANNEL)
                normal = s_normal[slot];
            const AlphaColumn column = AlphaColumn::at(conic_base2, s_opacity[slot].x, mean2d.x - center_x);
            const float dy0 = mean2d.y - center_y0;

            SplatSums sums{};
            float h[2];
#pragma unroll
            for (int k = 0; k < 2; ++k) {
                const bool eligible = inside[k] && position < last[k];
                const float unclamped = eligible ? column.alpha(conic_base2.z, dy0 - static_cast<float>(k)) : 0.0f;
                const bool accepted = eligible && alpha_accepted(unclamped);
                const float alpha = accepted ? clamped_alpha(unclamped) : 0.0f;
                const bool saturated = unclamped >= config::max_fragment_alpha;
                const float T_before = accepted ? T[k] * rcp_approx(1.0f - alpha) : T[k];
                const float w = alpha * T_before;
                const float3 pixel_grad_color = accepted ? grad_c[k] : make_float3(0.0f);
                const float pixel_grad_depth = accepted ? grad_d[k] : 0.0f;
                const float normal_dot_grad = NORMAL_CHANNEL && accepted ? dot(normal, grad_n[k]) : 0.0f;
                float3 dL_dcolor = w * pixel_grad_color * color_grad_factor;
                float variance_cost = 0.0f;
                if constexpr (COLOR_VARIANCE) {
                    const float3 residual = accepted ? color - target[k] : make_float3(0.0f);
                    variance_cost = variance_scale[k] * dot(residual, residual);
                    // Differentiate the rendered [0, max_blend_color] colors.
                    // Keep the photometric below-black recovery heuristic separate.
                    const float3 positive = make_float3(color.x > 0.0f ? 1.0f : 0.0f,
                                                        color.y > 0.0f ? 1.0f : 0.0f,
                                                        color.z > 0.0f ? 1.0f : 0.0f);
                    dL_dcolor += (2.0f * variance_scale[k] * w) * residual * positive * color_grad_factor;
                    variance_loss_local += w * variance_cost;
                }
                const float grad_T_after = accepted ? grad_T[k] : 0.0f;
                const float dL_dalpha = T_before * (dot(color, pixel_grad_color) + depth * pixel_grad_depth +
                                                    normal_dot_grad + variance_cost - grad_T_after);
                h[k] = accepted && !saturated ? -alpha * dL_dalpha : 0.0f;
                sums.u[kUColor] += dL_dcolor.x;
                sums.u[kUColor + 1] += dL_dcolor.y;
                sums.u[kUColor + 2] += dL_dcolor.z;
                sums.u[kUDepth] += w * pixel_grad_depth;
                if constexpr (NORMAL_CHANNEL) {
                    const float3 pixel_grad_normal = accepted ? grad_n[k] : make_float3(0.0f);
                    sums.u[kUNormal] += w * pixel_grad_normal.x;
                    sums.u[kUNormal + 1] += w * pixel_grad_normal.y;
                    sums.u[kUNormal + 2] += w * pixel_grad_normal.z;
                }
                if constexpr (DENSIFICATION_TYPE != DensificationType::None) {
                    sums.u[kUDensification] += w;
                    sums.u[kUDensification + 1] += w * pixel_error[k];
                }
                sums.u[kUEdge] += w * edge_weight[k];
                if constexpr (IMPACT_SCORE) {
                    const float removal_impact = w * (fabsf(color.x - behind[k].x) +
                                                      fabsf(color.y - behind[k].y) +
                                                      fabsf(color.z - behind[k].z));
                    sums.u[kUImpact] += removal_impact;
                    sums.peak = fmaxf(sums.peak, removal_impact);
                    behind[k] = accepted ? alpha * color + (1.0f - alpha) * behind[k] : behind[k];
                }
                const float next_grad_T = dot(pixel_grad_color, alpha * color) + alpha * depth * pixel_grad_depth +
                                          alpha * normal_dot_grad + alpha * variance_cost +
                                          grad_T_after * (1.0f - alpha);
                grad_T[k] = accepted ? next_grad_T : grad_T[k];
                T[k] = T_before;
                sums.contributed = sums.contributed || accepted;
            }
            // Rows are dy0 and dy0 - 1: sum h dy = dy0 H - h1 and
            // sum h dy^2 = dy0 (dy0 H - 2 h1) + h1.
            sums.h = h[0] + h[1];
            sums.h_dy = __fmaf_rn(dy0, sums.h, -h[1]);
            sums.u[kUConicZ] = 0.5f * __fmaf_rn(dy0, sums.h_dy - h[1], h[1]);
            return sums;
        };
        // Mean, conic and opacity terms of one column from its reduced sums.
        auto column_terms = [&](const uint slot, const float h, const float h_dy) {
            const float dx = s_mean2d[slot].x - center_x;
            const float3 conic = s_conic[slot];
            const float h_dx = dx * h;
            struct {
                float mean_x, mean_y, conic_x, conic_y, opacity;
            } terms{conic.x * h_dx + conic.y * h_dy,
                    conic.y * h_dx + conic.z * h_dy,
                    0.5f * dx * h_dx,
                    0.5f * dx * h_dy,
                    -h * s_opacity[slot].y};
            return terms;
        };

        // Lane-constant scatter targets of both reductions. After the three
        // horizontal exchanges a lane holds kOut totals of its array.
        constexpr int kUVertical = (kU + 3) / 4;
        constexpr int kSingle = 2 + kUVertical;
        constexpr int kSingleOut = (kSingle + 7) / 8;
        int single_fields[kSingleOut];
#pragma unroll
        for (int f = 0; f < kSingleOut; ++f) {
            const int index = halving_index<kSingle, 4u, 3>(lane, f);
            int field = -1;
            if (index == 0)
                field = row_pair == 0 ? kFieldMeanX : row_pair == 1 ? kFieldMeanY
                                                  : row_pair == 2   ? kFieldConicX
                                                                    : kFieldConicY;
            else if (index == 1)
                field = row_pair == 0 ? kFieldOpacity : -1;
            else if (index > 1) {
                const int u = halving_index<kU, 16u, 2>(lane, index - 2);
                field = u >= 0 ? u_field(u) : -1;
            }
            single_fields[f] = field;
        }
        constexpr int kUFolded = (kU + 1) / 2;
        constexpr int kPair = 3 + kUFolded;
        constexpr int kPairOut = (kPair + 7) / 8;
        const bool second_row_pair = (lane & 8u) != 0;
        int pair_fields[kPairOut];
#pragma unroll
        for (int f = 0; f < kPairOut; ++f) {
            const int index = halving_index<kPair, 4u, 3>(lane, f);
            int field = -1;
            if (index == 0)
                field = second_row_pair ? kFieldMeanY : kFieldMeanX;
            else if (index == 1)
                field = second_row_pair ? kFieldConicY : kFieldConicX;
            else if (index == 2)
                field = second_row_pair ? -1 : kFieldOpacity;
            else if (index >= 3) {
                const int u = halving_index<kU, 8u, 1>(lane, index - 3);
                field = u >= 0 ? u_field(u) : -1;
            }
            pair_fields[f] = field;
        }

        // WARP_BWD_WALK_BEGIN: per-warp staging and reductions only, no block barrier.
        for (int end = traversal_count; end > 0; end -= 32) {
            const int entry = end - 1 - static_cast<int>(lane);
            const uint slot = warp_base + lane;
            bool staged = false;
            if (entry >= 0) {
                const uint instance_idx = tile_instance_range.x + static_cast<uint>(entry);
                const unsigned coverage = contributor_masks != nullptr ? contributor_masks[instance_idx] : 0xffu;
                if (WARP_CULL_MODE == 1 || (coverage & warp_regions) != 0u) {
                    const uint primitive_idx = instance_primitive_indices[instance_idx];
                    uint work_idx = 0u;
                    bool valid = primitive_idx < n_primitives;
                    if (valid) {
                        work_idx = primitive_work_indices[primitive_idx];
                        valid = work_idx < n_visible;
                    }
                    if (!valid) {
                        report_fastgs_status(
                            status,
                            kFastGSForwardStatusPrimitiveIndexOutOfRange,
                            instance_idx,
                            tile_idx,
                            primitive_idx,
                            make_uint4(tile_instance_range.x, tile_instance_range.y, static_cast<uint>(entry), n_instances),
                            n_primitives,
                            primitive_idx);
                    } else {
                        staged = true;
                        if (primitive_touched != nullptr)
                            primitive_touched[work_idx] = 1u;
                        s_position[slot] = contributor_positions != nullptr ? contributor_positions[instance_idx] : static_cast<uint>(entry);
                        s_primitive[slot] = primitive_idx;
                        s_work[slot] = work_idx;
                        const ProjectedRecord record = projected[work_idx];
                        s_mean2d[slot] = record.geometry.mean2d;
                        const float3 conic = make_float3(record.conic_opacity);
                        s_conic[slot] = conic;
                        s_conic_base2[slot] = base2_conic(conic);
                        const float opacity = record.conic_opacity.w;
                        s_opacity[slot] = make_float2(opacity, opacity > 0.0f ? 1.0f / opacity : 0.0f);
                        const float3 color_unclamped = make_float3(record.color);
                        const float3 color_clamped = fminf(fmaxf(color_unclamped, 0.0f), config::max_blend_color);
                        const unsigned factor_bits = (color_unclamped.x <= config::max_blend_color ? 1u : 0u) |
                                                     (color_unclamped.y <= config::max_blend_color ? 2u : 0u) |
                                                     (color_unclamped.z <= config::max_blend_color ? 4u : 0u);
                        s_color[slot] = make_float4(color_clamped.x, color_clamped.y, color_clamped.z, __uint_as_float(factor_bits));
                        if (stage_depth)
                            s_depth[slot] = primitive_depths[work_idx];
                        if constexpr (NORMAL_CHANNEL)
                            s_normal[slot] = primitive_normals[work_idx];
                    }
                }
            }
            unsigned mask = __ballot_sync(0xffffffffu, staged);
            if constexpr (WARP_CULL_MODE == 2)
                mask = 0u;
            __syncwarp();

            // Two splats at a time: one per half-warp after the first exchange.
            while (false && __popc(mask) >= 2) {
                const uint slot0 = warp_base + (__ffs(mask) - 1);
                mask &= mask - 1u;
                const uint slot1 = warp_base + (__ffs(mask) - 1);
                mask &= mask - 1u;
                const SplatSums first = walk_splat(slot0);
                const SplatSums second = walk_splat(slot1);
                if (!__any_sync(0xffffffffu, first.contributed || second.contributed))
                    continue;
                const bool upper = (lane & 16u) != 0;
                auto fold = [&](const float a, const float b) {
                    return (upper ? b : a) + __shfl_xor_sync(0xffffffffu, upper ? a : b, 16);
                };
                const uint slot = upper ? slot1 : slot0;
                float h = fold(first.h, second.h);
                float h_dy = fold(first.h_dy, second.h_dy);
                float values[kPair];
                float folded[kU];
#pragma unroll
                for (int u = 0; u < kU; ++u)
                    folded[u] = fold(first.u[u], second.u[u]);
                halving_step<kU, 8>(folded, lane);
                h += __shfl_xor_sync(0xffffffffu, h, 8);
                h_dy += __shfl_xor_sync(0xffffffffu, h_dy, 8);
                const auto terms = column_terms(slot, h, h_dy);
                values[0] = second_row_pair ? terms.mean_y : terms.mean_x;
                values[1] = second_row_pair ? terms.conic_y : terms.conic_x;
                values[2] = second_row_pair ? 0.0f : terms.opacity;
#pragma unroll
                for (int u = 0; u < kUFolded; ++u)
                    values[3 + u] = folded[u];
                halving_step<kPair, 4>(values, lane);
                halving_step<(kPair + 1) / 2, 2>(values, lane);
                halving_step<((kPair + 1) / 2 + 1) / 2, 1>(values, lane);
#pragma unroll
                for (int f = 0; f < kPairOut; ++f)
                    store_field(pair_fields[f], values[f], s_primitive[slot], s_work[slot]);
                if constexpr (IMPACT_SCORE) {
                    float peak = fmaxf(upper ? second.peak : first.peak,
                                       __shfl_xor_sync(0xffffffffu, upper ? first.peak : second.peak, 16));
                    peak = reduce_max<8>(peak);
                    if ((lane & 15u) == 0u && peak > 0.0f)
                        atomicMax(reinterpret_cast<int*>(impact_peak_out + s_primitive[slot]), __float_as_int(peak));
                }
            }
            while (mask != 0u) {
                const uint slot = warp_base + (__ffs(mask) - 1);
                mask &= mask - 1u;
                const SplatSums sums = walk_splat(slot);
                if (__any_sync(0xffffffffu, sums.contributed)) {
                    float h = sums.h;
                    float h_dy = sums.h_dy;
                    float vertical[kU];
#pragma unroll
                    for (int u = 0; u < kU; ++u)
                        vertical[u] = sums.u[u];
                    halving_step<kU, 16>(vertical, lane);
                    halving_step<(kU + 1) / 2, 8>(vertical, lane);
#pragma unroll
                    for (int offset = 16; offset >= 8; offset /= 2) {
                        h += __shfl_xor_sync(0xffffffffu, h, offset);
                        h_dy += __shfl_xor_sync(0xffffffffu, h_dy, offset);
                    }
                    const auto terms = column_terms(slot, h, h_dy);
                    float values[kSingle];
                    values[0] = row_pair == 0 ? terms.mean_x : row_pair == 1 ? terms.mean_y
                                                           : row_pair == 2   ? terms.conic_x
                                                                             : terms.conic_y;
                    values[1] = row_pair == 0 ? terms.opacity : 0.0f;
#pragma unroll
                    for (int u = 0; u < kUVertical; ++u)
                        values[2 + u] = vertical[u];
                    halving_step<kSingle, 4>(values, lane);
                    halving_step<(kSingle + 1) / 2, 2>(values, lane);
                    halving_step<((kSingle + 1) / 2 + 1) / 2, 1>(values, lane);
#pragma unroll
                    for (int f = 0; f < kSingleOut; ++f)
                        store_field(single_fields[f], values[f], s_primitive[slot], s_work[slot]);
                    if constexpr (IMPACT_SCORE) {
                        const float peak = reduce_max<16>(sums.peak);
                        if (lane == 0u && peak > 0.0f)
                            atomicMax(reinterpret_cast<int*>(impact_peak_out + s_primitive[slot]), __float_as_int(peak));
                    }
                }
            }
            __syncwarp();
        }
        // WARP_BWD_WALK_END
        if constexpr (COLOR_VARIANCE) {
            if (color_variance.loss_out != nullptr) {
                const float sum = lfs::core::warp_ops::block_reduce_sum(variance_loss_local);
                if (thread_rank == 0)
                    atomicAdd(color_variance.loss_out, sum);
            }
        }
    }

} // namespace fast_lfs::rasterization::kernels::backward

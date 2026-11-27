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
#include "lfs/training/ray_depth_step.cuh"
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
    // CAMERA_GRAD: adds the camera gradient to grad_w2c and grad_cam_position (FastGSFusedExtraGradients).
    template <bool MIP_FILTER, int ACTIVE_SH_BASES, bool DISABLED_FEATURES = false, bool CAMERA_GRAD = false>
    __global__ void __launch_bounds__(config::block_size_preprocess_backward, 3) preprocess_backward_cu(
        // These four alias fused_adam.{means,scaling,rotation,opacity}.param (Adam writes).
        const float3* means,
        const float3* raw_scales,
        const float4* raw_rotations,
        const float4* __restrict__ sh_coefficients_rest, // compact float4-packed swizzled layout
        const float4* __restrict__ w2c,
        const float3* __restrict__ cam_position,
        const float* raw_opacities,
        const std::uint8_t* __restrict__ active_block_flag,
        const uint* __restrict__ primitive_work_indices,
        const std::uint8_t* __restrict__ primitive_touched,
        const ProjectedRecord* __restrict__ projected,
        const float2* __restrict__ grad_mean2d,
        const float3* __restrict__ grad_conic,
        const float* __restrict__ grad_depth,
        const float3* __restrict__ grad_normal,
        float* __restrict__ grad_opacity_helper,
        float3* __restrict__ grad_color_helper,
        float4* __restrict__ grad_w2c,
        float3* __restrict__ grad_cam_position,
        float* __restrict__ densification_weight,
        float* __restrict__ densification_error,
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
        const float dilation_scale,
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

        const bool block_has_visible =
            (active_block_flag == nullptr || active_block_flag[blockIdx.x] != 0u);
        const uint work_idx = (in_range && block_has_visible) ? primitive_work_indices[primitive_idx] : 0xffffffffu;
        const bool invisible = in_range && work_idx == 0xffffffffu;
        const bool visible = in_range && !invisible;
        const bool contributed = visible && (primitive_touched == nullptr || primitive_touched[work_idx] != 0u);
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
                bool need_colour_check = fused_adam.sh_rest_reg_weight > 0.0f;
                if (!need_colour_check && fused_adam.dc_reg_weight > 0.0f) {
                    constexpr float kSh0 = 0.28209479177387814f;
                    const float limit = 0.5f / kSh0;
                    const int stride = fused_adam.sh0.n_attributes > 0 ? fused_adam.sh0.n_attributes : 3;
                    const float* coeff = fused_adam.sh0.param + static_cast<size_t>(primitive_idx) * static_cast<size_t>(stride);
                    need_colour_check = fabsf(coeff[0]) > limit || fabsf(coeff[1]) > limit || fabsf(coeff[2]) > limit;
                }
                if (need_colour_check) {
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
        }

        // Compute SH backward gradients before entering the geometry path.
        bool skip_shape_reg = false;
        if constexpr (!DISABLED_FEATURES)
            skip_shape_reg = fused_adam.reg_rendered_only;
        if (in_range && !contributed && skip_shape_reg) {
            // MRNF: the shape regularizers act on rendered splats only; momentum decay elsewhere.
            opacity_grads[0] = opacity_extra_grad(fused_adam, fused_adam.opacity, static_cast<uint>(primitive_idx));
        } else if (in_range && !contributed) {
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

        // dL/d(w2c rows 0-2) then dL/d(cam_position), summed over the block after the geometry.
        [[maybe_unused]] float camera_grad[15] = {};

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
            const float kernel_size = (MIP_FILTER ? config::dilation_mip_filter : config::dilation) * dilation_scale;
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
                // The forward's factor: it keeps every splat with det >= min_cov2d_determinant.
                // It scales the opacity gradient only and stays detached from the covariance: feeding this
                // determinant ratio into the scale and rotation gradients creates long splats.
                opacity_compensation = sqrtf(fmaxf(raw_a * raw_c - raw_b * raw_b, 0.0f) * determinant_rcp);
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

            // Camera-space normal W n: n is the axis of least variance, turned to face the camera.
            float3 dL_dnormal_cam = make_float3(0.0f);
            float3 normal_world = make_float3(0.0f);
            if (grad_normal != nullptr) {
                dL_dnormal_cam = grad_normal[work_idx];
                const float3 g_world = make_float3(
                    w2c_r1.x * dL_dnormal_cam.x + w2c_r2.x * dL_dnormal_cam.y + w2c_r3.x * dL_dnormal_cam.z,
                    w2c_r1.y * dL_dnormal_cam.x + w2c_r2.y * dL_dnormal_cam.y + w2c_r3.y * dL_dnormal_cam.z,
                    w2c_r1.z * dL_dnormal_cam.x + w2c_r2.z * dL_dnormal_cam.y + w2c_r3.z * dL_dnormal_cam.z);
                const int axis = (variance.x <= variance.y && variance.x <= variance.z) ? 0
                                 : (variance.y <= variance.z)                           ? 1
                                                                                        : 2;
                const float3 axis_dir = axis == 0   ? make_float3(rotation.m11, rotation.m21, rotation.m31)
                                        : axis == 1 ? make_float3(rotation.m12, rotation.m22, rotation.m32)
                                                    : make_float3(rotation.m13, rotation.m23, rotation.m33);
                const float sign = dot(axis_dir, mean3d - cam_position[0]) > 0.0f ? -1.0f : 1.0f;
                normal_world = sign * axis_dir;
                const float3 dL_daxis = sign * g_world;
                if (axis == 0) {
                    dL_drotation.m11 += dL_daxis.x;
                    dL_drotation.m21 += dL_daxis.y;
                    dL_drotation.m31 += dL_daxis.z;
                } else if (axis == 1) {
                    dL_drotation.m12 += dL_daxis.x;
                    dL_drotation.m22 += dL_daxis.y;
                    dL_drotation.m32 += dL_daxis.z;
                } else {
                    dL_drotation.m13 += dL_daxis.x;
                    dL_drotation.m23 += dL_daxis.y;
                    dL_drotation.m33 += dL_daxis.z;
                }
            }

            if constexpr (CAMERA_GRAD) {
                // Through the camera-space centre W x + b, the covariance (J W) Sigma (J W)^T with J held (its
                // dependence on the centre is in dL_dmean3d_cam) and the normal; the position only enters the colour.
                const float3 dL_dW0 = dL_dmean3d_cam.x * mean3d + j11 * dL_djw_r1 + dL_dnormal_cam.x * normal_world;
                const float3 dL_dW1 = dL_dmean3d_cam.y * mean3d + j22 * dL_djw_r2 + dL_dnormal_cam.y * normal_world;
                const float3 dL_dW2 = dL_dmean3d_cam.z * mean3d + j13 * dL_djw_r1 + j23 * dL_djw_r2 +
                                      dL_dnormal_cam.z * normal_world;
                const float values[15] = {
                    dL_dW0.x, dL_dW0.y, dL_dW0.z, dL_dmean3d_cam.x,
                    dL_dW1.x, dL_dW1.y, dL_dW1.z, dL_dmean3d_cam.y,
                    dL_dW2.x, dL_dW2.y, dL_dW2.z, dL_dmean3d_cam.z,
                    -dL_dmean3d_from_color.x, -dL_dmean3d_from_color.y, -dL_dmean3d_from_color.z};
#pragma unroll
                for (int i = 0; i < 15; ++i)
                    camera_grad[i] = values[i];
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

        if constexpr (CAMERA_GRAD) {
            lfs::core::warp_ops::block_reduce_sum_n(camera_grad);
            if (threadIdx.x == 0) {
                float* w2c_out = reinterpret_cast<float*>(grad_w2c);
#pragma unroll
                for (int i = 0; i < 12; ++i)
                    atomicAdd(w2c_out + i, camera_grad[i]);
                atomicAdd(&grad_cam_position->x, camera_grad[12]);
                atomicAdd(&grad_cam_position->y, camera_grad[13]);
                atomicAdd(&grad_cam_position->z, camera_grad[14]);
            }
        }

        if (visible && densification_error != nullptr) {
            if (densification_weight != nullptr)
                densification_weight[primitive_idx] += 1.0f;
            densification_error[primitive_idx] += length(grad_mean2d[work_idx] * make_float2(0.5f * w, 0.5f * h));
        }

        // Ray-depth proposal from the rendered parameters; applied after the geometry rows' steps.
        [[maybe_unused]] lfs::training::ray_depth::Proposal ray_depth;
        [[maybe_unused]] const bool ray_depth_row = !DISABLED_FEATURES && contributed &&
                                                    fused_adam.ray_depth_state != nullptr &&
                                                    fused_adam.means.gradient_out == nullptr &&
                                                    fused_adam.means.param != nullptr &&
                                                    fused_adam.scaling.param != nullptr &&
                                                    fused_adam.rotation.param != nullptr &&
                                                    primitive_idx < static_cast<uint>(fused_adam.ray_depth_n);
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
            float mean_step_ratio = 1.0f;
            if (far_mean_step && fused_adam.scaling.param != nullptr) {
                const uint sb = primitive_idx * 3u;
                if (sb + 2u < static_cast<uint>(fused_adam.scaling.n_elements)) {
                    const float* s = fused_adam.scaling.param + sb;
                    mean_step_ratio = lfs::training::per_splat_mean_step_ratio(
                        s[0], s[1], s[2],
                        fused_adam.mean_step_median_extent);
                    means_p.step_size *= mean_step_ratio;
                }
            }
            if (ray_depth_row) {
                const uint b3 = primitive_idx * 3u;
                const float lr_scale = lfs::training::ray_depth::row_lr_scale(
                    primitive_idx, fused_adam.means.frozen_mask, fused_adam.means.frozen_mask_size,
                    fused_adam.means.frozen_lr_scale, fused_adam.means.crop_damping_mask,
                    fused_adam.means.crop_damping_mask_size, fused_adam.means.cropbox_lr_scale);
                const float3 camera = cam_position[0];
                const float camera_xyz[3] = {camera.x, camera.y, camera.z};
                ray_depth = lfs::training::ray_depth::propose(
                    fused_adam.ray_depth_state + primitive_idx * lfs::training::ray_depth::kStateStride,
                    fused_adam.means.param + b3, fused_adam.scaling.param + b3,
                    fused_adam.rotation.param + primitive_idx * 4u, camera_xyz, mean_grads,
                    fused_adam.ray_depth_lr * lr_scale,
                    lfs::training::ray_depth::kSidewaysSteps * fused_adam.ray_depth_mean_lr * lr_scale *
                        mean_step_ratio);
            }
            adam_step_row(mean_grads, means_p, primitive_idx, 3,
                          fused_adam.beta1, fused_adam.beta2, fused_adam.eps);
        }
        adam_step_row(rotation_grads, fused_adam.rotation, primitive_idx, 4, fused_adam.beta1, fused_adam.beta2, fused_adam.eps);
        adam_step_row(scale_grads, fused_adam.scaling, primitive_idx, 3, fused_adam.beta1, fused_adam.beta2, fused_adam.eps);
        adam_step_row(opacity_grads, fused_adam.opacity, primitive_idx, 1, fused_adam.beta1, fused_adam.beta2, fused_adam.eps);
        if constexpr (!DISABLED_FEATURES) {
            if (ray_depth_row) {
                const float3 camera = cam_position[0];
                const float camera_xyz[3] = {camera.x, camera.y, camera.z};
                lfs::training::ray_depth::commit(
                    fused_adam.ray_depth_state + primitive_idx * lfs::training::ray_depth::kStateStride,
                    fused_adam.means.param + primitive_idx * 3u, fused_adam.scaling.param + primitive_idx * 3u,
                    camera_xyz, ray_depth.dt, ray_depth.bperp2 >= 0.0f, fused_adam.ray_depth_far_limit);
            }
            if (in_range && fused_adam.lazy_last_step != nullptr && fused_adam.step_seed > 0u &&
                primitive_idx < static_cast<uint>(fused_adam.lazy_last_step_n)) {
                fused_adam.lazy_last_step[primitive_idx] = encode_lazy_step(fused_adam.step_seed);
            }
        }
    }

    // Tight initial expand-only block bounds for sparse single-pass backward preprocessing.
    // Blocks expand tightly on actual overflow and periodic re-tightening shrinks 8-bit SH
    // blocks back down after transient spikes. Zero-exact hi/lo ratios map center codes
    // (32768 for 16-bit, 128 for 8-bit) bit-exactly to 0.0f so decayed moments reach true zero.
    inline constexpr float kSparseMinU16 = 8.0f;
    inline constexpr float kSparseMinS16 = 44.0f;
    inline constexpr float kSparseMinU8 = 8.0f;
    inline constexpr float kSparseMinS8 = 36.0f;
    inline constexpr float kSparseMinVHalf = 0.25f;
    inline constexpr float kQ16ZeroExactHiRatio = 32767.0f / 32768.0f;
    inline constexpr float kQ16ZeroExactHiInvRatio = 32768.0f / 32767.0f;
    inline constexpr float kQ8ZeroExactHiRatio = 127.0f / 128.0f;
    inline constexpr float kQ8ZeroExactHiInvRatio = 128.0f / 127.0f;

    __global__ void prepare_sparse_bounds_cu(
        FusedAdamSettings fused_adam,
        SparseReboundState rebound,
        const uint sh_layout_slots) {
        const uint bidx = blockIdx.x * blockDim.x + threadIdx.x;
        if (bidx >= rebound.n_blocks)
            return;

        auto check_joint = [&](const FusedAdamParam& p, const int slot, const int expected_bits,
                               const float min_u, const float min_s, const float hi_ratio) {
            if (!p.enabled || p.joint_bits != expected_bits ||
                p.joint_packed == nullptr || p.joint_bounds == nullptr)
                return;
            if (p.n_primitives > 0 && bidx * 256u >= static_cast<uint>(p.n_primitives))
                return;
            float4* const bptr = reinterpret_cast<float4*>(p.joint_bounds) + bidx;
            const float4 mm = *bptr;
            if (mm.x == 0.0f && mm.y == 0.0f && mm.z == 0.0f && mm.w == 0.0f) {
                // Zero-initialized block: every cell has s_q==0, which still decodes
                // to exact (m,v)=(0,0) under s_min==0.0f without touching packed bytes.
                *bptr = make_float4(-min_u, min_u * hi_ratio, 0.0f, min_s);
                return;
            }
            // Only pre-rebound blocks whose range excludes zero (u=0 or s=0) or has
            // degenerate spans; healthy tight bounds stay untouched until overflow.
            if (mm.x > 0.0f || mm.y < 0.0f || mm.z > 0.0f ||
                mm.y <= mm.x + 1e-6f || mm.w <= mm.z + 1e-6f) {
                const uint pos = atomicAdd(rebound.counts_pre + slot, 1u);
                if (pos < rebound.n_blocks) {
                    rebound.rebound_list[static_cast<size_t>(slot) * rebound.n_blocks + pos] = bidx;
                }
            }
        };

        check_joint(fused_adam.means, 0, 16, kSparseMinU16, kSparseMinS16, kQ16ZeroExactHiRatio);
        check_joint(fused_adam.sh0, 1, 16, kSparseMinU16, kSparseMinS16, kQ16ZeroExactHiRatio);
        check_joint(fused_adam.scaling, 2, 16, kSparseMinU16, kSparseMinS16, kQ16ZeroExactHiRatio);
        check_joint(fused_adam.rotation, 3, 16, kSparseMinU16, kSparseMinS16, kQ16ZeroExactHiRatio);
        check_joint(fused_adam.opacity, 4, 16, kSparseMinU16, kSparseMinS16, kQ16ZeroExactHiRatio);
        if (sh_layout_slots > 0u) {
            check_joint(fused_adam.shN, 5, 8, kSparseMinU8, kSparseMinS8, kQ8ZeroExactHiRatio);
            const FusedAdamParam& shN = fused_adam.shN;
            if (shN.enabled && shN.sh_value_bits == 16 &&
                shN.sh_value_bounds != nullptr && shN.sh_value_n_cells > 0) {
                if (shN.n_primitives <= 0 || bidx * 256u < static_cast<uint>(shN.n_primitives)) {
                    const float2 vmm = reinterpret_cast<const float2*>(shN.sh_value_bounds)[bidx];
                    if (vmm.y <= vmm.x + 1e-6f) {
                        const uint pos = atomicAdd(rebound.counts_pre + 6, 1u);
                        if (pos < rebound.n_blocks) {
                            rebound.rebound_list[static_cast<size_t>(6) * rebound.n_blocks + pos] = bidx;
                        }
                    }
                }
            }
        }
    }

    // Periodic re-tightening of 8-bit shN joint-moment bounds so transient spikes
    // (early steps or post-refinement children) never leave blocks permanently coarse.
    __global__ void tighten_shN_joint_bounds_cu(
        const FusedAdamParam p,
        const uint n_primitives,
        const uint sh_layout_slots) {
        using C8 = lfs::training::joint_adam::DeviceCodec<8>;
        const uint bidx = blockIdx.x;
        const uint n_prims_param = p.n_primitives > 0 ? static_cast<uint>(p.n_primitives) : n_primitives;
        if (!p.enabled || p.joint_bits != 8 || p.joint_packed == nullptr || p.joint_bounds == nullptr ||
            sh_layout_slots == 0u || bidx * 256u >= n_prims_param)
            return;

        const float4 old_mm = reinterpret_cast<const float4*>(p.joint_bounds)[bidx];
        if (old_mm.x == 0.0f && old_mm.y == 0.0f && old_mm.z == 0.0f && old_mm.w == 0.0f)
            return;

        const uint prim = bidx * 256u + threadIdx.x;
        float local_max_u = 0.0f;
        float local_max_s = 0.0f;
        if (prim < n_prims_param) {
            const float u_span = old_mm.y - old_mm.x;
            const float s_span = old_mm.w - old_mm.z;
            const uint slot0 = shAt(prim, 0u, sh_layout_slots);
            const uint2* const packed_base = reinterpret_cast<const uint2*>(p.joint_packed) + slot0;
            for (uint k = 0; k < sh_layout_slots; ++k) {
                const uint2 words = packed_base[k * config::sh_reorder_size];
#pragma unroll
                for (uint c = 0; c < 4u; ++c) {
                    const uint32_t word = (c < 2u) ? words.x : words.y;
                    const uint32_t shift = (c & 1u) ? 16u : 0u;
                    const uint32_t pair = (word >> shift) & 0xffffu;
                    const uint32_t s_q = pair >> 8u;
                    if (s_q != 0u || old_mm.z > 0.0f) {
                        const float u_val = old_mm.x + u_span * (static_cast<float>(pair & 0xffu) * C8::kInvQMax);
                        const float s_val = old_mm.z + s_span * (static_cast<float>(s_q) * C8::kInvQMax);
                        local_max_u = fmaxf(local_max_u, fabsf(u_val));
                        local_max_s = fmaxf(local_max_s, s_val);
                    }
                }
            }
        }

        const float2 red = lfs::core::warp_ops::block_reduce_min2(make_float2(-local_max_u, -local_max_s));
        __shared__ float4 sm_new_mm;
        __shared__ int sm_do_reencode;
        if (threadIdx.x == 0) {
            const float max_u = -red.x;
            const float max_s = -red.y;
            const float target_u = fmaxf(max_u * kQ8ZeroExactHiInvRatio * 1.2f, 0.25f);
            const float target_s = fmaxf(max_s + 2.0f, 16.0f);
            const float cur_u = fmaxf(fabsf(old_mm.x), fabsf(old_mm.y) * kQ8ZeroExactHiInvRatio);
            const bool shrink = (cur_u > target_u * 1.35f) || (old_mm.w > target_s + 4.0f) || (old_mm.z > 0.0f);
            sm_do_reencode = shrink ? 1 : 0;
            sm_new_mm = make_float4(-target_u, target_u * kQ8ZeroExactHiRatio, 0.0f, target_s);
        }
        __syncthreads();
        if (!sm_do_reencode)
            return;

        const float4 new_mm = sm_new_mm;
        const float inv_u = 1.0f / fmaxf(new_mm.y - new_mm.x, lfs::training::joint_adam::kEpsDevice);
        const float inv_s = 1.0f / fmaxf(new_mm.w - new_mm.z, lfs::training::joint_adam::kEpsDevice);
        if (prim < n_prims_param) {
            for (uint k = 0; k < sh_layout_slots; ++k) {
                const uint sh_slot = shAt(prim, k, sh_layout_slots);
#pragma unroll
                for (int c = 0; c < 4; ++c) {
                    const int64_t cell = static_cast<int64_t>(sh_slot) * 4 + c;
                    const float2 us = C8::decode_us(p.joint_packed, cell, old_mm);
                    C8::encode_us(p.joint_packed, cell, us.x, us.y,
                                  new_mm.x, new_mm.z, inv_u, inv_s);
                }
            }
        }
        __syncthreads();
        if (threadIdx.x == 0)
            reinterpret_cast<float4*>(p.joint_bounds)[bidx] = new_mm;
    }

    template <bool IS_POST_OVERFLOW>
    __global__ void rebound_sparse_blocks_cu(
        FusedAdamSettings fused_adam,
        SparseReboundState rebound,
        const uint n_primitives,
        const uint sh_layout_slots) {
        const uint slot = blockIdx.y;
        if (slot >= static_cast<uint>(kSparseReboundSlots))
            return;
        const uint count = (IS_POST_OVERFLOW ? rebound.counts_post : rebound.counts_pre)[slot];
        if (count == 0u)
            return;

        if (slot < 5u) {
            using C16 = lfs::training::joint_adam::DeviceCodec<16>;
            const FusedAdamParam& p = (slot == 0u)   ? fused_adam.means
                                      : (slot == 1u) ? fused_adam.sh0
                                      : (slot == 2u) ? fused_adam.scaling
                                      : (slot == 3u) ? fused_adam.rotation
                                                     : fused_adam.opacity;
            if (!p.enabled || p.joint_bits != 16 || p.joint_packed == nullptr || p.joint_bounds == nullptr ||
                p.n_attributes <= 0)
                return;
            const uint n_prims_param = p.n_primitives > 0 ? static_cast<uint>(p.n_primitives) : n_primitives;
            const uint n_attr = static_cast<uint>(p.n_attributes);
            for (uint item = blockIdx.x; item < count; item += gridDim.x) {
                const uint bidx = rebound.rebound_list[static_cast<size_t>(slot) * rebound.n_blocks + item];
                const float4 old_mm = reinterpret_cast<const float4*>(p.joint_bounds)[bidx];
                const float cur_u = fmaxf(fabsf(old_mm.x), fabsf(old_mm.y) * kQ16ZeroExactHiInvRatio);
                float4 new_mm;
                if constexpr (IS_POST_OVERFLOW) {
                    const float need_u = __uint_as_float(
                        rebound.needed_u[static_cast<size_t>(slot) * rebound.n_blocks + bidx]);
                    const float need_s = __uint_as_float(
                        rebound.needed_s[static_cast<size_t>(slot) * rebound.n_blocks + bidx]);
                    const float new_u = fmaxf(cur_u, fmaxf(need_u * kQ16ZeroExactHiInvRatio * 1.25f, kSparseMinU16));
                    const float new_s = fmaxf(old_mm.w, fmaxf(need_s + 2.0f, kSparseMinS16));
                    new_mm = make_float4(-new_u, new_u * kQ16ZeroExactHiRatio, 0.0f, new_s);
                } else {
                    const float new_u = fmaxf(cur_u, kSparseMinU16);
                    new_mm = make_float4(-new_u, new_u * kQ16ZeroExactHiRatio, 0.0f,
                                         fmaxf(old_mm.w, kSparseMinS16));
                }
                const float inv_u = 1.0f / fmaxf(new_mm.y - new_mm.x, lfs::training::joint_adam::kEpsDevice);
                const float inv_s = 1.0f / fmaxf(new_mm.w - new_mm.z, lfs::training::joint_adam::kEpsDevice);
                const uint prim = bidx * 256u + threadIdx.x;
                if (prim < n_prims_param) {
                    const uint base = prim * n_attr;
                    const uint row = base < static_cast<uint>(p.n_elements)
                                         ? min(n_attr, static_cast<uint>(p.n_elements) - base)
                                         : 0u;
                    for (uint i = 0; i < row; ++i) {
                        const int64_t cell = static_cast<int64_t>(base + i);
                        const float2 us = C16::decode_us(p.joint_packed, cell, old_mm);
                        C16::encode_us(p.joint_packed, cell, us.x, us.y,
                                       new_mm.x, new_mm.z, inv_u, inv_s);
                    }
                }
                __syncthreads();
                if (threadIdx.x == 0)
                    reinterpret_cast<float4*>(p.joint_bounds)[bidx] = new_mm;
            }
            return;
        }

        if (slot == 5u) {
            using C8 = lfs::training::joint_adam::DeviceCodec<8>;
            const FusedAdamParam& p = fused_adam.shN;
            if (!p.enabled || p.joint_bits != 8 || p.joint_packed == nullptr || p.joint_bounds == nullptr ||
                sh_layout_slots == 0u)
                return;
            const uint n_prims_param = p.n_primitives > 0 ? static_cast<uint>(p.n_primitives) : n_primitives;
            for (uint item = blockIdx.x; item < count; item += gridDim.x) {
                const uint bidx = rebound.rebound_list[static_cast<size_t>(5) * rebound.n_blocks + item];
                const float4 old_mm = reinterpret_cast<const float4*>(p.joint_bounds)[bidx];
                const float cur_u = fmaxf(fabsf(old_mm.x), fabsf(old_mm.y) * kQ8ZeroExactHiInvRatio);
                float4 new_mm;
                if constexpr (IS_POST_OVERFLOW) {
                    const float need_u = __uint_as_float(
                        rebound.needed_u[static_cast<size_t>(5) * rebound.n_blocks + bidx]);
                    const float need_s = __uint_as_float(
                        rebound.needed_s[static_cast<size_t>(5) * rebound.n_blocks + bidx]);
                    const float new_u = fmaxf(cur_u, fmaxf(need_u * kQ8ZeroExactHiInvRatio * 1.25f, kSparseMinU8));
                    const float new_s = fmaxf(old_mm.w, fmaxf(need_s + 2.0f, kSparseMinS8));
                    new_mm = make_float4(-new_u, new_u * kQ8ZeroExactHiRatio, 0.0f, new_s);
                } else {
                    const float new_u = fmaxf(cur_u, kSparseMinU8);
                    new_mm = make_float4(-new_u, new_u * kQ8ZeroExactHiRatio, 0.0f,
                                         fmaxf(old_mm.w, kSparseMinS8));
                }
                const float inv_u = 1.0f / fmaxf(new_mm.y - new_mm.x, lfs::training::joint_adam::kEpsDevice);
                const float inv_s = 1.0f / fmaxf(new_mm.w - new_mm.z, lfs::training::joint_adam::kEpsDevice);
                const uint prim = bidx * 256u + threadIdx.x;
                if (prim < n_prims_param) {
                    for (uint k = 0; k < sh_layout_slots; ++k) {
                        const uint sh_slot = shAt(prim, k, sh_layout_slots);
#pragma unroll
                        for (int c = 0; c < 4; ++c) {
                            const int64_t cell = static_cast<int64_t>(sh_slot) * 4 + c;
                            const float2 us = C8::decode_us(p.joint_packed, cell, old_mm);
                            C8::encode_us(p.joint_packed, cell, us.x, us.y,
                                          new_mm.x, new_mm.z, inv_u, inv_s);
                        }
                    }
                }
                __syncthreads();
                if (threadIdx.x == 0)
                    reinterpret_cast<float4*>(p.joint_bounds)[bidx] = new_mm;
            }
            return;
        }

        // slot == 6: q16 SH-rest value bounds.
        using VC = lfs::core::sh_value::DeviceCodec16;
        const FusedAdamParam& p = fused_adam.shN;
        if (!p.enabled || p.sh_value_bits != 16 || p.sh_value_bounds == nullptr || p.sh_value_n_cells <= 0)
            return;
        const uint n_prims_param = p.n_primitives > 0 ? static_cast<uint>(p.n_primitives) : n_primitives;
        const uint n_cells = static_cast<uint>(p.sh_value_n_cells);
        uint16_t* const param_u16 = reinterpret_cast<uint16_t*>(p.param);
        for (uint item = blockIdx.x; item < count; item += gridDim.x) {
            const uint bidx = rebound.rebound_list[static_cast<size_t>(6) * rebound.n_blocks + item];
            const float2 old_vmm = reinterpret_cast<const float2*>(p.sh_value_bounds)[bidx];
            const float cur_half = fmaxf(fabsf(old_vmm.x), fabsf(old_vmm.y) * kQ16ZeroExactHiInvRatio);
            float new_half;
            if constexpr (IS_POST_OVERFLOW) {
                const float need_v = __uint_as_float(
                    rebound.needed_u[static_cast<size_t>(6) * rebound.n_blocks + bidx]);
                new_half = fmaxf(cur_half, fmaxf(need_v * kQ16ZeroExactHiInvRatio * 1.25f, kSparseMinVHalf));
            } else {
                new_half = fmaxf(cur_half, kSparseMinVHalf);
            }
            const float2 new_vmm = make_float2(-new_half, new_half * kQ16ZeroExactHiRatio);
            const float inv_range = 1.0f / fmaxf(new_vmm.y - new_vmm.x, lfs::core::sh_value::kEpsDevice);
            const uint prim = bidx * 256u + threadIdx.x;
            if (prim < n_prims_param) {
                for (uint c = 0; c < n_cells; ++c) {
                    const uint idx = lfs::core::sh_value::shAtU16(prim, c, n_cells);
                    const float v = VC::decode(param_u16[idx], old_vmm.x, old_vmm.y);
                    const float qf = fminf(
                        fmaxf(roundf(lfs::core::sh_value::kQMaxDevice * (v - new_vmm.x) * inv_range), 0.0f),
                        lfs::core::sh_value::kQMaxDevice);
                    param_u16[idx] = static_cast<uint16_t>(qf);
                }
            }
            __syncthreads();
            if (threadIdx.x == 0)
                reinterpret_cast<float2*>(p.sh_value_bounds)[bidx] = new_vmm;
        }
    }

    struct SparseShAdamArgs {
        FusedAdamParam shN;
        uint16_t* lazy_last_step = nullptr;
        int lazy_last_step_n = 0;
        uint32_t current_step = 0u;
        float beta1 = 0.9f;
        float beta2 = 0.999f;
        float eps = 1e-15f;
        float sh_rest_reg_weight = 0.0f;
        float* sh_rest_reg_loss_out = nullptr;
        bool has_sh0 = false;
    };

    // One visible row of the MRNF per-step stat fold: identical to the standalone
    // sparse fold kernel, run in the thread that just finished the row's backward.
    __device__ __forceinline__ void fold_mrnf_step_stats_row(const MrnfFoldArgs& fold, const uint prim) {
        const size_t N = fold.n_primitives;
        if (static_cast<size_t>(prim) >= N)
            return;
        const bool frozen = (fold.frozen_mask != nullptr && prim < fold.frozen_n && fold.frozen_mask[prim]);
        if (fold.densification_info != nullptr && fold.refine_weight != nullptr) {
            constexpr float kRefineErrorCompressViewPower = 0.2f;
            float* const info = fold.densification_info;
            const float coverage = info[prim];
            const float error = info[N + prim];
            if (coverage != 0.0f)
                info[prim] = 0.0f;
            if (error != 0.0f)
                info[N + prim] = 0.0f;
            if (fold.coverage_views != nullptr && coverage > 0.0f)
                fold.coverage_views[prim] += 1.0f;
            if (error > 0.0f) {
                const float s = powf(error, kRefineErrorCompressViewPower);
                if (!frozen)
                    fold.refine_weight[prim] += s;
                if (s > 0.0f)
                    info[3 * N + prim] += 1.0f;
            }
            if (frozen) {
                fold.refine_weight[prim] = 0.0f;
                info[2 * N + prim] = 0.0f;
            } else if (coverage != 0.0f) {
                info[2 * N + prim] += coverage;
            }
        }
        if (fold.step_peak != nullptr && fold.peak != nullptr && fold.renders != nullptr) {
            const float step = fold.step_peak[prim];
            if (step != 0.0f) {
                fold.step_peak[prim] = 0.0f;
                if (step > 0.0f) {
                    fold.renders[prim] += 1.0f;
                    const float scaled_step = step * fold.inv_peak_scale;
                    if (scaled_step > fold.peak[prim])
                        fold.peak[prim] = scaled_step;
                }
            }
        }
    }

    // MRNF low-opacity mean noise for one row after its Adam step. The variates are
    // bit-identical to curand_init(seed, idx, 0) + curand_normal4 (Philox4x32-10 +
    // Box-Muller), computed statelessly in registers.
    __device__ __forceinline__ void apply_mrnf_noise_fused(
        const MrnfNoiseArgs& noise, float* const means_param, const float* const opacity_param, const uint idx) {
        if (!noise.enabled || means_param == nullptr || opacity_param == nullptr || noise.vis_count == nullptr)
            return;
        const float raw_opac = opacity_param[idx];
        if (raw_opac > noise.max_logit)
            return;
        if (noise.frozen_mask != nullptr && idx < noise.frozen_mask_size && noise.frozen_mask[idx])
            return;
        if (noise.vis_count[idx] <= 0.0f)
            return;
        const float inv_opac = 1.0f / (1.0f + expf(raw_opac));
        const float weight = powf(fmaxf(inv_opac, 0.0f), 150.0f) * noise.lr_mean_noise_weight;
        if (weight < 1e-12f)
            return;

        uint c0 = 0u;
        uint c1 = 0u;
        uint c2 = idx;
        uint c3 = 0u;
        uint k0 = static_cast<uint>(noise.seed);
        uint k1 = static_cast<uint>(noise.seed >> 32);
#pragma unroll
        for (int round = 0; round < 10; ++round) {
            const uint hi0 = __umulhi(0xD2511F53u, c0);
            const uint lo0 = 0xD2511F53u * c0;
            const uint hi1 = __umulhi(0xCD9E8D57u, c2);
            const uint lo1 = 0xCD9E8D57u * c2;
            c0 = hi1 ^ c1 ^ k0;
            c1 = lo1;
            c2 = hi0 ^ c3 ^ k1;
            c3 = lo0;
            k0 += 0x9E3779B9u;
            k1 += 0xBB67AE85u;
        }
        constexpr float kInv32 = 2.3283064e-10f;
        constexpr float kInv32TwoPi = 2.3283064e-10f * 6.2831855f;
        const float u0 = c0 * kInv32 + (kInv32 / 2);
        const float v0 = c1 * kInv32TwoPi + (kInv32TwoPi / 2);
        const float u1 = c2 * kInv32 + (kInv32 / 2);
        const float v1 = c3 * kInv32TwoPi + (kInv32TwoPi / 2);
        const float r0 = sqrtf(-2.0f * logf(u0));
        const float r1 = sqrtf(-2.0f * logf(u1));
        float s0;
        float cs0;
        float s1;
        float cs1;
        __sincosf(v0, &s0, &cs0);
        __sincosf(v1, &s1, &cs1);
        static_cast<void>(cs1);
        const float m = noise.median_scale;
        float* const mp = means_param + static_cast<size_t>(idx) * 3u;
        mp[0] += fminf(fmaxf((s0 * r0) * weight, -m), m);
        mp[1] += fminf(fmaxf((cs0 * r0) * weight, -m), m);
        mp[2] += fminf(fmaxf((s1 * r1) * weight, -m), m);
    }

    struct SparseGeomAdamArgs {
        FusedAdamParam means;
        FusedAdamParam sh0;
        FusedAdamParam scaling;
        FusedAdamParam rotation;
        FusedAdamParam opacity;
        // Per-splat last step at which rotation/sh0 moments were current; when set,
        // non-contributing rows coast lazily and are caught up here in closed form.
        uint16_t* geom_last_step = nullptr;
        int geom_last_step_n = 0;
        uint32_t current_step = 0u;
        float beta1 = 0.9f;
        float beta2 = 0.999f;
        float eps = 1e-15f;
        float dc_reg_weight = 0.0f;
        float scale_reg_weight = 0.0f;
        bool scale_reg_log = false;
        float scale_reg_normalizer = 1.0f;
        float erank_reg_weight = 0.0f;
        float flatten_reg_weight = 0.0f;
        float* rendered_count = nullptr;
        float* dc_reg_loss_out = nullptr;
        float* scale_reg_loss_out = nullptr;
        float* erank_reg_loss_out = nullptr;
        bool per_splat_mean_step = false;
        float mean_step_median_extent = 0.0f;
        const bool* mean_step_far_mask = nullptr;
        int mean_step_far_mask_n = 0;
        float* ray_depth_state = nullptr;
        int ray_depth_n = 0;
        float ray_depth_lr = 0.0f;
        float ray_depth_mean_lr = 0.0f;
        float ray_depth_far_limit = 0.0f;
        MrnfFoldArgs mrnf_fold{};
        MrnfNoiseArgs mrnf_noise{};
    };

    struct SparseInvisGeomArgs {
        FusedAdamParam means;
        FusedAdamParam sh0;
        FusedAdamParam scaling;
        FusedAdamParam rotation;
        FusedAdamParam opacity;
        uint16_t* geom_last_step = nullptr;
        int geom_last_step_n = 0;
        uint32_t step_seed = 0u;
        float beta1 = 0.9f;
        float beta2 = 0.999f;
        float eps = 1e-15f;
        float dc_reg_weight = 0.0f;
        float scale_reg_weight = 0.0f;
        bool scale_reg_log = false;
        float scale_reg_normalizer = 1.0f;
        float erank_reg_weight = 0.0f;
        float* dc_reg_loss_out = nullptr;
        float* scale_reg_loss_out = nullptr;
        float* erank_reg_loss_out = nullptr;
        bool per_splat_mean_step = false;
        float mean_step_median_extent = 0.0f;
        const bool* mean_step_far_mask = nullptr;
        int mean_step_far_mask_n = 0;
        MrnfNoiseArgs mrnf_noise{};
    };

    // Split SH-rest half of sparse backward preprocessing: launched per 256-splat block
    // so every 32-splat SH tile is 100% warp-coalesced and block bounds live in CTA
    // shared memory. Inactive blocks exit on a single L1-cached byte read; active blocks
    // precompute a 256-entry SMEM LUT for inverse_sqrt_g2 (1 expf/thread vs 45/splat).
    template <int ACTIVE_SH_BASES, bool UPDATE_SHN, bool VALUE_Q16, bool REPORT_REG_LOSS = false>
    __global__ void __launch_bounds__(config::block_size_preprocess_backward, 3) preprocess_backward_sparse_sh_cu(
        const float3* __restrict__ means,
        const float4* sh_coefficients_rest,
        const float3* __restrict__ cam_position,
        const std::uint8_t* __restrict__ active_block_flag,
        const uint* __restrict__ active_block_ids,
        const uint* __restrict__ primitive_work_indices,
        const std::uint8_t* __restrict__ primitive_touched,
        const ProjectedRecord* __restrict__ projected,
        const float3* __restrict__ grad_color_helper,
        float3* __restrict__ grad_color_mean_out,
        const uint n_primitives,
        const uint sh_layout_slots,
        const SparseShAdamArgs sh_adam,
        const SparseReboundState rebound,
        const float2* __restrict__ shN_value_bounds,
        const uint shN_value_n_cells,
        const uint shN_value_bits) {
        const uint bidx = (active_block_ids != nullptr) ? active_block_ids[blockIdx.x] : blockIdx.x;
        if (active_block_ids == nullptr && active_block_flag != nullptr && active_block_flag[bidx] == 0u)
            return;

        const uint primitive_idx = bidx * blockDim.x + threadIdx.x;
        const bool in_range = primitive_idx < n_primitives;
        const uint work_idx = in_range ? primitive_work_indices[primitive_idx] : 0xffffffffu;
        const bool contributed =
            in_range && work_idx != 0xffffffffu &&
            (primitive_touched == nullptr || primitive_touched[work_idx] != 0u);

        __shared__ float s_sqrt_g2_lut[256];
        __shared__ float3 s_lazy_aux[256];
        __shared__ uint32_t s_over_state[4]; // [0]=flags(bit0:joint,bit1:val), [1]=u, [2]=s, [3]=v
        __shared__ ShBlockMeta s_meta;
        if constexpr (UPDATE_SHN) {
            if (__syncthreads_or(contributed) == 0)
                return;
            using C8 = lfs::training::joint_adam::DeviceCodec<8>;
            const float4 mm_lut = *reinterpret_cast<const float4*>(sh_adam.shN.joint_bounds + 4u * bidx);
            const float s_span_lut = mm_lut.w - mm_lut.z;
            const float s_dec_lut = s_span_lut * C8::kInvQMax;
            s_sqrt_g2_lut[threadIdx.x] =
                C8::inverse_sqrt_g2(mm_lut.z + static_cast<float>(threadIdx.x) * s_dec_lut);
            if (threadIdx.x < 4u)
                s_over_state[threadIdx.x] = 0u;
            if (threadIdx.x == 0u) {
                const float u_span_lut = mm_lut.y - mm_lut.x;
                s_meta.u_min = mm_lut.x;
                s_meta.s_min = mm_lut.z;
                s_meta.u_dec = u_span_lut * C8::kInvQMax;
                s_meta.u_enc = C8::kQMax / fmaxf(u_span_lut, lfs::training::joint_adam::kEpsDevice);
                s_meta.s_enc = C8::kQMax / fmaxf(s_span_lut, lfs::training::joint_adam::kEpsDevice);
                if constexpr (VALUE_Q16) {
                    const float2 vmm = *reinterpret_cast<const float2*>(sh_adam.shN.sh_value_bounds + 2u * bidx);
                    const float v_span = vmm.y - vmm.x;
                    s_meta.v_min = vmm.x;
                    s_meta.v_dec = v_span * (1.0f / 65535.0f);
                    s_meta.v_enc = 65535.0f / fmaxf(v_span, lfs::core::sh_value::kEpsDevice);
                    s_meta.n_value_cells = static_cast<uint>(sh_adam.shN.sh_value_n_cells);
                } else {
                    s_meta.v_min = 0.0f;
                    s_meta.v_dec = 0.0f;
                    s_meta.v_enc = 0.0f;
                    s_meta.n_value_cells = 0u;
                }
                constexpr uint kActiveCells = static_cast<uint>(ACTIVE_SH_BASES - 1) * 3u;
                s_meta.reg_scale = (sh_adam.sh_rest_reg_weight > 0.0f && sh_adam.shN.n_primitives > 0)
                                       ? sh_adam.sh_rest_reg_weight /
                                             (static_cast<float>(sh_adam.shN.n_primitives) * static_cast<float>(kActiveCells))
                                       : 0.0f;
                s_meta.bc2_sqrt_rcp = sh_adam.shN.bias_correction2_sqrt_rcp;
            }
            __syncthreads();
        }

        const uint active_mask = __ballot_sync(0xffffffffu, contributed);
        if constexpr (!UPDATE_SHN && !REPORT_REG_LOSS) {
            if (active_mask == 0u)
                return;
        }

        float sh_rest_loss_local = 0.0f;

        if (active_mask != 0u && contributed) {
            const float3 colour = make_float3(projected[work_idx].color);
            float3 image_grad = grad_color_helper[work_idx];
            if (colour.x < 0.0f && image_grad.x >= 0.0f)
                image_grad.x = 0.0f;
            if (colour.y < 0.0f && image_grad.y >= 0.0f)
                image_grad.y = 0.0f;
            if (colour.z < 0.0f && image_grad.z >= 0.0f)
                image_grad.z = 0.0f;

            uint colour_reg_mask = 7u;
            if (sh_adam.has_sh0) {
                colour_reg_mask =
                    (colour.x >= 0.0f || image_grad.x < 0.0f ? 1u : 0u) |
                    (colour.y >= 0.0f || image_grad.y < 0.0f ? 2u : 0u) |
                    (colour.z >= 0.0f || image_grad.z < 0.0f ? 4u : 0u);
            }

            float3 dL_dmean3d_from_color;
            if constexpr (UPDATE_SHN) {
                const uint32_t missed = lazy_missed_steps(
                    sh_adam.lazy_last_step, sh_adam.lazy_last_step_n, primitive_idx, sh_adam.current_step);
                const LazyAdamFactors lazy = compute_lazy_adam_factors(missed, sh_adam.beta1, sh_adam.beta2);
                s_lazy_aux[threadIdx.x] = make_float3(lazy.decay2_half, lazy.coast_sum, lazy.reg_sum);
                dL_dmean3d_from_color =
                    shN_backward_and_adam_sparse<ACTIVE_SH_BASES, VALUE_Q16, REPORT_REG_LOSS>(
                        sh_adam.shN, means[primitive_idx], cam_position[0],
                        primitive_idx, sh_layout_slots, image_grad,
                        colour_reg_mask, sh_adam.beta1, sh_adam.beta2, sh_adam.eps,
                        lazy.decay1, s_lazy_aux, s_meta, s_sqrt_g2_lut, s_over_state, sh_rest_loss_local);
                if (sh_adam.lazy_last_step != nullptr && sh_adam.current_step > 0u &&
                    primitive_idx < static_cast<uint>(sh_adam.lazy_last_step_n)) {
                    sh_adam.lazy_last_step[primitive_idx] = encode_lazy_step(sh_adam.current_step);
                }
            } else {
                float dummy_sh0[3];
                dL_dmean3d_from_color = convert_sh_to_color_backward_grads_from_color<ACTIVE_SH_BASES>(
                    sh_coefficients_rest, image_grad, means[primitive_idx], cam_position[0],
                    primitive_idx, sh_layout_slots, dummy_sh0,
                    shN_value_bounds,
                    shN_value_bounds != nullptr ? shN_value_n_cells : 0u,
                    shN_value_bits == 16u ? 16u : 0u);
            }
            grad_color_mean_out[work_idx] = dL_dmean3d_from_color;
        }

        if constexpr (UPDATE_SHN) {
            __syncthreads();
            if (threadIdx.x == 0u) {
                const uint32_t ov = s_over_state[0];
                if (__builtin_expect((ov & 1u) != 0u, 0)) {
                    sparse_rebound_mark(rebound, 5, bidx,
                                        __uint_as_float(s_over_state[1]),
                                        __uint_as_float(s_over_state[2]));
                }
                if constexpr (VALUE_Q16) {
                    if (__builtin_expect((ov & 2u) != 0u, 0)) {
                        sparse_rebound_mark(rebound, 6, bidx,
                                            __uint_as_float(s_over_state[3]), 0.0f);
                    }
                }
            }
        }

        if constexpr (REPORT_REG_LOSS) {
            if (sh_adam.sh_rest_reg_loss_out != nullptr) {
                const float block_sum = lfs::core::warp_ops::block_reduce_sum(sh_rest_loss_local);
                if (threadIdx.x == 0 && block_sum != 0.0f)
                    atomicAdd(sh_adam.sh_rest_reg_loss_out, block_sum);
            }
        }
    }

    // Visible geometry + sh0 half of sparse backward preprocessing: launched per 256-splat
    // block so all geometry + 16-bit moment accesses are 100% coalesced and bounds are CTA-uniform.
    // Inactive blocks exit immediately via active_block_flag; contributing rows perform
    // closed-form lazy catch-up over missed steps before applying the current step gradient.
    template <bool MIP_FILTER, bool HAS_COLOR_MEAN_GRAD, bool REPORT_REG_LOSS = false>
    __global__ void __launch_bounds__(config::block_size_preprocess_backward, 3) preprocess_backward_sparse_geom_cu(
        const float3* means,
        const float3* raw_scales,
        const float4* raw_rotations,
        const float4* __restrict__ w2c,
        const float3* __restrict__ cam_position,
        const float* raw_opacities,
        const std::uint8_t* __restrict__ active_block_flag,
        const uint* __restrict__ active_block_ids,
        const uint* __restrict__ primitive_work_indices,
        const std::uint8_t* __restrict__ primitive_touched,
        const ProjectedRecord* __restrict__ projected,
        const float2* __restrict__ grad_mean2d,
        const float3* __restrict__ grad_conic,
        const float* __restrict__ grad_depth,
        const float3* __restrict__ grad_normal,
        const float* __restrict__ grad_opacity_helper,
        const float3* __restrict__ grad_color_helper,
        const float3* __restrict__ grad_mean3d_from_color_in,
        float* __restrict__ densification_weight,
        float* __restrict__ densification_error,
        const uint n_primitives,
        const float w,
        const float h,
        const float fx,
        const float fy,
        const float clip_left,
        const float clip_right,
        const float clip_top,
        const float clip_bottom,
        const float dilation_scale,
        const SparseGeomAdamArgs geom_adam,
        const SparseReboundState rebound) {
        const uint bidx = (active_block_ids != nullptr) ? active_block_ids[blockIdx.x] : blockIdx.x;
        if (active_block_ids == nullptr && active_block_flag != nullptr && active_block_flag[bidx] == 0u)
            return;

        const uint primitive_idx = bidx * blockDim.x + threadIdx.x;
        const bool in_range = primitive_idx < n_primitives;
        const uint work_idx = in_range ? primitive_work_indices[primitive_idx] : 0xffffffffu;
        const bool visible = in_range && work_idx != 0xffffffffu;
        if constexpr (!REPORT_REG_LOSS) {
            if (__ballot_sync(0xffffffffu, visible) == 0u)
                return;
        }

        if (visible) {
            if (geom_adam.rendered_count != nullptr)
                geom_adam.rendered_count[primitive_idx] += 1.0f;
            if (densification_error != nullptr) {
                if (densification_weight != nullptr)
                    densification_weight[primitive_idx] += 1.0f;
                densification_error[primitive_idx] +=
                    length(grad_mean2d[work_idx] * make_float2(0.5f * w, 0.5f * h));
            }
            if (geom_adam.mrnf_fold.n_primitives != 0u)
                fold_mrnf_step_stats_row(geom_adam.mrnf_fold, primitive_idx);
        }

        const bool contributed =
            visible && (primitive_touched == nullptr || primitive_touched[work_idx] != 0u);
        const uint contrib_mask = __ballot_sync(0xffffffffu, contributed);
        if constexpr (!REPORT_REG_LOSS) {
            if (contrib_mask == 0u)
                return;
        }

        float dc_loss_local = 0.0f;
        float scale_loss_local = 0.0f;
        float erank_loss_local = 0.0f;

        if (contrib_mask != 0u && contributed) {
            // Rotation and sh0 rows skipped while non-contributing are caught up in
            // closed form inside their Adam step.
            const uint32_t geom_missed = lazy_missed_steps(
                geom_adam.geom_last_step, geom_adam.geom_last_step_n, primitive_idx, geom_adam.current_step);
            const LazyAdamFactors geom_lazy =
                compute_lazy_adam_factors(geom_missed, geom_adam.beta1, geom_adam.beta2);
            // Update sh0 first so its temporaries die before EWA/covariance live range.
            {
                const float3 colour = make_float3(projected[work_idx].color);
                float3 image_grad = grad_color_helper[work_idx];
                if (colour.x < 0.0f && image_grad.x >= 0.0f)
                    image_grad.x = 0.0f;
                if (colour.y < 0.0f && image_grad.y >= 0.0f)
                    image_grad.y = 0.0f;
                if (colour.z < 0.0f && image_grad.z >= 0.0f)
                    image_grad.z = 0.0f;

                uint colour_reg_mask = 7u;
                if (geom_adam.sh0.param != nullptr) {
                    colour_reg_mask =
                        (colour.x >= 0.0f || image_grad.x < 0.0f ? 1u : 0u) |
                        (colour.y >= 0.0f || image_grad.y < 0.0f ? 2u : 0u) |
                        (colour.z >= 0.0f || image_grad.z < 0.0f ? 4u : 0u);
                }

                const float3 dL_dsh0 = 0.28209479177387814f * image_grad;
                float sh0_grads[3] = {dL_dsh0.x, dL_dsh0.y, dL_dsh0.z};
                constexpr float kSh0 = 0.28209479177387814f;
                constexpr float kDcLimit = 0.5f / kSh0;
                const float dc_grad_scale = (geom_adam.dc_reg_weight > 0.0f && geom_adam.sh0.param != nullptr)
                                                ? (2.0f * geom_adam.dc_reg_weight / 3.0f)
                                                : 0.0f;
                const float dc_loss_weight =
                    (geom_adam.dc_reg_weight > 0.0f && n_primitives > 0u)
                        ? (geom_adam.dc_reg_weight / (3.0f * static_cast<float>(n_primitives)))
                        : 0.0f;
                adam_step_row_sparse<3, true, REPORT_REG_LOSS>(
                    sh0_grads, geom_adam.sh0, primitive_idx, bidx,
                    geom_adam.beta1, geom_adam.beta2, geom_adam.eps, rebound, 1,
                    1.0f, geom_lazy, geom_missed, kDcLimit, dc_grad_scale, dc_loss_weight,
                    colour_reg_mask, &dc_loss_local);
            }

            const float3 mean3d = means[primitive_idx];
            const float4 w2c_r3 = w2c[2];
            const float depth = w2c_r3.x * mean3d.x + w2c_r3.y * mean3d.y + w2c_r3.z * mean3d.z + w2c_r3.w;
            const float depth_safe = fmaxf(depth, 1e-4f);
            const float inv_depth = 1.0f / depth_safe;
            const float4 w2c_r1 = w2c[0];
            const float x = (w2c_r1.x * mean3d.x + w2c_r1.y * mean3d.y + w2c_r1.z * mean3d.z + w2c_r1.w) * inv_depth;
            const float4 w2c_r2 = w2c[1];
            const float y = (w2c_r2.x * mean3d.x + w2c_r2.y * mean3d.y + w2c_r2.z * mean3d.z + w2c_r2.w) * inv_depth;

            const float3 raw_scale = raw_scales[primitive_idx];
            const float3 clamped_scale = make_float3(
                fminf(raw_scale.x, config::max_raw_scale),
                fminf(raw_scale.y, config::max_raw_scale),
                fminf(raw_scale.z, config::max_raw_scale));
            const float3 variance = make_float3(
                expf(2.0f * clamped_scale.x), expf(2.0f * clamped_scale.y), expf(2.0f * clamped_scale.z));
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

            const float kernel_size = (MIP_FILTER ? config::dilation_mip_filter : config::dilation) * dilation_scale;
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
            const float3 dL_dcov2d = determinant_rcp_sq * make_float3(
                                                              2.0f * bc * dL_dconic.y - cc * dL_dconic.x - bb * dL_dconic.z,
                                                              bc * dL_dconic.x - (ac + bb) * dL_dconic.y + ab * dL_dconic.z,
                                                              2.0f * ab * dL_dconic.y - bb * dL_dconic.x - aa * dL_dconic.z);

            const float grad_compensated_opacity = grad_opacity_helper[work_idx];
            float opacity_compensation = 1.0f;
            if constexpr (MIP_FILTER) {
                opacity_compensation = sqrtf(fmaxf(raw_a * raw_c - raw_b * raw_b, 0.0f) * determinant_rcp);
            }
            const float sigmoid_derivative = lfs::core::sigmoid_derivative_from_logit(raw_opacities[primitive_idx]);
            const float opacity_grads[1] = {grad_compensated_opacity * opacity_compensation * sigmoid_derivative};
            adam_step_row_sparse<1>(
                opacity_grads, geom_adam.opacity, primitive_idx, bidx,
                geom_adam.beta1, geom_adam.beta2, geom_adam.eps, rebound, 4);

            const mat3x3_triu dL_dcov3d = {
                (jw_r1.x * jw_r1.x) * dL_dcov2d.x + 2.0f * (jw_r1.x * jw_r2.x) * dL_dcov2d.y + (jw_r2.x * jw_r2.x) * dL_dcov2d.z,
                (jw_r1.x * jw_r1.y) * dL_dcov2d.x + (jw_r1.x * jw_r2.y + jw_r1.y * jw_r2.x) * dL_dcov2d.y + (jw_r2.x * jw_r2.y) * dL_dcov2d.z,
                (jw_r1.x * jw_r1.z) * dL_dcov2d.x + (jw_r1.x * jw_r2.z + jw_r1.z * jw_r2.x) * dL_dcov2d.y + (jw_r2.x * jw_r2.z) * dL_dcov2d.z,
                (jw_r1.y * jw_r1.y) * dL_dcov2d.x + 2.0f * (jw_r1.y * jw_r2.y) * dL_dcov2d.y + (jw_r2.y * jw_r2.y) * dL_dcov2d.z,
                (jw_r1.y * jw_r1.z) * dL_dcov2d.x + (jw_r1.y * jw_r2.z + jw_r1.z * jw_r2.y) * dL_dcov2d.y + (jw_r2.y * jw_r2.z) * dL_dcov2d.z,
                (jw_r1.z * jw_r1.z) * dL_dcov2d.x + 2.0f * (jw_r1.z * jw_r2.z) * dL_dcov2d.y + (jw_r2.z * jw_r2.z) * dL_dcov2d.z,
            };

            const float3 dL_djw_r1 = 2.0f * make_float3(
                                                jwc_r1.x * dL_dcov2d.x + jwc_r2.x * dL_dcov2d.y,
                                                jwc_r1.y * dL_dcov2d.x + jwc_r2.y * dL_dcov2d.y,
                                                jwc_r1.z * dL_dcov2d.x + jwc_r2.z * dL_dcov2d.y);
            const float3 dL_djw_r2 = 2.0f * make_float3(
                                                jwc_r1.x * dL_dcov2d.y + jwc_r2.x * dL_dcov2d.z,
                                                jwc_r1.y * dL_dcov2d.y + jwc_r2.y * dL_dcov2d.z,
                                                jwc_r1.z * dL_dcov2d.y + jwc_r2.z * dL_dcov2d.z);

            const float dL_dj11 = w2c_r1.x * dL_djw_r1.x + w2c_r1.y * dL_djw_r1.y + w2c_r1.z * dL_djw_r1.z;
            const float dL_dj22 = w2c_r2.x * dL_djw_r2.x + w2c_r2.y * dL_djw_r2.y + w2c_r2.z * dL_djw_r2.z;
            const float dL_dj13 = w2c_r3.x * dL_djw_r1.x + w2c_r3.y * dL_djw_r1.y + w2c_r3.z * dL_djw_r1.z;
            const float dL_dj23 = w2c_r3.x * dL_djw_r2.x + w2c_r3.y * dL_djw_r2.y + w2c_r3.z * dL_djw_r2.z;

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
            dL_dmean3d_cam.z +=
                (j11 * (factor_x * tx * dL_dj13 - dL_dj11) + j22 * (factor_y * ty * dL_dj23 - dL_dj22)) * inv_depth;

            const float3 dL_dmean3d_from_splatting = make_float3(
                w2c_r1.x * dL_dmean3d_cam.x + w2c_r2.x * dL_dmean3d_cam.y + w2c_r3.x * dL_dmean3d_cam.z,
                w2c_r1.y * dL_dmean3d_cam.x + w2c_r2.y * dL_dmean3d_cam.y + w2c_r3.y * dL_dmean3d_cam.z,
                w2c_r1.z * dL_dmean3d_cam.x + w2c_r2.z * dL_dmean3d_cam.y + w2c_r3.z * dL_dmean3d_cam.z);
            const float3 dL_dmean3d_from_color =
                HAS_COLOR_MEAN_GRAD ? grad_mean3d_from_color_in[work_idx] : make_float3(0.0f, 0.0f, 0.0f);
            const float3 clamped_mean = clamp_grad3(dL_dmean3d_from_splatting + dL_dmean3d_from_color);
            float mean_grads[3] = {clamped_mean.x, clamped_mean.y, clamped_mean.z};

            const float dL_dvariance_x =
                rotation.m11 * rotation.m11 * dL_dcov3d.m11 + rotation.m21 * rotation.m21 * dL_dcov3d.m22 +
                rotation.m31 * rotation.m31 * dL_dcov3d.m33 +
                2.0f * (rotation.m11 * rotation.m21 * dL_dcov3d.m12 + rotation.m11 * rotation.m31 * dL_dcov3d.m13 +
                        rotation.m21 * rotation.m31 * dL_dcov3d.m23);
            const float dL_dvariance_y =
                rotation.m12 * rotation.m12 * dL_dcov3d.m11 + rotation.m22 * rotation.m22 * dL_dcov3d.m22 +
                rotation.m32 * rotation.m32 * dL_dcov3d.m33 +
                2.0f * (rotation.m12 * rotation.m22 * dL_dcov3d.m12 + rotation.m12 * rotation.m32 * dL_dcov3d.m13 +
                        rotation.m22 * rotation.m32 * dL_dcov3d.m23);
            const float dL_dvariance_z =
                rotation.m13 * rotation.m13 * dL_dcov3d.m11 + rotation.m23 * rotation.m23 * dL_dcov3d.m22 +
                rotation.m33 * rotation.m33 * dL_dcov3d.m33 +
                2.0f * (rotation.m13 * rotation.m23 * dL_dcov3d.m12 + rotation.m13 * rotation.m33 * dL_dcov3d.m13 +
                        rotation.m23 * rotation.m33 * dL_dcov3d.m23);
            const float3 dL_draw_scale = make_float3(
                (raw_scale.x < config::max_raw_scale) ? 2.0f * variance.x * dL_dvariance_x : 0.0f,
                (raw_scale.y < config::max_raw_scale) ? 2.0f * variance.y * dL_dvariance_y : 0.0f,
                (raw_scale.z < config::max_raw_scale) ? 2.0f * variance.z * dL_dvariance_z : 0.0f);
            const float3 clamped_scale_grad = clamp_grad3(dL_draw_scale);
            float scale_grads[3] = {clamped_scale_grad.x, clamped_scale_grad.y, clamped_scale_grad.z};

            if (geom_adam.scale_reg_weight > 0.0f && geom_adam.scaling.n_elements > 0) {
                const float inv_n = 1.0f / static_cast<float>(geom_adam.scaling.n_elements);
                if (geom_adam.scale_reg_log) {
                    const float g = geom_adam.scale_reg_weight * 0.01f * inv_n / geom_adam.scale_reg_normalizer;
                    if (raw_scale.x > -40.0f)
                        scale_grads[0] += g;
                    if (raw_scale.y > -40.0f)
                        scale_grads[1] += g;
                    if (raw_scale.z > -40.0f)
                        scale_grads[2] += g;
                    if constexpr (REPORT_REG_LOSS) {
                        scale_loss_local = g * (fmaxf(raw_scale.x, -40.0f) + fmaxf(raw_scale.y, -40.0f) +
                                                fmaxf(raw_scale.z, -40.0f));
                    }
                } else {
                    const float ex = expf(raw_scale.x);
                    const float ey = expf(raw_scale.y);
                    const float ez = expf(raw_scale.z);
                    const float coef = geom_adam.scale_reg_weight * inv_n;
                    scale_grads[0] += coef * ex;
                    scale_grads[1] += coef * ey;
                    scale_grads[2] += coef * ez;
                    if constexpr (REPORT_REG_LOSS)
                        scale_loss_local = coef * (ex + ey + ez);
                }
            }
            if (geom_adam.erank_reg_weight > 0.0f && geom_adam.scaling.n_primitives > 0) {
                const auto penalty =
                    lfs::training::effective_rank_penalty(raw_scale.x, raw_scale.y, raw_scale.z);
                const float coef =
                    geom_adam.erank_reg_weight / static_cast<float>(geom_adam.scaling.n_primitives);
                if constexpr (REPORT_REG_LOSS)
                    erank_loss_local = coef * penalty.value;
                scale_grads[0] += coef * penalty.gradient[0];
                scale_grads[1] += coef * penalty.gradient[1];
                scale_grads[2] += coef * penalty.gradient[2];
            }
            if (geom_adam.flatten_reg_weight > 0.0f && geom_adam.scaling.n_elements > 0) {
                const uint min_axis = (raw_scale.x <= raw_scale.y && raw_scale.x <= raw_scale.z) ? 0u
                                      : (raw_scale.y <= raw_scale.z)                             ? 1u
                                                                                                 : 2u;
                const float s_min = min_axis == 0u ? raw_scale.x : (min_axis == 1u ? raw_scale.y : raw_scale.z);
                const float fg = 3.0f * geom_adam.flatten_reg_weight * expf(s_min) /
                                 static_cast<float>(geom_adam.scaling.n_elements);
                if (min_axis == 0u)
                    scale_grads[0] += fg;
                else if (min_axis == 1u)
                    scale_grads[1] += fg;
                else
                    scale_grads[2] += fg;
            }

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

            const float3 cam_pos = cam_position[0];
            if (grad_normal != nullptr) {
                const float3 dL_dnormal_cam = grad_normal[work_idx];
                const float3 g_world = make_float3(
                    w2c_r1.x * dL_dnormal_cam.x + w2c_r2.x * dL_dnormal_cam.y + w2c_r3.x * dL_dnormal_cam.z,
                    w2c_r1.y * dL_dnormal_cam.x + w2c_r2.y * dL_dnormal_cam.y + w2c_r3.y * dL_dnormal_cam.z,
                    w2c_r1.z * dL_dnormal_cam.x + w2c_r2.z * dL_dnormal_cam.y + w2c_r3.z * dL_dnormal_cam.z);
                const int axis = (variance.x <= variance.y && variance.x <= variance.z) ? 0
                                 : (variance.y <= variance.z)                           ? 1
                                                                                        : 2;
                const float3 axis_dir = axis == 0   ? make_float3(rotation.m11, rotation.m21, rotation.m31)
                                        : axis == 1 ? make_float3(rotation.m12, rotation.m22, rotation.m32)
                                                    : make_float3(rotation.m13, rotation.m23, rotation.m33);
                const float sign = dot(axis_dir, mean3d - cam_pos) > 0.0f ? -1.0f : 1.0f;
                const float3 dL_daxis = sign * g_world;
                if (axis == 0) {
                    dL_drotation.m11 += dL_daxis.x;
                    dL_drotation.m21 += dL_daxis.y;
                    dL_drotation.m31 += dL_daxis.z;
                } else if (axis == 1) {
                    dL_drotation.m12 += dL_daxis.x;
                    dL_drotation.m22 += dL_daxis.y;
                    dL_drotation.m32 += dL_daxis.z;
                } else {
                    dL_drotation.m13 += dL_daxis.x;
                    dL_drotation.m23 += dL_daxis.y;
                    dL_drotation.m33 += dL_daxis.z;
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
            const float dL_dq_norm_helper =
                qxx * dL_dqxx + qyy * dL_dqyy + qzz * dL_dqzz + qxy * dL_dqxy + qxz * dL_dqxz +
                qyz * dL_dqyz + qrx * dL_dqrx + qry * dL_dqry + qrz * dL_dqrz;
            const float4 dL_draw_rotation =
                make_float4(qx * dL_dqrx + qy * dL_dqry + qz * dL_dqrz - qr * dL_dq_norm_helper,
                            2.0f * qx * dL_dqxx + qy * dL_dqxy + qz * dL_dqxz + qr * dL_dqrx - qx * dL_dq_norm_helper,
                            2.0f * qy * dL_dqyy + qx * dL_dqxy + qz * dL_dqyz + qr * dL_dqry - qy * dL_dq_norm_helper,
                            2.0f * qz * dL_dqzz + qx * dL_dqxz + qy * dL_dqyz + qr * dL_dqrz - qz * dL_dq_norm_helper) *
                inv_q_norm_sq;
            const float4 clamped_rotation = clamp_grad4(dL_draw_rotation);
            float rotation_grads[4] = {
                clamped_rotation.x, clamped_rotation.y, clamped_rotation.z, clamped_rotation.w};

            float mean_step_ratio = 1.0f;
            if (geom_adam.per_splat_mean_step &&
                geom_adam.mean_step_far_mask != nullptr &&
                primitive_idx < static_cast<uint>(geom_adam.mean_step_far_mask_n) &&
                geom_adam.mean_step_far_mask[primitive_idx] &&
                geom_adam.scaling.param != nullptr) {
                mean_step_ratio = lfs::training::per_splat_mean_step_ratio(
                    raw_scale.x, raw_scale.y, raw_scale.z, geom_adam.mean_step_median_extent);
            }
            lfs::training::ray_depth::Proposal ray_depth;
            const bool ray_depth_row =
                geom_adam.ray_depth_state != nullptr &&
                geom_adam.means.param != nullptr &&
                geom_adam.scaling.param != nullptr &&
                geom_adam.rotation.param != nullptr &&
                primitive_idx < static_cast<uint>(geom_adam.ray_depth_n);
            if (ray_depth_row) {
                const uint b3 = primitive_idx * 3u;
                const float lr_scale = lfs::training::ray_depth::row_lr_scale(
                    primitive_idx, geom_adam.means.frozen_mask, geom_adam.means.frozen_mask_size,
                    geom_adam.means.frozen_lr_scale, geom_adam.means.crop_damping_mask,
                    geom_adam.means.crop_damping_mask_size, geom_adam.means.cropbox_lr_scale);
                const float camera_xyz[3] = {cam_pos.x, cam_pos.y, cam_pos.z};
                ray_depth = lfs::training::ray_depth::propose(
                    geom_adam.ray_depth_state + primitive_idx * lfs::training::ray_depth::kStateStride,
                    geom_adam.means.param + b3, geom_adam.scaling.param + b3,
                    geom_adam.rotation.param + primitive_idx * 4u, camera_xyz, mean_grads,
                    geom_adam.ray_depth_lr * lr_scale,
                    lfs::training::ray_depth::kSidewaysSteps * geom_adam.ray_depth_mean_lr * lr_scale *
                        mean_step_ratio);
            }

            adam_step_row_sparse<3>(
                mean_grads, geom_adam.means, primitive_idx, bidx,
                geom_adam.beta1, geom_adam.beta2, geom_adam.eps, rebound, 0,
                mean_step_ratio);
            adam_step_row_sparse<4>(
                rotation_grads, geom_adam.rotation, primitive_idx, bidx,
                geom_adam.beta1, geom_adam.beta2, geom_adam.eps, rebound, 3,
                1.0f, geom_lazy, geom_missed);
            adam_step_row_sparse<3>(
                scale_grads, geom_adam.scaling, primitive_idx, bidx,
                geom_adam.beta1, geom_adam.beta2, geom_adam.eps, rebound, 2);
            if (geom_adam.geom_last_step != nullptr && geom_adam.current_step > 0u &&
                primitive_idx < static_cast<uint>(geom_adam.geom_last_step_n)) {
                geom_adam.geom_last_step[primitive_idx] = encode_lazy_step(geom_adam.current_step);
            }

            if (ray_depth_row) {
                const float camera_xyz[3] = {cam_pos.x, cam_pos.y, cam_pos.z};
                lfs::training::ray_depth::commit(
                    geom_adam.ray_depth_state + primitive_idx * lfs::training::ray_depth::kStateStride,
                    geom_adam.means.param + primitive_idx * 3u, geom_adam.scaling.param + primitive_idx * 3u,
                    camera_xyz, ray_depth.dt, ray_depth.bperp2 >= 0.0f, geom_adam.ray_depth_far_limit);
            }
            apply_mrnf_noise_fused(geom_adam.mrnf_noise, geom_adam.means.param, geom_adam.opacity.param, primitive_idx);
        }

        if constexpr (REPORT_REG_LOSS) {
            if (geom_adam.dc_reg_loss_out != nullptr) {
                const float block_sum = lfs::core::warp_ops::block_reduce_sum(dc_loss_local);
                if (threadIdx.x == 0 && block_sum != 0.0f)
                    atomicAdd(geom_adam.dc_reg_loss_out, block_sum);
            }
            if (geom_adam.scale_reg_loss_out != nullptr) {
                const float block_sum = lfs::core::warp_ops::block_reduce_sum(scale_loss_local);
                if (threadIdx.x == 0 && block_sum != 0.0f)
                    atomicAdd(geom_adam.scale_reg_loss_out, block_sum);
            }
            if (geom_adam.erank_reg_loss_out != nullptr) {
                const float block_sum = lfs::core::warp_ops::block_reduce_sum(erank_loss_local);
                if (threadIdx.x == 0 && block_sum != 0.0f)
                    atomicAdd(geom_adam.erank_reg_loss_out, block_sum);
            }
        }
    }

    // Cold helper kept out-of-line so SH3 basis evaluation does not inflate register
    // count in adam_decay_invisible_geom_cu (only invoked when |sh0| > 0.5/kSh0).
    template <int ACTIVE_SH_BASES>
    __device__ __noinline__ uint eval_invisible_sh_colour_reg_mask(
        const float* __restrict__ sh0_param,
        const float4* __restrict__ sh_coefficients_rest,
        const float3* __restrict__ means,
        const float3* __restrict__ cam_position,
        const uint primitive_idx,
        const uint sh_layout_slots,
        const float2* __restrict__ shN_value_bounds,
        const uint shN_value_n_cells,
        const uint shN_value_bits) {
        const float3 colour = convert_sh_to_color(
            reinterpret_cast<const float3*>(sh0_param), sh_coefficients_rest,
            means[primitive_idx], cam_position[0], primitive_idx, ACTIVE_SH_BASES,
            sh_layout_slots, shN_value_bounds,
            shN_value_bounds != nullptr ? shN_value_n_cells : 0u,
            shN_value_bits == 16u ? 16u : 0u);
        return (colour.x >= 0.0f ? 1u : 0u) |
               (colour.y >= 0.0f ? 2u : 0u) |
               (colour.z >= 0.0f ? 4u : 0u);
    }

    // Exact 16-bit joint Adam momentum decay for non-contributing geometry + sh0 rows.
    // Skips reading primitive_work_indices on inactive blocks via active_block_flag,
    // skips s_q==0 cells, skips param load/store when m==0, and uses exact log-space
    // sqrt(g2) cancellation for normal cells.
    template <int ACTIVE_SH_BASES, bool REPORT_REG_LOSS = false>
    __global__ void __launch_bounds__(config::block_size_preprocess_backward, 6) adam_decay_invisible_geom_cu(
        const float3* __restrict__ means,
        const float4* __restrict__ sh_coefficients_rest,
        const float3* __restrict__ cam_position,
        const std::uint8_t* __restrict__ active_block_flag,
        const uint* __restrict__ primitive_work_indices,
        const std::uint8_t* __restrict__ primitive_touched,
        const ProjectedRecord* __restrict__ projected,
        const uint n_primitives,
        const uint sh_layout_slots,
        const SparseInvisGeomArgs invis_adam,
        const SparseReboundState rebound,
        const float2* __restrict__ shN_value_bounds,
        const uint shN_value_n_cells,
        const uint shN_value_bits) {
        const uint primitive_idx = blockIdx.x * blockDim.x + threadIdx.x;
        const bool in_range = primitive_idx < n_primitives;
        if constexpr (!REPORT_REG_LOSS) {
            if (!in_range)
                return;
        }
        const bool block_has_visible =
            (active_block_flag == nullptr || active_block_flag[blockIdx.x] != 0u);
        const uint work_idx =
            (in_range && block_has_visible) ? primitive_work_indices[primitive_idx] : 0xffffffffu;
        const bool visible = in_range && work_idx != 0xffffffffu;
        const bool contributed =
            visible && (primitive_touched == nullptr || primitive_touched[work_idx] != 0u);
        if constexpr (!REPORT_REG_LOSS) {
            if (contributed)
                return;
        }

        float dc_loss_local = 0.0f;
        float scale_loss_local = 0.0f;
        float erank_loss_local = 0.0f;

        if (in_range && !contributed) {
            // 1. sh0 (+ cold dc_reg when |coeff| > 0.5/kSh0).
            bool has_dc_reg = false;
            float3 dc_grads = make_float3(0.0f, 0.0f, 0.0f);
            if (invis_adam.dc_reg_weight > 0.0f && invis_adam.sh0.param != nullptr) {
                constexpr float kSh0 = 0.28209479177387814f;
                const float limit = 0.5f / kSh0;
                const int stride = invis_adam.sh0.n_attributes > 0 ? invis_adam.sh0.n_attributes : 3;
                const float* const coeff =
                    invis_adam.sh0.param + static_cast<size_t>(primitive_idx) * static_cast<size_t>(stride);
                const float c0 = coeff[0];
                const float c1 = coeff[1];
                const float c2 = coeff[2];
                if (__builtin_expect(fabsf(c0) > limit || fabsf(c1) > limit || fabsf(c2) > limit, 0)) {
                    uint colour_reg_mask = 7u;
                    if (visible) {
                        const float3 colour = make_float3(projected[work_idx].color);
                        colour_reg_mask = (colour.x >= 0.0f ? 1u : 0u) |
                                          (colour.y >= 0.0f ? 2u : 0u) |
                                          (colour.z >= 0.0f ? 4u : 0u);
                    } else {
                        colour_reg_mask = eval_invisible_sh_colour_reg_mask<ACTIVE_SH_BASES>(
                            invis_adam.sh0.param, sh_coefficients_rest, means, cam_position,
                            primitive_idx, sh_layout_slots,
                            shN_value_bounds, shN_value_n_cells, shN_value_bits);
                    }
                    const float dc_grad_scale = 2.0f * invis_adam.dc_reg_weight / 3.0f;
                    const float reported_loss_weight =
                        invis_adam.dc_reg_weight / (3.0f * static_cast<float>(n_primitives));
                    auto clamp_delta = [limit](const float v) {
                        return (v > limit) ? (v - limit) : ((v < -limit) ? (v + limit) : 0.0f);
                    };
                    const float d0 = clamp_delta(c0);
                    const float d1 = clamp_delta(c1);
                    const float d2 = clamp_delta(c2);
                    if (colour_reg_mask & 1u)
                        dc_grads.x = dc_grad_scale * d0;
                    if (colour_reg_mask & 2u)
                        dc_grads.y = dc_grad_scale * d1;
                    if (colour_reg_mask & 4u)
                        dc_grads.z = dc_grad_scale * d2;
                    has_dc_reg = (dc_grads.x != 0.0f || dc_grads.y != 0.0f || dc_grads.z != 0.0f);
                    if constexpr (REPORT_REG_LOSS)
                        dc_loss_local += reported_loss_weight * (d0 * d0 + d1 * d1 + d2 * d2);
                }
            }
            // Lazy rows skip the zero-gradient rotation/sh0 coast; rows whose sh0 is under
            // dc_reg pressure are first caught up and then decayed exactly this step.
            bool dense_rot_sh0 = true;
            if (invis_adam.geom_last_step != nullptr && invis_adam.step_seed > 0u &&
                primitive_idx < static_cast<uint>(invis_adam.geom_last_step_n)) {
                const uint32_t missed = lazy_missed_steps(
                    invis_adam.geom_last_step, invis_adam.geom_last_step_n, primitive_idx, invis_adam.step_seed);
                const bool tracked = invis_adam.geom_last_step[primitive_idx] != 0u;
                if (tracked && !has_dc_reg) {
                    dense_rot_sh0 = false;
                } else {
                    if (missed > 0u) {
                        const LazyAdamFactors lazy =
                            compute_lazy_adam_factors(missed, invis_adam.beta1, invis_adam.beta2);
                        constexpr float kSh0Lazy = 0.28209479177387814f;
                        const float dc_scale =
                            (invis_adam.dc_reg_weight > 0.0f) ? (2.0f * invis_adam.dc_reg_weight / 3.0f) : 0.0f;
                        adam_catchup_row_16_lazy<4, false>(
                            invis_adam.rotation, primitive_idx, blockIdx.x, invis_adam.beta2,
                            invis_adam.eps, lazy, missed, invis_adam.step_seed, 3);
                        adam_catchup_row_16_lazy<3, true>(
                            invis_adam.sh0, primitive_idx, blockIdx.x, invis_adam.beta2,
                            invis_adam.eps, lazy, missed, invis_adam.step_seed, 1, 0.5f / kSh0Lazy, dc_scale);
                    }
                    invis_adam.geom_last_step[primitive_idx] = encode_lazy_step(invis_adam.step_seed);
                }
            }
            if (dense_rot_sh0) {
                if (__builtin_expect(has_dc_reg, 0)) {
                    adam_decay_row_16_invisible<3, false, true>(
                        invis_adam.sh0, primitive_idx, invis_adam.beta1, invis_adam.beta2, invis_adam.eps,
                        rebound, 1, invis_adam.step_seed, 1.0f, dc_grads);
                } else {
                    adam_decay_row_16_invisible<3, false, false>(
                        invis_adam.sh0, primitive_idx, invis_adam.beta1, invis_adam.beta2, invis_adam.eps,
                        rebound, 1, invis_adam.step_seed, 1.0f);
                }
            }

            // 2. means (reads scaling.param before scaling updates, matching dense order).
            float mean_step_ratio = 1.0f;
            if (invis_adam.per_splat_mean_step &&
                invis_adam.mean_step_far_mask != nullptr &&
                primitive_idx < static_cast<uint>(invis_adam.mean_step_far_mask_n) &&
                invis_adam.mean_step_far_mask[primitive_idx] &&
                invis_adam.scaling.param != nullptr) {
                const uint sb = primitive_idx * 3u;
                if (sb + 2u < static_cast<uint>(invis_adam.scaling.n_elements)) {
                    const float* const s = invis_adam.scaling.param + sb;
                    mean_step_ratio = lfs::training::per_splat_mean_step_ratio(
                        s[0], s[1], s[2], invis_adam.mean_step_median_extent);
                }
            }
            adam_decay_row_16_invisible<3, false>(
                invis_adam.means, primitive_idx, invis_adam.beta1, invis_adam.beta2, invis_adam.eps,
                rebound, 0, invis_adam.step_seed, mean_step_ratio);

            // 3. rotation.
            if (dense_rot_sh0) {
                adam_decay_row_16_invisible<4, false>(
                    invis_adam.rotation, primitive_idx, invis_adam.beta1, invis_adam.beta2, invis_adam.eps,
                    rebound, 3, invis_adam.step_seed);
            }

            // 4. scaling (+ screen_share hinge + optional loss reporting before update).
            if constexpr (REPORT_REG_LOSS) {
                if (invis_adam.scaling.param != nullptr && invis_adam.scaling.n_elements > 0) {
                    const uint sb = primitive_idx * 3u;
                    const float s0 = invis_adam.scaling.param[sb];
                    const float s1 = invis_adam.scaling.param[sb + 1u];
                    const float s2 = invis_adam.scaling.param[sb + 2u];
                    if (invis_adam.scale_reg_loss_out != nullptr && invis_adam.scale_reg_weight > 0.0f) {
                        const float inv_n = 1.0f / static_cast<float>(invis_adam.scaling.n_elements);
                        if (invis_adam.scale_reg_log) {
                            scale_loss_local = invis_adam.scale_reg_weight * 0.01f * inv_n /
                                               invis_adam.scale_reg_normalizer *
                                               (fmaxf(s0, -40.0f) + fmaxf(s1, -40.0f) + fmaxf(s2, -40.0f));
                        } else {
                            scale_loss_local = invis_adam.scale_reg_weight * inv_n *
                                               (expf(s0) + expf(s1) + expf(s2));
                        }
                    }
                    if (invis_adam.erank_reg_loss_out != nullptr && invis_adam.erank_reg_weight > 0.0f &&
                        invis_adam.scaling.n_primitives > 0) {
                        const auto penalty = lfs::training::effective_rank_penalty(s0, s1, s2);
                        erank_loss_local = (invis_adam.erank_reg_weight /
                                            static_cast<float>(invis_adam.scaling.n_primitives)) *
                                           penalty.value;
                    }
                }
            }
            adam_decay_row_16_invisible<3, true>(
                invis_adam.scaling, primitive_idx, invis_adam.beta1, invis_adam.beta2, invis_adam.eps,
                rebound, 2, invis_adam.step_seed);

            // 5. opacity.
            adam_decay_row_16_invisible<1, false>(
                invis_adam.opacity, primitive_idx, invis_adam.beta1, invis_adam.beta2, invis_adam.eps,
                rebound, 4, invis_adam.step_seed);

            // 6. MRNF mean noise on the updated row (contributing rows take it in sparse geometry).
            apply_mrnf_noise_fused(invis_adam.mrnf_noise, invis_adam.means.param, invis_adam.opacity.param, primitive_idx);
        }

        if constexpr (REPORT_REG_LOSS) {
            if (invis_adam.dc_reg_loss_out != nullptr) {
                const float block_sum = lfs::core::warp_ops::block_reduce_sum(dc_loss_local);
                if (threadIdx.x == 0 && block_sum != 0.0f)
                    atomicAdd(invis_adam.dc_reg_loss_out, block_sum);
            }
            if (invis_adam.scale_reg_loss_out != nullptr) {
                const float block_sum = lfs::core::warp_ops::block_reduce_sum(scale_loss_local);
                if (threadIdx.x == 0 && block_sum != 0.0f)
                    atomicAdd(invis_adam.scale_reg_loss_out, block_sum);
            }
            if (invis_adam.erank_reg_loss_out != nullptr) {
                const float block_sum = lfs::core::warp_ops::block_reduce_sum(erank_loss_local);
                if (threadIdx.x == 0 && block_sum != 0.0f)
                    atomicAdd(invis_adam.erank_reg_loss_out, block_sum);
            }
        }
    }

    // Flush pending lazy rotation/sh0 coast steps before full-model consumers
    // (refine, Morton reorder, evaluation, save).
    __global__ void __launch_bounds__(config::block_size_preprocess_backward, 4) flush_lazy_geom_cu(
        const uint n_primitives,
        const SparseInvisGeomArgs geom) {
        const uint primitive_idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (primitive_idx >= n_primitives)
            return;
        const uint32_t dt = lazy_flush_steps(
            geom.geom_last_step, geom.geom_last_step_n, primitive_idx, geom.step_seed);
        if (dt == 0u)
            return;
        const LazyAdamFactors lazy = compute_lazy_adam_factors(dt, geom.beta1, geom.beta2);
        constexpr float kSh0 = 0.28209479177387814f;
        const float dc_scale = (geom.dc_reg_weight > 0.0f) ? (2.0f * geom.dc_reg_weight / 3.0f) : 0.0f;
        adam_catchup_row_16_lazy<4, false>(
            geom.rotation, primitive_idx, blockIdx.x, geom.beta2, geom.eps,
            lazy, dt, geom.step_seed, 3);
        adam_catchup_row_16_lazy<3, true>(
            geom.sh0, primitive_idx, blockIdx.x, geom.beta2, geom.eps,
            lazy, dt, geom.step_seed, 1, 0.5f / kSh0, dc_scale);
        geom.geom_last_step[primitive_idx] = encode_lazy_step(geom.step_seed);
    }

    // Flush pending lazy coast + SH-rest reg steps for 8-bit joint Adam SH-rest rows.
    template <int ACTIVE_SH_BASES, bool VALUE_Q16>
    __global__ void __launch_bounds__(config::block_size_preprocess_backward, 3) flush_lazy_shN_cu(
        const uint n_primitives,
        const uint sh_layout_slots,
        const SparseShAdamArgs sh_adam) {
        const uint primitive_idx = blockIdx.x * blockDim.x + threadIdx.x;
        const bool in_range = primitive_idx < n_primitives;
        const uint32_t dt =
            in_range ? lazy_flush_steps(sh_adam.lazy_last_step, sh_adam.lazy_last_step_n,
                                        primitive_idx, sh_adam.current_step)
                     : 0u;
        if (__syncthreads_or(dt != 0u) == 0)
            return;

        __shared__ float s_sqrt_g2_lut[256];
        {
            using C8 = lfs::training::joint_adam::DeviceCodec<8>;
            float4* const bptr = reinterpret_cast<float4*>(sh_adam.shN.joint_bounds) + blockIdx.x;
            float4 mm_lut = *bptr;
            if (mm_lut.x == 0.0f && mm_lut.y == 0.0f && mm_lut.z == 0.0f && mm_lut.w == 0.0f) {
                mm_lut = make_float4(-kSparseMinU8, kSparseMinU8 * kQ8ZeroExactHiRatio, 0.0f, kSparseMinS8);
                if (threadIdx.x == 0)
                    *bptr = mm_lut;
            }
            const float s_dec_lut = (mm_lut.w - mm_lut.z) * C8::kInvQMax;
            s_sqrt_g2_lut[threadIdx.x] =
                C8::inverse_sqrt_g2(mm_lut.z + static_cast<float>(threadIdx.x) * s_dec_lut);
        }
        __syncthreads();

        if (dt == 0u)
            return;

        const LazyAdamFactors lazy = compute_lazy_adam_factors(dt, sh_adam.beta1, sh_adam.beta2);
        shN_flush_lazy_sparse<ACTIVE_SH_BASES, VALUE_Q16>(
            sh_adam.shN, primitive_idx, blockIdx.x, sh_layout_slots,
            sh_adam.sh_rest_reg_weight, sh_adam.eps, lazy, s_sqrt_g2_lut);
        if (sh_adam.lazy_last_step != nullptr && sh_adam.current_step > 0u &&
            primitive_idx < static_cast<uint>(sh_adam.lazy_last_step_n)) {
            sh_adam.lazy_last_step[primitive_idx] = encode_lazy_step(sh_adam.current_step);
        }
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
    // distinct lanes, so they leave in one atomic instruction). Splats whose
    // coverage hits only one 8x4 sub-tile are dual-issued across the two
    // half-warps in exact per-pixel back-to-front order without offset=16 shuffles.
    // ---------------------------------------------------------------------------
    template <DensificationType DENSIFICATION_TYPE, bool DEPTH_CHANNEL, bool NORMAL_CHANNEL, bool BLEND_PEAK, bool EDGE_CHANNEL, int WARP_CULL_MODE = 0, bool COLOR_VARIANCE = false>
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
        float* __restrict__ densification_weight,
        float* __restrict__ densification_error,
        const float* __restrict__ densification_error_map,
        const float* __restrict__ edge_weight_map,
        float* __restrict__ edge_score_out,
        float* __restrict__ blend_peak_out,
        std::uint8_t* __restrict__ primitive_touched,
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
        const unsigned top_bit = 1u << top_region;
        const unsigned bot_bit = 1u << (top_region + 2u);
        const unsigned warp_regions = top_bit | bot_bit;

        const uint n_pixels = width * height;
        const uint pixel_x = (tile_idx % grid_width) * static_cast<uint>(config::tile_width) + tile_x;
        const uint pixel_y0 = (tile_idx / grid_width) * static_cast<uint>(config::tile_height) + tile_y0;
        const float center_x = __uint2float_rn(pixel_x) + 0.5f;
        const float center_y0 = __uint2float_rn(pixel_y0) + 0.5f;

        // Reverse-walk state of the thread's two pixels (rows y0 and y0 + 1).
        bool inside[2];
        uint pixel_idx[2], last[2];
        float T[2], grad_T[2], pixel_error[2], variance_scale[2];
        [[maybe_unused]] float grad_d[2];
        [[maybe_unused]] float edge_weight[2];
        float3 grad_c[2], grad_n[2], target[2];
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
            if constexpr (DEPTH_CHANNEL) {
                grad_d[k] = (inside[k] && grad_depth_map != nullptr) ? grad_depth_map[pixel_idx[k]] : 0.0f;
            } else {
                grad_d[k] = 0.0f;
            }
            grad_n[k] = make_float3(0.0f);
            if constexpr (NORMAL_CHANNEL) {
                if (inside[k] && grad_normal_map != nullptr) {
                    grad_n[k] = make_float3(grad_normal_map[pixel_idx[k]],
                                            grad_normal_map[n_pixels + pixel_idx[k]],
                                            grad_normal_map[2 * n_pixels + pixel_idx[k]]);
                }
            }
            pixel_error[k] = 1.0f;
            if constexpr (DENSIFICATION_TYPE == DensificationType::MCMC) {
                pixel_error[k] = inside[k] ? densification_error_map[pixel_idx[k]] : 1.0f;
            } else if constexpr (DENSIFICATION_TYPE != DensificationType::None) {
                if (densification_error_map != nullptr && inside[k])
                    pixel_error[k] = densification_error_map[pixel_idx[k]];
            }
            if constexpr (EDGE_CHANNEL) {
                edge_weight[k] = (edge_weight_map != nullptr && inside[k]) ? edge_weight_map[pixel_idx[k]] : 0.0f;
            } else {
                edge_weight[k] = 0.0f;
            }
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
        constexpr int kUNormal = kUDepth + (DEPTH_CHANNEL ? 1 : 0);
        // MRNF sums weight and error; MCMC reads the error row alone.
        constexpr int kDensificationFields = DENSIFICATION_TYPE == DensificationType::MRNF   ? 2
                                             : DENSIFICATION_TYPE == DensificationType::MCMC ? 1
                                                                                             : 0;
        constexpr int kUDensification = kUNormal + (NORMAL_CHANNEL ? 3 : 0);
        constexpr int kUEdge = kUDensification + kDensificationFields;
        constexpr int kU = kUEdge + (EDGE_CHANNEL ? 1 : 0);
        // Output field ids of the accumulation targets.
        constexpr int kFieldMeanX = 0, kFieldMeanY = 1, kFieldConicX = 2, kFieldConicY = 3, kFieldConicZ = 4;
        constexpr int kFieldDepth = 5, kFieldOpacity = 6, kFieldColor = 7, kFieldNormal = 10;
        constexpr int kFieldDensification = kFieldNormal + (NORMAL_CHANNEL ? 3 : 0);
        constexpr int kFieldEdge = kFieldDensification + kDensificationFields;
        auto u_field = [](const int u) {
            if (u == kUConicZ)
                return kFieldConicZ;
            if (u < kUDepth)
                return kFieldColor + u - kUColor;
            if constexpr (DEPTH_CHANNEL) {
                if (u == kUDepth)
                    return kFieldDepth;
            }
            if constexpr (NORMAL_CHANNEL) {
                if (u >= kUNormal && u < kUDensification)
                    return kFieldNormal + u - kUNormal;
            }
            if (u < kUEdge)
                return kFieldDensification + u - kUDensification;
            return kFieldEdge;
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
            else if (field == kFieldDepth) {
                if constexpr (DEPTH_CHANNEL)
                    target_ptr = grad_depth + work;
            } else if (field == kFieldOpacity)
                target_ptr = grad_compensated_opacity + work;
            else if (field < kFieldNormal)
                target_ptr = reinterpret_cast<float*>(grad_color) + 3 * row + field - kFieldColor;
            else if (field < kFieldDensification) {
                if constexpr (NORMAL_CHANNEL)
                    target_ptr = reinterpret_cast<float*>(grad_normal) + 3 * row + field - kFieldNormal;
            } else if (field < kFieldEdge)
                target_ptr = (DENSIFICATION_TYPE == DensificationType::MRNF && field == kFieldDensification
                                  ? densification_weight
                                  : densification_error) +
                             primitive;
            else if constexpr (EDGE_CHANNEL)
                target_ptr = edge_score_out != nullptr ? edge_score_out + primitive : nullptr;
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
        __shared__ float s_depth[DEPTH_CHANNEL ? config::block_size_blend_backward : 1];
        __shared__ float3 s_normal[NORMAL_CHANNEL ? config::block_size_blend_backward : 1];
        const bool stage_depth = DEPTH_CHANNEL && grad_depth_map != nullptr;
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
            float depth = 0.0f;
            if constexpr (DEPTH_CHANNEL)
                depth = stage_depth ? s_depth[slot] : 0.0f;
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
                if constexpr (BLEND_PEAK)
                    sums.peak = fmaxf(sums.peak, w);
                const float3 pixel_grad_color = accepted ? grad_c[k] : make_float3(0.0f);
                float pixel_grad_depth = 0.0f;
                if constexpr (DEPTH_CHANNEL)
                    pixel_grad_depth = accepted ? grad_d[k] : 0.0f;
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
                float contrib_dot = dot(color, pixel_grad_color);
                if constexpr (DEPTH_CHANNEL)
                    contrib_dot += depth * pixel_grad_depth;
                if constexpr (NORMAL_CHANNEL)
                    contrib_dot += normal_dot_grad;
                if constexpr (COLOR_VARIANCE)
                    contrib_dot += variance_cost;
                const float dL_dalpha = T_before * (contrib_dot - grad_T_after);
                h[k] = accepted && !saturated ? -alpha * dL_dalpha : 0.0f;
                sums.u[kUColor] += dL_dcolor.x;
                sums.u[kUColor + 1] += dL_dcolor.y;
                sums.u[kUColor + 2] += dL_dcolor.z;
                if constexpr (DEPTH_CHANNEL)
                    sums.u[kUDepth] += w * pixel_grad_depth;
                if constexpr (NORMAL_CHANNEL) {
                    const float3 pixel_grad_normal = accepted ? grad_n[k] : make_float3(0.0f);
                    sums.u[kUNormal] += w * pixel_grad_normal.x;
                    sums.u[kUNormal + 1] += w * pixel_grad_normal.y;
                    sums.u[kUNormal + 2] += w * pixel_grad_normal.z;
                }
                if constexpr (DENSIFICATION_TYPE == DensificationType::MRNF) {
                    sums.u[kUDensification] += w;
                    sums.u[kUDensification + 1] += w * pixel_error[k];
                } else if constexpr (DENSIFICATION_TYPE == DensificationType::MCMC) {
                    sums.u[kUDensification] += w * pixel_error[k];
                }
                if constexpr (EDGE_CHANNEL)
                    sums.u[kUEdge] += w * edge_weight[k];
                float next_grad_T = dot(pixel_grad_color, alpha * color) + grad_T_after * (1.0f - alpha);
                if constexpr (DEPTH_CHANNEL)
                    next_grad_T += alpha * depth * pixel_grad_depth;
                if constexpr (NORMAL_CHANNEL)
                    next_grad_T += alpha * normal_dot_grad;
                if constexpr (COLOR_VARIANCE)
                    next_grad_T += alpha * variance_cost;
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
        const bool upper = (lane & 16u) != 0u;
        for (int end = traversal_count; end > 0; end -= 32) {
            const int entry = end - 1 - static_cast<int>(lane);
            const uint slot = warp_base + lane;
            bool hit_top = false;
            bool hit_bot = false;
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
                        hit_top = (WARP_CULL_MODE == 1) || ((coverage & top_bit) != 0u);
                        hit_bot = (WARP_CULL_MODE == 1) || ((coverage & bot_bit) != 0u);
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
                        if constexpr (DEPTH_CHANNEL) {
                            if (stage_depth)
                                s_depth[slot] = primitive_depths[work_idx];
                        }
                        if constexpr (NORMAL_CHANNEL)
                            s_normal[slot] = primitive_normals[work_idx];
                    }
                }
            }
            unsigned rem_top = __ballot_sync(0xffffffffu, hit_top);
            unsigned rem_bot = __ballot_sync(0xffffffffu, hit_bot);
            if constexpr (WARP_CULL_MODE == 2) {
                rem_top = 0u;
                rem_bot = 0u;
            }
            __syncwarp();

            // Walk staged splats back to front. Splats hitting both 8x4 sub-tiles run
            // on the full warp; single-subtile splats dual-issue on the two half-warps
            // with 16-lane reductions so neither half waits on sibling-only splats.
            while ((rem_top | rem_bot) != 0u) {
                const uint t_bit = rem_top != 0u ? static_cast<uint>(__ffs(rem_top) - 1) : 32u;
                const uint b_bit = rem_bot != 0u ? static_cast<uint>(__ffs(rem_bot) - 1) : 32u;
                if (t_bit == b_bit) {
                    const unsigned bit_mask = 1u << t_bit;
                    rem_top &= ~bit_mask;
                    rem_bot &= ~bit_mask;
                    const uint slot = warp_base + t_bit;
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
                        if constexpr (BLEND_PEAK) {
                            const float peak = reduce_max<16>(sums.peak);
                            if (lane == 0u && peak > 0.0f)
                                atomicMax(reinterpret_cast<int*>(blend_peak_out + s_primitive[slot]), __float_as_int(peak));
                        }
                    }
                } else {
                    const bool t_run = t_bit < 32u && ((rem_bot & (1u << t_bit)) == 0u);
                    const bool b_run = b_bit < 32u && ((rem_top & (1u << b_bit)) == 0u);
                    if (t_run)
                        rem_top &= ~(1u << t_bit);
                    if (b_run)
                        rem_bot &= ~(1u << b_bit);
                    const bool half_active = upper ? b_run : t_run;
                    const uint my_slot = warp_base + (upper ? (b_run ? b_bit : 0u) : (t_run ? t_bit : 0u));
                    const SplatSums sums = half_active ? walk_splat(my_slot) : SplatSums{};
                    const unsigned contrib_ballot = __ballot_sync(0xffffffffu, half_active && sums.contributed);
                    if (contrib_ballot != 0u) {
                        const bool my_half_contrib = (contrib_ballot & (upper ? 0xffff0000u : 0x0000ffffu)) != 0u;
                        float h = sums.h;
                        float h_dy = sums.h_dy;
                        float folded[kU];
#pragma unroll
                        for (int u = 0; u < kU; ++u)
                            folded[u] = sums.u[u];
                        halving_step<kU, 8>(folded, lane);
                        h += __shfl_xor_sync(0xffffffffu, h, 8);
                        h_dy += __shfl_xor_sync(0xffffffffu, h_dy, 8);
                        float values[kPair];
                        if (my_half_contrib) {
                            const auto terms = column_terms(my_slot, h, h_dy);
                            values[0] = second_row_pair ? terms.mean_y : terms.mean_x;
                            values[1] = second_row_pair ? terms.conic_y : terms.conic_x;
                            values[2] = second_row_pair ? 0.0f : terms.opacity;
                        } else {
                            values[0] = 0.0f;
                            values[1] = 0.0f;
                            values[2] = 0.0f;
                        }
#pragma unroll
                        for (int u = 0; u < kUFolded; ++u)
                            values[3 + u] = folded[u];
                        halving_step<kPair, 4>(values, lane);
                        halving_step<(kPair + 1) / 2, 2>(values, lane);
                        halving_step<((kPair + 1) / 2 + 1) / 2, 1>(values, lane);
                        if (my_half_contrib) {
#pragma unroll
                            for (int f = 0; f < kPairOut; ++f)
                                store_field(pair_fields[f], values[f], s_primitive[my_slot], s_work[my_slot]);
                        }
                        if constexpr (BLEND_PEAK) {
                            const float peak = reduce_max<8>(half_active ? sums.peak : 0.0f);
                            if (my_half_contrib && (lane & 15u) == 0u && peak > 0.0f)
                                atomicMax(reinterpret_cast<int*>(blend_peak_out + s_primitive[my_slot]), __float_as_int(peak));
                        }
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

    __global__ void prepare_tile_lpt_keys_bwd_cu(
        const uint2* __restrict__ tile_instance_ranges,
        uint* __restrict__ tile_counts,
        uint* __restrict__ tile_ids,
        const uint n_tiles) {
        const uint idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx < n_tiles) {
            const uint2 r = tile_instance_ranges[idx];
            tile_counts[idx] = (r.y > r.x) ? (r.y - r.x) : 0u;
            tile_ids[idx] = idx;
        }
    }

    // ---------------------------------------------------------------------------
    // blend_backward_subtile_cu: 64-thread (2-warp) sub-tile, 1-pixel-per-thread walk.
    //
    // Four 64-thread blocks cover each 16x16 tile (two adjacent 8x4 sub-tiles
    // per block matching the 8 bits of contributor_masks), eliminating block
    // barriers and cutting intra-block straggler idle time 4x. Compact 48 B/splat
    // per-warp staging (3 KB/block) fits 20 blocks/SM inside a 64 KB SMEM /
    // 64 KB L1 split (40 warps/SM = 83% occupancy, 48 regs/thread, 0 spills).
    // ---------------------------------------------------------------------------
    template <DensificationType DENSIFICATION_TYPE, bool BLEND_PEAK, bool EDGE_CHANNEL>
    __global__ void __launch_bounds__(64, 20) blend_backward_subtile_cu(
        const uint2* __restrict__ tile_instance_ranges,
        const uint* __restrict__ tile_order,
        const uint* __restrict__ instance_primitive_indices,
        const unsigned char* __restrict__ contributor_masks,
        const uint* __restrict__ contributor_positions,
        const uint* __restrict__ primitive_work_indices,
        const ProjectedRecord* __restrict__ projected,
        const float* __restrict__ grad_image,
        const BackgroundAlphaGradient background_grad,
        const uint* __restrict__ tile_n_contributions,
        const float* __restrict__ tile_final_transmittance,
        float2* __restrict__ grad_mean2d,
        float3* __restrict__ grad_conic,
        float* __restrict__ grad_compensated_opacity,
        float3* __restrict__ grad_color,
        float* __restrict__ densification_weight,
        float* __restrict__ densification_error,
        const float* __restrict__ densification_error_map,
        const float* __restrict__ edge_weight_map,
        float* __restrict__ edge_score_out,
        float* __restrict__ blend_peak_out,
        std::uint8_t* __restrict__ primitive_touched,
        FastGSForwardStatus* __restrict__ status,
        const uint n_instances,
        const uint n_primitives,
        const uint n_visible,
        const uint width,
        const uint height,
        const uint grid_width) {
        static_assert(config::tile_width == 16 && config::tile_height == 16, "subtile layout assumes 16x16 tiles");
        static_assert(config::warp_subtile_width == 8 && config::warp_subtile_height == 4,
                      "coverage bits address 8x4 regions");

        const uint tile_pos = blockIdx.x >> 2;
        const uint sub_base = (blockIdx.x & 3u) << 1;
        const uint tile_idx = (tile_order != nullptr) ? tile_order[tile_pos] : tile_pos;
        const uint thread_rank = threadIdx.x;
        const uint local_warp = thread_rank >> 5;
        const uint warp = sub_base + local_warp;
        const uint lane = thread_rank & 31u;

        const uint2 tile_instance_range = tile_instance_ranges[tile_idx];
        if (tile_instance_range.x > tile_instance_range.y || tile_instance_range.y > n_instances) {
            if (sub_base == 0u && thread_rank == 0u) {
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

        const uint local_x = lane & 7u;
        const uint local_y = lane >> 3;
        const uint tile_x = (warp & 1u) * 8u + local_x;
        const uint tile_y = (warp >> 1) * 4u + local_y;
        const unsigned subtile_bit = 1u << warp;

        const uint n_pixels = width * height;
        const uint origin_x = (tile_idx % grid_width) * 16u;
        const uint origin_y = (tile_idx / grid_width) * 16u;
        const uint pixel_x = origin_x + tile_x;
        const uint pixel_y = origin_y + tile_y;
        const bool inside = pixel_x < width && pixel_y < height;
        const uint pixel_idx = inside ? (width * pixel_y + pixel_x) : 0u;
        const uint pixel_rank = tile_y * 16u + tile_x;

        const float center_x = __uint2float_rn(pixel_x) + 0.5f;
        const float center_y0 = __uint2float_rn(pixel_y & ~1u) + 0.5f;
        const float dy_row_off = static_cast<float>(local_y & 1u);

        const uint last = inside ? tile_n_contributions[pixel_idx] : 0u;
        float T = inside ? tile_final_transmittance[tile_idx * config::block_size_blend + pixel_rank] : 1.0f;
        const float3 grad_c = inside ? make_float3(grad_image[pixel_idx],
                                                   grad_image[n_pixels + pixel_idx],
                                                   grad_image[2u * n_pixels + pixel_idx])
                                     : make_float3(0.0f);
        float grad_T = inside ? background_transmittance_grad(background_grad, grad_c, pixel_idx, n_pixels) : 0.0f;
        float pixel_error = 1.0f;
        if constexpr (DENSIFICATION_TYPE == DensificationType::MCMC) {
            pixel_error = inside ? densification_error_map[pixel_idx] : 1.0f;
        } else if constexpr (DENSIFICATION_TYPE == DensificationType::MRNF) {
            if (densification_error_map != nullptr && inside)
                pixel_error = densification_error_map[pixel_idx];
        }
        float edge_weight = 0.0f;
        if constexpr (EDGE_CHANNEL) {
            if (edge_weight_map != nullptr && inside)
                edge_weight = edge_weight_map[pixel_idx];
        }

#if __CUDA_ARCH__ >= 800
        const uint warp_last = __reduce_max_sync(0xffffffffu, last);
#else
        uint warp_last = last;
#pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1)
            warp_last = ::max(warp_last, __shfl_xor_sync(0xffffffffu, warp_last, offset));
#endif
        int traversal_count = 0;
        if (warp_last > 0u && tile_n_primitives > 0) {
            if (contributor_positions != nullptr) {
                int lo = 0;
                int hi = tile_n_primitives;
                while (lo < hi) {
                    const int mid = (lo + hi) >> 1;
                    if (contributor_positions[tile_instance_range.x + static_cast<uint>(mid)] < warp_last)
                        lo = mid + 1;
                    else
                        hi = mid;
                }
                traversal_count = lo;
            } else {
                traversal_count = ::min(tile_n_primitives, static_cast<int>(warp_last));
            }
        }
        if (sub_base == 0u && thread_rank == 0u && status)
            status->expected_count = static_cast<unsigned>(traversal_count);
        if (traversal_count <= 0)
            return;

        constexpr int kUConicZ = 0;
        constexpr int kUColor = 1;
        constexpr int kDensFields = DENSIFICATION_TYPE == DensificationType::MRNF   ? 2
                                    : DENSIFICATION_TYPE == DensificationType::MCMC ? 1
                                                                                    : 0;
        constexpr int kUDens = 4;
        constexpr int kUEdge = kUDens + kDensFields;
        constexpr int kU = kUEdge + (EDGE_CHANNEL ? 1 : 0);

        constexpr int kFieldMeanX = 0, kFieldMeanY = 1, kFieldConicX = 2, kFieldConicY = 3, kFieldConicZ = 4;
        constexpr int kFieldOpacity = 5, kFieldColor = 6;
        constexpr int kFieldDens = 9;
        constexpr int kFieldEdge = kFieldDens + kDensFields;

        constexpr int kUVertical = (kU + 3) / 4;
        constexpr int kSingle = 2 + kUVertical;
        static_assert((kSingle + 7) / 8 == 1, "each lane holds at most one scatter field");

        const int out_index = halving_index<kSingle, 4u, 3>(lane, 0);
        int out_field = -1;
        if (out_index == 0) {
            out_field = local_y == 0u ? kFieldMeanX : local_y == 1u ? kFieldMeanY
                                                  : local_y == 2u   ? kFieldConicX
                                                                    : kFieldConicY;
        } else if (out_index == 1) {
            out_field = local_y == 0u ? kFieldOpacity : -1;
        } else if (out_index > 1) {
            const int u = halving_index<kU, 16u, 2>(lane, out_index - 2);
            if (u == kUConicZ)
                out_field = kFieldConicZ;
            else if (u >= kUColor && u < kUDens)
                out_field = kFieldColor + (u - kUColor);
            else if (u >= kUDens && u < kUEdge)
                out_field = kFieldDens + (u - kUDens);
            else if (u >= kUEdge)
                out_field = kFieldEdge;
        }

        float* out_base = nullptr;
        uint out_meta = 0u; // bits 0..1: stride (1..3), bit 2: uses_prim, bits 3..5: color_bit
        if (out_field >= 0) {
            if (out_field < kFieldConicX) {
                out_base = reinterpret_cast<float*>(grad_mean2d) + out_field;
                out_meta = 2u;
            } else if (out_field < kFieldOpacity) {
                out_base = reinterpret_cast<float*>(grad_conic) + (out_field - kFieldConicX);
                out_meta = 3u;
            } else if (out_field == kFieldOpacity) {
                out_base = grad_compensated_opacity;
                out_meta = 1u;
            } else if (out_field < kFieldDens) {
                const uint ch = static_cast<uint>(out_field - kFieldColor);
                out_base = reinterpret_cast<float*>(grad_color) + ch;
                out_meta = 3u | ((1u << ch) << 3);
            } else if (out_field < kFieldEdge) {
                out_base = (DENSIFICATION_TYPE == DensificationType::MRNF && out_field == kFieldDens)
                               ? densification_weight
                               : densification_error;
                out_meta = 1u | 4u;
            } else if constexpr (EDGE_CHANNEL) {
                out_base = edge_score_out;
                out_meta = 1u | 4u;
            }
        }

        // Compact 48 B/splat per-warp staging: 64 * 48 B = 3 KB per 2-warp block.
        __shared__ uint4 s_meta[64];  // {primitive_idx, work_idx, pos|(factor<<29), float_as_uint(color.z)}
        __shared__ float4 s_geom[64]; // {mean2d.x, mean2d.y, opacity, color.x}
        __shared__ float4 s_cb2[64];  // {cb2.x, cb2.y, cb2.z, color.y}
        const uint warp_base = local_warp * 32u;

        constexpr float kInvDiag = -1.3862943611198906f;  // -2 * ln(2)
        constexpr float kInvCross = -0.6931471805599453f; // -ln(2)

        // Hit entries gathered across consecutive coverage chunks, back to front.
        __shared__ int s_hit_entry[64];
        const unsigned lanes_below = (1u << lane) - 1u;
        int end = traversal_count;
        while (end > 0) {
            // Scan up to four 32-entry coverage chunks in one load wave and compact their
            // hits (at most 32) so a single parallel wave gathers the projected records.
            unsigned char coverage[4];
#pragma unroll
            for (int c = 0; c < 4; ++c) {
                const int e = end - 1 - c * 32 - static_cast<int>(lane);
                coverage[c] = e >= 0 ? contributor_masks[tile_instance_range.x + static_cast<uint>(e)] : 0u;
            }
            int n_hits = 0;
#pragma unroll
            for (int c = 0; c < 4; ++c) {
                if (end <= 0)
                    break;
                const bool chunk_hit = (coverage[c] & subtile_bit) != 0u;
                const unsigned chunk_mask = __ballot_sync(0xffffffffu, chunk_hit);
                const int count = __popc(chunk_mask);
                if (n_hits + count > 32)
                    break;
                if (chunk_hit)
                    s_hit_entry[warp_base + static_cast<uint>(n_hits) + __popc(chunk_mask & lanes_below)] =
                        end - 1 - static_cast<int>(lane);
                n_hits += count;
                end -= 32;
            }
            if (n_hits == 0)
                continue;
            __syncwarp();

            const bool hit = static_cast<int>(lane) < n_hits;
            const int entry = hit ? s_hit_entry[warp_base + lane] : 0;
            const uint instance_idx = tile_instance_range.x + static_cast<uint>(entry);
            unsigned mask = 0u;

            bool valid_hit = false;
            if (hit) {
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
                    valid_hit = true;
                    // Most hits revisit an already-marked splat; skip the redundant
                    // byte store so its L2 line is not re-dirtied.
                    if (primitive_touched != nullptr && primitive_touched[work_idx] == 0u)
                        primitive_touched[work_idx] = 1u;
                    const uint slot = warp_base + lane;
                    const ProjectedRecord record = projected[work_idx];
                    const float3 conic = make_float3(record.conic_opacity);
                    const float3 cb2 = base2_conic(conic);
                    const float opacity = record.conic_opacity.w;
                    const float3 color_unclamped = make_float3(record.color);
                    const float3 color_clamped = fminf(fmaxf(color_unclamped, 0.0f), config::max_blend_color);
                    const uint factor_bits = (color_unclamped.x <= config::max_blend_color ? 1u : 0u) |
                                             (color_unclamped.y <= config::max_blend_color ? 2u : 0u) |
                                             (color_unclamped.z <= config::max_blend_color ? 4u : 0u);
                    const uint pos = (contributor_positions != nullptr) ? contributor_positions[instance_idx]
                                                                        : static_cast<uint>(entry);
                    const uint pos_factor = (pos & 0x1fffffffu) | (factor_bits << 29);
                    s_meta[slot] = make_uint4(primitive_idx, work_idx, pos_factor, __float_as_uint(color_clamped.z));
                    s_geom[slot] = make_float4(record.geometry.mean2d.x, record.geometry.mean2d.y, opacity, color_clamped.x);
                    s_cb2[slot] = make_float4(cb2.x, cb2.y, cb2.z, color_clamped.y);
                }
            }
            mask = __ballot_sync(0xffffffffu, valid_hit);
            __syncwarp();

            while (mask != 0u) {
                const uint bit = static_cast<uint>(__ffs(mask) - 1);
                mask &= mask - 1u;
                const uint slot = warp_base + bit;

                const uint position = s_meta[slot].z & 0x1fffffffu;
                const bool eligible = position < last;
                const float dx = s_geom[slot].x - center_x;
                const float dy = (s_geom[slot].y - center_y0) - dy_row_off;
                const float unclamped =
                    eligible
                        ? AlphaColumn::at(make_float3(s_cb2[slot].x, s_cb2[slot].y, s_cb2[slot].z),
                                          s_geom[slot].z, dx)
                              .alpha(s_cb2[slot].z, dy)
                        : 0.0f;
                const bool accepted = eligible && alpha_accepted(unclamped);
                if (!__any_sync(0xffffffffu, accepted))
                    continue;

                float h = 0.0f;
                float h_dy = 0.0f;
                float u[kU];
#pragma unroll
                for (int k = 0; k < kU; ++k)
                    u[k] = 0.0f;
                float peak = 0.0f;

                if (accepted) {
                    const float3 color = make_float3(s_geom[slot].w, s_cb2[slot].w, __uint_as_float(s_meta[slot].w));
                    const float alpha = clamped_alpha(unclamped);
                    const bool saturated = unclamped >= config::max_fragment_alpha;
                    const float T_before = T * rcp_approx(1.0f - alpha);
                    const float w = alpha * T_before;
                    if constexpr (BLEND_PEAK)
                        peak = w;

                    const float contrib_dot = dot(color, grad_c);
                    const float dL_dalpha = T_before * (contrib_dot - grad_T);
                    h = saturated ? 0.0f : (-alpha * dL_dalpha);
                    h_dy = h * dy;
                    u[kUConicZ] = 0.5f * h_dy * dy;
                    u[kUColor + 0] = w * grad_c.x;
                    u[kUColor + 1] = w * grad_c.y;
                    u[kUColor + 2] = w * grad_c.z;
                    if constexpr (DENSIFICATION_TYPE == DensificationType::MRNF) {
                        u[kUDens + 0] = w;
                        u[kUDens + 1] = w * pixel_error;
                    } else if constexpr (DENSIFICATION_TYPE == DensificationType::MCMC) {
                        u[kUDens + 0] = w * pixel_error;
                    }
                    if constexpr (EDGE_CHANNEL) {
                        u[kUEdge] = w * edge_weight;
                    }
                    grad_T = __fmaf_rn(alpha, contrib_dot, grad_T * (1.0f - alpha));
                    T = T_before;
                }

                halving_step<kU, 16>(u, lane);
                halving_step<(kU + 1) / 2, 8>(u, lane);
                h += __shfl_xor_sync(0xffffffffu, h, 16);
                h += __shfl_xor_sync(0xffffffffu, h, 8);
                h_dy += __shfl_xor_sync(0xffffffffu, h_dy, 16);
                h_dy += __shfl_xor_sync(0xffffffffu, h_dy, 8);

                const float h_dx = dx * h;
                const float cx = s_cb2[slot].x * kInvDiag;
                const float cy = s_cb2[slot].y * kInvCross;
                const float cz = s_cb2[slot].z * kInvDiag;
                float values[kSingle];
                values[0] = local_y == 0u   ? (cx * h_dx + cy * h_dy)
                            : local_y == 1u ? (cy * h_dx + cz * h_dy)
                            : local_y == 2u ? (0.5f * dx * h_dx)
                                            : (0.5f * dx * h_dy);
                const float opacity_s = s_geom[slot].z;
                values[1] = (local_y == 0u && opacity_s > 0.0f) ? (-h * __frcp_rn(opacity_s)) : 0.0f;
#pragma unroll
                for (int k = 0; k < kUVertical; ++k)
                    values[2 + k] = u[k];

                halving_step<kSingle, 4>(values, lane);
                halving_step<(kSingle + 1) / 2, 2>(values, lane);
                halving_step<((kSingle + 1) / 2 + 1) / 2, 1>(values, lane);

                if (out_base != nullptr && values[0] != 0.0f) {
                    const uint color_bit = out_meta >> 3;
                    const uint factor_bits = s_meta[slot].z >> 29;
                    if (color_bit == 0u || (factor_bits & color_bit) != 0u) {
                        const bool uses_prim = (out_meta & 4u) != 0u;
                        const size_t offset =
                            static_cast<size_t>(uses_prim ? s_meta[slot].x : s_meta[slot].y) * (out_meta & 3u);
                        atomicAdd(out_base + offset, uses_prim ? values[0] : clamp_grad(values[0]));
                    }
                }

                if constexpr (BLEND_PEAK) {
#if __CUDA_ARCH__ >= 800
                    const uint max_bits = __reduce_max_sync(0xffffffffu, __float_as_uint(peak));
                    if (lane == 0u && max_bits != 0u)
                        atomicMax(reinterpret_cast<int*>(blend_peak_out + s_meta[slot].x),
                                  static_cast<int>(max_bits));
#else
                    const float max_p = reduce_max<16>(peak);
                    if (lane == 0u && max_p > 0.0f)
                        atomicMax(reinterpret_cast<int*>(blend_peak_out + s_meta[slot].x), __float_as_int(max_p));
#endif
                }
            }
            __syncwarp();
        }
    }

} // namespace fast_lfs::rasterization::kernels::backward

/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include "buffer_utils.h"
#include "helper_math.h"
#include "kernel_utils.cuh"
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

    template <bool MIP_FILTER, int ACTIVE_SH_BASES>
    __global__ void preprocess_backward_cu(
        const float3* __restrict__ means,
        const float3* __restrict__ raw_scales,
        const float4* __restrict__ raw_rotations,
        const float3* __restrict__ sh_coefficients_rest,
        const float4* __restrict__ w2c,
        const float3* __restrict__ cam_position,
        const float* __restrict__ raw_opacities,
        const uint* __restrict__ primitive_n_touched_tiles,
        const float2* __restrict__ grad_mean2d,
        const float* __restrict__ grad_conic,
        float* __restrict__ grad_opacity_helper,
        float3* __restrict__ grad_color_helper,
        float4* __restrict__ grad_w2c,
        float* __restrict__ densification_info,
        const uint n_primitives,
        const uint total_bases_sh_rest,
        const float w,
        const float h,
        const float fx,
        const float fy,
        const float cx,
        const float cy,
        FusedAdamSettings fused_adam) {
        auto primitive_idx = cg::this_grid().thread_rank();
        if (primitive_idx >= n_primitives || primitive_n_touched_tiles[primitive_idx] == 0)
            return;

        // load 3d mean
        const float3 mean3d = means[primitive_idx];

        // sh evaluation backward
        const float3 dL_dmean3d_from_color = convert_sh_to_color_backward<ACTIVE_SH_BASES>(
            sh_coefficients_rest, grad_color_helper,
            fused_adam,
            mean3d, cam_position[0],
            primitive_idx, total_bases_sh_rest);

        const float4 w2c_r3 = w2c[2];
        const float depth = w2c_r3.x * mean3d.x + w2c_r3.y * mean3d.y + w2c_r3.z * mean3d.z + w2c_r3.w;
        const float depth_safe = fmaxf(depth, 1e-4f);
        const float4 w2c_r1 = w2c[0];
        const float x = (w2c_r1.x * mean3d.x + w2c_r1.y * mean3d.y + w2c_r1.z * mean3d.z + w2c_r1.w) / depth_safe;
        const float4 w2c_r2 = w2c[1];
        const float y = (w2c_r2.x * mean3d.x + w2c_r2.y * mean3d.y + w2c_r2.z * mean3d.z + w2c_r2.w) / depth_safe;

        // compute 3d covariance from raw scale and rotation
        const float3 raw_scale = raw_scales[primitive_idx];
        const float3 clamped_scale = make_float3(
            fminf(raw_scale.x, config::max_raw_scale),
            fminf(raw_scale.y, config::max_raw_scale),
            fminf(raw_scale.z, config::max_raw_scale));
        const float3 variance = make_float3(expf(2.0f * clamped_scale.x), expf(2.0f * clamped_scale.y), expf(2.0f * clamped_scale.z));
        auto [qr, qx, qy, qz] = raw_rotations[primitive_idx];
        const float qrr_raw = qr * qr, qxx_raw = qx * qx, qyy_raw = qy * qy, qzz_raw = qz * qz;
        const float q_norm_sq = qrr_raw + qxx_raw + qyy_raw + qzz_raw;
        const float q_norm_sq_safe = fmaxf(q_norm_sq, 1e-7f);
        const float qxx = 2.0f * qxx_raw / q_norm_sq_safe, qyy = 2.0f * qyy_raw / q_norm_sq_safe, qzz = 2.0f * qzz_raw / q_norm_sq_safe;
        const float qxy = 2.0f * qx * qy / q_norm_sq_safe, qxz = 2.0f * qx * qz / q_norm_sq_safe, qyz = 2.0f * qy * qz / q_norm_sq_safe;
        const float qrx = 2.0f * qr * qx / q_norm_sq_safe, qry = 2.0f * qr * qy / q_norm_sq_safe, qrz = 2.0f * qr * qz / q_norm_sq_safe;
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

        // ewa splatting gradient helpers
        const float clip_left = (-0.15f * w - cx) / fx;
        const float clip_right = (1.15f * w - cx) / fx;
        const float clip_top = (-0.15f * h - cy) / fy;
        const float clip_bottom = (1.15f * h - cy) / fy;
        const float tx = clamp(x, clip_left, clip_right);
        const float ty = clamp(y, clip_top, clip_bottom);
        const float j11 = fx / depth_safe;
        const float j13 = -j11 * tx;
        const float j22 = fy / depth_safe;
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
        const float3 dL_dconic = make_float3(
            grad_conic[primitive_idx],
            grad_conic[n_primitives + primitive_idx],
            grad_conic[2 * n_primitives + primitive_idx]);
        float3 dL_dcov2d = determinant_rcp_sq * make_float3(
                                                    2.0f * bc * dL_dconic.y - cc * dL_dconic.x - bb * dL_dconic.z,
                                                    bc * dL_dconic.x - (ac + bb) * dL_dconic.y + ab * dL_dconic.z,
                                                    2.0f * ab * dL_dconic.y - bb * dL_dconic.x - aa * dL_dconic.z);

        const float original_opacity = __frcp_rn(1.0f + __expf(-raw_opacities[primitive_idx]));
        const float grad_compensated_opacity = grad_opacity_helper[primitive_idx];
        float opacity_compensation = 1.0f;
        if constexpr (MIP_FILTER) {
            const float det_raw = raw_a * raw_c - raw_b * raw_b;
            if (det_raw > config::min_cov2d_determinant && determinant > config::min_cov2d_determinant) {
                const float det_raw_rcp = 1.0f / det_raw;
                opacity_compensation = sqrtf(det_raw * determinant_rcp);
                const float compensation_grad_scale = 0.5f * grad_compensated_opacity * original_opacity * opacity_compensation;
                dL_dcov2d.x += compensation_grad_scale * (raw_c * det_raw_rcp - c * determinant_rcp);
                dL_dcov2d.y += compensation_grad_scale * (-2.0f * raw_b * det_raw_rcp + 2.0f * b * determinant_rcp);
                dL_dcov2d.z += compensation_grad_scale * (raw_a * det_raw_rcp - a * determinant_rcp);
            } else {
                opacity_compensation = 0.0f;
            }
        }

        const float sigmoid_derivative = original_opacity * (1.0f - original_opacity);
        const float opacity_grad = grad_compensated_opacity * opacity_compensation * sigmoid_derivative +
                                   opacity_extra_grad(fused_adam, fused_adam.opacity, primitive_idx);
        adam_step_helper(
            opacity_grad,
            fused_adam.opacity,
            primitive_idx,
            0,
            fused_adam.beta1,
            fused_adam.beta2,
            fused_adam.eps);

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
        float3 dL_dmean3d_cam = make_float3(
            j11 * dL_dmean2d.x,
            j22 * dL_dmean2d.y,
            -j11 * x * dL_dmean2d.x - j22 * y * dL_dmean2d.y);
        const bool valid_x = x >= clip_left && x <= clip_right;
        const bool valid_y = y >= clip_top && y <= clip_bottom;
        if (valid_x)
            dL_dmean3d_cam.x -= j11 * dL_dj13 / depth_safe;
        if (valid_y)
            dL_dmean3d_cam.y -= j22 * dL_dj23 / depth_safe;
        const float factor_x = 1.0f + static_cast<float>(valid_x);
        const float factor_y = 1.0f + static_cast<float>(valid_y);
        dL_dmean3d_cam.z += (j11 * (factor_x * tx * dL_dj13 - dL_dj11) + j22 * (factor_y * ty * dL_dj23 - dL_dj22)) / depth_safe;

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

        // 3d mean gradient from splatting
        const float3 dL_dmean3d_from_splatting = make_float3(
            w2c_r1.x * dL_dmean3d_cam.x + w2c_r2.x * dL_dmean3d_cam.y + w2c_r3.x * dL_dmean3d_cam.z,
            w2c_r1.y * dL_dmean3d_cam.x + w2c_r2.y * dL_dmean3d_cam.y + w2c_r3.y * dL_dmean3d_cam.z,
            w2c_r1.z * dL_dmean3d_cam.x + w2c_r2.z * dL_dmean3d_cam.y + w2c_r3.z * dL_dmean3d_cam.z);

        const float3 dL_dmean3d = dL_dmean3d_from_splatting + dL_dmean3d_from_color;
        const float3 clamped_mean = clamp_grad3(dL_dmean3d);
        adam_step_helper(clamped_mean.x, fused_adam.means, primitive_idx, 0, fused_adam.beta1, fused_adam.beta2, fused_adam.eps);
        adam_step_helper(clamped_mean.y, fused_adam.means, primitive_idx, 1, fused_adam.beta1, fused_adam.beta2, fused_adam.eps);
        adam_step_helper(clamped_mean.z, fused_adam.means, primitive_idx, 2, fused_adam.beta1, fused_adam.beta2, fused_adam.eps);

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
        adam_step_helper(clamped_scale_grad.x + scale_regularization_grad(fused_adam, fused_adam.scaling, scale_base), fused_adam.scaling, primitive_idx, 0, fused_adam.beta1, fused_adam.beta2, fused_adam.eps);
        adam_step_helper(clamped_scale_grad.y + scale_regularization_grad(fused_adam, fused_adam.scaling, scale_base + 1), fused_adam.scaling, primitive_idx, 1, fused_adam.beta1, fused_adam.beta2, fused_adam.eps);
        adam_step_helper(clamped_scale_grad.z + scale_regularization_grad(fused_adam, fused_adam.scaling, scale_base + 2), fused_adam.scaling, primitive_idx, 2, fused_adam.beta1, fused_adam.beta2, fused_adam.eps);

        // raw rotation gradient
        const mat3x3 dL_drotation = {
            2.0f * (rotation_scaled.m11 * dL_dcov3d.m11 + rotation_scaled.m21 * dL_dcov3d.m12 + rotation_scaled.m31 * dL_dcov3d.m13),
            2.0f * (rotation_scaled.m12 * dL_dcov3d.m11 + rotation_scaled.m22 * dL_dcov3d.m12 + rotation_scaled.m32 * dL_dcov3d.m13),
            2.0f * (rotation_scaled.m13 * dL_dcov3d.m11 + rotation_scaled.m23 * dL_dcov3d.m12 + rotation_scaled.m33 * dL_dcov3d.m13),
            2.0f * (rotation_scaled.m11 * dL_dcov3d.m12 + rotation_scaled.m21 * dL_dcov3d.m22 + rotation_scaled.m31 * dL_dcov3d.m23),
            2.0f * (rotation_scaled.m12 * dL_dcov3d.m12 + rotation_scaled.m22 * dL_dcov3d.m22 + rotation_scaled.m32 * dL_dcov3d.m23),
            2.0f * (rotation_scaled.m13 * dL_dcov3d.m12 + rotation_scaled.m23 * dL_dcov3d.m22 + rotation_scaled.m33 * dL_dcov3d.m23),
            2.0f * (rotation_scaled.m11 * dL_dcov3d.m13 + rotation_scaled.m21 * dL_dcov3d.m23 + rotation_scaled.m31 * dL_dcov3d.m33),
            2.0f * (rotation_scaled.m12 * dL_dcov3d.m13 + rotation_scaled.m22 * dL_dcov3d.m23 + rotation_scaled.m32 * dL_dcov3d.m33),
            2.0f * (rotation_scaled.m13 * dL_dcov3d.m13 + rotation_scaled.m23 * dL_dcov3d.m23 + rotation_scaled.m33 * dL_dcov3d.m33)};
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
        const float4 dL_draw_rotation = 2.0f * make_float4(qx * dL_dqrx + qy * dL_dqry + qz * dL_dqrz - qr * dL_dq_norm_helper, 2.0f * qx * dL_dqxx + qy * dL_dqxy + qz * dL_dqxz + qr * dL_dqrx - qx * dL_dq_norm_helper, 2.0f * qy * dL_dqyy + qx * dL_dqxy + qz * dL_dqyz + qr * dL_dqry - qy * dL_dq_norm_helper, 2.0f * qz * dL_dqzz + qx * dL_dqxz + qy * dL_dqyz + qr * dL_dqrz - qz * dL_dq_norm_helper) / q_norm_sq_safe;
        const float4 clamped_rotation = clamp_grad4(dL_draw_rotation);
        adam_step_helper(clamped_rotation.x, fused_adam.rotation, primitive_idx, 0, fused_adam.beta1, fused_adam.beta2, fused_adam.eps);
        adam_step_helper(clamped_rotation.y, fused_adam.rotation, primitive_idx, 1, fused_adam.beta1, fused_adam.beta2, fused_adam.eps);
        adam_step_helper(clamped_rotation.z, fused_adam.rotation, primitive_idx, 2, fused_adam.beta1, fused_adam.beta2, fused_adam.eps);
        adam_step_helper(clamped_rotation.w, fused_adam.rotation, primitive_idx, 3, fused_adam.beta1, fused_adam.beta2, fused_adam.eps);

        // TODO: only needed for adaptive density control from the original 3dgs
        if (densification_info != nullptr) {
            densification_info[primitive_idx] += 1.0f;
            densification_info[n_primitives + primitive_idx] += length(dL_dmean2d * make_float2(0.5f * w, 0.5f * h));
        }
    }

    __global__ void adam_step_invisible(
        const uint* __restrict__ primitive_n_touched_tiles,
        FusedAdamParam param,
        FusedAdamSettings fused_adam,
        const int extra_grad_kind,
        const float beta1,
        const float beta2,
        const float eps) {
        const int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (!param.enabled || idx >= param.n_elements || param.n_attributes <= 0)
            return;

        const uint primitive_idx = static_cast<uint>(idx / param.n_attributes);
        if (primitive_n_touched_tiles[primitive_idx] != 0)
            return;

        float grad = 0.0f;
        if (extra_grad_kind == 1) {
            grad = scale_regularization_grad(fused_adam, param, static_cast<uint>(idx));
        } else if (extra_grad_kind == 2) {
            grad = opacity_extra_grad(fused_adam, param, static_cast<uint>(idx));
        }

        const float moment1 = fmaf(beta1, param.exp_avg[idx] - grad, grad);
        const float moment2 = fmaf(beta2, param.exp_avg_sq[idx] - grad * grad, grad * grad);
        const float denom = sqrtf(moment2) * param.bias_correction2_sqrt_rcp + eps;
        param.param[idx] -= param.step_size * moment1 / denom;
        param.exp_avg[idx] = moment1;
        param.exp_avg_sq[idx] = moment2;
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
        float densification_weight;
        float densification_error_weighted;
    };

    __device__ __forceinline__ BlendBackwardAccum make_zero_blend_backward_accum() {
        return {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
    }

    template <DensificationType DENSIFICATION_TYPE>
    __global__ void __launch_bounds__(config::block_size_blend_backward) blend_backward_cu(
        const uint2* __restrict__ tile_instance_ranges,
        const uint* __restrict__ instance_primitive_indices,
        const float2* __restrict__ primitive_mean2d,
        const float4* __restrict__ primitive_conic_opacity,
        const float3* __restrict__ primitive_color,
        const float* __restrict__ grad_image,
        const float* __restrict__ grad_alpha_map,
        const float* __restrict__ image,
        const float* __restrict__ alpha_map,
        const uint* __restrict__ tile_n_contributions,
        const float* __restrict__ tile_final_transmittance,
        float2* __restrict__ grad_mean2d,
        float* __restrict__ grad_conic,
        float* __restrict__ grad_compensated_opacity,
        float3* __restrict__ grad_color,
        float* __restrict__ densification_info,
        const float* __restrict__ densification_error_map,
        const uint n_primitives,
        const uint width,
        const uint height,
        const uint grid_width) {
        (void)image;
        (void)alpha_map;
        auto block = cg::this_thread_block();
        const uint tile_idx = block.group_index().x;
        const uint thread_rank = block.thread_rank();
        const uint2 tile_instance_range = tile_instance_ranges[tile_idx];
        const int tile_n_primitives = tile_instance_range.y - tile_instance_range.x;
        if (tile_n_primitives <= 0)
            return;

        const uint n_pixels = width * height;
        const uint2 tile_coords = {tile_idx % grid_width, tile_idx / grid_width};
        const uint2 start_pixel_coords = {tile_coords.x * config::tile_width, tile_coords.y * config::tile_height};
        static_assert(config::block_size_blend_backward <= config::block_size_blend);

        __shared__ uint s_last_contributor[config::block_size_blend];
        __shared__ float s_transmittance_state[config::block_size_blend];
        __shared__ float3 s_grad_color_state[config::block_size_blend];
        __shared__ float s_grad_transmittance_state[config::block_size_blend];

        const uint tile_pixel_state_base = tile_idx * config::block_size_blend;
        for (int pixel_rank = static_cast<int>(thread_rank);
             pixel_rank < config::block_size_blend;
             pixel_rank += config::block_size_blend_backward) {
            const uint2 pixel_coords = {start_pixel_coords.x + static_cast<uint>(pixel_rank % config::tile_width),
                                        start_pixel_coords.y + static_cast<uint>(pixel_rank / config::tile_width)};
            const bool valid_pixel = pixel_coords.x < width && pixel_coords.y < height;
            const uint pixel_idx = valid_pixel ? width * pixel_coords.y + pixel_coords.x : 0;

            s_last_contributor[pixel_rank] = valid_pixel ? tile_n_contributions[pixel_idx] : 0;
            s_transmittance_state[pixel_rank] = valid_pixel ? tile_final_transmittance[tile_pixel_state_base + pixel_rank] : 1.0f;
            s_grad_color_state[pixel_rank] = valid_pixel
                                                 ? make_float3(grad_image[pixel_idx], grad_image[n_pixels + pixel_idx], grad_image[2 * n_pixels + pixel_idx])
                                                 : make_float3(0.0f);
            s_grad_transmittance_state[pixel_rank] = valid_pixel ? -grad_alpha_map[pixel_idx] : 0.0f;
        }
        block.sync();

        int splat_batch_size = config::block_size_blend_backward;
        if (tile_n_primitives <= 4) {
            splat_batch_size = 32;
        } else if (tile_n_primitives <= 16) {
            splat_batch_size = 64;
        } else if (tile_n_primitives <= 36) {
            splat_batch_size = 96;
        }

        for (int batch_base = 0; batch_base < tile_n_primitives; batch_base += splat_batch_size) {
            const int n_splats_in_batch = ((tile_n_primitives - batch_base) < splat_batch_size)
                                              ? (tile_n_primitives - batch_base)
                                              : splat_batch_size;
            const int lane = static_cast<int>(thread_rank);
            const bool valid_splat = lane < n_splats_in_batch;
            const int tile_primitive_idx = tile_n_primitives - batch_base - lane - 1;

            uint primitive_idx = 0;
            float2 mean2d = make_float2(0.0f, 0.0f);
            float3 conic = make_float3(0.0f);
            float compensated_opacity = 0.0f;
            float3 color = make_float3(0.0f);
            float3 color_grad_factor = make_float3(0.0f);

            if (valid_splat) {
                primitive_idx = instance_primitive_indices[tile_instance_range.x + tile_primitive_idx];
                mean2d = primitive_mean2d[primitive_idx];
                const float4 conic_opacity = primitive_conic_opacity[primitive_idx];
                conic = make_float3(conic_opacity);
                compensated_opacity = conic_opacity.w;
                const float3 color_unclamped = primitive_color[primitive_idx];
                color = fminf(fmaxf(color_unclamped, 0.0f), config::max_blend_color);
                color_grad_factor = make_float3(
                    (color_unclamped.x >= 0.0f && color_unclamped.x <= config::max_blend_color) ? 1.0f : 0.0f,
                    (color_unclamped.y >= 0.0f && color_unclamped.y <= config::max_blend_color) ? 1.0f : 0.0f,
                    (color_unclamped.z >= 0.0f && color_unclamped.z <= config::max_blend_color) ? 1.0f : 0.0f);
            }

            BlendBackwardAccum accum = make_zero_blend_backward_accum();
            bool has_contribution = false;

            for (int diagonal = 0; diagonal < n_splats_in_batch + config::block_size_blend - 1; ++diagonal) {
                const int pixel_local_rank = diagonal - lane;
                if (valid_splat && pixel_local_rank >= 0 && pixel_local_rank < config::block_size_blend) {
                    const int pixel_rank = pixel_local_rank;
                    const uint last_contributor = s_last_contributor[pixel_rank];
                    if (static_cast<uint>(tile_primitive_idx) < last_contributor) {
                        const uint2 pixel_coords = {start_pixel_coords.x + static_cast<uint>(pixel_rank % config::tile_width),
                                                    start_pixel_coords.y + static_cast<uint>(pixel_rank / config::tile_width)};
                        const float2 pixel = make_float2(__uint2float_rn(pixel_coords.x), __uint2float_rn(pixel_coords.y)) + 0.5f;
                        const float2 delta = mean2d - pixel;
                        const float sigma_over_2 = 0.5f * (conic.x * delta.x * delta.x + conic.z * delta.y * delta.y) +
                                                   conic.y * delta.x * delta.y;
                        if (sigma_over_2 >= 0.0f) {
                            const float gaussian = expf(-sigma_over_2);
                            const float unclamped_alpha = compensated_opacity * gaussian;
                            const float alpha = fminf(unclamped_alpha, config::max_fragment_alpha);
                            if (alpha >= config::min_alpha_threshold) {
                                has_contribution = true;
                                const bool alpha_saturated = unclamped_alpha >= config::max_fragment_alpha;
                                const float one_minus_alpha = 1.0f - alpha;
                                const float one_minus_alpha_safe = fmaxf(one_minus_alpha, 1e-4f);
                                const float transmittance_after = s_transmittance_state[pixel_rank];
                                const float transmittance_before = transmittance_after / one_minus_alpha_safe;
                                const float blending_weight = transmittance_before * alpha;
                                const float3 grad_color_pixel = s_grad_color_state[pixel_rank];
                                const float grad_transmittance_after = s_grad_transmittance_state[pixel_rank];

                                if constexpr (DENSIFICATION_TYPE == DensificationType::MCMC) {
                                    const uint pixel_idx = width * pixel_coords.y + pixel_coords.x;
                                    const float pixel_error = densification_error_map[pixel_idx];
                                    accum.densification_weight += blending_weight;
                                    accum.densification_error_weighted += blending_weight * pixel_error;
                                }

                                const float3 dL_dcolor = blending_weight * grad_color_pixel * color_grad_factor;
                                accum.color_x += dL_dcolor.x;
                                accum.color_y += dL_dcolor.y;
                                accum.color_z += dL_dcolor.z;

                                const float dL_dalpha = dot(transmittance_before * color, grad_color_pixel) -
                                                        grad_transmittance_after * transmittance_before;
                                accum.compensated_opacity += alpha_saturated ? 0.0f : gaussian * dL_dalpha;

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
                                    const uint pixel_idx = width * pixel_coords.y + pixel_coords.x;
                                    const float pixel_error = (densification_error_map != nullptr)
                                                                  ? densification_error_map[pixel_idx]
                                                                  : 1.0f;
                                    accum.densification_weight += blending_weight;
                                    accum.densification_error_weighted += blending_weight * pixel_error;
                                }

                                s_transmittance_state[pixel_rank] = transmittance_before;
                                s_grad_transmittance_state[pixel_rank] = dot(grad_color_pixel, alpha * color) +
                                                                         grad_transmittance_after * one_minus_alpha;
                            }
                        }
                    }
                }
                block.sync();
            }

            if (valid_splat && has_contribution && primitive_idx < n_primitives) {
                atomicAdd(&grad_mean2d[primitive_idx].x, clamp_grad(accum.mean_x));
                atomicAdd(&grad_mean2d[primitive_idx].y, clamp_grad(accum.mean_y));
                atomicAdd(&grad_conic[primitive_idx], clamp_grad(accum.conic_x));
                atomicAdd(&grad_conic[n_primitives + primitive_idx], clamp_grad(accum.conic_y));
                atomicAdd(&grad_conic[2 * n_primitives + primitive_idx], clamp_grad(accum.conic_z));

                atomicAdd(&grad_compensated_opacity[primitive_idx],
                          clamp_grad(accum.compensated_opacity));

                atomicAdd(&grad_color[primitive_idx].x, clamp_grad(accum.color_x));
                atomicAdd(&grad_color[primitive_idx].y, clamp_grad(accum.color_y));
                atomicAdd(&grad_color[primitive_idx].z, clamp_grad(accum.color_z));

                if constexpr (DENSIFICATION_TYPE != DensificationType::None) {
                    atomicAdd(&densification_info[primitive_idx], accum.densification_weight);
                    atomicAdd(&densification_info[n_primitives + primitive_idx], accum.densification_error_weighted);
                }
            }
            block.sync();
        }
    }

} // namespace fast_lfs::rasterization::kernels::backward

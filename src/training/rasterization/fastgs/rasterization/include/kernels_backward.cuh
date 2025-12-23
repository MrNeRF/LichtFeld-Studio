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

    __global__ void preprocess_backward_cu(
        const float3* means,
        const float3* raw_scales,
        const float4* raw_rotations,
        const float3* sh_coefficients_rest,
        const float4* w2c,
        const float3* cam_position,
        const uint* primitive_n_touched_tiles,
        const float2* grad_mean2d,
        const float* grad_conic,
        float3* grad_means,
        float3* grad_raw_scales,
        float4* grad_raw_rotations,
        float3* grad_sh_coefficients_0,
        float3* grad_sh_coefficients_rest,
        float4* grad_w2c,
        float* densification_info,
        const uint n_primitives,
        const uint active_sh_bases,
        const uint total_bases_sh_rest,
        const float w,
        const float h,
        const float fx,
        const float fy,
        const float cx,
        const float cy,
        const bool mip_filter) {
        auto primitive_idx = cg::this_grid().thread_rank();
        if (primitive_idx >= n_primitives || primitive_n_touched_tiles[primitive_idx] == 0)
            return;

        // load 3d mean
        const float3 mean3d = means[primitive_idx];

        // sh evaluation backward
        const float3 dL_dmean3d_from_color = convert_sh_to_color_backward(
            sh_coefficients_rest, grad_sh_coefficients_0, grad_sh_coefficients_rest,
            mean3d, cam_position[0],
            primitive_idx, active_sh_bases, total_bases_sh_rest);

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
        const float kernel_size = mip_filter ? config::dilation_mip_filter : config::dilation;
        const float a = dot(jwc_r1, jw_r1) + kernel_size, b = dot(jwc_r1, jw_r2), c = dot(jwc_r2, jw_r2) + kernel_size;
        const float aa = a * a, bb = b * b, cc = c * c;
        const float ac = a * c, ab = a * b, bc = b * c;
        const float determinant = ac - bb;
        const float determinant_safe = fmaxf(determinant, 1e-8f);
        const float determinant_rcp = 1.0f / determinant_safe;
        const float determinant_rcp_sq = determinant_rcp * determinant_rcp;
        const float3 dL_dconic = make_float3(
            grad_conic[primitive_idx],
            grad_conic[n_primitives + primitive_idx],
            grad_conic[2 * n_primitives + primitive_idx]);
        const float3 dL_dcov2d = determinant_rcp_sq * make_float3(
                                                          2.0f * bc * dL_dconic.y - cc * dL_dconic.x - bb * dL_dconic.z,
                                                          bc * dL_dconic.x - (ac + bb) * dL_dconic.y + ab * dL_dconic.z,
                                                          2.0f * ab * dL_dconic.y - bb * dL_dconic.x - aa * dL_dconic.z);

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

        // mean3d camera space gradient from J and mean2d
        // TODO: original 3dgs accounts for clamping of tx/ty here, but it seems that this is not necessary
        float djwr1_dz_helper = dL_dj11 - 2.0f * tx * dL_dj13;
        float djwr2_dz_helper = dL_dj22 - 2.0f * ty * dL_dj23;
        const float2 dL_dmean2d = grad_mean2d[primitive_idx];
        const float3 dL_dmean3d_cam = make_float3(
            j11 * (dL_dmean2d.x - dL_dj13 / depth_safe),
            j22 * (dL_dmean2d.y - dL_dj23 / depth_safe),
            -j11 * (x * dL_dmean2d.x + djwr1_dz_helper / depth_safe) - j22 * (y * dL_dmean2d.y + djwr2_dz_helper / depth_safe));

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
        grad_means[primitive_idx] += clamp_grad3(dL_dmean3d);

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
        grad_raw_scales[primitive_idx] += clamp_grad3(dL_draw_scale);

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
        grad_raw_rotations[primitive_idx] += clamp_grad4(dL_draw_rotation);

        // TODO: only needed for adaptive density control from the original 3dgs
        if (densification_info != nullptr) {
            densification_info[primitive_idx] += 1.0f;
            densification_info[n_primitives + primitive_idx] += length(dL_dmean2d * make_float2(0.5f * w, 0.5f * h));
        }
    }

    // based on https://github.com/humansensinglab/taming-3dgs/blob/fd0f7d9edfe135eb4eefd3be82ee56dada7f2a16/submodules/diff-gaussian-rasterization/cuda_rasterizer/backward.cu#L404
    __global__ void blend_backward_cu(
        const uint2* tile_instance_ranges,
        const uint* tile_bucket_offsets,
        const uint* instance_primitive_indices,
        const float2* primitive_mean2d,
        const float4* primitive_conic_opacity,
        const float3* primitive_color,
        const float* raw_opacities,
        const float* grad_image,
        const float* grad_alpha_map,
        const float* image,
        const float* alpha_map,
        const uint* tile_max_n_contributions,
        const uint* tile_n_contributions,
        const uint* bucket_tile_index,
        const float4* bucket_color_transmittance,
        float2* grad_mean2d,
        float* grad_conic,
        float* grad_raw_opacity,
        float3* grad_color,
        const uint n_buckets,
        const uint n_primitives,
        const uint width,
        const uint height,
        const uint grid_width,
        const bool mip_filter) {
        auto block = cg::this_thread_block();
        const uint bucket_idx = block.group_index().x;
        if (bucket_idx >= n_buckets)
            return;
        auto warp = cg::tiled_partition<32>(block);
        const uint lane_idx = warp.thread_rank();

        const uint tile_idx = bucket_tile_index[bucket_idx];
        const uint2 tile_instance_range = tile_instance_ranges[tile_idx];
        const int tile_n_primitives = tile_instance_range.y - tile_instance_range.x;
        const uint tile_first_bucket_offset = tile_idx == 0 ? 0 : tile_bucket_offsets[tile_idx - 1];
        const int tile_bucket_idx = bucket_idx - tile_first_bucket_offset;
        if (tile_bucket_idx * 32 >= tile_max_n_contributions[tile_idx])
            return;

        const int tile_primitive_idx = tile_bucket_idx * 32 + lane_idx;
        const int instance_idx = tile_instance_range.x + tile_primitive_idx;
        const bool valid_primitive = tile_primitive_idx < tile_n_primitives;

        // load gaussian data
        uint primitive_idx = 0;
        float2 mean2d = {0.0f, 0.0f};
        float3 conic = {0.0f, 0.0f, 0.0f};
        float compensated_opacity = 0.0f;
        float original_opacity = 0.0f;
        float3 color = {0.0f, 0.0f, 0.0f};
        float3 color_grad_factor = {0.0f, 0.0f, 0.0f};
        if (valid_primitive) {
            primitive_idx = instance_primitive_indices[instance_idx];
            mean2d = primitive_mean2d[primitive_idx];
            const float4 conic_opacity = primitive_conic_opacity[primitive_idx];
            conic = make_float3(conic_opacity);
            compensated_opacity = conic_opacity.w;
            original_opacity = mip_filter ? __frcp_rn(1.0f + __expf(-raw_opacities[primitive_idx])) : compensated_opacity;
            const float3 color_unclamped = primitive_color[primitive_idx];
            color = fmaxf(color_unclamped, 0.0f);
            if (color_unclamped.x >= 0.0f)
                color_grad_factor.x = 1.0f;
            if (color_unclamped.y >= 0.0f)
                color_grad_factor.y = 1.0f;
            if (color_unclamped.z >= 0.0f)
                color_grad_factor.z = 1.0f;
        }

        // helpers
        const uint n_pixels = width * height;

        // gradient accumulation
        float2 dL_dmean2d_accum = {0.0f, 0.0f};
        float3 dL_dconic_accum = {0.0f, 0.0f, 0.0f};
        float dL_draw_opacity_partial_accum = 0.0f;
        float3 dL_dcolor_accum = {0.0f, 0.0f, 0.0f};

        // tile metadata
        const uint2 tile_coords = {tile_idx % grid_width, tile_idx / grid_width};
        const uint2 start_pixel_coords = {tile_coords.x * config::tile_width, tile_coords.y * config::tile_height};

        uint last_contributor;
        float3 color_pixel_after;
        float transmittance;
        float3 grad_color_pixel;
        float grad_alpha_common;

        bucket_color_transmittance += bucket_idx * config::block_size_blend;
        __shared__ uint collected_last_contributor[32];
        __shared__ float4 collected_color_pixel_after_transmittance[32];
        __shared__ float4 collected_grad_info_pixel[32];

// iterate over all pixels in the tile
#pragma unroll
        for (int i = 0; i < config::block_size_blend + 31; ++i) {
            if (i % 32 == 0) {
                const uint local_idx = i + lane_idx;
                const float4 color_transmittance = bucket_color_transmittance[local_idx];
                const uint2 pixel_coords = {start_pixel_coords.x + local_idx % config::tile_width, start_pixel_coords.y + local_idx / config::tile_width};
                const uint pixel_idx = width * pixel_coords.y + pixel_coords.x;
                // final values from forward pass before background blend and the respective gradients
                float3 color_pixel = {0.0f, 0.0f, 0.0f};
                float3 grad_color_pixel = {0.0f, 0.0f, 0.0f};
                float alpha_pixel = 0.0f;
                float grad_alpha_pixel = 0.0f;
                uint last_contrib_val = 0;
                if (pixel_coords.x < width && pixel_coords.y < height) {
                    color_pixel = make_float3(
                        image[pixel_idx],
                        image[n_pixels + pixel_idx],
                        image[2 * n_pixels + pixel_idx]);
                    grad_color_pixel = make_float3(
                        grad_image[pixel_idx],
                        grad_image[n_pixels + pixel_idx],
                        grad_image[2 * n_pixels + pixel_idx]);
                    alpha_pixel = alpha_map[pixel_idx];
                    grad_alpha_pixel = grad_alpha_map[pixel_idx];
                    last_contrib_val = tile_n_contributions[pixel_idx];
                }
                collected_color_pixel_after_transmittance[lane_idx] = make_float4(
                    color_pixel - make_float3(color_transmittance),
                    color_transmittance.w);
                collected_grad_info_pixel[lane_idx] = make_float4(
                    grad_color_pixel,
                    grad_alpha_pixel * (1.0f - alpha_pixel));
                collected_last_contributor[lane_idx] = last_contrib_val;
                __syncwarp();
            }

            if (i > 0) {
                last_contributor = warp.shfl_up(last_contributor, 1);
                color_pixel_after.x = warp.shfl_up(color_pixel_after.x, 1);
                color_pixel_after.y = warp.shfl_up(color_pixel_after.y, 1);
                color_pixel_after.z = warp.shfl_up(color_pixel_after.z, 1);
                transmittance = warp.shfl_up(transmittance, 1);
                grad_color_pixel.x = warp.shfl_up(grad_color_pixel.x, 1);
                grad_color_pixel.y = warp.shfl_up(grad_color_pixel.y, 1);
                grad_color_pixel.z = warp.shfl_up(grad_color_pixel.z, 1);
                grad_alpha_common = warp.shfl_up(grad_alpha_common, 1);
            }

            // which pixel index should this thread deal with?
            const int idx = i - static_cast<int>(lane_idx);
            const uint2 pixel_coords = {start_pixel_coords.x + idx % config::tile_width, start_pixel_coords.y + idx / config::tile_width};
            const bool valid_pixel = pixel_coords.x < width && pixel_coords.y < height;

            // leader thread loads values from shared memory into registers
            if (valid_primitive && valid_pixel && lane_idx == 0 && idx < config::block_size_blend) {
                const int current_shmem_index = i % 32;
                last_contributor = collected_last_contributor[current_shmem_index];
                const float4 color_pixel_after_transmittance = collected_color_pixel_after_transmittance[current_shmem_index];
                color_pixel_after = make_float3(color_pixel_after_transmittance);
                transmittance = color_pixel_after_transmittance.w;
                const float4 grad_info_pixel = collected_grad_info_pixel[current_shmem_index];
                grad_color_pixel = make_float3(grad_info_pixel);
                grad_alpha_common = grad_info_pixel.w;
            }

            const bool skip = !valid_primitive || !valid_pixel || idx < 0 || idx >= config::block_size_blend || tile_primitive_idx >= last_contributor;
            if (skip)
                continue;

            const float2 pixel = make_float2(__uint2float_rn(pixel_coords.x), __uint2float_rn(pixel_coords.y)) + 0.5f;
            const float2 delta = mean2d - pixel;
            const float sigma_over_2 = 0.5f * (conic.x * delta.x * delta.x + conic.z * delta.y * delta.y) + conic.y * delta.x * delta.y;
            if (sigma_over_2 < 0.0f)
                continue;
            const float gaussian = expf(-sigma_over_2);
            const float unclamped_alpha = compensated_opacity * gaussian;
            const float alpha = fminf(unclamped_alpha, config::max_fragment_alpha);
            if (alpha < config::min_alpha_threshold)
                continue;
            const bool alpha_saturated = unclamped_alpha >= config::max_fragment_alpha;
            const float one_minus_alpha = 1.0f - alpha;

            const float blending_weight = transmittance * alpha;

            // color gradient
            const float3 dL_dcolor = blending_weight * grad_color_pixel * color_grad_factor;
            dL_dcolor_accum += dL_dcolor;

            color_pixel_after -= blending_weight * color;

            // alpha gradient
            const float one_minus_alpha_safe = fmaxf(one_minus_alpha, 1e-4f);
            const float one_minus_alpha_rcp = 1.0f / one_minus_alpha_safe;
            const float dL_dalpha_from_color = dot(transmittance * color - color_pixel_after * one_minus_alpha_rcp, grad_color_pixel);
            const float dL_dalpha_from_alpha = grad_alpha_common * one_minus_alpha_rcp;
            const float dL_dalpha = dL_dalpha_from_color + dL_dalpha_from_alpha;
            // opacity gradient w.r.t. compensated_opacity (zero when alpha is clamped - no gradient flows through clamp)
            // dL/d(compensated_opacity) = dL/d(alpha) * d(alpha)/d(compensated_opacity) = dL_dalpha * gaussian
            const float dL_dcompensated_opacity = alpha_saturated ? 0.0f : gaussian * dL_dalpha;
            dL_draw_opacity_partial_accum += dL_dcompensated_opacity;

            // conic and mean2d gradient (zero when alpha is clamped)
            const float gaussian_grad_helper = alpha_saturated ? 0.0f : -alpha * dL_dalpha;
            const float3 dL_dconic = 0.5f * gaussian_grad_helper * make_float3(delta.x * delta.x, delta.x * delta.y, delta.y * delta.y);
            dL_dconic_accum += dL_dconic;
            const float2 dL_dmean2d = gaussian_grad_helper * make_float2(
                                                                 conic.x * delta.x + conic.y * delta.y,
                                                                 conic.y * delta.x + conic.z * delta.y);
            dL_dmean2d_accum += dL_dmean2d;

            transmittance *= one_minus_alpha;
        }

        // Add clamped gradients using atomics
        if (valid_primitive) {
            const float2 clamped_mean2d = make_float2(clamp_grad(dL_dmean2d_accum.x), clamp_grad(dL_dmean2d_accum.y));
            atomicAdd(&grad_mean2d[primitive_idx].x, clamped_mean2d.x);
            atomicAdd(&grad_mean2d[primitive_idx].y, clamped_mean2d.y);
            const float3 clamped_conic = clamp_grad3(dL_dconic_accum);
            atomicAdd(&grad_conic[primitive_idx], clamped_conic.x);
            atomicAdd(&grad_conic[n_primitives + primitive_idx], clamped_conic.y);
            atomicAdd(&grad_conic[2 * n_primitives + primitive_idx], clamped_conic.z);
            // Chain rule: dL/d(raw_opacity) = dL/d(comp_opacity) * d(comp)/d(orig) * d(orig)/d(raw)
            // d(comp)/d(orig) = conv_factor = comp_opacity / orig_opacity (when mip_filter)
            // d(orig)/d(raw) = sigmoid_derivative = orig_opacity * (1 - orig_opacity)
            const float conv_factor = mip_filter ? compensated_opacity / fmaxf(original_opacity, 1e-6f) : 1.0f;
            const float sigmoid_derivative = original_opacity * (1.0f - original_opacity);
            const float dL_draw_opacity = clamp_grad(dL_draw_opacity_partial_accum * conv_factor * sigmoid_derivative);
            atomicAdd(&grad_raw_opacity[primitive_idx], dL_draw_opacity);
            const float3 clamped_color = clamp_grad3(dL_dcolor_accum);
            atomicAdd(&grad_color[primitive_idx].x, clamped_color.x);
            atomicAdd(&grad_color[primitive_idx].y, clamped_color.y);
            atomicAdd(&grad_color[primitive_idx].z, clamped_color.z);
        }
    }

} // namespace fast_lfs::rasterization::kernels::backward
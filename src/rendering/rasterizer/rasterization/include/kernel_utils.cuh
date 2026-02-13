/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include "helper_math.h"
#include "rasterization_config.h"
#include "utils.h"
#include <cooperative_groups.h>
namespace cg = cooperative_groups;

namespace lfs::rendering::kernels {

    __device__ inline float3 mat3_transpose_mul_vec3(
        const mat3x3& m,
        const float3& v) {
        return make_float3(
            m.m11 * v.x + m.m21 * v.y + m.m31 * v.z,
            m.m12 * v.x + m.m22 * v.y + m.m32 * v.z,
            m.m13 * v.x + m.m23 * v.y + m.m33 * v.z);
    }

    __device__ inline float3 convert_sh_to_color_from_dir(
        const float3* sh_coefficients_0,
        const float3* sh_coefficients_rest,
        const float3& view_dir,
        const uint primitive_idx,
        const uint active_sh_bases,
        const uint total_bases_sh_rest) {
        // computation adapted from https://github.com/NVlabs/tiny-cuda-nn/blob/212104156403bd87616c1a4f73a1c5f2c2e172a9/include/tiny-cuda-nn/common_device.h#L340
        float3 result = 0.5f + 0.28209479177387814f * sh_coefficients_0[primitive_idx];
        if (active_sh_bases > 1) {
            const float3* coefficients_ptr = sh_coefficients_rest + primitive_idx * total_bases_sh_rest;
            auto [x, y, z] = normalize(view_dir);
            result = result + (-0.48860251190291987f * y) * coefficients_ptr[0] + (0.48860251190291987f * z) * coefficients_ptr[1] + (-0.48860251190291987f * x) * coefficients_ptr[2];
            if (active_sh_bases > 4) {
                const float xx = x * x, yy = y * y, zz = z * z;
                const float xy = x * y, xz = x * z, yz = y * z;
                result = result + (1.0925484305920792f * xy) * coefficients_ptr[3] + (-1.0925484305920792f * yz) * coefficients_ptr[4] + (0.94617469575755997f * zz - 0.31539156525251999f) * coefficients_ptr[5] + (-1.0925484305920792f * xz) * coefficients_ptr[6] + (0.54627421529603959f * xx - 0.54627421529603959f * yy) * coefficients_ptr[7];
                if (active_sh_bases > 9) {
                    result = result + (0.59004358992664352f * y * (-3.0f * xx + yy)) * coefficients_ptr[8] + (2.8906114426405538f * xy * z) * coefficients_ptr[9] + (0.45704579946446572f * y * (1.0f - 5.0f * zz)) * coefficients_ptr[10] + (0.3731763325901154f * z * (5.0f * zz - 3.0f)) * coefficients_ptr[11] + (0.45704579946446572f * x * (1.0f - 5.0f * zz)) * coefficients_ptr[12] + (1.4453057213202769f * z * (xx - yy)) * coefficients_ptr[13] + (0.59004358992664352f * x * (-xx + 3.0f * yy)) * coefficients_ptr[14];
                }
            }
        }
        return result;
    }

    // based on https://github.com/r4dl/StopThePop-Rasterization/blob/d8cad09919ff49b11be3d693d1e71fa792f559bb/cuda_rasterizer/stopthepop/stopthepop_common.cuh#L131
    __device__ inline bool will_primitive_contribute(
        const float2& mean,
        const float3& conic,
        const uint tile_x,
        const uint tile_y,
        const float power_threshold) {
        const float2 rect_min = make_float2(static_cast<float>(tile_x * config::tile_width), static_cast<float>(tile_y * config::tile_height));
        const float2 rect_max = make_float2(static_cast<float>((tile_x + 1) * config::tile_width - 1), static_cast<float>((tile_y + 1) * config::tile_height - 1));

        const float x_min_diff = rect_min.x - mean.x;
        const float x_left = static_cast<float>(x_min_diff > 0.0f);
        const float not_in_x_range = x_left + static_cast<float>(mean.x > rect_max.x);
        const float y_min_diff = rect_min.y - mean.y;
        const float y_above = static_cast<float>(y_min_diff > 0.0f);
        const float not_in_y_range = y_above + static_cast<float>(mean.y > rect_max.y);

        // let's hope the compiler optimizes this properly
        if (not_in_y_range + not_in_x_range == 0.0f) {
            return true;
        }
        const float2 closest_corner = make_float2(
            fast_lerp(rect_max.x, rect_min.x, x_left),
            fast_lerp(rect_max.y, rect_min.y, y_above));
        const float2 diff = mean - closest_corner;

        const float2 d = make_float2(
            copysignf(static_cast<float>(config::tile_width - 1), x_min_diff),
            copysignf(static_cast<float>(config::tile_height - 1), y_min_diff));
        const float2 t = make_float2(
            not_in_y_range * __saturatef((d.x * conic.x * diff.x + d.x * conic.y * diff.y) / (d.x * conic.x * d.x)),
            not_in_x_range * __saturatef((d.y * conic.y * diff.x + d.y * conic.z * diff.y) / (d.y * conic.z * d.y)));
        const float2 max_contribution_point = closest_corner + t * d;
        const float2 delta = mean - max_contribution_point;
        const float max_power_in_tile = 0.5f * (conic.x * delta.x * delta.x + conic.z * delta.y * delta.y) + conic.y * delta.x * delta.y;
        return max_power_in_tile <= power_threshold;
    }

    // based on https://github.com/r4dl/StopThePop-Rasterization/blob/d8cad09919ff49b11be3d693d1e71fa792f559bb/cuda_rasterizer/stopthepop/stopthepop_common.cuh#L177
    __device__ inline uint compute_exact_n_touched_tiles(
        const float2& mean2d,
        const float3& conic,
        const uint4& screen_bounds,
        const float power_threshold,
        const uint tile_count,
        const bool active) {
        const float2 mean2d_shifted = mean2d - 0.5f;

        uint n_touched_tiles = 0;
        if (active) {
            const uint screen_bounds_width = screen_bounds.y - screen_bounds.x;
            for (uint instance_idx = 0; instance_idx < tile_count && instance_idx < config::n_sequential_threshold; instance_idx++) {
                const uint tile_y = screen_bounds.z + (instance_idx / screen_bounds_width);
                const uint tile_x = screen_bounds.x + (instance_idx % screen_bounds_width);
                if (will_primitive_contribute(mean2d_shifted, conic, tile_x, tile_y, power_threshold))
                    n_touched_tiles++;
            }
        }

        const uint lane_idx = cg::this_thread_block().thread_rank() % 32u;
        const uint warp_idx = cg::this_thread_block().thread_rank() / 32u;

        const int compute_cooperatively = active && tile_count > config::n_sequential_threshold;
        const uint remaining_threads = __ballot_sync(0xffffffffu, compute_cooperatively);
        if (remaining_threads == 0)
            return n_touched_tiles;

        const uint n_remaining_threads = __popc(remaining_threads);
        for (int n = 0; n < n_remaining_threads && n < 32; n++) {
            const uint current_lane = __fns(remaining_threads, 0, n + 1); // find lane index of next remaining thread

            const uint4 screen_bounds_coop = make_uint4(
                __shfl_sync(0xffffffffu, screen_bounds.x, current_lane),
                __shfl_sync(0xffffffffu, screen_bounds.y, current_lane),
                __shfl_sync(0xffffffffu, screen_bounds.z, current_lane),
                __shfl_sync(0xffffffffu, screen_bounds.w, current_lane));
            const uint screen_bounds_width_coop = screen_bounds_coop.y - screen_bounds_coop.x;
            const uint tile_count_coop = (screen_bounds_coop.w - screen_bounds_coop.z) * screen_bounds_width_coop;

            const float2 mean2d_shifted_coop = make_float2(
                __shfl_sync(0xffffffffu, mean2d_shifted.x, current_lane),
                __shfl_sync(0xffffffffu, mean2d_shifted.y, current_lane));
            const float3 conic_coop = make_float3(
                __shfl_sync(0xffffffffu, conic.x, current_lane),
                __shfl_sync(0xffffffffu, conic.y, current_lane),
                __shfl_sync(0xffffffffu, conic.z, current_lane));
            const float power_threshold_coop = __shfl_sync(0xffffffffu, power_threshold, current_lane);

            const uint remaining_tile_count = tile_count_coop - config::n_sequential_threshold;
            const int n_iterations = div_round_up(remaining_tile_count, 32u);
            for (int i = 0; i < n_iterations; i++) {
                const int instance_idx = i * 32 + lane_idx + config::n_sequential_threshold;
                const int active_current = instance_idx < tile_count_coop;
                const uint tile_y = screen_bounds_coop.z + (instance_idx / screen_bounds_width_coop);
                const uint tile_x = screen_bounds_coop.x + (instance_idx % screen_bounds_width_coop);
                const uint contributes = active_current && will_primitive_contribute(mean2d_shifted_coop, conic_coop, tile_x, tile_y, power_threshold_coop);
                const uint contributes_ballot = __ballot_sync(0xffffffffu, contributes);
                const uint n_contributes = __popc(contributes_ballot);
                n_touched_tiles += (current_lane == lane_idx) * n_contributes;
            }
        }

        return n_touched_tiles;
    }

    // Projection result: screen position and Jacobian rows for EWA splatting
    struct ProjectionResult {
        float2 mean2d;
        float3 jw_r1; // First row of J * W (Jacobian * world-to-camera rotation)
        float3 jw_r2; // Second row of J * W
    };

    // Project 3D covariance to 2D using Jacobian rows: cov2d = J * cov3d * J^T
    __device__ inline float3 project_cov3d(
        const float3& jw_r1,
        const float3& jw_r2,
        const mat3x3_triu& cov3d) {
        const float3 jwc_r1 = make_float3(
            jw_r1.x * cov3d.m11 + jw_r1.y * cov3d.m12 + jw_r1.z * cov3d.m13,
            jw_r1.x * cov3d.m12 + jw_r1.y * cov3d.m22 + jw_r1.z * cov3d.m23,
            jw_r1.x * cov3d.m13 + jw_r1.y * cov3d.m23 + jw_r1.z * cov3d.m33);
        const float3 jwc_r2 = make_float3(
            jw_r2.x * cov3d.m11 + jw_r2.y * cov3d.m12 + jw_r2.z * cov3d.m13,
            jw_r2.x * cov3d.m12 + jw_r2.y * cov3d.m22 + jw_r2.z * cov3d.m23,
            jw_r2.x * cov3d.m13 + jw_r2.y * cov3d.m23 + jw_r2.z * cov3d.m33);
        return make_float3(dot(jwc_r1, jw_r1), dot(jwc_r1, jw_r2), dot(jwc_r2, jw_r2));
    }

    // Orthographic projection: linear mapping from camera space to screen
    __device__ inline ProjectionResult project_orthographic(
        const float cam_x,
        const float cam_y,
        const float cx,
        const float cy,
        const float ortho_scale,
        const float4& w2c_r1,
        const float4& w2c_r2) {
        return {
            .mean2d = make_float2(cam_x * ortho_scale + cx, cam_y * ortho_scale + cy),
            .jw_r1 = make_float3(ortho_scale * w2c_r1.x, ortho_scale * w2c_r1.y, ortho_scale * w2c_r1.z),
            .jw_r2 = make_float3(ortho_scale * w2c_r2.x, ortho_scale * w2c_r2.y, ortho_scale * w2c_r2.z)};
    }

    // Perspective projection: divide by depth with clamped Jacobian
    __device__ inline ProjectionResult project_perspective(
        const float cam_x,
        const float cam_y,
        const float depth,
        const float fx,
        const float fy,
        const float cx,
        const float cy,
        const float w,
        const float h,
        const float4& w2c_r1,
        const float4& w2c_r2,
        const float4& w2c_r3) {
        const float x = cam_x / depth;
        const float y = cam_y / depth;
        // Clamp to 15% beyond viewport to stabilize Jacobian
        const float tx = clamp(x, (-0.15f * w - cx) / fx, (1.15f * w - cx) / fx);
        const float ty = clamp(y, (-0.15f * h - cy) / fy, (1.15f * h - cy) / fy);
        const float j11 = fx / depth;
        const float j22 = fy / depth;
        return {
            .mean2d = make_float2(x * fx + cx, y * fy + cy),
            .jw_r1 = make_float3(
                j11 * w2c_r1.x - j11 * tx * w2c_r3.x,
                j11 * w2c_r1.y - j11 * tx * w2c_r3.y,
                j11 * w2c_r1.z - j11 * tx * w2c_r3.z),
            .jw_r2 = make_float3(
                j22 * w2c_r2.x - j22 * ty * w2c_r3.x,
                j22 * w2c_r2.y - j22 * ty * w2c_r3.y,
                j22 * w2c_r2.z - j22 * ty * w2c_r3.z)};
    }

} // namespace lfs::rendering::kernels

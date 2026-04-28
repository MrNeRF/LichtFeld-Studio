/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include "helper_math.h"
#include "rasterization_config.h"
#include "utils.h"

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

    __device__ inline float power_threshold_for_opacity(const float opacity) {
        const float safe_opacity = fmaxf(opacity, config::min_alpha_threshold);
        const float alpha_cutoff_power = logf(safe_opacity * config::min_alpha_threshold_rcp);
        return fmaxf(config::max_power_threshold, alpha_cutoff_power);
    }

    __device__ inline float stddev_for_power_threshold(const float power_threshold) {
        return sqrtf(2.0f * power_threshold);
    }

    // based on https://github.com/r4dl/StopThePop-Rasterization/blob/d8cad09919ff49b11be3d693d1e71fa792f559bb/cuda_rasterizer/stopthepop/stopthepop_common.cuh#L177
    __device__ inline float2 ellipse_range_bound(
        const float3& conic,
        const float radius_sq,
        const float y0,
        const float y1) {
        const float a = conic.x;
        const float b = conic.y;
        const float c = conic.z;
        const float det = fmaxf(a * c - b * b, 1e-20f);
        const float ym = -b / c * sqrtf(fmaxf(c * radius_sq / det, 0.0f));

        const float v0 = fminf(fmaxf(-ym, y0), y1);
        const float v1 = fminf(fmaxf(ym, y0), y1);
        const float bv0 = -b * v0;
        const float bv1 = -b * v1;

        const float inv_a = 1.0f / a;
        const float x0 = inv_a * (bv0 - sqrtf(fmaxf(bv0 * bv0 - a * (c * v0 * v0 - radius_sq), 0.0f)));
        const float x1 = inv_a * (bv1 + sqrtf(fmaxf(bv1 * bv1 - a * (c * v1 * v1 - radius_sq), 0.0f)));
        return make_float2(x0, x1);
    }

    __device__ inline uint floor_tile_clamped(
        const float coord,
        const uint min_tile,
        const uint max_tile,
        const uint tile_size) {
        const int tile = __float2int_rd(coord / static_cast<float>(tile_size));
        return static_cast<uint>(min(max(tile, static_cast<int>(min_tile)), static_cast<int>(max_tile)));
    }

    __device__ inline uint ceil_tile_clamped(
        const float coord,
        const uint min_tile,
        const uint max_tile,
        const uint tile_size) {
        const int tile = __float2int_ru(coord / static_cast<float>(tile_size));
        return static_cast<uint>(min(max(tile, static_cast<int>(min_tile)), static_cast<int>(max_tile)));
    }

    __device__ inline uint compute_exact_n_touched_tiles(
        const float2& mean2d,
        const float3& conic,
        const uint4& screen_bounds,
        const float power_threshold,
        const bool active) {
        if (!active)
            return 0;

        const float2 mean2d_shifted = mean2d - 0.5f;
        const float radius_sq = 2.0f * power_threshold;
        if (radius_sq <= 0.0f)
            return 0;

        const uint screen_bounds_width = screen_bounds.y - screen_bounds.x;
        const uint screen_bounds_height = screen_bounds.w - screen_bounds.z;

        uint n_touched_tiles = 0;

        if (screen_bounds_height <= screen_bounds_width) {
            for (uint tile_y = screen_bounds.z; tile_y < screen_bounds.w; tile_y++) {
                const float y0 = static_cast<float>(tile_y * config::tile_height) - mean2d_shifted.y;
                const float y1 = y0 + static_cast<float>(config::tile_height);
                const float2 bound = ellipse_range_bound(conic, radius_sq, y0, y1);
                const uint min_x = floor_tile_clamped(bound.x + mean2d_shifted.x, screen_bounds.x, screen_bounds.y, config::tile_width);
                const uint max_x = ceil_tile_clamped(bound.y + mean2d_shifted.x, screen_bounds.x, screen_bounds.y, config::tile_width);
                n_touched_tiles += max_x - min_x;
            }
        } else {
            const float3 conic_transposed = make_float3(conic.z, conic.y, conic.x);
            for (uint tile_x = screen_bounds.x; tile_x < screen_bounds.y; tile_x++) {
                const float x0 = static_cast<float>(tile_x * config::tile_width) - mean2d_shifted.x;
                const float x1 = x0 + static_cast<float>(config::tile_width);
                const float2 bound = ellipse_range_bound(conic_transposed, radius_sq, x0, x1);
                const uint min_y = floor_tile_clamped(bound.x + mean2d_shifted.y, screen_bounds.z, screen_bounds.w, config::tile_height);
                const uint max_y = ceil_tile_clamped(bound.y + mean2d_shifted.y, screen_bounds.z, screen_bounds.w, config::tile_height);
                n_touched_tiles += max_y - min_y;
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

/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include "fused_adam_types.h"
#include "helper_math.h"
#include "rasterization_config.h"
#include "utils.h"

namespace fast_lfs::rasterization::kernels {

    // SH swizzle index: swizzled layout is [ceil(N/R), 16, R, 3] where R = sh_reorder_size.
    // Adjacent primitives in a warp hit adjacent lanes -> fully coalesced memory access.
    // Returns the float3 slot index (multiply by 3 for float offset).
    __device__ __host__ __forceinline__ unsigned int shAt(unsigned int primitive_idx, unsigned int coeff_idx) {
        const unsigned int R = config::sh_reorder_size;
        const unsigned int K = config::sh_max_coeffs;
        const unsigned int block = primitive_idx / R;
        const unsigned int lane = primitive_idx % R;
        return block * (K * R) + coeff_idx * R + lane;
    }

    // Safe normalize: returns (0,0,1) for degenerate vectors to prevent NaN
    __device__ inline float3 safe_normalize(const float3 v) {
        constexpr float NORM_SQ_MIN = 1e-12f;
        const float norm_sq = dot(v, v);
        if (norm_sq < NORM_SQ_MIN) {
            return make_float3(0.0f, 0.0f, 1.0f);
        }
        return v * rsqrtf(norm_sq);
    }

    __device__ inline float3 convert_sh_to_color(
        const float3* sh_coefficients_0,
        const float3* sh_coefficients_rest,
        const float3& position,
        const float3& cam_position,
        const uint primitive_idx,
        const uint active_sh_bases,
        const uint /*total_bases_sh_rest*/) {
        // sh_coefficients_rest is the swizzled buffer: indexed via shAt(p, k).
        // Adjacent lanes in a warp hit adjacent memory -> coalesced loads.
        // computation adapted from https://github.com/NVlabs/tiny-cuda-nn/blob/212104156403bd87616c1a4f73a1c5f2c2e172a9/include/tiny-cuda-nn/common_device.h#L340
        float3 result = 0.5f + 0.28209479177387814f * sh_coefficients_0[primitive_idx];
        if (active_sh_bases > 1) {
            auto [x, y, z] = safe_normalize(position - cam_position);
            const float3 c0 = sh_coefficients_rest[shAt(primitive_idx, 0)];
            const float3 c1 = sh_coefficients_rest[shAt(primitive_idx, 1)];
            const float3 c2 = sh_coefficients_rest[shAt(primitive_idx, 2)];
            result = result + (-0.48860251190291987f * y) * c0 + (0.48860251190291987f * z) * c1 + (-0.48860251190291987f * x) * c2;
            if (active_sh_bases > 4) {
                const float xx = x * x, yy = y * y, zz = z * z;
                const float xy = x * y, xz = x * z, yz = y * z;
                const float3 c3 = sh_coefficients_rest[shAt(primitive_idx, 3)];
                const float3 c4 = sh_coefficients_rest[shAt(primitive_idx, 4)];
                const float3 c5 = sh_coefficients_rest[shAt(primitive_idx, 5)];
                const float3 c6 = sh_coefficients_rest[shAt(primitive_idx, 6)];
                const float3 c7 = sh_coefficients_rest[shAt(primitive_idx, 7)];
                result = result + (1.0925484305920792f * xy) * c3 + (-1.0925484305920792f * yz) * c4 + (0.94617469575755997f * zz - 0.31539156525251999f) * c5 + (-1.0925484305920792f * xz) * c6 + (0.54627421529603959f * xx - 0.54627421529603959f * yy) * c7;
                if (active_sh_bases > 9) {
                    const float3 c8 = sh_coefficients_rest[shAt(primitive_idx, 8)];
                    const float3 c9 = sh_coefficients_rest[shAt(primitive_idx, 9)];
                    const float3 c10 = sh_coefficients_rest[shAt(primitive_idx, 10)];
                    const float3 c11 = sh_coefficients_rest[shAt(primitive_idx, 11)];
                    const float3 c12 = sh_coefficients_rest[shAt(primitive_idx, 12)];
                    const float3 c13 = sh_coefficients_rest[shAt(primitive_idx, 13)];
                    const float3 c14 = sh_coefficients_rest[shAt(primitive_idx, 14)];
                    result = result + (0.59004358992664352f * y * (-3.0f * xx + yy)) * c8 + (2.8906114426405538f * xy * z) * c9 + (0.45704579946446572f * y * (1.0f - 5.0f * zz)) * c10 + (0.3731763325901154f * z * (5.0f * zz - 3.0f)) * c11 + (0.45704579946446572f * x * (1.0f - 5.0f * zz)) * c12 + (1.4453057213202769f * z * (xx - yy)) * c13 + (0.59004358992664352f * x * (-xx + 3.0f * yy)) * c14;
                }
            }
        }
        return result;
    }

    __device__ inline void adam_step_helper(
        const float grad,
        const FusedAdamParam& param,
        const uint primitive_idx,
        const uint offset,
        const float beta1,
        const float beta2,
        const float eps) {
        const uint element_idx = primitive_idx * static_cast<uint>(param.n_attributes) + offset;
        if (!param.enabled || element_idx >= static_cast<uint>(param.n_elements))
            return;

        const float moment1_prev = param.exp_avg[element_idx];
        const float moment2_prev = param.exp_avg_sq[element_idx];
        const float grad_sq = grad * grad;
        const float moment1 = fmaf(beta1, moment1_prev - grad, grad);
        const float moment2 = fmaf(beta2, moment2_prev - grad_sq, grad_sq);
        const float denom = sqrtf(moment2) * param.bias_correction2_sqrt_rcp + eps;
        param.param[element_idx] -= param.step_size * moment1 / denom;
        param.exp_avg[element_idx] = moment1;
        param.exp_avg_sq[element_idx] = moment2;
    }

    __device__ inline float sigmoid(const float x) {
        return 1.0f / (1.0f + expf(-x));
    }

    __device__ inline float scale_regularization_grad(
        const FusedAdamSettings& fused_adam,
        const FusedAdamParam& param,
        const uint element_idx) {
        if (fused_adam.scale_reg_weight <= 0.0f || param.n_elements <= 0)
            return 0.0f;
        return fused_adam.scale_reg_weight * expf(param.param[element_idx]) /
               static_cast<float>(param.n_elements);
    }

    __device__ inline float opacity_extra_grad(
        const FusedAdamSettings& fused_adam,
        const FusedAdamParam& param,
        const uint element_idx) {
        float grad = 0.0f;
        if (fused_adam.opacity_reg_weight > 0.0f && param.n_elements > 0) {
            const float opa = sigmoid(param.param[element_idx]);
            grad += fused_adam.opacity_reg_weight * opa * (1.0f - opa) /
                    static_cast<float>(param.n_elements);
        }
        if (fused_adam.sparsity_opa_sigmoid != nullptr &&
            fused_adam.sparsity_z != nullptr &&
            fused_adam.sparsity_u != nullptr &&
            element_idx < static_cast<uint>(fused_adam.sparsity_n)) {
            const float opa = fused_adam.sparsity_opa_sigmoid[element_idx];
            grad += fused_adam.sparsity_rho *
                    (opa - fused_adam.sparsity_z[element_idx] + fused_adam.sparsity_u[element_idx]) *
                    opa * (1.0f - opa) *
                    fused_adam.sparsity_grad_loss;
        }
        return grad;
    }

    // Adam-step helper for shN's swizzled buffer: element index is shAt(p,k)*3 + c.
    // The FusedAdamParam.exp_avg / exp_avg_sq / param buffers for shN are laid out in the
    // same swizzled order, so 32 warp lanes hit consecutive memory locations -> coalesced.
    __device__ inline void adam_step_helper_shN(
        const float grad,
        const FusedAdamParam& param,
        const uint primitive_idx,
        const uint basis_idx,
        const uint channel,
        const float beta1,
        const float beta2,
        const float eps) {
        const uint element_idx = shAt(primitive_idx, basis_idx) * 3u + channel;
        if (!param.enabled || element_idx >= static_cast<uint>(param.n_elements))
            return;

        const float moment1_prev = param.exp_avg[element_idx];
        const float moment2_prev = param.exp_avg_sq[element_idx];
        const float grad_sq = grad * grad;
        const float moment1 = fmaf(beta1, moment1_prev - grad, grad);
        const float moment2 = fmaf(beta2, moment2_prev - grad_sq, grad_sq);
        const float denom = sqrtf(moment2) * param.bias_correction2_sqrt_rcp + eps;
        param.param[element_idx] -= param.step_size * moment1 / denom;
        param.exp_avg[element_idx] = moment1;
        param.exp_avg_sq[element_idx] = moment2;
    }

    __device__ inline void apply_shN_grad(
        const uint basis_idx,
        const float3 grad,
        const FusedAdamSettings& fused_adam,
        const uint primitive_idx) {
        adam_step_helper_shN(grad.x, fused_adam.shN, primitive_idx, basis_idx, 0, fused_adam.beta1, fused_adam.beta2, fused_adam.eps);
        adam_step_helper_shN(grad.y, fused_adam.shN, primitive_idx, basis_idx, 1, fused_adam.beta1, fused_adam.beta2, fused_adam.eps);
        adam_step_helper_shN(grad.z, fused_adam.shN, primitive_idx, basis_idx, 2, fused_adam.beta1, fused_adam.beta2, fused_adam.eps);
    }

    template <int ACTIVE_SH_BASES>
    __device__ inline float3 convert_sh_to_color_backward(
        const float3* sh_coefficients_rest,
        float3* grad_color_helper,
        const FusedAdamSettings& fused_adam,
        const float3& position,
        const float3& cam_position,
        const uint primitive_idx,
        const uint total_bases_sh_rest) {
        // computation adapted from https://github.com/NVlabs/tiny-cuda-nn/blob/212104156403bd87616c1a4f73a1c5f2c2e172a9/include/tiny-cuda-nn/common_device.h#L340
        const float3 grad_color = grad_color_helper[primitive_idx];
        const float3 dL_dsh0 = 0.28209479177387814f * grad_color;
        adam_step_helper(dL_dsh0.x, fused_adam.sh0, primitive_idx, 0, fused_adam.beta1, fused_adam.beta2, fused_adam.eps);
        adam_step_helper(dL_dsh0.y, fused_adam.sh0, primitive_idx, 1, fused_adam.beta1, fused_adam.beta2, fused_adam.eps);
        adam_step_helper(dL_dsh0.z, fused_adam.sh0, primitive_idx, 2, fused_adam.beta1, fused_adam.beta2, fused_adam.eps);
        float3 dcolor_dposition = make_float3(0.0f);
        if constexpr (ACTIVE_SH_BASES > 1) {
            // sh_coefficients_rest is the swizzled buffer; read coefficients via shAt.
            auto [x_raw, y_raw, z_raw] = position - cam_position;
            auto [x, y, z] = safe_normalize(make_float3(x_raw, y_raw, z_raw));
            const float3 c0 = sh_coefficients_rest[shAt(primitive_idx, 0)];
            const float3 c1 = sh_coefficients_rest[shAt(primitive_idx, 1)];
            const float3 c2 = sh_coefficients_rest[shAt(primitive_idx, 2)];
            apply_shN_grad(0, (-0.48860251190291987f * y) * grad_color, fused_adam, primitive_idx);
            apply_shN_grad(1, (0.48860251190291987f * z) * grad_color, fused_adam, primitive_idx);
            apply_shN_grad(2, (-0.48860251190291987f * x) * grad_color, fused_adam, primitive_idx);
            float3 grad_direction_x = -0.48860251190291987f * c2;
            float3 grad_direction_y = -0.48860251190291987f * c0;
            float3 grad_direction_z = 0.48860251190291987f * c1;
            if constexpr (ACTIVE_SH_BASES > 4) {
                const float xx = x * x, yy = y * y, zz = z * z;
                const float xy = x * y, xz = x * z, yz = y * z;
                const float3 c3 = sh_coefficients_rest[shAt(primitive_idx, 3)];
                const float3 c4 = sh_coefficients_rest[shAt(primitive_idx, 4)];
                const float3 c5 = sh_coefficients_rest[shAt(primitive_idx, 5)];
                const float3 c6 = sh_coefficients_rest[shAt(primitive_idx, 6)];
                const float3 c7 = sh_coefficients_rest[shAt(primitive_idx, 7)];
                apply_shN_grad(3, (1.0925484305920792f * xy) * grad_color, fused_adam, primitive_idx);
                apply_shN_grad(4, (-1.0925484305920792f * yz) * grad_color, fused_adam, primitive_idx);
                apply_shN_grad(5, (0.94617469575755997f * zz - 0.31539156525251999f) * grad_color, fused_adam, primitive_idx);
                apply_shN_grad(6, (-1.0925484305920792f * xz) * grad_color, fused_adam, primitive_idx);
                apply_shN_grad(7, (0.54627421529603959f * xx - 0.54627421529603959f * yy) * grad_color, fused_adam, primitive_idx);
                grad_direction_x = grad_direction_x + (1.0925484305920792f * y) * c3 + (-1.0925484305920792f * z) * c6 + (1.0925484305920792f * x) * c7;
                grad_direction_y = grad_direction_y + (1.0925484305920792f * x) * c3 + (-1.0925484305920792f * z) * c4 + (-1.0925484305920792f * y) * c7;
                grad_direction_z = grad_direction_z + (-1.0925484305920792f * y) * c4 + (1.8923493915151202f * z) * c5 + (-1.0925484305920792f * x) * c6;
                if constexpr (ACTIVE_SH_BASES > 9) {
                    const float3 c8 = sh_coefficients_rest[shAt(primitive_idx, 8)];
                    const float3 c9 = sh_coefficients_rest[shAt(primitive_idx, 9)];
                    const float3 c10 = sh_coefficients_rest[shAt(primitive_idx, 10)];
                    const float3 c11 = sh_coefficients_rest[shAt(primitive_idx, 11)];
                    const float3 c12 = sh_coefficients_rest[shAt(primitive_idx, 12)];
                    const float3 c13 = sh_coefficients_rest[shAt(primitive_idx, 13)];
                    const float3 c14 = sh_coefficients_rest[shAt(primitive_idx, 14)];
                    apply_shN_grad(8, (0.59004358992664352f * y * (-3.0f * xx + yy)) * grad_color, fused_adam, primitive_idx);
                    apply_shN_grad(9, (2.8906114426405538f * xy * z) * grad_color, fused_adam, primitive_idx);
                    apply_shN_grad(10, (0.45704579946446572f * y * (1.0f - 5.0f * zz)) * grad_color, fused_adam, primitive_idx);
                    apply_shN_grad(11, (0.3731763325901154f * z * (5.0f * zz - 3.0f)) * grad_color, fused_adam, primitive_idx);
                    apply_shN_grad(12, (0.45704579946446572f * x * (1.0f - 5.0f * zz)) * grad_color, fused_adam, primitive_idx);
                    apply_shN_grad(13, (1.4453057213202769f * z * (xx - yy)) * grad_color, fused_adam, primitive_idx);
                    apply_shN_grad(14, (0.59004358992664352f * x * (-xx + 3.0f * yy)) * grad_color, fused_adam, primitive_idx);
                    grad_direction_x = grad_direction_x + (-3.5402615395598609f * xy) * c8 + (2.8906114426405538f * yz) * c9 + (0.45704579946446572f - 2.2852289973223288f * zz) * c12 + (2.8906114426405538f * xz) * c13 + (-1.7701307697799304f * xx + 1.7701307697799304f * yy) * c14;
                    grad_direction_y = grad_direction_y + (-1.7701307697799304f * xx + 1.7701307697799304f * yy) * c8 + (2.8906114426405538f * xz) * c9 + (0.45704579946446572f - 2.2852289973223288f * zz) * c10 + (-2.8906114426405538f * yz) * c13 + (3.5402615395598609f * xy) * c14;
                    grad_direction_z = grad_direction_z + (2.8906114426405538f * xy) * c9 + (-4.5704579946446566f * yz) * c10 + (5.597644988851731f * zz - 1.1195289977703462f) * c11 + (-4.5704579946446566f * xz) * c12 + (1.4453057213202769f * xx - 1.4453057213202769f * yy) * c13;
                }
            }

            const float3 grad_direction = make_float3(
                dot(grad_direction_x, grad_color),
                dot(grad_direction_y, grad_color),
                dot(grad_direction_z, grad_color));
            const float xx_raw = x_raw * x_raw, yy_raw = y_raw * y_raw, zz_raw = z_raw * z_raw;
            const float xy_raw = x_raw * y_raw, xz_raw = x_raw * z_raw, yz_raw = y_raw * z_raw;
            const float norm_sq = xx_raw + yy_raw + zz_raw;
            constexpr float NORM_SQ_GRAD_MIN = 1e-6f;
            constexpr float INV_NORM_CUBED_MAX = 1e6f;
            const float norm_sq_safe = fmaxf(norm_sq, NORM_SQ_GRAD_MIN);
            const float inv_norm_cubed = fminf(rsqrtf(norm_sq_safe * norm_sq_safe * norm_sq_safe), INV_NORM_CUBED_MAX);
            dcolor_dposition = make_float3(
                                   (yy_raw + zz_raw) * grad_direction.x - xy_raw * grad_direction.y - xz_raw * grad_direction.z,
                                   -xy_raw * grad_direction.x + (xx_raw + zz_raw) * grad_direction.y - yz_raw * grad_direction.z,
                                   -xz_raw * grad_direction.x - yz_raw * grad_direction.y + (xx_raw + yy_raw) * grad_direction.z) *
                               inv_norm_cubed;
        }
        return dcolor_dposition;
    }

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

} // namespace fast_lfs::rasterization::kernels

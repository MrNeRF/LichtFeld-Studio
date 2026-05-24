/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include "fused_adam_types.h"
#include "helper_math.h"
#include "rasterization_config.h"
#include "utils.h"

namespace fast_lfs::rasterization::kernels {

    // SH swizzle index: swizzled layout is [ceil(N/R), K_F4, R] of float4 where
    // R = config::sh_reorder_size and K_F4 = config::sh_rest_float4_per_primitive,
    // matching vksplat/vksplat/slang/spherical_harmonics.slang. Adjacent primitives in a warp hit
    // adjacent float4 slots -> a single 16B vector load per coefficient slot per lane.
    // Returns the float4 slot index (multiply by 4 to get the float offset).
    __device__ __host__ __forceinline__ unsigned int shSlotsForBases(unsigned int active_sh_bases) {
        const unsigned int rest_coeffs = active_sh_bases > 1u ? active_sh_bases - 1u : 0u;
        const unsigned int slots = (rest_coeffs * 3u + 3u) / 4u;
        return slots > config::sh_rest_float4_per_primitive ? config::sh_rest_float4_per_primitive : slots;
    }

    __device__ __host__ __forceinline__ unsigned int shAt(
        unsigned int primitive_idx,
        unsigned int float4_slot,
        unsigned int slots_per_primitive) {
        constexpr unsigned int R = config::sh_reorder_size;
        const unsigned int block = primitive_idx / R;
        const unsigned int lane = primitive_idx % R;
        return block * (slots_per_primitive * R) + float4_slot * R + lane;
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

    // Load all 15 shN coefficients (c0..c14) from the swizzled float4 buffer. Performs the
    // vksplat float4-pack shuffle (see sh_layout.cuh for the slot layout).
    // Up to ACTIVE_BASES selects which slots to read; remaining coeffs are left as float3(0).
    // Cost (SH3): 12 coalesced float4 loads per warp vs the old 15 misaligned float3 loads
    // (= 45 4-byte sectors per warp).
    __device__ inline void load_shN_coeffs(
        const float4* __restrict__ sh_f4,
        const uint primitive_idx,
        const uint active_sh_bases,
        const uint sh_layout_slots,
        float3 (&c)[15]) {
#pragma unroll
        for (int i = 0; i < 15; ++i)
            c[i] = make_float3(0.0f, 0.0f, 0.0f);

        if (active_sh_bases <= 1)
            return;

        const uint slots_per_primitive = sh_layout_slots;
        if (slots_per_primitive == 0u)
            return;
        const float4 a0 = sh_f4[shAt(primitive_idx, 0, slots_per_primitive)];
        const float4 a1 = sh_f4[shAt(primitive_idx, 1, slots_per_primitive)];
        const float4 a2 = sh_f4[shAt(primitive_idx, 2, slots_per_primitive)];
        c[0] = make_float3(a0.x, a0.y, a0.z);
        c[1] = make_float3(a0.w, a1.x, a1.y);
        c[2] = make_float3(a1.z, a1.w, a2.x);
        // c[3] also lives in a2 (a2.y, a2.z, a2.w). Read it now even if active_sh_bases==4 so the
        // unconditional load saves a branch; the value is unused upstream.
        c[3] = make_float3(a2.y, a2.z, a2.w);

        if (active_sh_bases <= 4)
            return;

        const float4 a3 = sh_f4[shAt(primitive_idx, 3, slots_per_primitive)];
        const float4 a4 = sh_f4[shAt(primitive_idx, 4, slots_per_primitive)];
        const float4 a5 = sh_f4[shAt(primitive_idx, 5, slots_per_primitive)];
        c[4] = make_float3(a3.x, a3.y, a3.z);
        c[5] = make_float3(a3.w, a4.x, a4.y);
        c[6] = make_float3(a4.z, a4.w, a5.x);
        c[7] = make_float3(a5.y, a5.z, a5.w);

        if (active_sh_bases <= 9)
            return;

        const float4 a6 = sh_f4[shAt(primitive_idx, 6, slots_per_primitive)];
        const float4 a7 = sh_f4[shAt(primitive_idx, 7, slots_per_primitive)];
        const float4 a8 = sh_f4[shAt(primitive_idx, 8, slots_per_primitive)];
        const float4 a9 = sh_f4[shAt(primitive_idx, 9, slots_per_primitive)];
        const float4 a10 = sh_f4[shAt(primitive_idx, 10, slots_per_primitive)];
        const float4 a11 = sh_f4[shAt(primitive_idx, 11, slots_per_primitive)];
        c[8] = make_float3(a6.x, a6.y, a6.z);
        c[9] = make_float3(a6.w, a7.x, a7.y);
        c[10] = make_float3(a7.z, a7.w, a8.x);
        c[11] = make_float3(a8.y, a8.z, a8.w);
        c[12] = make_float3(a9.x, a9.y, a9.z);
        c[13] = make_float3(a9.w, a10.x, a10.y);
        c[14] = make_float3(a10.z, a10.w, a11.x);
        // a11.y / a11.z / a11.w are tail padding (always zero).
    }

    __device__ inline float3 convert_sh_to_color(
        const float3* sh_coefficients_0,
        const float4* sh_coefficients_rest,
        const float3& position,
        const float3& cam_position,
        const uint primitive_idx,
        const uint active_sh_bases,
        const uint sh_layout_slots) {
        // computation adapted from https://github.com/NVlabs/tiny-cuda-nn/blob/212104156403bd87616c1a4f73a1c5f2c2e172a9/include/tiny-cuda-nn/common_device.h#L340
        float3 result = 0.5f + 0.28209479177387814f * sh_coefficients_0[primitive_idx];
        if (active_sh_bases > 1) {
            auto [x, y, z] = safe_normalize(position - cam_position);
            float3 c[15];
            load_shN_coeffs(sh_coefficients_rest, primitive_idx, active_sh_bases, sh_layout_slots, c);
            result = result + (-0.48860251190291987f * y) * c[0] + (0.48860251190291987f * z) * c[1] + (-0.48860251190291987f * x) * c[2];
            if (active_sh_bases > 4) {
                const float xx = x * x, yy = y * y, zz = z * z;
                const float xy = x * y, xz = x * z, yz = y * z;
                result = result + (1.0925484305920792f * xy) * c[3] + (-1.0925484305920792f * yz) * c[4] + (0.94617469575755997f * zz - 0.31539156525251999f) * c[5] + (-1.0925484305920792f * xz) * c[6] + (0.54627421529603959f * xx - 0.54627421529603959f * yy) * c[7];
                if (active_sh_bases > 9) {
                    result = result + (0.59004358992664352f * y * (-3.0f * xx + yy)) * c[8] + (2.8906114426405538f * xy * z) * c[9] + (0.45704579946446572f * y * (1.0f - 5.0f * zz)) * c[10] + (0.3731763325901154f * z * (5.0f * zz - 3.0f)) * c[11] + (0.45704579946446572f * x * (1.0f - 5.0f * zz)) * c[12] + (1.4453057213202769f * z * (xx - yy)) * c[13] + (0.59004358992664352f * x * (-xx + 3.0f * yy)) * c[14];
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

    // Single float4 Adam step. Reads/writes one float4 slot in each of param/exp_avg/exp_avg_sq.
    // Used by the shN packed-grad path so that 32 warp lanes hit 32 consecutive float4 slots
    // (perfectly coalesced 16B vector accesses, vs the old per-channel 4B scalar accesses).
    __device__ inline void adam_step_f4(
        const float4 grad,
        const FusedAdamParam& param,
        const uint float4_slot,
        const float beta1,
        const float beta2,
        const float eps) {
        const uint float_idx = float4_slot * 4u;
        if (!param.enabled || float_idx + 3u >= static_cast<uint>(param.n_elements))
            return;
        float4* p_f4 = reinterpret_cast<float4*>(param.param);
        float4* m1_f4 = reinterpret_cast<float4*>(param.exp_avg);
        float4* m2_f4 = reinterpret_cast<float4*>(param.exp_avg_sq);

        const float4 m1_prev = m1_f4[float4_slot];
        const float4 m2_prev = m2_f4[float4_slot];
        const float4 p_prev = p_f4[float4_slot];
        const float4 grad_sq = make_float4(grad.x * grad.x, grad.y * grad.y, grad.z * grad.z, grad.w * grad.w);
        const float4 m1 = make_float4(
            fmaf(beta1, m1_prev.x - grad.x, grad.x),
            fmaf(beta1, m1_prev.y - grad.y, grad.y),
            fmaf(beta1, m1_prev.z - grad.z, grad.z),
            fmaf(beta1, m1_prev.w - grad.w, grad.w));
        const float4 m2 = make_float4(
            fmaf(beta2, m2_prev.x - grad_sq.x, grad_sq.x),
            fmaf(beta2, m2_prev.y - grad_sq.y, grad_sq.y),
            fmaf(beta2, m2_prev.z - grad_sq.z, grad_sq.z),
            fmaf(beta2, m2_prev.w - grad_sq.w, grad_sq.w));
        const float4 p_new = make_float4(
            p_prev.x - param.step_size * m1.x / (sqrtf(m2.x) * param.bias_correction2_sqrt_rcp + eps),
            p_prev.y - param.step_size * m1.y / (sqrtf(m2.y) * param.bias_correction2_sqrt_rcp + eps),
            p_prev.z - param.step_size * m1.z / (sqrtf(m2.z) * param.bias_correction2_sqrt_rcp + eps),
            p_prev.w - param.step_size * m1.w / (sqrtf(m2.w) * param.bias_correction2_sqrt_rcp + eps));
        p_f4[float4_slot] = p_new;
        m1_f4[float4_slot] = m1;
        m2_f4[float4_slot] = m2;
    }

    // Apply 15 float3 grad coefficients (c0..c14) to the swizzled shN Adam state.
    // Packs the 15 float3 grads into 12 float4 grads using the same shuffle as the read path,
    // then runs adam_step_f4 on each slot. Adam decay on the 3 tail-padding floats of slot 11
    // is mathematically a no-op (their state stays at zero from initialisation).
    // n_slots_to_update controls how many of the 12 slots to write, derived from active SH:
    //     active_sh_bases <= 1 : 0 slots (caller must skip)
    //     active_sh_bases <= 4 : 3 slots  (covers c0..c3.x grads; slot 2's c3 lane is 0 grad)
    //     active_sh_bases <= 9 : 6 slots  (covers c0..c7 grads)
    //     active_sh_bases > 9  : 12 slots (covers c0..c14 grads)
    __device__ inline void apply_shN_grads_packed(
        const FusedAdamSettings& fused_adam,
        const uint primitive_idx,
        const float3 (&g)[15],
        const uint n_slots_to_update,
        const uint sh_layout_slots) {
        const FusedAdamParam& p = fused_adam.shN;
        if (!p.enabled || n_slots_to_update == 0u || sh_layout_slots == 0u)
            return;

#pragma unroll
        for (uint k = 0; k < 12u; ++k) {
            if (k >= n_slots_to_update)
                break;
            float4 gk;
            // Same shuffle as load_shN_coeffs, but for write.
            switch (k) {
            case 0: gk = make_float4(g[0].x, g[0].y, g[0].z, g[1].x); break;
            case 1: gk = make_float4(g[1].y, g[1].z, g[2].x, g[2].y); break;
            case 2: gk = make_float4(g[2].z, g[3].x, g[3].y, g[3].z); break;
            case 3: gk = make_float4(g[4].x, g[4].y, g[4].z, g[5].x); break;
            case 4: gk = make_float4(g[5].y, g[5].z, g[6].x, g[6].y); break;
            case 5: gk = make_float4(g[6].z, g[7].x, g[7].y, g[7].z); break;
            case 6: gk = make_float4(g[8].x, g[8].y, g[8].z, g[9].x); break;
            case 7: gk = make_float4(g[9].y, g[9].z, g[10].x, g[10].y); break;
            case 8: gk = make_float4(g[10].z, g[11].x, g[11].y, g[11].z); break;
            case 9: gk = make_float4(g[12].x, g[12].y, g[12].z, g[13].x); break;
            case 10: gk = make_float4(g[13].y, g[13].z, g[14].x, g[14].y); break;
            case 11: gk = make_float4(g[14].z, 0.0f, 0.0f, 0.0f); break;
            default: gk = make_float4(0.0f, 0.0f, 0.0f, 0.0f); break;
            }
            adam_step_f4(gk, p, shAt(primitive_idx, k, sh_layout_slots),
                         fused_adam.beta1, fused_adam.beta2, fused_adam.eps);
        }
    }

    template <int ACTIVE_SH_BASES>
    __device__ inline float3 convert_sh_to_color_backward(
        const float4* sh_coefficients_rest,
        float3* grad_color_helper,
        const FusedAdamSettings& fused_adam,
        const float3& position,
        const float3& cam_position,
        const uint primitive_idx,
        const uint sh_layout_slots) {
        // computation adapted from https://github.com/NVlabs/tiny-cuda-nn/blob/212104156403bd87616c1a4f73a1c5f2c2e172a9/include/tiny-cuda-nn/common_device.h#L340
        const float3 grad_color = grad_color_helper[primitive_idx];
        const float3 dL_dsh0 = 0.28209479177387814f * grad_color;
        adam_step_helper(dL_dsh0.x, fused_adam.sh0, primitive_idx, 0, fused_adam.beta1, fused_adam.beta2, fused_adam.eps);
        adam_step_helper(dL_dsh0.y, fused_adam.sh0, primitive_idx, 1, fused_adam.beta1, fused_adam.beta2, fused_adam.eps);
        adam_step_helper(dL_dsh0.z, fused_adam.sh0, primitive_idx, 2, fused_adam.beta1, fused_adam.beta2, fused_adam.eps);
        float3 dcolor_dposition = make_float3(0.0f);
        if constexpr (ACTIVE_SH_BASES > 1) {
            auto [x_raw, y_raw, z_raw] = position - cam_position;
            auto [x, y, z] = safe_normalize(make_float3(x_raw, y_raw, z_raw));

            // Load all coeffs we need via the float4-packed shuffle.
            float3 c[15];
            load_shN_coeffs(sh_coefficients_rest, primitive_idx, ACTIVE_SH_BASES, sh_layout_slots, c);

            // Compute grad-of-coeff (15 float3 grads); inactive lanes left at 0.
            float3 g[15];
#pragma unroll
            for (int i = 0; i < 15; ++i)
                g[i] = make_float3(0.0f, 0.0f, 0.0f);

            g[0] = (-0.48860251190291987f * y) * grad_color;
            g[1] = (0.48860251190291987f * z) * grad_color;
            g[2] = (-0.48860251190291987f * x) * grad_color;
            float3 grad_direction_x = -0.48860251190291987f * c[2];
            float3 grad_direction_y = -0.48860251190291987f * c[0];
            float3 grad_direction_z = 0.48860251190291987f * c[1];
            if constexpr (ACTIVE_SH_BASES > 4) {
                const float xx = x * x, yy = y * y, zz = z * z;
                const float xy = x * y, xz = x * z, yz = y * z;
                g[3] = (1.0925484305920792f * xy) * grad_color;
                g[4] = (-1.0925484305920792f * yz) * grad_color;
                g[5] = (0.94617469575755997f * zz - 0.31539156525251999f) * grad_color;
                g[6] = (-1.0925484305920792f * xz) * grad_color;
                g[7] = (0.54627421529603959f * xx - 0.54627421529603959f * yy) * grad_color;
                grad_direction_x = grad_direction_x + (1.0925484305920792f * y) * c[3] + (-1.0925484305920792f * z) * c[6] + (1.0925484305920792f * x) * c[7];
                grad_direction_y = grad_direction_y + (1.0925484305920792f * x) * c[3] + (-1.0925484305920792f * z) * c[4] + (-1.0925484305920792f * y) * c[7];
                grad_direction_z = grad_direction_z + (-1.0925484305920792f * y) * c[4] + (1.8923493915151202f * z) * c[5] + (-1.0925484305920792f * x) * c[6];
                if constexpr (ACTIVE_SH_BASES > 9) {
                    g[8] = (0.59004358992664352f * y * (-3.0f * xx + yy)) * grad_color;
                    g[9] = (2.8906114426405538f * xy * z) * grad_color;
                    g[10] = (0.45704579946446572f * y * (1.0f - 5.0f * zz)) * grad_color;
                    g[11] = (0.3731763325901154f * z * (5.0f * zz - 3.0f)) * grad_color;
                    g[12] = (0.45704579946446572f * x * (1.0f - 5.0f * zz)) * grad_color;
                    g[13] = (1.4453057213202769f * z * (xx - yy)) * grad_color;
                    g[14] = (0.59004358992664352f * x * (-xx + 3.0f * yy)) * grad_color;
                    grad_direction_x = grad_direction_x + (-3.5402615395598609f * xy) * c[8] + (2.8906114426405538f * yz) * c[9] + (0.45704579946446572f - 2.2852289973223288f * zz) * c[12] + (2.8906114426405538f * xz) * c[13] + (-1.7701307697799304f * xx + 1.7701307697799304f * yy) * c[14];
                    grad_direction_y = grad_direction_y + (-1.7701307697799304f * xx + 1.7701307697799304f * yy) * c[8] + (2.8906114426405538f * xz) * c[9] + (0.45704579946446572f - 2.2852289973223288f * zz) * c[10] + (-2.8906114426405538f * yz) * c[13] + (3.5402615395598609f * xy) * c[14];
                    grad_direction_z = grad_direction_z + (2.8906114426405538f * xy) * c[9] + (-4.5704579946446566f * yz) * c[10] + (5.597644988851731f * zz - 1.1195289977703462f) * c[11] + (-4.5704579946446566f * xz) * c[12] + (1.4453057213202769f * xx - 1.4453057213202769f * yy) * c[13];
                }
            }

            // How many float4 slots cover the active coeffs:
            //   bases > 9 : 12 slots cover c0..c14
            //   bases > 4 : 6 slots cover c0..c7
            //   bases > 1 : 3 slots cover c0..c2 (slot 2's c3 lane has 0 grad -> harmless decay)
            constexpr uint n_slots = (ACTIVE_SH_BASES > 9) ? 12u : (ACTIVE_SH_BASES > 4) ? 6u
                                                                                         : 3u;
            apply_shN_grads_packed(fused_adam, primitive_idx, g, n_slots, sh_layout_slots);

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

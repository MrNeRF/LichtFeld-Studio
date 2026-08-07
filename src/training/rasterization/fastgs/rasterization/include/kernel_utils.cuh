/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include "fused_adam_types.h"
#include "helper_math.h"
#include "lfs/core/warp_reduce.cuh"
#include "lfs/training/joint_adam_codec.cuh"
#include "lfs/training/sh_value_codec.cuh"
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
    //
    // Phase 2.1 q16: when sh_value_bounds != nullptr, `sh_f4` is actually a bitcast of
    // uint16 cell-linear storage (pad-dropped: n_cells = coeffs_rest*3). Decode in registers
    // at the use site — do NOT materialise a float workspace (FIX-2.2 live-range lesson).
    __device__ inline void load_shN_coeffs(
        const float4* __restrict__ sh_f4,
        const uint primitive_idx,
        const uint active_sh_bases,
        const uint sh_layout_slots,
        float3 (&c)[15],
        const float2* __restrict__ sh_value_bounds = nullptr,
        const uint sh_value_n_cells = 0u) {
#pragma unroll
        for (int i = 0; i < 15; ++i)
            c[i] = make_float3(0.0f, 0.0f, 0.0f);

        if (active_sh_bases <= 1)
            return;

        // ---- q16 decode-in-registers path (pad-dropped cell-linear) ----
        if (sh_value_bounds != nullptr && sh_value_n_cells > 0u) {
            using DC = lfs::training::sh_value::DeviceCodec16;
            const uint16_t* sh_u16 = reinterpret_cast<const uint16_t*>(sh_f4);
            const float2 mm = sh_value_bounds[primitive_idx / 256u];
            const uint n_cells = sh_value_n_cells;
#pragma unroll
            for (int i = 0; i < 15; ++i) {
                const uint base = static_cast<uint>(i) * 3u;
                if (base + 2u >= n_cells)
                    break;
                if (i >= 3 && active_sh_bases <= 4)
                    break;
                if (i >= 8 && active_sh_bases <= 9)
                    break;
                c[i] = make_float3(
                    DC::decode(sh_u16[lfs::training::sh_value::shAtU16(primitive_idx, base + 0, n_cells)],
                               mm.x, mm.y),
                    DC::decode(sh_u16[lfs::training::sh_value::shAtU16(primitive_idx, base + 1, n_cells)],
                               mm.x, mm.y),
                    DC::decode(sh_u16[lfs::training::sh_value::shAtU16(primitive_idx, base + 2, n_cells)],
                               mm.x, mm.y));
            }
            return;
        }

        // ---- fp32 float4 path ----
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
        const uint sh_layout_slots,
        const float2* __restrict__ sh_value_bounds = nullptr,
        const uint sh_value_n_cells = 0u) {
        // computation adapted from https://github.com/NVlabs/tiny-cuda-nn/blob/212104156403bd87616c1a4f73a1c5f2c2e172a9/include/tiny-cuda-nn/common_device.h#L340
        float3 result = 0.5f + 0.28209479177387814f * sh_coefficients_0[primitive_idx];
        if (active_sh_bases > 1) {
            const float3 direction = safe_normalize(position - cam_position);
            const float x = direction.x;
            const float y = direction.y;
            const float z = direction.z;
            float3 c[15];
            load_shN_coeffs(sh_coefficients_rest, primitive_idx, active_sh_bases, sh_layout_slots, c,
                            sh_value_bounds, sh_value_n_cells);
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

    // Quantised Adam moment helpers (see fast_lfs::optimizer::kernels::adam for the rationale).
    //   m: signed int8 around zero-point 128, scale = max|m| / 127.
    //   v: quantised sqrt(v), scale = sqrt(max v) / 255.
    constexpr int FUSED_MOMENT_ZERO_POINT = 128;
    // Max contiguous (non-shN) attributes per primitive (rotation = 4 is the largest).
    constexpr int MAX_FUSED_ADAM_ATTRIBUTES = 4;

    __device__ inline float dequant_m(const std::uint8_t q, const float scale) {
        return scale == 0.0f ? 0.0f : (static_cast<int>(q) - FUSED_MOMENT_ZERO_POINT) * scale;
    }

    __device__ inline std::uint8_t quantize_m(const float value, const float scale) {
        if (scale == 0.0f)
            return static_cast<std::uint8_t>(FUSED_MOMENT_ZERO_POINT);
        const int q = static_cast<int>(roundf(value / scale)) + FUSED_MOMENT_ZERO_POINT;
        return static_cast<std::uint8_t>(min(255, max(0, q)));
    }

    __device__ inline float dequant_sqrt_v(const std::uint8_t q, const float scale) {
        return scale == 0.0f ? 0.0f : static_cast<float>(q) * scale;
    }

    __device__ inline std::uint8_t quantize_sqrt_v(const float v, const float scale) {
        if (scale == 0.0f)
            return 0;
        const float s = sqrtf(fmaxf(v, 0.0f));
        const int q = static_cast<int>(roundf(s / scale));
        if (v > 0.0f && q == 0)
            return 1;
        return static_cast<std::uint8_t>(min(255, max(0, q)));
    }

    // Joint (u, log_s) Adam step for a contiguous param row. ALL threads in the CUDA
    // block must call this together (block_reduce + __syncthreads). Disabled params
    // early-out uniformly (same `enabled` for every thread).
    //
    // Frozen / crop-damped rows with zero lr skip the moment+param update but still
    // re-encode their current (u,log_s) under the new block bounds so codes stay valid.
    template <int BITS>
    __device__ inline void adam_step_row_joint(
        const float* grads,
        const FusedAdamParam& param,
        const uint primitive_idx,
        const uint row_elements,
        const float beta1,
        const float beta2,
        const float eps) {
        using C = lfs::training::joint_adam::DeviceCodec<BITS>;
        constexpr float kInf = 1e30f;

        float row_step_size = param.step_size;
        bool apply_step = true;
        bool touch = param.enabled && param.n_attributes > 0 &&
                     param.joint_packed != nullptr && param.joint_bounds != nullptr;
        if (touch && param.frozen_mask != nullptr &&
            primitive_idx < static_cast<uint>(param.frozen_mask_size) &&
            param.frozen_mask[primitive_idx]) {
            if (param.frozen_lr_scale == 0.0f)
                apply_step = false;
            else
                row_step_size *= param.frozen_lr_scale;
        }
        if (touch && param.crop_damping_mask != nullptr &&
            primitive_idx < static_cast<uint>(param.crop_damping_mask_size) &&
            param.crop_damping_mask[primitive_idx]) {
            if (param.cropbox_lr_scale == 0.0f)
                apply_step = false;
            else
                row_step_size *= param.cropbox_lr_scale;
        }

        const uint n_attr = param.n_attributes > 0 ? static_cast<uint>(param.n_attributes) : 1u;
        const uint base = primitive_idx * n_attr;
        if (touch && base >= static_cast<uint>(param.n_elements))
            touch = false;
        const uint row = touch ? min(n_attr, static_cast<uint>(param.n_elements) - base) : 0u;
        const uint active = min(row_elements, row);

        const int bidx = static_cast<int>(blockIdx.x);
        const float4 old_mm = touch
                                  ? *reinterpret_cast<const float4*>(param.joint_bounds + 4 * bidx)
                                  : make_float4(0.0f, 0.0f, 0.0f, 0.0f);

        float local_u_min = kInf, local_u_max = -kInf;
        float local_s_min = kInf, local_s_max = -kInf;
        float us_u[MAX_FUSED_ADAM_ATTRIBUTES];
        float us_s[MAX_FUSED_ADAM_ATTRIBUTES];

        if (touch) {
            for (uint i = 0; i < row; ++i) {
                // Packed layout: [N, n_attr * bytes_per_cell] — cell = base+i
                const int64_t cell = static_cast<int64_t>(base + i);
                const float2 mv = C::decode_g1g2(param.joint_packed, cell, old_mm);
                float m = mv.x;
                float v = mv.y;
                if (apply_step) {
                    const float grad = (i < active) ? grads[i] : 0.0f;
                    m = beta1 * mv.x + (1.0f - beta1) * grad;
                    v = beta2 * mv.y + (1.0f - beta2) * grad * grad;
                    if (i < active) {
                        const float denom = sqrtf(v) * param.bias_correction2_sqrt_rcp + eps;
                        param.param[base + i] -= row_step_size * m / denom;
                    }
                }
                const float2 prim = C::g1g2_to_us(m, v);
                us_u[i] = prim.x;
                us_s[i] = prim.y;
                local_u_min = fminf(local_u_min, prim.x);
                local_u_max = fmaxf(local_u_max, prim.x);
                local_s_min = fminf(local_s_min, prim.y);
                local_s_max = fmaxf(local_s_max, prim.y);
            }
        }

        // FIX-2.2 F2: fused min4 on {u_min,-u_max,s_min,-s_max} → 1 barrier
        // (plus the sm_bounds sync below = 2/section) instead of 4 separate
        // min/max reduces that raced on static __shared__.
        const float4 red = lfs::core::warp_ops::block_reduce_min4(
            make_float4(local_u_min, -local_u_max, local_s_min, -local_s_max));
        const float u_min = red.x;
        const float u_max = -red.y;
        const float s_min = red.z;
        const float s_max = -red.w;

        __shared__ float4 sm_bounds;
        if (threadIdx.x == 0) {
            float4 nb;
            if (u_min > u_max) {
                nb = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
            } else {
                nb = make_float4(u_min, u_max, s_min, s_max);
            }
            sm_bounds = nb;
            if (param.enabled && param.joint_bounds != nullptr) {
                *reinterpret_cast<float4*>(param.joint_bounds + 4 * bidx) = nb;
            }
        }
        __syncthreads();
        const float4 new_mm = sm_bounds;
        // FIX-2.2 F3: hoist block-uniform inv ranges (2 fdiv → FMA per cell).
        const float inv_u_range = 1.0f / fmaxf(new_mm.y - new_mm.x, lfs::training::joint_adam::kEpsDevice);
        const float inv_s_range = 1.0f / fmaxf(new_mm.w - new_mm.z, lfs::training::joint_adam::kEpsDevice);

        if (touch) {
            for (uint i = 0; i < row; ++i) {
                C::encode_us(param.joint_packed, static_cast<int64_t>(base + i),
                             us_u[i], us_s[i], new_mm.x, new_mm.z, inv_u_range, inv_s_range);
            }
        }
    }

    // Two-pass quantised Adam step over a contiguous [n_attributes] row of one primitive
    // (means / sh0 / scaling / rotation / opacity). Pass 1 derives the per-primitive scales
    // from max|m| / max(v); pass 2 writes the param update and requantises. `grads` holds
    // `row_elements` values; any resident attributes beyond that get pure momentum decay.
    // Joint codec path uses block bounds (all threads must call when joint_bits != 0).
    __device__ inline void adam_step_row(
        const float* grads,
        const FusedAdamParam& param,
        const uint primitive_idx,
        const uint row_elements,
        const float beta1,
        const float beta2,
        const float eps) {
        if (param.joint_bits == 16) {
            adam_step_row_joint<16>(grads, param, primitive_idx, row_elements, beta1, beta2, eps);
            return;
        }
        if (param.joint_bits == 8) {
            adam_step_row_joint<8>(grads, param, primitive_idx, row_elements, beta1, beta2, eps);
            return;
        }
        if (!param.enabled || param.n_attributes <= 0)
            return;
        float row_step_size = param.step_size;
        if (param.frozen_mask != nullptr &&
            primitive_idx < static_cast<uint>(param.frozen_mask_size) &&
            param.frozen_mask[primitive_idx]) {
            if (param.frozen_lr_scale == 0.0f)
                return;
            row_step_size *= param.frozen_lr_scale;
        }
        if (param.crop_damping_mask != nullptr &&
            primitive_idx < static_cast<uint>(param.crop_damping_mask_size) &&
            param.crop_damping_mask[primitive_idx]) {
            if (param.cropbox_lr_scale == 0.0f)
                return;
            row_step_size *= param.cropbox_lr_scale;
        }
        const uint n_attr = static_cast<uint>(param.n_attributes);
        const uint base = primitive_idx * n_attr;
        if (base >= static_cast<uint>(param.n_elements))
            return;
        const uint row = min(n_attr, static_cast<uint>(param.n_elements) - base);
        const uint active = min(row_elements, row);

        const float old_m_scale = param.exp_avg_scale[primitive_idx];
        const float old_v_scale = param.exp_avg_sq_scale[primitive_idx];

        float max_abs_m = 0.0f;
        float max_v = 0.0f;
        for (uint i = 0; i < row; ++i) {
            const float old_m = dequant_m(param.exp_avg_q[base + i], old_m_scale);
            const float sqrt_old_v = dequant_sqrt_v(param.exp_avg_sq_q[base + i], old_v_scale);
            const float old_v = sqrt_old_v * sqrt_old_v;
            const float grad = (i < active) ? grads[i] : 0.0f;
            const float m = beta1 * old_m + (1.0f - beta1) * grad;
            const float v = beta2 * old_v + (1.0f - beta2) * grad * grad;
            max_abs_m = fmaxf(max_abs_m, fabsf(m));
            max_v = fmaxf(max_v, v);
        }

        const float new_m_scale = max_abs_m > 0.0f ? max_abs_m / 127.0f : 0.0f;
        const float new_v_scale = max_v > 0.0f ? sqrtf(max_v) / 255.0f : 0.0f;

        for (uint i = 0; i < row; ++i) {
            const float old_m = dequant_m(param.exp_avg_q[base + i], old_m_scale);
            const float sqrt_old_v = dequant_sqrt_v(param.exp_avg_sq_q[base + i], old_v_scale);
            const float old_v = sqrt_old_v * sqrt_old_v;
            const float grad = (i < active) ? grads[i] : 0.0f;
            const float m = beta1 * old_m + (1.0f - beta1) * grad;
            const float v = beta2 * old_v + (1.0f - beta2) * grad * grad;
            if (i < active) {
                const float denom = sqrtf(v) * param.bias_correction2_sqrt_rcp + eps;
                param.param[base + i] -= row_step_size * m / denom;
            }
            param.exp_avg_q[base + i] = quantize_m(m, new_m_scale);
            param.exp_avg_sq_q[base + i] = quantize_sqrt_v(v, new_v_scale);
        }

        param.exp_avg_scale[primitive_idx] = new_m_scale;
        param.exp_avg_sq_scale[primitive_idx] = new_v_scale;
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

    // L = weight * mean_over_primitives(exp(min raw scale)): flattens splats
    // along their thinnest axis so the min-axis normal is well-defined
    // (PGSR-style). The argmin is treated as a constant.
    __device__ inline void add_flatten_regularization_grads(
        const FusedAdamSettings& fused_adam,
        const FusedAdamParam& param,
        const uint scale_base,
        float (&grads)[3]) {
        if (fused_adam.flatten_reg_weight <= 0.0f || param.n_elements <= 0)
            return;
        const float s0 = param.param[scale_base];
        const float s1 = param.param[scale_base + 1];
        const float s2 = param.param[scale_base + 2];
        const uint min_axis = (s0 <= s1 && s0 <= s2) ? 0u : (s1 <= s2) ? 1u
                                                                       : 2u;
        const float s_min = min_axis == 0u ? s0 : (min_axis == 1u ? s1 : s2);
        // n_elements counts all three scale channels per primitive.
        grads[min_axis] += 3.0f * fused_adam.flatten_reg_weight * expf(s_min) /
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

    // Shuffle the 15 float3 grad coefficients (c0..c14) into float4 slot k, matching the
    // swizzled load_shN_coeffs packing. Slot 11's tail lanes are padding (0 grad).
    __device__ inline float4 shN_grad_for_slot(const float3 (&g)[15], const uint k) {
        switch (k) {
        case 0: return make_float4(g[0].x, g[0].y, g[0].z, g[1].x);
        case 1: return make_float4(g[1].y, g[1].z, g[2].x, g[2].y);
        case 2: return make_float4(g[2].z, g[3].x, g[3].y, g[3].z);
        case 3: return make_float4(g[4].x, g[4].y, g[4].z, g[5].x);
        case 4: return make_float4(g[5].y, g[5].z, g[6].x, g[6].y);
        case 5: return make_float4(g[6].z, g[7].x, g[7].y, g[7].z);
        case 6: return make_float4(g[8].x, g[8].y, g[8].z, g[9].x);
        case 7: return make_float4(g[9].y, g[9].z, g[10].x, g[10].y);
        case 8: return make_float4(g[10].z, g[11].x, g[11].y, g[11].z);
        case 9: return make_float4(g[12].x, g[12].y, g[12].z, g[13].x);
        case 10: return make_float4(g[13].y, g[13].z, g[14].x, g[14].y);
        case 11: return make_float4(g[14].z, 0.0f, 0.0f, 0.0f);
        default: return make_float4(0.0f, 0.0f, 0.0f, 0.0f);
        }
    }

    // Joint 8-bit shN Adam (swizzled float cells × 2 B). ALL threads must call when joint.
    // Walks ALL layout slots (not only active SH) so inactive bands re-encode under new
    // bounds and stay true-zero when their codes represent (u,log_s)=(0,0).
    //
    // Phase 2.1: when p.sh_value_bits==16, param is pad-dropped uint16 codes; decode in
    // registers, Adam-update, then single re-encode after value bounds reduce. Moments still
    // use the float4-slot cell indexing (48 cells). Value re-encode is the single writer.
    __device__ inline void apply_shN_grads_packed_joint(
        const FusedAdamSettings& fused_adam,
        const uint primitive_idx,
        const float3 (&g)[15],
        const uint n_slots_to_update,
        const uint sh_layout_slots) {
        using C = lfs::training::joint_adam::DeviceCodec<8>;
        using VC = lfs::training::sh_value::DeviceCodec16;
        constexpr float kInf = 1e30f;
        const FusedAdamParam& p = fused_adam.shN;
        const bool value_q16 = p.sh_value_bits == 16 && p.sh_value_bounds != nullptr &&
                               p.sh_value_n_cells > 0;
        const uint n_value_cells = value_q16 ? static_cast<uint>(p.sh_value_n_cells) : 0u;

        float row_step_size = p.step_size;
        bool apply_step = true;
        bool touch = p.enabled && sh_layout_slots > 0u &&
                     p.joint_packed != nullptr && p.joint_bounds != nullptr;
        if (touch && p.frozen_mask != nullptr &&
            primitive_idx < static_cast<uint>(p.frozen_mask_size) &&
            p.frozen_mask[primitive_idx]) {
            if (p.frozen_lr_scale == 0.0f)
                apply_step = false;
            else
                row_step_size *= p.frozen_lr_scale;
        }
        if (touch && p.crop_damping_mask != nullptr &&
            primitive_idx < static_cast<uint>(p.crop_damping_mask_size) &&
            p.crop_damping_mask[primitive_idx]) {
            if (p.cropbox_lr_scale == 0.0f)
                apply_step = false;
            else
                row_step_size *= p.cropbox_lr_scale;
        }

        const int bidx = static_cast<int>(blockIdx.x);
        const float4 old_mm = touch
                                  ? *reinterpret_cast<const float4*>(p.joint_bounds + 4 * bidx)
                                  : make_float4(0.0f, 0.0f, 0.0f, 0.0f);
        const float2 old_vmm = value_q16
                                   ? *reinterpret_cast<const float2*>(p.sh_value_bounds + 2 * bidx)
                                   : make_float2(0.0f, 0.0f);
        const float beta1 = fused_adam.beta1, beta2 = fused_adam.beta2, eps = fused_adam.eps;
        float4* param4 = (!value_q16 && touch) ? reinterpret_cast<float4*>(p.param) : nullptr;
        uint16_t* param_u16 = (value_q16 && touch) ? reinterpret_cast<uint16_t*>(p.param) : nullptr;

        float local_u_min = kInf, local_u_max = -kInf;
        float local_s_min = kInf, local_s_max = -kInf;
        float local_v_min = kInf, local_v_max = -kInf;
        // Up to 12 slots × 4 cells; store (u,log_s) for encode after bounds reduce.
        float us_u[48], us_s[48];
        // Updated param values (q16 path holds them until value bounds reduce).
        float pval[48];

        if (touch) {
            int n_cells_local = 0;
#pragma unroll
            for (uint k = 0; k < 12u; ++k) {
                if (k >= sh_layout_slots)
                    break;
                const uint slot = shAt(primitive_idx, k, sh_layout_slots);
                const bool active_slot = k < n_slots_to_update;
                const float4 gk = active_slot ? shN_grad_for_slot(g, k)
                                              : make_float4(0.0f, 0.0f, 0.0f, 0.0f);
                const float gc[4] = {gk.x, gk.y, gk.z, gk.w};
                float pc[4];
                if (value_q16) {
#pragma unroll
                    for (int c = 0; c < 4; ++c) {
                        const uint cell_lin = k * 4u + static_cast<uint>(c);
                        if (cell_lin < n_value_cells) {
                            pc[c] = VC::decode(
                                param_u16[lfs::training::sh_value::shAtU16(
                                    primitive_idx, cell_lin, n_value_cells)],
                                old_vmm.x, old_vmm.y);
                        } else {
                            pc[c] = 0.0f;
                        }
                    }
                } else {
                    float4 pv = param4[slot];
                    pc[0] = pv.x;
                    pc[1] = pv.y;
                    pc[2] = pv.z;
                    pc[3] = pv.w;
                }
#pragma unroll
                for (int c = 0; c < 4; ++c) {
                    const int64_t cell = static_cast<int64_t>(slot) * 4 + c;
                    const float2 mv = C::decode_g1g2(p.joint_packed, cell, old_mm);
                    float m = mv.x;
                    float v = mv.y;
                    if (apply_step) {
                        m = beta1 * mv.x + (1.0f - beta1) * gc[c];
                        v = beta2 * mv.y + (1.0f - beta2) * gc[c] * gc[c];
                        if (active_slot) {
                            const float denom = sqrtf(v) * p.bias_correction2_sqrt_rcp + eps;
                            pc[c] -= row_step_size * m / denom;
                        }
                    }
                    const float2 prim = C::g1g2_to_us(m, v);
                    us_u[n_cells_local] = prim.x;
                    us_s[n_cells_local] = prim.y;
                    pval[n_cells_local] = pc[c];
                    if (value_q16) {
                        const uint cell_lin = k * 4u + static_cast<uint>(c);
                        if (cell_lin < n_value_cells) {
                            local_v_min = fminf(local_v_min, pc[c]);
                            local_v_max = fmaxf(local_v_max, pc[c]);
                        }
                    }
                    ++n_cells_local;
                    local_u_min = fminf(local_u_min, prim.x);
                    local_u_max = fmaxf(local_u_max, prim.x);
                    local_s_min = fminf(local_s_min, prim.y);
                    local_s_max = fmaxf(local_s_max, prim.y);
                }
                if (!value_q16 && apply_step && active_slot)
                    param4[slot] = make_float4(pc[0], pc[1], pc[2], pc[3]);
            }
            (void)n_cells_local;
        }

        // FIX-2.2 F2: fused min4 bounds reduce for Adam moment (u,log_s).
        const float4 red = lfs::core::warp_ops::block_reduce_min4(
            make_float4(local_u_min, -local_u_max, local_s_min, -local_s_max));
        const float u_min = red.x;
        const float u_max = -red.y;
        const float s_min = red.z;
        const float s_max = -red.w;
        // Phase 2.1 value bounds (separate from moment bounds).
        const float v_min = value_q16 ? lfs::core::warp_ops::block_reduce_min(local_v_min) : 0.0f;
        const float v_max = value_q16 ? lfs::core::warp_ops::block_reduce_max(local_v_max) : 0.0f;

        __shared__ float4 sm_bounds;
        __shared__ float2 sm_vbounds;
        if (threadIdx.x == 0) {
            float4 nb = (u_min > u_max) ? make_float4(0.0f, 0.0f, 0.0f, 0.0f)
                                        : make_float4(u_min, u_max, s_min, s_max);
            sm_bounds = nb;
            if (p.enabled && p.joint_bounds != nullptr) {
                *reinterpret_cast<float4*>(p.joint_bounds + 4 * bidx) = nb;
            }
            if (value_q16) {
                float2 vb = (v_min > v_max) ? make_float2(0.0f, 0.0f)
                                            : make_float2(v_min, v_max);
                sm_vbounds = vb;
                *reinterpret_cast<float2*>(p.sh_value_bounds + 2 * bidx) = vb;
            }
        }
        __syncthreads();
        const float4 new_mm = sm_bounds;
        const float2 new_vmm = value_q16 ? sm_vbounds : make_float2(0.0f, 0.0f);
        // FIX-2.2 F3: hoist block-uniform inv ranges (2 fdiv → FMA per cell).
        const float inv_u_range = 1.0f / fmaxf(new_mm.y - new_mm.x, lfs::training::joint_adam::kEpsDevice);
        const float inv_s_range = 1.0f / fmaxf(new_mm.w - new_mm.z, lfs::training::joint_adam::kEpsDevice);

        if (touch) {
            int ci = 0;
#pragma unroll
            for (uint k = 0; k < 12u; ++k) {
                if (k >= sh_layout_slots)
                    break;
                const uint slot = shAt(primitive_idx, k, sh_layout_slots);
#pragma unroll
                for (int c = 0; c < 4; ++c) {
                    const int64_t cell = static_cast<int64_t>(slot) * 4 + c;
                    C::encode_us(p.joint_packed, cell, us_u[ci], us_s[ci],
                                 new_mm.x, new_mm.z, inv_u_range, inv_s_range);
                    // Always re-encode under new block bounds (frozen/crop-damped too),
                    // matching joint-moment encode. apply_step only gates the Adam update.
                    if (value_q16) {
                        const uint cell_lin = k * 4u + static_cast<uint>(c);
                        if (cell_lin < n_value_cells) {
                            param_u16[lfs::training::sh_value::shAtU16(
                                primitive_idx, cell_lin, n_value_cells)] =
                                VC::encode(pval[ci], new_vmm.x, new_vmm.y);
                        }
                    }
                    ++ci;
                }
            }
        }
    }

    // Quantised two-pass Adam step over the swizzled shN moments of one primitive. moments are
    // uchar4 at the same swizzled float4 slots as the fp32 param (32 warp lanes -> 32 consecutive
    // uchar4 = coalesced 128B loads). Pass 1 derives the per-primitive scales over the active
    // slots; pass 2 updates param and requantises. n_slots_to_update is derived from active SH:
    //     active_sh_bases <= 1 : 0 slots (caller skips)   <= 4 : 3 slots
    //     active_sh_bases <= 9 : 6 slots                   > 9 : 12 slots
    // Joint path (joint_bits==8): block-bounded (u,log_s); all threads must call.
    __device__ inline void apply_shN_grads_packed(
        const FusedAdamSettings& fused_adam,
        const uint primitive_idx,
        const float3 (&g)[15],
        const uint n_slots_to_update,
        const uint sh_layout_slots) {
        const FusedAdamParam& p = fused_adam.shN;
        if (p.joint_bits == 8) {
            apply_shN_grads_packed_joint(fused_adam, primitive_idx, g, n_slots_to_update, sh_layout_slots);
            return;
        }
        if (!p.enabled || n_slots_to_update == 0u || sh_layout_slots == 0u)
            return;
        float row_step_size = p.step_size;
        if (p.frozen_mask != nullptr &&
            primitive_idx < static_cast<uint>(p.frozen_mask_size) &&
            p.frozen_mask[primitive_idx]) {
            if (p.frozen_lr_scale == 0.0f)
                return;
            row_step_size *= p.frozen_lr_scale;
        }
        if (p.crop_damping_mask != nullptr &&
            primitive_idx < static_cast<uint>(p.crop_damping_mask_size) &&
            p.crop_damping_mask[primitive_idx]) {
            if (p.cropbox_lr_scale == 0.0f)
                return;
            row_step_size *= p.cropbox_lr_scale;
        }

        float4* param4 = reinterpret_cast<float4*>(p.param);
        uchar4* m4 = reinterpret_cast<uchar4*>(p.exp_avg_q);
        uchar4* v4 = reinterpret_cast<uchar4*>(p.exp_avg_sq_q);
        const float beta1 = fused_adam.beta1, beta2 = fused_adam.beta2, eps = fused_adam.eps;
        const float old_m_scale = p.exp_avg_scale[primitive_idx];
        const float old_v_scale = p.exp_avg_sq_scale[primitive_idx];

        float max_abs_m = 0.0f;
        float max_v = 0.0f;
#pragma unroll
        for (uint k = 0; k < 12u; ++k) {
            if (k >= n_slots_to_update)
                break;
            const uint slot = shAt(primitive_idx, k, sh_layout_slots);
            const float4 gk = shN_grad_for_slot(g, k);
            const uchar4 qm = m4[slot];
            const uchar4 qv = v4[slot];
            const float gc[4] = {gk.x, gk.y, gk.z, gk.w};
            const std::uint8_t mc[4] = {qm.x, qm.y, qm.z, qm.w};
            const std::uint8_t vc[4] = {qv.x, qv.y, qv.z, qv.w};
#pragma unroll
            for (int c = 0; c < 4; ++c) {
                const float sqrt_old_v = dequant_sqrt_v(vc[c], old_v_scale);
                const float old_v = sqrt_old_v * sqrt_old_v;
                const float m = beta1 * dequant_m(mc[c], old_m_scale) + (1.0f - beta1) * gc[c];
                const float v = beta2 * old_v + (1.0f - beta2) * gc[c] * gc[c];
                max_abs_m = fmaxf(max_abs_m, fabsf(m));
                max_v = fmaxf(max_v, v);
            }
        }

        const float new_m_scale = max_abs_m > 0.0f ? max_abs_m / 127.0f : 0.0f;
        const float new_v_scale = max_v > 0.0f ? sqrtf(max_v) / 255.0f : 0.0f;

#pragma unroll
        for (uint k = 0; k < 12u; ++k) {
            if (k >= n_slots_to_update)
                break;
            const uint slot = shAt(primitive_idx, k, sh_layout_slots);
            const float4 gk = shN_grad_for_slot(g, k);
            const uchar4 qm = m4[slot];
            const uchar4 qv = v4[slot];
            const float gc[4] = {gk.x, gk.y, gk.z, gk.w};
            const std::uint8_t mc[4] = {qm.x, qm.y, qm.z, qm.w};
            const std::uint8_t vc[4] = {qv.x, qv.y, qv.z, qv.w};
            float4 pv = param4[slot];
            float pc[4] = {pv.x, pv.y, pv.z, pv.w};
            std::uint8_t nm[4];
            std::uint8_t nv[4];
#pragma unroll
            for (int c = 0; c < 4; ++c) {
                const float sqrt_old_v = dequant_sqrt_v(vc[c], old_v_scale);
                const float old_v = sqrt_old_v * sqrt_old_v;
                const float m = beta1 * dequant_m(mc[c], old_m_scale) + (1.0f - beta1) * gc[c];
                const float v = beta2 * old_v + (1.0f - beta2) * gc[c] * gc[c];
                const float denom = sqrtf(v) * p.bias_correction2_sqrt_rcp + eps;
                pc[c] -= row_step_size * m / denom;
                nm[c] = quantize_m(m, new_m_scale);
                nv[c] = quantize_sqrt_v(v, new_v_scale);
            }
            param4[slot] = make_float4(pc[0], pc[1], pc[2], pc[3]);
            m4[slot] = make_uchar4(nm[0], nm[1], nm[2], nm[3]);
            v4[slot] = make_uchar4(nv[0], nv[1], nv[2], nv[3]);
        }

        p.exp_avg_scale[primitive_idx] = new_m_scale;
        p.exp_avg_sq_scale[primitive_idx] = new_v_scale;
    }

    // SH backward: fills sh0_grads[3] and shN_grads[15]; does NOT apply Adam
    // (caller runs a unified adam section so joint block-bounds can sync).
    template <int ACTIVE_SH_BASES>
    __device__ inline float3 convert_sh_to_color_backward_grads(
        const float4* sh_coefficients_rest,
        float3* grad_color_helper,
        const float3& position,
        const float3& cam_position,
        const uint primitive_idx,
        const uint sh_layout_slots,
        float* __restrict__ sh0_grads_out,
        float3* __restrict__ shN_grads_out,
        const float2* __restrict__ sh_value_bounds = nullptr,
        const uint sh_value_n_cells = 0u) {
        const float3 grad_color = grad_color_helper[primitive_idx];
        const float3 dL_dsh0 = 0.28209479177387814f * grad_color;
        sh0_grads_out[0] = dL_dsh0.x;
        sh0_grads_out[1] = dL_dsh0.y;
        sh0_grads_out[2] = dL_dsh0.z;
#pragma unroll
        for (int i = 0; i < 15; ++i)
            shN_grads_out[i] = make_float3(0.0f, 0.0f, 0.0f);

        float3 dcolor_dposition = make_float3(0.0f);
        if constexpr (ACTIVE_SH_BASES > 1) {
            const float3 raw_direction = position - cam_position;
            const float x_raw = raw_direction.x;
            const float y_raw = raw_direction.y;
            const float z_raw = raw_direction.z;
            const float3 direction = safe_normalize(raw_direction);
            const float x = direction.x;
            const float y = direction.y;
            const float z = direction.z;

            float3 c[15];
            load_shN_coeffs(sh_coefficients_rest, primitive_idx, ACTIVE_SH_BASES, sh_layout_slots, c,
                            sh_value_bounds, sh_value_n_cells);

            float3* g = shN_grads_out;
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

    // Legacy wrapper: compute SH grads then apply Adam immediately (legacy codec path).
    template <int ACTIVE_SH_BASES>
    __device__ inline float3 convert_sh_to_color_backward(
        const float4* sh_coefficients_rest,
        float3* grad_color_helper,
        const FusedAdamSettings& fused_adam,
        const float3& position,
        const float3& cam_position,
        const uint primitive_idx,
        const uint sh_layout_slots) {
        float sh0_grads[3] = {0.0f, 0.0f, 0.0f};
        float3 g[15];
#pragma unroll
        for (int i = 0; i < 15; ++i)
            g[i] = make_float3(0.0f, 0.0f, 0.0f);
        const float3 dcolor_dposition = convert_sh_to_color_backward_grads<ACTIVE_SH_BASES>(
            sh_coefficients_rest, grad_color_helper, position, cam_position,
            primitive_idx, sh_layout_slots, sh0_grads, g);
        adam_step_row(sh0_grads, fused_adam.sh0, primitive_idx, 3, fused_adam.beta1, fused_adam.beta2, fused_adam.eps);
        if constexpr (ACTIVE_SH_BASES > 1) {
            constexpr uint n_slots = (ACTIVE_SH_BASES > 9) ? 12u : (ACTIVE_SH_BASES > 4) ? 6u
                                                                                         : 3u;
            apply_shN_grads_packed(fused_adam, primitive_idx, g, n_slots, sh_layout_slots);
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

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

#include <cuda_fp16.h>

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
    // q16: when sh_value_bounds != nullptr, `sh_f4` is actually a bitcast of
    // uint16 cell-linear storage (pad-dropped: n_cells = coeffs_rest*3). Decode in registers
    // at the use site — do NOT materialise a float workspace (live-range lesson).
    //
    // IEEE f16 float4-swizzle (GUI exportable): sh_value_bits==16 with null bounds.
    // Same slot topology as fp32; each float4 slot is stored as 4×__half (8 bytes).
    __device__ inline void load_shN_coeffs(
        const float4* __restrict__ sh_f4,
        const uint primitive_idx,
        const uint active_sh_bases,
        const uint sh_layout_slots,
        float3 (&c)[15],
        const float2* __restrict__ sh_value_bounds = nullptr,
        const uint sh_value_n_cells = 0u,
        const uint sh_value_bits = 0u) {
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

        const uint slots_per_primitive = sh_layout_slots;
        if (slots_per_primitive == 0u)
            return;

        // ---- IEEE f16 float4-swizzle (exportable GUI path) ----
        if (sh_value_bits == 16u) {
            const __half* sh_h = reinterpret_cast<const __half*>(sh_f4);
            auto load_half4 = [&](const uint slot) -> float4 {
                const uint base = shAt(primitive_idx, slot, slots_per_primitive) * 4u;
                return make_float4(__half2float(sh_h[base + 0]),
                                   __half2float(sh_h[base + 1]),
                                   __half2float(sh_h[base + 2]),
                                   __half2float(sh_h[base + 3]));
            };
            const float4 a0 = load_half4(0);
            const float4 a1 = load_half4(1);
            const float4 a2 = load_half4(2);
            c[0] = make_float3(a0.x, a0.y, a0.z);
            c[1] = make_float3(a0.w, a1.x, a1.y);
            c[2] = make_float3(a1.z, a1.w, a2.x);
            c[3] = make_float3(a2.y, a2.z, a2.w);
            if (active_sh_bases <= 4)
                return;
            const float4 a3 = load_half4(3);
            const float4 a4 = load_half4(4);
            const float4 a5 = load_half4(5);
            c[4] = make_float3(a3.x, a3.y, a3.z);
            c[5] = make_float3(a3.w, a4.x, a4.y);
            c[6] = make_float3(a4.z, a4.w, a5.x);
            c[7] = make_float3(a5.y, a5.z, a5.w);
            if (active_sh_bases <= 9)
                return;
            const float4 a6 = load_half4(6);
            const float4 a7 = load_half4(7);
            const float4 a8 = load_half4(8);
            const float4 a9 = load_half4(9);
            const float4 a10 = load_half4(10);
            const float4 a11 = load_half4(11);
            c[8] = make_float3(a6.x, a6.y, a6.z);
            c[9] = make_float3(a6.w, a7.x, a7.y);
            c[10] = make_float3(a7.z, a7.w, a8.x);
            c[11] = make_float3(a8.y, a8.z, a8.w);
            c[12] = make_float3(a9.x, a9.y, a9.z);
            c[13] = make_float3(a9.w, a10.x, a10.y);
            c[14] = make_float3(a10.z, a10.w, a11.x);
            return;
        }

        // ---- fp32 float4 path ----
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

    // Streaming SH-rest loaders. One coefficient (or the 1–2 float4 slots it
    // occupies) so the 15-coeff working set never lands in local memory.
    // Decode matches load_shN_coeffs: q16 cell-linear, IEEE f16 float4-swizzle,
    // fp32 float4-swizzle. COEFF is a compile-time index (0..14).
    template <int LANE>
    __device__ __forceinline__ float shN_float4_lane(const float4 a) {
        static_assert(LANE >= 0 && LANE < 4, "float4 lane");
        if constexpr (LANE == 0)
            return a.x;
        else if constexpr (LANE == 1)
            return a.y;
        else if constexpr (LANE == 2)
            return a.z;
        else
            return a.w;
    }

    __device__ __forceinline__ float4 load_shN_float4_slot(
        const float4* __restrict__ sh_f4,
        const uint primitive_idx,
        const uint slot,
        const uint slots_per_primitive,
        const uint sh_value_bits) {
        if (sh_value_bits == 16u) {
            const __half* sh_h = reinterpret_cast<const __half*>(sh_f4);
            const uint base = shAt(primitive_idx, slot, slots_per_primitive) * 4u;
            return make_float4(__half2float(sh_h[base + 0]),
                               __half2float(sh_h[base + 1]),
                               __half2float(sh_h[base + 2]),
                               __half2float(sh_h[base + 3]));
        }
        return sh_f4[shAt(primitive_idx, slot, slots_per_primitive)];
    }

    template <int COEFF>
    __device__ __forceinline__ float3 load_shN_coeff(
        const float4* __restrict__ sh_f4,
        const uint primitive_idx,
        const uint sh_layout_slots,
        const float2* __restrict__ sh_value_bounds,
        const uint sh_value_n_cells,
        const uint sh_value_bits) {
        static_assert(COEFF >= 0 && COEFF < 15, "SH rest coeff 0..14");

        if (sh_value_bounds != nullptr && sh_value_n_cells > 0u) {
            using DC = lfs::training::sh_value::DeviceCodec16;
            const uint16_t* sh_u16 = reinterpret_cast<const uint16_t*>(sh_f4);
            const float2 mm = sh_value_bounds[primitive_idx / 256u];
            const uint base = static_cast<uint>(COEFF) * 3u;
            if (base + 2u >= sh_value_n_cells)
                return make_float3(0.0f, 0.0f, 0.0f);
            return make_float3(
                DC::decode(sh_u16[lfs::training::sh_value::shAtU16(primitive_idx, base + 0, sh_value_n_cells)],
                           mm.x, mm.y),
                DC::decode(sh_u16[lfs::training::sh_value::shAtU16(primitive_idx, base + 1, sh_value_n_cells)],
                           mm.x, mm.y),
                DC::decode(sh_u16[lfs::training::sh_value::shAtU16(primitive_idx, base + 2, sh_value_n_cells)],
                           mm.x, mm.y));
        }

        if (sh_layout_slots == 0u)
            return make_float3(0.0f, 0.0f, 0.0f);

        constexpr int lin0 = COEFF * 3;
        constexpr uint slot0 = static_cast<uint>(lin0) / 4u;
        constexpr uint slot1 = static_cast<uint>(lin0 + 1) / 4u;
        constexpr uint slot2 = static_cast<uint>(lin0 + 2) / 4u;
        const float4 a0 = load_shN_float4_slot(sh_f4, primitive_idx, slot0, sh_layout_slots, sh_value_bits);
        const float x = shN_float4_lane<(lin0 & 3)>(a0);
        float y, z;
        if constexpr (slot2 == slot0) {
            y = shN_float4_lane<((lin0 + 1) & 3)>(a0);
            z = shN_float4_lane<((lin0 + 2) & 3)>(a0);
        } else {
            const float4 a2 = load_shN_float4_slot(sh_f4, primitive_idx, slot2, sh_layout_slots, sh_value_bits);
            if constexpr (slot1 == slot0)
                y = shN_float4_lane<((lin0 + 1) & 3)>(a0);
            else
                y = shN_float4_lane<((lin0 + 1) & 3)>(a2);
            z = shN_float4_lane<((lin0 + 2) & 3)>(a2);
        }
        return make_float3(x, y, z);
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
        const uint sh_value_n_cells = 0u,
        const uint sh_value_bits = 0u) {
        // computation adapted from https://github.com/NVlabs/tiny-cuda-nn/blob/212104156403bd87616c1a4f73a1c5f2c2e172a9/include/tiny-cuda-nn/common_device.h#L340
        float3 result = 0.5f + 0.28209479177387814f * sh_coefficients_0[primitive_idx];
        if (active_sh_bases > 1) {
            const float3 direction = safe_normalize(position - cam_position);
            const float x = direction.x;
            const float y = direction.y;
            const float z = direction.z;
            float3 c[15];
            load_shN_coeffs(sh_coefficients_rest, primitive_idx, active_sh_bases, sh_layout_slots, c,
                            sh_value_bounds, sh_value_n_cells, sh_value_bits);
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

    // Max contiguous (non-shN) attributes per primitive (rotation = 4 is the largest).
    constexpr int MAX_FUSED_ADAM_ATTRIBUTES = 4;

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

        // fused min4 on {u_min,-u_max,s_min,-s_max} → 1 barrier
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
        // hoist block-uniform inv ranges (2 fdiv → FMA per cell).
        const float inv_u_range = 1.0f / fmaxf(new_mm.y - new_mm.x, lfs::training::joint_adam::kEpsDevice);
        const float inv_s_range = 1.0f / fmaxf(new_mm.w - new_mm.z, lfs::training::joint_adam::kEpsDevice);

        if (touch) {
            for (uint i = 0; i < row; ++i) {
                C::encode_us(param.joint_packed, static_cast<int64_t>(base + i),
                             us_u[i], us_s[i], new_mm.x, new_mm.z, inv_u_range, inv_s_range);
            }
        }
    }

    // Joint (u, log_s) Adam step over a contiguous [n_attributes] row of one primitive
    // (means / sh0 / scaling / rotation / opacity). Block-bounded; all threads must call
    // when joint_bits != 0.
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
        // Non-joint state is unsupported (joint is the only codec).
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

    // Per-coefficient SH-rest color gradient: basis_i(dir) * dL/dcolor.
    // Unused rest indices (i >= ACTIVE_SH_BASES-1) are zero, matching the
    // previous g[15] fill (degree-1 leaves g[3] at 0 even though slot 2 packs it).
    template <int I>
    __device__ __forceinline__ float3 shN_coeff_color_grad(
        const float x,
        const float y,
        const float z,
        const float xx,
        const float yy,
        const float zz,
        const float xy,
        const float xz,
        const float yz,
        const float3 grad_color) {
        static_assert(I >= 0 && I < 15, "SH rest coeff 0..14");
        if constexpr (I == 0)
            return (-0.48860251190291987f * y) * grad_color;
        else if constexpr (I == 1)
            return (0.48860251190291987f * z) * grad_color;
        else if constexpr (I == 2)
            return (-0.48860251190291987f * x) * grad_color;
        else if constexpr (I == 3)
            return (1.0925484305920792f * xy) * grad_color;
        else if constexpr (I == 4)
            return (-1.0925484305920792f * yz) * grad_color;
        else if constexpr (I == 5)
            return (0.94617469575755997f * zz - 0.31539156525251999f) * grad_color;
        else if constexpr (I == 6)
            return (-1.0925484305920792f * xz) * grad_color;
        else if constexpr (I == 7)
            return (0.54627421529603959f * xx - 0.54627421529603959f * yy) * grad_color;
        else if constexpr (I == 8)
            return (0.59004358992664352f * y * (-3.0f * xx + yy)) * grad_color;
        else if constexpr (I == 9)
            return (2.8906114426405538f * xy * z) * grad_color;
        else if constexpr (I == 10)
            return (0.45704579946446572f * y * (1.0f - 5.0f * zz)) * grad_color;
        else if constexpr (I == 11)
            return (0.3731763325901154f * z * (5.0f * zz - 3.0f)) * grad_color;
        else if constexpr (I == 12)
            return (0.45704579946446572f * x * (1.0f - 5.0f * zz)) * grad_color;
        else if constexpr (I == 13)
            return (1.4453057213202769f * z * (xx - yy)) * grad_color;
        else
            return (0.59004358992664352f * x * (-xx + 3.0f * yy)) * grad_color;
    }

    template <int ACTIVE_SH_BASES, int I>
    __device__ __forceinline__ float3 shN_active_coeff_grad(
        const float x,
        const float y,
        const float z,
        const float xx,
        const float yy,
        const float zz,
        const float xy,
        const float xz,
        const float yz,
        const float3 grad_color) {
        constexpr int N_REST = ACTIVE_SH_BASES > 1 ? ACTIVE_SH_BASES - 1 : 0;
        if constexpr (I >= N_REST)
            return make_float3(0.0f, 0.0f, 0.0f);
        else
            return shN_coeff_color_grad<I>(x, y, z, xx, yy, zz, xy, xz, yz, grad_color);
    }

    // Pack one float4-slot of SH-rest grads from the basis — no g[15] array.
    // `k` is the unrolled slot index; the switch constant-folds.
    template <int ACTIVE_SH_BASES>
    __device__ __forceinline__ float4 shN_slot_grad_from_basis(
        const uint k,
        const float x,
        const float y,
        const float z,
        const float xx,
        const float yy,
        const float zz,
        const float xy,
        const float xz,
        const float yz,
        const float3 grad_color) {
        switch (k) {
        case 0: {
            const float3 a = shN_active_coeff_grad<ACTIVE_SH_BASES, 0>(x, y, z, xx, yy, zz, xy, xz, yz, grad_color);
            const float3 b = shN_active_coeff_grad<ACTIVE_SH_BASES, 1>(x, y, z, xx, yy, zz, xy, xz, yz, grad_color);
            return make_float4(a.x, a.y, a.z, b.x);
        }
        case 1: {
            const float3 a = shN_active_coeff_grad<ACTIVE_SH_BASES, 1>(x, y, z, xx, yy, zz, xy, xz, yz, grad_color);
            const float3 b = shN_active_coeff_grad<ACTIVE_SH_BASES, 2>(x, y, z, xx, yy, zz, xy, xz, yz, grad_color);
            return make_float4(a.y, a.z, b.x, b.y);
        }
        case 2: {
            const float3 a = shN_active_coeff_grad<ACTIVE_SH_BASES, 2>(x, y, z, xx, yy, zz, xy, xz, yz, grad_color);
            const float3 b = shN_active_coeff_grad<ACTIVE_SH_BASES, 3>(x, y, z, xx, yy, zz, xy, xz, yz, grad_color);
            return make_float4(a.z, b.x, b.y, b.z);
        }
        case 3: {
            const float3 a = shN_active_coeff_grad<ACTIVE_SH_BASES, 4>(x, y, z, xx, yy, zz, xy, xz, yz, grad_color);
            const float3 b = shN_active_coeff_grad<ACTIVE_SH_BASES, 5>(x, y, z, xx, yy, zz, xy, xz, yz, grad_color);
            return make_float4(a.x, a.y, a.z, b.x);
        }
        case 4: {
            const float3 a = shN_active_coeff_grad<ACTIVE_SH_BASES, 5>(x, y, z, xx, yy, zz, xy, xz, yz, grad_color);
            const float3 b = shN_active_coeff_grad<ACTIVE_SH_BASES, 6>(x, y, z, xx, yy, zz, xy, xz, yz, grad_color);
            return make_float4(a.y, a.z, b.x, b.y);
        }
        case 5: {
            const float3 a = shN_active_coeff_grad<ACTIVE_SH_BASES, 6>(x, y, z, xx, yy, zz, xy, xz, yz, grad_color);
            const float3 b = shN_active_coeff_grad<ACTIVE_SH_BASES, 7>(x, y, z, xx, yy, zz, xy, xz, yz, grad_color);
            return make_float4(a.z, b.x, b.y, b.z);
        }
        case 6: {
            const float3 a = shN_active_coeff_grad<ACTIVE_SH_BASES, 8>(x, y, z, xx, yy, zz, xy, xz, yz, grad_color);
            const float3 b = shN_active_coeff_grad<ACTIVE_SH_BASES, 9>(x, y, z, xx, yy, zz, xy, xz, yz, grad_color);
            return make_float4(a.x, a.y, a.z, b.x);
        }
        case 7: {
            const float3 a = shN_active_coeff_grad<ACTIVE_SH_BASES, 9>(x, y, z, xx, yy, zz, xy, xz, yz, grad_color);
            const float3 b = shN_active_coeff_grad<ACTIVE_SH_BASES, 10>(x, y, z, xx, yy, zz, xy, xz, yz, grad_color);
            return make_float4(a.y, a.z, b.x, b.y);
        }
        case 8: {
            const float3 a = shN_active_coeff_grad<ACTIVE_SH_BASES, 10>(x, y, z, xx, yy, zz, xy, xz, yz, grad_color);
            const float3 b = shN_active_coeff_grad<ACTIVE_SH_BASES, 11>(x, y, z, xx, yy, zz, xy, xz, yz, grad_color);
            return make_float4(a.z, b.x, b.y, b.z);
        }
        case 9: {
            const float3 a = shN_active_coeff_grad<ACTIVE_SH_BASES, 12>(x, y, z, xx, yy, zz, xy, xz, yz, grad_color);
            const float3 b = shN_active_coeff_grad<ACTIVE_SH_BASES, 13>(x, y, z, xx, yy, zz, xy, xz, yz, grad_color);
            return make_float4(a.x, a.y, a.z, b.x);
        }
        case 10: {
            const float3 a = shN_active_coeff_grad<ACTIVE_SH_BASES, 13>(x, y, z, xx, yy, zz, xy, xz, yz, grad_color);
            const float3 b = shN_active_coeff_grad<ACTIVE_SH_BASES, 14>(x, y, z, xx, yy, zz, xy, xz, yz, grad_color);
            return make_float4(a.y, a.z, b.x, b.y);
        }
        case 11: {
            const float3 a = shN_active_coeff_grad<ACTIVE_SH_BASES, 14>(x, y, z, xx, yy, zz, xy, xz, yz, grad_color);
            return make_float4(a.z, 0.0f, 0.0f, 0.0f);
        }
        default:
            return make_float4(0.0f, 0.0f, 0.0f, 0.0f);
        }
    }

    template <typename C>
    __device__ __forceinline__ float2 shN_adam_moment_us(
        const float grad,
        const int64_t cell,
        const uint8_t* __restrict__ joint_packed,
        const float4 old_mm,
        const bool apply_step,
        const bool update_param,
        const float beta1,
        const float beta2,
        const float row_step_size,
        const float eps,
        const float bias_correction2_sqrt_rcp,
        float& pc) {
        const float2 mv = C::decode_g1g2(joint_packed, cell, old_mm);
        float m = mv.x;
        float v = mv.y;
        if (apply_step) {
            m = beta1 * mv.x + (1.0f - beta1) * grad;
            v = beta2 * mv.y + (1.0f - beta2) * grad * grad;
            if (update_param) {
                const float denom = sqrtf(v) * bias_correction2_sqrt_rcp + eps;
                pc -= row_step_size * m / denom;
            }
        }
        return C::g1g2_to_us(m, v);
    }

    __device__ __forceinline__ float4 load_shN_param_slot(
        const bool value_q16,
        const bool value_f16,
        const uint16_t* __restrict__ param_u16,
        const __half* __restrict__ param_h,
        const float4* __restrict__ param4,
        const uint primitive_idx,
        const uint k,
        const uint slot,
        const uint n_value_cells,
        const float2 old_vmm) {
        using VC = lfs::training::sh_value::DeviceCodec16;
        if (value_q16) {
            float px = 0.0f, py = 0.0f, pz = 0.0f, pw = 0.0f;
            if (k * 4u + 0u < n_value_cells)
                px = VC::decode(param_u16[lfs::training::sh_value::shAtU16(primitive_idx, k * 4u + 0u, n_value_cells)], old_vmm.x, old_vmm.y);
            if (k * 4u + 1u < n_value_cells)
                py = VC::decode(param_u16[lfs::training::sh_value::shAtU16(primitive_idx, k * 4u + 1u, n_value_cells)], old_vmm.x, old_vmm.y);
            if (k * 4u + 2u < n_value_cells)
                pz = VC::decode(param_u16[lfs::training::sh_value::shAtU16(primitive_idx, k * 4u + 2u, n_value_cells)], old_vmm.x, old_vmm.y);
            if (k * 4u + 3u < n_value_cells)
                pw = VC::decode(param_u16[lfs::training::sh_value::shAtU16(primitive_idx, k * 4u + 3u, n_value_cells)], old_vmm.x, old_vmm.y);
            return make_float4(px, py, pz, pw);
        }
        if (value_f16) {
            const uint base = slot * 4u;
            return make_float4(__half2float(param_h[base + 0]),
                               __half2float(param_h[base + 1]),
                               __half2float(param_h[base + 2]),
                               __half2float(param_h[base + 3]));
        }
        return param4[slot];
    }

    // Joint 8-bit shN Adam (swizzled float cells × 2 B). ALL threads must call when joint.
    // Walks ALL layout slots (not only active SH) so inactive bands re-encode under new
    // bounds and stay true-zero when their codes represent (u,log_s)=(0,0).
    //
    // when p.sh_value_bits==16, param is pad-dropped uint16 codes; decode in
    // registers, Adam-update, then single re-encode after value bounds reduce. Moments still
    // use the float4-slot cell indexing (48 cells). Value re-encode is the single writer.
    //
    // Streaming: one float4 slot at a time. Pass 1 applies Adam and reduces bounds
    // without storing the 48-cell (u, log_s, pval) workspace. Pass 2 recomputes the
    // same per-cell Adam from still-original packed moments (and q16 values) and
    // encodes. Slot / cell order is unchanged.
    template <int ACTIVE_SH_BASES>
    __device__ inline void apply_shN_grads_packed_joint(
        const FusedAdamSettings& fused_adam,
        const uint primitive_idx,
        const uint sh_layout_slots,
        const float3 mean3d,
        const float3 cam_position,
        const float3 grad_color,
        const bool compute_sh_grads) {
        using C = lfs::training::joint_adam::DeviceCodec<8>;
        using VC = lfs::training::sh_value::DeviceCodec16;
        constexpr float kInf = 1e30f;
        const FusedAdamParam& p = fused_adam.shN;
        const bool value_q16 = p.sh_value_bits == 16 && p.sh_value_bounds != nullptr &&
                               p.sh_value_n_cells > 0;
        // IEEE f16 float4-swizzle (exportable GUI): bits==16, no bounds.
        const bool value_f16 = p.sh_value_bits == 16 && !value_q16;
        const uint n_value_cells = value_q16 ? static_cast<uint>(p.sh_value_n_cells) : 0u;

        float row_step_size = p.step_size;
        bool apply_step = true;
        bool touch = p.enabled && sh_layout_slots > 0u &&
                     p.joint_packed != nullptr && p.joint_bounds != nullptr;
        // grid overhang threads (ceil(N/256)*256 - N) must stay in the reduce
        // with ±inf identities — never decode/encode into capacity slack / OOB.
        if (touch && p.n_primitives > 0 &&
            primitive_idx >= static_cast<uint>(p.n_primitives)) {
            touch = false;
        }
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
        float4* param4 =
            (!value_q16 && !value_f16 && touch) ? reinterpret_cast<float4*>(p.param) : nullptr;
        uint16_t* param_u16 = (value_q16 && touch) ? reinterpret_cast<uint16_t*>(p.param) : nullptr;
        __half* param_h = (value_f16 && touch) ? reinterpret_cast<__half*>(p.param) : nullptr;

        float local_u_min = kInf, local_u_max = -kInf;
        float local_s_min = kInf, local_s_max = -kInf;
        float local_v_min = kInf, local_v_max = -kInf;

        constexpr uint N_SLOTS = (ACTIVE_SH_BASES > 9) ? 12u : (ACTIVE_SH_BASES > 4) ? 6u
                                                                                     : 3u;
        float bx = 0.0f, by = 0.0f, bz = 0.0f;
        float bxx = 0.0f, byy = 0.0f, bzz = 0.0f, bxy = 0.0f, bxz = 0.0f, byz = 0.0f;
        if (compute_sh_grads) {
            const float3 direction = safe_normalize(mean3d - cam_position);
            bx = direction.x;
            by = direction.y;
            bz = direction.z;
            bxx = bx * bx;
            byy = by * by;
            bzz = bz * bz;
            bxy = bx * by;
            bxz = bx * bz;
            byz = by * bz;
        }

        if (touch) {
#pragma unroll
            for (uint k = 0; k < 12u; ++k) {
                if (k >= sh_layout_slots)
                    break;
                const uint slot = shAt(primitive_idx, k, sh_layout_slots);
                const bool active_slot = k < N_SLOTS;
                const float4 gk = (compute_sh_grads && active_slot)
                                      ? shN_slot_grad_from_basis<ACTIVE_SH_BASES>(
                                            k, bx, by, bz, bxx, byy, bzz, bxy, bxz, byz, grad_color)
                                      : make_float4(0.0f, 0.0f, 0.0f, 0.0f);
                float4 pc = load_shN_param_slot(value_q16, value_f16, param_u16, param_h, param4,
                                                primitive_idx, k, slot, n_value_cells, old_vmm);
#pragma unroll
                for (int c = 0; c < 4; ++c) {
                    float pci = (c == 0) ? pc.x : (c == 1) ? pc.y
                                              : (c == 2)   ? pc.z
                                                           : pc.w;
                    const float gci = (c == 0) ? gk.x : (c == 1) ? gk.y
                                                    : (c == 2)   ? gk.z
                                                                 : gk.w;
                    const int64_t cell = static_cast<int64_t>(slot) * 4 + c;
                    const float2 prim = shN_adam_moment_us<C>(
                        gci, cell, p.joint_packed, old_mm, apply_step, active_slot,
                        beta1, beta2, row_step_size, eps, p.bias_correction2_sqrt_rcp, pci);
                    if (c == 0)
                        pc.x = pci;
                    else if (c == 1)
                        pc.y = pci;
                    else if (c == 2)
                        pc.z = pci;
                    else
                        pc.w = pci;
                    if (value_q16) {
                        const uint cell_lin = k * 4u + static_cast<uint>(c);
                        if (cell_lin < n_value_cells) {
                            local_v_min = fminf(local_v_min, pci);
                            local_v_max = fmaxf(local_v_max, pci);
                        }
                    }
                    local_u_min = fminf(local_u_min, prim.x);
                    local_u_max = fmaxf(local_u_max, prim.x);
                    local_s_min = fminf(local_s_min, prim.y);
                    local_s_max = fmaxf(local_s_max, prim.y);
                }
                if (apply_step && active_slot) {
                    if (value_f16) {
                        const uint base = slot * 4u;
                        param_h[base + 0] = __float2half(pc.x);
                        param_h[base + 1] = __float2half(pc.y);
                        param_h[base + 2] = __float2half(pc.z);
                        param_h[base + 3] = __float2half(pc.w);
                    } else if (!value_q16) {
                        param4[slot] = pc;
                    }
                }
            }
        }

        // fused min4 bounds reduce for Adam moment (u,log_s).
        const float4 red = lfs::core::warp_ops::block_reduce_min4(
            make_float4(local_u_min, -local_u_max, local_s_min, -local_s_max));
        const float u_min = red.x;
        const float u_max = -red.y;
        const float s_min = red.z;
        const float s_max = -red.w;
        // value bounds (separate from moment bounds).
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
        // hoist block-uniform inv ranges (2 fdiv → FMA per cell).
        const float inv_u_range = 1.0f / fmaxf(new_mm.y - new_mm.x, lfs::training::joint_adam::kEpsDevice);
        const float inv_s_range = 1.0f / fmaxf(new_mm.w - new_mm.z, lfs::training::joint_adam::kEpsDevice);

        if (touch) {
#pragma unroll
            for (uint k = 0; k < 12u; ++k) {
                if (k >= sh_layout_slots)
                    break;
                const uint slot = shAt(primitive_idx, k, sh_layout_slots);
                const bool active_slot = k < N_SLOTS;
                const float4 gk = (compute_sh_grads && active_slot)
                                      ? shN_slot_grad_from_basis<ACTIVE_SH_BASES>(
                                            k, bx, by, bz, bxx, byy, bzz, bxy, bxz, byz, grad_color)
                                      : make_float4(0.0f, 0.0f, 0.0f, 0.0f);
                float4 pc = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
                if (value_q16) {
                    pc = load_shN_param_slot(true, false, param_u16, param_h, param4,
                                             primitive_idx, k, slot, n_value_cells, old_vmm);
                }
#pragma unroll
                for (int c = 0; c < 4; ++c) {
                    float pci = (c == 0) ? pc.x : (c == 1) ? pc.y
                                              : (c == 2)   ? pc.z
                                                           : pc.w;
                    const float gci = (c == 0) ? gk.x : (c == 1) ? gk.y
                                                    : (c == 2)   ? gk.z
                                                                 : gk.w;
                    const int64_t cell = static_cast<int64_t>(slot) * 4 + c;
                    const float2 prim = shN_adam_moment_us<C>(
                        gci, cell, p.joint_packed, old_mm, apply_step, active_slot,
                        beta1, beta2, row_step_size, eps, p.bias_correction2_sqrt_rcp, pci);
                    C::encode_us(p.joint_packed, cell, prim.x, prim.y,
                                 new_mm.x, new_mm.z, inv_u_range, inv_s_range);
                    // Always re-encode under new block bounds (frozen/crop-damped too),
                    // matching joint-moment encode. apply_step only gates the Adam update.
                    if (value_q16) {
                        const uint cell_lin = k * 4u + static_cast<uint>(c);
                        if (cell_lin < n_value_cells) {
                            param_u16[lfs::training::sh_value::shAtU16(
                                primitive_idx, cell_lin, n_value_cells)] =
                                VC::encode(pci, new_vmm.x, new_vmm.y);
                        }
                    }
                }
            }
        }
    }

    // Joint (u, log_s) Adam step over swizzled shN moments of one primitive.
    // Slot count is derived from ACTIVE_SH_BASES. All threads must call
    // when joint_bits==8 (block bounds).
    template <int ACTIVE_SH_BASES>
    __device__ inline void apply_shN_grads_packed(
        const FusedAdamSettings& fused_adam,
        const uint primitive_idx,
        const uint sh_layout_slots,
        const float3 mean3d,
        const float3 cam_position,
        const float3 grad_color,
        const bool compute_sh_grads) {
        const FusedAdamParam& p = fused_adam.shN;
        if (p.joint_bits == 8) {
            apply_shN_grads_packed_joint<ACTIVE_SH_BASES>(
                fused_adam, primitive_idx, sh_layout_slots,
                mean3d, cam_position, grad_color, compute_sh_grads);
        }
        // Non-joint state is unsupported (joint is the only codec).
    }

    // SH backward: fills sh0_grads[3] and returns dL/dmean from view-dir.
    // Per-coefficient shN color grads are independent (basis_i * dL/dcolor) and
    // are recomputed in the fused-Adam stream — they are not materialized here.
    // dL/ddir keeps the original grouped accumulation (degree-1 init from
    // c2,c0,c1; degree-2 adds c3,c6,c7 / c3,c4,c7 / c4,c5,c6; degree-3 adds
    // c8,c9,c12,c13,c14 / c8,c9,c10,c13,c14 / c9,c10,c11,c12,c13).
    template <int ACTIVE_SH_BASES>
    __device__ inline float3 convert_sh_to_color_backward_grads(
        const float4* sh_coefficients_rest,
        float3* grad_color_helper,
        const float3& position,
        const float3& cam_position,
        const uint primitive_idx,
        const uint sh_layout_slots,
        float* __restrict__ sh0_grads_out,
        const float2* __restrict__ sh_value_bounds = nullptr,
        const uint sh_value_n_cells = 0u,
        const uint sh_value_bits = 0u) {
        const float3 grad_color = grad_color_helper[primitive_idx];
        const float3 dL_dsh0 = 0.28209479177387814f * grad_color;
        sh0_grads_out[0] = dL_dsh0.x;
        sh0_grads_out[1] = dL_dsh0.y;
        sh0_grads_out[2] = dL_dsh0.z;

        float3 dcolor_dposition = make_float3(0.0f);
        if constexpr (ACTIVE_SH_BASES > 1) {
            const float3 raw_direction = position - cam_position;
            const float x_raw = raw_direction.x;
            const float y_raw = raw_direction.y;
            const float z_raw = raw_direction.z;
            const float3 c0 = load_shN_coeff<0>(sh_coefficients_rest, primitive_idx, sh_layout_slots,
                                                sh_value_bounds, sh_value_n_cells, sh_value_bits);
            const float3 c1 = load_shN_coeff<1>(sh_coefficients_rest, primitive_idx, sh_layout_slots,
                                                sh_value_bounds, sh_value_n_cells, sh_value_bits);
            const float3 c2 = load_shN_coeff<2>(sh_coefficients_rest, primitive_idx, sh_layout_slots,
                                                sh_value_bounds, sh_value_n_cells, sh_value_bits);
            float3 grad_direction_x = -0.48860251190291987f * c2;
            float3 grad_direction_y = -0.48860251190291987f * c0;
            float3 grad_direction_z = 0.48860251190291987f * c1;
            if constexpr (ACTIVE_SH_BASES > 4) {
                const float3 direction = safe_normalize(raw_direction);
                const float x = direction.x;
                const float y = direction.y;
                const float z = direction.z;
                const float xx = x * x, yy = y * y, zz = z * z;
                const float xy = x * y, xz = x * z, yz = y * z;
                const float3 c3 = load_shN_coeff<3>(sh_coefficients_rest, primitive_idx, sh_layout_slots,
                                                    sh_value_bounds, sh_value_n_cells, sh_value_bits);
                const float3 c4 = load_shN_coeff<4>(sh_coefficients_rest, primitive_idx, sh_layout_slots,
                                                    sh_value_bounds, sh_value_n_cells, sh_value_bits);
                const float3 c5 = load_shN_coeff<5>(sh_coefficients_rest, primitive_idx, sh_layout_slots,
                                                    sh_value_bounds, sh_value_n_cells, sh_value_bits);
                const float3 c6 = load_shN_coeff<6>(sh_coefficients_rest, primitive_idx, sh_layout_slots,
                                                    sh_value_bounds, sh_value_n_cells, sh_value_bits);
                const float3 c7 = load_shN_coeff<7>(sh_coefficients_rest, primitive_idx, sh_layout_slots,
                                                    sh_value_bounds, sh_value_n_cells, sh_value_bits);
                grad_direction_x = grad_direction_x + (1.0925484305920792f * y) * c3 + (-1.0925484305920792f * z) * c6 + (1.0925484305920792f * x) * c7;
                grad_direction_y = grad_direction_y + (1.0925484305920792f * x) * c3 + (-1.0925484305920792f * z) * c4 + (-1.0925484305920792f * y) * c7;
                grad_direction_z = grad_direction_z + (-1.0925484305920792f * y) * c4 + (1.8923493915151202f * z) * c5 + (-1.0925484305920792f * x) * c6;
                if constexpr (ACTIVE_SH_BASES > 9) {
                    const float3 c8 = load_shN_coeff<8>(sh_coefficients_rest, primitive_idx, sh_layout_slots,
                                                        sh_value_bounds, sh_value_n_cells, sh_value_bits);
                    const float3 c9 = load_shN_coeff<9>(sh_coefficients_rest, primitive_idx, sh_layout_slots,
                                                        sh_value_bounds, sh_value_n_cells, sh_value_bits);
                    const float3 c10 = load_shN_coeff<10>(sh_coefficients_rest, primitive_idx, sh_layout_slots,
                                                          sh_value_bounds, sh_value_n_cells, sh_value_bits);
                    const float3 c11 = load_shN_coeff<11>(sh_coefficients_rest, primitive_idx, sh_layout_slots,
                                                          sh_value_bounds, sh_value_n_cells, sh_value_bits);
                    const float3 c12 = load_shN_coeff<12>(sh_coefficients_rest, primitive_idx, sh_layout_slots,
                                                          sh_value_bounds, sh_value_n_cells, sh_value_bits);
                    const float3 c13 = load_shN_coeff<13>(sh_coefficients_rest, primitive_idx, sh_layout_slots,
                                                          sh_value_bounds, sh_value_n_cells, sh_value_bits);
                    const float3 c14 = load_shN_coeff<14>(sh_coefficients_rest, primitive_idx, sh_layout_slots,
                                                          sh_value_bounds, sh_value_n_cells, sh_value_bits);
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

    // Ellipse–AABB overlap (exact). Port of alphablend_shader.slang / utils.slang
    // ellipse_box_overlap_test. Ellipse centered at origin, defined by inv_cov with r=1:
    //   form = a x^2 + 2 b x y + c y^2  (inv_cov.y is the half cross-term, same as conic.y).
    // Box is axis-aligned [x0,x1] × [y0,y1] in the same frame.
    // Used for warp sub-tile culling: tighter than pixel AABB, still
    // never drops a sample whose power is below the contribution threshold when inv_cov
    // has been scaled by 1/(2 * power_threshold).
    __device__ __forceinline__ bool ellipse_box_overlap_test(
        const float3& inv_cov,
        const float x0,
        const float x1,
        const float y0,
        const float y1) {
        const float a = inv_cov.x;
        const float b0 = inv_cov.y;
        const float c = inv_cov.z;

        // Parabola vertices on the four edges (branchless clamp).
        const float wx = -b0 / c;
        const float wy = -b0 / a;
        const float u0 = fminf(fmaxf(x0 * wx, y0), y1);
        const float u1 = fminf(fmaxf(x1 * wx, y0), y1);
        const float v0 = fminf(fmaxf(y0 * wy, x0), x1);
        const float v1 = fminf(fmaxf(y1 * wy, x0), x1);
        const float b = 2.0f * b0;
        const float mx = fminf(
            a * x0 * x0 + b * x0 * u0 + c * u0 * u0,
            a * x1 * x1 + b * x1 * u1 + c * u1 * u1);
        const float my = fminf(
            a * v0 * v0 + b * v0 * y0 + c * y0 * y0,
            a * v1 * v1 + b * v1 * y1 + c * y1 * y1);

        // Center of ellipse inside box (negative if yes). mc == 0 means the
        // mean sits on the box boundary (a pixel center on the sub-tile edge);
        // fmin(mx, my) == 1 means the unit ellipse exactly touches an edge.
        // Blend includes the contribution boundary (alpha >= min_alpha), so
        // this test is closed: <= 0, not < 0.
        const float mc = fmaxf(fmaxf(x0, -x1), fmaxf(y0, -y1));
        return fminf(mc, fminf(mx, my) - 1.0f) <= 0.0f;
    }

    // True if the contribution ellipse of (mean2d, conic, opacity) overlaps the
    // axis-aligned sub-tile whose absolute integer pixel corners are
    // [sub_x0, sub_x0+sub_w) × [sub_y0, sub_y0+sub_h). Evaluation points are
    // pixel centers (+0.5); the continuous box covers those centers.
    __device__ __forceinline__ bool splat_overlaps_subtile_ellipse(
        const float2 mean2d,
        const float3 conic,
        const float opacity,
        const float sub_x0,
        const float sub_y0,
        const float sub_w,
        const float sub_h) {
        if (!(opacity >= config::min_alpha_threshold))
            return false;
        const float power_threshold = logf(opacity * config::min_alpha_threshold_rcp);
        // Continuous box of pixel centers in mean-relative coordinates.
        const float x0 = (sub_x0 + 0.5f) - mean2d.x;
        const float x1 = (sub_x0 + sub_w - 0.5f) - mean2d.x;
        const float y0 = (sub_y0 + 0.5f) - mean2d.y;
        const float y1 = (sub_y0 + sub_h - 0.5f) - mean2d.y;
        if (!(power_threshold > 0.0f)) {
            // Degenerate point ellipse: blend still shades a pixel whose center
            // coincides with the mean when opacity == min_alpha (power == 0).
            if (!(power_threshold == 0.0f))
                return false;
            const float mc = fmaxf(fmaxf(x0, -x1), fmaxf(y0, -y1));
            return mc <= 0.0f;
        }
        // Normalize so contribution boundary is the unit ellipse of inv_cov.
        const float inv_scale = 1.0f / (2.0f * power_threshold);
        const float3 inv_cov = make_float3(
            conic.x * inv_scale,
            conic.y * inv_scale,
            conic.z * inv_scale);
        return ellipse_box_overlap_test(inv_cov, x0, x1, y0, y1);
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

    // Exclusive end of a *closed* interval in shifted pixel-center space
    // (integers are pixel centers). `ceil(coord/tile)` drops the next tile
    // when coord lands bit-exactly on a tile boundary — that pixel center
    // belongs to the next tile and blend will shade it. floor+1 keeps it.
    __device__ inline uint ceil_tile_clamped(
        const float coord,
        const uint min_tile,
        const uint max_tile,
        const uint tile_size) {
        const int tile = __float2int_rd(coord / static_cast<float>(tile_size)) + 1;
        return static_cast<uint>(min(max(tile, static_cast<int>(min_tile)), static_cast<int>(max_tile)));
    }

    // AABB walk domain for exact ellipse binning, half-open [min, max).
    // `extent_*` is preprocess max(E - 0.5, 0), so mean±extent is a
    // pixel-center-adjusted edge. A closed contribution interval must keep
    // the extra tile only when that edge lands bit-exactly on a tile
    // boundary: inclusive start ceil(t)-1, exclusive end floor(t)+1.
    // Those match floor/ceil for every non-integer.
    __device__ inline uint4 compute_screen_tile_bounds(
        const float2 mean2d,
        const float extent_x,
        const float extent_y,
        const uint grid_width,
        const uint grid_height) {
        const float tw = static_cast<float>(config::tile_width);
        const float th = static_cast<float>(config::tile_height);
        const int x_min = max(0, __float2int_ru((mean2d.x - extent_x) / tw) - 1);
        const int x_max = max(0, __float2int_rd((mean2d.x + extent_x) / tw) + 1);
        const int y_min = max(0, __float2int_ru((mean2d.y - extent_y) / th) - 1);
        const int y_max = max(0, __float2int_rd((mean2d.y + extent_y) / th) + 1);
        return make_uint4(
            min(grid_width, static_cast<uint>(x_min)),
            min(grid_width, static_cast<uint>(x_max)),
            min(grid_height, static_cast<uint>(y_min)),
            min(grid_height, static_cast<uint>(y_max)));
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
        // radius_sq == 0 is a point ellipse: blend still shades a pixel whose
        // center coincides with the mean. Negative is not a valid ellipse.
        if (radius_sq < 0.0f)
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
                n_touched_tiles += max_x >= min_x ? max_x - min_x : 0;
            }
        } else {
            const float3 conic_transposed = make_float3(conic.z, conic.y, conic.x);
            for (uint tile_x = screen_bounds.x; tile_x < screen_bounds.y; tile_x++) {
                const float x0 = static_cast<float>(tile_x * config::tile_width) - mean2d_shifted.x;
                const float x1 = x0 + static_cast<float>(config::tile_width);
                const float2 bound = ellipse_range_bound(conic_transposed, radius_sq, x0, x1);
                const uint min_y = floor_tile_clamped(bound.x + mean2d_shifted.y, screen_bounds.z, screen_bounds.w, config::tile_height);
                const uint max_y = ceil_tile_clamped(bound.y + mean2d_shifted.y, screen_bounds.z, screen_bounds.w, config::tile_height);
                n_touched_tiles += max_y >= min_y ? max_y - min_y : 0;
            }
        }

        return n_touched_tiles;
    }

} // namespace fast_lfs::rasterization::kernels

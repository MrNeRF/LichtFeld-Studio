/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include "core/cuda/sigmoid.cuh"
#include "core/sh_value_codec.cuh"
#include "fused_adam_types.h"
#include "helper_math.h"
#include "lfs/core/warp_reduce.cuh"
#include "lfs/training/joint_adam_codec.cuh"
#include "lfs/training/screen_share.cuh"
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
            using DC = lfs::core::sh_value::DeviceCodec16;
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
                    DC::decode(sh_u16[lfs::core::sh_value::shAtU16(primitive_idx, base + 0, n_cells)],
                               mm.x, mm.y),
                    DC::decode(sh_u16[lfs::core::sh_value::shAtU16(primitive_idx, base + 1, n_cells)],
                               mm.x, mm.y),
                    DC::decode(sh_u16[lfs::core::sh_value::shAtU16(primitive_idx, base + 2, n_cells)],
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
            using DC = lfs::core::sh_value::DeviceCodec16;
            const uint16_t* sh_u16 = reinterpret_cast<const uint16_t*>(sh_f4);
            const float2 mm = sh_value_bounds[primitive_idx / 256u];
            const uint base = static_cast<uint>(COEFF) * 3u;
            if (base + 2u >= sh_value_n_cells)
                return make_float3(0.0f, 0.0f, 0.0f);
            return make_float3(
                DC::decode(sh_u16[lfs::core::sh_value::shAtU16(primitive_idx, base + 0, sh_value_n_cells)],
                           mm.x, mm.y),
                DC::decode(sh_u16[lfs::core::sh_value::shAtU16(primitive_idx, base + 1, sh_value_n_cells)],
                           mm.x, mm.y),
                DC::decode(sh_u16[lfs::core::sh_value::shAtU16(primitive_idx, base + 2, sh_value_n_cells)],
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
            auto coefficient = [&]<int Index>() {
                return load_shN_coeff<Index>(sh_coefficients_rest, primitive_idx, sh_layout_slots,
                                             sh_value_bounds, sh_value_n_cells, sh_value_bits);
            };
            // Consume each decoded coefficient before loading the next one.
            // Keep the addition order and the existing storage-specific decoders.
            result = result + (-0.48860251190291987f * y) * coefficient.template operator()<0>();
            result = result + (0.48860251190291987f * z) * coefficient.template operator()<1>();
            result = result + (-0.48860251190291987f * x) * coefficient.template operator()<2>();
            if (active_sh_bases > 4) {
                const float xx = x * x, yy = y * y, zz = z * z;
                const float xy = x * y, xz = x * z, yz = y * z;
                result = result + (1.0925484305920792f * xy) * coefficient.template operator()<3>();
                result = result + (-1.0925484305920792f * yz) * coefficient.template operator()<4>();
                result = result + (0.94617469575755997f * zz - 0.31539156525251999f) * coefficient.template operator()<5>();
                result = result + (-1.0925484305920792f * xz) * coefficient.template operator()<6>();
                result = result + (0.54627421529603959f * xx - 0.54627421529603959f * yy) * coefficient.template operator()<7>();
                if (active_sh_bases > 9) {
                    result = result + (0.59004358992664352f * y * (-3.0f * xx + yy)) * coefficient.template operator()<8>();
                    result = result + (2.8906114426405538f * xy * z) * coefficient.template operator()<9>();
                    result = result + (0.45704579946446572f * y * (1.0f - 5.0f * zz)) * coefficient.template operator()<10>();
                    result = result + (0.3731763325901154f * z * (5.0f * zz - 3.0f)) * coefficient.template operator()<11>();
                    result = result + (0.45704579946446572f * x * (1.0f - 5.0f * zz)) * coefficient.template operator()<12>();
                    result = result + (1.4453057213202769f * z * (xx - yy)) * coefficient.template operator()<13>();
                    result = result + (0.59004358992664352f * x * (-xx + 3.0f * yy)) * coefficient.template operator()<14>();
                }
            }
        }
        return result;
    }

    // Specialized forward SH evaluation for single-pass preprocess: hoists q16 block
    // bounds and base pointer once per visible splat instead of per-coefficient.
    template <int ACTIVE_SH_BASES, bool SH_Q16>
    __device__ __forceinline__ float3 convert_sh_to_color_fast(
        const float3* __restrict__ sh_coefficients_0,
        const float4* __restrict__ sh_coefficients_rest,
        const float3& position,
        const float3& cam_position,
        const uint primitive_idx,
        const uint sh_layout_slots,
        const float2* __restrict__ sh_value_bounds,
        const uint sh_value_n_cells,
        const uint sh_value_bits) {
        float3 result = 0.5f + 0.28209479177387814f * sh_coefficients_0[primitive_idx];
        if constexpr (ACTIVE_SH_BASES <= 1) {
            return result;
        }
        if constexpr (!SH_Q16) {
            return convert_sh_to_color(
                sh_coefficients_0, sh_coefficients_rest, position, cam_position,
                primitive_idx, static_cast<uint>(ACTIVE_SH_BASES), sh_layout_slots,
                sh_value_bounds, sh_value_n_cells, sh_value_bits);
        } else {
            constexpr uint kRequiredCells = static_cast<uint>(ACTIVE_SH_BASES - 1) * 3u;
            if (__builtin_expect(sh_value_n_cells < kRequiredCells, 0)) {
                return convert_sh_to_color(
                    sh_coefficients_0, sh_coefficients_rest, position, cam_position,
                    primitive_idx, static_cast<uint>(ACTIVE_SH_BASES), sh_layout_slots,
                    sh_value_bounds, sh_value_n_cells, sh_value_bits);
            }
            const float3 direction = safe_normalize(position - cam_position);
            const float x = direction.x;
            const float y = direction.y;
            const float z = direction.z;

            const float2 mm = sh_value_bounds[primitive_idx >> 8];
            const float v_min = mm.x;
            const float v_dec = (mm.y - mm.x) * lfs::core::sh_value::kInvQMaxDevice;
            const uint16_t* const base_ptr =
                reinterpret_cast<const uint16_t*>(sh_coefficients_rest) +
                static_cast<size_t>(primitive_idx >> 5) * (static_cast<size_t>(sh_value_n_cells) * 32u) +
                (primitive_idx & 31u);

            auto coeff = [&]<int Index>() -> float3 {
                constexpr uint off = static_cast<uint>(Index) * 96u;
                return make_float3(
                    fmaf(static_cast<float>(base_ptr[off + 0u]), v_dec, v_min),
                    fmaf(static_cast<float>(base_ptr[off + 32u]), v_dec, v_min),
                    fmaf(static_cast<float>(base_ptr[off + 64u]), v_dec, v_min));
            };

            result = result + (-0.48860251190291987f * y) * coeff.template operator()<0>();
            result = result + (0.48860251190291987f * z) * coeff.template operator()<1>();
            result = result + (-0.48860251190291987f * x) * coeff.template operator()<2>();
            if constexpr (ACTIVE_SH_BASES > 4) {
                const float xx = x * x, yy = y * y, zz = z * z;
                const float xy = x * y, xz = x * z, yz = y * z;
                result = result + (1.0925484305920792f * xy) * coeff.template operator()<3>();
                result = result + (-1.0925484305920792f * yz) * coeff.template operator()<4>();
                result = result + (0.94617469575755997f * zz - 0.31539156525251999f) * coeff.template operator()<5>();
                result = result + (-1.0925484305920792f * xz) * coeff.template operator()<6>();
                result = result + (0.54627421529603959f * xx - 0.54627421529603959f * yy) * coeff.template operator()<7>();
                if constexpr (ACTIVE_SH_BASES > 9) {
                    result = result + (0.59004358992664352f * y * (-3.0f * xx + yy)) * coeff.template operator()<8>();
                    result = result + (2.8906114426405538f * xy * z) * coeff.template operator()<9>();
                    result = result + (0.45704579946446572f * y * (1.0f - 5.0f * zz)) * coeff.template operator()<10>();
                    result = result + (0.3731763325901154f * z * (5.0f * zz - 3.0f)) * coeff.template operator()<11>();
                    result = result + (0.45704579946446572f * x * (1.0f - 5.0f * zz)) * coeff.template operator()<12>();
                    result = result + (1.4453057213202769f * z * (xx - yy)) * coeff.template operator()<13>();
                    result = result + (0.59004358992664352f * x * (-xx + 3.0f * yy)) * coeff.template operator()<14>();
                }
            }
            return result;
        }
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
                    float grad = (i < active) ? grads[i] : 0.0f;
                    float hinge = 0.0f;
                    if (i < active && param.screen_share_max != nullptr &&
                        primitive_idx < static_cast<uint>(param.screen_share_n)) {
                        hinge = lfs::training::screen_share_hinge_extra_grad(
                            param.screen_share_max[primitive_idx],
                            param.screen_share_limit,
                            param.screen_share_penalty,
                            mv.y,
                            param.bias_correction2_sqrt_rcp,
                            eps);
                    }
                    m = beta1 * mv.x + (1.0f - beta1) * (grad + hinge);
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
        if (param.gradient_out != nullptr) {
            if (primitive_idx < static_cast<uint>(param.n_primitives)) {
                for (uint i = 0; i < row_elements; ++i)
                    param.gradient_out[primitive_idx * row_elements + i] += grads[i];
            }
            return;
        }
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

    template <bool RUNTIME_LOG_SCALE = true>
    __device__ inline float scale_regularization_grad(
        const FusedAdamSettings& fused_adam,
        const FusedAdamParam& param,
        const uint element_idx) {
        if (fused_adam.scale_reg_weight <= 0.0f || param.n_elements <= 0)
            return 0.0f;
        if constexpr (RUNTIME_LOG_SCALE) {
            if (fused_adam.scale_reg_log) {
                if (param.param[element_idx] <= -40.0f)
                    return 0.0f;
                return fused_adam.scale_reg_weight * 0.01f /
                       (static_cast<float>(param.n_elements) * fused_adam.scale_reg_normalizer);
            }
        }
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
            const float derivative = lfs::core::sigmoid_derivative_from_logit(param.param[element_idx]);
            grad += fused_adam.opacity_reg_weight * derivative /
                    static_cast<float>(param.n_elements);
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
        using VC = lfs::core::sh_value::DeviceCodec16;
        if (value_q16) {
            float px = 0.0f, py = 0.0f, pz = 0.0f, pw = 0.0f;
            if (k * 4u + 0u < n_value_cells)
                px = VC::decode(param_u16[lfs::core::sh_value::shAtU16(primitive_idx, k * 4u + 0u, n_value_cells)], old_vmm.x, old_vmm.y);
            if (k * 4u + 1u < n_value_cells)
                py = VC::decode(param_u16[lfs::core::sh_value::shAtU16(primitive_idx, k * 4u + 1u, n_value_cells)], old_vmm.x, old_vmm.y);
            if (k * 4u + 2u < n_value_cells)
                pz = VC::decode(param_u16[lfs::core::sh_value::shAtU16(primitive_idx, k * 4u + 2u, n_value_cells)], old_vmm.x, old_vmm.y);
            if (k * 4u + 3u < n_value_cells)
                pw = VC::decode(param_u16[lfs::core::sh_value::shAtU16(primitive_idx, k * 4u + 3u, n_value_cells)], old_vmm.x, old_vmm.y);
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

    // Per-cell shN gradient: fused path reconstructs from dL/dcolor and the SH basis.
    template <int ACTIVE_SH_BASES>
    struct ShNGradFromColor {
        float3 grad_color;
        float bx, by, bz, bxx, byy, bzz, bxy, bxz, byz;
        bool compute;

        __device__ __forceinline__ ShNGradFromColor(
            const float3 mean3d,
            const float3 cam_position,
            const float3 gc,
            const bool compute_sh_grads)
            : grad_color(gc),
              bx(0.0f),
              by(0.0f),
              bz(0.0f),
              bxx(0.0f),
              byy(0.0f),
              bzz(0.0f),
              bxy(0.0f),
              bxz(0.0f),
              byz(0.0f),
              compute(compute_sh_grads) {
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
        }

        __device__ __forceinline__ float4 operator()(
            const uint /*primitive_idx*/,
            const uint k,
            const uint /*slot*/,
            const bool active_slot) const {
            if (!compute || !active_slot)
                return make_float4(0.0f, 0.0f, 0.0f, 0.0f);
            return shN_slot_grad_from_basis<ACTIVE_SH_BASES>(
                k, bx, by, bz, bxx, byy, bzz, bxy, bxz, byz, grad_color);
        }
    };

    // Per-cell shN gradient: standalone path loads the already-swizzled fp32 grad buffer.
    struct ShNGradFromSwizzle {
        const float4* __restrict__ grad4;

        __device__ __forceinline__ float4 operator()(
            const uint /*primitive_idx*/,
            const uint /*k*/,
            const uint slot,
            const bool active_slot) const {
            if (!active_slot || grad4 == nullptr)
                return make_float4(0.0f, 0.0f, 0.0f, 0.0f);
            return grad4[slot];
        }
    };

    // L2 pull of one shN-rest coefficient toward zero: adds its gradient to grad (unless the
    // colour gate blocks the channel) and returns its loss term.
    template <int ACTIVE_SH_BASES>
    __device__ __forceinline__ float add_sh_rest_regularization(
        const FusedAdamSettings& fused_adam,
        const int n_primitives,
        const uint cell_lin,
        const uint colour_reg_mask,
        const float value,
        float& grad) {
        constexpr uint kCoefficientCount = (ACTIVE_SH_BASES - 1u) * 3u;
        if (fused_adam.sh_rest_reg_weight <= 0.0f || cell_lin >= kCoefficientCount || n_primitives <= 0)
            return 0.0f;
        const float reg_scale = fused_adam.sh_rest_reg_weight /
                                (static_cast<float>(n_primitives) * static_cast<float>(kCoefficientCount));
        if (colour_reg_mask & (1u << (cell_lin % 3u)))
            grad += 2.0f * reg_scale * value;
        return reg_scale * value * value;
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
    //
    // GradSource(primitive_idx, k, slot, active_slot) -> float4 cell grads. The fused
    // path reconstructs from color; the standalone path loads a swizzled buffer.
    // blockIdx.x is the 256-splat joint-bounds row (256 threads per block).
    template <int ACTIVE_SH_BASES, bool SEPT_REGULARIZERS = true, typename GradSource>
    __device__ inline void apply_shN_grads_packed_joint(
        const FusedAdamSettings& fused_adam,
        const uint primitive_idx,
        const uint sh_layout_slots,
        GradSource grad_source,
        const uint colour_reg_mask = 7u) {
        using C = lfs::training::joint_adam::DeviceCodec<8>;
        using VC = lfs::core::sh_value::DeviceCodec16;
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
        float local_sh_rest_loss = 0.0f;

        constexpr uint N_SLOTS = (ACTIVE_SH_BASES > 9) ? 12u : (ACTIVE_SH_BASES > 4) ? 6u
                                                                                     : 3u;

        if (touch) {
#pragma unroll
            for (uint k = 0; k < 12u; ++k) {
                if (k >= sh_layout_slots)
                    break;
                const uint slot = shAt(primitive_idx, k, sh_layout_slots);
                const bool active_slot = k < N_SLOTS;
                const float4 gk = grad_source(primitive_idx, k, slot, active_slot);
                float4 pc = load_shN_param_slot(value_q16, value_f16, param_u16, param_h, param4,
                                                primitive_idx, k, slot, n_value_cells, old_vmm);
#pragma unroll
                for (int c = 0; c < 4; ++c) {
                    float pci = (c == 0) ? pc.x : (c == 1) ? pc.y
                                              : (c == 2)   ? pc.z
                                                           : pc.w;
                    float gci = (c == 0) ? gk.x : (c == 1) ? gk.y
                                              : (c == 2)   ? gk.z
                                                           : gk.w;
                    const int64_t cell = static_cast<int64_t>(slot) * 4 + c;
                    if constexpr (SEPT_REGULARIZERS && ACTIVE_SH_BASES > 1)
                        local_sh_rest_loss += add_sh_rest_regularization<ACTIVE_SH_BASES>(
                            fused_adam, p.n_primitives, k * 4u + static_cast<uint>(c), colour_reg_mask, pci, gci);
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
        // value bounds: block-uniform fused min2 (own shared alloc, distinct from min4).
        float v_min = 0.0f;
        float v_max = 0.0f;
        if (value_q16) {
            const float2 r = lfs::core::warp_ops::block_reduce_min2(
                make_float2(local_v_min, -local_v_max));
            v_min = r.x;
            v_max = -r.y;
        }

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
                const float4 gk = grad_source(primitive_idx, k, slot, active_slot);
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
                    float gci = (c == 0) ? gk.x : (c == 1) ? gk.y
                                              : (c == 2)   ? gk.z
                                                           : gk.w;
                    const int64_t cell = static_cast<int64_t>(slot) * 4 + c;
                    // The first pass accumulated the loss; this pass re-encodes the regularized moments.
                    if constexpr (SEPT_REGULARIZERS && ACTIVE_SH_BASES > 1)
                        add_sh_rest_regularization<ACTIVE_SH_BASES>(
                            fused_adam, p.n_primitives, k * 4u + static_cast<uint>(c), colour_reg_mask, pci, gci);
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
                            param_u16[lfs::core::sh_value::shAtU16(
                                primitive_idx, cell_lin, n_value_cells)] =
                                VC::encode(pci, new_vmm.x, new_vmm.y);
                        }
                    }
                }
            }
        }

        // Every thread reaches this reduction, including overhang and masked
        // rows. This leaves at most one global scalar update per block.
        if constexpr (SEPT_REGULARIZERS)
            if (fused_adam.sh_rest_reg_loss_out != nullptr) {
                const float block_sum = lfs::core::warp_ops::block_reduce_sum(local_sh_rest_loss);
                if (threadIdx.x == 0 && block_sum != 0.0f)
                    atomicAdd(fused_adam.sh_rest_reg_loss_out, block_sum);
            }
    }

    // Joint (u, log_s) Adam step over swizzled shN moments of one primitive.
    // Slot count is derived from ACTIVE_SH_BASES. All threads must call
    // when joint_bits==8 (block bounds).
    template <int ACTIVE_SH_BASES, bool SEPT_REGULARIZERS = true>
    __device__ inline void apply_shN_grads_packed(
        const FusedAdamSettings& fused_adam,
        const uint primitive_idx,
        const uint sh_layout_slots,
        const float3 mean3d,
        const float3 cam_position,
        const float3 grad_color,
        const bool compute_sh_grads,
        const uint colour_reg_mask = 7u) {
        const FusedAdamParam& p = fused_adam.shN;
        if (p.gradient_out != nullptr) {
            if (primitive_idx < static_cast<uint>(p.n_primitives)) {
                constexpr uint n_slots = (ACTIVE_SH_BASES > 9) ? 12u : (ACTIVE_SH_BASES > 4) ? 6u
                                                                                             : 3u;
                const ShNGradFromColor<ACTIVE_SH_BASES> source(mean3d, cam_position, grad_color, compute_sh_grads);
                auto* out = reinterpret_cast<float4*>(p.gradient_out);
                for (uint k = 0; k < sh_layout_slots; ++k) {
                    const uint slot = shAt(primitive_idx, k, sh_layout_slots);
                    out[slot] += source(primitive_idx, k, slot, k < n_slots);
                }
            }
            return;
        }
        if (p.joint_bits == 8) {
            apply_shN_grads_packed_joint<ACTIVE_SH_BASES, SEPT_REGULARIZERS>(
                fused_adam, primitive_idx, sh_layout_slots,
                ShNGradFromColor<ACTIVE_SH_BASES>(
                    mean3d, cam_position, grad_color, compute_sh_grads),
                colour_reg_mask);
        }
        // Non-joint state is unsupported (joint is the only codec).
    }

    struct LazyAdamFactors {
        float decay1;      // beta1^dt (1.0f when dt == 0)
        float decay2_half; // beta2^(dt/2)
        float coast_sum;   // r * (1 - r^dt) / (1 - r), r = beta1 / sqrt(beta2)
        float reg_sum;     // dt - beta1 * (1 - beta1^dt) / (1 - beta1)
    };

    __device__ __forceinline__ uint16_t encode_lazy_step(const uint32_t step) {
        if (step == 0u)
            return 0u;
        return static_cast<uint16_t>(1u + ((step - 1u) % 65535u));
    }

    __device__ __forceinline__ uint32_t lazy_missed_steps(
        const uint16_t* __restrict__ last_step,
        const int last_step_n,
        const uint primitive_idx,
        const uint32_t current_step) {
        if (last_step == nullptr || current_step == 0u ||
            primitive_idx >= static_cast<uint>(last_step_n))
            return 0u;
        const uint32_t prev = static_cast<uint32_t>(last_step[primitive_idx]);
        if (prev == 0u)
            return 0u;
        const uint32_t cur = static_cast<uint32_t>(encode_lazy_step(current_step));
        const uint32_t diff = (cur >= prev) ? (cur - prev) : (65535u - prev + cur);
        if (diff <= 1u)
            return 0u;
        return min(diff - 1u, 4096u);
    }

    // Number of pending lazy coast steps up to and including `current_step`
    // for an explicit flush before refine / eval / save.
    __device__ __forceinline__ uint32_t lazy_flush_steps(
        const uint16_t* __restrict__ last_step,
        const int last_step_n,
        const uint primitive_idx,
        const uint32_t current_step) {
        if (last_step == nullptr || current_step == 0u ||
            primitive_idx >= static_cast<uint>(last_step_n))
            return 0u;
        const uint32_t prev = static_cast<uint32_t>(last_step[primitive_idx]);
        if (prev == 0u)
            return 0u;
        const uint32_t cur = static_cast<uint32_t>(encode_lazy_step(current_step));
        const uint32_t diff = (cur >= prev) ? (cur - prev) : (65535u - prev + cur);
        return min(diff, 4096u);
    }

    __device__ __forceinline__ LazyAdamFactors compute_lazy_adam_factors(
        const uint32_t dt,
        const float beta1,
        const float beta2) {
        LazyAdamFactors f{1.0f, 1.0f, 0.0f, 0.0f};
        if (dt == 0u)
            return f;
        const float sqrt_b2 = sqrtf(beta2);
        if (dt == 1u) {
            f.decay1 = beta1;
            f.decay2_half = sqrt_b2;
            f.coast_sum = beta1 / sqrt_b2;
            f.reg_sum = 1.0f - beta1;
            return f;
        }
        const float dtf = static_cast<float>(dt);
        const float d1 = __powf(beta1, dtf);
        const float d2_half = __powf(sqrt_b2, dtf);
        const float r = beta1 / sqrt_b2;
        const float r_dt = d1 / fmaxf(d2_half, 1e-30f);
        const float coast_sum = r * (1.0f - r_dt) / fmaxf(1.0f - r, 1e-6f);
        const float reg_sum = dtf - beta1 * (1.0f - d1) / fmaxf(1.0f - beta1, 1e-6f);
        f.decay1 = d1;
        f.decay2_half = d2_half;
        f.coast_sum = coast_sum;
        f.reg_sum = fmaxf(reg_sum, 0.0f);
        return f;
    }

    __device__ __forceinline__ void sparse_rebound_mark(
        const SparseReboundState& rebound,
        const int slot,
        const uint bidx,
        const float needed_u_abs,
        const float needed_s_hi) {
        if (bidx >= rebound.n_blocks)
            return;
        if (needed_u_abs > 0.0f && isfinite(needed_u_abs)) {
            atomicMax(rebound.needed_u + static_cast<size_t>(slot) * rebound.n_blocks + bidx,
                      __float_as_uint(needed_u_abs));
        }
        if (slot < 6 && needed_s_hi > 0.0f && isfinite(needed_s_hi)) {
            atomicMax(rebound.needed_s + static_cast<size_t>(slot) * rebound.n_blocks + bidx,
                      __float_as_uint(needed_s_hi));
        }
        uint32_t* const slot_dirty =
            rebound.dirty_bits + static_cast<size_t>(slot) * rebound.words_per_slot;
        const uint word = bidx >> 5;
        const uint bit = 1u << (bidx & 31u);
        if ((atomicOr(slot_dirty + word, bit) & bit) == 0u) {
            const uint pos = atomicAdd(rebound.counts_post + slot, 1u);
            if (pos < rebound.n_blocks) {
                rebound.rebound_list[static_cast<size_t>(slot) * rebound.n_blocks + pos] = bidx;
            }
        }
    }

    // Row learning rate after the frozen and crop-box damping masks. Returns false
    // when a zero mask scale freezes the row, which then must not step.
    __device__ __forceinline__ bool effective_row_step_size(
        const FusedAdamParam& param,
        const uint primitive_idx,
        const float base_step,
        float& out) {
        out = base_step;
        if (param.frozen_mask != nullptr &&
            primitive_idx < static_cast<uint>(param.frozen_mask_size) &&
            param.frozen_mask[primitive_idx]) {
            if (param.frozen_lr_scale == 0.0f)
                return false;
            out *= param.frozen_lr_scale;
        }
        if (param.crop_damping_mask != nullptr &&
            primitive_idx < static_cast<uint>(param.crop_damping_mask_size) &&
            param.crop_damping_mask[primitive_idx]) {
            if (param.cropbox_lr_scale == 0.0f)
                return false;
            out *= param.cropbox_lr_scale;
        }
        return true;
    }

    // Single-pass 16-bit joint Adam step for one contributing primitive under
    // read-only expand-only block bounds, with optional closed-form lazy catch-up
    // over missed steps (incl. dc_reg on sh0 and screen_share hinge on scaling).
    template <uint ROW_DIM, bool HAS_DC_REG = false, bool REPORT_DC_LOSS = false>
    __device__ __forceinline__ void adam_step_row_sparse(
        const float (&grads)[ROW_DIM],
        const FusedAdamParam& param,
        const uint primitive_idx,
        const uint bidx,
        const float beta1,
        const float beta2,
        const float eps,
        const SparseReboundState& rebound,
        const int rebound_slot,
        const float step_size_scale = 1.0f,
        const LazyAdamFactors& lazy = LazyAdamFactors{1.0f, 1.0f, 0.0f, 0.0f},
        const uint32_t missed_steps = 0u,
        const float dc_limit = 0.0f,
        const float dc_grad_scale = 0.0f,
        const float dc_reported_loss_weight = 0.0f,
        const uint colour_reg_mask = 7u,
        float* dc_loss_out = nullptr) {
        using C = lfs::training::joint_adam::DeviceCodec<16>;
        if (!param.enabled || param.n_attributes <= 0 || param.joint_bits != 16 ||
            param.joint_packed == nullptr || param.joint_bounds == nullptr)
            return;

        float row_step_size;
        if (!effective_row_step_size(param, primitive_idx, param.step_size * step_size_scale, row_step_size))
            return;

        const uint n_attr = static_cast<uint>(param.n_attributes);
        const uint base = primitive_idx * n_attr;
        if (base >= static_cast<uint>(param.n_elements))
            return;
        const uint row = min(n_attr, static_cast<uint>(param.n_elements) - base);

        const float4 mm = *reinterpret_cast<const float4*>(param.joint_bounds + 4u * bidx);
        const float u_span = mm.y - mm.x;
        const float s_span = mm.w - mm.z;
        const float u_dec = u_span * C::kInvQMax;
        const float s_dec = s_span * C::kInvQMax;
        const float u_enc = C::kQMax / fmaxf(u_span, lfs::training::joint_adam::kEpsDevice);
        const float s_enc = C::kQMax / fmaxf(s_span, lfs::training::joint_adam::kEpsDevice);
        uint32_t* const packed_u32 = reinterpret_cast<uint32_t*>(param.joint_packed);
        const bool has_screen_share =
            param.screen_share_max != nullptr && primitive_idx < static_cast<uint>(param.screen_share_n);
        const float screen_share_val = has_screen_share ? param.screen_share_max[primitive_idx] : 0.0f;
        const bool has_hinge = has_screen_share &&
                               param.screen_share_limit > 0.0f &&
                               param.screen_share_limit < 1.0f &&
                               screen_share_val > param.screen_share_limit &&
                               param.screen_share_penalty > 0.0f;

#pragma unroll
        for (uint i = 0; i < ROW_DIM; ++i) {
            if (i < row) {
                const int64_t cell = static_cast<int64_t>(base + i);
                const uint32_t word = packed_u32[cell];
                const float u_val = mm.x + static_cast<float>(word & 0xffffu) * u_dec;
                const float log_s_val = mm.z + static_cast<float>(word >> 16u) * s_dec;
                const float sqrt_g2 = C::inverse_sqrt_g2(log_s_val);
                float g2 = sqrt_g2 * sqrt_g2;
                float g1 = (g2 == 0.0f) ? 0.0f : u_val * (sqrt_g2 + lfs::training::joint_adam::kEpsDevice);

                float p_val = param.param[base + i];

                if (lazy.decay1 < 1.0f) {
                    if constexpr (HAS_DC_REG) {
                        const float delta_miss = (p_val > dc_limit)
                                                     ? (p_val - dc_limit)
                                                     : ((p_val < -dc_limit) ? (p_val + dc_limit) : 0.0f);
                        const bool dc_on = (dc_grad_scale > 0.0f) && ((colour_reg_mask & (1u << i)) != 0u);
                        const float g_reg_miss = dc_on ? (dc_grad_scale * delta_miss) : 0.0f;
                        if (g2 > 0.0f || g_reg_miss != 0.0f) {
                            float step_miss = 0.0f;
                            if (g2 > 0.0f) {
                                const float denom_old = sqrt_g2 * param.bias_correction2_sqrt_rcp + eps;
                                step_miss = row_step_size * (g1 / denom_old) * lazy.coast_sum;
                            }
                            float g_reg_eff = g_reg_miss;
                            if (g_reg_miss != 0.0f) {
                                const float v_mid =
                                    g2 * lazy.decay2_half + g_reg_miss * g_reg_miss * (1.0f - lazy.decay2_half);
                                const float denom_reg = sqrtf(v_mid) * param.bias_correction2_sqrt_rcp + eps;
                                const float raw_reg_step = row_step_size * (g_reg_miss / denom_reg) * lazy.reg_sum;
                                float reg_step = raw_reg_step;
                                if (fabsf(raw_reg_step) >= fabsf(delta_miss)) {
                                    reg_step = copysignf(fabsf(delta_miss), raw_reg_step);
                                    g_reg_eff = 0.0f;
                                }
                                step_miss += reg_step;
                            }
                            p_val -= step_miss;
                            g1 = g1 * lazy.decay1 + g_reg_eff * (1.0f - lazy.decay1);
                            const float d2 = lazy.decay2_half * lazy.decay2_half;
                            g2 = g2 * d2 + g_reg_miss * g_reg_miss * (1.0f - d2);
                        }
                    } else if (__builtin_expect(has_hinge, 0)) {
                        const uint steps = min(missed_steps, 32u);
                        for (uint s = 0u; s < steps; ++s) {
                            const float h = lfs::training::screen_share_hinge_extra_grad(
                                screen_share_val, param.screen_share_limit, param.screen_share_penalty,
                                g2, param.bias_correction2_sqrt_rcp, eps);
                            g1 = beta1 * g1 + (1.0f - beta1) * h;
                            g2 = beta2 * g2;
                            if (g1 != 0.0f) {
                                const float d = sqrtf(g2) * param.bias_correction2_sqrt_rcp + eps;
                                p_val -= row_step_size * g1 / d;
                            }
                        }
                    } else if (g2 > 0.0f) {
                        const float denom_old = sqrt_g2 * param.bias_correction2_sqrt_rcp + eps;
                        p_val -= row_step_size * (g1 / denom_old) * lazy.coast_sum;
                        g1 *= lazy.decay1;
                        const float d2 = lazy.decay2_half * lazy.decay2_half;
                        g2 *= d2;
                    }
                }

                float grad = grads[i];
                if constexpr (HAS_DC_REG) {
                    if (dc_grad_scale > 0.0f) {
                        const float delta = (p_val > dc_limit)
                                                ? (p_val - dc_limit)
                                                : ((p_val < -dc_limit) ? (p_val + dc_limit) : 0.0f);
                        if (colour_reg_mask & (1u << i))
                            grad += dc_grad_scale * delta;
                        if constexpr (REPORT_DC_LOSS) {
                            if (dc_loss_out != nullptr)
                                *dc_loss_out += dc_reported_loss_weight * delta * delta;
                        }
                    }
                }
                const float hinge = has_hinge
                                        ? lfs::training::screen_share_hinge_extra_grad(
                                              screen_share_val,
                                              param.screen_share_limit,
                                              param.screen_share_penalty,
                                              g2,
                                              param.bias_correction2_sqrt_rcp,
                                              eps)
                                        : 0.0f;
                const float m = beta1 * g1 + (1.0f - beta1) * (grad + hinge);
                const float v = beta2 * g2 + (1.0f - beta2) * grad * grad;
                const float sqrt_v = sqrtf(fmaxf(v, 0.0f));
                const float denom = sqrt_v * param.bias_correction2_sqrt_rcp + eps;
                p_val -= row_step_size * m / denom;
                param.param[base + i] = p_val;

                const float prim_x = m / (sqrt_v + lfs::training::joint_adam::kEpsDevice);
                const float prim_y = C::forward_sqrt_g2(sqrt_v);
                const float u_norm = (prim_x - mm.x) * u_enc;
                const float s_norm = (prim_y - mm.z) * s_enc;
                if (__builtin_expect(u_norm < 0.0f || u_norm > C::kQMax || s_norm < 0.0f || s_norm > C::kQMax, 0)) {
                    sparse_rebound_mark(rebound, rebound_slot, bidx, fabsf(prim_x), prim_y);
                }
                const uint32_t u_q = static_cast<uint32_t>(fminf(fmaxf(roundf(u_norm), 0.0f), C::kQMax));
                const uint32_t s_q = static_cast<uint32_t>(fminf(fmaxf(roundf(s_norm), 0.0f), C::kQMax));
                packed_u32[cell] = u_q | (s_q << 16u);
            }
        }
    }

    __device__ __forceinline__ uint32_t sparse_sr_hash(
        const uint32_t step_seed,
        const uint32_t primitive_idx,
        const uint32_t channel_tag) {
        uint32_t h = (step_seed * 0x9e3779b9u) ^ (primitive_idx * 0x85ebca6bu) ^ (channel_tag * 0xc2b2ae35u);
        h ^= h >> 16u;
        h *= 0x7feb352du;
        h ^= h >> 15u;
        h *= 0x846ca68bu;
        h ^= h >> 16u;
        return h;
    }

    // Ultra-lightweight exact invisible-row 16-bit joint Adam decay helper.
    // Skips zero-variance cells completely, skips param load/store when m==0,
    // cancels sqrt(g2) in log-space for normal cells (log_s > 16, g==0, eps<=1e-14),
    // and uses unbiased stochastic rounding on non-zero (u, log_s) updates so fixed
    // expand-only block bounds never freeze u or systematically bias log_s decay.
    template <uint ROW_DIM, bool CHECK_SCREEN_SHARE = false, bool HAS_EXTRA_GRAD = false>
    __device__ __forceinline__ void adam_decay_row_16_invisible(
        const FusedAdamParam& param,
        const uint primitive_idx,
        const float beta1,
        const float beta2,
        const float eps,
        const SparseReboundState& rebound,
        const int rebound_slot,
        const uint32_t step_seed,
        const float step_size_scale = 1.0f,
        const float3 extra_grad = make_float3(0.0f, 0.0f, 0.0f)) {
        using C = lfs::training::joint_adam::DeviceCodec<16>;
        if (!param.enabled || param.n_attributes <= 0 || param.joint_bits != 16 ||
            param.joint_packed == nullptr || param.joint_bounds == nullptr)
            return;

        float row_step_size;
        if (!effective_row_step_size(param, primitive_idx, param.step_size * step_size_scale, row_step_size))
            return;

        const uint n_attr = static_cast<uint>(param.n_attributes);
        const uint base = primitive_idx * n_attr;
        if (base >= static_cast<uint>(param.n_elements))
            return;
        const uint row = min(n_attr, static_cast<uint>(param.n_elements) - base);

        const uint bidx = blockIdx.x;
        const float4 mm = *reinterpret_cast<const float4*>(param.joint_bounds + 4u * bidx);
        const float u_span = mm.y - mm.x;
        const float s_span = mm.w - mm.z;
        const float u_dec = u_span * C::kInvQMax;
        const float s_dec = s_span * C::kInvQMax;
        const float u_enc = C::kQMax / fmaxf(u_span, lfs::training::joint_adam::kEpsDevice);
        const float s_enc = C::kQMax / fmaxf(s_span, lfs::training::joint_adam::kEpsDevice);
        uint32_t* const packed_u32 = reinterpret_cast<uint32_t*>(param.joint_packed);

        const float sqrt_b2 = sqrtf(beta2);
        const float inv_sqrt_b2 = 1.0f / sqrt_b2;
        const float u_decay_ratio = beta1 * inv_sqrt_b2;
        const float log_sqrt_b2 = __logf(sqrt_b2);
        const float u_step_scale = row_step_size * (u_decay_ratio / param.bias_correction2_sqrt_rcp);
        const bool eps_negligible = (eps <= 1e-14f);
        const float u_zero_tol = 1e-3f * u_dec;

        float screen_share_val = 0.0f;
        bool has_hinge = false;
        if constexpr (CHECK_SCREEN_SHARE) {
            if (param.screen_share_max != nullptr &&
                primitive_idx < static_cast<uint>(param.screen_share_n)) {
                screen_share_val = param.screen_share_max[primitive_idx];
                has_hinge = (param.screen_share_limit > 0.0f &&
                             param.screen_share_limit < 1.0f &&
                             screen_share_val > param.screen_share_limit &&
                             param.screen_share_penalty > 0.0f);
            }
        }

#pragma unroll
        for (uint i = 0; i < ROW_DIM; ++i) {
            if (i < row) {
                const int64_t cell = static_cast<int64_t>(base + i);
                const uint32_t word = packed_u32[cell];
                const uint32_t s_q_in = word >> 16u;
                float g_reg = 0.0f;
                if constexpr (HAS_EXTRA_GRAD) {
                    g_reg = (i == 0u) ? extra_grad.x : (i == 1u) ? extra_grad.y
                                                                 : extra_grad.z;
                }
                if (s_q_in == 0u && g_reg == 0.0f && !has_hinge)
                    continue;

                const float u_raw = mm.x + static_cast<float>(word & 0xffffu) * u_dec;
                const float u_val = (fabsf(u_raw) <= u_zero_tol) ? 0.0f : u_raw;
                const float log_s_val = mm.z + static_cast<float>(s_q_in) * s_dec;

                float2 prim;
                if (__builtin_expect(g_reg == 0.0f && !has_hinge && eps_negligible && log_s_val > 16.0f, 1)) {
                    if (u_val != 0.0f) {
                        param.param[base + i] -= u_step_scale * u_val;
                    }
                    prim = make_float2(u_val * u_decay_ratio, log_s_val + log_sqrt_b2);
                } else {
                    const float sqrt_g2 = C::inverse_sqrt_g2(log_s_val);
                    const float g2 = sqrt_g2 * sqrt_g2;
                    const float g1 = (g2 == 0.0f) ? 0.0f : u_val * (sqrt_g2 + lfs::training::joint_adam::kEpsDevice);

                    float hinge = 0.0f;
                    if constexpr (CHECK_SCREEN_SHARE) {
                        if (has_hinge) {
                            hinge = lfs::training::screen_share_hinge_extra_grad(
                                screen_share_val, param.screen_share_limit, param.screen_share_penalty,
                                g2, param.bias_correction2_sqrt_rcp, eps);
                        }
                    }

                    const float m = beta1 * g1 + (1.0f - beta1) * (g_reg + hinge);
                    const float v = beta2 * g2 + (1.0f - beta2) * (g_reg * g_reg);
                    if (m != 0.0f) {
                        const float denom = sqrtf(v) * param.bias_correction2_sqrt_rcp + eps;
                        param.param[base + i] -= row_step_size * m / denom;
                    }
                    prim = C::g1g2_to_us(m, v);
                }

                const float u_norm = (prim.x - mm.x) * u_enc;
                const float s_norm = (prim.y - mm.z) * s_enc;
                if constexpr (HAS_EXTRA_GRAD || CHECK_SCREEN_SHARE) {
                    if (__builtin_expect((g_reg != 0.0f || has_hinge) &&
                                             (u_norm < 0.0f || u_norm > C::kQMax || s_norm < 0.0f || s_norm > C::kQMax),
                                         0)) {
                        sparse_rebound_mark(rebound, rebound_slot, bidx, fabsf(prim.x), prim.y);
                    }
                }
                const uint32_t h = sparse_sr_hash(step_seed, primitive_idx,
                                                  (static_cast<uint32_t>(rebound_slot) << 3u) | i);
                const float r_u = static_cast<float>(h & 0xffffu) * (1.0f / 65536.0f);
                const float r_s = static_cast<float>(h >> 16u) * (1.0f / 65536.0f);
                const bool use_sr_s = (g_reg == 0.0f && !has_hinge && log_s_val > 16.0f && s_norm >= 2.0f);
                const bool use_sr_u = (use_sr_s && prim.x != 0.0f && fabsf(u_val) < 16.0f * u_dec);
                const float u_qf = use_sr_u ? floorf(u_norm + r_u) : roundf(u_norm);
                const float s_qf = (use_sr_s && prim.y != 0.0f) ? floorf(s_norm + r_s) : roundf(s_norm);
                const uint32_t u_q = static_cast<uint32_t>(fminf(fmaxf(u_qf, 0.0f), C::kQMax));
                const uint32_t s_q = static_cast<uint32_t>(fminf(fmaxf(s_qf, 0.0f), C::kQMax));
                const uint32_t out_word = u_q | (s_q << 16u);
                if (out_word != word)
                    packed_u32[cell] = out_word;
            }
        }
    }

    // Closed-form multi-step catch-up for one 16-bit joint Adam row under read-only
    // block bounds. Applies the coast of lazily skipped zero-gradient steps when a
    // splat contributes again, and in flush_lazy_geom_cu before full-model consumers.
    template <uint ROW_DIM, bool HAS_DC_REG = false>
    __device__ __forceinline__ void adam_catchup_row_16_lazy(
        const FusedAdamParam& param,
        const uint primitive_idx,
        const uint bidx,
        const float beta2,
        const float eps,
        const LazyAdamFactors& lazy,
        const uint32_t missed_steps,
        const uint32_t step_seed,
        const int slot_tag,
        const float dc_limit = 0.0f,
        const float dc_grad_scale = 0.0f) {
        using C = lfs::training::joint_adam::DeviceCodec<16>;
        if (missed_steps == 0u || lazy.decay1 >= 1.0f || !param.enabled ||
            param.n_attributes <= 0 || param.joint_bits != 16 ||
            param.joint_packed == nullptr || param.joint_bounds == nullptr)
            return;

        float row_step_size;
        if (!effective_row_step_size(param, primitive_idx, param.step_size, row_step_size))
            return;

        const uint n_attr = static_cast<uint>(param.n_attributes);
        const uint base = primitive_idx * n_attr;
        if (base >= static_cast<uint>(param.n_elements))
            return;
        const uint row = min(n_attr, static_cast<uint>(param.n_elements) - base);

        const float4 mm = *reinterpret_cast<const float4*>(param.joint_bounds + 4u * bidx);
        const float u_span = mm.y - mm.x;
        const float s_span = mm.w - mm.z;
        if (u_span <= 1e-6f || s_span <= 1e-6f)
            return;
        const float u_dec = u_span * C::kInvQMax;
        const float s_dec = s_span * C::kInvQMax;
        const float u_enc = C::kQMax / u_span;
        const float s_enc = C::kQMax / s_span;
        uint32_t* const packed_u32 = reinterpret_cast<uint32_t*>(param.joint_packed);

        const float log_sqrt_b2 = __logf(sqrtf(beta2));
        const float d_log_s = static_cast<float>(missed_steps) * log_sqrt_b2;
        const float u_decay_dt = lazy.decay1 / fmaxf(lazy.decay2_half, 1e-30f);
        const float u_step_dt = (row_step_size / param.bias_correction2_sqrt_rcp) * lazy.coast_sum;
        const bool eps_negligible = (eps <= 1e-14f);
        const float u_zero_tol = 1e-3f * u_dec;

#pragma unroll
        for (uint i = 0; i < ROW_DIM; ++i) {
            if (i < row) {
                const int64_t cell = static_cast<int64_t>(base + i);
                const uint32_t word = packed_u32[cell];
                const uint32_t s_q_in = word >> 16u;

                float g_reg_miss = 0.0f;
                float delta_miss = 0.0f;
                float p_val = 0.0f;
                bool p_loaded = false;
                if constexpr (HAS_DC_REG) {
                    if (dc_grad_scale > 0.0f) {
                        p_val = param.param[base + i];
                        p_loaded = true;
                        delta_miss = (p_val > dc_limit)
                                         ? (p_val - dc_limit)
                                         : ((p_val < -dc_limit) ? (p_val + dc_limit) : 0.0f);
                        g_reg_miss = dc_grad_scale * delta_miss;
                    }
                }
                if (s_q_in == 0u && g_reg_miss == 0.0f)
                    continue;

                const float u_raw = mm.x + static_cast<float>(word & 0xffffu) * u_dec;
                const float u_val = (fabsf(u_raw) <= u_zero_tol) ? 0.0f : u_raw;
                const float log_s_val = mm.z + static_cast<float>(s_q_in) * s_dec;

                float2 prim;
                if (__builtin_expect(g_reg_miss == 0.0f && eps_negligible && log_s_val > 16.0f, 1)) {
                    if (u_val != 0.0f) {
                        if (!p_loaded)
                            p_val = param.param[base + i];
                        param.param[base + i] = p_val - u_step_dt * u_val;
                    }
                    prim = make_float2(u_val * u_decay_dt, fmaxf(log_s_val + d_log_s, 0.0f));
                } else {
                    if (!p_loaded)
                        p_val = param.param[base + i];
                    const float sqrt_g2 = C::inverse_sqrt_g2(log_s_val);
                    float g2 = sqrt_g2 * sqrt_g2;
                    float g1 = (g2 == 0.0f) ? 0.0f : u_val * (sqrt_g2 + lfs::training::joint_adam::kEpsDevice);
                    if constexpr (HAS_DC_REG) {
                        float step_miss = 0.0f;
                        if (g2 > 0.0f) {
                            const float denom_old = sqrt_g2 * param.bias_correction2_sqrt_rcp + eps;
                            step_miss = row_step_size * (g1 / denom_old) * lazy.coast_sum;
                        }
                        float g_reg_eff = g_reg_miss;
                        if (g_reg_miss != 0.0f) {
                            const float v_mid =
                                g2 * lazy.decay2_half + g_reg_miss * g_reg_miss * (1.0f - lazy.decay2_half);
                            const float denom_reg = sqrtf(v_mid) * param.bias_correction2_sqrt_rcp + eps;
                            const float raw_reg_step = row_step_size * (g_reg_miss / denom_reg) * lazy.reg_sum;
                            float reg_step = raw_reg_step;
                            if (fabsf(raw_reg_step) >= fabsf(delta_miss)) {
                                reg_step = copysignf(fabsf(delta_miss), raw_reg_step);
                                g_reg_eff = 0.0f;
                            }
                            step_miss += reg_step;
                        }
                        if (step_miss != 0.0f) {
                            p_val -= step_miss;
                            param.param[base + i] = p_val;
                        }
                        g1 = g1 * lazy.decay1 + g_reg_eff * (1.0f - lazy.decay1);
                        const float d2 = lazy.decay2_half * lazy.decay2_half;
                        g2 = g2 * d2 + g_reg_miss * g_reg_miss * (1.0f - d2);
                        prim = C::g1g2_to_us(g1, g2);
                    } else {
                        if (g2 > 0.0f) {
                            const float denom_old = sqrt_g2 * param.bias_correction2_sqrt_rcp + eps;
                            p_val -= row_step_size * (g1 / denom_old) * lazy.coast_sum;
                            param.param[base + i] = p_val;
                            g1 *= lazy.decay1;
                            const float d2 = lazy.decay2_half * lazy.decay2_half;
                            g2 *= d2;
                        }
                        prim = C::g1g2_to_us(g1, g2);
                    }
                }

                const float u_norm = (prim.x - mm.x) * u_enc;
                const float s_norm = (prim.y - mm.z) * s_enc;
                const uint32_t h = sparse_sr_hash(step_seed, primitive_idx,
                                                  (static_cast<uint32_t>(slot_tag) << 3u) | i);
                const float r_u = static_cast<float>(h & 0xffffu) * (1.0f / 65536.0f);
                const float r_s = static_cast<float>(h >> 16u) * (1.0f / 65536.0f);
                const bool use_sr_s = (g_reg_miss == 0.0f && log_s_val > 16.0f && s_norm >= 2.0f);
                const bool use_sr_u = (use_sr_s && prim.x != 0.0f && fabsf(u_val) < 16.0f * u_dec);
                const float u_qf = use_sr_u ? floorf(u_norm + r_u) : roundf(u_norm);
                const float s_qf = (use_sr_s && prim.y != 0.0f) ? floorf(s_norm + r_s) : roundf(s_norm);
                const uint32_t u_q = static_cast<uint32_t>(fminf(fmaxf(u_qf, 0.0f), C::kQMax));
                const uint32_t s_q = static_cast<uint32_t>(fminf(fmaxf(s_qf, 0.0f), C::kQMax));
                const uint32_t out_word = u_q | (s_q << 16u);
                if (out_word != word)
                    packed_u32[cell] = out_word;
            }
        }
    }

    // Evaluate SH-rest basis function B_i(dir) and its partial derivatives (dB_i/dx, dB_i/dy, dB_i/dz)
    // for rest coefficient index i in [0, 14]. Recomputes quadratic terms on the fly to keep
    // register pressure low inside streamed SH backward+Adam loops.
    __device__ __forceinline__ float4 shN_basis_val_and_deriv(
        const uint coeff_idx,
        float x,
        float y,
        float z) {
        // Prevent LICM from hoisting degree-2/3 quadratic products out of the caller loop,
        // which otherwise forces local-memory register spills across all 45 SH channels.
        asm("" : "+f"(x), "+f"(y), "+f"(z));
        switch (coeff_idx) {
        case 0u:
            return make_float4(-0.48860251190291987f * y, 0.0f, -0.48860251190291987f, 0.0f);
        case 1u:
            return make_float4(0.48860251190291987f * z, 0.0f, 0.0f, 0.48860251190291987f);
        case 2u:
            return make_float4(-0.48860251190291987f * x, -0.48860251190291987f, 0.0f, 0.0f);
        case 3u:
            return make_float4(1.0925484305920792f * x * y, 1.0925484305920792f * y, 1.0925484305920792f * x, 0.0f);
        case 4u:
            return make_float4(-1.0925484305920792f * y * z, 0.0f, -1.0925484305920792f * z, -1.0925484305920792f * y);
        case 5u:
            return make_float4(0.94617469575755997f * z * z - 0.31539156525251999f, 0.0f, 0.0f, 1.8923493915151202f * z);
        case 6u:
            return make_float4(-1.0925484305920792f * x * z, -1.0925484305920792f * z, 0.0f, -1.0925484305920792f * x);
        case 7u:
            return make_float4(0.54627421529603959f * (x * x - y * y),
                               1.0925484305920792f * x, -1.0925484305920792f * y, 0.0f);
        case 8u: {
            const float xx = x * x, yy = y * y;
            return make_float4(0.59004358992664352f * y * (-3.0f * xx + yy),
                               -3.5402615395598609f * x * y,
                               -1.7701307697799304f * (xx - yy),
                               0.0f);
        }
        case 9u:
            return make_float4(2.8906114426405538f * x * y * z,
                               2.8906114426405538f * y * z,
                               2.8906114426405538f * x * z,
                               2.8906114426405538f * x * y);
        case 10u: {
            const float zz = z * z;
            return make_float4(0.45704579946446572f * y * (1.0f - 5.0f * zz),
                               0.0f,
                               0.45704579946446572f - 2.2852289973223288f * zz,
                               -4.5704579946446566f * y * z);
        }
        case 11u: {
            const float zz = z * z;
            return make_float4(0.3731763325901154f * z * (5.0f * zz - 3.0f),
                               0.0f,
                               0.0f,
                               5.597644988851731f * zz - 1.1195289977703462f);
        }
        case 12u: {
            const float zz = z * z;
            return make_float4(0.45704579946446572f * x * (1.0f - 5.0f * zz),
                               0.45704579946446572f - 2.2852289973223288f * zz,
                               0.0f,
                               -4.5704579946446566f * x * z);
        }
        case 13u: {
            const float xx = x * x, yy = y * y;
            return make_float4(1.4453057213202769f * z * (xx - yy),
                               2.8906114426405538f * x * z,
                               -2.8906114426405538f * y * z,
                               1.4453057213202769f * (xx - yy));
        }
        case 14u: {
            const float xx = x * x, yy = y * y;
            return make_float4(0.59004358992664352f * x * (-xx + 3.0f * yy),
                               -1.7701307697799304f * (xx - yy),
                               3.5402615395598609f * x * y,
                               0.0f);
        }
        default:
            return make_float4(0.0f, 0.0f, 0.0f, 0.0f);
        }
    }

    template <int ACTIVE_SH_BASES>
    __device__ inline float3 convert_sh_to_color_backward_grads_from_color(
        const float4* sh_coefficients_rest,
        const float3 grad_color,
        const float3& position,
        const float3& cam_position,
        const uint primitive_idx,
        const uint sh_layout_slots,
        float* __restrict__ sh0_grads_out,
        const float2* __restrict__ sh_value_bounds = nullptr,
        const uint sh_value_n_cells = 0u,
        const uint sh_value_bits = 0u);

    struct ShBlockMeta {
        float u_min;
        float s_min;
        float u_dec;
        float u_enc;
        float s_enc;
        float v_min;
        float v_dec;
        float v_enc;
        float reg_scale;
        float bc2_sqrt_rcp;
        uint n_value_cells;
    };

    // Fused single-read SH-rest view-direction backward + single-pass 8-bit joint Adam
    // with closed-form lazy catch-up over missed steps (incl. SH-rest regularization).
    // Invoked per 256-splat block (bidx = blockIdx.x) so joint/value bounds live in CTA
    // shared memory and every 32-splat SH tile load/store is 100% warp-coalesced.
    template <int ACTIVE_SH_BASES, bool VALUE_Q16, bool REPORT_REG_LOSS = false>
    __device__ __forceinline__ float3 shN_backward_and_adam_sparse(
        const FusedAdamParam& p,
        const float3 mean3d,
        const float3 cam_pos,
        const uint primitive_idx,
        const uint sh_layout_slots,
        const float3 grad_color,
        const uint colour_reg_mask,
        const float beta1,
        const float beta2,
        const float eps,
        const float lazy_decay1,
        const float3* __restrict__ s_lazy_aux,
        const ShBlockMeta& meta,
        const float* __restrict__ s_sqrt_g2_lut,
        uint32_t* __restrict__ s_over_state,
        float& sh_rest_loss_out) {
        using C = lfs::training::joint_adam::DeviceCodec<8>;

        float row_step_size = p.step_size;
        uint flags = colour_reg_mask | 32u; // bit 5: apply_step
        if (p.frozen_mask != nullptr &&
            primitive_idx < static_cast<uint>(p.frozen_mask_size) &&
            p.frozen_mask[primitive_idx]) {
            if (p.frozen_lr_scale == 0.0f)
                flags &= ~32u;
            else
                row_step_size *= p.frozen_lr_scale;
        }
        if (p.crop_damping_mask != nullptr &&
            primitive_idx < static_cast<uint>(p.crop_damping_mask_size) &&
            p.crop_damping_mask[primitive_idx]) {
            if (p.cropbox_lr_scale == 0.0f)
                flags &= ~32u;
            else
                row_step_size *= p.cropbox_lr_scale;
        }

        const bool value_f16 = (!VALUE_Q16) && (p.sh_value_bits == 16);

        const float3 raw_direction = mean3d - cam_pos;
        const float norm_sq = raw_direction.x * raw_direction.x + raw_direction.y * raw_direction.y +
                              raw_direction.z * raw_direction.z;
        const float inv_norm = rsqrtf(fmaxf(norm_sq, 1e-12f));
        const float bx = raw_direction.x * inv_norm;
        const float by = raw_direction.y * inv_norm;
        const float bz = raw_direction.z * inv_norm;
        constexpr float NORM_SQ_GRAD_MIN = 1e-6f;
        constexpr float INV_NORM_CUBED_MAX = 1e6f;
        const float norm_sq_safe = fmaxf(norm_sq, NORM_SQ_GRAD_MIN);
        const float dir_grad_scale =
            norm_sq * fminf(rsqrtf(norm_sq_safe * norm_sq_safe * norm_sq_safe), INV_NORM_CUBED_MAX);

        constexpr uint N_SLOTS = (ACTIVE_SH_BASES > 9) ? 12u : (ACTIVE_SH_BASES > 4) ? 6u
                                                                                     : 3u;
        constexpr uint kActiveCoeffs = static_cast<uint>(ACTIVE_SH_BASES - 1);
        constexpr uint kActiveCells = kActiveCoeffs * 3u;
        const uint slot0 = shAt(primitive_idx, 0u, sh_layout_slots);
        uint2* packed_ptr = reinterpret_cast<uint2*>(p.joint_packed) + slot0;
        uint16_t* u16_ptr =
            VALUE_Q16 ? (reinterpret_cast<uint16_t*>(p.param) +
                         lfs::core::sh_value::shAtU16(primitive_idx, 0u, meta.n_value_cells))
                      : nullptr;

        float gd_x = 0.0f;
        float gd_y = 0.0f;
        float gd_z = 0.0f;
        float local_sh_rest_loss = 0.0f;

        uint coeff_idx = 0u;
        uint rgb = 0u;
        float4 bd = shN_basis_val_and_deriv(0u, bx, by, bz);

#pragma unroll 1
        for (uint k = 0; k < N_SLOTS; ++k) {
            if (k >= sh_layout_slots)
                break;
            const bool apply_step = (flags & 32u) != 0u;
            uint2 m_words = apply_step ? *packed_ptr : make_uint2(0u, 0u);
            uint32_t q_pack0 = 0u;
            uint32_t q_pack1 = 0u;
            float4 pc_nonq16 = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
            if constexpr (VALUE_Q16) {
                const uint cell0 = k * 4u;
                const uint nvc = meta.n_value_cells;
                const uint32_t q0 = (cell0 + 0u < kActiveCells && cell0 + 0u < nvc) ? u16_ptr[0u] : 0u;
                const uint32_t q1 = (cell0 + 1u < kActiveCells && cell0 + 1u < nvc) ? u16_ptr[32u] : 0u;
                const uint32_t q2 = (cell0 + 2u < kActiveCells && cell0 + 2u < nvc) ? u16_ptr[64u] : 0u;
                const uint32_t q3 = (cell0 + 3u < kActiveCells && cell0 + 3u < nvc) ? u16_ptr[96u] : 0u;
                q_pack0 = q0 | (q1 << 16u);
                q_pack1 = q2 | (q3 << 16u);
            } else {
                const uint slot = slot0 + k * config::sh_reorder_size;
                if (value_f16) {
                    const __half* const ph = reinterpret_cast<const __half*>(p.param) + slot * 4u;
                    pc_nonq16 = make_float4(__half2float(ph[0]), __half2float(ph[1]),
                                            __half2float(ph[2]), __half2float(ph[3]));
                } else {
                    pc_nonq16 = reinterpret_cast<const float4*>(p.param)[slot];
                }
            }

#pragma unroll 1
            for (uint c = 0; c < 4u; ++c) {
                const uint cell_lin = k * 4u + c;
                if (cell_lin >= kActiveCells)
                    break;
                float pci;
                if constexpr (VALUE_Q16) {
                    if (cell_lin < meta.n_value_cells) {
                        const uint32_t q_word = (c < 2u) ? q_pack0 : q_pack1;
                        const uint32_t q_in = (q_word >> ((c & 1u) ? 16u : 0u)) & 0xffffu;
                        pci = meta.v_min + static_cast<float>(q_in) * meta.v_dec;
                    } else {
                        pci = 0.0f;
                    }
                } else {
                    pci = (c == 0u) ? pc_nonq16.x : (c == 1u) ? pc_nonq16.y
                                                : (c == 2u)   ? pc_nonq16.z
                                                              : pc_nonq16.w;
                }

                const float gc_ch = (rgb == 0u) ? grad_color.x : (rgb == 1u) ? grad_color.y
                                                                             : grad_color.z;
                const float cg = pci * gc_ch;
                gd_x += bd.y * cg;
                gd_y += bd.z * cg;
                gd_z += bd.w * cg;

                if (apply_step) {
                    const uint32_t word = (c < 2u) ? m_words.x : m_words.y;
                    const uint32_t shift = (c & 1u) ? 16u : 0u;
                    const uint32_t pair = (word >> shift) & 0xffffu;
                    const float u_old = meta.u_min + static_cast<float>(pair & 0xffu) * meta.u_dec;
                    const float sqrt_g2 = s_sqrt_g2_lut[pair >> 8u];
                    float v_prev = sqrt_g2 * sqrt_g2;
                    float m_prev = (v_prev == 0.0f) ? 0.0f
                                                    : u_old * (sqrt_g2 + lfs::training::joint_adam::kEpsDevice);

                    const float reg_scale = meta.reg_scale;
                    const float bc2 = meta.bc2_sqrt_rcp;
                    const bool reg_on = (reg_scale > 0.0f) && ((flags & (1u << rgb)) != 0u);
                    if (lazy_decay1 < 1.0f) {
                        const float g_reg_miss = reg_on ? (2.0f * reg_scale * pci) : 0.0f;
                        if (v_prev > 0.0f || g_reg_miss != 0.0f) {
                            const float3 lazy_aux = s_lazy_aux[threadIdx.x];
                            float step_miss = 0.0f;
                            if (v_prev > 0.0f) {
                                const float denom_old = sqrt_g2 * bc2 + eps;
                                step_miss = row_step_size * (m_prev / denom_old) * lazy_aux.y;
                            }
                            float g_reg_eff = g_reg_miss;
                            if (g_reg_miss != 0.0f) {
                                const float v_mid =
                                    v_prev * lazy_aux.x + g_reg_miss * g_reg_miss * (1.0f - lazy_aux.x);
                                const float denom_reg = sqrtf(v_mid) * bc2 + eps;
                                const float raw_reg_step = row_step_size * (g_reg_miss / denom_reg) * lazy_aux.z;
                                float reg_step = raw_reg_step;
                                if (fabsf(raw_reg_step) >= fabsf(pci)) {
                                    reg_step = copysignf(fabsf(pci), raw_reg_step);
                                    g_reg_eff = 0.0f;
                                }
                                step_miss += reg_step;
                            }
                            pci -= step_miss;
                            m_prev = m_prev * lazy_decay1 + g_reg_eff * (1.0f - lazy_decay1);
                            const float d2 = lazy_aux.x * lazy_aux.x;
                            v_prev = v_prev * d2 + g_reg_miss * g_reg_miss * (1.0f - d2);
                        }
                    }

                    float gci = bd.x * gc_ch;
                    if (reg_scale > 0.0f) {
                        if (reg_on)
                            gci += 2.0f * reg_scale * pci;
                        if constexpr (REPORT_REG_LOSS)
                            local_sh_rest_loss += reg_scale * pci * pci;
                    }

                    const float m_new = beta1 * m_prev + (1.0f - beta1) * gci;
                    const float v_new = beta2 * v_prev + (1.0f - beta2) * gci * gci;
                    const float sqrt_v_new = sqrtf(fmaxf(v_new, 0.0f));
                    const float denom = sqrt_v_new * bc2 + eps;
                    pci -= row_step_size * m_new / denom;

                    const float prim_x = m_new / (sqrt_v_new + lfs::training::joint_adam::kEpsDevice);
                    const float prim_y = C::forward_sqrt_g2(sqrt_v_new);
                    const float u_norm = (prim_x - meta.u_min) * meta.u_enc;
                    const float s_norm = (prim_y - meta.s_min) * meta.s_enc;
                    if (__builtin_expect(u_norm < 0.0f || u_norm > C::kQMax || s_norm < 0.0f || s_norm > C::kQMax, 0)) {
                        atomicOr(&s_over_state[0], 1u);
                        const float ax = fabsf(prim_x);
                        if (isfinite(ax))
                            atomicMax(&s_over_state[1], __float_as_uint(ax));
                        if (prim_y > 0.0f && isfinite(prim_y))
                            atomicMax(&s_over_state[2], __float_as_uint(prim_y));
                    }

                    const uint32_t u_q = static_cast<uint32_t>(fminf(fmaxf(roundf(u_norm), 0.0f), C::kQMax));
                    const uint32_t s_q = static_cast<uint32_t>(fminf(fmaxf(roundf(s_norm), 0.0f), C::kQMax));
                    const uint32_t out_pair = u_q | (s_q << 8u);
                    if (c < 2u) {
                        m_words.x = (m_words.x & ~(0xffffu << shift)) | (out_pair << shift);
                    } else {
                        m_words.y = (m_words.y & ~(0xffffu << shift)) | (out_pair << shift);
                    }

                    if constexpr (VALUE_Q16) {
                        if (cell_lin < meta.n_value_cells) {
                            const float v_norm = (pci - meta.v_min) * meta.v_enc;
                            if (__builtin_expect(v_norm < 0.0f || v_norm > lfs::core::sh_value::kQMaxDevice, 0)) {
                                atomicOr(&s_over_state[0], 2u);
                                const float av = fabsf(pci);
                                if (isfinite(av))
                                    atomicMax(&s_over_state[3], __float_as_uint(av));
                            }
                            const float qf = fminf(fmaxf(roundf(v_norm), 0.0f), lfs::core::sh_value::kQMaxDevice);
                            const uint32_t q_out = static_cast<uint32_t>(static_cast<uint16_t>(qf));
                            const uint32_t q_old = (((c < 2u) ? q_pack0 : q_pack1) >> ((c & 1u) ? 16u : 0u)) & 0xffffu;
                            // Unchanged codes keep their sector clean.
                            if (q_out != q_old)
                                u16_ptr[c * 32u] = static_cast<uint16_t>(q_out);
                        }
                    } else {
                        if (c == 0u)
                            pc_nonq16.x = pci;
                        else if (c == 1u)
                            pc_nonq16.y = pci;
                        else if (c == 2u)
                            pc_nonq16.z = pci;
                        else
                            pc_nonq16.w = pci;
                    }
                }

                if (++rgb == 3u) {
                    rgb = 0u;
                    ++coeff_idx;
                    if (coeff_idx < kActiveCoeffs)
                        bd = shN_basis_val_and_deriv(coeff_idx, bx, by, bz);
                }
            }
            if (apply_step) {
                *packed_ptr = m_words;
                if constexpr (!VALUE_Q16) {
                    const uint slot = slot0 + k * config::sh_reorder_size;
                    if (value_f16) {
                        __half* const ph = reinterpret_cast<__half*>(p.param) + slot * 4u;
                        ph[0] = __float2half(pc_nonq16.x);
                        ph[1] = __float2half(pc_nonq16.y);
                        ph[2] = __float2half(pc_nonq16.z);
                        ph[3] = __float2half(pc_nonq16.w);
                    } else {
                        reinterpret_cast<float4*>(p.param)[slot] = pc_nonq16;
                    }
                }
            }
            packed_ptr += config::sh_reorder_size;
            if constexpr (VALUE_Q16) {
                u16_ptr += 128u;
            }
        }
        sh_rest_loss_out = local_sh_rest_loss;

        const float b_dot_gd = bx * gd_x + by * gd_y + bz * gd_z;
        return make_float3(gd_x - b_dot_gd * bx, gd_y - b_dot_gd * by, gd_z - b_dot_gd * bz) * dir_grad_scale;
    }

    // Closed-form flush of pending lazy coast + SH-rest regularization steps for one
    // primitive's 8-bit joint Adam SH-rest row.
    template <int ACTIVE_SH_BASES, bool VALUE_Q16>
    __device__ __forceinline__ void shN_flush_lazy_sparse(
        const FusedAdamParam& p,
        const uint primitive_idx,
        const uint bidx,
        const uint sh_layout_slots,
        const float sh_rest_reg_weight,
        const float eps,
        const LazyAdamFactors& lazy,
        const float* __restrict__ s_sqrt_g2_lut) {
        using C = lfs::training::joint_adam::DeviceCodec<8>;
        if (lazy.decay1 >= 1.0f || !p.enabled || sh_layout_slots == 0u ||
            p.joint_bits != 8 || p.joint_packed == nullptr || p.joint_bounds == nullptr)
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

        const float4 mm = *reinterpret_cast<const float4*>(p.joint_bounds + 4u * bidx);
        const float u_min = mm.x;
        const float s_min = mm.z;
        const float u_span = mm.y - mm.x;
        const float s_span = mm.w - mm.z;
        const float u_dec = u_span * C::kInvQMax;
        const float u_enc = C::kQMax / fmaxf(u_span, lfs::training::joint_adam::kEpsDevice);
        const float s_enc = C::kQMax / fmaxf(s_span, lfs::training::joint_adam::kEpsDevice);

        float v_min = 0.0f;
        float v_dec = 0.0f;
        float v_enc = 0.0f;
        uint n_value_cells = 0u;
        if constexpr (VALUE_Q16) {
            const float2 vmm = *reinterpret_cast<const float2*>(p.sh_value_bounds + 2u * bidx);
            const float v_span = vmm.y - vmm.x;
            v_min = vmm.x;
            v_dec = v_span * (1.0f / 65535.0f);
            v_enc = 65535.0f / fmaxf(v_span, lfs::core::sh_value::kEpsDevice);
            n_value_cells = static_cast<uint>(p.sh_value_n_cells);
        }
        const bool value_f16 = (!VALUE_Q16) && (p.sh_value_bits == 16);

        constexpr uint N_SLOTS = (ACTIVE_SH_BASES > 9) ? 12u : (ACTIVE_SH_BASES > 4) ? 6u
                                                                                     : 3u;
        constexpr uint kActiveCoeffs = static_cast<uint>(ACTIVE_SH_BASES - 1);
        constexpr uint kActiveCells = kActiveCoeffs * 3u;
        const float reg_scale = (sh_rest_reg_weight > 0.0f && p.n_primitives > 0)
                                    ? sh_rest_reg_weight /
                                          (static_cast<float>(p.n_primitives) * static_cast<float>(kActiveCells))
                                    : 0.0f;
        const uint slot0 = shAt(primitive_idx, 0u, sh_layout_slots);
        uint2* packed_ptr = reinterpret_cast<uint2*>(p.joint_packed) + slot0;
        uint16_t* u16_ptr =
            VALUE_Q16 ? (reinterpret_cast<uint16_t*>(p.param) +
                         lfs::core::sh_value::shAtU16(primitive_idx, 0u, n_value_cells))
                      : nullptr;

#pragma unroll 1
        for (uint k = 0; k < N_SLOTS; ++k) {
            if (k >= sh_layout_slots)
                break;
            uint2 m_words = *packed_ptr;
            const uint2 m_words_orig = m_words;
            const bool any_nz =
                ((m_words.x & 0xff00ff00u) != 0u) || ((m_words.y & 0xff00ff00u) != 0u);
            if (!any_nz && reg_scale <= 0.0f) {
                packed_ptr += config::sh_reorder_size;
                if constexpr (VALUE_Q16)
                    u16_ptr += 128u;
                continue;
            }
            float4 pc_nonq16 = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
            bool param_dirty = false;
            if constexpr (!VALUE_Q16) {
                const uint slot = slot0 + k * config::sh_reorder_size;
                if (value_f16) {
                    const __half* const ph = reinterpret_cast<const __half*>(p.param) + slot * 4u;
                    pc_nonq16 = make_float4(__half2float(ph[0]), __half2float(ph[1]),
                                            __half2float(ph[2]), __half2float(ph[3]));
                } else {
                    pc_nonq16 = reinterpret_cast<const float4*>(p.param)[slot];
                }
            }
#pragma unroll 1
            for (uint c = 0; c < 4u; ++c) {
                const uint cell_lin = k * 4u + c;
                if (cell_lin >= kActiveCells)
                    break;
                const uint32_t word = (c < 2u) ? m_words.x : m_words.y;
                const uint32_t shift = (c & 1u) ? 16u : 0u;
                const uint32_t pair = (word >> shift) & 0xffffu;
                const uint32_t s_q_in = pair >> 8u;
                if (s_q_in == 0u && reg_scale <= 0.0f)
                    continue;
                float pci;
                if constexpr (VALUE_Q16) {
                    if (cell_lin < n_value_cells)
                        pci = v_min + static_cast<float>(u16_ptr[c * 32u]) * v_dec;
                    else
                        pci = 0.0f;
                } else {
                    pci = (c == 0u) ? pc_nonq16.x : (c == 1u) ? pc_nonq16.y
                                                : (c == 2u)   ? pc_nonq16.z
                                                              : pc_nonq16.w;
                }
                const float g_reg_miss = (reg_scale > 0.0f) ? (2.0f * reg_scale * pci) : 0.0f;
                const float sqrt_g2 = s_sqrt_g2_lut[s_q_in];
                float v_prev = sqrt_g2 * sqrt_g2;
                if (v_prev == 0.0f && g_reg_miss == 0.0f)
                    continue;
                const float u_old = u_min + static_cast<float>(pair & 0xffu) * u_dec;
                float m_prev = (v_prev == 0.0f) ? 0.0f
                                                : u_old * (sqrt_g2 + lfs::training::joint_adam::kEpsDevice);
                float step_miss = 0.0f;
                if (v_prev > 0.0f) {
                    const float denom_old = sqrt_g2 * p.bias_correction2_sqrt_rcp + eps;
                    step_miss = row_step_size * (m_prev / denom_old) * lazy.coast_sum;
                }
                float g_reg_eff = g_reg_miss;
                if (g_reg_miss != 0.0f) {
                    const float v_mid =
                        v_prev * lazy.decay2_half + g_reg_miss * g_reg_miss * (1.0f - lazy.decay2_half);
                    const float denom_reg = sqrtf(v_mid) * p.bias_correction2_sqrt_rcp + eps;
                    const float raw_reg_step = row_step_size * (g_reg_miss / denom_reg) * lazy.reg_sum;
                    float reg_step = raw_reg_step;
                    if (fabsf(raw_reg_step) >= fabsf(pci)) {
                        reg_step = copysignf(fabsf(pci), raw_reg_step);
                        g_reg_eff = 0.0f;
                    }
                    step_miss += reg_step;
                }
                if (step_miss != 0.0f) {
                    pci -= step_miss;
                    if constexpr (VALUE_Q16) {
                        if (cell_lin < n_value_cells) {
                            const float qf = fminf(fmaxf(roundf((pci - v_min) * v_enc), 0.0f),
                                                   lfs::core::sh_value::kQMaxDevice);
                            u16_ptr[c * 32u] = static_cast<uint16_t>(qf);
                        }
                    } else {
                        param_dirty = true;
                        if (c == 0u)
                            pc_nonq16.x = pci;
                        else if (c == 1u)
                            pc_nonq16.y = pci;
                        else if (c == 2u)
                            pc_nonq16.z = pci;
                        else
                            pc_nonq16.w = pci;
                    }
                }
                m_prev = m_prev * lazy.decay1 + g_reg_eff * (1.0f - lazy.decay1);
                const float d2 = lazy.decay2_half * lazy.decay2_half;
                v_prev = v_prev * d2 + g_reg_miss * g_reg_miss * (1.0f - d2);
                const float sqrt_v = sqrtf(fmaxf(v_prev, 0.0f));
                const float prim_x = m_prev / (sqrt_v + lfs::training::joint_adam::kEpsDevice);
                const float prim_y = C::forward_sqrt_g2(sqrt_v);
                const uint32_t u_q = static_cast<uint32_t>(fminf(fmaxf(roundf((prim_x - u_min) * u_enc), 0.0f), C::kQMax));
                const uint32_t s_q = static_cast<uint32_t>(fminf(fmaxf(roundf((prim_y - s_min) * s_enc), 0.0f), C::kQMax));
                const uint32_t out_pair = u_q | (s_q << 8u);
                if (c < 2u)
                    m_words.x = (m_words.x & ~(0xffffu << shift)) | (out_pair << shift);
                else
                    m_words.y = (m_words.y & ~(0xffffu << shift)) | (out_pair << shift);
            }
            if (m_words.x != m_words_orig.x || m_words.y != m_words_orig.y)
                *packed_ptr = m_words;
            if constexpr (!VALUE_Q16) {
                if (param_dirty) {
                    const uint slot = slot0 + k * config::sh_reorder_size;
                    if (value_f16) {
                        __half* const ph = reinterpret_cast<__half*>(p.param) + slot * 4u;
                        ph[0] = __float2half(pc_nonq16.x);
                        ph[1] = __float2half(pc_nonq16.y);
                        ph[2] = __float2half(pc_nonq16.z);
                        ph[3] = __float2half(pc_nonq16.w);
                    } else {
                        reinterpret_cast<float4*>(p.param)[slot] = pc_nonq16;
                    }
                }
            }
            packed_ptr += config::sh_reorder_size;
            if constexpr (VALUE_Q16)
                u16_ptr += 128u;
        }
    }

    // SH backward: fills sh0_grads[3] and returns dL/dmean from view-dir.
    // Per-coefficient shN color grads are independent (basis_i * dL/dcolor) and
    // are recomputed in the fused-Adam stream — they are not materialized here.
    // dL/ddir keeps the original grouped accumulation (degree-1 init from
    // c2,c0,c1; degree-2 adds c3,c6,c7 / c3,c4,c7 / c4,c5,c6; degree-3 adds
    // c8,c9,c12,c13,c14 / c8,c9,c10,c13,c14 / c9,c10,c11,c12,c13).
    template <int ACTIVE_SH_BASES>
    __device__ inline float3 convert_sh_to_color_backward_grads_from_color(
        const float4* sh_coefficients_rest,
        const float3 grad_color,
        const float3& position,
        const float3& cam_position,
        const uint primitive_idx,
        const uint sh_layout_slots,
        float* __restrict__ sh0_grads_out,
        const float2* __restrict__ sh_value_bounds,
        const uint sh_value_n_cells,
        const uint sh_value_bits) {
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

    template <int ACTIVE_SH_BASES>
    __device__ inline float3 convert_sh_to_color_backward_grads(
        const float4* sh_coefficients_rest,
        float3* grad_color_helper,
        const float3& position,
        const float3& cam_position,
        const uint primitive_idx,
        const uint grad_color_idx,
        const uint sh_layout_slots,
        float* __restrict__ sh0_grads_out,
        const float2* __restrict__ sh_value_bounds = nullptr,
        const uint sh_value_n_cells = 0u,
        const uint sh_value_bits = 0u) {
        return convert_sh_to_color_backward_grads_from_color<ACTIVE_SH_BASES>(
            sh_coefficients_rest, grad_color_helper[grad_color_idx], position, cam_position,
            primitive_idx, sh_layout_slots, sh0_grads_out,
            sh_value_bounds, sh_value_n_cells, sh_value_bits);
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

    // Per-row (or per-column) ellipse walk used by both the count path and
    // create_instances emit. Float math is verbatim on both sides so the
    // kFastGSForwardStatusInstanceWriteMismatch check cannot fire from a
    // count/emit divergence. scan_along_x: walk tile rows, span in x.
    __device__ __forceinline__ uint2 ellipse_touched_tile_span(
        const float3& conic,
        const float radius_sq,
        const float2 mean2d_shifted,
        const bool scan_along_x,
        const uint scan_index,
        const uint cross_min,
        const uint cross_max) {
        if (scan_along_x) {
            const float y0 = static_cast<float>(scan_index * config::tile_height) - mean2d_shifted.y;
            const float y1 = y0 + static_cast<float>(config::tile_height);
            const float2 bound = ellipse_range_bound(conic, radius_sq, y0, y1);
            const uint lo = floor_tile_clamped(bound.x + mean2d_shifted.x, cross_min, cross_max, config::tile_width);
            const uint hi = ceil_tile_clamped(bound.y + mean2d_shifted.x, cross_min, cross_max, config::tile_width);
            return make_uint2(lo, hi);
        }
        const float3 conic_transposed = make_float3(conic.z, conic.y, conic.x);
        const float x0 = static_cast<float>(scan_index * config::tile_width) - mean2d_shifted.x;
        const float x1 = x0 + static_cast<float>(config::tile_width);
        const float2 bound = ellipse_range_bound(conic_transposed, radius_sq, x0, x1);
        const uint lo = floor_tile_clamped(bound.x + mean2d_shifted.y, cross_min, cross_max, config::tile_height);
        const uint hi = ceil_tile_clamped(bound.y + mean2d_shifted.y, cross_min, cross_max, config::tile_height);
        return make_uint2(lo, hi);
    }

    __device__ __forceinline__ uint tile_span_count(const uint2 span) {
        return span.y >= span.x ? span.y - span.x : 0u;
    }

    __device__ __forceinline__ uint warp_exclusive_scan_uint(uint val) {
        uint inclusive = val;
#pragma unroll
        for (int offset = 1; offset < 32; offset <<= 1) {
            const uint n = __shfl_up_sync(0xffffffffu, inclusive, offset);
            if ((threadIdx.x & 31) >= static_cast<unsigned>(offset))
                inclusive += n;
        }
        return inclusive - val;
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
        const bool scan_along_x = screen_bounds_height <= screen_bounds_width;
        const uint scan0 = scan_along_x ? screen_bounds.z : screen_bounds.x;
        const uint scan1 = scan_along_x ? screen_bounds.w : screen_bounds.y;
        const uint cross0 = scan_along_x ? screen_bounds.x : screen_bounds.z;
        const uint cross1 = scan_along_x ? screen_bounds.y : screen_bounds.w;

        uint n_touched_tiles = 0;
        for (uint scan_index = scan0; scan_index < scan1; scan_index++) {
            const uint2 span = ellipse_touched_tile_span(
                conic, radius_sq, mean2d_shifted, scan_along_x, scan_index, cross0, cross1);
            n_touched_tiles += tile_span_count(span);
        }
        return n_touched_tiles;
    }

} // namespace fast_lfs::rasterization::kernels

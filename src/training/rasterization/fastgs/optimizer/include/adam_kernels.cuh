/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include "lfs/core/warp_reduce.cuh"
#include "lfs/training/joint_adam_codec.cuh"

#include <cstdint>

namespace fast_lfs::optimizer::kernels::adam {

    // Non-fused joint Adam step for contiguous [n_prims, n_attr] params.
    // Grid = ceil(n_prims/256), block = 256 so blockIdx.x == bounds block index
    // (matches fused preprocess_backward / joint_adam::kBlockSizeDevice).
    // Frozen / crop-damped rows with zero lr skip the Adam update but still
    // re-encode under the new block bounds (same as fused adam_step_row_joint).
    template <int BITS>
    __global__ void adam_step_joint_contiguous_cu(
        float* param,
        uint8_t* packed,
        float* bounds, // float4 per 256-splat block
        const float* param_grad,
        const bool* frozen_mask,
        const int frozen_mask_size,
        const float frozen_lr_scale,
        const bool* crop_damping_mask,
        const int crop_damping_mask_size,
        const float cropbox_lr_scale,
        const int n_prims,
        const int n_attr,
        const float lr,
        const float beta1,
        const float beta2,
        const float eps,
        const float bias_correction1_rcp,
        const float bias_correction2_sqrt_rcp) {
        using C = lfs::training::joint_adam::DeviceCodec<BITS>;
        constexpr float kInf = 1e30f;
        constexpr int kBS = lfs::training::joint_adam::kBlockSizeDevice;

        const int prim = static_cast<int>(blockIdx.x) * kBS + static_cast<int>(threadIdx.x);
        const bool in_range = prim < n_prims && n_attr > 0;

        float row_lr = lr;
        bool apply_step = in_range;
        if (in_range && frozen_mask != nullptr && prim < frozen_mask_size && frozen_mask[prim]) {
            if (frozen_lr_scale == 0.0f)
                apply_step = false;
            else
                row_lr *= frozen_lr_scale;
        }
        if (in_range && crop_damping_mask != nullptr && prim < crop_damping_mask_size &&
            crop_damping_mask[prim]) {
            if (cropbox_lr_scale == 0.0f)
                apply_step = false;
            else
                row_lr *= cropbox_lr_scale;
        }

        const int bidx = static_cast<int>(blockIdx.x);
        const float4 old_mm = (bounds != nullptr)
                                  ? *reinterpret_cast<const float4*>(bounds + 4 * bidx)
                                  : make_float4(0.0f, 0.0f, 0.0f, 0.0f);

        // Means/scale/rot/opacity/sh0 attrs are small (≤4); sh0 is 3.
        constexpr int kMaxAttr = 16;
        float us_u[kMaxAttr];
        float us_s[kMaxAttr];
        float local_u_min = kInf, local_u_max = -kInf;
        float local_s_min = kInf, local_s_max = -kInf;
        const int row = in_range ? min(n_attr, kMaxAttr) : 0;
        const float step_size = row_lr * bias_correction1_rcp;
        const float beta1_comp = 1.0f - beta1;
        const float beta2_comp = 1.0f - beta2;

        if (in_range) {
            for (int i = 0; i < row; ++i) {
                const int64_t cell = static_cast<int64_t>(prim) * n_attr + i;
                const float2 mv = C::decode_g1g2(packed, cell, old_mm);
                float m = mv.x;
                float v = mv.y;
                if (apply_step) {
                    const float grad = param_grad[static_cast<int64_t>(prim) * n_attr + i];
                    m = beta1 * mv.x + beta1_comp * grad;
                    v = beta2 * mv.y + beta2_comp * grad * grad;
                    const float denom = sqrtf(v) * bias_correction2_sqrt_rcp + eps;
                    param[static_cast<int64_t>(prim) * n_attr + i] -= step_size * m / denom;
                }
                const float2 prim_us = C::g1g2_to_us(m, v);
                us_u[i] = prim_us.x;
                us_s[i] = prim_us.y;
                local_u_min = fminf(local_u_min, prim_us.x);
                local_u_max = fmaxf(local_u_max, prim_us.x);
                local_s_min = fminf(local_s_min, prim_us.y);
                local_s_max = fmaxf(local_s_max, prim_us.y);
            }
        }

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
            if (bounds != nullptr) {
                *reinterpret_cast<float4*>(bounds + 4 * bidx) = nb;
            }
        }
        __syncthreads();
        const float4 new_mm = sm_bounds;
        const float inv_u = 1.0f / fmaxf(new_mm.y - new_mm.x, lfs::training::joint_adam::kEpsDevice);
        const float inv_s = 1.0f / fmaxf(new_mm.w - new_mm.z, lfs::training::joint_adam::kEpsDevice);

        if (in_range) {
            for (int i = 0; i < row; ++i) {
                C::encode_us(packed, static_cast<int64_t>(prim) * n_attr + i,
                             us_u[i], us_s[i], new_mm.x, new_mm.z, inv_u, inv_s);
            }
        }
    }

    // expand block bounds to include (u,log_s)=(0,0) so encode_us(0,0) is
    // representable. When expansion is needed, re-encode every live cell in the
    // block under the new bounds (decode under old first) so non-target prims
    // stay consistent. Raw zero codes under bounds that exclude 0 decode to the
    // block minimum, not (m,v)=(0,0).
    __device__ __forceinline__ float4 joint_expand_bounds_include_zero(const float4 mm) {
        return make_float4(fminf(mm.x, 0.0f), fmaxf(mm.y, 0.0f),
                           fminf(mm.z, 0.0f), fmaxf(mm.w, 0.0f));
    }

    __device__ __forceinline__ bool joint_bounds_equal(const float4 a, const float4 b) {
        return a.x == b.x && a.y == b.y && a.z == b.z && a.w == b.w;
    }

    // Encode (u,log_s)=(0,0) under (possibly widened) block bounds → true (m,v)=(0,0).
    // Contiguous [N, n_attr] cells; packed is [N, n_attr * bpc] uint8.
    // bounds is mutable; n_prims is the live primitive count for block re-encode.
    template <int BITS>
    __global__ void joint_encode_zero_rows_cu(
        uint8_t* packed,
        float* bounds,
        const int64_t* indices,
        const int n_indices,
        const int n_attr,
        const int n_prims) {
        using C = lfs::training::joint_adam::DeviceCodec<BITS>;
        const int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx >= n_indices)
            return;
        const int64_t prim = indices[idx];
        if (prim < 0 || (n_prims > 0 && prim >= static_cast<int64_t>(n_prims)))
            return;
        const int bidx = static_cast<int>(prim / 256);
        float4* mm_ptr = reinterpret_cast<float4*>(bounds + 4 * bidx);
        const float4 old_mm = *mm_ptr;
        const float4 new_mm = joint_expand_bounds_include_zero(old_mm);
        if (!joint_bounds_equal(old_mm, new_mm)) {
            const int block_begin = bidx * 256;
            const int block_end = (n_prims > 0) ? min(block_begin + 256, n_prims)
                                                : (block_begin + 256);
            for (int p = block_begin; p < block_end; ++p) {
                for (int a = 0; a < n_attr; ++a) {
                    const int64_t cell = static_cast<int64_t>(p) * n_attr + a;
                    const float2 us = C::decode_us(packed, cell, old_mm);
                    C::encode_us(packed, cell, us.x, us.y, new_mm);
                }
            }
            *mm_ptr = new_mm;
        }
        for (int a = 0; a < n_attr; ++a) {
            const int64_t cell = prim * static_cast<int64_t>(n_attr) + a;
            C::encode_us(packed, cell, 0.0f, 0.0f, new_mm);
        }
    }

    // Swizzled shN: one thread per primitive; zero all float cells in layout slots.
    template <int BITS>
    __global__ void joint_encode_zero_shN_cu(
        uint8_t* packed,
        float* bounds,
        const int64_t* indices,
        const int n_indices,
        const int slots_per_primitive,
        const int n_prims) {
        using C = lfs::training::joint_adam::DeviceCodec<BITS>;
        const int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx >= n_indices)
            return;
        const int64_t prim = indices[idx];
        if (prim < 0 || (n_prims > 0 && prim >= static_cast<int64_t>(n_prims)))
            return;
        const int bidx = static_cast<int>(prim / 256);
        float4* mm_ptr = reinterpret_cast<float4*>(bounds + 4 * bidx);
        const float4 old_mm = *mm_ptr;
        const float4 new_mm = joint_expand_bounds_include_zero(old_mm);
        const uint32_t slots = static_cast<uint32_t>(slots_per_primitive);
        constexpr uint32_t R = 32u;

        auto sh_cell = [&](const uint32_t p, const uint32_t k, const int c) -> int64_t {
            const uint32_t slot = (p / R) * (slots * R) + k * R + (p % R);
            return static_cast<int64_t>(slot) * 4 + c;
        };

        if (!joint_bounds_equal(old_mm, new_mm) && slots > 0u) {
            const int block_begin = bidx * 256;
            const int block_end = (n_prims > 0) ? min(block_begin + 256, n_prims)
                                                : (block_begin + 256);
            for (int p = block_begin; p < block_end; ++p) {
                for (uint32_t k = 0; k < slots; ++k) {
                    for (int c = 0; c < 4; ++c) {
                        const int64_t cell = sh_cell(static_cast<uint32_t>(p), k, c);
                        const float2 us = C::decode_us(packed, cell, old_mm);
                        C::encode_us(packed, cell, us.x, us.y, new_mm);
                    }
                }
            }
            *mm_ptr = new_mm;
        }
        for (uint32_t k = 0; k < slots; ++k) {
            for (int c = 0; c < 4; ++c) {
                C::encode_us(packed, sh_cell(static_cast<uint32_t>(prim), k, c),
                             0.0f, 0.0f, new_mm);
            }
        }
    }

    // after raw gather of packed codes across blocks, re-encode each new
    // row under its destination block bounds (decode under source bounds).
    template <int BITS>
    __global__ void joint_transcode_gathered_rows_cu(
        uint8_t* packed,
        const float* bounds,
        const int64_t* indices,
        const int n_new,
        const int old_N,
        const int n_attr) {
        using C = lfs::training::joint_adam::DeviceCodec<BITS>;
        const int k = blockIdx.x * blockDim.x + threadIdx.x;
        if (k >= n_new)
            return;
        const int64_t src = indices[k];
        const int64_t dst = static_cast<int64_t>(old_N) + k;
        if (src < 0)
            return;
        const float4 src_mm = *reinterpret_cast<const float4*>(bounds + 4 * static_cast<int>(src / 256));
        const float4 dst_mm = *reinterpret_cast<const float4*>(bounds + 4 * static_cast<int>(dst / 256));
        // Same block + identical bounds: codes already correct after raw gather.
        if (src_mm.x == dst_mm.x && src_mm.y == dst_mm.y &&
            src_mm.z == dst_mm.z && src_mm.w == dst_mm.w &&
            (src / 256) == (dst / 256)) {
            return;
        }
        for (int a = 0; a < n_attr; ++a) {
            // Raw gather left src codes at dst; decode with src bounds, encode with dst.
            const int64_t dst_cell = dst * n_attr + a;
            const float2 us = C::decode_us(packed, dst_cell, src_mm);
            C::encode_us(packed, dst_cell, us.x, us.y, dst_mm);
        }
    }

} // namespace fast_lfs::optimizer::kernels::adam

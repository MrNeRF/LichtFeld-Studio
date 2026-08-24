/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#pragma once

#include <cmath>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

namespace lfs::core::nn::device {

    __device__ __forceinline__ float gelu_erf(const float x) {
        return 0.5f * x * (1.0f + erff(x * 0.7071067811865476f));
    }

    __device__ __forceinline__ float gelu_tanh(const float x) {
        const float inner = 0.7978845608028654f * (x + 0.044715f * x * x * x);
        return 0.5f * x * (1.0f + tanhf(inner));
    }

    __device__ __forceinline__ float silu(const float x) {
        return x / (1.0f + expf(-x));
    }

    __device__ __forceinline__ float apply_activation(const float x, const int act) {
        switch (act) {
        case 1:
            return fmaxf(x, 0.0f);
        case 2:
            return gelu_tanh(x);
        case 3:
            return gelu_erf(x);
        case 4:
            return silu(x);
        default:
            return x;
        }
    }

    __device__ __forceinline__ float ld_f32(const void* ptr, const bool is_half) {
        if (is_half) {
            return __half2float(*static_cast<const __half*>(ptr));
        }
        return *static_cast<const float*>(ptr);
    }

    __device__ __forceinline__ void st_f32(void* ptr, const float value, const bool is_half) {
        if (is_half) {
            *static_cast<__half*>(ptr) = __float2half_rn(value);
        } else {
            *static_cast<float*>(ptr) = value;
        }
    }

    __device__ __forceinline__ float ld_strided(const void* base, const long long index,
                                                const bool is_half) {
        if (is_half) {
            return __half2float(static_cast<const __half*>(base)[index]);
        }
        return static_cast<const float*>(base)[index];
    }

    __device__ __forceinline__ void st_strided(void* base, const long long index,
                                               const float value, const bool is_half) {
        if (is_half) {
            static_cast<__half*>(base)[index] = __float2half_rn(value);
        } else {
            static_cast<float*>(base)[index] = value;
        }
    }

    __device__ __forceinline__ float resize_coord(const int dst, const int src_len,
                                                  const int dst_len, const int coord) {
        if (dst_len == 1) {
            return coord == 2 ? 0.0f : (src_len > 0 ? 0.0f : 0.0f);
        }
        if (coord == 2) {
            if (src_len <= 1) {
                return 0.0f;
            }
            return static_cast<float>(dst) * static_cast<float>(src_len - 1) /
                   static_cast<float>(dst_len - 1);
        }
        const float scale = static_cast<float>(src_len) / static_cast<float>(dst_len);
        if (coord == 1) {
            return static_cast<float>(dst) * scale;
        }
        return (static_cast<float>(dst) + 0.5f) * scale - 0.5f;
    }

    __device__ __forceinline__ void cp_async16(void* smem_addr, const void* glob_addr) {
#if __CUDA_ARCH__ >= 800
        const unsigned smem = __cvta_generic_to_shared(smem_addr);
        asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" ::"r"(smem), "l"(glob_addr));
#else
        *reinterpret_cast<uint4*>(smem_addr) = *reinterpret_cast<const uint4*>(glob_addr);
#endif
    }

    __device__ __forceinline__ void cp_async_commit() {
#if __CUDA_ARCH__ >= 800
        asm volatile("cp.async.commit_group;\n");
#endif
    }

    __device__ __forceinline__ void cp_async_wait0() {
#if __CUDA_ARCH__ >= 800
        asm volatile("cp.async.wait_group 0;\n");
#endif
    }

    // 8-half (16-byte) xor swizzle. Consecutive rows at a fixed 16-byte
    // column hit distinct smem banks (32 banks x 4 bytes).
    __device__ __forceinline__ int xor_swizzle_col(const int row, const int col) {
        return col ^ ((row & 7) << 3);
    }

} // namespace lfs::core::nn::device

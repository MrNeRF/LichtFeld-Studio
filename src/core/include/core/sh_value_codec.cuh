/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

/**
 * Device-side 16-bit SH-rest value codec.
 * Mirrors host math in lfs/training/sh_value_codec.hpp.
 */

#include <cstdint>
#include <cuda_runtime.h>

namespace lfs::core::sh_value {

    constexpr float kQMaxDevice = 65535.0f;
    constexpr float kInvQMaxDevice = 1.0f / 65535.0f;
    constexpr float kEpsDevice = 1e-20f;

#ifdef __CUDACC__
    struct DeviceCodec16 {
        // Canonical host/export rounding, unaffected by CUDA fast-math FTZ.
        __device__ static __forceinline__ float decode_host_rounded(const uint16_t q, const float lo, const float hi) {
            float range, normalized, value;
            const float code = static_cast<float>(q);
            asm("sub.rn.f32 %0, %1, %2;" : "=f"(range) : "f"(hi), "f"(lo));
            asm("mul.rn.f32 %0, %1, %2;" : "=f"(normalized) : "f"(code), "f"(kInvQMaxDevice));
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(value) : "f"(range), "f"(normalized), "f"(lo));
            return value;
        }

        __device__ static inline float decode(const uint16_t q, const float lo, const float hi) {
            return lo + (hi - lo) * (static_cast<float>(q) * kInvQMaxDevice);
        }

        __device__ static inline uint16_t encode(const float v, const float lo, const float hi) {
            const float range = fmaxf(hi - lo, kEpsDevice);
            const float qf = fminf(fmaxf(roundf(kQMaxDevice * (v - lo) / range), 0.0f), kQMaxDevice);
            return static_cast<uint16_t>(qf);
        }
    };
#endif

    // Cell-linear swizzle index for q16 values (pad-dropped layout).
    // Layout [ceil(N/R), n_cells, R] of uint16, R=32.
    __device__ __host__ __forceinline__ unsigned int shAtU16(
        unsigned int primitive_idx,
        unsigned int cell,
        unsigned int n_cells) {
        constexpr unsigned int R = 32u;
        const unsigned int block = primitive_idx / R;
        const unsigned int lane = primitive_idx % R;
        return block * (n_cells * R) + cell * R + lane;
    }

} // namespace lfs::core::sh_value

/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

/**
 * Device-side 16-bit SH-rest value codec (Phase 2.1).
 * Mirrors host math in sh_value_codec.hpp.
 */

#include <cstdint>
#include <cuda_runtime.h>

namespace lfs::training::sh_value {

    constexpr int kBlockSizeDevice = 256;
    constexpr float kQMaxDevice = 65535.0f;
    constexpr float kInvQMaxDevice = 1.0f / 65535.0f;
    constexpr float kEpsDevice = 1e-20f;

    struct DeviceCodec16 {
        __device__ static inline float decode(const uint16_t q, const float lo, const float hi) {
            return lo + (hi - lo) * (static_cast<float>(q) * kInvQMaxDevice);
        }

        __device__ static inline uint16_t encode(const float v, const float lo, const float hi) {
            const float range = fmaxf(hi - lo, kEpsDevice);
            const float qf = fminf(fmaxf(roundf(kQMaxDevice * (v - lo) / range), 0.0f), kQMaxDevice);
            return static_cast<uint16_t>(qf);
        }

        /// Decode one float4 slot (4 cells) from packed uint16.
        __device__ static inline float4 decode_slot(const uint16_t* __restrict__ packed,
                                                    const int64_t float4_slot,
                                                    const float2 mm) {
            const int64_t base = float4_slot * 4;
            return make_float4(decode(packed[base + 0], mm.x, mm.y),
                               decode(packed[base + 1], mm.x, mm.y),
                               decode(packed[base + 2], mm.x, mm.y),
                               decode(packed[base + 3], mm.x, mm.y));
        }

        __device__ static inline void encode_slot(uint16_t* __restrict__ packed,
                                                  const int64_t float4_slot,
                                                  const float4 v,
                                                  const float2 mm) {
            const int64_t base = float4_slot * 4;
            packed[base + 0] = encode(v.x, mm.x, mm.y);
            packed[base + 1] = encode(v.y, mm.x, mm.y);
            packed[base + 2] = encode(v.z, mm.x, mm.y);
            packed[base + 3] = encode(v.w, mm.x, mm.y);
        }
    };

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

} // namespace lfs::training::sh_value

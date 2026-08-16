/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

/**
 * Device-side joint (u, log_s) Adam codec.
 * Mirrors host math in joint_adam_codec.hpp.
 */

#include <cstdint>
#include <cuda_runtime.h>

namespace lfs::training::joint_adam {

    constexpr int kBlockSizeDevice = 256;
    constexpr float kEpsDevice = 1e-15f;

    template <int BITS>
    struct DeviceCodec {
        static_assert(BITS == 8 || BITS == 16, "joint Adam codec supports 8 or 16 bits");
        static constexpr int kBits = BITS;
        static constexpr float kQMax = static_cast<float>((1 << BITS) - 1);
        static constexpr float kInvQMax = 1.0f / kQMax;
        static constexpr int kBytesPerCell = (BITS == 16) ? 4 : 2;

        // guarded fast transcode. __logf/__expf for the bulk of the
        // range; log1pf/expm1f near zero so 0↔0 fixed point stays exact.
        // Thresholds: x>0.125 → __logf(1+x); log_s>0.118 → __expf-1.
        __device__ static inline float forward_sqrt_g2(const float sqrt_g2) {
            const float x = fmaxf(sqrt_g2, 0.0f) * (1.0f / kEpsDevice);
            return (x > 0.125f) ? __logf(1.0f + x) : log1pf(x);
        }
        __device__ static inline float inverse_sqrt_g2(const float log_s) {
            const float e1 = (log_s > 0.118f) ? (__expf(log_s) - 1.0f) : expm1f(log_s);
            return kEpsDevice * e1;
        }

        __device__ static inline float2 g1g2_to_us(const float g1, const float g2) {
            const float sqrt_g2 = sqrtf(fmaxf(g2, 0.0f));
            return make_float2(g1 / (sqrt_g2 + kEpsDevice), forward_sqrt_g2(sqrt_g2));
        }

        __device__ static inline float2 decode_us(const uint8_t* __restrict__ packed,
                                                  const int64_t idx,
                                                  const float4 mm) {
            float u_q, s_q;
            if constexpr (BITS == 16) {
                const auto* p = reinterpret_cast<const uint16_t*>(packed);
                u_q = static_cast<float>(p[idx * 2 + 0]);
                s_q = static_cast<float>(p[idx * 2 + 1]);
            } else {
                u_q = static_cast<float>(packed[idx * 2 + 0]);
                s_q = static_cast<float>(packed[idx * 2 + 1]);
            }
            return make_float2(mm.x + (mm.y - mm.x) * (u_q * kInvQMax),
                               mm.z + (mm.w - mm.z) * (s_q * kInvQMax));
        }

        __device__ static inline float2 decode_g1g2(const uint8_t* __restrict__ packed,
                                                    const int64_t idx,
                                                    const float4 mm) {
            const float2 prim = decode_us(packed, idx, mm);
            const float sqrt_g2 = inverse_sqrt_g2(prim.y);
            return make_float2(prim.x * (sqrt_g2 + kEpsDevice), sqrt_g2 * sqrt_g2);
        }

        // Block-uniform ranges: caller hoists inv_u_range / inv_s_range once
        // per encode pass (2 fdiv → FMA per cell).
        __device__ static inline void encode_us(uint8_t* __restrict__ packed,
                                                const int64_t idx,
                                                const float u_val,
                                                const float log_s_val,
                                                const float umin,
                                                const float smin,
                                                const float inv_u_range,
                                                const float inv_s_range) {
            const float u_qf = fminf(fmaxf(roundf(kQMax * (u_val - umin) * inv_u_range), 0.0f), kQMax);
            const float s_qf = fminf(fmaxf(roundf(kQMax * (log_s_val - smin) * inv_s_range), 0.0f), kQMax);
            if constexpr (BITS == 16) {
                auto* p = reinterpret_cast<uint16_t*>(packed);
                p[idx * 2 + 0] = static_cast<uint16_t>(u_qf);
                p[idx * 2 + 1] = static_cast<uint16_t>(s_qf);
            } else {
                packed[idx * 2 + 0] = static_cast<uint8_t>(u_qf);
                packed[idx * 2 + 1] = static_cast<uint8_t>(s_qf);
            }
        }

        __device__ static inline void encode_us(uint8_t* __restrict__ packed,
                                                const int64_t idx,
                                                const float u_val,
                                                const float log_s_val,
                                                const float4 mm) {
            const float inv_u = 1.0f / fmaxf(mm.y - mm.x, kEpsDevice);
            const float inv_s = 1.0f / fmaxf(mm.w - mm.z, kEpsDevice);
            encode_us(packed, idx, u_val, log_s_val, mm.x, mm.z, inv_u, inv_s);
        }
    };

} // namespace lfs::training::joint_adam

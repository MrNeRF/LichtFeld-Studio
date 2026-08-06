/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

/**
 * Device-side joint (u, log_s) Adam codec (Phase 2.2).
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

        __device__ static inline float forward_sqrt_g2(const float sqrt_g2) {
            return log1pf(fmaxf(sqrt_g2, 0.0f) * (1.0f / kEpsDevice));
        }
        __device__ static inline float inverse_sqrt_g2(const float log_s) {
            return kEpsDevice * expm1f(log_s);
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

        __device__ static inline void encode_us(uint8_t* __restrict__ packed,
                                                const int64_t idx,
                                                const float u_val,
                                                const float log_s_val,
                                                const float4 mm) {
            const float u_range = fmaxf(mm.y - mm.x, kEpsDevice);
            const float s_range = fmaxf(mm.w - mm.z, kEpsDevice);
            const float u_qf = fminf(fmaxf(roundf(kQMax * (u_val - mm.x) / u_range), 0.0f), kQMax);
            const float s_qf = fminf(fmaxf(roundf(kQMax * (log_s_val - mm.z) / s_range), 0.0f), kQMax);
            if constexpr (BITS == 16) {
                auto* p = reinterpret_cast<uint16_t*>(packed);
                p[idx * 2 + 0] = static_cast<uint16_t>(u_qf);
                p[idx * 2 + 1] = static_cast<uint16_t>(s_qf);
            } else {
                packed[idx * 2 + 0] = static_cast<uint8_t>(u_qf);
                packed[idx * 2 + 1] = static_cast<uint8_t>(s_qf);
            }
        }

        __device__ static inline void encode_g1g2(uint8_t* __restrict__ packed,
                                                  const int64_t idx,
                                                  const float g1,
                                                  const float g2,
                                                  const float4 mm) {
            const float2 prim = g1g2_to_us(g1, g2);
            encode_us(packed, idx, prim.x, prim.y, mm);
        }
    };

} // namespace lfs::training::joint_adam

/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

/**
 * @file joint_adam_codec.hpp
 * @brief Joint (u, log_s) Adam moment codec.
 *
 * Per cell stores two linearly-quantized primitives:
 *   u     = m / (sqrt(v) + eps)
 *   log_s = log1p(sqrt(v) / eps)
 * Bounds are float4 (u_min, u_max, log_s_min, log_s_max) per 256-splat block.
 * Endpoint-exact; all-zero packed+bounds decode to (m,v)=(0,0).
 *
 * Bit depths: 16-bit non-SH (4 B/cell), 8-bit SH (2 B/cell).
 * Joint is the sole Adam moment codec.
 */

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>

namespace lfs::training::joint_adam {

    inline constexpr int kBlockSize = 256;
    inline constexpr float kEps = 1e-15f;

    [[nodiscard]] inline std::size_t n_bounds_for_prims(std::size_t n_prims) {
        if (n_prims == 0)
            return 0;
        return (n_prims + static_cast<std::size_t>(kBlockSize) - 1) /
               static_cast<std::size_t>(kBlockSize);
    }

    [[nodiscard]] inline int bytes_per_cell(int bits) {
        return bits == 16 ? 4 : (bits == 8 ? 2 : 0);
    }

    [[nodiscard]] inline std::size_t packed_bytes(std::size_t n_cells, int bits) {
        return n_cells * static_cast<std::size_t>(bytes_per_cell(bits));
    }

    /// Host-side codec (mirrors device math in joint_adam_codec.cuh).
    template <int BITS>
    struct Codec {
        static_assert(BITS == 8 || BITS == 16, "joint Adam codec supports 8 or 16 bits");
        static constexpr int kBits = BITS;
        static constexpr float kQMax = static_cast<float>((1 << BITS) - 1);
        static constexpr float kInvQMax = 1.0f / kQMax;
        static constexpr int kBytesPerCell = (BITS == 16) ? 4 : 2;

        // Host mirror of device fast-path thresholds (log1p/expm1 near 0;
        // std::log/exp for the bulk). Keeps 0↔0 fixed point exact.
        static float forward_sqrt_g2(float sqrt_g2) {
            const float x = std::max(sqrt_g2, 0.0f) * (1.0f / kEps);
            return (x > 0.125f) ? std::log(1.0f + x) : std::log1p(x);
        }
        static float inverse_sqrt_g2(float log_s) {
            const float e1 = (log_s > 0.118f) ? (std::exp(log_s) - 1.0f) : std::expm1(log_s);
            return kEps * e1;
        }

        /// (m, v) -> (u, log_s)
        static void g1g2_to_us(float g1, float g2, float& u, float& log_s) {
            const float sqrt_g2 = std::sqrt(std::max(g2, 0.0f));
            u = g1 / (sqrt_g2 + kEps);
            log_s = forward_sqrt_g2(sqrt_g2);
        }

        static void us_to_g1g2(float u, float log_s, float& g1, float& g2) {
            const float sqrt_g2 = inverse_sqrt_g2(log_s);
            g1 = u * (sqrt_g2 + kEps);
            g2 = sqrt_g2 * sqrt_g2;
        }

        /// Decode (u, log_s) from packed cell `idx` with bounds mm = (umin,umax,smin,smax).
        static void decode_us(const std::uint8_t* packed, std::size_t idx,
                              float umin, float umax, float smin, float smax,
                              float& u, float& log_s) {
            float u_q = 0.0f, s_q = 0.0f;
            if constexpr (BITS == 16) {
                const auto* p = reinterpret_cast<const std::uint16_t*>(packed);
                u_q = static_cast<float>(p[idx * 2 + 0]);
                s_q = static_cast<float>(p[idx * 2 + 1]);
            } else {
                u_q = static_cast<float>(packed[idx * 2 + 0]);
                s_q = static_cast<float>(packed[idx * 2 + 1]);
            }
            u = umin + (umax - umin) * (u_q * kInvQMax);
            log_s = smin + (smax - smin) * (s_q * kInvQMax);
        }

        static void decode_g1g2(const std::uint8_t* packed, std::size_t idx,
                                float umin, float umax, float smin, float smax,
                                float& g1, float& g2) {
            float u = 0.0f, log_s = 0.0f;
            decode_us(packed, idx, umin, umax, smin, smax, u, log_s);
            us_to_g1g2(u, log_s, g1, g2);
        }

        /// Host encode mirrors the device fast path: inverse-range multiplication, not per-cell division.
        static void encode_us(std::uint8_t* packed, std::size_t idx,
                              float u_val, float log_s_val,
                              float umin, float umax, float smin, float smax) {
            const float inv_u = 1.0f / std::max(umax - umin, kEps);
            const float inv_s = 1.0f / std::max(smax - smin, kEps);
            const float u_qf = std::min(std::max(std::round(kQMax * (u_val - umin) * inv_u), 0.0f), kQMax);
            const float s_qf = std::min(std::max(std::round(kQMax * (log_s_val - smin) * inv_s), 0.0f), kQMax);
            if constexpr (BITS == 16) {
                auto* p = reinterpret_cast<std::uint16_t*>(packed);
                p[idx * 2 + 0] = static_cast<std::uint16_t>(u_qf);
                p[idx * 2 + 1] = static_cast<std::uint16_t>(s_qf);
            } else {
                packed[idx * 2 + 0] = static_cast<std::uint8_t>(u_qf);
                packed[idx * 2 + 1] = static_cast<std::uint8_t>(s_qf);
            }
        }

        static void encode_g1g2(std::uint8_t* packed, std::size_t idx,
                                float g1, float g2,
                                float umin, float umax, float smin, float smax) {
            float u = 0.0f, log_s = 0.0f;
            g1g2_to_us(g1, g2, u, log_s);
            encode_us(packed, idx, u, log_s, umin, umax, smin, smax);
        }

        /// Reduce bounds over `n` (g1,g2) pairs; writes float4-style out[4].
        static void reduce_bounds(const float* g1, const float* g2, std::size_t n,
                                  float out[4]) {
            if (n == 0) {
                out[0] = out[1] = out[2] = out[3] = 0.0f;
                return;
            }
            float umin = 1e30f, umax = -1e30f, smin = 1e30f, smax = -1e30f;
            for (std::size_t i = 0; i < n; ++i) {
                float u = 0.0f, log_s = 0.0f;
                g1g2_to_us(g1[i], g2[i], u, log_s);
                umin = std::min(umin, u);
                umax = std::max(umax, u);
                smin = std::min(smin, log_s);
                smax = std::max(smax, log_s);
            }
            out[0] = umin;
            out[1] = umax;
            out[2] = smin;
            out[3] = smax;
        }
    };

    using Codec16 = Codec<16>;
    using Codec8 = Codec<8>;

    /// Hand-computed optimizer B/splat for SH3 joint codec (large-N limit, no pad waste):
    /// non-SH 14 cells × 4 B = 56; SH 48 swizzled cells × 2 B = 96; bounds ≪ 1 → ~152.
    /// (Unpadded K×3=45 SH cells would be ~146.)
    inline constexpr std::size_t kOptimBpsJointSh3 = 152;
    /// Legacy uint8 + per-primitive scales.
    inline constexpr std::size_t kOptimBpsLegacySh3 = 172;

} // namespace lfs::training::joint_adam

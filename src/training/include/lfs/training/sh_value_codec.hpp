/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

/**
 * @file sh_value_codec.hpp
 * @brief Spirulae-style 16-bit linear SH-rest value codec (Phase 2.1).
 *
 * Per cell: endpoint-exact linear quant to uint16 over float2 (min,max) bounds.
 * Bounds layout: one float2 per 256-splat block (FPBO / per-splat-block).
 * All SH cells of a splat share that splat-block's bound.
 *
 * Runtime: default OFF until densify/export fully wired (ISS-2.1).
 *   LFS_SH_VALUE_QUANT=1  → enable 16-bit SH value quant
 *   LFS_SH_VALUE_FP32=1   → force fp32 (overrides quant)
 */

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <optional>

namespace lfs::training::sh_value {

    inline constexpr int kBlockSize = 256;
    inline constexpr int kBits = 16;
    inline constexpr float kQMax = 65535.0f;
    inline constexpr float kInvQMax = 1.0f / 65535.0f;
    inline constexpr float kEps = 1e-20f;

    [[nodiscard]] bool sh_value_quant_enabled();
    void set_sh_value_quant_enabled_for_testing(std::optional<bool> enabled);

    [[nodiscard]] inline std::size_t n_bounds_for_prims(std::size_t n_prims) {
        if (n_prims == 0)
            return 0;
        return (n_prims + static_cast<std::size_t>(kBlockSize) - 1) /
               static_cast<std::size_t>(kBlockSize);
    }

    /// Host codec (mirrors device math in sh_value_codec.cuh).
    struct Codec16 {
        static constexpr float kQMaxV = kQMax;
        static constexpr float kInvQMaxV = kInvQMax;

        static float decode(std::uint16_t q, float lo, float hi) {
            return lo + (hi - lo) * (static_cast<float>(q) * kInvQMaxV);
        }

        static std::uint16_t encode(float v, float lo, float hi) {
            const float range = std::max(hi - lo, kEps);
            const float qf = std::min(std::max(std::round(kQMaxV * (v - lo) / range), 0.0f), kQMaxV);
            return static_cast<std::uint16_t>(qf);
        }

        /// Endpoint-exact: encoding lo/hi recovers them after decode.
        static void reduce_bounds(const float* vals, std::size_t n, float out[2]) {
            if (n == 0) {
                out[0] = out[1] = 0.0f;
                return;
            }
            float lo = vals[0], hi = vals[0];
            for (std::size_t i = 1; i < n; ++i) {
                lo = std::min(lo, vals[i]);
                hi = std::max(hi, vals[i]);
            }
            out[0] = lo;
            out[1] = hi;
        }
    };

    /// Hand-computed param B/splat for SH3 with q16 swizzled pad:
    /// non-SH 56 + shN 48 cells × 2 B = 96 + bounds ≪1 → params 152 (vs fp32 248).
    /// Optim unchanged by 2.1 (joint 152). Densify 8. Total ≈ 312 B/splat large-N.
    /// (Spirulae 45 cells → shN 90 → params 146 → total ~300.)
    inline constexpr std::size_t kParamsBpsFp32Sh3 = 248;
    inline constexpr std::size_t kParamsBpsQ16Sh3 = 152; // 56 + 96
    inline constexpr std::size_t kShNBpsFp32Sh3 = 192;
    inline constexpr std::size_t kShNBpsQ16Sh3 = 96;

} // namespace lfs::training::sh_value

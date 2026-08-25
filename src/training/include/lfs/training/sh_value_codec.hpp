/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

/**
 * @file sh_value_codec.hpp
 * @brief 16-bit linear SH-rest value codec.
 *
 * Per cell: endpoint-exact linear quant to uint16 over float2 (min,max) bounds.
 * Bounds layout: one float2 per 256-splat block (FPBO / per-splat-block).
 * All SH cells of a splat share that splat-block's bound.
 *
 * Runtime: SH value quantization is permanently ON in production.
 * Tests may force off via set_sh_value_quant_enabled_for_testing.
 *
 * Storage (pad-dropped): n_cells = coeffs_rest * 3 per primitive
 * (45 for SH3 — no float4 tail pad). Swizzle [ceil(N/R), n_cells, R] of uint16.
 * Bounds: float2 per 256-splat block (FPBO). Params ≈ 56+90 = 146 B/splat SH3.
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

    /// Host codec (mirrors device math in core/sh_value_codec.cuh).
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

    /// Hand-computed param B/splat for SH3 with q16 pad-dropped (2.4):
    /// non-SH 56 + shN 45 cells × 2 B = 90 + bounds ≪1 → params 146 (vs fp32 248).
    /// Optim (joint, still 48-cell moments) ≈ 152. Densify 8. Total ≈ 306 large-N.
    inline constexpr std::size_t kParamsBpsFp32Sh3 = 248;
    inline constexpr std::size_t kParamsBpsQ16Sh3 = 146; // 56 + 90
    inline constexpr std::size_t kShNBpsFp32Sh3 = 192;
    inline constexpr std::size_t kShNBpsQ16Sh3 = 90; // 45 cells × 2 (pad-dropped)

    /// Number of uint16 value cells per primitive (no float4 pad).
    [[nodiscard]] inline constexpr std::uint32_t n_value_cells_per_prim(
        std::uint32_t coeffs_rest) noexcept {
        return coeffs_rest * 3u; // 0 / 9 / 24 / 45 for deg 0/1/2/3
    }

    /// Total uint16 cells in the cell-linear swizzled buffer for n primitives.
    /// Layout: [ceil(N/R), n_cells, R] with R = 32 (same reorder as float4 path).
    [[nodiscard]] inline constexpr std::size_t sh_value_u16_count(
        std::size_t n_prims,
        std::uint32_t coeffs_rest) noexcept {
        if (n_prims == 0 || coeffs_rest == 0)
            return 0;
        constexpr std::size_t R = 32;
        const std::size_t blocks = (n_prims + R - 1) / R;
        return blocks * R * static_cast<std::size_t>(n_value_cells_per_prim(coeffs_rest));
    }

} // namespace lfs::training::sh_value

/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

/**
 * Core-side SH value quant size helpers + runtime flag.
 * Codec math lives in lfs/training/sh_value_codec.hpp; this header is for SplatData
 * allocation without a core→training dependency.
 */

#include "core/cuda/sh_layout.cuh"

#include <atomic>
#include <cstdint>
#include <optional>

namespace lfs::core::sh_value_quant {

    inline constexpr int kBlockSize = 256;

    [[nodiscard]] inline std::size_t n_bounds_for_prims(std::size_t n_prims) {
        if (n_prims == 0)
            return 0;
        return (n_prims + static_cast<std::size_t>(kBlockSize) - 1) /
               static_cast<std::size_t>(kBlockSize);
    }

    /// Pad-dropped cells per primitive (no float4 tail pad).
    [[nodiscard]] inline constexpr std::uint32_t n_value_cells_per_prim(
        std::uint32_t coeffs_rest) noexcept {
        return coeffs_rest * 3u;
    }

    /// Total uint16 cells: [ceil(N/R), n_cells, R] with R = kShReorderSize.
    [[nodiscard]] inline constexpr std::size_t sh_value_u16_count(
        std::size_t n_prims,
        std::uint32_t coeffs_rest) noexcept {
        if (n_prims == 0 || coeffs_rest == 0)
            return 0;
        return sh_swizzled_padded_n(n_prims) *
               static_cast<std::size_t>(n_value_cells_per_prim(coeffs_rest));
    }

    // Production: SH value quantization permanently ON.
    // Tests may force off via set_enabled_for_testing (footprint tables, etc.).
    inline std::atomic<int>& override_flag() {
        static std::atomic<int> g{-1};
        return g;
    }

    inline void set_enabled_for_testing(std::optional<bool> enabled) {
        if (!enabled.has_value()) {
            override_flag().store(-1, std::memory_order_relaxed);
            return;
        }
        override_flag().store(*enabled ? 1 : 0, std::memory_order_relaxed);
    }

    [[nodiscard]] inline bool enabled() {
        const int o = override_flag().load(std::memory_order_relaxed);
        if (o >= 0)
            return o != 0;
        return true; // production default: always ON
    }

} // namespace lfs::core::sh_value_quant

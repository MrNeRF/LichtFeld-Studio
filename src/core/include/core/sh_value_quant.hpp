/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

/**
 * Core-side SH value quant size helpers + runtime flag (Phase 2.1 / WO-G3).
 * Codec math lives in lfs/training/sh_value_codec.hpp; this header is for SplatData
 * allocation without a core→training dependency.
 */

#include "core/cuda/sh_layout.cuh"

#include <atomic>
#include <cstdint>
#include <cstdlib>
#include <cstring>
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

    // Runtime flag (mirrors training::sh_value::sh_value_quant_enabled).
    // Default OFF until densify re-encode after N-growth is gate-green (ISS-2.1).
    // Opt-in: LFS_SH_VALUE_QUANT=1. Force off: LFS_SH_VALUE_FP32=1 or QUANT=0.
    [[nodiscard]] inline bool env_quant_enabled() {
        const char* force_fp32 = std::getenv("LFS_SH_VALUE_FP32");
        if (force_fp32 != nullptr && force_fp32[0] != '\0' &&
            !(force_fp32[0] == '0' && force_fp32[1] == '\0') &&
            std::strcmp(force_fp32, "false") != 0 &&
            std::strcmp(force_fp32, "off") != 0) {
            return false;
        }
        const char* opt = std::getenv("LFS_SH_VALUE_QUANT");
        if (opt == nullptr || opt[0] == '\0')
            return false; // default OFF — densify re-encode after growth still open
        if (opt[0] == '0' && opt[1] == '\0')
            return false;
        if (std::strcmp(opt, "false") == 0 || std::strcmp(opt, "off") == 0)
            return false;
        return true;
    }

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
        return env_quant_enabled();
    }

} // namespace lfs::core::sh_value_quant

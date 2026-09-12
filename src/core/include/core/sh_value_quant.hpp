/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

/**
 * Core-side SH value quant size helpers + runtime flag.
 * Codec math lives in lfs/training/sh_value_codec.hpp; this header is for SplatData
 * allocation without a core→training dependency.
 */

#include "core/cuda/sh_layout.cuh"
#include "core/export.hpp"
#include "core/tensor_fwd.hpp"

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
    // Implemented in lfs_core so the flag is process-wide across DSOs.
    LFS_CORE_API void set_enabled_for_testing(std::optional<bool> enabled);
    [[nodiscard]] LFS_CORE_API bool enabled();

    /// Tensor-program encode of float4-swizzled SH-rest into pad-dropped u16
    /// codes (Float16 container) plus float2 bounds per 256-splat block.
    LFS_CORE_API void encode_shN_float4_to_u16_tensor(
        const Tensor& src_float4_swizzled,
        std::size_t n_primitives,
        std::uint32_t slots_per_prim,
        std::uint32_t n_cells_per_prim,
        Tensor& codes_out,
        Tensor& bounds_out);

} // namespace lfs::core::sh_value_quant

/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

/**
 * @file sh_value_storage.hpp
 * @brief Convert SplatData.shN between fp32 float4-swizzled and q16 pad-dropped storage.
 *
 * Call apply_shN_value_quant once after model load / before training when the flag is ON.
 * Densify entry points should call ensure_shN_fp32_for_mutation / commit_shN_after_mutation
 * around float-native densify ops.
 *
 * Exportable (GUI) path: commit re-encodes via a bounded staging chunk into the single
 * resident q16 region (ISS-027 follow-up / WO-SH-DOUBLEBUFFER). Peak during refine is
 * q16 + staging chunk — not full fp32 and not 2× q16. Publish bumps a generation so
 * viewers/training consumers never read a chunk mid-write.
 */

#include "core/splat_data.hpp"

#include <cstddef>
#include <cstdint>
#include <optional>

namespace lfs::training::sh_value {

    /// If quant is enabled and shN is still fp32, convert to Float16 u16 + bounds.
    /// No-op when already quantized or flag off. Returns true if converted.
    bool apply_shN_value_quant(core::SplatData& splat);

    /// If quant is on and shN is u16, expand to a float4-swizzled temp in-place for densify.
    /// Pair with commit_shN_after_mutation.
    bool ensure_shN_fp32_for_mutation(core::SplatData& splat);

    /// After densify mutated float shN, re-encode to u16 + bounds (if quant on).
    /// Exportable-backed models use chunked staging publish into the live q16 region.
    bool commit_shN_after_mutation(core::SplatData& splat);

    /// Drop the refine-window staging buffer (call at stop_refine / teardown).
    void release_shN_publish_staging();

    /// Generation bumped after each successful chunked publish into the live q16 region.
    [[nodiscard]] std::uint64_t shN_publish_generation() noexcept;

    /// Staging budget used for chunked exportable re-encode (bytes). Default 96 MiB.
    [[nodiscard]] std::size_t shN_publish_staging_budget_bytes() noexcept;

    /// Test-only: force a smaller/larger staging budget (nullopt restores default).
    void set_shN_publish_staging_budget_for_testing(std::optional<std::size_t> bytes);

    /// Bind device constants for FastGS forward decode (call each step when quant on).
    void bind_shN_quant_for_raster(const core::SplatData& splat);

    /// Clear device constants (fp32 path / teardown).
    void clear_shN_quant_for_raster();

} // namespace lfs::training::sh_value

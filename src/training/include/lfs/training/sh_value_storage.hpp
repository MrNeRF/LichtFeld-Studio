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
 */

#include "core/splat_data.hpp"

namespace lfs::training::sh_value {

    /// If quant is enabled and shN is still fp32, convert to Float16 u16 + bounds.
    /// No-op when already quantized or flag off. Returns true if converted.
    bool apply_shN_value_quant(core::SplatData& splat);

    /// If quant is on and shN is u16, expand to a float4-swizzled temp in-place for densify.
    /// Pair with commit_shN_after_mutation.
    bool ensure_shN_fp32_for_mutation(core::SplatData& splat);

    /// After densify mutated float shN, re-encode to u16 + bounds (if quant on).
    bool commit_shN_after_mutation(core::SplatData& splat);

    /// Bind device constants for FastGS forward decode (call each step when quant on).
    void bind_shN_quant_for_raster(const core::SplatData& splat);

    /// Clear device constants (fp32 path / teardown).
    void clear_shN_quant_for_raster();

} // namespace lfs::training::sh_value

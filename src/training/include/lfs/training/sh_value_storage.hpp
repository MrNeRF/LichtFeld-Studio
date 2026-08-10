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
 * The exportable block holds pad-dropped q16
 * SH at all times. densify expands a float workspace under trainer render_mutex
 * exclusive, then commit re-encodes into the same live q16 region before the lock
 * drops. Passive viewport preview acquires the step-boundary lock at a bounded
 * cadence so it never projects concurrent with densify/re-encode.
 */

#include "core/splat_data.hpp"

#include <string_view>

namespace lfs::training::sh_value {

    /// If quant is enabled and shN is still fp32, convert to Float16 u16 + bounds.
    /// No-op when already quantized or flag off. Returns true if converted.
    bool apply_shN_value_quant(core::SplatData& splat);

    /// If quant is on and shN is u16, expand to a float4-swizzled temp in-place for densify.
    /// Pair with commit_shN_after_mutation.
    bool ensure_shN_fp32_for_mutation(core::SplatData& splat);

    /// After densify mutated float shN, re-encode to u16 + bounds (if quant on).
    /// Uses the model tensor allocator so exportable/GUI lands codes in the live block.
    bool commit_shN_after_mutation(core::SplatData& splat);

    /// Scope-exit commit for densify helpers with early-return paths. Commit can
    /// allocate and throw; the destructor contains and logs failures so unwinding
    /// never escalates to std::terminate.
    class ShNCommitGuard final {
    public:
        ShNCommitGuard(core::SplatData& splat, bool expanded, std::string_view site) noexcept
            : splat_(&splat), expanded_(expanded), site_(site) {}
        ~ShNCommitGuard() noexcept;

        ShNCommitGuard(const ShNCommitGuard&) = delete;
        ShNCommitGuard& operator=(const ShNCommitGuard&) = delete;

    private:
        core::SplatData* splat_ = nullptr;
        bool expanded_ = false;
        std::string_view site_;
    };

} // namespace lfs::training::sh_value

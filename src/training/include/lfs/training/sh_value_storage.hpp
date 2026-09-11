/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

/**
 * @file sh_value_storage.hpp
 * @brief Convert SplatData.shN between fp32 float4-swizzled and q16 pad-dropped storage.
 *
 * Call apply_shN_value_quant once after model load / before training when the flag is ON.
 *
 * Refine / densify keeps the q16 region authoritative. Mutation helpers
 * gather-decode only the source index set and re-encode only the touched
 * 256-splat bound blocks (or 256-aligned dest chunks for compact/append).
 * ensure_shN_fp32_for_mutation remains for non-refine callers that still
 * need a full-model float workspace (GUI transform bake, IEEE-f16 expand,
 * tests, Morton's fp32 fallback).
 */

#include "core/sh_value_quant_kernels.hpp"
#include "core/splat_data.hpp"

#include <cstddef>
#include <memory>
#include <string_view>

namespace lfs::training::sh_value {

    using lfs::core::sh_value_quant::decode_shN_f16_range_to_canonical;
    using lfs::core::sh_value_quant::decode_shN_u16_range_to_canonical;
    using lfs::core::sh_value_quant::decode_shN_u16_to_float4;
    using lfs::core::sh_value_quant::encode_shN_float4_to_u16;

    /// If quant is enabled and shN is still fp32, convert to Float16 u16 + bounds.
    /// No-op when already quantized or flag off. Returns true if converted.
    bool apply_shN_value_quant(core::SplatData& splat);

    /// If quant is on and shN is u16, expand to a float4-swizzled temp in-place for densify.
    /// Pair with commit_shN_after_mutation.
    bool ensure_shN_fp32_for_mutation(core::SplatData& splat);

    /// After densify mutated float shN, re-encode to u16 + bounds (if quant on).
    /// Uses the model tensor allocator so exportable/GUI lands codes in the live block.
    bool commit_shN_after_mutation(core::SplatData& splat);

    /// Gather-decode selected source primitives into canonical [K, rest, 3] fp32.
    /// q16: decode only those source splats. fp32: swizzled gather-to-linear.
    /// n_src_primitives=0 uses splat.size(). Pass the encoded source N when
    /// means have already grown past the q16 region.
    void gather_shN_to_canonical(core::SplatData& splat,
                                 const core::Tensor& src_indices,
                                 core::Tensor& dest_canonical,
                                 std::size_t n_src_primitives = 0);

    /// Scatter canonical [K, rest, 3] into dest primitive indices.
    /// q16: decode/replace/re-encode only the touched 256-splat bound blocks.
    void scatter_canonical_into_shN(core::SplatData& splat,
                                    const core::Tensor& dest_indices,
                                    const core::Tensor& src_canonical);

    /// Append canonical [K, rest, 3] at dest_offset (typically the pre-append N).
    /// q16: grow cells/bounds then re-encode only blocks covering
    /// [dest_offset, dest_offset+K).
    void append_canonical_to_shN(core::SplatData& splat,
                                 const core::Tensor& src_canonical,
                                 std::size_t dest_offset);

    /// Zero SH-rest at dest indices. q16: touched 256-splat blocks only.
    void zero_shN_at_indices(core::SplatData& splat, const core::Tensor& dest_indices);

    /// Accumulate dest writes from one refine event, then decode/overlay/encode
    /// each touched 256-splat block once. Recorded tensors are held by reference
    /// count until flush(). A single recorded op flushes through the matching
    /// direct helper above.
    class ShNMutationBatch final {
    public:
        explicit ShNMutationBatch(core::SplatData& splat);
        ~ShNMutationBatch();

        ShNMutationBatch(const ShNMutationBatch&) = delete;
        ShNMutationBatch& operator=(const ShNMutationBatch&) = delete;
        ShNMutationBatch(ShNMutationBatch&&) noexcept;
        ShNMutationBatch& operator=(ShNMutationBatch&&) noexcept;

        void scatter(const core::Tensor& dest_indices, const core::Tensor& src_canonical);
        void zero(const core::Tensor& dest_indices);
        void append(const core::Tensor& src_canonical, std::size_t dest_offset);
        void flush();

    private:
        struct Impl;
        std::unique_ptr<Impl> impl_;
    };

    /// Compact shN so dest[i] = src[keep[i]], sized to keep.numel().
    /// n_src is the pre-compact primitive count (means may already be compacted).
    /// dest_cap_prims is the reserved primitive capacity (0 = new N).
    void compact_shN_gather(core::SplatData& splat,
                            const core::Tensor& keep_indices,
                            std::size_t n_src,
                            std::size_t dest_cap_prims = 0);

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

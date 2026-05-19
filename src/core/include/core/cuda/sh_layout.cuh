/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include <cstddef>
#include <cstdint>
#include <cuda_runtime.h>

namespace lfs::core {

    // SH coefficient swizzle layout (port of vksplat shAt).
    //
    // Canonical layout: [N, K, 3] row-major, where K = active SH-rest coefficient count
    //                   (0 / 3 / 8 / 15 for SH degree 0 / 1 / 2 / 3).
    // Swizzled layout : interpret as [ceil(N/R), SH_MAX_COEFFS, R, 3] where R = SH_REORDER_SIZE.
    //                   For primitive p, coefficient k, channel c:
    //                       swizzled_float_idx = (shAt(p, k) * 3) + c
    //                       shAt(p, k) = (p / R) * (SH_MAX_COEFFS * R) + k * R + (p % R)
    //                   Adjacent primitives in one block hit adjacent lanes -> coalesced memory.
    //
    // We always allocate SH_MAX_COEFFS=16 slots per primitive (full SH3), even when the active
    // SH degree is lower. This keeps the per-block stride invariant under SH-degree promotion
    // and matches vksplat's allocation policy. Dead coefficient slots are zero-initialised.

    inline constexpr std::uint32_t SH_REORDER_SIZE = 32u;
    inline constexpr std::uint32_t SH_MAX_COEFFS = 16u;      // SH3 = 1 (DC) + 15 (rest); we hold all 16 in swizzled.
    inline constexpr std::uint32_t SH_MAX_COEFFS_REST = 15u; // shN excludes the DC term.
    inline constexpr std::uint32_t SH_CHANNELS = 3u;

    // Number of primitives in the swizzled buffer (rounded up to multiple of SH_REORDER_SIZE).
    [[nodiscard]] inline constexpr std::size_t sh_swizzled_block_count(std::size_t n) noexcept {
        return (n + SH_REORDER_SIZE - 1) / SH_REORDER_SIZE;
    }

    [[nodiscard]] inline constexpr std::size_t sh_swizzled_padded_n(std::size_t n) noexcept {
        return sh_swizzled_block_count(n) * SH_REORDER_SIZE;
    }

    // Total float count in the swizzled SH buffer for n primitives.
    // (ceil(n/R) * SH_MAX_COEFFS_REST * R * 3) — we only store the "rest" coefficients here;
    // the DC term (sh0) is a separate tensor.
    [[nodiscard]] inline constexpr std::size_t sh_swizzled_float_count(std::size_t n) noexcept {
        return sh_swizzled_block_count(n) * SH_MAX_COEFFS_REST * SH_REORDER_SIZE * SH_CHANNELS;
    }

    [[nodiscard]] inline constexpr std::size_t sh_swizzled_byte_count(std::size_t n) noexcept {
        return sh_swizzled_float_count(n) * sizeof(float);
    }

    // Host index helper. Identical math to the device `shAt` in fastgs kernel_utils.cuh.
    // Returns the float3-slot index; multiply by 3 to get the float offset.
    [[nodiscard]] inline std::uint32_t sh_swizzled_index(std::uint32_t primitive_idx, std::uint32_t coeff_idx) noexcept {
        const std::uint32_t block = primitive_idx / SH_REORDER_SIZE;
        const std::uint32_t lane = primitive_idx % SH_REORDER_SIZE;
        return block * (SH_MAX_COEFFS_REST * SH_REORDER_SIZE) + coeff_idx * SH_REORDER_SIZE + lane;
    }

    // Reorder canonical [N, K, 3] (K = active_coeffs_rest, contiguous, row-major) into the
    // swizzled buffer. Trailing lanes in the last block AND coefficient slots beyond K are
    // zero-filled. dst must be at least sh_swizzled_float_count(n) floats.
    void reorder_sh_to_swizzled(
        const float* src_canonical,
        float* dst_swizzled,
        std::size_t n_primitives,
        std::uint32_t active_coeffs_rest,
        cudaStream_t stream = nullptr);

    // Inverse of reorder_sh_to_swizzled: copy the first `active_coeffs_rest` coefficients of
    // the first n primitives back into canonical [N, K, 3] layout. dst must be at least
    // n * active_coeffs_rest * 3 floats.
    void undo_reorder_sh_from_swizzled(
        const float* src_swizzled,
        float* dst_canonical,
        std::size_t n_primitives,
        std::uint32_t active_coeffs_rest,
        cudaStream_t stream = nullptr);

    // Zero entire primitive rows in the swizzled buffer (all SH_MAX_COEFFS_REST coefficients).
    // Used by densification prune paths.
    void shN_swizzled_zero_at_indices(
        float* buffer_swizzled,
        const int* indices,
        std::size_t n_indices,
        cudaStream_t stream = nullptr);

    // int64 variant for callers holding device-side Int64 index buffers (e.g. relocate).
    void shN_swizzled_zero_at_indices_i64(
        float* buffer_swizzled,
        const std::int64_t* indices,
        std::size_t n_indices,
        cudaStream_t stream = nullptr);

    // Gather n_dst primitives' SH from src_indices[i] into dst position (dst_offset + i).
    // src and dst MAY alias (in-place append-gather) as long as the source range
    // [0, dst_offset) and the destination range [dst_offset, dst_offset + n_dst) are
    // disjoint, which is the case for MCMC growth (indices < dst_offset).
    void shN_swizzled_gather_self(
        const float* src_swizzled,
        float* dst_swizzled,
        const int* src_indices,
        std::size_t n_dst,
        std::size_t dst_offset = 0,
        cudaStream_t stream = nullptr);

    // int64 variant for callers holding indices in Int64 (Tensor's nonzero/multinomial).
    void shN_swizzled_gather_self_i64(
        const float* src_swizzled,
        float* dst_swizzled,
        const std::int64_t* src_indices,
        std::size_t n_dst,
        std::size_t dst_offset = 0,
        cudaStream_t stream = nullptr);

    // Append n_src linear rows (laid out as [n_src, active_coeffs_rest, 3]) into the
    // swizzled buffer starting at primitive index dst_offset.
    void shN_swizzled_gather_from_linear(
        float* dst_swizzled,
        std::size_t dst_offset,
        const float* src_linear,
        std::size_t n_src,
        std::uint32_t active_coeffs_rest,
        cudaStream_t stream = nullptr);

    // Scatter linear rows ([n_src, active_coeffs_rest, 3]) into specific primitive indices
    // of the swizzled buffer (equivalent of index_put_ on dim 0).
    void shN_swizzled_scatter_linear(
        float* dst_swizzled,
        const int* dst_indices,
        const float* src_linear,
        std::size_t n_src,
        std::uint32_t active_coeffs_rest,
        cudaStream_t stream = nullptr);

} // namespace lfs::core

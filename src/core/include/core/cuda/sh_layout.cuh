/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#pragma once

#include "core/cuda_types.hpp"
#include "core/sh_layout.hpp"

#include <cstddef>
#include <cstdint>

namespace lfs::core {

    // Canonical rows contain `src_coeffs_rest` coefficients but the
    // destination swizzled buffer has room for `layout_coeffs_rest`. Coefficients beyond
    // src_coeffs_rest are zero-filled in the destination layout.
    void reorder_sh_to_swizzled(
        const float* src_canonical,
        float* dst_swizzled,
        std::size_t n_primitives,
        std::uint32_t src_coeffs_rest,
        std::uint32_t layout_coeffs_rest,
        cudaStream_t stream = nullptr);

    // Inverse of reorder_sh_to_swizzled: copy dst_coeffs_rest coefficients from
    // source rows with layout_coeffs_rest slots into canonical [N, K, 3] rows.
    void undo_reorder_sh_from_swizzled(
        const float* src_swizzled,
        float* dst_canonical,
        std::size_t n_primitives,
        std::uint32_t dst_coeffs_rest,
        std::uint32_t layout_coeffs_rest,
        cudaStream_t stream = nullptr);

    // Bounded snapshot variant. Writes canonical floats in
    // [canonical_float_offset, canonical_float_offset + float_count) to a
    // zero-based scratch destination, avoiding a checkpoint-sized D2D buffer.
    void undo_reorder_sh_range_from_swizzled(
        const float* src_swizzled,
        float* dst_canonical_scratch,
        std::uint64_t canonical_float_offset,
        std::uint64_t float_count,
        std::size_t n_primitives,
        std::uint32_t dst_coeffs_rest,
        std::uint32_t layout_coeffs_rest,
        cudaStream_t stream = nullptr);

    // Zero entire primitive rows in the swizzled buffer (all active float4 slots).
    // Used by densification prune paths.
    void shN_swizzled_zero_at_indices(
        float* buffer_swizzled,
        const int* indices,
        std::size_t n_indices,
        std::uint32_t active_coeffs_rest,
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
        std::size_t dst_offset,
        std::uint32_t active_coeffs_rest,
        cudaStream_t stream = nullptr);

    // int64 variant for callers holding indices in Int64 (Tensor's nonzero/multinomial).
    void shN_swizzled_gather_self_i64(
        const float* src_swizzled,
        float* dst_swizzled,
        const std::int64_t* src_indices,
        std::size_t n_dst,
        std::size_t dst_offset,
        std::uint32_t active_coeffs_rest,
        cudaStream_t stream = nullptr);

    // uint8 (quantised Adam moment) variants of shN_swizzled_gather_self. Same swizzled slot
    // permutation, just uchar4 elements instead of float4.
    void shN_swizzled_gather_self_u8(
        const std::uint8_t* src_swizzled,
        std::uint8_t* dst_swizzled,
        const int* src_indices,
        std::size_t n_dst,
        std::size_t dst_offset,
        std::uint32_t active_coeffs_rest,
        cudaStream_t stream = nullptr);

    // Copy n_src primitives starting at src_offset from one swizzled buffer into
    // dst_offset in another swizzled buffer.
    void shN_swizzled_copy_range(
        const float* src_swizzled,
        float* dst_swizzled,
        std::size_t src_offset,
        std::size_t n_src,
        std::size_t dst_offset,
        std::uint32_t src_active_coeffs_rest,
        std::uint32_t dst_active_coeffs_rest,
        cudaStream_t stream = nullptr);

    // Gather selected primitives from swizzled storage into contiguous linear rows laid out as
    // [n_src, dst_coeffs_rest, 3]. This is the selected-row inverse of
    // reorder_sh_to_swizzled and is used by densification paths that only need child rows.
    void shN_swizzled_gather_to_linear(
        const float* src_swizzled,
        const int* src_indices,
        float* dst_linear,
        std::size_t n_src,
        std::uint32_t dst_coeffs_rest,
        std::uint32_t layout_coeffs_rest,
        cudaStream_t stream = nullptr);

    // int64 variant for callers holding Tensor nonzero/multinomial indices.
    void shN_swizzled_gather_to_linear_i64(
        const float* src_swizzled,
        const std::int64_t* src_indices,
        float* dst_linear,
        std::size_t n_src,
        std::uint32_t dst_coeffs_rest,
        std::uint32_t layout_coeffs_rest,
        cudaStream_t stream = nullptr);

    // Append n_src linear rows (laid out as [n_src, src_coeffs_rest, 3]) into the
    // swizzled buffer starting at primitive index dst_offset.
    void shN_swizzled_gather_from_linear(
        float* dst_swizzled,
        std::size_t dst_offset,
        const float* src_linear,
        std::size_t n_src,
        std::uint32_t src_coeffs_rest,
        std::uint32_t layout_coeffs_rest,
        cudaStream_t stream = nullptr);

    // Scatter linear rows ([n_src, src_coeffs_rest, 3]) into specific primitive indices
    // of the swizzled buffer (equivalent of index_put_ on dim 0).
    void shN_swizzled_scatter_linear(
        float* dst_swizzled,
        const int* dst_indices,
        const float* src_linear,
        std::size_t n_src,
        std::uint32_t src_coeffs_rest,
        std::uint32_t layout_coeffs_rest,
        cudaStream_t stream = nullptr);

} // namespace lfs::core

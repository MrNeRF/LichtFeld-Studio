/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include <cstddef>
#include <cstdint>
#include <cuda_runtime.h>

namespace lfs::core {

    // SH coefficient swizzle layout (vksplat float4 packing, ported from
    // vksplat/vksplat/slang/spherical_harmonics.slang).
    //
    // Canonical layout (input/export):
    //     [N, K, 3] row-major, K = active SH-rest coefficient count (0 / 3 / 8 / 15 for SH 0-3).
    //
    // Swizzled layout (resident GPU storage):
    //     [ceil(N/R), kShRestFloat4PerPrimitive, R] of float4, where R = kShReorderSize.
    //     For a warp of R lanes the float4 reads/writes coalesce into a single 16-byte vector
    //     load per coefficient slot — fixing the stride-12 misaligned loads of the old float3
    //     layout.
    //
    // Packing of 15 float3 coefficients (c0..c14) into 12 float4 slots, identical to vksplat
    // but with the DC term (which we keep in a separate sh0 tensor) replaced by tail padding:
    //
    //     slot 0  : (c0.x, c0.y, c0.z, c1.x)         slot 6  : (c8.x,  c8.y,  c8.z,  c9.x)
    //     slot 1  : (c1.y, c1.z, c2.x, c2.y)         slot 7  : (c9.y,  c9.z,  c10.x, c10.y)
    //     slot 2  : (c2.z, c3.x, c3.y, c3.z)         slot 8  : (c10.z, c11.x, c11.y, c11.z)
    //     slot 3  : (c4.x, c4.y, c4.z, c5.x)         slot 9  : (c12.x, c12.y, c12.z, c13.x)
    //     slot 4  : (c5.y, c5.z, c6.x, c6.y)         slot 10 : (c13.y, c13.z, c14.x, c14.y)
    //     slot 5  : (c6.z, c7.x, c7.y, c7.z)         slot 11 : (c14.z, 0,     0,     0    )
    //
    // shAt(p, k) returns the float4 slot index (multiply by 4 to get the float offset).
    //
    //     shAt(p, k) = (p / R) * (kShRestFloat4PerPrimitive * R) + k * R + (p % R)
    //
    // We always allocate the full 12 float4 slots per primitive (even when the active SH degree
    // is lower); slots past the active range are zero-initialised. This keeps the per-block
    // stride invariant under SH-degree promotion.

    inline constexpr std::uint32_t kShReorderSize = 32u;
    inline constexpr std::uint32_t kShMaxCoeffsRest = 15u;          // canonical float3 coeff count (shN)
    inline constexpr std::uint32_t kShRestFloat4PerPrimitive = 12u; // packed float4 slot count per primitive
    inline constexpr std::uint32_t kShChannels = 3u;                // canonical layout channel count

    // Number of primitives in the swizzled buffer (rounded up to multiple of kShReorderSize).
    [[nodiscard]] inline constexpr std::size_t sh_swizzled_block_count(std::size_t n) noexcept {
        return (n + kShReorderSize - 1) / kShReorderSize;
    }

    [[nodiscard]] inline constexpr std::size_t sh_swizzled_padded_n(std::size_t n) noexcept {
        return sh_swizzled_block_count(n) * kShReorderSize;
    }

    // Total float count in the swizzled SH buffer for n primitives.
    // (ceil(n/R) * kShRestFloat4PerPrimitive * R * 4) — packed float4 layout.
    // The DC term (sh0) is stored in a separate tensor; this buffer only holds shN-rest.
    [[nodiscard]] inline constexpr std::size_t sh_swizzled_float_count(std::size_t n) noexcept {
        return sh_swizzled_block_count(n) * kShRestFloat4PerPrimitive * kShReorderSize * 4u;
    }

    // Total float4 slot count in the swizzled SH buffer for n primitives.
    [[nodiscard]] inline constexpr std::size_t sh_swizzled_float4_count(std::size_t n) noexcept {
        return sh_swizzled_block_count(n) * kShRestFloat4PerPrimitive * kShReorderSize;
    }

    [[nodiscard]] inline constexpr std::size_t sh_swizzled_byte_count(std::size_t n) noexcept {
        return sh_swizzled_float_count(n) * sizeof(float);
    }

    // Host index helper. Returns the float4-slot index (multiply by 4 to get the float offset).
    [[nodiscard]] inline std::uint32_t sh_swizzled_index(std::uint32_t primitive_idx, std::uint32_t float4_slot) noexcept {
        const std::uint32_t block = primitive_idx / kShReorderSize;
        const std::uint32_t lane = primitive_idx % kShReorderSize;
        return block * (kShRestFloat4PerPrimitive * kShReorderSize) + float4_slot * kShReorderSize + lane;
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

    // Zero entire primitive rows in the swizzled buffer (all 12 float4 slots).
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

    // Copy the first n_src primitives from one swizzled buffer into dst_offset in another
    // swizzled buffer. Source and destination may have different padded block boundaries.
    void shN_swizzled_copy_contiguous(
        const float* src_swizzled,
        float* dst_swizzled,
        std::size_t n_src,
        std::size_t dst_offset = 0,
        cudaStream_t stream = nullptr);

    // Gather selected primitives from swizzled storage into contiguous linear rows laid out as
    // [n_src, active_coeffs_rest, 3]. This is the selected-row inverse of
    // reorder_sh_to_swizzled and is used by densification paths that only need child rows.
    void shN_swizzled_gather_to_linear(
        const float* src_swizzled,
        const int* src_indices,
        float* dst_linear,
        std::size_t n_src,
        std::uint32_t active_coeffs_rest,
        cudaStream_t stream = nullptr);

    // int64 variant for callers holding Tensor nonzero/multinomial indices.
    void shN_swizzled_gather_to_linear_i64(
        const float* src_swizzled,
        const std::int64_t* src_indices,
        float* dst_linear,
        std::size_t n_src,
        std::uint32_t active_coeffs_rest,
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

    // Pack split resident SH storage into VkSplat's full 16-coefficient packed layout:
    // sh0 [N, 1, 3] or [N, 3] + swizzled shN rest -> [ceil(N/32), 12, 32] float4.
    // dst_full_swizzled must have sh_swizzled_float_count(n) floats. src_shN_swizzled
    // may be null, in which case the rest coefficients are packed as zeros.
    void sh_swizzled_pack_full_from_split(
        const float* src_sh0,
        const float* src_shN_swizzled,
        float* dst_full_swizzled,
        std::size_t n_primitives,
        cudaStream_t stream = nullptr);

} // namespace lfs::core

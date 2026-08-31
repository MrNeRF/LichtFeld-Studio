/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include <cstddef>
#include <cstdint>
#include <cuda_runtime.h>

namespace lfs::core::sh_value_quant {

    struct SortedBlockRunScratch {
        std::int32_t* flags = nullptr;
        std::int32_t* compact = nullptr;
        void* scan = nullptr;
        std::size_t scan_bytes = 0;
    };

    /// Encode float4-swizzled SH-rest into pad-dropped u16 + float2 bounds / 256.
    void encode_shN_float4_to_u16(
        const float* src_float4_swizzled,
        std::uint16_t* dst_u16,
        float* bounds_float2,
        std::size_t n_primitives,
        std::uint32_t coeffs_rest,
        cudaStream_t stream = nullptr);

    /// Decode u16 + bounds into float4-swizzled (zeros float4 tail pad).
    void decode_shN_u16_to_float4(
        const std::uint16_t* src_u16,
        const float* bounds_float2,
        float* dst_float4_swizzled,
        std::size_t n_primitives,
        std::uint32_t coeffs_rest,
        cudaStream_t stream = nullptr);

    /// Gather-decode: dest-local primitive i (0..n_dst-1) reads source
    /// primitive perm[dest_offset + i] and writes float4-swizzled storage as
    /// if those n_dst primitives started at index 0. Source bounds are the
    /// live per-256-block table indexed by the source primitive.
    void decode_shN_u16_gathered_to_float4(
        const std::uint16_t* src_u16,
        const float* src_bounds_float2,
        const std::int64_t* perm,
        float* dst_float4_swizzled,
        std::size_t dest_offset,
        std::size_t n_dst,
        std::size_t n_src_primitives,
        std::uint32_t coeffs_rest,
        cudaStream_t stream = nullptr);

    /// Sequential decode of prims [src_offset, src_offset + n_dst) into a
    /// compact float4-swizzled buffer that treats those prims as 0..n_dst-1.
    /// Prims with source index >= n_src_primitives are written as zeros.
    void decode_shN_u16_range_to_float4(
        const std::uint16_t* src_u16,
        const float* src_bounds_float2,
        float* dst_float4_swizzled,
        std::size_t src_offset,
        std::size_t n_dst,
        std::size_t n_src_primitives,
        std::uint32_t coeffs_rest,
        cudaStream_t stream = nullptr);

    /// Gather-decode selected source prims into canonical [n_dst, rest, 3].
    void decode_shN_u16_gathered_to_canonical(
        const std::uint16_t* src_u16,
        const float* src_bounds_float2,
        const std::int64_t* perm,
        float* dst_canonical,
        std::size_t n_dst,
        std::size_t n_src_primitives,
        std::uint32_t coeffs_rest,
        cudaStream_t stream = nullptr);

    /// Overlay canonical [K, rest, 3] rows into a compact float4-swizzled
    /// chunk of n_chunk prims that represents global prims
    /// [block_start, block_start + n_chunk). dest_indices[group_offset + i]
    /// is the global primitive for canonical row group_offset + i.
    void overlay_canonical_into_float4_chunk(
        const float* src_canonical,
        const std::int64_t* dest_indices,
        float* dst_float4_swizzled,
        std::size_t group_offset,
        std::size_t n_group,
        std::size_t block_start,
        std::size_t n_chunk,
        std::uint32_t coeffs_rest,
        cudaStream_t stream = nullptr);

    /// block_id[i] = dest_indices[i] / 256 as float32 (exact for training-scale N).
    void fill_quant_block_ids_f32(
        const std::int64_t* dest_indices,
        float* block_ids_f32,
        std::size_t n,
        cudaStream_t stream = nullptr);

    /// Compact unique block-ids and run starts from already-sorted float block ids.
    /// unique_block_ids[0..n_runs) and run_offsets[0..n_runs) stay on device.
    /// n_runs_device[0] is the run count. No host readback of block ids.
    void build_sorted_block_runs(
        const float* sorted_block_ids,
        std::int32_t* unique_block_ids,
        std::int32_t* run_offsets,
        std::int32_t* n_runs_device,
        std::size_t n,
        cudaStream_t stream = nullptr,
        const SortedBlockRunScratch* scratch = nullptr);

    /// Query the CUB scan storage needed by build_sorted_block_runs.
    std::size_t sorted_block_runs_scan_workspace_bytes(
        std::size_t n,
        cudaStream_t stream = nullptr);

    /// One launch over unique touched 256-splat blocks: decode (or zero-init
    /// past n_decode_src), overlay the sorted canonical run, reduce bounds,
    /// encode in place. Grid may be K; extra blocks no-op via n_runs_device.
    void reencode_touched_q16_blocks(
        std::uint16_t* codes,
        float* bounds_float2,
        const float* sorted_canonical,
        const std::int64_t* sorted_dest,
        const std::int32_t* unique_block_ids,
        const std::int32_t* run_offsets,
        const std::int32_t* n_runs_device,
        std::size_t n_sorted,
        std::size_t n_prims,
        std::size_t n_decode_src,
        std::uint32_t coeffs_rest,
        cudaStream_t stream = nullptr);

    /// Gather-decode perm[dest] with source per-block bounds, reduce dest
    /// bounds, encode dest 256-splat blocks. One launch for the whole dest.
    void encode_shN_u16_gathered(
        const std::uint16_t* src_u16,
        const float* src_bounds_float2,
        const std::int64_t* perm,
        std::uint16_t* dest_u16,
        float* dest_bounds_float2,
        std::size_t n_dst,
        std::size_t n_src_primitives,
        std::uint32_t coeffs_rest,
        cudaStream_t stream = nullptr);

    void decode_shN_u16_range_to_canonical(
        const std::uint16_t* src_u16,
        const float* bounds_float2,
        float* dst_canonical,
        std::uint64_t canonical_float_offset,
        std::uint64_t float_count,
        std::size_t n_primitives,
        std::uint32_t dst_coeffs_rest,
        std::uint32_t layout_coeffs_rest,
        cudaStream_t stream = nullptr);

    void decode_shN_f16_range_to_canonical(
        const std::uint16_t* src_f16,
        float* dst_canonical,
        std::uint64_t canonical_float_offset,
        std::uint64_t float_count,
        std::size_t n_primitives,
        std::uint32_t dst_coeffs_rest,
        std::uint32_t layout_coeffs_rest,
        cudaStream_t stream = nullptr);

} // namespace lfs::core::sh_value_quant

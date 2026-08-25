// densification_kernels.hpp
/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include <cstdint>
#include <cuda_runtime.h>

namespace lfs::training {
    struct PositiveMedianScratch;
}

namespace lfs::training::kernels {

    /**
     * @brief Custom CUDA kernels for Gaussian densification (LAS path).
     *
     * Long-axis split densification kernels minimize intermediate allocations.
     */

    /**
     * @brief In-place Long-Axis-Split selected Gaussians
     *
     * Modifies original Gaussians in-place for first split result,
     * writes second split result to separate output arrays.
     *
     * This kernel is designed for soft-deletion mode where kept Gaussians
     * remain in their original positions (no compaction).
     *
     * Output layout:
     * - First split result: written back to original positions (in-place)
     * - Second split result: written to second_* arrays [num_split, ...]
     *
     * @param positions Input/output positions [N, 3] - modified in-place for split Gaussians
     * @param rotations Input/output rotations [N, 4] - unchanged (same rotation for both splits)
     * @param scales Input/output scales [N, 3] - modified in-place for split Gaussians
     * @param sh0 Input SH degree 0 [N, 3] - unchanged
     * @param shN Input SH higher degrees [N, shN_dim] - unchanged
     * @param opacities Input/output opacities [N, 1]
     * @param second_positions Output positions for second split [num_split, 3]
     * @param second_rotations Output rotations for second split [num_split, 4]
     * @param second_scales Output scales for second split [num_split, 3]
     * @param second_sh0 Output SH degree 0 for second split [num_split, 3]
     * @param second_shN Output SH higher degrees for second split [num_split, shN_dim]
     * @param second_opacities Output opacities for second split [num_split, 1]
     * @param split_indices Indices of Gaussians to split [num_split]
     * @param num_split Number of Gaussians to split
     * @param shN_dim SH higher degree dimension
     * @param stream CUDA stream
     */
    void launch_long_axis_split_gaussians_inplace(
        float* positions,
        float* rotations,
        float* scales,
        const float* sh0,
        const float* shN,
        float* opacities,
        float* second_positions,
        float* second_rotations,
        float* second_scales,
        float* second_sh0,
        float* second_shN,
        float* second_opacities,
        const int64_t* split_indices,
        int num_split,
        int shN_dim,
        cudaStream_t stream = nullptr);

    /**
     * fused free-slot write.
     *
     * For each i in [0, n_fill): writes child row i into param row target_indices[i]
     * (means/rot/scale/sh0/opacity), zeros up to 12 Adam per-primitive scale buffers
     * at that index, and sets free_mask[target]=false.
     *
     * sh0 layout: source is always 3 floats per row (flat). Destination is either
     * [N,3] (sh0_dst_stride3=true) or [N,1,3] contiguous (false → write 3 floats at
     * index*3 still works for [N,1,3] since middle dim is 1).
     * opacity_dim: 0 → [N], 1 → [N,1].
     * adam_scale_ptrs: nullable array of length n_adam_scales; each is [N] float.
     */
    void launch_fill_free_slots_fused(
        const int64_t* target_indices,
        size_t n_fill,
        const float* src_means,
        const float* src_rotations,
        const float* src_scales,
        const float* src_sh0,
        const float* src_opacities,
        float* dst_means,
        float* dst_rotations,
        float* dst_scales,
        float* dst_sh0,
        float* dst_opacities,
        int opacity_dim,
        float* const* adam_scale_ptrs,
        int n_adam_scales,
        bool* free_mask,
        size_t N,
        cudaStream_t stream = nullptr);

    /**
     * Zero Adam per-primitive scales at indices (quantized moments dequant to 0).
     * Used for parent-split reset without a full free-slot write.
     */
    void launch_zero_adam_scales_at_indices(
        const int64_t* indices,
        size_t n_indices,
        float* const* adam_scale_ptrs,
        int n_adam_scales,
        size_t N,
        cudaStream_t stream = nullptr);

    /**
     * pack up to 4 host-needed counts into one device buffer for a
     * single D2H. Each input may be null (skipped → out[i]=0).
     *   slot 0/1: count of true bools over n_bool0/1
     *   slot 2/3: count of strictly-positive floats over n_f0/1
     */
    void launch_packed_refine_counts(
        const bool* bool0,
        size_t n_bool0,
        const bool* bool1,
        size_t n_bool1,
        const float* float0,
        size_t n_float0,
        const float* float1,
        size_t n_float1,
        int64_t* out_counts4,
        cudaStream_t stream = nullptr);

    /**
     * positive-median normalize without full-tensor sort.
     * Compact positives → radix-sort the compact buffer → median at count/2 →
     * divide data in-place by max(median, 1e-9). NaN/non-positive left as-is
     * after a pre-pass that zeros NaNs (caller may pre-masked_fill).
     */
    void launch_normalize_by_positive_median(
        float* data,
        size_t n,
        cudaStream_t stream = nullptr,
        lfs::training::PositiveMedianScratch* scratch = nullptr);

} // namespace lfs::training::kernels

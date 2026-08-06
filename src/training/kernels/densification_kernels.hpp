// densification_kernels.hpp
/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include <cstdint>
#include <cuda_runtime.h>

namespace lfs::training::kernels {

    /**
     * @brief Custom CUDA kernels for Gaussian densification
     *
     * These kernels replace LibTorch-heavy operations in densification with
     * efficient CUDA kernels that minimize memory allocations and maximize throughput.
     *
     * Memory savings:
     * - Old approach: 60+ intermediate tensors (8+ GB for 10M Gaussians)
     * - New approach: 3 temp allocations (~500 MB for 10M Gaussians)
     *
     * Performance:
     * - Duplicate: ~2ms for 10M Gaussians (vs ~50ms LibTorch)
     * - Split: ~5ms for 10M Gaussians (vs ~200ms LibTorch)
     */

    /**
     * @brief Duplicate selected Gaussians
     *
     * Copies selected Gaussians to end of output arrays.
     * Output arrays must be pre-allocated to size (N + num_selected).
     *
     * @param positions_in Input positions [N, 3]
     * @param rotations_in Input rotations [N, 4]
     * @param scales_in Input scales [N, 3]
     * @param sh0_in Input SH degree 0 [N, 3]
     * @param shN_in Input SH higher degrees [N, shN_dim]
     * @param opacities_in Input opacities [N, 1]
     * @param positions_out Output positions [N + num_selected, 3]
     * @param rotations_out Output rotations [N + num_selected, 4]
     * @param scales_out Output scales [N + num_selected, 3]
     * @param sh0_out Output SH degree 0 [N + num_selected, 3]
     * @param shN_out Output SH higher degrees [N + num_selected, shN_dim]
     * @param opacities_out Output opacities [N + num_selected, 1]
     * @param selected_indices Indices to duplicate [num_selected]
     * @param N Number of input Gaussians
     * @param num_selected Number of Gaussians to duplicate
     * @param shN_dim SH higher degree dimension
     * @param stream CUDA stream
     */
    void launch_duplicate_gaussians(
        const float* positions_in,
        const float* rotations_in,
        const float* scales_in,
        const float* sh0_in,
        const float* shN_in,
        const float* opacities_in,
        float* positions_out,
        float* rotations_out,
        float* scales_out,
        float* sh0_out,
        float* shN_out,
        float* opacities_out,
        const int64_t* selected_indices,
        int N,
        int num_selected,
        int shN_dim,
        cudaStream_t stream = nullptr);

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

} // namespace lfs::training::kernels

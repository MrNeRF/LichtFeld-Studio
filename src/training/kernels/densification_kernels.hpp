// densification_kernels.hpp
/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include <cstdint>
#include <cuda_runtime.h>

namespace lfs::training::kernels {

    /**
     * @brief Custom CUDA kernels for Gaussian densification (LAS path).
     *
     * Long-axis split densification kernels minimize intermediate allocations
     * versus the historical LibTorch-heavy densify path.
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

} // namespace lfs::training::kernels

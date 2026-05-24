/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include <cstddef>
#include <cuda_runtime_api.h>

namespace lfs::vis::vksplat::detail {

    [[nodiscard]] cudaError_t launchPackActivatedRotations(
        const float* rotation_raw,
        float* rotations_dst,
        std::size_t num_splats,
        cudaStream_t stream);

    [[nodiscard]] cudaError_t launchPackScalesOpacs(
        const float* scaling_raw,
        const float* opacity_raw,
        float* scales_opacs_dst,
        std::size_t num_splats,
        cudaStream_t stream);

    // Copy opacity_raw → opacity_dst, but force soft-deleted entries to a
    // strongly-negative raw value so sigmoid(raw) ≈ 0 in the rasterizer. Used
    // to honor SplatData::deleted() through the VkSplat path without modifying
    // the shared opacity tensor. `deleted_mask` is the bool tensor's storage
    // (1 byte per element); the kernel only checks non-zero.
    [[nodiscard]] cudaError_t launchPackOpacityMaskingDeleted(
        const float* opacity_raw,
        const bool* deleted_mask,
        float* opacity_dst,
        std::size_t num_splats,
        cudaStream_t stream);

} // namespace lfs::vis::vksplat::detail

/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include "core/tensor.hpp"
#include <cstdint>
#include <cuda_runtime.h>

namespace lfs::io {

    using lfs::core::Tensor;

    __host__ __device__ constexpr uint64_t morton_spread(uint64_t x) {
        x &= 0x1fffffULL;
        x = (x | (x << 32)) & 0x1f00000000ffffULL;
        x = (x | (x << 16)) & 0x1f0000ff0000ffULL;
        x = (x | (x << 8)) & 0x100f00f00f00f00fULL;
        x = (x | (x << 4)) & 0x10c30c30c30c30c3ULL;
        return (x | (x << 2)) & 0x1249249249249249ULL;
    }

    __host__ __device__ constexpr uint64_t morton_encode(uint32_t x, uint32_t y, uint32_t z) {
        return morton_spread(x) | (morton_spread(y) << 1) | (morton_spread(z) << 2);
    }

    __host__ __device__ constexpr float morton_multiplier(float extent) {
        return extent == 0 ? 0 : float(1u << 21) / extent;
    }

    __host__ __device__ constexpr uint32_t morton_coordinate(float position, float low, float multiplier) {
        constexpr uint32_t axis_max = (1u << 21) - 1;
        const float normalized = (position - low) * multiplier;
        return normalized <= 0 ? 0 : normalized >= axis_max ? axis_max
                                                            : static_cast<uint32_t>(normalized);
    }

    /**
     * @brief Compute 63-bit Morton codes and sort indices on the GPU.
     *
     * SOG export uses 21 bits per axis over the global bounds. This finer grid
     * avoids the dense-cell scrambling caused by a single 10-bit pass while
     * retaining the source order for equal keys.
     *
     * @param positions Tensor of shape [N, 3] containing 3D positions (Float32, CUDA)
     * @param sorted_keys Optional output for the correspondingly sorted 63-bit keys.
     * @return Tensor of sorted indices (Int32, CUDA)
     */
    Tensor morton_sort_indices_for_positions(const Tensor& positions, Tensor* sorted_keys = nullptr);

} // namespace lfs::io

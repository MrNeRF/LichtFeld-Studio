/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include <cstdint>

#if defined(__CUDACC__)
#define LFS_MORTON_HD __host__ __device__
#else
#define LFS_MORTON_HD
#endif

namespace lfs::core {

    LFS_MORTON_HD constexpr uint64_t morton_spread(uint64_t x) {
        x &= 0x1fffffULL;
        x = (x | (x << 32)) & 0x1f00000000ffffULL;
        x = (x | (x << 16)) & 0x1f0000ff0000ffULL;
        x = (x | (x << 8)) & 0x100f00f00f00f00fULL;
        x = (x | (x << 4)) & 0x10c30c30c30c30c3ULL;
        return (x | (x << 2)) & 0x1249249249249249ULL;
    }

    LFS_MORTON_HD constexpr uint64_t morton_encode(uint32_t x, uint32_t y, uint32_t z) {
        return morton_spread(x) | (morton_spread(y) << 1) | (morton_spread(z) << 2);
    }

    LFS_MORTON_HD constexpr float morton_multiplier(float extent) {
        return extent == 0 ? 0 : float(1u << 21) / extent;
    }

    LFS_MORTON_HD constexpr uint32_t morton_coordinate(float position, float low, float multiplier) {
        constexpr uint32_t axis_max = (1u << 21) - 1;
        const float normalized = (position - low) * multiplier;
        return normalized <= 0 ? 0 : normalized >= axis_max ? axis_max
                                                            : static_cast<uint32_t>(normalized);
    }

} // namespace lfs::core

#undef LFS_MORTON_HD

/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#pragma once
#include "sharpness_score.hpp"
#include <cstddef>

#ifdef __CUDACC__
#define LFS_SHARPNESS_HD __host__ __device__
#else
#define LFS_SHARPNESS_HD
#endif
namespace lfs::io::detail {
    inline double edgeRatio(unsigned long long edges, int w, int h) {
        return static_cast<double>(edges) * 100.0 / static_cast<double>(static_cast<long long>(w) * h);
    }

    LFS_SHARPNESS_HD inline int magnitude(int value) { return value < 0 ? -value : value; }
    // Shared integer predicate: host and device count exactly the same pixels.
    LFS_SHARPNESS_HD inline bool sharpEdge(const uint8_t* p, std::ptrdiff_t stride, SharpnessAlgorithm algorithm) {
        if (algorithm == SharpnessAlgorithm::LAPLACIAN || algorithm == SharpnessAlgorithm::COMBINED) {
            const int lap = magnitude(p[0] * 4 - p[-stride] - p[3] - p[-3] - p[stride]);
            if (algorithm == SharpnessAlgorithm::LAPLACIAN || lap > 10)
                return lap > 10;
        }
        const int gx = -p[-stride - 3] + p[-stride + 3] - p[-3] * 2 + p[3] * 2 - p[stride - 3] + p[stride + 3];
        const int gy = -p[-stride - 3] - p[-stride] * 2 - p[-stride + 3] + p[stride - 3] + p[stride] * 2 + p[stride + 3];
        return magnitude(gx) + magnitude(gy) > 40;
    }
} // namespace lfs::io::detail
#undef LFS_SHARPNESS_HD

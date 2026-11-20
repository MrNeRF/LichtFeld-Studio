/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#pragma once
#include "sharpness_score.hpp"
#include <cstddef>

namespace lfs::io::detail {
    inline int magnitude(int value) { return value < 0 ? -value : value; }
    // Count an edge using the green channel and the selected integer stencil.
    inline bool sharpEdge(const uint8_t* p, std::ptrdiff_t stride, SharpnessAlgorithm algorithm) {
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

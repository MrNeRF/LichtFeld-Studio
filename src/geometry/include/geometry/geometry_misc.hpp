/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include <glm/glm.hpp>
#include <span>

namespace lfs {
    namespace geometry {

        // Compute the geometric median (L1 median) of a set of 3D points using
        // Weiszfeld's iterative algorithm. O(N * max_iter), converges in ~30-50 iterations.
        glm::vec3 geometric_median(std::span<const glm::vec3> points,
                                   int max_iter = 100,
                                   float tol = 1e-6f,
                                   int max_points_to_sample = 50000);

        // Median Absolute Deviation: median of L2 distances from each point to the given median.
        float geometric_mad(std::span<const glm::vec3> points, const glm::vec3& median);

    } // namespace geometry
} // namespace lfs

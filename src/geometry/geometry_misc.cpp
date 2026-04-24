/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "geometry/geometry_misc.hpp"
#include <glm/geometric.hpp>

namespace lfs {
    namespace geometry {

        glm::vec3 geometric_median(std::span<const glm::vec3> points,
                                   int max_iter,
                                   float tol) {
            if (points.empty()) return glm::vec3{0.0f};
            if (points.size() == 1) return points[0];
            if (points.size() == 2) return (points[0] + points[1]) * 0.5f;

            // Initialize at arithmetic mean
            glm::vec3 y{0.0f};
            for (const auto& p : points)
                y += p;
            y /= static_cast<float>(points.size());

            constexpr float eps = 1e-8f;

            for (int iter = 0; iter < max_iter; ++iter) {
                glm::vec3 num{0.0f};
                float den = 0.0f;

                for (const auto& p : points) {
                    float d = glm::distance(y, p);
                    if (d < eps) continue;
                    float w = 1.0f / d;
                    num += w * p;
                    den += w;
                }

                if (den < eps) break;

                glm::vec3 y_new = num / den;

                if (glm::distance(y_new, y) < tol) {
                    y = y_new;
                    break;
                }
                y = y_new;
            }

            return y;
        }

    } // namespace geometry
} // namespace lfs

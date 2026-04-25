/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "geometry/geometry_misc.hpp"
#include <algorithm>
#include <glm/geometric.hpp>
#include <numeric>
#include <random>

namespace lfs {
    namespace geometry {

        glm::vec3 geometric_median(std::span<const glm::vec3> points,
                                   int max_iter,
                                   float tol,
                                   int max_points_to_sample) {
            if (points.empty())
                return glm::vec3{0.0f};
            if (points.size() == 1)
                return points[0];
            if (points.size() == 2)
                return (points[0] + points[1]) * 0.5f;

            // Sample if point count exceeds the threshold
            std::vector<glm::vec3> sample_buf;
            std::span<const glm::vec3> work = points;
            if (max_points_to_sample > 0 &&
                static_cast<int>(points.size()) > max_points_to_sample) {
                sample_buf.resize(static_cast<size_t>(max_points_to_sample));
                std::vector<size_t> indices(points.size());
                std::iota(indices.begin(), indices.end(), size_t{0});
                std::mt19937 rng{42};
                std::ranges::shuffle(indices, rng);
                for (int i = 0; i < max_points_to_sample; ++i)
                    sample_buf[static_cast<size_t>(i)] = points[indices[static_cast<size_t>(i)]];
                work = sample_buf;
            }

            // Initialize at arithmetic mean
            glm::vec3 y{0.0f};
            for (const auto& p : work)
                y += p;
            y /= static_cast<float>(work.size());

            constexpr float eps = 1e-8f;

            for (int iter = 0; iter < max_iter; ++iter) {
                glm::vec3 num{0.0f};
                float den = 0.0f;

                for (const auto& p : work) {
                    float d = glm::distance(y, p);
                    if (d < eps)
                        continue;
                    float w = 1.0f / d;
                    num += w * p;
                    den += w;
                }

                if (den < eps)
                    break;

                glm::vec3 y_new = num / den;

                if (glm::distance(y_new, y) < tol) {
                    y = y_new;
                    break;
                }
                y = y_new;
            }

            return y;
        }

        float geometric_mad(std::span<const glm::vec3> points, const glm::vec3& median) {
            if (points.empty())
                return 0.0f;

            std::vector<float> distances(points.size());
            for (size_t i = 0; i < points.size(); ++i)
                distances[i] = glm::distance(points[i], median);

            auto mid = distances.begin() + static_cast<ptrdiff_t>(distances.size() / 2);
            std::nth_element(distances.begin(), mid, distances.end());
            return *mid;
        }

    } // namespace geometry
} // namespace lfs

/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include <algorithm>
#include <cmath>
#include <glm/glm.hpp>
#include <optional>
#include <span>
#include <utility>

namespace lfs::vis::gui {
    struct GuideClipPlane {
        glm::vec3 normal;
        float offset;
    };

    [[nodiscard]] inline std::optional<std::pair<glm::vec3, glm::vec3>> clipGuideSegment(
        const glm::vec3& a, const glm::vec3& b, const std::span<const GuideClipPlane> planes) {
        const glm::dvec3 da(a), db(b);
        double enter = 0.0;
        double exit = 1.0;
        for (const auto& plane : planes) {
            // A large crop box can cross the camera plane. Float distances lose
            // the small near-plane offset, and float interpolation then places
            // the endpoint at z=0, discarding the entire edge on projection.
            const double distance_a = glm::dot(glm::dvec3(plane.normal), da) + plane.offset;
            const double distance_b = glm::dot(glm::dvec3(plane.normal), db) + plane.offset;
            if (!std::isfinite(distance_a) || !std::isfinite(distance_b) ||
                (distance_a < 0.0 && distance_b < 0.0)) {
                return std::nullopt;
            }
            if (distance_a < 0.0) {
                enter = std::max(enter, distance_a / (distance_a - distance_b));
            } else if (distance_b < 0.0) {
                exit = std::min(exit, distance_a / (distance_a - distance_b));
            }
        }
        if (enter >= exit) {
            return std::nullopt;
        }
        return std::pair{glm::vec3(glm::mix(da, db, enter)), glm::vec3(glm::mix(da, db, exit))};
    }
} // namespace lfs::vis::gui

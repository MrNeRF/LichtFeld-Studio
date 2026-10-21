/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#pragma once
#include <glm/glm.hpp>
#include <stdexcept>

namespace lfs::vis {
    struct SplatDragAxes {
        int up = -2;

        [[nodiscard]] bool valid() const {
            return up == -2 || up == 2 || up == -3 || up == 3;
        }
        [[nodiscard]] glm::mat4 matrix() const {
            if (!valid())
                throw std::invalid_argument("Splat drag up axis must be -Y, +Y, -Z or +Z");
            const auto vector = [](int axis) {
                glm::vec3 value(0.0f);
                value[(axis < 0 ? -axis : axis) - 1] = axis < 0 ? -1.0f : 1.0f;
                return value;
            };
            const auto up_vector = vector(up);
            // Preserve +X right and rotate around X. This determines forward
            // unambiguously for the four supported up axes.
            const int forward = up == -2 ? 3 : up == 2 ? -3
                                           : up == -3  ? -2
                                                       : 2;
            const auto forward_vector = vector(forward);
            // Source coordinates are RDF: +X right, -Y up, +Z forward.
            // Keep a proper rotation so covariance, quaternions and SH agree.
            glm::mat4 result(1.0f);
            result[0] = glm::vec4(glm::cross(forward_vector, up_vector), 0.0f);
            result[1] = glm::vec4(-up_vector, 0.0f);
            result[2] = glm::vec4(forward_vector, 0.0f);
            return result;
        }
        [[nodiscard]] static SplatDragAxes blender() { return {3}; }
    };
} // namespace lfs::vis

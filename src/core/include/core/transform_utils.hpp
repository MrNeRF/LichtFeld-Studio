/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include <algorithm>
#include <cmath>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_inverse.hpp>
#include <optional>

namespace lfs::core {

    // Return the local transform that preserves `world` under `parent_world`.
    // Reject singular parents and conversions that cannot reproduce the requested world transform.
    [[nodiscard]] inline std::optional<glm::mat4> finiteLocalTransform(
        const glm::mat4& parent_world, const glm::mat4& world) {
        const glm::mat4 parent_inverse = glm::inverse(parent_world);
        for (int column = 0; column < 4; ++column) {
            for (int row = 0; row < 4; ++row) {
                if (!std::isfinite(parent_inverse[column][row]))
                    return std::nullopt;
            }
        }
        glm::mat4 local = parent_inverse * world;
        for (int column = 0; column < 4; ++column) {
            for (int row = 0; row < 4; ++row) {
                if (!std::isfinite(local[column][row]))
                    return std::nullopt;
            }
        }

        const glm::mat4 reconstructed_world = parent_world * local;
        float max_world_magnitude = 0.0f;
        float max_error = 0.0f;
        for (int column = 0; column < 4; ++column) {
            for (int row = 0; row < 3; ++row) {
                const float world_value = world[column][row];
                const float reconstructed_value = reconstructed_world[column][row];
                if (!std::isfinite(world_value) || !std::isfinite(reconstructed_value))
                    return std::nullopt;
                max_world_magnitude = std::max(max_world_magnitude, std::abs(world_value));
                max_error = std::max(max_error, std::abs(reconstructed_value - world_value));
            }
        }
        constexpr float relative_tolerance = 1.0e-4f;
        if (max_error > relative_tolerance * std::max(1.0f, max_world_magnitude))
            return std::nullopt;
        return local;
    }

} // namespace lfs::core

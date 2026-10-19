/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#pragma once

#include "rendering/coordinate_conventions.hpp"
#include <optional>
#include <string_view>

namespace lfs::vis {
    // LegacyY preserves the historical visualizer +Y vertical. DataZ refers
    // to raw dataset +Z, which the existing display boundary maps to -Z.
    enum class NavigationUpAxis { LegacyY,
                                  DataZ };

    inline std::string_view navigationUpAxisName(NavigationUpAxis axis) {
        return axis == NavigationUpAxis::DataZ ? "data_z" : "legacy_y";
    }
    inline std::optional<NavigationUpAxis> navigationUpAxisFromName(std::string_view name) {
        if (name == "legacy_y")
            return NavigationUpAxis::LegacyY;
        if (name == "data_z")
            return NavigationUpAxis::DataZ;
        return std::nullopt;
    }
    inline glm::mat3 navigationFrame(NavigationUpAxis axis) {
        if (axis == NavigationUpAxis::LegacyY)
            return glm::mat3(1.0f);
        const auto up = rendering::visualizerWorldPointFromDataWorld(glm::vec3(0, 0, 1));
        const glm::vec3 right(1, 0, 0);
        return glm::mat3(right, up, glm::cross(right, up));
    }
} // namespace lfs::vis

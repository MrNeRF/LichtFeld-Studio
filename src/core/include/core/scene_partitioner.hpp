/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include "core/export.hpp"
#include <glm/vec3.hpp>
#include <string>
#include <vector>

namespace lfs::core {
    struct PointCloud;

    // Bounds are expressed in a frame rotated counterclockwise about Z by angle_degrees.
    struct SceneBounds {
        glm::vec3 min{0.0f};
        glm::vec3 max{0.0f};
        float angle_degrees = 0.0f;
    };

    struct TileGridConfig {
        int columns = 0;
        int rows = 0;
        float target_size = 0.0f;
        float overlap_ratio = 0.15f;
        bool align_pca = true;
    };

    struct TileSpec {
        glm::vec3 min{0.0f};
        glm::vec3 max{0.0f};
        glm::vec3 rotation_degrees{0.0f};
        int i = 0;
        int j = 0;
        std::string name;
    };

    [[nodiscard]] LFS_CORE_API SceneBounds compute_robust_scene_bounds(
        const PointCloud& pc, const std::vector<glm::vec3>& camera_centers,
        float quantile = 0.02f, float angle_degrees = 0.0f);
    [[nodiscard]] LFS_CORE_API float compute_dominant_orientation_pca(const PointCloud& pc);
    [[nodiscard]] LFS_CORE_API std::vector<TileSpec> generate_tile_grid(
        const SceneBounds& bounds, const TileGridConfig& config);
} // namespace lfs::core

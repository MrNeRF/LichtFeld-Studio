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

    enum class OrientationMode {
        Obb3D,       // Full 3D PCA Oriented Bounding Box
        PlanarYaw,   // Horizontal orientation with vertical locked to Z
        PlanarPitch, // Orientation in the XZ plane around Y
        PlanarRoll,  // Orientation in the YZ plane around X
        AxisAligned  // Axis-aligned bounding box (AABB)
    };

    enum class BoundsSource {
        Both,
        Points,
        Cameras
    };

    struct SceneBounds {
        glm::vec3 min{0.0f};
        glm::vec3 max{0.0f};
        float angle_degrees = 0.0f;
        glm::vec3 rotation_degrees{0.0f};

        SceneBounds() = default;
        SceneBounds(glm::vec3 min_, glm::vec3 max_, float angle_deg = 0.0f)
            : min(min_), max(max_), angle_degrees(angle_deg), rotation_degrees(0.0f, 0.0f, angle_deg) {}
        SceneBounds(glm::vec3 min_, glm::vec3 max_, glm::vec3 rot_deg)
            : min(min_), max(max_), angle_degrees(rot_deg.z), rotation_degrees(rot_deg) {}
    };

    struct TileGridConfig {
        int columns = 0;
        int rows = 0;
        float target_size = 0.0f;
        float overlap_ratio = 0.15f;
        bool align_pca = true;
        OrientationMode orientation_mode = OrientationMode::Obb3D;
        float quantile = 0.02f;
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

    [[nodiscard]] LFS_CORE_API SceneBounds compute_robust_scene_bounds_obb(
        const PointCloud& pc, const std::vector<glm::vec3>& camera_centers,
        float quantile = 0.02f,
        OrientationMode mode = OrientationMode::Obb3D,
        BoundsSource source = BoundsSource::Both);

    [[nodiscard]] LFS_CORE_API float compute_dominant_orientation_pca(const PointCloud& pc);
    [[nodiscard]] LFS_CORE_API glm::vec3 compute_dominant_orientation_3d(
        const std::vector<glm::vec3>& points, float quantile = 0.02f);
    [[nodiscard]] LFS_CORE_API std::vector<TileSpec> generate_tile_grid(
        const SceneBounds& bounds, const TileGridConfig& config);
} // namespace lfs::core

/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include "core/export.hpp"
#include <glm/vec3.hpp>
#include <array>
#include <cstddef>
#include <memory>
#include <string>
#include <vector>

namespace lfs::core {
    struct PointCloud;
    class Camera;

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

    // Snapshot of the SfM evidence needed by the partitioner. The matrix is the
    // row-major world-to-camera transform; no image pixels are loaded.
    struct TileCameraEvidence {
        int uid = -1;
        std::string image_name;
        std::shared_ptr<const Camera> source;
        glm::vec3 center{0.0f};
        std::array<float, 16> world_to_camera{};
        float fx = 0.0f, fy = 0.0f, cx = 0.0f, cy = 0.0f;
        int width = 0, height = 0;
        bool pinhole = false;
        std::vector<glm::vec3> observations;
    };

    // Planning estimates from sampled SfM data. Work units compare tiles under
    // the same training settings; they are not measured seconds or GPU memory.
    struct TilePlanDiagnostics {
        size_t sampled_points = 0;
        size_t sampled_crop_points = 0;
        int supporting_cameras = 0;
        int estimated_selected_cameras = 0;
        int occupied_cells = 0;
        int covered_cells = 0;
        float mean_image_megapixels = 0.0f;
        double estimated_work_units = 0.0;
    };

    struct AdaptiveTileOptions {
        bool all_cameras_for_training = false;
        int resize_factor = -1;
        int max_width = 0;
    };

    [[nodiscard]] LFS_CORE_API std::vector<TileCameraEvidence> collect_tile_camera_evidence(
        const std::vector<std::shared_ptr<Camera>>& cameras);
    [[nodiscard]] LFS_CORE_API std::vector<TileSpec> generate_adaptive_tiles(
        const SceneBounds& bounds, int count, float overlap_ratio,
        const SceneBounds& outer_bounds, const PointCloud& points,
        const std::vector<TileCameraEvidence>& cameras,
        std::vector<TilePlanDiagnostics>* diagnostics = nullptr,
        AdaptiveTileOptions options = {});
    [[nodiscard]] LFS_CORE_API std::vector<int> select_tile_camera_uids(
        const TileSpec& tile, const std::vector<TileCameraEvidence>& cameras);
    [[nodiscard]] LFS_CORE_API std::vector<int> select_tile_initial_point_indices(
        const PointCloud& points, const TileSpec& tile,
        const std::vector<TileCameraEvidence>& selected_cameras);

    [[nodiscard]] LFS_CORE_API SceneBounds compute_robust_scene_bounds(
        const PointCloud& pc, const std::vector<glm::vec3>& camera_centers,
        float quantile = 0.02f, float angle_degrees = 0.0f);

    [[nodiscard]] LFS_CORE_API SceneBounds compute_robust_scene_bounds_obb(
        const PointCloud& pc, const std::vector<glm::vec3>& camera_centers,
        float quantile = 0.02f,
        OrientationMode mode = OrientationMode::Obb3D,
        BoundsSource source = BoundsSource::Both);

    // Uses the partition's orientation, so the outer limits can extend only its boundary cells.
    [[nodiscard]] LFS_CORE_API SceneBounds compute_outer_scene_bounds(
        const PointCloud& pc, const std::vector<glm::vec3>& camera_centers,
        const SceneBounds& partition_bounds, float quantile = 0.02f,
        BoundsSource source = BoundsSource::Both);

    [[nodiscard]] LFS_CORE_API float compute_dominant_orientation_pca(const PointCloud& pc);
    [[nodiscard]] LFS_CORE_API glm::vec3 compute_dominant_orientation_3d(
        const std::vector<glm::vec3>& points, float quantile = 0.02f);
    [[nodiscard]] LFS_CORE_API std::vector<TileSpec> generate_tile_grid(
        const SceneBounds& bounds, const TileGridConfig& config);
    [[nodiscard]] LFS_CORE_API std::vector<TileSpec> generate_tile_grid(
        const SceneBounds& bounds, const TileGridConfig& config,
        const SceneBounds& outer_bounds);
} // namespace lfs::core

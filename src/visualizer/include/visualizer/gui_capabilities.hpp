/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include "core/apriltag_scene_scale.hpp"
#include "core/error.hpp"
#include "core/export.hpp"
#include "core/scene.hpp"

#include <cstddef>
#include <cstdint>
#include <expected>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

#include <glm/glm.hpp>

namespace lfs::vis {
    class RenderingManager;
    class SceneManager;
} // namespace lfs::vis

namespace lfs::vis::cap {

    enum class TransformTargetPolicy : uint8_t {
        AllowEditableSubset,
        RequireAllEditable,
    };

    struct LFS_VIS_API TransformComponents {
        glm::vec3 translation{0.0f};
        glm::vec3 rotation{0.0f};
        glm::vec3 scale{1.0f};
    };

    struct LFS_VIS_API ResolvedTransformTargets {
        std::vector<std::string> node_names;
        glm::vec3 local_center{0.0f};
        glm::vec3 world_center{0.0f};
    };

    struct LFS_VIS_API SelectionSnapshot {
        int64_t selected_count = 0;
        std::vector<int64_t> indices;
        bool truncated = false;
    };

    struct LFS_VIS_API CropBoxUpdate {
        std::optional<glm::vec3> min_bounds;
        std::optional<glm::vec3> max_bounds;
        std::optional<glm::vec3> translation;
        std::optional<glm::vec3> rotation;
        std::optional<glm::vec3> scale;
        bool has_inverse = false;
        bool inverse = false;
        bool has_enabled = false;
        bool enabled = false;
        bool has_show = false;
        bool show = false;
        bool has_use = false;
        bool use = false;
    };

    struct LFS_VIS_API EllipsoidUpdate {
        std::optional<glm::vec3> radii;
        std::optional<glm::vec3> translation;
        std::optional<glm::vec3> rotation;
        std::optional<glm::vec3> scale;
        bool has_inverse = false;
        bool inverse = false;
        bool has_enabled = false;
        bool enabled = false;
        bool has_show = false;
        bool show = false;
        bool has_use = false;
        bool use = false;
    };

    struct LFS_VIS_API CropPolygonUpdate {
        std::optional<float> half_depth;
        bool has_inverse = false;
        bool inverse = false;
        bool has_enabled = false;
        bool enabled = false;
    };

    [[nodiscard]] LFS_VIS_API bool isTransformableNodeType(core::NodeType type);
    // Node types the 3-point align operator can transform (not the broader transformable set).
    [[nodiscard]] LFS_VIS_API bool isAlignTransformTargetType(core::NodeType type);
    [[nodiscard]] LFS_VIS_API TransformComponents decomposeTransform(const glm::mat4& matrix);
    [[nodiscard]] LFS_VIS_API glm::mat4 composeTransform(const TransformComponents& components);

    [[nodiscard]] LFS_VIS_API SelectionSnapshot getSelectionSnapshot(const core::Scene& scene,
                                                                     int max_indices = 100000);
    [[nodiscard]] LFS_VIS_API std::expected<void, std::string> clearGaussianSelection(SceneManager& scene_manager);
    [[nodiscard]] LFS_VIS_API std::expected<void, std::string> clearNodeSelection(SceneManager& scene_manager);
    [[nodiscard]] LFS_VIS_API std::expected<void, std::string> selectNode(SceneManager& scene_manager,
                                                                          const std::string& name,
                                                                          std::string_view mode = "replace");
    [[nodiscard]] LFS_VIS_API std::expected<void, std::string> selectNodes(SceneManager& scene_manager,
                                                                           const std::vector<std::string>& names,
                                                                           std::string_view mode = "replace");

    [[nodiscard]] LFS_VIS_API std::expected<std::vector<std::string>, std::string> resolveTransformTargets(
        const SceneManager& scene_manager,
        const std::optional<std::string>& requested_node);
    [[nodiscard]] LFS_VIS_API std::expected<ResolvedTransformTargets, std::string> resolveEditableTransformSelection(
        const SceneManager& scene_manager,
        const std::optional<std::string>& requested_node,
        TransformTargetPolicy policy = TransformTargetPolicy::AllowEditableSubset);
    [[nodiscard]] LFS_VIS_API std::expected<void, std::string> setTransform(
        SceneManager& scene_manager,
        const std::vector<std::string>& targets,
        const std::optional<glm::vec3>& translation,
        const std::optional<glm::vec3>& rotation,
        const std::optional<glm::vec3>& scale,
        std::string_view undo_label = "transform.set");
    [[nodiscard]] LFS_VIS_API std::expected<void, std::string> setTransformMatrix(
        SceneManager& scene_manager,
        const std::vector<std::string>& targets,
        const glm::mat4& transform,
        std::string_view undo_label = "transform.set_matrix");
    [[nodiscard]] LFS_VIS_API std::expected<void, std::string> translateNodes(
        SceneManager& scene_manager,
        const std::vector<std::string>& targets,
        const glm::vec3& value,
        std::string_view undo_label = "transform.translate");
    [[nodiscard]] LFS_VIS_API std::expected<void, std::string> rotateNodes(
        SceneManager& scene_manager,
        const std::vector<std::string>& targets,
        const glm::vec3& value,
        std::string_view undo_label = "transform.rotate");
    [[nodiscard]] LFS_VIS_API std::expected<void, std::string> scaleNodes(
        SceneManager& scene_manager,
        const std::vector<std::string>& targets,
        const glm::vec3& value,
        std::string_view undo_label = "transform.scale");

    struct LFS_VIS_API AprilTagSceneApplyResult {
        bool applied = false;
        std::string error_message;
        std::vector<std::string> transformed_nodes;
        core::AprilTagSceneScaleReport scale_report;
    };

    [[nodiscard]] LFS_VIS_API AprilTagSceneApplyResult applyUniformScaleAboutDataPivot(
        SceneManager& scene_manager,
        const std::vector<std::string>& requested_targets,
        const glm::vec3& data_world_pivot,
        float scale_factor,
        std::string_view undo_label = "transform.scale_apriltag");

    [[nodiscard]] LFS_VIS_API AprilTagSceneApplyResult applyAprilTagScaleReport(
        SceneManager& scene_manager,
        const core::AprilTagSceneSnapshot& snapshot,
        core::AprilTagSceneScaleReport report,
        const std::vector<std::string>& requested_targets = {});

    [[nodiscard]] LFS_VIS_API AprilTagSceneApplyResult autoScaleSceneWithAprilTags(
        SceneManager& scene_manager,
        std::span<const geometry::apriltag::TagDistanceConstraint> constraints,
        const core::AprilTagSceneScaleOptions& options = {},
        const std::vector<std::string>& requested_targets = {},
        std::string_view undo_label = "transform.scale_apriltag",
        const core::AprilTagProgressCallback& progress_callback = {});

    [[nodiscard]] LFS_VIS_API std::expected<size_t, std::string> bakeNodeTransforms(
        SceneManager& scene_manager,
        const std::vector<std::string>& targets,
        std::string_view undo_label = "transform.bake");
    [[nodiscard]] LFS_VIS_API lfs::Result<void> bakeSplatTransformPreservingStorage(
        core::SplatData& model, const glm::mat4& transform);

    [[nodiscard]] LFS_VIS_API std::expected<void, std::string> writeGaussianField(
        SceneManager& scene_manager,
        RenderingManager* rendering_manager,
        const std::string& node_name,
        std::string_view field_name,
        const std::vector<int>& indices,
        const std::vector<float>& values);

    [[nodiscard]] LFS_VIS_API std::expected<core::NodeId, std::string> resolveCropBoxParentId(
        const SceneManager& scene_manager,
        const std::optional<std::string>& requested_node);
    [[nodiscard]] LFS_VIS_API std::expected<core::NodeId, std::string> resolveCropBoxId(
        const SceneManager& scene_manager,
        const std::optional<std::string>& requested_node);
    [[nodiscard]] LFS_VIS_API std::expected<core::NodeId, std::string> ensureCropBox(
        SceneManager& scene_manager,
        RenderingManager* rendering_manager,
        core::NodeId parent_id);
    [[nodiscard]] LFS_VIS_API std::expected<void, std::string> updateCropBox(
        SceneManager& scene_manager,
        RenderingManager* rendering_manager,
        core::NodeId cropbox_id,
        const CropBoxUpdate& update);
    [[nodiscard]] LFS_VIS_API std::expected<void, std::string> fitCropBoxToParent(
        SceneManager& scene_manager,
        RenderingManager* rendering_manager,
        core::NodeId cropbox_id,
        bool use_percentile);
    [[nodiscard]] LFS_VIS_API std::expected<void, std::string> resetCropBox(
        SceneManager& scene_manager,
        RenderingManager* rendering_manager,
        core::NodeId cropbox_id);

    [[nodiscard]] LFS_VIS_API std::expected<core::NodeId, std::string> resolveEllipsoidParentId(
        const SceneManager& scene_manager,
        const std::optional<std::string>& requested_node);
    [[nodiscard]] LFS_VIS_API std::expected<core::NodeId, std::string> resolveEllipsoidId(
        const SceneManager& scene_manager,
        const std::optional<std::string>& requested_node);
    [[nodiscard]] LFS_VIS_API std::expected<core::NodeId, std::string> ensureEllipsoid(
        SceneManager& scene_manager,
        RenderingManager* rendering_manager,
        core::NodeId parent_id);
    [[nodiscard]] LFS_VIS_API std::expected<void, std::string> updateEllipsoid(
        SceneManager& scene_manager,
        RenderingManager* rendering_manager,
        core::NodeId ellipsoid_id,
        const EllipsoidUpdate& update);
    [[nodiscard]] LFS_VIS_API std::expected<void, std::string> fitEllipsoidToParent(
        SceneManager& scene_manager,
        RenderingManager* rendering_manager,
        core::NodeId ellipsoid_id,
        bool use_percentile);
    [[nodiscard]] LFS_VIS_API std::expected<void, std::string> resetEllipsoid(
        SceneManager& scene_manager,
        RenderingManager* rendering_manager,
        core::NodeId ellipsoid_id);

    // The named crop polygon or the one under the named node; without a name, the selected one.
    [[nodiscard]] LFS_VIS_API lfs::Result<core::NodeId> resolveCropPolygonId(
        const SceneManager& scene_manager,
        const std::optional<std::string>& requested_node);
    // Sets the polygon's depth, inverse or enabled flag in one history entry; the outline is unchanged.
    [[nodiscard]] LFS_VIS_API lfs::Status updateCropPolygon(
        SceneManager& scene_manager,
        RenderingManager* rendering_manager,
        core::NodeId polygon_id,
        const CropPolygonUpdate& update);

    // Keeps the outline and the node's rotation; scales each local axis and moves
    // the prism so its local box is the parent's bounds (the trimmed bounds with
    // use_percentile) seen from the unscaled polygon frame. One history entry.
    [[nodiscard]] LFS_VIS_API lfs::Result<void> fitCropPolygonToParent(
        SceneManager& scene_manager,
        RenderingManager* rendering_manager,
        core::NodeId polygon_id,
        bool use_percentile);
    // Back to the drawn size, scale 1 on every axis, keeping position and rotation.
    [[nodiscard]] LFS_VIS_API lfs::Result<void> resetCropPolygon(
        SceneManager& scene_manager,
        RenderingManager* rendering_manager,
        core::NodeId polygon_id);

    // Replaces a polygon's outline (node-local XY) in one history entry. The outline's
    // box is re-centred on the origin and the node moves to compensate, so the prism
    // stays in place. Rejects fewer than 3 distinct vertices, no area or over 64 vertices.
    [[nodiscard]] LFS_VIS_API lfs::Result<void> setCropPolygonOutline(
        SceneManager& scene_manager,
        RenderingManager* rendering_manager,
        core::NodeId polygon_id,
        std::vector<glm::vec2> vertices);

    // Adds an enabled polygon under a splat in one history entry, replacing the
    // splat's existing crop box, ellipsoid or polygon.
    [[nodiscard]] LFS_VIS_API lfs::Result<core::NodeId> createCropPolygon(
        SceneManager& scene_manager,
        RenderingManager* rendering_manager,
        core::NodeId parent_id,
        core::CropPolygonData data,
        const glm::mat4& local_transform);

} // namespace lfs::vis::cap

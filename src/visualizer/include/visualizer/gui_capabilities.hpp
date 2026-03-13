/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include "core/export.hpp"
#include "core/scene.hpp"

#include <expected>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

#include <glm/glm.hpp>

namespace lfs::vis {
    class RenderingManager;
    class SceneManager;
}

namespace lfs::vis::cap {

    struct LFS_VIS_API TransformComponents {
        glm::vec3 translation{0.0f};
        glm::vec3 rotation{0.0f};
        glm::vec3 scale{1.0f};
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

} // namespace lfs::vis::cap

/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include "core/editor_context.hpp"
#include "core/export.hpp"
#include "core/scene.hpp"
#include <glm/glm.hpp>
#include <glm/gtc/quaternion.hpp>
#include <string>
#include <vector>
#include <imgui.h>

namespace lfs::vis::gui {

    enum class GizmoTargetType {
        Node,
        CropBox,
        Ellipsoid
    };

    struct GizmoTransformContext {
        GizmoTargetType type = GizmoTargetType::Node;
        std::vector<std::string> target_names;

        // Frozen at drag start
        glm::vec3 pivot_world{0.0f};
        glm::vec3 pivot_local{0.0f};

        // Per-target original state captured at drag start
        struct TargetState {
            std::string name;
            glm::mat4 visualizer_world_transform{1.0f};
            glm::mat3 rotation{1.0f};

            glm::vec3 bounds_min{0.0f};
            glm::vec3 bounds_max{0.0f};
            glm::vec3 radii{1.0f};
        };
        std::vector<TargetState> targets;

        // Cumulative tracking - prevents drift by accumulating from original state
        glm::mat3 cumulative_rotation{1.0f};
        glm::vec3 cumulative_scale{1.0f};
        glm::vec3 cumulative_translation{0.0f};

        // Settings at drag start
        bool use_world_space = false;
        PivotMode pivot_mode = PivotMode::Origin;

        bool isActive() const { return !target_names.empty(); }
        void reset();
    };

    namespace gizmo_ops {

        struct LFS_VIS_API NodeLocalTransformResult {
            std::string name;
            glm::mat4 local_transform{1.0f};
        };

        // Matrix decomposition helpers
        glm::mat3 extractRotation(const glm::mat4& m);
        glm::vec3 extractScale(const glm::mat4& m);
        glm::vec3 extractTranslation(const glm::mat4& m);
        void setNodeVisualizerWorldTransform(core::Scene& scene,
                                             const std::string& name,
                                             const glm::mat4& visualizer_world_transform);

        // Multi-node scene graph gizmos operate on selected top-level nodes only.
        // If both a parent and one of its descendants are selected, transforming
        // the parent already carries the child, so the descendant is omitted.
        LFS_VIS_API std::vector<std::string> topLevelTransformTargets(
            const core::Scene& scene,
            const std::vector<std::string>& target_names);

        LFS_VIS_API std::vector<NodeLocalTransformResult> computeNodeSharedSelectionLocalTransforms(
            const core::Scene& scene,
            const std::vector<std::string>& target_names,
            const std::vector<glm::mat4>& original_visualizer_world_transforms,
            const glm::mat4& visualizer_world_delta);

        LFS_VIS_API std::vector<NodeLocalTransformResult> computeNodeIndividualLocalTransforms(
            const core::Scene& scene,
            const std::vector<std::string>& target_names,
            const std::vector<glm::mat4>& original_visualizer_world_transforms,
            const glm::mat4& local_visualizer_delta);

        LFS_VIS_API bool computeCombinedVisualizerWorldBounds(
            const core::Scene& scene,
            const std::vector<std::string>& target_names,
            glm::vec3& out_min,
            glm::vec3& out_max);

        // Compute gizmo display matrix for the custom transform gizmos
        glm::mat4 computeGizmoMatrix(
            const glm::vec3& pivot_world,
            const glm::mat3& rotation,
            const glm::vec3& scale,
            bool use_world_space,
            bool is_scale_operation);

        // Capture context at drag start
        GizmoTransformContext captureCropBox(
            const core::Scene& scene,
            const std::string& name,
            const glm::vec3& pivot_world,
            const glm::vec3& pivot_local,
            TransformSpace space,
            PivotMode pivot_mode);

        GizmoTransformContext captureEllipsoid(
            const core::Scene& scene,
            const std::string& name,
            const glm::vec3& pivot_world,
            const glm::vec3& pivot_local,
            TransformSpace space,
            PivotMode pivot_mode);

        // Apply cumulative transforms - updates scene nodes
        void applyTranslation(
            GizmoTransformContext& ctx,
            core::Scene& scene,
            const glm::vec3& new_pivot_world);

        void applyRotation(
            GizmoTransformContext& ctx,
            core::Scene& scene,
            const glm::mat3& delta_rotation);

        // For cropbox/ellipsoid bounds scaling
        void applyBoundsScale(
            GizmoTransformContext& ctx,
            core::Scene& scene,
            const glm::vec3& new_size);

    } // namespace gizmo_ops

} // namespace lfs::vis::gui

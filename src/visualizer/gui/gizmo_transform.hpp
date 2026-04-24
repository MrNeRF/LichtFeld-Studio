/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include "core/editor_context.hpp"
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

        // Matrix decomposition helpers
        glm::mat3 extractRotation(const glm::mat4& m);
        glm::vec3 extractScale(const glm::mat4& m);
        glm::vec3 extractTranslation(const glm::mat4& m);
        void setNodeVisualizerWorldTransform(core::Scene& scene,
                                             const std::string& name,
                                             const glm::mat4& visualizer_world_transform);

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

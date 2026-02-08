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
#include <ImGuizmo.h>

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
            glm::mat4 local_transform{1.0f};
            glm::vec3 world_position{0.0f};
            glm::mat4 parent_world_inverse{1.0f};
            glm::mat3 rotation{1.0f};
            glm::vec3 scale{1.0f};

            // CropBox specific
            glm::vec3 bounds_min{0.0f};
            glm::vec3 bounds_max{0.0f};

            // Ellipsoid specific
            glm::vec3 radii{1.0f};
        };
        std::vector<TargetState> targets;

        // Cumulative tracking - prevents drift by accumulating from original state
        glm::mat3 cumulative_rotation{1.0f};
        glm::vec3 cumulative_scale{1.0f};
        glm::vec3 cumulative_translation{0.0f};

        // Current operation being performed
        ImGuizmo::OPERATION operation = ImGuizmo::TRANSLATE;

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

        // World-to-local conversion via sandwich product
        // local_delta = parent_rot_inv * world_delta * parent_rot
        glm::mat3 worldToLocalRotation(const glm::mat3& world_delta, const glm::mat4& parent_world_inverse);
        glm::mat3 worldToLocalScale(const glm::vec3& world_scale, const glm::mat4& parent_world_inverse);

        // Compute local pivot based on pivot mode and target type
        glm::vec3 computeLocalPivot(
            const core::Scene& scene,
            core::NodeId target_id,
            PivotMode mode,
            GizmoTargetType type);

        // Compute gizmo display matrix for ImGuizmo
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
            PivotMode pivot_mode,
            ImGuizmo::OPERATION operation);

        GizmoTransformContext captureEllipsoid(
            const core::Scene& scene,
            const std::string& name,
            const glm::vec3& pivot_world,
            const glm::vec3& pivot_local,
            TransformSpace space,
            PivotMode pivot_mode,
            ImGuizmo::OPERATION operation);

        // Apply cumulative transforms - updates scene nodes
        void applyTranslation(
            GizmoTransformContext& ctx,
            core::Scene& scene,
            const glm::vec3& new_pivot_world);

        void applyRotation(
            GizmoTransformContext& ctx,
            core::Scene& scene,
            const glm::mat3& delta_rotation);

        void applyScale(
            GizmoTransformContext& ctx,
            core::Scene& scene,
            const glm::vec3& delta_scale,
            const glm::vec3& new_pivot_world);

        // For cropbox/ellipsoid bounds scaling
        void applyBoundsScale(
            GizmoTransformContext& ctx,
            core::Scene& scene,
            const glm::vec3& new_size);

        // Multi-node capture for transform panel (simpler than GizmoTransformContext)
        struct MultiNodeCapture {
            std::vector<std::string> node_names;
            std::vector<glm::mat4> local_transforms;
            std::vector<glm::vec3> world_positions;
            std::vector<glm::mat4> parent_world_inverses;
            std::vector<glm::mat3> rotations;
            std::vector<glm::vec3> scales;

            bool empty() const { return node_names.empty(); }
            size_t size() const { return node_names.size(); }
        };

        // Capture state for multiple nodes at edit start
        // Filters out nodes whose ancestors are also selected (prevents double-transform)
        MultiNodeCapture captureNodes(
            const core::Scene& scene,
            const std::vector<std::string>& selected_names);

        // Apply cumulative transforms to captured nodes around pivot
        void applyMultiTranslation(
            const MultiNodeCapture& capture,
            core::Scene& scene,
            const glm::vec3& cumulative_delta);

        void applyMultiRotation(
            const MultiNodeCapture& capture,
            core::Scene& scene,
            const glm::mat3& cumulative_rotation,
            const glm::vec3& pivot_world);

        void applyMultiScale(
            const MultiNodeCapture& capture,
            core::Scene& scene,
            const glm::vec3& cumulative_scale,
            const glm::vec3& pivot_world);

    } // namespace gizmo_ops

} // namespace lfs::vis::gui

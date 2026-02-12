/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include "gui/gizmo_transform.hpp"
#include "gui/panel_layout.hpp"
#include "gui/pie_menu.hpp"
#include "gui/ui_context.hpp"
#include <chrono>
#include <glm/glm.hpp>
#include <glm/gtc/quaternion.hpp>
#include <string>
#include <vector>
#include <ImGuizmo.h>

namespace lfs::vis {
    class VisualizerImpl;

    namespace gui {

        class GizmoManager {
        public:
            explicit GizmoManager(VisualizerImpl* viewer);

            void setupEvents();
            void updateToolState(const UIContext& ctx, bool ui_hidden);

            void renderNodeTransformGizmo(const UIContext& ctx, const ViewportLayout& viewport);
            void renderCropBoxGizmo(const UIContext& ctx, const ViewportLayout& viewport);
            void renderEllipsoidGizmo(const UIContext& ctx, const ViewportLayout& viewport);
            void renderCropGizmoMiniToolbar(const UIContext& ctx);
            void renderViewportGizmo(const ViewportLayout& viewport);
            void updateCropFlash();
            void deactivateAllTools();
            void setSelectionSubMode(SelectionSubMode mode);

            [[nodiscard]] TransformSpace getTransformSpace() const { return transform_space_; }
            void setTransformSpace(TransformSpace space) { transform_space_ = space; }
            [[nodiscard]] PivotMode getPivotMode() const { return pivot_mode_; }
            void setPivotMode(PivotMode mode) { pivot_mode_ = mode; }
            [[nodiscard]] ImGuizmo::OPERATION getCurrentOperation() const { return current_operation_; }
            void setCurrentOperation(ImGuizmo::OPERATION op) { current_operation_ = op; }
            [[nodiscard]] SelectionSubMode getSelectionSubMode() const { return selection_mode_; }

            [[nodiscard]] bool isCropboxGizmoActive() const { return cropbox_gizmo_active_; }
            [[nodiscard]] bool isEllipsoidGizmoActive() const { return ellipsoid_gizmo_active_; }
            [[nodiscard]] bool isViewportGizmoDragging() const { return viewport_gizmo_dragging_; }
            [[nodiscard]] bool isPositionInViewportGizmo(double x, double y) const;
            [[nodiscard]] ToolType getCurrentToolMode() const;

            // Pie menu
            void openPieMenu(ImVec2 cursor_pos);
            void closePieMenu();
            void renderPieMenu();
            void onPieMenuKeyRelease();
            void onPieMenuMouseMove(ImVec2 pos);
            void onPieMenuClick(ImVec2 pos);
            [[nodiscard]] bool isPieMenuOpen() const { return pie_menu_.isOpen(); }

        private:
            VisualizerImpl* viewer_;

            // Transform gizmo settings
            ImGuizmo::OPERATION current_operation_ = ImGuizmo::TRANSLATE;
            SelectionSubMode selection_mode_ = SelectionSubMode::Centers;
            TransformSpace transform_space_ = TransformSpace::Local;
            PivotMode pivot_mode_ = PivotMode::Origin;

            // Node transform gizmo
            bool show_node_gizmo_ = false;
            ImGuizmo::OPERATION node_gizmo_operation_ = ImGuizmo::TRANSLATE;
            bool node_gizmo_active_ = false;
            std::vector<std::string> node_gizmo_node_names_;
            std::vector<glm::mat4> node_transforms_before_drag_;
            std::vector<glm::vec3> node_original_world_positions_;
            std::vector<glm::mat4> node_parent_world_inverses_;
            std::vector<glm::mat3> node_original_rotations_;
            std::vector<glm::vec3> node_original_scales_;
            glm::vec3 gizmo_pivot_{0.0f};
            glm::mat3 gizmo_cumulative_rotation_{1.0f};
            glm::vec3 gizmo_cumulative_scale_{1.0f};

            // Cropbox gizmo
            bool cropbox_gizmo_active_ = false;
            std::string cropbox_node_name_;

            // Ellipsoid gizmo
            bool ellipsoid_gizmo_active_ = false;
            std::string ellipsoid_node_name_;

            // Unified gizmo context
            GizmoTransformContext gizmo_context_;

            // Viewport gizmo
            bool viewport_gizmo_dragging_ = false;
            glm::dvec2 gizmo_drag_start_cursor_{0.0, 0.0};
            bool show_viewport_gizmo_ = true;
            static constexpr float VIEWPORT_GIZMO_SIZE = 95.0f;
            static constexpr float VIEWPORT_GIZMO_MARGIN_X = 10.0f;
            static constexpr float VIEWPORT_GIZMO_MARGIN_Y = 10.0f;

            void triggerCropFlash();

            // Crop flash effect
            std::chrono::steady_clock::time_point crop_flash_start_;
            bool crop_flash_active_ = false;

            // Bounds-mode scale gizmo state (single selection only)
            bool node_bounds_scale_active_ = false;
            glm::vec3 node_bounds_min_{0.0f};
            glm::vec3 node_bounds_max_{0.0f};
            glm::vec3 node_bounds_orig_scale_{1.0f};
            glm::mat3 node_bounds_orig_rotation_{1.0f};
            glm::vec3 node_bounds_world_scale_{1.0f};

            // Display cache to avoid per-frame compute_bounds on large splats
            bool node_bounds_cache_valid_ = false;
            core::NodeId node_bounds_cache_node_id_ = core::NULL_NODE;
            glm::vec3 node_bounds_cache_min_{0.0f};
            glm::vec3 node_bounds_cache_max_{0.0f};

            // Axis hover state for gizmo interaction
            bool node_hovered_axis_ = false;
            bool cropbox_hovered_axis_ = false;
            bool ellipsoid_hovered_axis_ = false;

            // Tool tracking
            std::string previous_tool_id_;
            SelectionSubMode previous_selection_mode_ = SelectionSubMode::Centers;

            // Pie menu
            PieMenu pie_menu_;

            void handlePieMenuSelection();
            void handleCropAction(const std::string& action_id);
            void addCropObject(bool is_cropbox);
        };

    } // namespace gui
} // namespace lfs::vis

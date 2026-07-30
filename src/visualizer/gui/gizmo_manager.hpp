/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include "core/export.hpp"
#include "core/scene.hpp"
#include "gui/gizmo_transform.hpp"
#include "gui/panel_layout.hpp"
#include "gui/pie_menu.hpp"
#include "gui/ui_context.hpp"
#include "rendering/rendering_types.hpp"
#include "selection/selection_service.hpp"
#include <chrono>
#include <cstdint>
#include <glm/glm.hpp>
#include <glm/gtc/quaternion.hpp>
#include <memory>
#include <optional>
#include <string>
#include <vector>

namespace lfs::vis {
    class VisualizerImpl;

    namespace gui {

        enum class GizmoOperation {
            Translate,
            Rotate,
            Scale
        };

        // Selection mode is the shared-pivot transform for the current multi-selection;
        // it is unrelated to scene graph GROUP nodes.
        enum class MultiTransformMode {
            Selection = 0,
            Individual = 1
        };

        constexpr MultiTransformMode normalizeMultiTransformMode(const MultiTransformMode mode) {
            return mode == MultiTransformMode::Individual
                       ? MultiTransformMode::Individual
                       : MultiTransformMode::Selection;
        }

        class GizmoManager {
        public:
            explicit GizmoManager(VisualizerImpl* viewer);

            void setupEvents();
            void updateToolState(const UIContext& ctx, bool ui_hidden);

            void renderNodeTransformGizmo(const UIContext& ctx, const ViewportLayout& viewport);
            void renderCropBoxGizmo(const UIContext& ctx, const ViewportLayout& viewport);
            void renderEllipsoidGizmo(const UIContext& ctx, const ViewportLayout& viewport);
            void renderViewportGizmo(const ViewportLayout& viewport);
            void updateCropFlash();
            void deactivateAllTools();
            void setSelectionSubMode(SelectionSubMode mode);
            void setSelectionVolumeFromDrag(SelectionSubMode mode,
                                            SelectionMode apply_mode,
                                            uint64_t source_generation,
                                            const glm::vec3& center_world,
                                            float radius);

            [[nodiscard]] TransformSpace getTransformSpace() const { return transform_space_; }
            void setTransformSpace(TransformSpace space);
            [[nodiscard]] PivotMode getPivotMode() const { return pivot_mode_; }
            void setPivotMode(PivotMode mode);
            [[nodiscard]] MultiTransformMode getMultiTransformMode() const { return multi_transform_mode_; }
            void setMultiTransformMode(MultiTransformMode mode);
            [[nodiscard]] SelectionSubMode getSelectionSubMode() const { return selection_mode_; }

            [[nodiscard]] bool isCropboxGizmoActive() const;
            [[nodiscard]] bool isEllipsoidGizmoActive() const;
            LFS_VIS_API void setCropToolShape(const std::string& shape);
            [[nodiscard]] LFS_VIS_API std::string cropToolShape() const;
            LFS_VIS_API void setCropToolOperation(const std::string& operation);
            [[nodiscard]] LFS_VIS_API std::string cropToolOperation() const;
            LFS_VIS_API void fitActiveCropTool(bool use_percentile);
            LFS_VIS_API void applyActiveCropTool();
            LFS_VIS_API void deleteActiveCropToolVolume();
            [[nodiscard]] bool isViewportGizmoDragging() const { return viewport_gizmo_dragging_; }
            [[nodiscard]] bool isPositionInViewportGizmo(double x, double y) const;
            [[nodiscard]] ToolType getCurrentToolMode() const;

            // Pie menu
            void openPieMenu(glm::vec2 cursor_pos);
            void closePieMenu();
            void renderPieMenu();
            void onPieMenuKeyRelease();
            void onPieMenuMouseMove(glm::vec2 pos);
            void onPieMenuClick(glm::vec2 pos);
            [[nodiscard]] bool isPieMenuOpen() const { return pie_menu_.isOpen(); }

        private:
            struct ToolStateStamp {
                bool valid = false;
                bool ui_hidden = false;
                bool has_scene_manager = false;
                bool has_selected_node = false;
                const void* align_tool = nullptr;
                const void* selection_tool = nullptr;
                const void* rendering_manager = nullptr;
                std::string active_tool_id;
                std::string gizmo_type;
                std::string selected_node_name;
                SelectionSubMode selection_mode = SelectionSubMode::Centers;

                bool operator==(const ToolStateStamp&) const = default;
            };

            VisualizerImpl* viewer_;
            [[nodiscard]] std::optional<core::NodeId> selectedCropVolumeNodeId(core::NodeId target_id) const;
            [[nodiscard]] bool ensureCropVolumeForCurrentSelection();

            // Transform gizmo settings
            GizmoOperation current_operation_ = GizmoOperation::Translate;
            SelectionSubMode selection_mode_ = SelectionSubMode::Centers;
            TransformSpace transform_space_ = TransformSpace::Local;
            PivotMode pivot_mode_ = PivotMode::Origin;
            MultiTransformMode multi_transform_mode_ = MultiTransformMode::Selection;

            // Node transform gizmo
            bool show_node_gizmo_ = false;
            GizmoOperation node_gizmo_operation_ = GizmoOperation::Translate;
            bool node_gizmo_active_ = false;
            std::vector<std::string> node_gizmo_node_names_;
            std::vector<glm::mat4> node_transforms_before_drag_;
            std::vector<glm::mat4> node_original_visualizer_world_transforms_;
            glm::vec3 gizmo_pivot_{0.0f};
            glm::mat3 gizmo_cumulative_rotation_{1.0f};
            glm::vec3 gizmo_cumulative_scale_{1.0f};

            // Cropbox gizmo
            bool cropbox_gizmo_active_ = false;
            std::string cropbox_node_name_;
            core::CropBoxData cropbox_data_before_drag_;
            glm::mat4 cropbox_transform_before_drag_{1.0f};

            // Ellipsoid gizmo
            bool ellipsoid_gizmo_active_ = false;
            std::string ellipsoid_node_name_;
            core::EllipsoidData ellipsoid_data_before_drag_;
            glm::mat4 ellipsoid_transform_before_drag_{1.0f};

            enum class CropToolShape {
                Box,
                Ellipsoid,
            };

            CropToolShape crop_tool_shape_ = CropToolShape::Box;
            bool crop_tool_initialized_ = false;
            core::NodeId crop_tool_target_node_id_ = core::NULL_NODE;
            core::NodeId crop_tool_volume_node_id_ = core::NULL_NODE;
            bool crop_tool_drag_active_ = false;
            bool crop_tool_drag_changed_ = false;
            glm::vec3 crop_tool_box_min_{-0.5f};
            glm::vec3 crop_tool_box_max_{0.5f};
            glm::vec3 crop_tool_ellipsoid_radii_{1.0f};
            glm::mat4 crop_tool_visualizer_transform_{1.0f};
            SelectionMode selection_volume_apply_mode_ = SelectionMode::Replace;
            uint64_t selection_volume_source_generation_ = 0;
            std::shared_ptr<core::Tensor> selection_volume_base_mask_;
            core::Scene::SelectionStateSnapshot selection_volume_selection_before_drag_;
            bool selection_volume_gizmo_active_ = false;
            bool selection_volume_drag_changed_ = false;

            // Unified gizmo context
            GizmoTransformContext gizmo_context_;

            // Viewport gizmo
            bool viewport_gizmo_dragging_ = false;
            SplitViewPanelId viewport_gizmo_active_panel_ = SplitViewPanelId::Left;
            glm::dvec2 gizmo_drag_start_cursor_{0.0, 0.0};
            bool show_viewport_gizmo_ = true;
            static constexpr float VIEWPORT_GIZMO_SIZE = 95.0f;
            static constexpr float VIEWPORT_GIZMO_MARGIN_X = 10.0f;
            static constexpr float VIEWPORT_GIZMO_MARGIN_Y = 10.0f;

            void triggerCropFlash();

            // Crop flash effect
            std::chrono::steady_clock::time_point crop_flash_start_;
            bool crop_flash_active_ = false;

            // Bounds-mode scale gizmo state
            bool node_bounds_scale_active_ = false;
            glm::vec3 node_bounds_min_{0.0f};
            glm::vec3 node_bounds_max_{0.0f};
            glm::mat4 node_bounds_orig_visualizer_world_transform_{1.0f};
            glm::vec3 node_bounds_world_scale_{1.0f};
            bool node_selection_bounds_scale_active_ = false;
            glm::vec3 node_selection_bounds_min_{0.0f};
            glm::vec3 node_selection_bounds_max_{0.0f};

            // Display cache to avoid per-frame compute_bounds on large splats
            bool node_bounds_cache_valid_ = false;
            core::NodeId node_bounds_cache_node_id_ = core::NULL_NODE;
            glm::vec3 node_bounds_cache_min_{0.0f};
            glm::vec3 node_bounds_cache_max_{0.0f};
            bool node_selection_bounds_cache_valid_ = false;
            std::vector<core::NodeId> node_selection_bounds_cache_node_ids_;
            std::vector<glm::mat4> node_selection_bounds_cache_visualizer_world_transforms_;
            glm::vec3 node_selection_bounds_cache_min_{0.0f};
            glm::vec3 node_selection_bounds_cache_max_{0.0f};

            // Tool tracking
            SelectionSubMode previous_selection_mode_ = SelectionSubMode::Centers;
            ToolStateStamp last_tool_state_stamp_;

            // Pie menu
            PieMenu pie_menu_;

            void handlePieMenuSelection();
            void handleCropAction(const std::string& action_id);
            [[nodiscard]] bool isCropToolActive() const;
            [[nodiscard]] bool isSelectionVolumeMode() const;
            [[nodiscard]] bool isVolumeGizmoToolActive() const;
            [[nodiscard]] std::optional<core::NodeId> selectedCropTargetNodeId() const;
            [[nodiscard]] bool ensureCropToolState();
            [[nodiscard]] bool computeCropToolTargetBounds(core::NodeId target_id,
                                                           bool use_percentile,
                                                           glm::vec3& bounds_min,
                                                           glm::vec3& bounds_max) const;
            void setCropToolBounds(core::NodeId target_id,
                                   const glm::vec3& bounds_min,
                                   const glm::vec3& bounds_max);
            [[nodiscard]] bool syncCropToolStateFromNode(core::NodeId target_id, core::NodeId volume_node_id, bool* changed = nullptr);
            [[nodiscard]] bool persistActiveCropToolToNode(bool enable);
            void updateCropToolOverlayState();
            void clearCropToolOverlayState();
            void leaveCropTool(bool hide_volume, bool select_parent, bool clear_active_tool);
            void deleteCropVolumeAfterApply(core::NodeId volume_node_id);
            void clearSelectionVolumeState();
            void captureSelectionVolumeBase(uint64_t source_generation);
            [[nodiscard]] bool applySelectionVolumeFromGizmo(bool push_undo);
            void beginSelectionVolumeGizmoDrag();
            void finishSelectionVolumeGizmoDrag();
            void renderCropToolBoxGizmo(const UIContext& ctx, const ViewportLayout& viewport);
            void renderCropToolEllipsoidGizmo(const UIContext& ctx, const ViewportLayout& viewport);
        };

    } // namespace gui
} // namespace lfs::vis

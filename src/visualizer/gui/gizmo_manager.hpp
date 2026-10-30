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
    class CropVolumeSourceTest_GizmoEditsAreSavedAsSceneState_Test;

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
            LFS_VIS_API explicit GizmoManager(VisualizerImpl* viewer);

            LFS_VIS_API void setupEvents();
            LFS_VIS_API void updateToolState(const UIContext& ctx, bool ui_hidden);

            LFS_VIS_API void renderNodeTransformGizmo(const UIContext& ctx, const ViewportLayout& viewport);
            LFS_VIS_API void renderCropBoxGizmo(const UIContext& ctx, const ViewportLayout& viewport);
            LFS_VIS_API void renderEllipsoidGizmo(const UIContext& ctx, const ViewportLayout& viewport);
            LFS_VIS_API void renderViewportGizmo(const ViewportLayout& viewport);
            void updateCropFlash();
            LFS_VIS_API void deactivateAllTools();
            bool cancelActiveNodeTransformDrag();
            bool cancelActiveGizmoDrag();
            LFS_VIS_API void setSelectionSubMode(SelectionSubMode mode);
            LFS_VIS_API void setSelectionVolumeFromDrag(SelectionSubMode mode,
                                                        SelectionMode apply_mode,
                                                        uint64_t source_generation,
                                                        const glm::vec3& center_world,
                                                        float radius);

            [[nodiscard]] TransformSpace getTransformSpace() const { return transform_space_; }
            LFS_VIS_API void setTransformSpace(TransformSpace space);
            [[nodiscard]] GizmoOperation getOperation() const {
                return current_operation_;
            }
            void setOperation(GizmoOperation operation) {
                current_operation_ = operation;
                node_gizmo_operation_ = operation;
            }
            [[nodiscard]] PivotMode getPivotMode() const { return pivot_mode_; }
            LFS_VIS_API void setPivotMode(PivotMode mode);
            [[nodiscard]] MultiTransformMode getMultiTransformMode() const { return multi_transform_mode_; }
            void setMultiTransformMode(MultiTransformMode mode);
            [[nodiscard]] SelectionSubMode getSelectionSubMode() const { return selection_mode_; }

            [[nodiscard]] bool isCropboxGizmoActive() const;
            [[nodiscard]] bool isEllipsoidGizmoActive() const;
            [[nodiscard]] bool isCropPolygonGizmoActive() const;
            LFS_VIS_API void setCropToolShape(const std::string& shape);
            [[nodiscard]] LFS_VIS_API std::string cropToolShape() const;
            LFS_VIS_API void setCropToolOperation(const std::string& operation);
            [[nodiscard]] bool ensureCropToolStateForRestore() {
                return ensureCropToolState();
            }
            [[nodiscard]] LFS_VIS_API std::string cropToolOperation() const;
            LFS_VIS_API void finishCropTool();
            LFS_VIS_API void editCropTool(const core::Uuid& uuid);
            LFS_VIS_API bool changeCropToolShape(const std::string& shape);
            [[nodiscard]] LFS_VIS_API bool isCropToolShapeLocked() const;
            LFS_VIS_API void newCropTool(const std::string& shape);
            LFS_VIS_API void applyActiveCropTool();
            LFS_VIS_API void deleteActiveCropToolVolume();
            // Polygon button: edit the selected splat's or point cloud's polygon, or start drawing one.
            LFS_VIS_API void activateCropPolygonTool();
            // Fit or reset the selected crop polygon, or the selected splat's or point cloud's.
            LFS_VIS_API void fitSelectedCropPolygon(bool use_percentile);
            LFS_VIS_API void resetSelectedCropPolygon();
            // Invert the selected crop polygon, else crop box, else ellipsoid, in one undo step.
            LFS_VIS_API void toggleSelectedCropInverse();
            // Whether that crop volume is inverted; empty without one.
            [[nodiscard]] LFS_VIS_API std::optional<bool> selectedCropInverse() const;
            [[nodiscard]] bool isViewportGizmoDragging() const { return viewport_gizmo_dragging_; }
            // The panel whose axis widget is pressed or dragged, and the pressed marker (-1 for the sphere).
            struct ViewportGizmoPress {
                SplitViewPanelId panel = SplitViewPanelId::Left;
                int axis = -1;
            };
            [[nodiscard]] std::optional<ViewportGizmoPress> viewportGizmoPress() const;
            // What the axis widget shows under the pointer: -1 off every sphere,
            // otherwise the panel and hovered marker, so a change needs a redraw.
            [[nodiscard]] LFS_VIS_API int viewportGizmoHover(double x, double y) const;
            [[nodiscard]] bool isPositionInViewportGizmo(double x, double y) const;
            [[nodiscard]] ToolType getCurrentToolMode() const;

            // Pie menu
            LFS_VIS_API void openPieMenu(glm::vec2 cursor_pos);
            void closePieMenu();
            void renderPieMenu();
            void onPieMenuKeyRelease();
            LFS_VIS_API void onPieMenuMouseMove(glm::vec2 pos);
            LFS_VIS_API void onPieMenuClick(glm::vec2 pos);
            [[nodiscard]] bool isPieMenuOpen() const { return pie_menu_.isOpen(); }

        private:
            friend class lfs::vis::CropVolumeSourceTest_GizmoEditsAreSavedAsSceneState_Test;
            struct ToolStateStamp {
                bool valid = false;
                bool ui_hidden = false;
                bool has_scene_manager = false;
                bool has_selected_node = false;
                const void* align_tool = nullptr;
                const void* scale_distance_tool = nullptr;
                const void* selection_tool = nullptr;
                const void* rendering_manager = nullptr;
                std::string active_tool_id;
                std::string gizmo_type;
                std::string selected_node_name;
                SelectionSubMode selection_mode = SelectionSubMode::Centers;

                bool operator==(const ToolStateStamp&) const = default;
            };

            VisualizerImpl* viewer_;
            void beginViewportGizmoOrbit(::Viewport& viewport, const glm::vec2& mouse, float time);
            void orbitViewportGizmo(::Viewport& viewport, float time);
            void releaseViewportGizmoPointer();
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

            enum class CropToolShape {
                Box,
                Ellipsoid,
                Polygon,
            };

            CropToolShape crop_tool_shape_ = CropToolShape::Box;
            bool crop_tool_initialized_ = false;
            bool crop_tool_shape_locked_ = false;
            core::NodeId crop_tool_target_node_id_ = core::NULL_NODE;
            core::NodeId crop_tool_volume_node_id_ = core::NULL_NODE;
            bool crop_tool_drag_active_ = false;
            bool crop_tool_drag_changed_ = false;
            glm::vec3 crop_tool_box_min_before_drag_{-0.5f};
            glm::vec3 crop_tool_box_max_before_drag_{0.5f};
            glm::vec3 crop_tool_ellipsoid_radii_before_drag_{1.0f};
            glm::mat4 crop_tool_visualizer_transform_before_drag_{1.0f};
            glm::vec3 crop_tool_box_min_{-0.5f};
            glm::vec3 crop_tool_box_max_{0.5f};
            glm::vec3 crop_tool_ellipsoid_radii_{1.0f};
            glm::mat4 crop_tool_visualizer_transform_{1.0f};
            // The polygon's gizmos edit the transform; its vertex handles edit the outline.
            std::vector<glm::vec2> crop_tool_polygon_vertices_;
            float crop_tool_polygon_half_depth_ = 1.0f;
            glm::vec3 crop_tool_polygon_half_extents_{1.0f};
            // A vertex dragged on the cap plane z = plane_z. Vertices are node-local
            // as at drag start; the node changes once, on release.
            struct CropPolygonVertexDrag {
                size_t index = 0;
                float plane_z = 0.0f;
                glm::mat4 polygon_to_world{1.0f};
                std::vector<glm::vec2> vertices;
                bool moved = false;
            };
            std::optional<CropPolygonVertexDrag> crop_polygon_vertex_drag_;
            std::optional<glm::mat4> crop_tool_scale_start_;
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
            bool viewport_gizmo_pressed_ = false;
            int viewport_gizmo_pressed_axis_ = -1;
            glm::vec2 viewport_gizmo_press_pos_{0.0f};
            // Orbit input accumulates relative motion, so the window edge never stops a drag.
            glm::vec2 viewport_gizmo_drag_pos_{0.0f};
            SplitViewPanelId viewport_gizmo_active_panel_ = SplitViewPanelId::Left;
            glm::dvec2 gizmo_drag_start_cursor_{0.0, 0.0};
            glm::vec3 viewport_gizmo_camera_position_before_drag_{0.0f};
            glm::vec3 viewport_gizmo_camera_pivot_before_drag_{0.0f};
            glm::mat3 viewport_gizmo_camera_rotation_before_drag_{1.0f};
            bool viewport_gizmo_camera_snapshot_valid_ = false;
            bool show_viewport_gizmo_ = true;

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

            // Vertex handles of the polygon crop tool, on the cap facing the camera.
            struct CropPolygonHandles {
                std::vector<std::optional<glm::vec2>> screen;
                int hot = -1; // hovered or dragged vertex
                bool owns_pointer = false;
            };
            // Drags, inserts and removes outline vertices from this frame's pointer input.
            [[nodiscard]] CropPolygonHandles updateCropPolygonVertexEdit(glm::vec2 panel_pos, glm::vec2 panel_size,
                                                                         const glm::mat4& view,
                                                                         const glm::mat4& projection);
            void previewCropPolygonOutline(const std::vector<glm::vec2>& vertices, const glm::mat4& polygon_to_world);
            bool commitCropPolygonOutline(std::vector<glm::vec2> vertices);
            [[nodiscard]] bool isCropToolActive() const;
            [[nodiscard]] bool isSelectionVolumeMode() const;
            [[nodiscard]] bool isVolumeGizmoToolActive() const;
            [[nodiscard]] std::optional<core::NodeId> selectedCropTargetNodeId() const;
            [[nodiscard]] LFS_VIS_API bool ensureCropToolState();
            bool syncSelectedCropShape();
            [[nodiscard]] bool syncCropToolStateFromNode(core::NodeId target_id, core::NodeId volume_node_id, bool* changed = nullptr);
            [[nodiscard]] LFS_VIS_API bool persistActiveCropToolToNode(bool enable);
            void scaleCropToolTransform(const glm::vec3& total_scale);
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

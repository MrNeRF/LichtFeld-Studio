/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "gui/gizmo_manager.hpp"
#include "core/error_bus.hpp"
#include "core/events.hpp"
#include "core/logger.hpp"
#include "core/scene.hpp"
#include "core/splat_data_transform.hpp"
#include "core/splat_operations.hpp"
#include "gui/bounds_gizmo.hpp"
#include "gui/gui_focus_state.hpp"
#include "gui/gui_manager.hpp"
#include "gui/rotation_gizmo.hpp"
#include "gui/scale_gizmo.hpp"
#include "gui/translation_gizmo.hpp"
#include "gui/ui_widgets.hpp"
#include "gui/viewport_gizmo_geometry.hpp"
#include "input/input_controller.hpp"
#include "input/sdl_key_mapping.hpp"
#include "operation/undo_entry.hpp"
#include "operation/undo_history.hpp"
#include "operator/operator_id.hpp"
#include "operator/operator_registry.hpp"
#include "operator/ops/crop_polygon_ops.hpp"
#include "python/python_runtime.hpp"
#include "rendering/coordinate_conventions.hpp"
#include "rendering/rendering.hpp"
#include "rendering/rendering_manager.hpp"
#include "scene/scene_manager.hpp"
#include "tools/align_tool.hpp"
#include "tools/scale_distance_tool.hpp"
#include "tools/selection_tool.hpp"
#include "tools/unified_tool_registry.hpp"
#include "visualizer/app_store.hpp"
#include "visualizer/gui_capabilities.hpp"
#include "visualizer/scene_coordinate_utils.hpp"
#include "visualizer_impl.hpp"
#include <SDL3/SDL.h>
#include <algorithm>
#include <array>
#include <cassert>
#include <cmath>
#include <glm/gtc/matrix_transform.hpp>
#include <limits>
#include <optional>
#include <vector>

namespace lfs::vis::gui {

    using ToolType = lfs::vis::ToolType;

    namespace {
        void setPointPickToolsEnabled(ToolBase* align, ToolBase* scale_distance,
                                      const bool enable_align = false, const bool enable_scale_distance = false) {
            if (align && !enable_align)
                align->setEnabled(false);
            if (scale_distance && !enable_scale_distance)
                scale_distance->setEnabled(false);
            if (align && enable_align)
                align->setEnabled(true);
            if (scale_distance && enable_scale_distance)
                scale_distance->setEnabled(true);
        }

        [[nodiscard]] lfs::vis::SelectionPreviewMode toSelectionPreviewMode(const SelectionSubMode mode) {
            switch (mode) {
            case SelectionSubMode::Rectangle: return lfs::vis::SelectionPreviewMode::Rectangle;
            case SelectionSubMode::Polygon: return lfs::vis::SelectionPreviewMode::Polygon;
            case SelectionSubMode::Lasso: return lfs::vis::SelectionPreviewMode::Lasso;
            case SelectionSubMode::Rings: return lfs::vis::SelectionPreviewMode::Rings;
            case SelectionSubMode::Box: return lfs::vis::SelectionPreviewMode::Box;
            case SelectionSubMode::Sphere: return lfs::vis::SelectionPreviewMode::Sphere;
            case SelectionSubMode::Color: return lfs::vis::SelectionPreviewMode::Color;
            case SelectionSubMode::Centers:
            default: return lfs::vis::SelectionPreviewMode::Centers;
            }
        }

        [[nodiscard]] bool isSelectionVolumeSubMode(const SelectionSubMode mode) {
            return mode == SelectionSubMode::Box || mode == SelectionSubMode::Sphere;
        }

        [[nodiscard]] bool isCropVolumeType(const core::NodeType type) {
            return type == core::NodeType::CROPBOX || type == core::NodeType::ELLIPSOID ||
                   type == core::NodeType::CROP_POLYGON;
        }

        void emitCropPolygon(const core::Scene& scene, const core::SceneNode& node) {
            const auto& polygon = *node.crop_polygon;
            lfs::core::events::cmd::CropPLYPolygon{
                .world_transform = scene_coords::nodeDataWorldTransform(scene, node.id),
                .vertices = polygon.vertices,
                .half_depth = polygon.half_depth,
                .inverse = polygon.inverse,
                .target_node_id = node.parent_id}
                .emit();
        }

        [[nodiscard]] const char* selectionSubModeId(const SelectionSubMode mode) {
            switch (mode) {
            case SelectionSubMode::Centers: return "centers";
            case SelectionSubMode::Rectangle: return "rectangle";
            case SelectionSubMode::Polygon: return "polygon";
            case SelectionSubMode::Lasso: return "lasso";
            case SelectionSubMode::Rings: return "rings";
            case SelectionSubMode::Color: return "color";
            case SelectionSubMode::Box: return "box";
            case SelectionSubMode::Sphere: return "sphere";
            }
            return "centers";
        }

        struct ViewportGizmoPanelTarget {
            SplitViewPanelId panel = SplitViewPanelId::Left;
            Viewport* viewport = nullptr;
            glm::vec2 pos{0.0f};
            glm::vec2 size{0.0f};

            [[nodiscard]] bool valid() const {
                return viewport != nullptr && size.x > 0.0f && size.y > 0.0f;
            }
        };

        constexpr int NODE_GIZMO_ID_BASE = 100;
        constexpr int CROPBOX_GIZMO_ID_BASE = 200;
        constexpr int ELLIPSOID_GIZMO_ID_BASE = 300;

        [[nodiscard]] int panelGizmoId(const int base, const SplitViewPanelId panel) {
            return base + (panel == SplitViewPanelId::Right ? 1 : 0);
        }

        [[nodiscard]] NativeGizmoInput nativeGizmoInputFromFrame(const lfs::vis::FrameInputBuffer& frame_input) {
            return {
                .mouse_pos = {frame_input.mouse_x, frame_input.mouse_y},
                .mouse_left_down = frame_input.mouse_down[0],
                .mouse_left_clicked = frame_input.mouse_clicked[0],
            };
        }

        [[nodiscard]] bool nativeControlModifierDown(const lfs::vis::FrameInputBuffer& frame_input) {
            return (frame_input.key_mods & SDL_KMOD_CTRL) != 0;
        }

        constexpr float CROP_POLYGON_HANDLE_RADIUS_PX = 5.0f;
        constexpr OverlayColor CROP_POLYGON_HANDLE_COLOR = overlayColor(255, 214, 10, 235);
        constexpr OverlayColor CROP_POLYGON_HANDLE_HOT_COLOR = overlayColor(255, 128, 16, 245);
        constexpr OverlayColor CROP_POLYGON_HANDLE_OUTLINE = overlayColor(0, 0, 0, 160);

        // Projects like the native gizmos: visualizer world to panel pixels.
        [[nodiscard]] std::optional<glm::vec2> projectToPanel(const glm::mat4& view_projection, const glm::vec2 panel_pos,
                                                              const glm::vec2 panel_size, const glm::vec3& world) {
            const glm::vec4 clip = view_projection * glm::vec4(world, 1.0f);
            if (!(clip.w > std::numeric_limits<float>::epsilon()))
                return std::nullopt;
            const glm::vec2 ndc = glm::vec2(clip) / clip.w;
            if (!std::isfinite(ndc.x) || !std::isfinite(ndc.y))
                return std::nullopt;
            return glm::vec2(panel_pos.x + (ndc.x * 0.5f + 0.5f) * panel_size.x,
                             panel_pos.y + (-ndc.y * 0.5f + 0.5f) * panel_size.y);
        }

        // The polygon-local XY point under a panel pixel on the plane z = plane_z.
        [[nodiscard]] std::optional<glm::vec2> pixelOnPolygonPlane(const glm::mat4& view_projection, const glm::vec2 panel_pos,
                                                                   const glm::vec2 panel_size,
                                                                   const glm::mat4& polygon_to_world,
                                                                   const float plane_z, const glm::vec2 pixel) {
            const glm::mat4 local_to_clip = view_projection * polygon_to_world;
            const glm::mat4 clip_to_local = glm::inverse(local_to_clip);
            const glm::vec2 ndc((pixel.x - panel_pos.x) / panel_size.x * 2.0f - 1.0f,
                                1.0f - (pixel.y - panel_pos.y) / panel_size.y * 2.0f);
            // Any two depths give the pixel's ray, in perspective and orthographic views alike.
            const auto unproject = [&](const float depth) {
                const glm::vec4 point = clip_to_local * glm::vec4(ndc, depth, 1.0f);
                return glm::vec3(point) / point.w;
            };
            const glm::vec3 near_point = unproject(0.25f);
            const glm::vec3 far_point = unproject(0.75f);
            const float dz = far_point.z - near_point.z;
            if (!std::isfinite(dz) || std::abs(dz) < 1e-12f)
                return std::nullopt;
            const glm::vec3 hit = glm::mix(near_point, far_point, (plane_z - near_point.z) / dz);
            if (!std::isfinite(hit.x) || !std::isfinite(hit.y) || !((local_to_clip * glm::vec4(hit, 1.0f)).w > 0.0f))
                return std::nullopt;
            return glm::vec2(hit);
        }

        [[nodiscard]] float distanceSquaredToSegment(const glm::vec2 point, const glm::vec2 a, const glm::vec2 b,
                                                     float* const t_out) {
            const glm::vec2 ab = b - a;
            const float length_sq = glm::dot(ab, ab);
            const float t = length_sq > 0.0f ? std::clamp(glm::dot(point - a, ab) / length_sq, 0.0f, 1.0f) : 0.0f;
            *t_out = t;
            const glm::vec2 delta = point - (a + ab * t);
            return glm::dot(delta, delta);
        }

        // Screen-space hits on the projected outline, with the selection polygon's radii.
        [[nodiscard]] int findOutlineVertexAt(const std::vector<std::optional<glm::vec2>>& points, const glm::vec2 cursor) {
            const float radius_sq = POLYGON_VERTEX_HIT_RADIUS_PX * POLYGON_VERTEX_HIT_RADIUS_PX;
            for (size_t i = 0; i < points.size(); ++i) {
                if (points[i] && glm::dot(cursor - *points[i], cursor - *points[i]) <= radius_sq)
                    return static_cast<int>(i);
            }
            return -1;
        }

        [[nodiscard]] int findOutlineEdgeAt(const std::vector<std::optional<glm::vec2>>& points, const glm::vec2 cursor) {
            const float vertex_radius_sq = POLYGON_VERTEX_HIT_RADIUS_PX * POLYGON_VERTEX_HIT_RADIUS_PX;
            float best_distance_sq = POLYGON_EDGE_HIT_RADIUS_PX * POLYGON_EDGE_HIT_RADIUS_PX;
            int best_edge = -1;
            for (size_t i = 0; i < points.size(); ++i) {
                const auto& start = points[i];
                const auto& end = points[(i + 1) % points.size()];
                if (!start || !end || glm::dot(cursor - *start, cursor - *start) <= vertex_radius_sq ||
                    glm::dot(cursor - *end, cursor - *end) <= vertex_radius_sq)
                    continue;
                float t = 0.0f;
                const float distance_sq = distanceSquaredToSegment(cursor, *start, *end, &t);
                if (distance_sq <= best_distance_sq && t > 0.0f && t < 1.0f) {
                    best_distance_sq = distance_sq;
                    best_edge = static_cast<int>(i);
                }
            }
            return best_edge;
        }

        // Gizmo hover is refreshed only while that gizmo is drawn. A tool change can stop
        // drawing the hovered gizmo, and its stale hover would keep claiming viewport presses.
        void resetIdleGizmoHover() {
            if (!isTranslationGizmoActive())
                cancelTranslationGizmoDrag();
            if (!isRotationGizmoActive())
                cancelRotationGizmoDrag();
            if (!isScaleGizmoActive())
                cancelScaleGizmoDrag();
            if (!isBoundsGizmoActive())
                cancelBoundsGizmoDrag();
        }

        [[nodiscard]] float viewportGizmoUiScale() {
            return std::max(1.0f, lfs::python::get_shared_dpi_scale());
        }

        [[nodiscard]] std::optional<ViewportGizmoLayout> panelGizmoLayout(const ViewportGizmoPanelTarget& panel) {
            if (!panel.valid()) {
                return std::nullopt;
            }
            return buildViewportGizmoLayout(panel.viewport->getRotationMatrix(), panel.pos, panel.size,
                                            viewportGizmoUiScale());
        }

        [[nodiscard]] std::vector<ViewportGizmoPanelTarget> collectViewportGizmoPanels(
            VisualizerImpl* const viewer,
            const glm::vec2& viewport_pos,
            const glm::vec2& viewport_size) {
            std::vector<ViewportGizmoPanelTarget> panels;
            if (!viewer || viewport_size.x <= 0.0f || viewport_size.y <= 0.0f) {
                return panels;
            }

            auto* const rendering_manager = viewer->getRenderingManager();
            if (!rendering_manager || !rendering_manager->isIndependentSplitViewActive()) {
                panels.push_back({
                    .panel = SplitViewPanelId::Left,
                    .viewport = &viewer->getViewport(),
                    .pos = viewport_pos,
                    .size = viewport_size,
                });
                return panels;
            }

            if (const auto left_panel = rendering_manager->resolveViewerPanel(
                    viewer->getViewport(),
                    viewport_pos, viewport_size, std::nullopt, SplitViewPanelId::Left);
                left_panel && left_panel->valid()) {
                panels.push_back(ViewportGizmoPanelTarget{
                    .panel = SplitViewPanelId::Left,
                    .viewport = left_panel->viewport,
                    .pos = {left_panel->x, left_panel->y},
                    .size = {left_panel->width, left_panel->height},
                });
            }

            if (const auto right_panel = rendering_manager->resolveViewerPanel(
                    viewer->getViewport(),
                    viewport_pos, viewport_size, std::nullopt, SplitViewPanelId::Right);
                right_panel && right_panel->valid()) {
                panels.push_back(ViewportGizmoPanelTarget{
                    .panel = SplitViewPanelId::Right,
                    .viewport = right_panel->viewport,
                    .pos = {right_panel->x, right_panel->y},
                    .size = {right_panel->width, right_panel->height},
                });
            }

            if (panels.empty()) {
                panels.push_back({
                    .panel = SplitViewPanelId::Left,
                    .viewport = &viewer->getViewport(),
                    .pos = viewport_pos,
                    .size = viewport_size,
                });
            }

            return panels;
        }

        [[nodiscard]] std::optional<ViewportGizmoPanelTarget> resolveActiveGizmoPanel(
            VisualizerImpl* const viewer,
            const ViewportLayout& viewport) {
            const auto panels = collectViewportGizmoPanels(
                viewer,
                {viewport.pos.x, viewport.pos.y},
                {viewport.size.x, viewport.size.y});
            if (panels.empty()) {
                return std::nullopt;
            }

            auto* const rendering_manager = viewer ? viewer->getRenderingManager() : nullptr;
            if (!rendering_manager || !rendering_manager->isIndependentSplitViewActive()) {
                return panels.front();
            }

            const auto focused_panel = rendering_manager->getFocusedSplitPanel();
            for (const auto& panel : panels) {
                if (panel.panel == focused_panel && panel.valid()) {
                    return panel;
                }
            }

            return panels.front();
        }

        // Crop tool gizmo state in world space, shared by every split panel that draws it.
        struct CropToolGizmoFrame {
            GizmoOperation operation = GizmoOperation::Translate;
            glm::vec3 pivot_world{0.0f};
            glm::mat3 axes_orientation{1.0f};
            glm::vec3 bounds_center_world{0.0f};
            glm::mat3 bounds_orientation{1.0f};
            glm::vec3 bounds_half_extents_world{0.0f};
            glm::vec3 scale_pivot_world{0.0f};
        };

        // In an independent split view the focused panel owns crop tool input; every other panel shows the same
        // gizmo read-only, projected with its own camera and clipped to its own rectangle.
        void drawPassiveCropToolGizmos(VisualizerImpl* const viewer,
                                       const ViewportLayout& viewport,
                                       const SplitViewPanelId active_panel,
                                       const RenderSettings& settings,
                                       const int gizmo_id_base,
                                       const CropToolGizmoFrame& frame,
                                       NativeOverlayDrawList& draw_list) {
            auto* const rendering_manager = viewer ? viewer->getRenderingManager() : nullptr;
            if (!rendering_manager || !rendering_manager->isIndependentSplitViewActive())
                return;

            NativeGizmoInput no_input;
            no_input.mouse_pos = glm::vec2(-1.0e9f);
            for (const auto& panel : collectViewportGizmoPanels(
                     viewer, {viewport.pos.x, viewport.pos.y}, {viewport.size.x, viewport.size.y})) {
                if (!panel.valid() || panel.panel == active_panel)
                    continue;

                const glm::mat4 view = panel.viewport->getViewMatrix();
                const glm::ivec2 size(static_cast<int>(panel.size.x), static_cast<int>(panel.size.y));
                const glm::mat4 projection = lfs::rendering::createProjectionMatrixFromFocal(
                    size, settings.focal_length_mm, settings.orthographic,
                    panel.viewport->ortho_scale_override.value_or(settings.ortho_scale));
                const int id = panelGizmoId(gizmo_id_base, panel.panel);

                draw_list.PushClipRect(panel.pos, panel.pos + panel.size, true);
                if (frame.operation == GizmoOperation::Scale) {
                    BoundsGizmoConfig bounds_config;
                    bounds_config.id = id;
                    bounds_config.viewport_pos = panel.pos;
                    bounds_config.viewport_size = panel.size;
                    bounds_config.view = view;
                    bounds_config.projection = projection;
                    bounds_config.center_world = frame.bounds_center_world;
                    bounds_config.orientation_world = frame.bounds_orientation;
                    bounds_config.half_extents_world = frame.bounds_half_extents_world;
                    bounds_config.draw_list = &draw_list;
                    bounds_config.input = no_input;
                    bounds_config.input_enabled = false;
                    (void)drawBoundsGizmo(bounds_config);

                    ScaleGizmoConfig scale_config;
                    scale_config.id = id;
                    scale_config.viewport_pos = panel.pos;
                    scale_config.viewport_size = panel.size;
                    scale_config.view = view;
                    scale_config.projection = projection;
                    scale_config.pivot_world = frame.scale_pivot_world;
                    scale_config.orientation_world = frame.bounds_orientation;
                    scale_config.draw_list = &draw_list;
                    scale_config.input = no_input;
                    scale_config.input_enabled = false;
                    (void)drawScaleGizmo(scale_config);
                } else if (frame.operation == GizmoOperation::Translate) {
                    TranslationGizmoConfig translation_config;
                    translation_config.id = id;
                    translation_config.viewport_pos = panel.pos;
                    translation_config.viewport_size = panel.size;
                    translation_config.view = view;
                    translation_config.projection = projection;
                    translation_config.pivot_world = frame.pivot_world;
                    translation_config.orientation_world = frame.axes_orientation;
                    translation_config.draw_list = &draw_list;
                    translation_config.input = no_input;
                    translation_config.input_enabled = false;
                    (void)drawTranslationGizmo(translation_config);
                } else if (frame.operation == GizmoOperation::Rotate) {
                    RotationGizmoConfig rotation_config;
                    rotation_config.id = id;
                    rotation_config.viewport_pos = panel.pos;
                    rotation_config.viewport_size = panel.size;
                    rotation_config.view = view;
                    rotation_config.projection = projection;
                    rotation_config.pivot_world = frame.pivot_world;
                    rotation_config.orientation_world = frame.axes_orientation;
                    rotation_config.draw_list = &draw_list;
                    rotation_config.input = no_input;
                    rotation_config.input_enabled = false;
                    (void)drawRotationGizmo(rotation_config);
                }
                draw_list.PopClipRect();
            }
        }
    } // namespace
    constexpr float MIN_GIZMO_SCALE = 0.001f;
    constexpr float ROTATION_SNAP_DEGREES = 5.0f;
    constexpr float TRANSLATE_SNAP_UNITS = 0.1f;
    constexpr float SCALE_SNAP_RATIO = 0.1f;

    namespace {
        inline glm::mat3 extractRotation(const glm::mat4& m) {
            return glm::mat3(glm::normalize(glm::vec3(m[0])), glm::normalize(glm::vec3(m[1])),
                             glm::normalize(glm::vec3(m[2])));
        }

        inline glm::mat3 userFacingLocalRotation(const glm::mat4& visualizer_world_transform) {
            return extractRotation(visualizer_world_transform) * lfs::rendering::DATA_TO_VISUALIZER_WORLD_AXES;
        }

        inline glm::mat3 selectionVolumeLocalRotation(const glm::mat4& visualizer_world_transform) {
            return extractRotation(visualizer_world_transform);
        }

        inline glm::vec3 extractScale(const glm::mat4& m) {
            return glm::vec3(glm::length(glm::vec3(m[0])), glm::length(glm::vec3(m[1])),
                             glm::length(glm::vec3(m[2])));
        }

        [[nodiscard]] bool nearlyEqual(const glm::vec3& a, const glm::vec3& b) {
            constexpr float EPSILON = 1e-6f;
            return std::abs(a.x - b.x) <= EPSILON && std::abs(a.y - b.y) <= EPSILON &&
                   std::abs(a.z - b.z) <= EPSILON;
        }

        [[nodiscard]] bool nearlyEqual(const glm::mat4& a, const glm::mat4& b) {
            constexpr float EPSILON = 1e-6f;
            for (int col = 0; col < 4; ++col) {
                for (int row = 0; row < 4; ++row) {
                    if (std::abs(a[col][row] - b[col][row]) > EPSILON)
                        return false;
                }
            }
            return true;
        }

    } // namespace

    GizmoManager::GizmoManager(VisualizerImpl* viewer)
        : viewer_(viewer) {
        python::set_selection_submode(static_cast<int>(selection_mode_));
    }

    bool GizmoManager::isCropToolActive() const {
        return UnifiedToolRegistry::instance().getActiveTool() == "builtin.cropbox";
    }

    bool GizmoManager::isSelectionVolumeMode() const {
        return UnifiedToolRegistry::instance().getActiveTool() == "builtin.select" &&
               isSelectionVolumeSubMode(selection_mode_);
    }

    bool GizmoManager::isVolumeGizmoToolActive() const {
        return isCropToolActive() || isSelectionVolumeMode();
    }

    bool GizmoManager::isCropboxGizmoActive() const {
        return (isVolumeGizmoToolActive() && crop_tool_initialized_ && crop_tool_shape_ == CropToolShape::Box);
    }

    bool GizmoManager::isEllipsoidGizmoActive() const {
        return (isVolumeGizmoToolActive() && crop_tool_initialized_ && crop_tool_shape_ == CropToolShape::Ellipsoid);
    }

    bool GizmoManager::isCropPolygonGizmoActive() const {
        return isCropToolActive() && crop_tool_initialized_ && crop_tool_shape_ == CropToolShape::Polygon;
    }

    std::optional<core::NodeId> GizmoManager::selectedCropTargetNodeId() const {
        const auto* const sm = viewer_ ? viewer_->getSceneManager() : nullptr;
        if (!sm)
            return std::nullopt;

        const auto& scene = sm->getScene();
        const auto find_child_target = [&scene](const core::SceneNode& node,
                                                const auto& self) -> core::NodeId {
            for (const core::NodeId child_id : node.children) {
                const auto* child = scene.getNodeById(child_id);
                if (!child) {
                    continue;
                }
                if (child->type == core::NodeType::SPLAT || child->type == core::NodeType::POINTCLOUD) {
                    return child->id;
                }
                if (const core::NodeId nested = self(*child, self); nested != core::NULL_NODE) {
                    return nested;
                }
            }
            return core::NULL_NODE;
        };

        for (const auto& name : sm->getSelectedNodeNames()) {
            const auto* const node = scene.getNode(name);
            if (!node)
                continue;
            if (node->type == core::NodeType::SPLAT || node->type == core::NodeType::POINTCLOUD)
                return node->id;
            if (node->type == core::NodeType::CROPBOX || node->type == core::NodeType::ELLIPSOID ||
                node->type == core::NodeType::CROP_POLYGON) {
                const auto* parent = scene.getNodeById(node->parent_id);
                if (parent && (parent->type == core::NodeType::SPLAT || parent->type == core::NodeType::POINTCLOUD))
                    return parent->id;
                if (parent && parent->type == core::NodeType::DATASET) {
                    if (const core::NodeId child_target = find_child_target(*parent, find_child_target);
                        child_target != core::NULL_NODE) {
                        return child_target;
                    }
                }
            }
            if (node->type == core::NodeType::DATASET) {
                if (const core::NodeId child_target = find_child_target(*node, find_child_target);
                    child_target != core::NULL_NODE) {
                    return child_target;
                }
            }
        }
        return std::nullopt;
    }

    std::optional<core::NodeId> GizmoManager::selectedCropVolumeNodeId(const core::NodeId target_id) const {
        const auto* const sm = viewer_ ? viewer_->getSceneManager() : nullptr;
        if (!sm)
            return std::nullopt;

        const auto& scene = sm->getScene();
        const auto selected = sm->getSelectedNodeIds();
        const auto* exact = selected.size() == 1 ? scene.getNodeById(selected.front()) : nullptr;
        if (exact && isCropVolumeType(exact->type) && exact->parent_id == target_id)
            return exact->id;
        core::NodeId selected_id = core::NULL_NODE;
        core::NodeId attached_id = core::NULL_NODE;
        switch (crop_tool_shape_) {
        case CropToolShape::Box:
            selected_id = sm->getSelectedNodeCropBoxId();
            attached_id = scene.getCropBoxForSplat(target_id);
            break;
        case CropToolShape::Ellipsoid:
            selected_id = sm->getSelectedNodeEllipsoidId();
            attached_id = scene.getEllipsoidForSplat(target_id);
            break;
        case CropToolShape::Polygon:
            selected_id = sm->getSelectedNodeCropPolygonId();
            attached_id = scene.getCropPolygonForSplat(target_id);
            break;
        }
        if (selected_id != core::NULL_NODE) {
            const auto* const selected_node = scene.getNodeById(selected_id);
            if (selected_node && selected_node->parent_id == target_id) {
                return selected_id;
            }
        }

        if (attached_id != core::NULL_NODE) {
            return attached_id;
        }

        return std::nullopt;
    }

    bool GizmoManager::ensureCropVolumeForCurrentSelection() {
        auto* const sm = viewer_ ? viewer_->getSceneManager() : nullptr;
        if (!sm)
            return false;

        const auto target_id = selectedCropTargetNodeId();
        // A polygon exists only once drawn; there is no default one to create.
        if (!target_id || crop_tool_shape_ == CropToolShape::Polygon)
            return false;

        auto* const rm = viewer_ ? viewer_->getRenderingManager() : nullptr;
        const auto volume_id = crop_tool_shape_ == CropToolShape::Box
                                   ? cap::ensureCropBox(*sm, rm, *target_id)
                                   : cap::ensureEllipsoid(*sm, rm, *target_id);
        if (!volume_id) {
            LOG_WARN("Cannot create active crop tool volume: {}", volume_id.error());
            return false;
        }

        if (const auto* node = sm->getScene().getNodeById(*volume_id)) {
            const std::string node_name = node->name;
            if (!node->visible)
                sm->setNodeVisibilityTransient(*volume_id, true);
            sm->selectNode(node_name);
            return true;
        }
        return false;
    }

    bool GizmoManager::syncCropToolStateFromNode(const core::NodeId target_id,
                                                 const core::NodeId volume_node_id,
                                                 bool* const changed) {
        auto* const sm = viewer_ ? viewer_->getSceneManager() : nullptr;
        if (!sm)
            return false;

        const auto& scene = sm->getScene();
        const auto* const node = scene.getNodeById(volume_node_id);
        if (!node)
            return false;

        const glm::mat4 next_transform = scene_coords::nodeVisualizerWorldTransform(scene, volume_node_id);
        bool state_changed = !crop_tool_initialized_ || crop_tool_target_node_id_ != target_id ||
                             crop_tool_volume_node_id_ != volume_node_id ||
                             !nearlyEqual(crop_tool_visualizer_transform_, next_transform);

        if (crop_tool_shape_ == CropToolShape::Box) {
            if (!node->cropbox)
                return false;
            state_changed = state_changed || !nearlyEqual(crop_tool_box_min_, node->cropbox->min) ||
                            !nearlyEqual(crop_tool_box_max_, node->cropbox->max);
            crop_tool_box_min_ = node->cropbox->min;
            crop_tool_box_max_ = node->cropbox->max;
        } else if (crop_tool_shape_ == CropToolShape::Polygon) {
            if (!node->crop_polygon)
                return false;
            const auto& polygon = *node->crop_polygon;
            const glm::vec3 half_extents(core::crop_polygon::outlineHalfExtents(polygon.vertices), polygon.half_depth);
            state_changed = state_changed || crop_tool_polygon_vertices_ != polygon.vertices ||
                            !nearlyEqual(crop_tool_polygon_half_extents_, half_extents);
            crop_tool_polygon_vertices_ = polygon.vertices;
            crop_tool_polygon_half_depth_ = polygon.half_depth;
            crop_tool_polygon_half_extents_ = half_extents;
        } else {
            if (!node->ellipsoid)
                return false;
            state_changed = state_changed || !nearlyEqual(crop_tool_ellipsoid_radii_, node->ellipsoid->radii);
            crop_tool_ellipsoid_radii_ = node->ellipsoid->radii;
        }

        crop_tool_visualizer_transform_ = next_transform;
        crop_tool_target_node_id_ = target_id;
        crop_tool_volume_node_id_ = volume_node_id;
        crop_tool_initialized_ = true;
        if (changed)
            *changed = state_changed;
        return true;
    }

    void GizmoManager::scaleCropToolTransform(const glm::vec3& total_scale) {
        // Scale from the drag start so the floor stays reversible within one drag.
        if (!crop_tool_scale_start_)
            crop_tool_scale_start_ = crop_tool_visualizer_transform_;
        glm::mat4 transform = *crop_tool_scale_start_;
        for (int axis = 0; axis < 3; ++axis)
            transform[axis] *= total_scale[axis];
        gizmo_ops::clampVolumeAxisScale(transform);
        crop_tool_visualizer_transform_ = transform;
    }

    bool GizmoManager::persistActiveCropToolToNode(const bool enable) {
        auto* const sm = viewer_ ? viewer_->getSceneManager() : nullptr;
        auto* const rm = viewer_ ? viewer_->getRenderingManager() : nullptr;
        if (!sm || crop_tool_volume_node_id_ == core::NULL_NODE)
            return false;

        auto& scene = sm->getScene();
        auto* const node = scene.getNodeById(crop_tool_volume_node_id_);
        if (!node)
            return false;

        const glm::mat4 data_world_transform =
            rendering::visualizerWorldTransformToDataWorld(crop_tool_visualizer_transform_);
        const glm::mat4 parent_world = scene.getWorldTransform(node->parent_id);
        const auto local_transform = core::finiteLocalTransform(parent_world, data_world_transform);
        if (!local_transform) {
            LOG_WARN("Cannot commit crop tool transform: parent transform cannot preserve a finite world transform");
            return false;
        }

        if (crop_tool_shape_ == CropToolShape::Box) {
            if (!node->cropbox)
                return false;

            const auto before_data = *node->cropbox;
            const auto before_transform = sm->getNodeTransform(node->name);
            const bool show_before = node->visible;
            const bool use_before = node->cropbox->enabled;

            auto data = before_data;
            data.min = crop_tool_box_min_;
            data.max = crop_tool_box_max_;
            if (enable)
                data.enabled = true;
            scene.setCropBoxData(node->id, data);
            sm->setNodeTransform(node->name, *local_transform);
            // Crop volumes are scene-graph state. MODEL_CHANGED would mark the splat
            // payload dirty and make every following save capture the whole model.
            scene.notifyMutation(core::Scene::MutationType::TRANSFORM_CHANGED);
            if (rm)
                rm->markDirty(DirtyFlag::SPLATS | DirtyFlag::OVERLAY, lfs::vis::FrameReason::SceneChange);

            auto entry = std::make_unique<op::CropBoxUndoEntry>(
                *sm, rm, node->name, before_data, before_transform, show_before, use_before);
            if (entry->hasChanges())
                op::undoHistory().push(std::move(entry));
            return true;
        }

        if (crop_tool_shape_ == CropToolShape::Polygon) {
            if (!node->crop_polygon)
                return false;

            // Outline and depth are as drawn; the bounds gizmo writes the transform's scale.
            const auto before_data = *node->crop_polygon;
            const auto before_transform = sm->getNodeTransform(node->name);
            const bool show_before = node->visible;
            if (enable)
                node->crop_polygon->enabled = true;
            sm->setNodeTransform(node->name, *local_transform);
            scene.notifyMutation(core::Scene::MutationType::TRANSFORM_CHANGED);
            if (rm)
                rm->markDirty(DirtyFlag::SPLATS | DirtyFlag::OVERLAY, lfs::vis::FrameReason::SceneChange);

            auto entry = std::make_unique<op::CropPolygonUndoEntry>(
                *sm, node->name, before_data, before_transform, show_before);
            if (entry->hasChanges())
                op::undoHistory().push(std::move(entry));
            return true;
        }

        if (!node->ellipsoid)
            return false;

        const auto before_data = *node->ellipsoid;
        const auto before_transform = sm->getNodeTransform(node->name);
        const bool show_before = node->visible;
        const bool use_before = node->ellipsoid->enabled;

        auto data = before_data;
        data.radii = glm::max(crop_tool_ellipsoid_radii_, glm::vec3(MIN_GIZMO_SCALE));
        if (enable)
            data.enabled = true;
        scene.setEllipsoidData(node->id, data);
        sm->setNodeTransform(node->name, *local_transform);
        scene.notifyMutation(core::Scene::MutationType::TRANSFORM_CHANGED);
        if (rm)
            rm->markDirty(DirtyFlag::SPLATS | DirtyFlag::OVERLAY, lfs::vis::FrameReason::SceneChange);

        auto entry = std::make_unique<op::EllipsoidUndoEntry>(
            *sm, rm, node->name, before_data, before_transform, show_before, use_before);
        if (entry->hasChanges())
            op::undoHistory().push(std::move(entry));
        return true;
    }

    bool GizmoManager::syncSelectedCropShape() {
        auto* sm = viewer_ ? viewer_->getSceneManager() : nullptr;
        if (!sm || !isCropToolActive() || !sm->hasSelectedNode())
            return false;
        const auto type = sm->getSelectedNodeType();
        if (!isCropVolumeType(type))
            return false;
        const auto shape = type == core::NodeType::CROPBOX     ? CropToolShape::Box
                           : type == core::NodeType::ELLIPSOID ? CropToolShape::Ellipsoid
                                                               : CropToolShape::Polygon;
        if (shape == crop_tool_shape_)
            return false;
        crop_tool_shape_ = shape;
        crop_tool_initialized_ = false;
        clearCropToolOverlayState();
        return true;
    }

    bool GizmoManager::ensureCropToolState() {
        (void)syncSelectedCropShape();
        auto* const sm = viewer_ ? viewer_->getSceneManager() : nullptr;
        if (!sm)
            return false;

        if (isSelectionVolumeMode()) {
            if (!crop_tool_initialized_) {
                crop_tool_target_node_id_ = core::NULL_NODE;
                crop_tool_volume_node_id_ = core::NULL_NODE;
                clearCropToolOverlayState();
            }
            return crop_tool_initialized_;
        }

        if ((crop_tool_drag_active_ || crop_polygon_vertex_drag_) && crop_tool_initialized_)
            return true;

        const auto target_id = selectedCropTargetNodeId();
        if (!target_id) {
            crop_tool_initialized_ = false;
            crop_tool_target_node_id_ = core::NULL_NODE;
            crop_tool_volume_node_id_ = core::NULL_NODE;
            clearCropToolOverlayState();
            return false;
        }

        const auto volume_id = selectedCropVolumeNodeId(*target_id);
        if (!volume_id) {
            crop_tool_initialized_ = false;
            crop_tool_target_node_id_ = core::NULL_NODE;
            crop_tool_volume_node_id_ = core::NULL_NODE;
            clearCropToolOverlayState();
            return false;
        }

        bool state_changed = false;
        if (!syncCropToolStateFromNode(*target_id, *volume_id, &state_changed)) {
            crop_tool_initialized_ = false;
            crop_tool_target_node_id_ = core::NULL_NODE;
            crop_tool_volume_node_id_ = core::NULL_NODE;
            clearCropToolOverlayState();
            return false;
        }

        if (state_changed)
            updateCropToolOverlayState();
        return true;
    }

    void GizmoManager::clearCropToolOverlayState() {
        if (auto* const rm = viewer_ ? viewer_->getRenderingManager() : nullptr) {
            rm->setCropboxGizmoActive(false);
            rm->setEllipsoidGizmoActive(false);
            rm->setCropPolygonGizmoActive(false);
        }
        crop_tool_volume_node_id_ = core::NULL_NODE;
        crop_tool_drag_active_ = false;
        crop_tool_scale_start_.reset();
        crop_tool_drag_changed_ = false;
        crop_polygon_vertex_drag_.reset();
    }

    void GizmoManager::leaveCropTool(const bool hide_volume,
                                     const bool select_parent,
                                     const bool clear_active_tool) {
        crop_tool_shape_locked_ = false;
        auto* const sm = viewer_ ? viewer_->getSceneManager() : nullptr;
        core::NodeId volume_node_id = crop_tool_volume_node_id_;
        core::NodeId parent_node_id = core::NULL_NODE;

        if (sm) {
            auto& scene = sm->getScene();
            if (volume_node_id == core::NULL_NODE) {
                const core::NodeId cropbox_id = sm->getSelectedNodeCropBoxId();
                const core::NodeId ellipsoid_id = sm->getSelectedNodeEllipsoidId();
                const core::NodeId polygon_id = sm->getSelectedNodeCropPolygonId();
                const auto* cropbox_node = scene.getNodeById(cropbox_id);
                const auto* ellipsoid_node = scene.getNodeById(ellipsoid_id);
                if (cropbox_node && cropbox_node->type == core::NodeType::CROPBOX) {
                    volume_node_id = cropbox_id;
                } else if (ellipsoid_node && ellipsoid_node->type == core::NodeType::ELLIPSOID) {
                    volume_node_id = ellipsoid_id;
                } else if (polygon_id != core::NULL_NODE) {
                    volume_node_id = polygon_id;
                }
            }

            const auto* const node = scene.getNodeById(volume_node_id);
            if (node && (node->type == core::NodeType::CROPBOX || node->type == core::NodeType::ELLIPSOID ||
                         node->type == core::NodeType::CROP_POLYGON)) {
                parent_node_id = node->parent_id;
                if (hide_volume && node->visible) {
                    sm->setNodeVisibilityTransient(volume_node_id, false);
                }
            }
        }

        crop_tool_initialized_ = false;
        crop_tool_target_node_id_ = core::NULL_NODE;
        clearCropToolOverlayState();

        if (clear_active_tool) {
            python::cancel_active_operator();
            if (viewer_)
                viewer_->getEditorContext().clearActiveOperator();
            UnifiedToolRegistry::instance().setActiveTool("");
        }

        if (select_parent && sm && parent_node_id != core::NULL_NODE && sm->getScene().getNodeById(parent_node_id)) {
            sm->selectNode(parent_node_id);
        }
    }

    bool GizmoManager::deleteCropVolumeAfterApply(const core::NodeId volume_node_id) {
        auto* const sm = viewer_ ? viewer_->getSceneManager() : nullptr;
        if (!sm || volume_node_id == core::NULL_NODE) {
            leaveCropTool(false, true, true);
            return false;
        }

        const auto* const node = sm->getScene().getNodeById(volume_node_id);
        if (!node || (node->type != core::NodeType::CROPBOX && node->type != core::NodeType::ELLIPSOID &&
                      node->type != core::NodeType::CROP_POLYGON)) {
            leaveCropTool(false, true, true);
            return false;
        }

        const core::NodeId parent_node_id = node->parent_id;
        crop_tool_initialized_ = false;
        crop_tool_target_node_id_ = core::NULL_NODE;
        clearCropToolOverlayState();
        python::cancel_active_operator();
        if (viewer_)
            viewer_->getEditorContext().clearActiveOperator();
        UnifiedToolRegistry::instance().setActiveTool("");

        // The crop snapshot also records sibling order. Restore that order on
        // undo before replaying the geometry snapshot in the enclosing bake.
        if (auto result = sm->removeNodesByIdsWithResult({volume_node_id}, false); !result) {
            LOG_WARN("Failed to remove crop volume: {}", result.error());
            return false;
        }
        if (parent_node_id != core::NULL_NODE && sm->getScene().getNodeById(parent_node_id)) {
            sm->selectNode(parent_node_id);
        }
        return true;
    }

    void GizmoManager::clearSelectionVolumeState() {
        selection_volume_source_generation_ = 0;
        selection_volume_base_mask_.reset();
        selection_volume_selection_before_drag_ = {};
        selection_volume_gizmo_active_ = false;
        selection_volume_drag_changed_ = false;
        selection_volume_apply_mode_ = SelectionMode::Replace;
    }

    void GizmoManager::captureSelectionVolumeBase(const uint64_t source_generation) {
        if (selection_volume_source_generation_ == source_generation)
            return;

        selection_volume_source_generation_ = source_generation;
        selection_volume_selection_before_drag_ = {};
        selection_volume_gizmo_active_ = false;
        selection_volume_drag_changed_ = false;

        auto* const sm = viewer_ ? viewer_->getSceneManager() : nullptr;
        if (!sm) {
            selection_volume_base_mask_.reset();
            return;
        }

        auto& scene = sm->getScene();
        const size_t selection_count = scene.getSelectionGaussianCount();
        if (selection_count == 0) {
            selection_volume_base_mask_.reset();
            return;
        }

        const auto selection = scene.getSelectionMask();
        selection_volume_base_mask_ =
            (selection && selection->is_valid() && selection->numel() == selection_count)
                ? std::make_shared<core::Tensor>(selection->clone())
                : std::make_shared<core::Tensor>(
                      core::Tensor::zeros({selection_count}, core::Device::CUDA, core::DataType::Bool));
    }

    bool GizmoManager::applySelectionVolumeFromGizmo(const bool push_undo) {
        if (!isSelectionVolumeMode() || !crop_tool_initialized_)
            return false;

        auto* const sm = viewer_ ? viewer_->getSceneManager() : nullptr;
        auto* const service = sm ? sm->getSelectionService() : nullptr;
        if (!service)
            return false;

        SelectionCommitOptions options;
        options.base_selection = selection_volume_base_mask_.get();
        options.push_undo = push_undo;

        const auto result = crop_tool_shape_ == CropToolShape::Box
                                ? service->selectBoxVolume(selection_volume_apply_mode_, options)
                                : service->selectSphereVolume(selection_volume_apply_mode_, options);
        if (!result.success) {
            LOG_WARN("Failed to apply selection volume: {}", result.error);
            return false;
        }
        return true;
    }

    void GizmoManager::beginSelectionVolumeGizmoDrag() {
        if (selection_volume_gizmo_active_)
            return;

        crop_tool_box_min_before_drag_ = crop_tool_box_min_;
        crop_tool_box_max_before_drag_ = crop_tool_box_max_;
        crop_tool_ellipsoid_radii_before_drag_ = crop_tool_ellipsoid_radii_;
        crop_tool_visualizer_transform_before_drag_ = crop_tool_visualizer_transform_;
        selection_volume_selection_before_drag_ = {};
        if (auto* const sm = viewer_ ? viewer_->getSceneManager() : nullptr) {
            selection_volume_selection_before_drag_ = sm->getScene().captureSelectionState();
        }
        selection_volume_drag_changed_ = false;
        selection_volume_gizmo_active_ = true;
    }

    void GizmoManager::finishSelectionVolumeGizmoDrag() {
        if (!selection_volume_gizmo_active_)
            return;

        selection_volume_gizmo_active_ = false;
        if (!selection_volume_drag_changed_)
            return;

        if (auto* const sm = viewer_ ? viewer_->getSceneManager() : nullptr) {
            sm->getScene().restoreSelectionState(selection_volume_selection_before_drag_);
        }
        selection_volume_selection_before_drag_ = {};
        selection_volume_drag_changed_ = false;
        (void)applySelectionVolumeFromGizmo(true);
    }

    void GizmoManager::updateCropToolOverlayState() {
        auto* const rm = viewer_ ? viewer_->getRenderingManager() : nullptr;
        if (!rm || !crop_tool_initialized_ || !isVolumeGizmoToolActive()) {
            clearCropToolOverlayState();
            return;
        }

        bool affects_render = false;
        bool effectively_visible = true;
        int parent_node_index = -1;
        if (isCropToolActive()) {
            if (auto* const sm = viewer_ ? viewer_->getSceneManager() : nullptr) {
                const auto& scene = sm->getScene();
                if (const auto* const node = scene.getNodeById(crop_tool_volume_node_id_)) {
                    effectively_visible = scene.isNodeEffectivelyVisible(crop_tool_volume_node_id_);
                    parent_node_index = scene.getVisibleNodeIndex(node->parent_id);
                    switch (crop_tool_shape_) {
                    case CropToolShape::Box: affects_render = node->cropbox && node->cropbox->enabled; break;
                    case CropToolShape::Ellipsoid: affects_render = node->ellipsoid && node->ellipsoid->enabled; break;
                    case CropToolShape::Polygon:
                        affects_render = node->crop_polygon && node->crop_polygon->enabled;
                        break;
                    }
                }
            }
        }
        if (auto* manager = viewer_ ? viewer_->getSceneManager() : nullptr) {
            if (const auto* owner = manager->getScene().getNodeById(crop_tool_target_node_id_);
                owner && owner->type == core::NodeType::SPLAT && !owner->crop_effects_enabled)
                affects_render = false;
        }
        rm->setCropboxGizmoActive(false);
        rm->setEllipsoidGizmoActive(false);
        rm->setCropPolygonGizmoActive(false);
        if (!effectively_visible) {
            rm->markDirty(DirtyFlag::SPLATS | DirtyFlag::OVERLAY, lfs::vis::FrameReason::SceneChange);
            return;
        }
        switch (crop_tool_shape_) {
        case CropToolShape::Box:
            rm->setCropboxGizmoState(
                true, crop_tool_box_min_, crop_tool_box_max_, crop_tool_visualizer_transform_, affects_render, parent_node_index);
            break;
        case CropToolShape::Ellipsoid:
            rm->setEllipsoidGizmoState(
                true, crop_tool_ellipsoid_radii_, crop_tool_visualizer_transform_, affects_render, parent_node_index);
            break;
        case CropToolShape::Polygon:
            rm->setCropPolygonGizmoState(true, crop_tool_polygon_vertices_, crop_tool_polygon_half_depth_,
                                         crop_tool_visualizer_transform_, affects_render, parent_node_index);
            break;
        }
        rm->markDirty(DirtyFlag::SPLATS | DirtyFlag::OVERLAY, lfs::vis::FrameReason::SceneChange);
    }

    void GizmoManager::setCropToolShape(const std::string& shape) {
        if (isCropToolShapeLocked() && shape != cropToolShape())
            return;
        crop_tool_shape_ = shape == "ellipsoid" ? CropToolShape::Ellipsoid
                           : shape == "polygon" ? CropToolShape::Polygon
                                                : CropToolShape::Box;
        if (isVolumeGizmoToolActive() && ensureCropToolState())
            updateCropToolOverlayState();
    }

    std::string GizmoManager::cropToolShape() const {
        const auto* sm = viewer_ ? viewer_->getSceneManager() : nullptr;
        if (isCropToolActive() && sm && sm->hasSelectedNode()) {
            switch (sm->getSelectedNodeType()) {
            case core::NodeType::CROPBOX: return "box";
            case core::NodeType::ELLIPSOID: return "ellipsoid";
            case core::NodeType::CROP_POLYGON: return "polygon";
            default: break;
            }
        }
        switch (crop_tool_shape_) {
        case CropToolShape::Ellipsoid: return "ellipsoid";
        case CropToolShape::Polygon: return "polygon";
        case CropToolShape::Box: break;
        }
        return "box";
    }

    void GizmoManager::setCropToolOperation(const std::string& operation) {
        if (operation == "rotate") {
            current_operation_ = GizmoOperation::Rotate;
        } else if (operation == "scale") {
            current_operation_ = GizmoOperation::Scale;
        } else {
            current_operation_ = GizmoOperation::Translate;
        }
        if (isCropToolActive() && viewer_) {
            auto& editor = viewer_->getEditorContext();
            if (editor.getGizmoType() != cropToolOperation())
                editor.setActiveOperator("builtin.cropbox", cropToolOperation());
        }
        if (isVolumeGizmoToolActive() && ensureCropToolState()) {
            updateCropToolOverlayState();
        }
    }

    std::string GizmoManager::cropToolOperation() const {
        switch (current_operation_) {
        case GizmoOperation::Rotate: return "rotate";
        case GizmoOperation::Scale: return "scale";
        case GizmoOperation::Translate: return "translate";
        }
        return "translate";
    }

    void GizmoManager::finishCropTool() {
        if (isSelectionVolumeMode()) {
            (void)applySelectionVolumeFromGizmo(true);
            return;
        }
        if (!isCropToolActive() || !ensureCropToolState())
            return;
        auto* sm = viewer_ ? viewer_->getSceneManager() : nullptr;
        auto* editor = services().editorOrNull();
        if (!sm || sm->getScene().isNodeEffectivelyLocked(crop_tool_volume_node_id_) ||
            (editor && editor->isToolsDisabled()) || op::operators().hasModalOperator())
            return;
        if (persistActiveCropToolToNode(false))
            leaveCropTool(false, false, true);
    }

    void GizmoManager::editCropTool(const core::Uuid& uuid) {
        auto* sm = viewer_ ? viewer_->getSceneManager() : nullptr;
        if (!sm || viewer_->getEditorContext().isToolsDisabled() || op::operators().hasModalOperator())
            return;
        const auto* node = sm->getScene().getNodeByUuid(uuid);
        const auto* owner = node ? sm->getScene().getNodeById(node->parent_id) : nullptr;
        const auto* descriptor = node ? core::splatOperationDescriptor(node->type) : nullptr;
        if (!node || !owner || !descriptor || descriptor->category != core::SplatOperationCategory::CropVolume ||
            sm->getScene().isNodeEffectivelyLocked(node->id))
            return;
        crop_tool_shape_locked_ = false;
        sm->selectNode(node->id);
        sm->setNodeVisibilityTransient(node->id, true);
        viewer_->getEditorContext().setActiveOperator("builtin.cropbox", "translate");
        UnifiedToolRegistry::instance().setActiveTool("builtin.cropbox");
        setCropToolShape(node->type == core::NodeType::ELLIPSOID ? "ellipsoid" : node->type == core::NodeType::CROP_POLYGON ? "polygon"
                                                                                                                            : "box");
        crop_tool_shape_locked_ = true;
        if (ensureCropToolState())
            updateCropToolOverlayState();
    }

    bool GizmoManager::isCropToolShapeLocked() const {
        return crop_tool_shape_locked_ && isCropToolActive();
    }

    bool GizmoManager::changeCropToolShape(const std::string& shape) {
        if (shape != "box" && shape != "ellipsoid" && shape != "polygon")
            return false;
        if (isSelectionVolumeMode()) {
            setCropToolShape(shape);
            return true;
        }
        if (isCropToolShapeLocked())
            return shape == cropToolShape();
        auto* sm = viewer_ ? viewer_->getSceneManager() : nullptr;
        if (!sm || !isCropToolActive())
            return false;
        const auto selected = sm->getSelectedNodeIds();
        const auto* node = selected.size() == 1 ? sm->getScene().getNodeById(selected.front()) : nullptr;
        if (!node || !isCropVolumeType(node->type))
            return false;
        const auto uuid = node->uuid;
        const auto operation = cropToolOperation();
        const auto target = shape == "box" ? core::NodeType::CROPBOX : shape == "ellipsoid" ? core::NodeType::ELLIPSOID
                                                                                            : core::NodeType::CROP_POLYGON;
        auto result = cap::convertSplatCrop(*sm, viewer_->getRenderingManager(), uuid, target);
        if (!result) {
            lfs::ErrorBus::instance().publish(lfs::ErrorNotification{.error = result.error(), .surface = lfs::ErrorSurface::Toast});
            return false;
        }
        crop_tool_initialized_ = false;
        clearCropToolOverlayState();
        editCropTool(uuid);
        crop_tool_shape_locked_ = false;
        setCropToolOperation(operation);
        return true;
    }

    void GizmoManager::newCropTool(const std::string& shape) {
        auto* sm = viewer_ ? viewer_->getSceneManager() : nullptr;
        if (!sm || op::operators().hasModalOperator())
            return;
        auto& scene = sm->getScene();
        const auto selected = sm->getSelectedNodeIds();
        const auto* owner = selected.size() == 1 ? scene.getNodeById(selected[0]) : nullptr;
        if (owner && core::splatOperationDescriptor(owner->type))
            owner = scene.getNodeById(owner->parent_id);
        if (!owner || owner->type != core::NodeType::SPLAT || scene.isNodeEffectivelyLocked(owner->id) ||
            viewer_->getEditorContext().isToolsDisabled())
            return;
        if (shape != "box" && shape != "ellipsoid" && shape != "polygon")
            return;
        finishCropTool();
        crop_tool_shape_locked_ = false;
        if (shape == "polygon") {
            op::OperatorProperties props;
            props.set("target_node_id", static_cast<int>(owner->id));
            props.set("new_instance", true);
            (void)op::operators().invoke(op::BuiltinOp::CropPolygonDraw, &props);
            return;
        }
        auto created = cap::createSplatCrop(*sm, viewer_->getRenderingManager(), owner->uuid,
                                            shape == "box" ? core::NodeType::CROPBOX : core::NodeType::ELLIPSOID);
        if (created) {
            editCropTool(scene.getNodeById(*created)->uuid);
            crop_tool_shape_locked_ = false;
        } else
            lfs::ErrorBus::instance().publish(lfs::ErrorNotification{
                .error = created.error(),
                .surface = lfs::ErrorSurface::Toast});
    }

    void GizmoManager::applyActiveCropTool() {
        using namespace lfs::core::events;

        if (!ensureCropToolState())
            return;

        if (isSelectionVolumeMode()) {
            (void)applySelectionVolumeFromGizmo(true);
            return;
        }

        if (!isCropToolActive())
            return;

        auto* const sm = viewer_ ? viewer_->getSceneManager() : nullptr;
        if (!sm)
            return;

        if (!sm->canApplyCropToNode(crop_tool_target_node_id_))
            return;

        // Committing the volume, cropping and removing the volume are one user action,
        // so a single undo restores the splat and the volume.
        op::TransactionGuard transaction(crop_tool_shape_ == CropToolShape::Box       ? "Crop Box"
                                         : crop_tool_shape_ == CropToolShape::Polygon ? "Crop Polygon"
                                                                                      : "Crop Ellipsoid");
        if (!persistActiveCropToolToNode(true))
            return;

        const core::NodeId applied_volume_node_id = crop_tool_volume_node_id_;
        auto& scene = sm->getScene();
        if (crop_tool_shape_ == CropToolShape::Box) {
            const auto* cropbox_node = scene.getNodeById(crop_tool_volume_node_id_);
            if (!cropbox_node || !cropbox_node->cropbox) {
                LOG_WARN("Failed to apply active crop tool: crop box node is invalid");
                return;
            }

            lfs::geometry::BoundingBox crop_box;
            crop_box.setBounds(cropbox_node->cropbox->min, cropbox_node->cropbox->max);
            crop_box.setworld2BBox(glm::inverse(scene_coords::nodeDataWorldTransform(scene, cropbox_node->id)));
            cmd::CropPLY{.crop_box = crop_box, .inverse = cropbox_node->cropbox->inverse, .target_node_id = cropbox_node->parent_id}.emit();
        } else if (crop_tool_shape_ == CropToolShape::Polygon) {
            const auto* polygon_node = scene.getNodeById(crop_tool_volume_node_id_);
            if (!polygon_node || !polygon_node->crop_polygon) {
                LOG_WARN("Failed to apply active crop tool: crop polygon node is invalid");
                return;
            }
            emitCropPolygon(scene, *polygon_node);
        } else {
            const auto* ellipsoid_node = scene.getNodeById(crop_tool_volume_node_id_);
            if (!ellipsoid_node || !ellipsoid_node->ellipsoid) {
                LOG_WARN("Failed to apply active crop tool: ellipsoid node is invalid");
                return;
            }
            cmd::CropPLYEllipsoid{
                .world_transform = scene_coords::nodeDataWorldTransform(scene, ellipsoid_node->id),
                .radii = ellipsoid_node->ellipsoid->radii,
                .inverse = ellipsoid_node->ellipsoid->inverse,
                .target_node_id = ellipsoid_node->parent_id}
                .emit();
        }
        triggerCropFlash();
        if (deleteCropVolumeAfterApply(applied_volume_node_id))
            transaction.commit();
    }

    void GizmoManager::deleteActiveCropToolVolume() {
        if (!isCropToolActive() || !ensureCropToolState())
            return;

        deleteCropVolumeAfterApply(crop_tool_volume_node_id_);
    }

    void GizmoManager::activateCropPolygonTool() {
        auto* const sm = viewer_ ? viewer_->getSceneManager() : nullptr;
        const auto target_id = selectedCropTargetNodeId();
        if (!sm || !target_id)
            return;
        const auto* const target = sm->getScene().getNodeById(*target_id);
        if (!target || !op::isCropPolygonTarget(target->type)) {
            LOG_WARN("Crop polygons can only be drawn for splat or point cloud nodes");
            return;
        }

        const auto activate = [this] {
            viewer_->getEditorContext().setActiveOperator("builtin.cropbox", "translate");
            UnifiedToolRegistry::instance().setActiveTool("builtin.cropbox");
            setCropToolShape("polygon");
        };
        if (op::operators().activeModalId() == op::to_string(op::BuiltinOp::CropPolygonDraw)) {
            activate();
            return;
        }

        if (const core::NodeId polygon_id = sm->getSelectedNodeCropPolygonId();
            polygon_id != core::NULL_NODE) {
            python::cancel_active_operator();
            if (!sm->getScene().getNodeById(polygon_id)->visible)
                sm->setNodeVisibilityTransient(polygon_id, true);
            sm->selectNode(polygon_id);
            activate();
            return;
        }

        // The draw operator enters the polygon tool itself and restores the scene on cancel.
        python::cancel_active_operator();
        op::OperatorProperties props;
        props.set("target_node_id", static_cast<int>(*target_id));
        if (op::operators().invoke(op::BuiltinOp::CropPolygonDraw, &props).status != op::OperatorResult::RUNNING_MODAL)
            LOG_WARN("Cannot start drawing a crop polygon");
    }

    void GizmoManager::fitSelectedCropPolygon(const bool use_percentile) {
        auto* const sm = viewer_ ? viewer_->getSceneManager() : nullptr;
        const core::NodeId polygon_id = sm ? sm->getSelectedNodeCropPolygonId() : core::NULL_NODE;
        if (polygon_id == core::NULL_NODE)
            return;
        if (auto result = cap::fitCropPolygonToParent(*sm, viewer_->getRenderingManager(), polygon_id, use_percentile);
            !result)
            LOG_WARN("Failed to fit crop polygon: {}", result.error().user_message());
    }

    void GizmoManager::toggleSelectedCropInverse() {
        auto* const sm = viewer_->getSceneManager();
        if (!sm)
            return;

        if (const core::NodeId polygon_id = sm->getSelectedNodeCropPolygonId(); polygon_id != core::NULL_NODE) {
            auto* const node = sm->getScene().getNodeById(polygon_id);
            if (!node || !node->crop_polygon)
                return;
            const auto before = *node->crop_polygon;
            const auto before_transform = sm->getNodeTransform(node->name);
            node->crop_polygon->inverse = !node->crop_polygon->inverse;
            sm->getScene().notifyMutation(core::Scene::MutationType::TRANSFORM_CHANGED);
            if (auto* const rm = viewer_->getRenderingManager())
                rm->markDirty(DirtyFlag::SPLATS | DirtyFlag::OVERLAY, lfs::vis::FrameReason::SceneChange);
            auto entry = std::make_unique<op::CropPolygonUndoEntry>(
                *sm, node->name, before, before_transform, node->visible.get());
            if (entry->hasChanges())
                op::undoHistory().push(std::move(entry));
            return;
        }

        if (auto cropbox_id = cap::resolveCropBoxId(*sm, std::nullopt)) {
            const auto* node = sm->getScene().getNodeById(*cropbox_id);
            if (!node || !node->cropbox)
                return;

            cap::CropBoxUpdate update;
            update.has_inverse = true;
            update.inverse = !node->cropbox->inverse;
            if (auto result = cap::updateCropBox(*sm, viewer_->getRenderingManager(), *cropbox_id, update); !result) {
                LOG_WARN("Failed to toggle crop inverse: {}", result.error());
            }
            return;
        }

        const auto ellipsoid_id = cap::resolveEllipsoidId(*sm, std::nullopt);
        const auto* const ellipsoid_node = ellipsoid_id ? sm->getScene().getNodeById(*ellipsoid_id) : nullptr;
        if (!ellipsoid_node || !ellipsoid_node->ellipsoid)
            return;

        cap::EllipsoidUpdate update;
        update.has_inverse = true;
        update.inverse = !ellipsoid_node->ellipsoid->inverse;
        if (auto result = cap::updateEllipsoid(*sm, viewer_->getRenderingManager(), *ellipsoid_id, update); !result) {
            LOG_WARN("Failed to toggle ellipsoid inverse: {}", result.error());
        }
    }

    std::optional<bool> GizmoManager::selectedCropInverse() const {
        auto* const sm = viewer_->getSceneManager();
        if (!sm)
            return std::nullopt;
        if (const core::NodeId polygon_id = sm->getSelectedNodeCropPolygonId(); polygon_id != core::NULL_NODE) {
            const auto* const node = sm->getScene().getNodeById(polygon_id);
            return node && node->crop_polygon ? std::optional<bool>(node->crop_polygon->inverse) : std::nullopt;
        }
        if (const auto cropbox_id = cap::resolveCropBoxId(*sm, std::nullopt)) {
            const auto* const node = sm->getScene().getNodeById(*cropbox_id);
            return node && node->cropbox ? std::optional<bool>(node->cropbox->inverse) : std::nullopt;
        }
        if (const auto ellipsoid_id = cap::resolveEllipsoidId(*sm, std::nullopt)) {
            const auto* const node = sm->getScene().getNodeById(*ellipsoid_id);
            return node && node->ellipsoid ? std::optional<bool>(node->ellipsoid->inverse) : std::nullopt;
        }
        return std::nullopt;
    }

    void GizmoManager::resetSelectedCropPolygon() {
        auto* const sm = viewer_ ? viewer_->getSceneManager() : nullptr;
        const core::NodeId polygon_id = sm ? sm->getSelectedNodeCropPolygonId() : core::NULL_NODE;
        if (polygon_id == core::NULL_NODE)
            return;
        if (auto result = cap::resetCropPolygon(*sm, viewer_->getRenderingManager(), polygon_id); !result)
            LOG_WARN("Failed to reset crop polygon: {}", result.error().user_message());
    }

    GizmoManager::CropPolygonHandles GizmoManager::updateCropPolygonVertexEdit(const glm::vec2 panel_pos,
                                                                               const glm::vec2 panel_size,
                                                                               const glm::mat4& view,
                                                                               const glm::mat4& projection) {
        CropPolygonHandles handles;
        const glm::mat4 view_projection = projection * view;
        const auto& frame_input = viewer_->getWindowManager()->frameInput();
        const glm::vec2 mouse(frame_input.mouse_x, frame_input.mouse_y);
        const auto project = [&](const std::vector<glm::vec2>& vertices, const glm::mat4& to_world, const float z) {
            std::vector<std::optional<glm::vec2>> points;
            points.reserve(vertices.size());
            for (const glm::vec2& vertex : vertices)
                points.push_back(projectToPanel(view_projection, panel_pos, panel_size,
                                                glm::vec3(to_world * glm::vec4(vertex, z, 1.0f))));
            return points;
        };

        if (crop_polygon_vertex_drag_) {
            auto& drag = *crop_polygon_vertex_drag_;
            if (frame_input.mouse_down[0]) {
                if (const auto local = pixelOnPolygonPlane(view_projection, panel_pos, panel_size,
                                                           drag.polygon_to_world, drag.plane_z, mouse);
                    local && *local != drag.vertices[drag.index]) {
                    drag.vertices[drag.index] = *local;
                    drag.moved = true;
                    previewCropPolygonOutline(drag.vertices, drag.polygon_to_world);
                }
                handles.screen = project(drag.vertices, drag.polygon_to_world, drag.plane_z);
                handles.hot = static_cast<int>(drag.index);
                handles.owns_pointer = true;
                return handles;
            }
            // Released: the whole drag is one outline edit.
            auto vertices = std::move(drag.vertices);
            const bool moved = drag.moved;
            crop_polygon_vertex_drag_.reset();
            if (moved)
                (void)commitCropPolygonOutline(std::move(vertices));
            else
                (void)ensureCropToolState();
        }

        auto* const sm = viewer_->getSceneManager();
        if (!sm || !sm->getScene().isNodeEffectivelyVisible(crop_tool_volume_node_id_))
            return handles;

        // Handles sit on the cap that faces the camera.
        const glm::mat4& to_world = crop_tool_visualizer_transform_;
        const glm::vec3 camera(glm::inverse(view)[3]);
        const float plane_z = glm::dot(glm::vec3(to_world[2]), camera - glm::vec3(to_world[3])) >= 0.0f
                                  ? crop_tool_polygon_half_depth_
                                  : -crop_tool_polygon_half_depth_;
        handles.screen = project(crop_tool_polygon_vertices_, to_world, plane_z);

        // A gizmo drag keeps the pointer; elsewhere the vertex handles win only on themselves.
        if (crop_tool_drag_active_ || isBoundsGizmoActive() || isTranslationGizmoActive() ||
            isRotationGizmoActive() || isScaleGizmoActive())
            return handles;
        const bool gizmo_hovered = isBoundsGizmoHovered() || isTranslationGizmoHovered() ||
                                   isRotationGizmoHovered() || isScaleGizmoHovered();
        handles.hot = findOutlineVertexAt(handles.screen, mouse);

        // The key map of the controller that dispatches these presses decides insert and remove.
        const auto* const input_controller = viewer_->getWindowManager()->inputController();
        const int mods = input::sdlModsToAppMods(frame_input.key_mods);
        for (const auto& press : frame_input.mouse_button_events) {
            if (!press.down)
                continue;
            const glm::vec2 point(press.x, press.y);
            const int vertex = findOutlineVertexAt(handles.screen, point);
            const auto action = input_controller
                                    ? input_controller->getBindings().getActionForMouseButton(
                                          input::ToolMode::CROP_BOX, static_cast<input::MouseButton>(press.button), mods)
                                    : input::Action::NONE;
            if (action == input::Action::REMOVE_CROP_POLYGON_VERTEX) {
                if (vertex >= 0 && crop_tool_polygon_vertices_.size() > core::crop_polygon::kMinVertices) {
                    auto vertices = crop_tool_polygon_vertices_;
                    vertices.erase(vertices.begin() + vertex);
                    (void)commitCropPolygonOutline(std::move(vertices));
                }
                return handles;
            }
            if (action == input::Action::INSERT_CROP_POLYGON_VERTEX) {
                const int edge = gizmo_hovered ? -1 : findOutlineEdgeAt(handles.screen, point);
                const auto local = edge >= 0 ? pixelOnPolygonPlane(view_projection, panel_pos, panel_size, to_world,
                                                                   plane_z, point)
                                             : std::nullopt;
                if (local) {
                    auto vertices = crop_tool_polygon_vertices_;
                    vertices.insert(vertices.begin() + edge + 1, *local);
                    (void)commitCropPolygonOutline(std::move(vertices));
                }
                return handles;
            }
            if (press.button == static_cast<uint8_t>(input::AppMouseButton::LEFT) && vertex >= 0) {
                crop_polygon_vertex_drag_ = CropPolygonVertexDrag{
                    .index = static_cast<size_t>(vertex),
                    .plane_z = plane_z,
                    .polygon_to_world = to_world,
                    .vertices = crop_tool_polygon_vertices_,
                    .moved = false,
                };
                handles.hot = vertex;
            }
        }
        handles.owns_pointer = handles.hot >= 0;
        return handles;
    }

    void GizmoManager::previewCropPolygonOutline(const std::vector<glm::vec2>& vertices, const glm::mat4& polygon_to_world) {
        // Shown re-centred like the commit will store it, so the prism does not jump on release.
        glm::vec2 lo(std::numeric_limits<float>::max());
        glm::vec2 hi(std::numeric_limits<float>::lowest());
        for (const glm::vec2& vertex : vertices) {
            lo = glm::min(lo, vertex);
            hi = glm::max(hi, vertex);
        }
        const glm::vec2 centre = (lo + hi) * 0.5f;
        crop_tool_polygon_vertices_.clear();
        for (const glm::vec2& vertex : vertices)
            crop_tool_polygon_vertices_.push_back(vertex - centre);
        crop_tool_polygon_half_extents_ = glm::vec3(core::crop_polygon::outlineHalfExtents(crop_tool_polygon_vertices_),
                                                    crop_tool_polygon_half_depth_);
        crop_tool_visualizer_transform_ = glm::translate(polygon_to_world, glm::vec3(centre, 0.0f));
        updateCropToolOverlayState();
    }

    bool GizmoManager::commitCropPolygonOutline(std::vector<glm::vec2> vertices) {
        auto* const sm = viewer_ ? viewer_->getSceneManager() : nullptr;
        bool committed = false;
        if (sm && crop_tool_volume_node_id_ != core::NULL_NODE) {
            if (auto result = cap::setCropPolygonOutline(*sm, viewer_->getRenderingManager(), crop_tool_volume_node_id_,
                                                         std::move(vertices));
                result) {
                committed = true;
            } else {
                LOG_WARN("Crop polygon edit rejected: {}", result.error().user_message());
            }
        }
        // Back to the stored outline, edited or not.
        if (ensureCropToolState())
            updateCropToolOverlayState();
        return committed;
    }

    void GizmoManager::setupEvents() {
        using namespace lfs::core::events;

        ui::NodeSelected::when([this](const auto& event) {
            auto* const sm = viewer_ ? viewer_->getSceneManager() : nullptr;
            const auto* const selected_node = sm ? sm->getScene().getNode(event.path) : nullptr;
            const bool selected_crop_volume = selected_node && isCropVolumeType(selected_node->type);
            const bool active_crop_operator = UnifiedToolRegistry::instance().getActiveTool() == "builtin.cropbox" ||
                                              viewer_->getEditorContext().getActiveOperator() == "builtin.cropbox";
            if (!op::undoHistory().isPlaybackActive()) {
                if (active_crop_operator)
                    leaveCropTool(false, false, true);
                else
                    python::cancel_active_operator();
                // Helper visibility follows selection, independently of crop filtering.
                if (sm)
                    for (const auto* volume : sm->getScene().getNodes())
                        if (isCropVolumeType(volume->type)) {
                            const bool show = selected_crop_volume && volume->id == selected_node->id;
                            if (static_cast<bool>(volume->visible) != show)
                                sm->setNodeVisibilityTransient(volume->id, show);
                        }
            }

            setPointPickToolsEnabled(viewer_->getAlignTool(), viewer_->getScaleDistanceTool());
            if (sm)
                sm->syncCropBoxToRenderSettings();
            node_bounds_cache_valid_ = false;
            node_selection_bounds_cache_valid_ = false;
        });
        ui::NodeDeselected::when([this](const auto&) {
            if (isCropToolActive())
                leaveCropTool(false, false, true);
            python::cancel_active_operator();
            if (auto* sm = viewer_ ? viewer_->getSceneManager() : nullptr)
                for (const auto* node : sm->getScene().getNodes())
                    if (isCropVolumeType(node->type) && node->visible)
                        sm->setNodeVisibilityTransient(node->id, false);
            setPointPickToolsEnabled(viewer_->getAlignTool(), viewer_->getScaleDistanceTool());
            node_bounds_cache_valid_ = false;
            node_selection_bounds_cache_valid_ = false;
        });

        // Node removal changes the selection, not the chosen tool. Reset only
        // when the whole scene is cleared so the editor and registry stay in sync.
        state::SceneCleared::when([this](const auto&) { deactivateAllTools(); });

        lfs::core::events::tools::SetToolbarTool::when([this](const auto& e) {
            auto& editor = viewer_->getEditorContext();
            const auto tool = static_cast<ToolType>(e.tool_mode);

            auto& registry = UnifiedToolRegistry::instance();
            // Undo can reselect a crop volume without entering the crop tool.
            const auto* const sm = viewer_->getSceneManager();
            const bool selected_crop_volume = sm && sm->hasSelectedNode() &&
                                              isCropVolumeType(sm->getSelectedNodeType());
            if (registry.getActiveTool() == "builtin.cropbox" ||
                (tool != ToolType::None && selected_crop_volume)) {
                leaveCropTool(true, true, true);
            } else if (editor.hasActiveOperator() && tool != ToolType::Selection) {
                python::cancel_active_operator();
            }

            editor.setActiveTool(tool);
            const char* tool_id = nullptr;
            switch (tool) {
            case ToolType::None: break;
            case ToolType::Selection: tool_id = "builtin.select"; break;
            case ToolType::Translate: tool_id = "builtin.translate"; break;
            case ToolType::Rotate: tool_id = "builtin.rotate"; break;
            case ToolType::Scale: tool_id = "builtin.scale"; break;
            case ToolType::Mirror: tool_id = "builtin.mirror"; break;
            case ToolType::Align: tool_id = "builtin.align"; break;
            case ToolType::ScaleDistance: tool_id = "builtin.scale_distance"; break;
            }
            if (tool_id)
                registry.setActiveTool(tool_id);
            else
                registry.clearActiveTool();

            switch (tool) {
            case ToolType::Translate:
                current_operation_ = GizmoOperation::Translate;
                LOG_DEBUG("SetToolbarTool: TRANSLATE");
                break;
            case ToolType::Rotate:
                current_operation_ = GizmoOperation::Rotate;
                LOG_DEBUG("SetToolbarTool: ROTATE");
                break;
            case ToolType::Scale:
                current_operation_ = GizmoOperation::Scale;
                LOG_DEBUG("SetToolbarTool: SCALE");
                break;
            case ToolType::Selection:
                // Plain select activation returns to brush selection; explicit
                // flyout submode picks immediately override this event.
                setSelectionSubMode(SelectionSubMode::Centers);
                registry.setActiveSubmode("centers");
                break;
            default:
                LOG_DEBUG("SetToolbarTool: tool_mode={}", e.tool_mode);
                break;
            }

            // Selecting an editing tool gives the viewport its working space,
            // but clearing the active tool is a lifecycle operation and must
            // not mutate the live workspace.
            if (tool != ToolType::None) {
                if (auto* gui = viewer_->getGuiManager()) {
                    gui->panelLayout().setShowSequencer(false);
                }
            }
        });

        lfs::core::events::tools::SetSelectionSubMode::when([this](const auto& e) {
            setSelectionSubMode(static_cast<SelectionSubMode>(e.selection_mode));
            UnifiedToolRegistry::instance().setActiveSubmode(selectionSubModeId(selection_mode_));

            if (auto* const tool = viewer_->getSelectionTool()) {
                tool->onSelectionModeChanged();
            }
        });

        lfs::core::events::tools::ExecuteMirror::when([this](const auto& e) {
            auto* sm = viewer_->getSceneManager();
            if (sm) {
                sm->executeMirror(static_cast<lfs::core::MirrorAxis>(e.axis));
            }
        });

        lfs::core::events::tools::CancelActiveOperator::when([](const auto&) {
            lfs::python::cancel_active_operator();
        });

        cmd::ApplyCropBox::when([this](const auto&) {
            auto* const sm = viewer_->getSceneManager();
            if (isCropToolActive()) {
                applyActiveCropTool();
                return;
            }

            if (!sm)
                return;

            const core::NodeId cropbox_id = sm->getSelectedNodeCropBoxId();
            if (cropbox_id == core::NULL_NODE)
                return;

            const auto* cropbox_node = sm->getScene().getNodeById(cropbox_id);
            if (!cropbox_node || !cropbox_node->cropbox)
                return;

            if (!sm->canApplyCropToNode(cropbox_node->parent_id))
                return;

            op::TransactionGuard transaction("Crop Box");
            cap::CropBoxUpdate update;
            update.has_use = true;
            update.use = true;
            if (auto* rm = viewer_->getRenderingManager()) {
                if (auto result = cap::updateCropBox(*sm, rm, cropbox_id, update); !result) {
                    LOG_WARN("Failed to apply cropbox '{}': {}", cropbox_node->name, result.error());
                    return;
                }
            } else if (auto result = cap::updateCropBox(*sm, nullptr, cropbox_id, update); !result) {
                LOG_WARN("Failed to apply cropbox '{}': {}", cropbox_node->name, result.error());
                return;
            }
            if (auto* rm = viewer_->getRenderingManager()) {
                auto settings = rm->getSettings();
                settings.show_crop_box = cropbox_node->visible;
                settings.use_crop_box = true;
                rm->updateSettings(settings, DirtyFlag::SPLATS | DirtyFlag::OVERLAY);
                viewer_->projectMarkSessionChapterCommitted("VIEW");
            }

            lfs::geometry::BoundingBox crop_box;
            crop_box.setBounds(cropbox_node->cropbox->min, cropbox_node->cropbox->max);
            crop_box.setworld2BBox(glm::inverse(scene_coords::nodeDataWorldTransform(sm->getScene(), cropbox_id)));
            cmd::CropPLY{.crop_box = crop_box, .inverse = cropbox_node->cropbox->inverse, .target_node_id = cropbox_node->parent_id}.emit();
            triggerCropFlash();
            if (deleteCropVolumeAfterApply(cropbox_id))
                transaction.commit();
        });

        cmd::ApplyEllipsoid::when([this](const auto&) {
            if (isCropToolActive()) {
                applyActiveCropTool();
                return;
            }

            auto* const sm = viewer_->getSceneManager();
            if (!sm)
                return;

            const core::NodeId ellipsoid_id = sm->getSelectedNodeEllipsoidId();
            if (ellipsoid_id == core::NULL_NODE)
                return;

            const auto* ellipsoid_node = sm->getScene().getNodeById(ellipsoid_id);
            if (!ellipsoid_node || !ellipsoid_node->ellipsoid)
                return;

            if (!sm->canApplyCropToNode(ellipsoid_node->parent_id))
                return;

            op::TransactionGuard transaction("Crop Ellipsoid");
            cap::EllipsoidUpdate update;
            update.has_use = true;
            update.use = true;
            if (auto* rm = viewer_->getRenderingManager()) {
                if (auto result = cap::updateEllipsoid(*sm, rm, ellipsoid_id, update); !result) {
                    LOG_WARN("Failed to apply ellipsoid '{}': {}", ellipsoid_node->name, result.error());
                    return;
                }
            } else if (auto result = cap::updateEllipsoid(*sm, nullptr, ellipsoid_id, update); !result) {
                LOG_WARN("Failed to apply ellipsoid '{}': {}", ellipsoid_node->name, result.error());
                return;
            }
            if (auto* rm = viewer_->getRenderingManager()) {
                auto settings = rm->getSettings();
                settings.show_ellipsoid = ellipsoid_node->visible;
                settings.use_ellipsoid = true;
                rm->updateSettings(settings, DirtyFlag::SPLATS | DirtyFlag::OVERLAY);
                viewer_->projectMarkSessionChapterCommitted("VIEW");
            }

            cmd::CropPLYEllipsoid{
                .world_transform = scene_coords::nodeDataWorldTransform(sm->getScene(), ellipsoid_id),
                .radii = ellipsoid_node->ellipsoid->radii,
                .inverse = ellipsoid_node->ellipsoid->inverse,
                .target_node_id = ellipsoid_node->parent_id}
                .emit();
            triggerCropFlash();
            if (deleteCropVolumeAfterApply(ellipsoid_id))
                transaction.commit();
        });

        cmd::ApplyCropPolygon::when([this](const auto&) {
            if (isCropToolActive()) {
                applyActiveCropTool();
                return;
            }

            auto* const sm = viewer_->getSceneManager();
            const core::NodeId polygon_id = sm ? sm->getSelectedNodeCropPolygonId() : core::NULL_NODE;
            const auto* polygon_node = sm ? sm->getScene().getNodeById(polygon_id) : nullptr;
            if (!polygon_node || !polygon_node->crop_polygon || !sm->canApplyCropToNode(polygon_node->parent_id))
                return;

            op::TransactionGuard transaction("Crop Polygon");
            emitCropPolygon(sm->getScene(), *polygon_node);
            triggerCropFlash();
            if (deleteCropVolumeAfterApply(polygon_id))
                transaction.commit();
        });

        cmd::ToggleCropInverse::when([this](const auto&) { toggleSelectedCropInverse(); });

        cmd::CycleSelectionVisualization::when([this](const auto&) {
            if (viewer_->getEditorContext().getActiveTool() != ToolType::Selection)
                return;
            auto* const rm = viewer_->getRenderingManager();
            if (!rm)
                return;

            auto settings = rm->getSettings();
            const bool centers = settings.show_center_markers;
            const bool rings = settings.show_rings;

            settings.show_center_markers = !centers && !rings;
            settings.show_rings = centers && !rings;
            rm->updateSettings(settings);
            viewer_->projectMarkSessionChapterCommitted("VIEW");
        });
    }

    void GizmoManager::updateToolState(const UIContext& ctx, bool ui_hidden) {
        auto* const scene_manager = ctx.viewer->getSceneManager();
        auto* const align_tool = ctx.viewer->getAlignTool();
        auto* const scale_distance_tool = ctx.viewer->getScaleDistanceTool();
        auto* const selection_tool = ctx.viewer->getSelectionTool();
        auto* const rendering_manager = ctx.viewer->getRenderingManager();
        const bool has_selected_node = scene_manager && scene_manager->hasSelectedNode();
        const std::string active_tool_id = UnifiedToolRegistry::instance().getActiveTool();
        const std::string gizmo_type = ctx.editor ? ctx.editor->getGizmoType() : std::string{};
        const bool is_crop_tool = active_tool_id == "builtin.cropbox";
        const bool is_selection_mode = active_tool_id == "builtin.select";
        const bool is_selection_volume_tool = is_selection_mode && isSelectionVolumeSubMode(selection_mode_);
        const bool selected_crop_volume = has_selected_node && scene_manager &&
                                          isCropVolumeType(scene_manager->getSelectedNodeType());

        ToolStateStamp stamp;
        stamp.valid = true;
        stamp.ui_hidden = ui_hidden;
        stamp.has_scene_manager = scene_manager != nullptr;
        stamp.has_selected_node = has_selected_node;
        stamp.align_tool = align_tool;
        stamp.scale_distance_tool = scale_distance_tool;
        stamp.selection_tool = selection_tool;
        stamp.rendering_manager = rendering_manager;
        stamp.active_tool_id = active_tool_id;
        stamp.gizmo_type = gizmo_type;
        stamp.selected_node_name = scene_manager ? scene_manager->getSelectedNodeName() : std::string{};
        stamp.selection_mode = selection_mode_;
        if (stamp == last_tool_state_stamp_)
            return;
        last_tool_state_stamp_ = stamp;
        resetIdleGizmoHover();

        if (rendering_manager)
            rendering_manager->setGaussianSelectionVisible(is_selection_mode);

        if (scene_manager && !ui_hidden) {
            bool is_transform_tool = false;
            if (has_selected_node && !gizmo_type.empty()) {
                is_transform_tool = !is_crop_tool;
                if (gizmo_type == "translate") {
                    node_gizmo_operation_ = GizmoOperation::Translate;
                    current_operation_ = GizmoOperation::Translate;
                } else if (gizmo_type == "rotate") {
                    node_gizmo_operation_ = GizmoOperation::Rotate;
                    current_operation_ = GizmoOperation::Rotate;
                } else if (gizmo_type == "scale") {
                    node_gizmo_operation_ = GizmoOperation::Scale;
                    current_operation_ = GizmoOperation::Scale;
                } else {
                    is_transform_tool = false;
                }
            } else if (has_selected_node &&
                       (active_tool_id == "builtin.translate" || active_tool_id == "builtin.rotate" ||
                        active_tool_id == "builtin.scale")) {
                is_transform_tool = true;
                node_gizmo_operation_ = current_operation_;
            }
            if (selected_crop_volume)
                is_transform_tool = false;
            show_node_gizmo_ = is_transform_tool;
            if (!is_selection_volume_tool && selection_volume_source_generation_ != 0)
                clearSelectionVolumeState();

            if (is_crop_tool && has_selected_node) {
                if (ensureCropToolState())
                    updateCropToolOverlayState();
            } else if (is_selection_volume_tool) {
                if (ensureCropToolState())
                    updateCropToolOverlayState();
            } else {
                crop_tool_initialized_ = false;
                crop_tool_target_node_id_ = core::NULL_NODE;
                clearCropToolOverlayState();
            }

            const bool is_align_mode = (active_tool_id == "builtin.align");
            const bool is_scale_distance_mode = (active_tool_id == "builtin.scale_distance");

            const bool enable_align = is_align_mode && has_selected_node && !selected_crop_volume;
            const bool enable_scale_distance = is_scale_distance_mode && has_selected_node && !selected_crop_volume;

            setPointPickToolsEnabled(align_tool, scale_distance_tool, enable_align, enable_scale_distance);
            if (selection_tool)
                selection_tool->setEnabled(is_selection_mode && !selected_crop_volume);

            if (is_selection_mode) {
                if (rendering_manager) {
                    rendering_manager->setSelectionPreviewMode(toSelectionPreviewMode(selection_mode_));

                    if (selection_mode_ != previous_selection_mode_) {
                        if (selection_tool)
                            selection_tool->onSelectionModeChanged();
                        previous_selection_mode_ = selection_mode_;
                    }
                }
            }

        } else {
            show_node_gizmo_ = false;
            if (selection_volume_source_generation_ != 0)
                clearSelectionVolumeState();
            crop_tool_initialized_ = false;
            crop_tool_target_node_id_ = core::NULL_NODE;
            clearCropToolOverlayState();
            setPointPickToolsEnabled(ctx.viewer->getAlignTool(), ctx.viewer->getScaleDistanceTool());
            if (auto* const tool = ctx.viewer->getSelectionTool())
                tool->setEnabled(false);
        }
    }

    void GizmoManager::renderNodeTransformGizmo(const UIContext& ctx, const ViewportLayout& viewport) {
        if (!show_node_gizmo_)
            return;

        auto* scene_manager = ctx.viewer->getSceneManager();
        if (!scene_manager || !scene_manager->hasSelectedNode())
            return;

        const auto selected_type = scene_manager->getSelectedNodeType();
        if (isCropVolumeType(selected_type) ||
            selected_type == core::NodeType::KEYFRAME || selected_type == core::NodeType::KEYFRAME_GROUP)
            return;

        const auto& scene = scene_manager->getScene();
        const auto transform_targets = cap::resolveEditableTransformSelection(
            *scene_manager, std::nullopt, cap::TransformTargetPolicy::RequireAllEditable);
        if (!transform_targets)
            return;

        const auto& target_names = transform_targets->node_names;
        bool any_visible = false;
        for (const auto& name : target_names) {
            if (const auto* node = scene.getNode(name)) {
                if (scene.isNodeEffectivelyVisible(node->id)) {
                    any_visible = true;
                    break;
                }
            }
        }
        if (!any_visible)
            return;

        auto* render_manager = ctx.viewer->getRenderingManager();
        if (!render_manager)
            return;

        const auto& settings = render_manager->getSettings();
        const bool is_multi_selection = (target_names.size() > 1);
        const bool use_selection_mode = !is_multi_selection || multi_transform_mode_ == MultiTransformMode::Selection;
        const bool use_individual_mode = is_multi_selection && multi_transform_mode_ == MultiTransformMode::Individual;
        std::optional<std::vector<std::string>> top_level_target_names;
        const auto get_top_level_target_names = [&]() -> const std::vector<std::string>& {
            if (!top_level_target_names) {
                top_level_target_names = is_multi_selection
                                             ? gizmo_ops::topLevelTransformTargets(scene, target_names)
                                             : target_names;
            }
            return *top_level_target_names;
        };

        const auto active_panel = resolveActiveGizmoPanel(ctx.viewer, viewport);
        if (!active_panel || !active_panel->valid())
            return;

        auto& vp = *active_panel->viewport;
        const glm::mat4 view = vp.getViewMatrix();
        const glm::ivec2 vp_size(static_cast<int>(active_panel->size.x), static_cast<int>(active_panel->size.y));
        const glm::mat4 projection = lfs::rendering::createProjectionMatrixFromFocal(
            vp_size, settings.focal_length_mm, settings.orthographic, vp.ortho_scale_override.value_or(settings.ortho_scale));

        const bool use_world_space = (transform_space_ == TransformSpace::World) || is_multi_selection;

        const glm::vec3 local_pivot = (pivot_mode_ == PivotMode::Origin)
                                          ? glm::vec3(0.0f)
                                          : transform_targets->local_center;

        bool has_valid_bounds = false;
        const bool use_single_bounds_scale = !is_multi_selection && node_gizmo_operation_ == GizmoOperation::Scale;
        const bool use_selection_bounds_scale = is_multi_selection && use_selection_mode &&
                                                node_gizmo_operation_ == GizmoOperation::Scale;
        const bool use_bounds_scale = use_single_bounds_scale || use_selection_bounds_scale;

        const auto* first_node = (!is_multi_selection && !target_names.empty())
                                     ? scene.getNode(target_names.front())
                                     : nullptr;

        glm::vec3 bounds_min(0.0f), bounds_max(0.0f);
        glm::mat4 world_transform(1.0f);
        glm::vec3 world_scale(1.0f);
        glm::mat3 node_rotation(1.0f);

        if (use_single_bounds_scale && first_node) {
            world_transform = scene_coords::nodeVisualizerWorldTransform(scene, first_node->id);
            world_scale = extractScale(world_transform);
            node_rotation = extractRotation(world_transform);

            if (node_bounds_scale_active_) {
                has_valid_bounds = true;
                bounds_min = node_bounds_min_;
                bounds_max = node_bounds_max_;
            } else {
                if (!node_bounds_cache_valid_ || node_bounds_cache_node_id_ != first_node->id) {
                    if (scene.getNodeBounds(first_node->id, node_bounds_cache_min_, node_bounds_cache_max_)) {
                        node_bounds_cache_valid_ = true;
                        node_bounds_cache_node_id_ = first_node->id;
                    } else {
                        node_bounds_cache_valid_ = false;
                    }
                }
                if (node_bounds_cache_valid_) {
                    has_valid_bounds = true;
                    bounds_min = node_bounds_cache_min_;
                    bounds_max = node_bounds_cache_max_;
                }
            }
        }
        if (use_selection_bounds_scale) {
            if (node_selection_bounds_scale_active_) {
                has_valid_bounds = true;
                bounds_min = node_selection_bounds_min_;
                bounds_max = node_selection_bounds_max_;
            } else {
                struct GroupBoundsCacheKeyEntry {
                    core::NodeId id = core::NULL_NODE;
                    glm::mat4 visualizer_world_transform{1.0f};
                };

                const auto& top_level_names = get_top_level_target_names();
                std::vector<GroupBoundsCacheKeyEntry> key_entries;
                key_entries.reserve(top_level_names.size());
                for (const auto& name : top_level_names) {
                    if (const auto* node = scene.getNode(name)) {
                        key_entries.push_back(GroupBoundsCacheKeyEntry{
                            .id = node->id,
                            .visualizer_world_transform = scene_coords::nodeVisualizerWorldTransform(scene, node->id),
                        });
                    }
                }
                std::sort(key_entries.begin(), key_entries.end(), [](const auto& a, const auto& b) {
                    return a.id < b.id;
                });

                std::vector<core::NodeId> cache_node_ids;
                std::vector<glm::mat4> cache_world_transforms;
                cache_node_ids.reserve(key_entries.size());
                cache_world_transforms.reserve(key_entries.size());
                for (const auto& entry : key_entries) {
                    cache_node_ids.push_back(entry.id);
                    cache_world_transforms.push_back(entry.visualizer_world_transform);
                }

                bool cache_matches = node_selection_bounds_cache_valid_ &&
                                     node_selection_bounds_cache_node_ids_ == cache_node_ids &&
                                     node_selection_bounds_cache_visualizer_world_transforms_.size() ==
                                         cache_world_transforms.size();
                if (cache_matches) {
                    for (size_t i = 0; i < cache_world_transforms.size(); ++i) {
                        if (node_selection_bounds_cache_visualizer_world_transforms_[i] != cache_world_transforms[i]) {
                            cache_matches = false;
                            break;
                        }
                    }
                }

                if (cache_matches) {
                    has_valid_bounds = true;
                    bounds_min = node_selection_bounds_cache_min_;
                    bounds_max = node_selection_bounds_cache_max_;
                } else if (gizmo_ops::computeCombinedVisualizerWorldBounds(
                               scene, top_level_names, bounds_min, bounds_max)) {
                    has_valid_bounds = true;
                    node_selection_bounds_cache_valid_ = true;
                    node_selection_bounds_cache_node_ids_ = std::move(cache_node_ids);
                    node_selection_bounds_cache_visualizer_world_transforms_ = std::move(cache_world_transforms);
                    node_selection_bounds_cache_min_ = bounds_min;
                    node_selection_bounds_cache_max_ = bounds_max;
                } else {
                    node_selection_bounds_cache_valid_ = false;
                    node_selection_bounds_cache_node_ids_.clear();
                    node_selection_bounds_cache_visualizer_world_transforms_.clear();
                }
            }
            world_transform = glm::mat4(1.0f);
            world_scale = glm::vec3(1.0f);
            node_rotation = glm::mat3(1.0f);
        }

        const bool actually_using_bounds = use_bounds_scale && has_valid_bounds;

        const glm::vec3 transform_gizmo_position = (node_gizmo_active_ && !node_bounds_scale_active_ &&
                                                    !node_selection_bounds_scale_active_)
                                                       ? gizmo_pivot_
                                                       : (is_multi_selection
                                                              ? transform_targets->world_center
                                                              : (first_node
                                                                     ? glm::vec3(scene_coords::nodeVisualizerWorldTransform(scene, first_node->id) *
                                                                                 glm::vec4(local_pivot, 1.0f))
                                                                     : glm::vec3(0.0f)));
        glm::mat4 transform_gizmo_matrix(1.0f);
        transform_gizmo_matrix[3] = glm::vec4(transform_gizmo_position, 1.0f);
        if (!is_multi_selection) {
            const glm::mat3 rotation_scale(first_node ? scene_coords::nodeVisualizerWorldTransform(scene, first_node->id)
                                                      : glm::mat4(1.0f));
            transform_gizmo_matrix[0] = glm::vec4(rotation_scale[0], 0.0f);
            transform_gizmo_matrix[1] = glm::vec4(rotation_scale[1], 0.0f);
            transform_gizmo_matrix[2] = glm::vec4(rotation_scale[2], 0.0f);
        }

        glm::mat4 gizmo_matrix(1.0f);
        if (actually_using_bounds) {
            const glm::vec3 bounds_size = bounds_max - bounds_min;
            const glm::vec3 bounds_center_local = (bounds_min + bounds_max) * 0.5f;

            glm::vec3 center_world;
            if (use_selection_bounds_scale) {
                const glm::vec3 display_size = node_selection_bounds_scale_active_
                                                   ? (node_selection_bounds_max_ - node_selection_bounds_min_) * gizmo_cumulative_scale_
                                                   : bounds_size;
                center_world = node_selection_bounds_scale_active_ ? gizmo_pivot_ : bounds_center_local;
                gizmo_matrix[3] = glm::vec4(center_world, 1.0f);
                gizmo_matrix[0] = glm::vec4(display_size.x, 0.0f, 0.0f, 0.0f);
                gizmo_matrix[1] = glm::vec4(0.0f, display_size.y, 0.0f, 0.0f);
                gizmo_matrix[2] = glm::vec4(0.0f, 0.0f, display_size.z, 0.0f);
            } else if (node_bounds_scale_active_) {
                const glm::vec3 original_bounds_size = node_bounds_max_ - node_bounds_min_;
                const glm::vec3 current_size = original_bounds_size * gizmo_cumulative_scale_;
                const glm::vec3 scaled_center = (node_bounds_min_ + node_bounds_max_) * 0.5f;
                center_world = glm::vec3(world_transform * glm::vec4(scaled_center, 1.0f));

                gizmo_matrix[3] = glm::vec4(center_world, 1.0f);
                const glm::vec3 display_size = current_size * node_bounds_world_scale_;
                gizmo_matrix[0] = glm::vec4(node_rotation[0] * display_size.x, 0.0f);
                gizmo_matrix[1] = glm::vec4(node_rotation[1] * display_size.y, 0.0f);
                gizmo_matrix[2] = glm::vec4(node_rotation[2] * display_size.z, 0.0f);
            } else {
                center_world = glm::vec3(world_transform * glm::vec4(bounds_center_local, 1.0f));
                gizmo_matrix[3] = glm::vec4(center_world, 1.0f);
                const glm::vec3 display_size = bounds_size * world_scale;
                gizmo_matrix[0] = glm::vec4(node_rotation[0] * display_size.x, 0.0f);
                gizmo_matrix[1] = glm::vec4(node_rotation[1] * display_size.y, 0.0f);
                gizmo_matrix[2] = glm::vec4(node_rotation[2] * display_size.z, 0.0f);
            }
        } else {
            gizmo_matrix = transform_gizmo_matrix;
        }

        NativeOverlayDrawList overlay_drawlist;
        const glm::vec2 clip_min(active_panel->pos.x, active_panel->pos.y);
        const glm::vec2 clip_max(clip_min.x + active_panel->size.x, clip_min.y + active_panel->size.y);
        overlay_drawlist.PushClipRect(clip_min, clip_max, true);
        const auto& frame_input = viewer_->getWindowManager()->frameInput();
        const NativeGizmoInput gizmo_input = nativeGizmoInputFromFrame(frame_input);
        const bool snap_modifier = nativeControlModifierDown(frame_input);

        const bool gizmo_uses_local_axes = actually_using_bounds || !use_world_space;

        const bool use_bounds_gizmo = actually_using_bounds;
        const bool use_translation_gizmo = node_gizmo_operation_ == GizmoOperation::Translate && !actually_using_bounds;
        const bool use_rotation_gizmo = node_gizmo_operation_ == GizmoOperation::Rotate && !actually_using_bounds;
        const bool use_scale_gizmo = node_gizmo_operation_ == GizmoOperation::Scale;
        bool is_using = false;
        bool gizmo_changed = false;
        bool gizmo_released = false;
        bool gizmo_returned_to_start = false;
        glm::mat4 delta_matrix(1.0f);
        bool bounds_result_valid = false;
        bool bounds_gizmo_active = false;
        glm::vec3 bounds_result_center_world(0.0f);
        glm::vec3 bounds_result_local_size(0.0f);
        ScaleGizmoResult scale_result;
        const bool scale_gizmo_has_priority = use_scale_gizmo && (isScaleGizmoHovered() || isScaleGizmoActive());

        if (use_bounds_gizmo) {
            const glm::vec3 safe_world_scale = glm::max(node_bounds_scale_active_ ? node_bounds_world_scale_ : world_scale,
                                                        glm::vec3(1e-6f));
            BoundsGizmoConfig bounds_config;
            bounds_config.id = panelGizmoId(NODE_GIZMO_ID_BASE, active_panel->panel);
            bounds_config.viewport_pos = active_panel->pos;
            bounds_config.viewport_size = active_panel->size;
            bounds_config.view = view;
            bounds_config.projection = projection;
            bounds_config.center_world = glm::vec3(gizmo_matrix[3]);
            bounds_config.orientation_world = userFacingLocalRotation(gizmo_matrix);
            bounds_config.half_extents_world = extractScale(gizmo_matrix) * 0.5f;
            bounds_config.min_half_extents_world = safe_world_scale * (MIN_GIZMO_SCALE * 0.5f);
            bounds_config.draw_list = &overlay_drawlist;
            bounds_config.input = gizmo_input;
            bounds_config.input_enabled = !scale_gizmo_has_priority;
            bounds_config.snap = snap_modifier;
            bounds_config.snap_ratio = SCALE_SNAP_RATIO;

            const auto bounds_result = drawBoundsGizmo(bounds_config);
            is_using = is_using || bounds_result.active;
            gizmo_changed = gizmo_changed || bounds_result.changed;
            gizmo_released = gizmo_released || bounds_result.released;
            gizmo_returned_to_start = gizmo_returned_to_start || bounds_result.returned_to_start;
            bounds_gizmo_active = bounds_result.active;
            if (bounds_result.active) {
                const glm::mat3 box_rotation = extractRotation(gizmo_matrix);
                const glm::vec3 full_size = bounds_result.half_extents_world * 2.0f;
                gizmo_matrix[3] = glm::vec4(bounds_result.center_world, 1.0f);
                gizmo_matrix[0] = glm::vec4(box_rotation[0] * full_size.x, 0.0f);
                gizmo_matrix[1] = glm::vec4(box_rotation[1] * full_size.y, 0.0f);
                gizmo_matrix[2] = glm::vec4(box_rotation[2] * full_size.z, 0.0f);
            }
            if (bounds_result.changed) {
                bounds_result_valid = true;
                bounds_result_center_world = bounds_result.center_world;
                bounds_result_local_size =
                    glm::max((bounds_result.half_extents_world * 2.0f) / safe_world_scale,
                             glm::vec3(MIN_GIZMO_SCALE));
            }
            if (bounds_result.hovered || bounds_result.active) {
                guiFocusState().want_capture_mouse = true;
            }
        } else if (use_translation_gizmo) {
            TranslationGizmoConfig translation_config;
            translation_config.id = panelGizmoId(NODE_GIZMO_ID_BASE, active_panel->panel);
            translation_config.viewport_pos = active_panel->pos;
            translation_config.viewport_size = active_panel->size;
            translation_config.view = view;
            translation_config.projection = projection;
            translation_config.pivot_world = glm::vec3(gizmo_matrix[3]);
            translation_config.orientation_world =
                gizmo_uses_local_axes ? userFacingLocalRotation(gizmo_matrix) : glm::mat3(1.0f);
            translation_config.draw_list = &overlay_drawlist;
            translation_config.input = gizmo_input;
            translation_config.snap = snap_modifier;
            translation_config.snap_units = TRANSLATE_SNAP_UNITS;

            const auto translation_result = drawTranslationGizmo(translation_config);
            is_using = translation_result.active;
            gizmo_changed = translation_result.changed;
            gizmo_released = translation_result.released;
            gizmo_returned_to_start = translation_result.returned_to_start;
            delta_matrix = glm::translate(glm::mat4(1.0f), translation_result.delta_translation);
            if (translation_result.active) {
                gizmo_matrix[3] =
                    glm::vec4(translation_config.pivot_world + translation_result.total_translation, 1.0f);
            }
            if (translation_result.hovered || translation_result.active) {
                guiFocusState().want_capture_mouse = true;
            }
        } else if (use_rotation_gizmo) {
            RotationGizmoConfig rotation_config;
            rotation_config.id = panelGizmoId(NODE_GIZMO_ID_BASE, active_panel->panel);
            rotation_config.viewport_pos = active_panel->pos;
            rotation_config.viewport_size = active_panel->size;
            rotation_config.view = view;
            rotation_config.projection = projection;
            rotation_config.pivot_world = glm::vec3(gizmo_matrix[3]);
            rotation_config.orientation_world =
                gizmo_uses_local_axes ? userFacingLocalRotation(gizmo_matrix) : glm::mat3(1.0f);
            rotation_config.draw_list = &overlay_drawlist;
            rotation_config.input = gizmo_input;
            rotation_config.snap = snap_modifier;
            rotation_config.snap_degrees = ROTATION_SNAP_DEGREES;

            const auto rotation_result = drawRotationGizmo(rotation_config);
            is_using = rotation_result.active;
            gizmo_changed = rotation_result.changed;
            gizmo_released = rotation_result.released;
            gizmo_returned_to_start = rotation_result.returned_to_start;
            delta_matrix = glm::mat4(rotation_result.delta_rotation);
            if (rotation_result.hovered || rotation_result.active) {
                guiFocusState().want_capture_mouse = true;
            }
        }

        if (use_scale_gizmo) {
            ScaleGizmoConfig scale_config;
            scale_config.id = panelGizmoId(NODE_GIZMO_ID_BASE, active_panel->panel);
            scale_config.viewport_pos = active_panel->pos;
            scale_config.viewport_size = active_panel->size;
            scale_config.view = view;
            scale_config.projection = projection;
            scale_config.pivot_world = glm::vec3(transform_gizmo_matrix[3]);
            scale_config.orientation_world =
                gizmo_uses_local_axes ? userFacingLocalRotation(transform_gizmo_matrix) : glm::mat3(1.0f);
            scale_config.draw_list = &overlay_drawlist;
            scale_config.input = gizmo_input;
            scale_config.input_enabled = !isBoundsGizmoActive();
            scale_config.snap = snap_modifier;
            scale_config.snap_ratio = SCALE_SNAP_RATIO;

            scale_result = drawScaleGizmo(scale_config);
            is_using = is_using || scale_result.active;
            gizmo_changed = gizmo_changed || scale_result.changed;
            gizmo_released = gizmo_released || scale_result.released;
            gizmo_returned_to_start = gizmo_returned_to_start || scale_result.returned_to_start;
            if (scale_result.changed) {
                delta_matrix = glm::scale(glm::mat4(1.0f), scale_result.delta_scale);
                transform_gizmo_matrix[0] *= scale_result.delta_scale.x;
                transform_gizmo_matrix[1] *= scale_result.delta_scale.y;
                transform_gizmo_matrix[2] *= scale_result.delta_scale.z;
            }
            if (scale_result.active) {
                gizmo_matrix = transform_gizmo_matrix;
            }
            if (scale_result.hovered || scale_result.active) {
                guiFocusState().want_capture_mouse = true;
            }
        }

        if (is_using && !node_gizmo_active_) {
            node_gizmo_active_ = true;
            gizmo_pivot_ = glm::vec3(gizmo_matrix[3]);
            gizmo_cumulative_rotation_ = glm::mat3(1.0f);
            gizmo_cumulative_scale_ = glm::vec3(1.0f);

            if (actually_using_bounds && use_selection_bounds_scale && bounds_gizmo_active) {
                glm::vec3 fresh_min, fresh_max;
                if (gizmo_ops::computeCombinedVisualizerWorldBounds(
                        scene, get_top_level_target_names(), fresh_min, fresh_max)) {
                    node_selection_bounds_min_ = fresh_min;
                    node_selection_bounds_max_ = fresh_max;
                    node_selection_bounds_scale_active_ = true;
                }
            } else if (actually_using_bounds && first_node && bounds_gizmo_active) {
                glm::vec3 fresh_min, fresh_max;
                if (scene.getNodeBounds(first_node->id, fresh_min, fresh_max)) {
                    node_bounds_min_ = fresh_min;
                    node_bounds_max_ = fresh_max;
                    node_bounds_orig_visualizer_world_transform_ = scene_coords::nodeVisualizerWorldTransform(scene, first_node->id);
                    node_bounds_world_scale_ = world_scale;
                    node_bounds_scale_active_ = true;
                }
            }

            node_gizmo_node_names_ = get_top_level_target_names();

            node_transforms_before_drag_.clear();
            node_original_visualizer_world_transforms_.clear();

            for (const auto& name : node_gizmo_node_names_) {
                const auto* node = scene.getNode(name);
                if (!node)
                    continue;

                const glm::mat4 world_t = scene_coords::nodeVisualizerWorldTransform(scene, node->id);
                const glm::mat4 local_t = node->local_transform.get();
                node_transforms_before_drag_.push_back(local_t);
                node_original_visualizer_world_transforms_.push_back(world_t);
            }
        }

        if (gizmo_changed && (is_using || gizmo_released) && !gizmo_returned_to_start) {
            core::Scene::Transaction txn(scene_manager->getScene());
            if (node_gizmo_operation_ == GizmoOperation::Rotate) {
                const glm::mat3 delta_rot = extractRotation(delta_matrix);
                // Individual mode uses one drag-defined rotation path for the selection and
                // post-applies that cumulative delta to each node's drag-start transform.
                // This keeps each node rotating from its own origin while avoiding drift.
                gizmo_cumulative_rotation_ = delta_rot * gizmo_cumulative_rotation_;
                const auto transforms = use_individual_mode
                                            ? gizmo_ops::computeNodeIndividualLocalTransforms(
                                                  scene_manager->getScene(),
                                                  node_gizmo_node_names_,
                                                  node_original_visualizer_world_transforms_,
                                                  glm::mat4(gizmo_cumulative_rotation_))
                                            : gizmo_ops::computeNodeSharedSelectionLocalTransforms(
                                                  scene_manager->getScene(),
                                                  node_gizmo_node_names_,
                                                  node_original_visualizer_world_transforms_,
                                                  glm::translate(glm::mat4(1.0f), gizmo_pivot_) *
                                                      glm::mat4(gizmo_cumulative_rotation_) *
                                                      glm::translate(glm::mat4(1.0f), -gizmo_pivot_));

                for (const auto& transform : transforms) {
                    scene_manager->setNodeTransform(transform.name, transform.local_transform);
                }
            } else if (node_gizmo_operation_ == GizmoOperation::Scale &&
                       node_selection_bounds_scale_active_) {
                glm::vec3 new_world_size;
                glm::vec3 new_center_world;
                if (bounds_result_valid) {
                    new_world_size = bounds_result_local_size;
                    new_center_world = bounds_result_center_world;
                } else {
                    new_world_size = glm::max(extractScale(gizmo_matrix), glm::vec3(MIN_GIZMO_SCALE));
                    new_center_world = glm::vec3(gizmo_matrix[3]);
                }

                const glm::vec3 original_bounds_size = node_selection_bounds_max_ - node_selection_bounds_min_;
                const glm::vec3 safe_bounds = glm::max(original_bounds_size, glm::vec3(1e-6f));
                const glm::vec3 scale_ratio = new_world_size / safe_bounds;
                gizmo_cumulative_scale_ = scale_ratio;
                gizmo_pivot_ = new_center_world;

                const glm::vec3 original_center = (node_selection_bounds_min_ + node_selection_bounds_max_) * 0.5f;
                const glm::mat4 world_delta = glm::translate(glm::mat4(1.0f), new_center_world) *
                                              glm::scale(glm::mat4(1.0f), scale_ratio) *
                                              glm::translate(glm::mat4(1.0f), -original_center);

                for (const auto& transform : gizmo_ops::computeNodeSharedSelectionLocalTransforms(
                         scene_manager->getScene(),
                         node_gizmo_node_names_,
                         node_original_visualizer_world_transforms_,
                         world_delta)) {
                    scene_manager->setNodeTransform(transform.name, transform.local_transform);
                }
            } else if (node_gizmo_operation_ == GizmoOperation::Scale &&
                       !node_bounds_scale_active_ &&
                       !node_selection_bounds_scale_active_ &&
                       (is_multi_selection || use_world_space)) {
                gizmo_cumulative_scale_ *= extractScale(delta_matrix);
                const auto transforms = use_individual_mode
                                            ? gizmo_ops::computeNodeIndividualLocalTransforms(
                                                  scene_manager->getScene(),
                                                  node_gizmo_node_names_,
                                                  node_original_visualizer_world_transforms_,
                                                  glm::scale(glm::mat4(1.0f), gizmo_cumulative_scale_))
                                            : gizmo_ops::computeNodeSharedSelectionLocalTransforms(
                                                  scene_manager->getScene(),
                                                  node_gizmo_node_names_,
                                                  node_original_visualizer_world_transforms_,
                                                  glm::translate(glm::mat4(1.0f), gizmo_pivot_) *
                                                      glm::scale(glm::mat4(1.0f), gizmo_cumulative_scale_) *
                                                      glm::translate(glm::mat4(1.0f), -gizmo_pivot_));

                for (const auto& transform : transforms) {
                    scene_manager->setNodeTransform(transform.name, transform.local_transform);
                }
            } else if (is_multi_selection) {
                if (node_gizmo_operation_ == GizmoOperation::Translate) {
                    const glm::vec3 new_gizmo_pos(gizmo_matrix[3]);
                    const glm::vec3 delta = new_gizmo_pos - gizmo_pivot_;
                    const glm::mat4 world_delta = glm::translate(glm::mat4(1.0f), delta);

                    for (const auto& transform : gizmo_ops::computeNodeSharedSelectionLocalTransforms(
                             scene_manager->getScene(),
                             node_gizmo_node_names_,
                             node_original_visualizer_world_transforms_,
                             world_delta)) {
                        scene_manager->setNodeTransform(transform.name, transform.local_transform);
                    }
                }
            } else if (node_bounds_scale_active_) {
                assert(!is_multi_selection);
                glm::vec3 new_local_size;
                glm::vec3 new_center_world;
                if (bounds_result_valid) {
                    new_local_size = bounds_result_local_size;
                    new_center_world = bounds_result_center_world;
                } else {
                    const glm::vec3 safe_world_scale = glm::max(node_bounds_world_scale_, glm::vec3(1e-6f));
                    new_local_size = glm::max(
                        extractScale(gizmo_matrix) / safe_world_scale,
                        glm::vec3(MIN_GIZMO_SCALE));
                    new_center_world = glm::vec3(gizmo_matrix[3]);
                }

                const glm::vec3 original_bounds_size = node_bounds_max_ - node_bounds_min_;
                const glm::vec3 safe_bounds = glm::max(original_bounds_size, glm::vec3(1e-6f));
                const glm::vec3 scale_ratio = new_local_size / safe_bounds;

                gizmo_cumulative_scale_ = scale_ratio;

                const glm::vec3 bounds_center_local = (node_bounds_min_ + node_bounds_max_) * 0.5f;
                const auto* node = target_names.empty() ? nullptr : scene.getNode(target_names.front());
                if (node) {
                    glm::mat4 new_world_transform = node_bounds_orig_visualizer_world_transform_;
                    new_world_transform[0] *= scale_ratio.x;
                    new_world_transform[1] *= scale_ratio.y;
                    new_world_transform[2] *= scale_ratio.z;

                    const glm::mat3 new_rs(new_world_transform);
                    new_world_transform[3] = glm::vec4(new_center_world - new_rs * bounds_center_local, 1.0f);

                    if (const auto new_local_transform =
                            scene_coords::nodeLocalTransformFromVisualizerWorld(scene, node->id, new_world_transform)) {
                        scene_manager->setSelectedNodeTransform(*new_local_transform);
                    }
                }
            } else {
                const auto& sm_scene = scene_manager->getScene();
                const auto* node = target_names.empty() ? nullptr : sm_scene.getNode(target_names.front());
                if (node) {
                    const glm::mat4 new_world_transform =
                        gizmo_matrix * glm::translate(glm::mat4(1.0f), -local_pivot);
                    if (const auto new_local_transform =
                            scene_coords::nodeLocalTransformFromVisualizerWorld(sm_scene, node->id, new_world_transform)) {
                        scene_manager->setSelectedNodeTransform(*new_local_transform);
                    }
                }
            }
        }

        // Reset from the saved snapshot when the released pointer lands exactly at press.
        // Incremental rotation and scale deltas can otherwise leave rounding residue.
        if (gizmo_returned_to_start)
            (void)cancelActiveNodeTransformDrag();

        if (!is_using && node_gizmo_active_) {
            node_gizmo_active_ = false;
            node_bounds_scale_active_ = false;
            node_selection_bounds_scale_active_ = false;
            node_bounds_cache_valid_ = false;
            node_selection_bounds_cache_valid_ = false;
            if (render_manager) {
                render_manager->setCropboxGizmoActive(false);
                render_manager->setEllipsoidGizmoActive(false);
            }

            const size_t count = node_gizmo_node_names_.size();
            std::vector<glm::mat4> final_transforms;
            final_transforms.reserve(count);
            for (const auto& name : node_gizmo_node_names_) {
                final_transforms.push_back(scene_manager->getNodeTransform(name));
            }

            bool any_changed = false;
            for (size_t i = 0; i < count; ++i) {
                if (node_transforms_before_drag_[i] != final_transforms[i]) {
                    any_changed = true;
                    break;
                }
            }

            if (any_changed) {
                op::OperatorProperties props;
                props.set("node_names", node_gizmo_node_names_);
                props.set("old_transforms", node_transforms_before_drag_);
                op::operators().invoke(op::BuiltinOp::TransformApplyBatch, &props);

                if (is_multi_selection && use_individual_mode &&
                    (node_gizmo_operation_ == GizmoOperation::Rotate ||
                     node_gizmo_operation_ == GizmoOperation::Scale)) {
                    glm::vec3 updated_min, updated_max;
                    if (gizmo_ops::computeCombinedVisualizerWorldBounds(
                            scene_manager->getScene(), node_gizmo_node_names_, updated_min, updated_max)) {
                        gizmo_pivot_ = (updated_min + updated_max) * 0.5f;
                        if (render_manager) {
                            render_manager->markDirty(DirtyFlag::OVERLAY, FrameReason::SceneChange);
                        }
                    }
                }
            }
        }

        if (node_gizmo_active_ && render_manager) {
            for (const auto& name : target_names) {
                const auto* node = scene.getNode(name);
                if (!node || node->type != core::NodeType::SPLAT)
                    continue;

                const core::NodeId cropbox_id = scene.getCropBoxForSplat(node->id);
                if (cropbox_id != core::NULL_NODE) {
                    const auto* cropbox_node = scene.getNodeById(cropbox_id);
                    if (cropbox_node && cropbox_node->cropbox && scene.isNodeEffectivelyVisible(cropbox_id)) {
                        const glm::mat4 cropbox_world = scene_coords::nodeVisualizerWorldTransform(scene, cropbox_id);
                        const int parent_node_index = scene.getVisibleNodeIndex(node->id);
                        render_manager->setCropboxGizmoState(
                            true, cropbox_node->cropbox->min, cropbox_node->cropbox->max, cropbox_world,
                            cropbox_node->cropbox->enabled, parent_node_index);
                    }
                }

                const core::NodeId ellipsoid_id = scene.getEllipsoidForSplat(node->id);
                if (ellipsoid_id != core::NULL_NODE) {
                    const auto* ellipsoid_node = scene.getNodeById(ellipsoid_id);
                    if (ellipsoid_node && ellipsoid_node->ellipsoid && scene.isNodeEffectivelyVisible(ellipsoid_id)) {
                        const glm::mat4 ellipsoid_world = scene_coords::nodeVisualizerWorldTransform(scene, ellipsoid_id);
                        const int parent_node_index = scene.getVisibleNodeIndex(node->id);
                        render_manager->setEllipsoidGizmoState(
                            true, ellipsoid_node->ellipsoid->radii, ellipsoid_world,
                            ellipsoid_node->ellipsoid->enabled, parent_node_index);
                    }
                }
            }
        }

        overlay_drawlist.PopClipRect();
    }

    void GizmoManager::renderCropToolBoxGizmo(const UIContext& ctx, const ViewportLayout& viewport) {
        auto* const render_manager = ctx.viewer->getRenderingManager();
        if (!render_manager || crop_tool_shape_ != CropToolShape::Box || !ensureCropToolState())
            return;

        const auto active_panel = resolveActiveGizmoPanel(ctx.viewer, viewport);
        if (!active_panel || !active_panel->valid())
            return;

        const auto& settings = render_manager->getSettings();
        auto& vp = *active_panel->viewport;
        const glm::mat4 view = vp.getViewMatrix();
        const glm::ivec2 vp_size(static_cast<int>(active_panel->size.x), static_cast<int>(active_panel->size.y));
        const glm::mat4 projection = lfs::rendering::createProjectionMatrixFromFocal(
            vp_size, settings.focal_length_mm, settings.orthographic, vp.ortho_scale_override.value_or(settings.ortho_scale));

        const glm::vec3 local_size = crop_tool_box_max_ - crop_tool_box_min_;
        const glm::vec3 world_scale = glm::max(extractScale(crop_tool_visualizer_transform_), glm::vec3(1e-6f));
        const glm::mat3 rotation = extractRotation(crop_tool_visualizer_transform_);
        const glm::vec3 local_pivot = (crop_tool_box_min_ + crop_tool_box_max_) * 0.5f;
        const glm::vec3 pivot_world =
            glm::vec3(crop_tool_visualizer_transform_[3]) + rotation * (local_pivot * world_scale);

        const GizmoOperation gizmo_op = current_operation_;
        const bool use_bounds = gizmo_op == GizmoOperation::Scale;
        const bool use_world_space = transform_space_ == TransformSpace::World;
        const bool local_aligned = use_bounds || !use_world_space;

        glm::mat4 gizmo_matrix = glm::translate(glm::mat4(1.0f), pivot_world);
        if (local_aligned)
            gizmo_matrix *= glm::mat4(rotation);
        gizmo_matrix = glm::scale(gizmo_matrix, local_size * world_scale);
        const glm::mat3 local_orientation = isSelectionVolumeMode()
                                                ? selectionVolumeLocalRotation(gizmo_matrix)
                                                : userFacingLocalRotation(gizmo_matrix);

        NativeOverlayDrawList overlay_drawlist;
        const glm::vec2 clip_min(active_panel->pos.x, active_panel->pos.y);
        const glm::vec2 clip_max(clip_min.x + active_panel->size.x, clip_min.y + active_panel->size.y);
        overlay_drawlist.PushClipRect(clip_min, clip_max, true);
        const auto& frame_input = viewer_->getWindowManager()->frameInput();
        const NativeGizmoInput gizmo_input = nativeGizmoInputFromFrame(frame_input);
        const bool snap_modifier = nativeControlModifierDown(frame_input);

        bool changed = false;
        bool is_using = false;
        ScaleGizmoResult scale_result;
        const bool scale_gizmo_has_priority = use_bounds && (isScaleGizmoHovered() || isScaleGizmoActive());

        if (use_bounds) {
            BoundsGizmoConfig bounds_config;
            bounds_config.id = panelGizmoId(CROPBOX_GIZMO_ID_BASE, active_panel->panel);
            bounds_config.viewport_pos = active_panel->pos;
            bounds_config.viewport_size = active_panel->size;
            bounds_config.view = view;
            bounds_config.projection = projection;
            bounds_config.center_world = glm::vec3(gizmo_matrix[3]);
            bounds_config.orientation_world = local_orientation;
            bounds_config.half_extents_world = extractScale(gizmo_matrix) * 0.5f;
            // The crop tool scales the transform; its floor is the transform's axis scale.
            const glm::vec3 box_extent = glm::max(crop_tool_box_max_ - crop_tool_box_min_, glm::vec3(1e-6f));
            bounds_config.min_half_extents_world =
                isCropToolActive() ? box_extent * (gizmo_ops::MIN_VOLUME_AXIS_SCALE * 0.5f)
                                   : world_scale * (MIN_GIZMO_SCALE * 0.5f);
            bounds_config.draw_list = &overlay_drawlist;
            bounds_config.input = gizmo_input;
            bounds_config.input_enabled = !scale_gizmo_has_priority;
            bounds_config.snap = snap_modifier;
            bounds_config.snap_ratio = SCALE_SNAP_RATIO;

            const auto bounds_result = drawBoundsGizmo(bounds_config);
            is_using = is_using || bounds_result.active;
            changed = changed || bounds_result.changed;
            if (bounds_result.changed) {
                if (isCropToolActive()) {
                    const glm::vec3 ratio = bounds_result.half_extents_world * 2.0f / (world_scale * box_extent);
                    for (int axis = 0; axis < 3; ++axis)
                        crop_tool_visualizer_transform_[axis] *= ratio[axis];
                    const glm::vec3 center = (crop_tool_box_min_ + crop_tool_box_max_) * 0.5f;
                    crop_tool_visualizer_transform_[3] = glm::vec4(
                        bounds_result.center_world - glm::mat3(crop_tool_visualizer_transform_) * center, 1.0f);
                } else {
                    const glm::vec3 local_half =
                        glm::max(bounds_result.half_extents_world / world_scale, glm::vec3(MIN_GIZMO_SCALE * 0.5f));
                    crop_tool_box_min_ = -local_half;
                    crop_tool_box_max_ = local_half;
                    crop_tool_visualizer_transform_[3] = glm::vec4(bounds_result.center_world, 1.0f);
                }
            }
            if (bounds_result.hovered || bounds_result.active)
                guiFocusState().want_capture_mouse = true;
        } else if (gizmo_op == GizmoOperation::Translate) {
            TranslationGizmoConfig translation_config;
            translation_config.id = panelGizmoId(CROPBOX_GIZMO_ID_BASE, active_panel->panel);
            translation_config.viewport_pos = active_panel->pos;
            translation_config.viewport_size = active_panel->size;
            translation_config.view = view;
            translation_config.projection = projection;
            translation_config.pivot_world = pivot_world;
            translation_config.orientation_world = local_aligned ? local_orientation : glm::mat3(1.0f);
            translation_config.draw_list = &overlay_drawlist;
            translation_config.input = gizmo_input;
            translation_config.snap = snap_modifier;
            translation_config.snap_units = TRANSLATE_SNAP_UNITS;

            const auto translation_result = drawTranslationGizmo(translation_config);
            is_using = is_using || translation_result.active;
            changed = changed || translation_result.changed;
            if (translation_result.changed)
                crop_tool_visualizer_transform_[3] += glm::vec4(translation_result.delta_translation, 0.0f);
            if (translation_result.hovered || translation_result.active)
                guiFocusState().want_capture_mouse = true;
        } else if (gizmo_op == GizmoOperation::Rotate) {
            RotationGizmoConfig rotation_config;
            rotation_config.id = panelGizmoId(CROPBOX_GIZMO_ID_BASE, active_panel->panel);
            rotation_config.viewport_pos = active_panel->pos;
            rotation_config.viewport_size = active_panel->size;
            rotation_config.view = view;
            rotation_config.projection = projection;
            rotation_config.pivot_world = pivot_world;
            rotation_config.orientation_world = local_aligned ? local_orientation : glm::mat3(1.0f);
            rotation_config.draw_list = &overlay_drawlist;
            rotation_config.input = gizmo_input;
            rotation_config.snap = snap_modifier;
            rotation_config.snap_degrees = ROTATION_SNAP_DEGREES;

            const auto rotation_result = drawRotationGizmo(rotation_config);
            is_using = is_using || rotation_result.active;
            changed = changed || rotation_result.changed;
            if (rotation_result.changed) {
                crop_tool_visualizer_transform_ =
                    glm::translate(glm::mat4(1.0f), pivot_world) *
                    glm::mat4(rotation_result.delta_rotation) *
                    glm::translate(glm::mat4(1.0f), -pivot_world) *
                    crop_tool_visualizer_transform_;
            }
            if (rotation_result.hovered || rotation_result.active)
                guiFocusState().want_capture_mouse = true;
        }

        if (use_bounds) {
            ScaleGizmoConfig scale_config;
            scale_config.id = panelGizmoId(CROPBOX_GIZMO_ID_BASE, active_panel->panel);
            scale_config.viewport_pos = active_panel->pos;
            scale_config.viewport_size = active_panel->size;
            scale_config.view = view;
            scale_config.projection = projection;
            scale_config.pivot_world = glm::vec3(crop_tool_visualizer_transform_[3]);
            scale_config.orientation_world = local_orientation;
            scale_config.draw_list = &overlay_drawlist;
            scale_config.input = gizmo_input;
            scale_config.input_enabled = !isBoundsGizmoActive();
            scale_config.snap = snap_modifier;
            scale_config.snap_ratio = SCALE_SNAP_RATIO;

            scale_result = drawScaleGizmo(scale_config);
            is_using = is_using || scale_result.active;
            changed = changed || scale_result.changed;
            if (scale_result.changed) {
                if (isCropToolActive()) {
                    scaleCropToolTransform(scale_result.total_scale);
                } else {
                    crop_tool_box_min_ *= scale_result.delta_scale;
                    crop_tool_box_max_ *= scale_result.delta_scale;
                }
            }
            if (scale_result.hovered || scale_result.active)
                guiFocusState().want_capture_mouse = true;
        }

        if (isSelectionVolumeMode() && is_using && !selection_volume_gizmo_active_)
            beginSelectionVolumeGizmoDrag();
        if (isCropToolActive() && is_using && !crop_tool_drag_active_) {
            crop_tool_drag_active_ = true;
            crop_tool_drag_changed_ = false;
            crop_tool_box_min_before_drag_ = crop_tool_box_min_;
            crop_tool_box_max_before_drag_ = crop_tool_box_max_;
            crop_tool_ellipsoid_radii_before_drag_ = crop_tool_ellipsoid_radii_;
            crop_tool_visualizer_transform_before_drag_ = crop_tool_visualizer_transform_;
        }

        if (changed) {
            if (crop_tool_drag_active_)
                crop_tool_drag_changed_ = true;
            updateCropToolOverlayState();
            if (isSelectionVolumeMode()) {
                selection_volume_drag_changed_ = selection_volume_drag_changed_ || selection_volume_gizmo_active_;
                (void)applySelectionVolumeFromGizmo(false);
            }
        } else {
            bool affects_render = false;
            bool effectively_visible = true;
            int parent_node_index = -1;
            if (isCropToolActive()) {
                if (auto* const sm = viewer_ ? viewer_->getSceneManager() : nullptr) {
                    if (const auto* const node = sm->getScene().getNodeById(crop_tool_volume_node_id_)) {
                        effectively_visible = sm->getScene().isNodeEffectivelyVisible(crop_tool_volume_node_id_);
                        affects_render = node->cropbox && node->cropbox->enabled;
                        parent_node_index = sm->getScene().getVisibleNodeIndex(node->parent_id);
                    }
                }
            }
            if (effectively_visible) {
                render_manager->setCropboxGizmoState(
                    true, crop_tool_box_min_, crop_tool_box_max_, crop_tool_visualizer_transform_, affects_render, parent_node_index);
            } else {
                render_manager->setCropboxGizmoActive(false);
            }
        }

        if (isSelectionVolumeMode() && !is_using && selection_volume_gizmo_active_)
            finishSelectionVolumeGizmoDrag();
        if (isCropToolActive() && !is_using && crop_tool_drag_active_) {
            const bool should_persist = crop_tool_drag_changed_;
            crop_tool_drag_active_ = false;
            crop_tool_scale_start_.reset();
            crop_tool_drag_changed_ = false;
            if (should_persist) {
                (void)persistActiveCropToolToNode(false);
            }
        }

        overlay_drawlist.PopClipRect();

        const glm::vec3 shown_scale = glm::max(extractScale(crop_tool_visualizer_transform_), glm::vec3(1e-6f));
        const glm::mat3 shown_rotation = extractRotation(crop_tool_visualizer_transform_);
        const glm::vec3 shown_pivot = glm::vec3(crop_tool_visualizer_transform_[3]) +
                                      shown_rotation * (((crop_tool_box_min_ + crop_tool_box_max_) * 0.5f) * shown_scale);
        glm::mat4 shown_matrix = glm::translate(glm::mat4(1.0f), shown_pivot);
        if (local_aligned)
            shown_matrix *= glm::mat4(shown_rotation);
        shown_matrix = glm::scale(shown_matrix, (crop_tool_box_max_ - crop_tool_box_min_) * shown_scale);
        const glm::mat3 shown_orientation = isSelectionVolumeMode()
                                                ? selectionVolumeLocalRotation(shown_matrix)
                                                : userFacingLocalRotation(shown_matrix);
        CropToolGizmoFrame frame;
        frame.operation = gizmo_op;
        frame.pivot_world = shown_pivot;
        frame.axes_orientation = local_aligned ? shown_orientation : glm::mat3(1.0f);
        frame.bounds_center_world = glm::vec3(shown_matrix[3]);
        frame.bounds_orientation = shown_orientation;
        frame.bounds_half_extents_world = extractScale(shown_matrix) * 0.5f;
        frame.scale_pivot_world = glm::vec3(crop_tool_visualizer_transform_[3]);
        drawPassiveCropToolGizmos(ctx.viewer, viewport, active_panel->panel, settings, CROPBOX_GIZMO_ID_BASE, frame,
                                  overlay_drawlist);
    }

    void GizmoManager::renderCropBoxGizmo(const UIContext& ctx, const ViewportLayout& viewport) {
        (void)syncSelectedCropShape();
        // Selecting a crop only shows its scene-owned outline. Interactive handles
        // belong exclusively to an explicit crop edit or selection-volume tool.
        if (isVolumeGizmoToolActive() && crop_tool_shape_ == CropToolShape::Box)
            renderCropToolBoxGizmo(ctx, viewport);
    }

    void GizmoManager::renderCropToolEllipsoidGizmo(const UIContext& ctx, const ViewportLayout& viewport) {
        auto* const render_manager = ctx.viewer->getRenderingManager();
        const bool polygon = crop_tool_shape_ == CropToolShape::Polygon;
        if (!render_manager || (crop_tool_shape_ != CropToolShape::Ellipsoid && !polygon) || !ensureCropToolState())
            return;
        // The polygon tool only scales the transform, so its half extents stay as drawn.
        glm::vec3& half_extents = polygon ? crop_tool_polygon_half_extents_ : crop_tool_ellipsoid_radii_;

        const auto active_panel = resolveActiveGizmoPanel(ctx.viewer, viewport);
        if (!active_panel || !active_panel->valid())
            return;

        const auto& settings = render_manager->getSettings();
        auto& vp = *active_panel->viewport;
        const glm::mat4 view = vp.getViewMatrix();
        const glm::ivec2 vp_size(static_cast<int>(active_panel->size.x), static_cast<int>(active_panel->size.y));
        const glm::mat4 projection = lfs::rendering::createProjectionMatrixFromFocal(
            vp_size, settings.focal_length_mm, settings.orthographic, vp.ortho_scale_override.value_or(settings.ortho_scale));

        // Vertex handles take the pointer only on themselves; elsewhere the gizmos keep it.
        CropPolygonHandles polygon_handles;
        if (polygon && isCropToolActive())
            polygon_handles = updateCropPolygonVertexEdit(active_panel->pos, active_panel->size, view, projection);

        const glm::vec3 world_scale = glm::max(extractScale(crop_tool_visualizer_transform_), glm::vec3(1e-6f));
        const glm::mat3 rotation = extractRotation(crop_tool_visualizer_transform_);
        const glm::vec3 pivot_world(crop_tool_visualizer_transform_[3]);

        const GizmoOperation gizmo_op = current_operation_;
        const bool use_bounds = gizmo_op == GizmoOperation::Scale;
        const bool use_world_space = transform_space_ == TransformSpace::World;
        const bool local_aligned = use_bounds || !use_world_space;

        glm::mat4 gizmo_matrix = glm::translate(glm::mat4(1.0f), pivot_world);
        if (local_aligned)
            gizmo_matrix *= glm::mat4(rotation);
        gizmo_matrix = glm::scale(gizmo_matrix, half_extents * world_scale);
        const glm::mat3 local_orientation = isSelectionVolumeMode()
                                                ? selectionVolumeLocalRotation(gizmo_matrix)
                                                : userFacingLocalRotation(gizmo_matrix);

        NativeOverlayDrawList overlay_drawlist;
        const glm::vec2 clip_min(active_panel->pos.x, active_panel->pos.y);
        const glm::vec2 clip_max(clip_min.x + active_panel->size.x, clip_min.y + active_panel->size.y);
        overlay_drawlist.PushClipRect(clip_min, clip_max, true);
        const auto& frame_input = viewer_->getWindowManager()->frameInput();
        NativeGizmoInput gizmo_input = nativeGizmoInputFromFrame(frame_input);
        if (polygon_handles.owns_pointer) {
            gizmo_input.mouse_pos = glm::vec2(-1.0e6f);
            gizmo_input.mouse_left_clicked = false;
        }
        const bool snap_modifier = nativeControlModifierDown(frame_input);

        bool changed = false;
        bool is_using = false;
        ScaleGizmoResult scale_result;
        const bool scale_gizmo_has_priority = use_bounds && (isScaleGizmoHovered() || isScaleGizmoActive());

        if (use_bounds) {
            BoundsGizmoConfig bounds_config;
            bounds_config.id = panelGizmoId(ELLIPSOID_GIZMO_ID_BASE, active_panel->panel);
            bounds_config.viewport_pos = active_panel->pos;
            bounds_config.viewport_size = active_panel->size;
            bounds_config.view = view;
            bounds_config.projection = projection;
            bounds_config.center_world = glm::vec3(gizmo_matrix[3]);
            bounds_config.orientation_world = local_orientation;
            bounds_config.half_extents_world = extractScale(gizmo_matrix);
            const glm::vec3 radii = glm::max(half_extents, glm::vec3(1e-6f));
            bounds_config.min_half_extents_world =
                isCropToolActive() ? radii * gizmo_ops::MIN_VOLUME_AXIS_SCALE : world_scale * MIN_GIZMO_SCALE;
            bounds_config.draw_list = &overlay_drawlist;
            bounds_config.input = gizmo_input;
            bounds_config.input_enabled = !scale_gizmo_has_priority;
            bounds_config.snap = snap_modifier;
            bounds_config.snap_ratio = SCALE_SNAP_RATIO;

            const auto bounds_result = drawBoundsGizmo(bounds_config);
            is_using = is_using || bounds_result.active;
            changed = changed || bounds_result.changed;
            if (bounds_result.changed) {
                if (isCropToolActive()) {
                    const glm::vec3 ratio = bounds_result.half_extents_world / (world_scale * radii);
                    for (int axis = 0; axis < 3; ++axis)
                        crop_tool_visualizer_transform_[axis] *= ratio[axis];
                } else {
                    half_extents =
                        glm::max(bounds_result.half_extents_world / world_scale, glm::vec3(MIN_GIZMO_SCALE));
                }
                crop_tool_visualizer_transform_[3] = glm::vec4(bounds_result.center_world, 1.0f);
            }
            if (bounds_result.hovered || bounds_result.active)
                guiFocusState().want_capture_mouse = true;
        } else if (gizmo_op == GizmoOperation::Translate) {
            TranslationGizmoConfig translation_config;
            translation_config.id = panelGizmoId(ELLIPSOID_GIZMO_ID_BASE, active_panel->panel);
            translation_config.viewport_pos = active_panel->pos;
            translation_config.viewport_size = active_panel->size;
            translation_config.view = view;
            translation_config.projection = projection;
            translation_config.pivot_world = pivot_world;
            translation_config.orientation_world = local_aligned ? local_orientation : glm::mat3(1.0f);
            translation_config.draw_list = &overlay_drawlist;
            translation_config.input = gizmo_input;
            translation_config.snap = snap_modifier;
            translation_config.snap_units = TRANSLATE_SNAP_UNITS;

            const auto translation_result = drawTranslationGizmo(translation_config);
            is_using = is_using || translation_result.active;
            changed = changed || translation_result.changed;
            if (translation_result.changed)
                crop_tool_visualizer_transform_[3] += glm::vec4(translation_result.delta_translation, 0.0f);
            if (translation_result.hovered || translation_result.active)
                guiFocusState().want_capture_mouse = true;
        } else if (gizmo_op == GizmoOperation::Rotate) {
            RotationGizmoConfig rotation_config;
            rotation_config.id = panelGizmoId(ELLIPSOID_GIZMO_ID_BASE, active_panel->panel);
            rotation_config.viewport_pos = active_panel->pos;
            rotation_config.viewport_size = active_panel->size;
            rotation_config.view = view;
            rotation_config.projection = projection;
            rotation_config.pivot_world = pivot_world;
            rotation_config.orientation_world = local_aligned ? local_orientation : glm::mat3(1.0f);
            rotation_config.draw_list = &overlay_drawlist;
            rotation_config.input = gizmo_input;
            rotation_config.snap = snap_modifier;
            rotation_config.snap_degrees = ROTATION_SNAP_DEGREES;

            const auto rotation_result = drawRotationGizmo(rotation_config);
            is_using = is_using || rotation_result.active;
            changed = changed || rotation_result.changed;
            if (rotation_result.changed) {
                crop_tool_visualizer_transform_ =
                    glm::translate(glm::mat4(1.0f), pivot_world) *
                    glm::mat4(rotation_result.delta_rotation) *
                    glm::translate(glm::mat4(1.0f), -pivot_world) *
                    crop_tool_visualizer_transform_;
            }
            if (rotation_result.hovered || rotation_result.active)
                guiFocusState().want_capture_mouse = true;
        }

        if (use_bounds) {
            ScaleGizmoConfig scale_config;
            scale_config.id = panelGizmoId(ELLIPSOID_GIZMO_ID_BASE, active_panel->panel);
            scale_config.viewport_pos = active_panel->pos;
            scale_config.viewport_size = active_panel->size;
            scale_config.view = view;
            scale_config.projection = projection;
            scale_config.pivot_world = glm::vec3(crop_tool_visualizer_transform_[3]);
            scale_config.orientation_world = local_orientation;
            scale_config.draw_list = &overlay_drawlist;
            scale_config.input = gizmo_input;
            scale_config.input_enabled = !isBoundsGizmoActive();
            scale_config.snap = snap_modifier;
            scale_config.snap_ratio = SCALE_SNAP_RATIO;

            scale_result = drawScaleGizmo(scale_config);
            is_using = is_using || scale_result.active;
            changed = changed || scale_result.changed;
            if (scale_result.changed) {
                if (isCropToolActive()) {
                    scaleCropToolTransform(scale_result.total_scale);
                } else {
                    half_extents *= scale_result.delta_scale;
                }
            }
            if (scale_result.hovered || scale_result.active)
                guiFocusState().want_capture_mouse = true;
        }

        if (isSelectionVolumeMode() && is_using && !selection_volume_gizmo_active_)
            beginSelectionVolumeGizmoDrag();
        if (isCropToolActive() && is_using && !crop_tool_drag_active_) {
            crop_tool_drag_active_ = true;
            crop_tool_drag_changed_ = false;
            crop_tool_box_min_before_drag_ = crop_tool_box_min_;
            crop_tool_box_max_before_drag_ = crop_tool_box_max_;
            crop_tool_ellipsoid_radii_before_drag_ = crop_tool_ellipsoid_radii_;
            crop_tool_visualizer_transform_before_drag_ = crop_tool_visualizer_transform_;
        }

        if (changed) {
            if (crop_tool_drag_active_)
                crop_tool_drag_changed_ = true;
            updateCropToolOverlayState();
            if (isSelectionVolumeMode()) {
                selection_volume_drag_changed_ = selection_volume_drag_changed_ || selection_volume_gizmo_active_;
                (void)applySelectionVolumeFromGizmo(false);
            }
        } else {
            bool affects_render = false;
            bool effectively_visible = true;
            int parent_node_index = -1;
            if (isCropToolActive()) {
                if (auto* const sm = viewer_ ? viewer_->getSceneManager() : nullptr) {
                    if (const auto* const node = sm->getScene().getNodeById(crop_tool_volume_node_id_)) {
                        effectively_visible = sm->getScene().isNodeEffectivelyVisible(crop_tool_volume_node_id_);
                        affects_render = polygon ? node->crop_polygon && node->crop_polygon->enabled
                                                 : node->ellipsoid && node->ellipsoid->enabled;
                        parent_node_index = sm->getScene().getVisibleNodeIndex(node->parent_id);
                    }
                }
            }
            if (!effectively_visible) {
                render_manager->setEllipsoidGizmoActive(false);
                render_manager->setCropPolygonGizmoActive(false);
            } else if (polygon) {
                render_manager->setCropPolygonGizmoState(true, crop_tool_polygon_vertices_, crop_tool_polygon_half_depth_,
                                                         crop_tool_visualizer_transform_, affects_render,
                                                         parent_node_index);
            } else {
                render_manager->setEllipsoidGizmoState(
                    true, crop_tool_ellipsoid_radii_, crop_tool_visualizer_transform_, affects_render, parent_node_index);
            }
        }

        if (isSelectionVolumeMode() && !is_using && selection_volume_gizmo_active_)
            finishSelectionVolumeGizmoDrag();
        if (isCropToolActive() && !is_using && crop_tool_drag_active_) {
            const bool should_persist = crop_tool_drag_changed_;
            crop_tool_drag_active_ = false;
            crop_tool_scale_start_.reset();
            crop_tool_drag_changed_ = false;
            if (should_persist)
                (void)persistActiveCropToolToNode(false);
        }

        for (size_t i = 0; i < polygon_handles.screen.size(); ++i) {
            if (const auto& point = polygon_handles.screen[i]) {
                const bool hot = static_cast<int>(i) == polygon_handles.hot;
                overlay_drawlist.AddCircleFilled(*point, CROP_POLYGON_HANDLE_RADIUS_PX,
                                                 hot ? CROP_POLYGON_HANDLE_HOT_COLOR : CROP_POLYGON_HANDLE_COLOR);
                overlay_drawlist.AddCircle(*point, CROP_POLYGON_HANDLE_RADIUS_PX, CROP_POLYGON_HANDLE_OUTLINE, 16, 1.0f);
            }
        }

        overlay_drawlist.PopClipRect();

        const glm::vec3 shown_scale = glm::max(extractScale(crop_tool_visualizer_transform_), glm::vec3(1e-6f));
        const glm::vec3 shown_pivot(crop_tool_visualizer_transform_[3]);
        glm::mat4 shown_matrix = glm::translate(glm::mat4(1.0f), shown_pivot);
        if (local_aligned)
            shown_matrix *= glm::mat4(extractRotation(crop_tool_visualizer_transform_));
        shown_matrix = glm::scale(shown_matrix, half_extents * shown_scale);
        const glm::mat3 shown_orientation = isSelectionVolumeMode()
                                                ? selectionVolumeLocalRotation(shown_matrix)
                                                : userFacingLocalRotation(shown_matrix);
        CropToolGizmoFrame frame;
        frame.operation = gizmo_op;
        frame.pivot_world = shown_pivot;
        frame.axes_orientation = local_aligned ? shown_orientation : glm::mat3(1.0f);
        frame.bounds_center_world = glm::vec3(shown_matrix[3]);
        frame.bounds_orientation = shown_orientation;
        frame.bounds_half_extents_world = extractScale(shown_matrix);
        frame.scale_pivot_world = shown_pivot;
        drawPassiveCropToolGizmos(ctx.viewer, viewport, active_panel->panel, settings, ELLIPSOID_GIZMO_ID_BASE, frame,
                                  overlay_drawlist);
    }

    void GizmoManager::renderEllipsoidGizmo(const UIContext& ctx, const ViewportLayout& viewport) {
        (void)syncSelectedCropShape();
        // Polygon prisms share this explicit volume-edit path. There is no
        // selection-only fallback that can draw or hit-test transform handles.
        if (isVolumeGizmoToolActive() &&
            (crop_tool_shape_ == CropToolShape::Ellipsoid || crop_tool_shape_ == CropToolShape::Polygon))
            renderCropToolEllipsoidGizmo(ctx, viewport);
    }

    void GizmoManager::renderViewportGizmo(const ViewportLayout& viewport) {
        if (!show_viewport_gizmo_ || viewport.size.x <= 0 || viewport.size.y <= 0)
            return;

        auto* rendering_manager = viewer_->getRenderingManager();
        if (!rendering_manager)
            return;

        auto panels = collectViewportGizmoPanels(viewer_, {viewport.pos.x, viewport.pos.y},
                                                 {viewport.size.x, viewport.size.y});
        const auto find_panel = [&](const SplitViewPanelId panel_id) -> ViewportGizmoPanelTarget* {
            for (auto& panel : panels) {
                if (panel.panel == panel_id && panel.valid()) {
                    return &panel;
                }
            }
            return nullptr;
        };

        const auto& frame_input = viewer_->getWindowManager()->frameInput();
        const glm::vec2 mouse(frame_input.mouse_x, frame_input.mouse_y);
        const float time = static_cast<float>(SDL_GetTicks()) / 1000.0f;
        const auto snap_to_axis = [&](ViewportGizmoPanelTarget& panel, const int encoded_axis) {
            panel.viewport->camera.setAxisAlignedView(encoded_axis % 3, encoded_axis >= 3);
            rendering_manager->setGridPlaneForPanel(panel.panel, encoded_axis % 3);
            rendering_manager->markCameraPoseChanged();
        };

        // A press or drag owns the pointer until release, even over UI.
        if (viewport_gizmo_dragging_) {
            auto* const panel = find_panel(viewport_gizmo_active_panel_);
            if (panel && frame_input.mouse_down[0]) {
                const glm::vec2 delta(frame_input.mouse_dx, frame_input.mouse_dy);
                if (delta != glm::vec2(0.0f)) {
                    viewport_gizmo_drag_pos_ += delta;
                    orbitViewportGizmo(*panel->viewport, time);
                    rendering_manager->markCameraPoseChanged();
                }
                return;
            }
            if (panel) {
                panel->viewport->camera.endRotateAroundCenter();
                if (const auto* const input_controller = viewer_->getInputController();
                    input_controller && input_controller->cameraViewSnapEnabled()) {
                    constexpr float kAxisSnapAngleDegrees = 10.0f;
                    int snapped_axis = -1;
                    if (panel->viewport->camera.snapToNearestAxisView(kAxisSnapAngleDegrees, &snapped_axis, nullptr)) {
                        rendering_manager->setGridPlaneForPanel(panel->panel, snapped_axis);
                    }
                }
                rendering_manager->markCameraPoseChanged();
            }
            viewport_gizmo_dragging_ = false;
            viewport_gizmo_camera_snapshot_valid_ = false;
            releaseViewportGizmoPointer();
            return;
        }

        if (viewport_gizmo_pressed_) {
            auto* const panel = find_panel(viewport_gizmo_active_panel_);
            if (!panel) {
                viewport_gizmo_pressed_ = false;
                return;
            }
            if (frame_input.mouse_down[0]) {
                if (glm::distance(mouse, viewport_gizmo_press_pos_) >=
                    VIEWPORT_AXIS_DRAG_THRESHOLD * viewportGizmoUiScale()) {
                    beginViewportGizmoOrbit(*panel->viewport, mouse, time);
                }
                return;
            }
            // A click: release on the pressed marker views along its axis.
            viewport_gizmo_pressed_ = false;
            if (const auto layout = panelGizmoLayout(*panel);
                layout && hitTestViewportGizmoMarkers(layout->markers, mouse) == viewport_gizmo_pressed_axis_) {
                snap_to_axis(*panel, viewport_gizmo_pressed_axis_);
            }
            return;
        }

        if (!frame_input.mouse_clicked[0] || guiFocusState().want_capture_mouse)
            return;

        glm::vec2 press = mouse;
        for (const auto& event : frame_input.mouse_button_events) {
            if (event.button == 0 && event.down) {
                press = {event.x, event.y};
                break;
            }
        }
        for (auto& panel : panels) {
            const auto layout = panelGizmoLayout(panel);
            if (!layout || !layout->discContains(press)) {
                continue;
            }
            if (auto* const input_controller = viewer_->getInputController()) {
                input_controller->setFocusedSplitPanel(panel.panel);
            } else {
                rendering_manager->setFocusedSplitPanel(panel.panel);
            }
            viewport_gizmo_active_panel_ = panel.panel;
            viewport_gizmo_press_pos_ = press;
            viewport_gizmo_pressed_axis_ = hitTestViewportGizmoMarkers(layout->markers, press);
            float cursor_x = 0.0f;
            float cursor_y = 0.0f;
            SDL_GetMouseState(&cursor_x, &cursor_y);
            gizmo_drag_start_cursor_ = {cursor_x, cursor_y};
            if (viewport_gizmo_pressed_axis_ < 0) {
                // The sphere itself orbits at once.
                beginViewportGizmoOrbit(*panel.viewport, mouse, time);
            } else if (frame_input.mouse_down[0]) {
                viewport_gizmo_pressed_ = true;
            } else if (hitTestViewportGizmoMarkers(layout->markers, mouse) == viewport_gizmo_pressed_axis_) {
                snap_to_axis(panel, viewport_gizmo_pressed_axis_);
            }
            return;
        }
    }

    void GizmoManager::beginViewportGizmoOrbit(Viewport& viewport, const glm::vec2& mouse, const float time) {
        viewport_gizmo_pressed_ = false;
        viewport_gizmo_dragging_ = true;
        viewport_gizmo_camera_position_before_drag_ = viewport.camera.t;
        viewport_gizmo_camera_pivot_before_drag_ = viewport.camera.pivot;
        viewport_gizmo_camera_rotation_before_drag_ = viewport.camera.R;
        viewport_gizmo_camera_snapshot_valid_ = true;
        viewport.camera.startRotateAroundCenter(viewport_gizmo_press_pos_, time);
        viewport_gizmo_drag_pos_ = mouse;
        if (mouse != viewport_gizmo_press_pos_) {
            orbitViewportGizmo(viewport, time);
            if (auto* const rendering_manager = viewer_->getRenderingManager())
                rendering_manager->markCameraPoseChanged();
        }
        if (SDL_Window* const window = viewer_->getWindow())
            SDL_SetWindowRelativeMouseMode(window, true);
    }

    void GizmoManager::orbitViewportGizmo(Viewport& viewport, const float time) {
        if (const auto* const input_controller = viewer_->getInputController();
            input_controller &&
            input_controller->cameraNavigationMode() == InputController::CameraNavigationMode::Trackball) {
            viewport.camera.updateTrackballRotateAroundCenter(viewport_gizmo_drag_pos_, time);
        } else {
            viewport.camera.updateRotateAroundCenter(viewport_gizmo_drag_pos_, time);
        }
    }

    void GizmoManager::releaseViewportGizmoPointer() {
        if (SDL_Window* const window = viewer_ ? viewer_->getWindow() : nullptr) {
            SDL_SetWindowRelativeMouseMode(window, false);
            SDL_WarpMouseInWindow(window,
                                  static_cast<float>(gizmo_drag_start_cursor_.x),
                                  static_cast<float>(gizmo_drag_start_cursor_.y));
        }
    }

    std::optional<GizmoManager::ViewportGizmoPress> GizmoManager::viewportGizmoPress() const {
        if (!viewport_gizmo_dragging_ && !viewport_gizmo_pressed_)
            return std::nullopt;
        return ViewportGizmoPress{.panel = viewport_gizmo_active_panel_,
                                  .axis = viewport_gizmo_dragging_ ? -1 : viewport_gizmo_pressed_axis_};
    }

    void GizmoManager::triggerCropFlash() {
        crop_flash_active_ = true;
        crop_flash_start_ = std::chrono::steady_clock::now();
    }

    void GizmoManager::updateCropFlash() {
        if (!crop_flash_active_)
            return;

        auto* const sm = viewer_->getSceneManager();
        auto* const rm = viewer_->getRenderingManager();
        if (!sm || !rm)
            return;

        constexpr int DURATION_MS = 400;
        const auto elapsed_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                                    std::chrono::steady_clock::now() - crop_flash_start_)
                                    .count();

        const core::NodeId cropbox_id = sm->getSelectedNodeCropBoxId();
        if (cropbox_id == core::NULL_NODE) {
            crop_flash_active_ = false;
            return;
        }

        const auto* cropbox_ref = sm->getScene().getNodeById(cropbox_id);
        if (!cropbox_ref) {
            crop_flash_active_ = false;
            return;
        }
        auto* node = sm->getScene().getMutableNode(cropbox_ref->name);
        if (!node || !node->cropbox) {
            crop_flash_active_ = false;
            return;
        }

        if (elapsed_ms >= DURATION_MS) {
            crop_flash_active_ = false;
            node->cropbox->flash_intensity = 0.0f;
        } else {
            node->cropbox->flash_intensity = 1.0f - static_cast<float>(elapsed_ms) / DURATION_MS;
        }
        sm->getScene().invalidateCache();
        rm->markDirty(DirtyFlag::SPLATS | DirtyFlag::OVERLAY, lfs::vis::FrameReason::SceneChange);
    }

    void GizmoManager::deactivateAllTools() {
        python::cancel_active_operator();
        setPointPickToolsEnabled(viewer_->getAlignTool(), viewer_->getScaleDistanceTool());

        auto& editor = viewer_->getEditorContext();
        editor.setActiveTool(ToolType::None);
        current_operation_ = GizmoOperation::Translate;
    }

    bool GizmoManager::cancelActiveNodeTransformDrag() {
        if (!node_gizmo_active_ || node_gizmo_node_names_.empty() || node_transforms_before_drag_.empty()) {
            return false;
        }

        auto* const scene_manager = viewer_ ? viewer_->getSceneManager() : nullptr;
        if (!scene_manager) {
            return false;
        }

        const size_t restore_count = std::min(node_gizmo_node_names_.size(), node_transforms_before_drag_.size());
        for (size_t i = 0; i < restore_count; ++i) {
            scene_manager->setNodeTransform(node_gizmo_node_names_[i], node_transforms_before_drag_[i]);
        }

        cancelTranslationGizmoDrag();
        cancelRotationGizmoDrag();
        cancelScaleGizmoDrag();
        cancelBoundsGizmoDrag();
        node_gizmo_active_ = false;
        node_bounds_scale_active_ = false;
        node_selection_bounds_scale_active_ = false;
        node_gizmo_node_names_.clear();
        node_transforms_before_drag_.clear();
        node_original_visualizer_world_transforms_.clear();

        if (auto* const rendering_manager = viewer_->getRenderingManager()) {
            rendering_manager->setCropboxGizmoActive(false);
            rendering_manager->setEllipsoidGizmoActive(false);
            rendering_manager->markDirty(DirtyFlag::SPLATS | DirtyFlag::OVERLAY, FrameReason::SceneChange);
        }
        return true;
    }

    bool GizmoManager::cancelActiveGizmoDrag() {
        bool cancelled = cancelActiveNodeTransformDrag();
        auto* const scene_manager = viewer_ ? viewer_->getSceneManager() : nullptr;
        auto* const rendering_manager = viewer_ ? viewer_->getRenderingManager() : nullptr;

        if (crop_polygon_vertex_drag_) {
            crop_polygon_vertex_drag_.reset();
            if (ensureCropToolState())
                updateCropToolOverlayState();
            cancelled = true;
        }

        if (crop_tool_drag_active_ || selection_volume_gizmo_active_) {
            crop_tool_box_min_ = crop_tool_box_min_before_drag_;
            crop_tool_box_max_ = crop_tool_box_max_before_drag_;
            crop_tool_ellipsoid_radii_ = crop_tool_ellipsoid_radii_before_drag_;
            crop_tool_visualizer_transform_ = crop_tool_visualizer_transform_before_drag_;
            crop_tool_drag_active_ = false;
            crop_tool_scale_start_.reset();
            crop_tool_drag_changed_ = false;
            if (selection_volume_gizmo_active_) {
                if (scene_manager) {
                    scene_manager->getScene().restoreSelectionState(selection_volume_selection_before_drag_);
                }
                selection_volume_selection_before_drag_ = {};
                selection_volume_gizmo_active_ = false;
                selection_volume_drag_changed_ = false;
            }
            if (isVolumeGizmoToolActive())
                updateCropToolOverlayState();
            cancelled = true;
        }

        if (viewport_gizmo_dragging_) {
            Viewport* active_viewport = viewer_ ? &viewer_->getViewport() : nullptr;
            if (active_viewport && rendering_manager &&
                viewport_gizmo_active_panel_ == SplitViewPanelId::Right &&
                rendering_manager->isIndependentSplitViewActive()) {
                active_viewport = &rendering_manager->projectSecondaryViewport();
            }
            if (active_viewport) {
                active_viewport->camera.endRotateAroundCenter();
                if (viewport_gizmo_camera_snapshot_valid_) {
                    active_viewport->camera.t = viewport_gizmo_camera_position_before_drag_;
                    active_viewport->camera.pivot = viewport_gizmo_camera_pivot_before_drag_;
                    active_viewport->camera.R = viewport_gizmo_camera_rotation_before_drag_;
                }
            }
            viewport_gizmo_dragging_ = false;
            viewport_gizmo_camera_snapshot_valid_ = false;
            releaseViewportGizmoPointer();
            if (rendering_manager)
                rendering_manager->markCameraPoseChanged();
            cancelled = true;
        }
        if (viewport_gizmo_pressed_) {
            viewport_gizmo_pressed_ = false;
            cancelled = true;
        }

        if (cancelled) {
            cancelTranslationGizmoDrag();
            cancelRotationGizmoDrag();
            cancelScaleGizmoDrag();
            cancelBoundsGizmoDrag();
            if (rendering_manager) {
                rendering_manager->setCropboxGizmoActive(false);
                rendering_manager->setEllipsoidGizmoActive(false);
                rendering_manager->markDirty(DirtyFlag::SPLATS | DirtyFlag::OVERLAY,
                                             lfs::vis::FrameReason::SceneChange);
            }
        }
        return cancelled;
    }

    void GizmoManager::setSelectionSubMode(SelectionSubMode mode) {
        const bool was_volume_mode = isSelectionVolumeSubMode(selection_mode_);
        selection_mode_ = mode;
        python::set_selection_submode(static_cast<int>(mode));

        if (auto* rm = viewer_->getRenderingManager()) {
            rm->setSelectionPreviewMode(toSelectionPreviewMode(mode));
        }

        if (isSelectionVolumeSubMode(mode)) {
            crop_tool_shape_ = mode == SelectionSubMode::Sphere ? CropToolShape::Ellipsoid : CropToolShape::Box;
            if (!was_volume_mode) {
                current_operation_ = GizmoOperation::Scale;
            }
            clearSelectionVolumeState();
            crop_tool_initialized_ = false;
            crop_tool_target_node_id_ = core::NULL_NODE;
            clearCropToolOverlayState();
            return;
        }

        if (was_volume_mode && !isCropToolActive()) {
            clearSelectionVolumeState();
            crop_tool_initialized_ = false;
            crop_tool_target_node_id_ = core::NULL_NODE;
            clearCropToolOverlayState();
        }
    }

    void GizmoManager::setSelectionVolumeFromDrag(const SelectionSubMode mode,
                                                  const SelectionMode apply_mode,
                                                  const uint64_t source_generation,
                                                  const glm::vec3& center_world,
                                                  const float radius) {
        if (!isSelectionVolumeSubMode(mode) || !std::isfinite(radius)) {
            return;
        }

        captureSelectionVolumeBase(source_generation);
        selection_volume_apply_mode_ = apply_mode;
        selection_mode_ = mode;
        python::set_selection_submode(static_cast<int>(mode));
        crop_tool_shape_ = mode == SelectionSubMode::Sphere ? CropToolShape::Ellipsoid : CropToolShape::Box;
        crop_tool_initialized_ = true;
        crop_tool_target_node_id_ = selectedCropTargetNodeId().value_or(core::NULL_NODE);

        const float safe_radius = std::max(radius, MIN_GIZMO_SCALE);
        crop_tool_box_min_ = glm::vec3(-safe_radius);
        crop_tool_box_max_ = glm::vec3(safe_radius);
        crop_tool_ellipsoid_radii_ = glm::vec3(safe_radius);
        crop_tool_visualizer_transform_ = glm::mat4(1.0f);
        crop_tool_visualizer_transform_[3] = glm::vec4(center_world, 1.0f);

        if (auto* rm = viewer_ ? viewer_->getRenderingManager() : nullptr) {
            rm->setSelectionPreviewMode(toSelectionPreviewMode(mode));
        }
        updateCropToolOverlayState();
    }

    void GizmoManager::setTransformSpace(const TransformSpace space) {
        if (transform_space_ == space)
            return;
        transform_space_ = space;
        app_store().transform_space.set(static_cast<int>(transform_space_));
    }

    void GizmoManager::setPivotMode(const PivotMode mode) {
        if (pivot_mode_ == mode)
            return;
        pivot_mode_ = mode;
        app_store().pivot_mode.set(static_cast<int>(pivot_mode_));
    }

    void GizmoManager::setMultiTransformMode(const MultiTransformMode mode) {
        const auto normalized_mode = normalizeMultiTransformMode(mode);
        if (multi_transform_mode_ == normalized_mode)
            return;
        multi_transform_mode_ = normalized_mode;
        app_store().multi_transform_mode.set(static_cast<int>(multi_transform_mode_));
    }

    int GizmoManager::viewportGizmoHover(const double x, const double y) const {
        if (!show_viewport_gizmo_ || guiFocusState().want_capture_mouse)
            return -1;

        const glm::vec2 point(static_cast<float>(x), static_cast<float>(y));
        for (const auto& panel : collectViewportGizmoPanels(viewer_, viewer_->getGuiManager()->getViewportPos(),
                                                            viewer_->getGuiManager()->getViewportSize())) {
            if (const auto layout = panelGizmoLayout(panel); layout && layout->discContains(point)) {
                return static_cast<int>(panel.panel) * 8 + hitTestViewportGizmoMarkers(layout->markers, point) + 1;
            }
        }
        return -1;
    }

    bool GizmoManager::isPositionInViewportGizmo(const double x, const double y) const {
        if (!show_viewport_gizmo_)
            return false;

        const auto vp_pos = viewer_->getGuiManager()->getViewportPos();
        const auto vp_size = viewer_->getGuiManager()->getViewportSize();
        const glm::vec2 point(static_cast<float>(x), static_cast<float>(y));
        for (const auto& panel : collectViewportGizmoPanels(viewer_, vp_pos, vp_size)) {
            if (const auto layout = panelGizmoLayout(panel); layout && layout->discContains(point)) {
                return true;
            }
        }
        return false;
    }

    ToolType GizmoManager::getCurrentToolMode() const {
        return viewer_->getEditorContext().getActiveTool();
    }

    void GizmoManager::openPieMenu(glm::vec2 cursor_pos) {
        pie_menu_.updateItems(viewer_->getEditorContext());
        pie_menu_.open(cursor_pos);
    }

    void GizmoManager::closePieMenu() {
        pie_menu_.close();
    }

    void GizmoManager::onPieMenuKeyRelease() {
        pie_menu_.onKeyRelease();
        handlePieMenuSelection();
    }

    void GizmoManager::onPieMenuMouseMove(glm::vec2 pos) {
        pie_menu_.onMouseMove(pos);
    }

    void GizmoManager::onPieMenuClick(glm::vec2 pos) {
        pie_menu_.onMouseClick(pos);
        handlePieMenuSelection();
    }

    void GizmoManager::renderPieMenu() {
        if (!pie_menu_.isOpen())
            return;

        auto* const rm = viewer_ ? viewer_->getRenderingManager() : nullptr;
        auto* const overlay = rm ? rm->getScreenOverlayRenderer() : nullptr;
        if (!overlay || !overlay->isFrameActive())
            return;

        pie_menu_.draw(*overlay);
    }

    void GizmoManager::handlePieMenuSelection() {
        if (!pie_menu_.hasSelection())
            return;

        const auto& tool_id = pie_menu_.getSelectedId();

        if (!tool_id.empty()) {
            if (tool_id == "builtin.cropbox" || tool_id == "builtin.ellipsoid") {
                python::cancel_active_operator();
                setCropToolShape(tool_id == "builtin.ellipsoid" ? "ellipsoid" : "box");
                if (!ensureCropVolumeForCurrentSelection()) {
                    return;
                }
                viewer_->getEditorContext().setActiveOperator("builtin.cropbox", "translate");
                UnifiedToolRegistry::instance().setActiveTool("builtin.cropbox");
                current_operation_ = GizmoOperation::Translate;
            } else if (tool_id.starts_with("crop.")) {
                handleCropAction(tool_id);
            } else {
                const auto tool_type = pie_menu_.getSelectedToolType();
                if (tool_type != ToolType::None) {
                    lfs::core::events::tools::SetToolbarTool{.tool_mode = static_cast<int>(tool_type)}.emit();

                    const auto& submode_id = pie_menu_.getSelectedSubmodeId();
                    if (!submode_id.empty()) {
                        if (tool_type == ToolType::Selection) {
                            static const std::pair<const char*, SelectionSubMode> SUBMODE_MAP[] = {
                                {"centers", SelectionSubMode::Centers},
                                {"rectangle", SelectionSubMode::Rectangle},
                                {"polygon", SelectionSubMode::Polygon},
                                {"lasso", SelectionSubMode::Lasso},
                                {"rings", SelectionSubMode::Rings},
                                {"box", SelectionSubMode::Box},
                                {"sphere", SelectionSubMode::Sphere},
                                {"color", SelectionSubMode::Color},
                            };
                            for (const auto& [sm_id, sm_mode] : SUBMODE_MAP) {
                                if (submode_id == sm_id) {
                                    lfs::core::events::tools::SetSelectionSubMode{
                                        .selection_mode = static_cast<int>(sm_mode)}
                                        .emit();
                                    break;
                                }
                            }
                        } else if (tool_type == ToolType::Mirror) {
                            int axis = 0;
                            if (submode_id == "y")
                                axis = 1;
                            else if (submode_id == "z")
                                axis = 2;
                            lfs::core::events::tools::ExecuteMirror{.axis = axis}.emit();
                        }
                    }
                }
            }
        }

        pie_menu_.close();
    }

    void GizmoManager::handleCropAction(const std::string& action_id) {
        using namespace lfs::core::events;

        auto* sm = viewer_->getSceneManager();
        const auto selected_type = sm ? sm->getSelectedNodeType() : core::NodeType::SPLAT;
        const bool is_cropbox = selected_type == core::NodeType::CROPBOX;
        const bool is_polygon = selected_type == core::NodeType::CROP_POLYGON;

        if (action_id == "crop.translate") {
            setCropToolOperation("translate");
        } else if (action_id == "crop.rotate") {
            setCropToolOperation("rotate");
        } else if (action_id == "crop.scale") {
            setCropToolOperation("scale");
        } else if (action_id == "crop.apply") {
            finishCropTool();
        } else if (action_id == "crop.fit" || action_id == "crop.fit_trim") {
            const bool use_percentile = action_id == "crop.fit_trim";
            if (is_cropbox) {
                auto cropbox_id = cap::resolveCropBoxId(*sm, std::nullopt);
                if (!cropbox_id)
                    return;
                if (auto result =
                        cap::fitCropBoxToParent(*sm, viewer_->getRenderingManager(), *cropbox_id, use_percentile);
                    !result) {
                    LOG_WARN("Failed to fit cropbox: {}", result.error());
                }
            } else if (is_polygon) {
                fitSelectedCropPolygon(use_percentile);
            } else {
                cmd::FitEllipsoidToScene{.use_percentile = use_percentile}.emit();
            }
        } else if (action_id == "crop.invert") {
            cmd::ToggleCropInverse{}.emit();
        } else if (action_id == "crop.reset") {
            if (is_cropbox) {
                auto cropbox_id = cap::resolveCropBoxId(*sm, std::nullopt);
                if (!cropbox_id)
                    return;
                if (auto result = cap::resetCropBox(*sm, viewer_->getRenderingManager(), *cropbox_id); !result) {
                    LOG_WARN("Failed to reset cropbox: {}", result.error());
                }
            } else if (is_polygon) {
                resetSelectedCropPolygon();
            } else {
                cmd::ResetEllipsoid{}.emit();
            }
        } else if (action_id == "crop.delete") {
            if (!sm)
                return;
            const core::NodeId id = is_cropbox   ? sm->getSelectedNodeCropBoxId()
                                    : is_polygon ? sm->getSelectedNodeCropPolygonId()
                                                 : sm->getSelectedNodeEllipsoidId();
            if (id == core::NULL_NODE)
                return;
            const auto* node = sm->getScene().getNodeById(id);
            if (!node)
                return;
            cmd::RemovePLY{.name = node->name}.emit();
        }
    }

} // namespace lfs::vis::gui

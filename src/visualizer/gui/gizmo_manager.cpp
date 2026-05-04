/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "gui/gizmo_manager.hpp"
#include "core/events.hpp"
#include "core/logger.hpp"
#include "core/scene.hpp"
#include "gui/bounds_gizmo.hpp"
#include "gui/gui_focus_state.hpp"
#include "gui/gui_manager.hpp"
#include "gui/rotation_gizmo.hpp"
#include "gui/scale_gizmo.hpp"
#include "gui/translation_gizmo.hpp"
#include "gui/ui_widgets.hpp"
#include "input/input_controller.hpp"
#include "operation/undo_entry.hpp"
#include "operation/undo_history.hpp"
#include "operator/operator_id.hpp"
#include "operator/operator_registry.hpp"
#include "python/python_runtime.hpp"
#include "rendering/coordinate_conventions.hpp"
#include "rendering/rendering.hpp"
#include "rendering/rendering_manager.hpp"
#include "scene/scene_manager.hpp"
#include "tools/align_tool.hpp"
#include "tools/brush_tool.hpp"
#include "tools/selection_tool.hpp"
#include "tools/unified_tool_registry.hpp"
#include "visualizer/gui_capabilities.hpp"
#include "visualizer/scene_coordinate_utils.hpp"
#include "visualizer_impl.hpp"
#include <SDL3/SDL.h>
#include <algorithm>
#include <array>
#include <cassert>
#include <glm/gtc/matrix_transform.hpp>
#include <optional>
#include <unordered_set>
#include <vector>

namespace lfs::vis::gui {

    using ToolType = lfs::vis::ToolType;

    namespace {
        [[nodiscard]] lfs::vis::SelectionPreviewMode toSelectionPreviewMode(const SelectionSubMode mode) {
            switch (mode) {
            case SelectionSubMode::Rectangle: return lfs::vis::SelectionPreviewMode::Rectangle;
            case SelectionSubMode::Polygon: return lfs::vis::SelectionPreviewMode::Polygon;
            case SelectionSubMode::Lasso: return lfs::vis::SelectionPreviewMode::Lasso;
            case SelectionSubMode::Rings: return lfs::vis::SelectionPreviewMode::Rings;
            case SelectionSubMode::Centers:
            default: return lfs::vis::SelectionPreviewMode::Centers;
            }
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

        struct ViewportGizmoMarker {
            int encoded_axis = -1;
            int axis = 0;
            bool negative = false;
            glm::vec2 screen_pos{0.0f};
            float radius = 0.0f;
            float depth = 0.0f;
            bool visible = false;
        };

        struct ViewportGizmoLayoutData {
            glm::vec2 top_left{0.0f};
            glm::vec2 center{0.0f};
            float size = 0.0f;
            std::array<ViewportGizmoMarker, 6> markers{};
        };

        constexpr float VIEWPORT_GIZMO_DISTANCE = 2.8f;
        constexpr float VIEWPORT_GIZMO_FOV_DEGREES = 38.0f;
        constexpr float VIEWPORT_GIZMO_SPHERE_RADIUS = 0.198f;
        constexpr float VIEWPORT_GIZMO_LABEL_DISTANCE = 0.63f;
        constexpr float VIEWPORT_GIZMO_HIT_RADIUS_SCALE = 2.5f;

        [[nodiscard]] std::optional<ViewportGizmoLayoutData> buildViewportGizmoLayout(
            const ViewportGizmoPanelTarget& panel,
            const float size,
            const float margin_x,
            const float margin_y) {
            if (!panel.valid() || size <= 0.0f) {
                return std::nullopt;
            }

            ViewportGizmoLayoutData layout;
            layout.size = size;
            layout.top_left = {
                panel.pos.x + panel.size.x - size - margin_x,
                panel.pos.y + margin_y,
            };
            layout.center = layout.top_left + glm::vec2(size * 0.5f);

            glm::mat4 view = lfs::rendering::makeViewMatrix(panel.viewport->getRotationMatrix(), glm::vec3(0.0f));
            view[3][2] = -VIEWPORT_GIZMO_DISTANCE;
            const glm::mat4 proj =
                glm::perspective(glm::radians(VIEWPORT_GIZMO_FOV_DEGREES), 1.0f, 0.1f, 10.0f);
            const float projected_marker_radius =
                VIEWPORT_GIZMO_SPHERE_RADIUS *
                (1.0f / std::tan(glm::radians(VIEWPORT_GIZMO_FOV_DEGREES) * 0.5f)) /
                VIEWPORT_GIZMO_DISTANCE *
                size * 0.5f;

            const auto project_marker = [&](const int axis, const bool negative) {
                ViewportGizmoMarker marker;
                marker.axis = axis;
                marker.negative = negative;
                marker.encoded_axis = axis + (negative ? 3 : 0);

                glm::vec3 position(0.0f);
                position[axis] = negative ? -VIEWPORT_GIZMO_LABEL_DISTANCE : VIEWPORT_GIZMO_LABEL_DISTANCE;

                const glm::vec4 clip = proj * view * glm::vec4(position, 1.0f);
                if (clip.w <= 0.0f) {
                    return marker;
                }

                const glm::vec3 ndc = glm::vec3(clip) / clip.w;
                const float local_x = (ndc.x * 0.5f + 0.5f) * size;
                const float local_y = (1.0f - (ndc.y * 0.5f + 0.5f)) * size;
                marker.screen_pos = layout.top_left + glm::vec2(local_x, local_y);
                marker.radius = projected_marker_radius;
                marker.depth = clip.z / clip.w;
                marker.visible = true;
                return marker;
            };

            for (int axis = 0; axis < 3; ++axis) {
                layout.markers[static_cast<size_t>(axis)] = project_marker(axis, false);
                layout.markers[static_cast<size_t>(axis + 3)] = project_marker(axis, true);
            }
            return layout;
        }

        [[nodiscard]] int hitTestViewportGizmoLayout(
            const ViewportGizmoLayoutData& layout,
            const glm::vec2& mouse_pos) {
            for (const auto& marker : layout.markers) {
                if (!marker.visible) {
                    continue;
                }
                const glm::vec2 delta = mouse_pos - marker.screen_pos;
                const float radius = marker.radius * VIEWPORT_GIZMO_HIT_RADIUS_SCALE;
                if (glm::dot(delta, delta) <= radius * radius) {
                    return marker.encoded_axis;
                }
            }
            return -1;
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

        inline glm::vec3 extractScale(const glm::mat4& m) {
            return glm::vec3(glm::length(glm::vec3(m[0])), glm::length(glm::vec3(m[1])),
                             glm::length(glm::vec3(m[2])));
        }

    } // namespace

    GizmoManager::GizmoManager(VisualizerImpl* viewer)
        : viewer_(viewer) {
    }

    void GizmoManager::setupEvents() {
        using namespace lfs::core::events;

        ui::NodeSelected::when([this](const auto&) {
            python::cancel_active_operator();
            if (auto* const t = viewer_->getBrushTool())
                t->setEnabled(false);
            if (auto* const t = viewer_->getAlignTool())
                t->setEnabled(false);
            if (auto* const sm = viewer_->getSceneManager())
                sm->syncCropBoxToRenderSettings();
            node_bounds_cache_valid_ = false;
        });

        ui::NodeDeselected::when([this](const auto&) {
            python::cancel_active_operator();
            if (auto* const t = viewer_->getBrushTool())
                t->setEnabled(false);
            if (auto* const t = viewer_->getAlignTool())
                t->setEnabled(false);
            node_bounds_cache_valid_ = false;
        });

        state::PLYRemoved::when([this](const auto&) { deactivateAllTools(); });
        state::SceneCleared::when([this](const auto&) { deactivateAllTools(); });

        lfs::core::events::tools::SetToolbarTool::when([this](const auto& e) {
            auto& editor = viewer_->getEditorContext();
            const auto tool = static_cast<ToolType>(e.tool_mode);

            if (editor.hasActiveOperator() && tool != ToolType::Selection) {
                python::cancel_active_operator();
            }

            editor.setActiveTool(tool);

            auto& registry = UnifiedToolRegistry::instance();
            const char* tool_id = nullptr;
            switch (tool) {
            case ToolType::None: break;
            case ToolType::Selection: tool_id = "builtin.select"; break;
            case ToolType::Translate: tool_id = "builtin.translate"; break;
            case ToolType::Rotate: tool_id = "builtin.rotate"; break;
            case ToolType::Scale: tool_id = "builtin.scale"; break;
            case ToolType::Mirror: tool_id = "builtin.mirror"; break;
            case ToolType::Brush: tool_id = "builtin.brush"; break;
            case ToolType::Align: tool_id = "builtin.align"; break;
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
                break;
            default:
                LOG_DEBUG("SetToolbarTool: tool_mode={}", e.tool_mode);
                break;
            }

            if (auto* gui = viewer_->getGuiManager()) {
                gui->panelLayout().setShowSequencer(false);
            }
        });

        lfs::core::events::tools::SetSelectionSubMode::when([this](const auto& e) {
            setSelectionSubMode(static_cast<SelectionSubMode>(e.selection_mode));

            static constexpr std::array<const char*, 5> SUBMODE_IDS = {
                "centers", "rectangle", "polygon", "lasso", "rings"};
            const auto idx = static_cast<size_t>(e.selection_mode);
            if (idx < SUBMODE_IDS.size()) {
                UnifiedToolRegistry::instance().setActiveSubmode(SUBMODE_IDS[idx]);
            }

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
            if (!sm)
                return;

            const core::NodeId cropbox_id = sm->getSelectedNodeCropBoxId();
            if (cropbox_id == core::NULL_NODE)
                return;

            const auto* cropbox_node = sm->getScene().getNodeById(cropbox_id);
            if (!cropbox_node || !cropbox_node->cropbox)
                return;

            const glm::mat4 world_transform = scene_coords::nodeDataWorldTransform(sm->getScene(), cropbox_id);

            lfs::geometry::BoundingBox crop_box;
            crop_box.setBounds(cropbox_node->cropbox->min, cropbox_node->cropbox->max);
            crop_box.setworld2BBox(glm::inverse(world_transform));
            cmd::CropPLY{.crop_box = crop_box, .inverse = cropbox_node->cropbox->inverse}.emit();
            triggerCropFlash();
        });

        cmd::ApplyEllipsoid::when([this](const auto&) {
            auto* const sm = viewer_->getSceneManager();
            if (!sm)
                return;

            const core::NodeId ellipsoid_id = sm->getSelectedNodeEllipsoidId();
            if (ellipsoid_id == core::NULL_NODE)
                return;

            const auto* ellipsoid_node = sm->getScene().getNodeById(ellipsoid_id);
            if (!ellipsoid_node || !ellipsoid_node->ellipsoid)
                return;

            const glm::mat4 world_transform = scene_coords::nodeDataWorldTransform(sm->getScene(), ellipsoid_id);
            const glm::vec3 radii = ellipsoid_node->ellipsoid->radii;
            const bool inverse = ellipsoid_node->ellipsoid->inverse;

            cmd::CropPLYEllipsoid{
                .world_transform = world_transform,
                .radii = radii,
                .inverse = inverse}
                .emit();
            triggerCropFlash();
        });

        cmd::ToggleCropInverse::when([this](const auto&) {
            auto* const sm = viewer_->getSceneManager();
            if (!sm)
                return;

            auto cropbox_id = cap::resolveCropBoxId(*sm, std::nullopt);
            if (!cropbox_id)
                return;

            const auto* node = sm->getScene().getNodeById(*cropbox_id);
            if (!node || !node->cropbox)
                return;

            cap::CropBoxUpdate update;
            update.has_inverse = true;
            update.inverse = !node->cropbox->inverse;
            if (auto result = cap::updateCropBox(*sm, viewer_->getRenderingManager(), *cropbox_id, update); !result) {
                LOG_WARN("Failed to toggle crop inverse: {}", result.error());
            }
        });

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
        });
    }

    void GizmoManager::updateToolState(const UIContext& ctx, bool ui_hidden) {
        auto* const scene_manager = ctx.viewer->getSceneManager();
        if (scene_manager && scene_manager->hasSelectedNode() && !ui_hidden) {
            const auto& active_tool_id = UnifiedToolRegistry::instance().getActiveTool();
            const auto& gizmo_type = ctx.editor->getGizmoType();

            bool is_transform_tool = false;
            if (!gizmo_type.empty()) {
                is_transform_tool = true;
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
            } else if (active_tool_id == "builtin.translate" || active_tool_id == "builtin.rotate" ||
                       active_tool_id == "builtin.scale") {
                is_transform_tool = true;
                node_gizmo_operation_ = current_operation_;
            }
            show_node_gizmo_ = is_transform_tool;

            auto* const brush_tool = ctx.viewer->getBrushTool();
            auto* const align_tool = ctx.viewer->getAlignTool();
            auto* const selection_tool = ctx.viewer->getSelectionTool();
            const bool is_brush_mode = (active_tool_id == "builtin.brush");
            const bool is_align_mode = (active_tool_id == "builtin.align");
            const bool is_selection_mode = (active_tool_id == "builtin.select");

            if (previous_tool_id_ == "builtin.select" && active_tool_id != previous_tool_id_) {
                if (auto* const sm = ctx.viewer->getSceneManager()) {
                    sm->applyDeleted();
                }
            }
            previous_tool_id_ = active_tool_id;

            if (brush_tool)
                brush_tool->setEnabled(is_brush_mode);
            if (align_tool)
                align_tool->setEnabled(is_align_mode);
            if (selection_tool)
                selection_tool->setEnabled(is_selection_mode);

            if (is_selection_mode) {
                if (auto* const rm = ctx.viewer->getRenderingManager()) {
                    rm->setSelectionPreviewMode(toSelectionPreviewMode(selection_mode_));

                    if (selection_mode_ != previous_selection_mode_) {
                        if (selection_tool)
                            selection_tool->onSelectionModeChanged();
                        previous_selection_mode_ = selection_mode_;
                    }
                }
            }

        } else {
            show_node_gizmo_ = false;
            if (auto* const tool = ctx.viewer->getBrushTool())
                tool->setEnabled(false);
            if (auto* const tool = ctx.viewer->getAlignTool())
                tool->setEnabled(false);
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
        if (selected_type == core::NodeType::CROPBOX || selected_type == core::NodeType::ELLIPSOID ||
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

        const auto active_panel = resolveActiveGizmoPanel(ctx.viewer, viewport);
        if (!active_panel || !active_panel->valid())
            return;

        auto& vp = *active_panel->viewport;
        const glm::mat4 view = vp.getViewMatrix();
        const glm::ivec2 vp_size(static_cast<int>(active_panel->size.x), static_cast<int>(active_panel->size.y));
        const glm::mat4 projection = lfs::rendering::createProjectionMatrixFromFocal(
            vp_size, settings.focal_length_mm, settings.orthographic, settings.ortho_scale);

        const bool use_world_space = (transform_space_ == TransformSpace::World) || is_multi_selection;

        const glm::vec3 local_pivot = (pivot_mode_ == PivotMode::Origin)
                                          ? glm::vec3(0.0f)
                                          : transform_targets->local_center;

        bool has_valid_bounds = false;
        const bool use_bounds_scale = !is_multi_selection && node_gizmo_operation_ == GizmoOperation::Scale;

        const auto* first_node = (!is_multi_selection && !target_names.empty())
                                     ? scene.getNode(target_names.front())
                                     : nullptr;

        glm::vec3 bounds_min(0.0f), bounds_max(0.0f);
        glm::mat4 world_transform(1.0f);
        glm::vec3 world_scale(1.0f);
        glm::mat3 node_rotation(1.0f);

        if (use_bounds_scale && first_node) {
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

        const bool actually_using_bounds = use_bounds_scale && has_valid_bounds;

        const glm::vec3 transform_gizmo_position = (node_gizmo_active_ && !node_bounds_scale_active_)
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
            if (node_bounds_scale_active_) {
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

            if (actually_using_bounds && first_node && bounds_gizmo_active) {
                glm::vec3 fresh_min, fresh_max;
                if (scene.getNodeBounds(first_node->id, fresh_min, fresh_max)) {
                    node_bounds_min_ = fresh_min;
                    node_bounds_max_ = fresh_max;
                    node_bounds_orig_visualizer_world_transform_ = scene_coords::nodeVisualizerWorldTransform(scene, first_node->id);
                    node_bounds_world_scale_ = world_scale;
                    node_bounds_scale_active_ = true;
                }
            }

            std::unordered_set<core::NodeId> selected_ids;
            for (const auto& name : target_names) {
                if (const auto* node = scene.getNode(name)) {
                    selected_ids.insert(node->id);
                }
            }

            node_gizmo_node_names_.clear();
            for (const auto& name : target_names) {
                const auto* node = scene.getNode(name);
                if (!node)
                    continue;

                bool ancestor_selected = false;
                for (core::NodeId check_id = node->parent_id; check_id != core::NULL_NODE;) {
                    if (selected_ids.count(check_id)) {
                        ancestor_selected = true;
                        break;
                    }
                    const auto* parent = scene.getNodeById(check_id);
                    check_id = parent ? parent->parent_id : core::NULL_NODE;
                }

                if (!ancestor_selected) {
                    node_gizmo_node_names_.push_back(name);
                }
            }

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

        if (gizmo_changed && is_using) {
            if (node_gizmo_operation_ == GizmoOperation::Rotate) {
                const glm::mat3 delta_rot = extractRotation(delta_matrix);
                gizmo_cumulative_rotation_ = delta_rot * gizmo_cumulative_rotation_;
                const glm::mat4 world_delta = glm::translate(glm::mat4(1.0f), gizmo_pivot_) *
                                              glm::mat4(gizmo_cumulative_rotation_) *
                                              glm::translate(glm::mat4(1.0f), -gizmo_pivot_);

                for (size_t i = 0; i < node_gizmo_node_names_.size(); ++i) {
                    const glm::mat4 new_world_transform = world_delta * node_original_visualizer_world_transforms_[i];
                    if (const auto new_local_transform =
                            scene_coords::nodeLocalTransformFromVisualizerWorld(
                                scene_manager->getScene(),
                                node_gizmo_node_names_[i],
                                new_world_transform)) {
                        scene_manager->setNodeTransform(node_gizmo_node_names_[i], *new_local_transform);
                    }
                }
            } else if (node_gizmo_operation_ == GizmoOperation::Scale &&
                       !node_bounds_scale_active_ &&
                       (is_multi_selection || use_world_space)) {
                gizmo_cumulative_scale_ *= extractScale(delta_matrix);
                const glm::mat4 world_delta = glm::translate(glm::mat4(1.0f), gizmo_pivot_) *
                                              glm::scale(glm::mat4(1.0f), gizmo_cumulative_scale_) *
                                              glm::translate(glm::mat4(1.0f), -gizmo_pivot_);

                for (size_t i = 0; i < node_gizmo_node_names_.size(); ++i) {
                    const glm::mat4 new_world_transform = world_delta * node_original_visualizer_world_transforms_[i];
                    if (const auto new_local_transform =
                            scene_coords::nodeLocalTransformFromVisualizerWorld(
                                scene_manager->getScene(),
                                node_gizmo_node_names_[i],
                                new_world_transform)) {
                        scene_manager->setNodeTransform(node_gizmo_node_names_[i], *new_local_transform);
                    }
                }
            } else if (is_multi_selection) {
                if (node_gizmo_operation_ == GizmoOperation::Translate) {
                    const glm::vec3 new_gizmo_pos(gizmo_matrix[3]);
                    const glm::vec3 delta = new_gizmo_pos - gizmo_pivot_;
                    const glm::mat4 world_delta = glm::translate(glm::mat4(1.0f), delta);

                    for (size_t i = 0; i < node_gizmo_node_names_.size(); ++i) {
                        const glm::mat4 new_world_transform = world_delta * node_original_visualizer_world_transforms_[i];
                        if (const auto new_local_transform =
                                scene_coords::nodeLocalTransformFromVisualizerWorld(
                                    scene_manager->getScene(),
                                    node_gizmo_node_names_[i],
                                    new_world_transform)) {
                            scene_manager->setNodeTransform(node_gizmo_node_names_[i], *new_local_transform);
                        }
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

        if (!is_using && node_gizmo_active_) {
            node_gizmo_active_ = false;
            node_bounds_scale_active_ = false;
            node_bounds_cache_valid_ = false;
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
                    if (cropbox_node && cropbox_node->cropbox) {
                        const glm::mat4 cropbox_world = scene_coords::nodeVisualizerWorldTransform(scene, cropbox_id);
                        render_manager->setCropboxGizmoState(true, cropbox_node->cropbox->min,
                                                             cropbox_node->cropbox->max, cropbox_world);
                    }
                }

                const core::NodeId ellipsoid_id = scene.getEllipsoidForSplat(node->id);
                if (ellipsoid_id != core::NULL_NODE) {
                    const auto* ellipsoid_node = scene.getNodeById(ellipsoid_id);
                    if (ellipsoid_node && ellipsoid_node->ellipsoid) {
                        const glm::mat4 ellipsoid_world = scene_coords::nodeVisualizerWorldTransform(scene, ellipsoid_id);
                        render_manager->setEllipsoidGizmoState(true, ellipsoid_node->ellipsoid->radii,
                                                               ellipsoid_world);
                    }
                }
            }
        }

        overlay_drawlist.PopClipRect();
    }

    void GizmoManager::renderCropBoxGizmo(const UIContext& ctx, const ViewportLayout& viewport) {
        auto* const render_manager = ctx.viewer->getRenderingManager();
        auto* const scene_manager = ctx.viewer->getSceneManager();
        if (!render_manager || !scene_manager)
            return;

        const auto& settings = render_manager->getSettings();
        if (!settings.show_crop_box)
            return;

        core::NodeId cropbox_id = core::NULL_NODE;
        const core::SceneNode* cropbox_node = nullptr;

        if (scene_manager->getSelectedNodeType() == core::NodeType::CROPBOX) {
            cropbox_id = scene_manager->getSelectedNodeCropBoxId();
        }

        if (cropbox_id == core::NULL_NODE)
            return;

        cropbox_node = scene_manager->getScene().getNodeById(cropbox_id);
        if (!cropbox_node || !cropbox_node->visible || !cropbox_node->cropbox)
            return;
        if (!scene_manager->getScene().isNodeEffectivelyVisible(cropbox_id))
            return;

        const auto active_panel = resolveActiveGizmoPanel(ctx.viewer, viewport);
        if (!active_panel || !active_panel->valid())
            return;

        auto& vp = *active_panel->viewport;
        const glm::mat4 view = vp.getViewMatrix();
        const glm::ivec2 vp_size(static_cast<int>(active_panel->size.x), static_cast<int>(active_panel->size.y));
        const glm::mat4 projection = lfs::rendering::createProjectionMatrixFromFocal(
            vp_size, settings.focal_length_mm, settings.orthographic, settings.ortho_scale);

        const glm::vec3 cropbox_min = cropbox_node->cropbox->min;
        const glm::vec3 cropbox_max = cropbox_node->cropbox->max;
        const glm::mat4 world_transform = scene_coords::nodeVisualizerWorldTransform(scene_manager->getScene(), cropbox_id);

        const glm::vec3 local_size = cropbox_max - cropbox_min;
        const glm::vec3 world_scale = gizmo_ops::extractScale(world_transform);
        const glm::mat3 rotation = gizmo_ops::extractRotation(world_transform);
        const glm::vec3 translation = gizmo_ops::extractTranslation(world_transform);

        const bool use_world_space = (transform_space_ == TransformSpace::World);
        const GizmoOperation gizmo_op = current_operation_;

        const glm::vec3 local_pivot = (cropbox_min + cropbox_max) * 0.5f;
        const glm::vec3 pivot_world = translation + rotation * (local_pivot * world_scale);

        const bool gizmo_local_aligned = (gizmo_op == GizmoOperation::Scale) || !use_world_space;
        glm::vec3 transform_gizmo_pivot_world = pivot_world;
        glm::mat4 gizmo_matrix;
        if (cropbox_gizmo_active_ && gizmo_context_.isActive()) {
            const auto& target = gizmo_context_.targets[0];
            const glm::vec3 original_size = target.bounds_max - target.bounds_min;
            const glm::vec3 current_size = original_size * gizmo_context_.cumulative_scale;
            const glm::mat3 current_rotation = gizmo_context_.cumulative_rotation * target.rotation;
            const glm::vec3 current_pivot = gizmo_context_.pivot_world + gizmo_context_.cumulative_translation;
            transform_gizmo_pivot_world = current_pivot;

            gizmo_matrix = gizmo_ops::computeGizmoMatrix(
                current_pivot, current_rotation, current_size * world_scale,
                gizmo_context_.use_world_space, gizmo_op == GizmoOperation::Scale);
        } else {
            const glm::vec3 scaled_size = local_size * world_scale;
            gizmo_matrix = glm::translate(glm::mat4(1.0f), pivot_world);
            if (gizmo_local_aligned) {
                gizmo_matrix = gizmo_matrix * glm::mat4(rotation);
            }
            gizmo_matrix = glm::scale(gizmo_matrix, scaled_size);
        }

        const bool use_bounds = (gizmo_op == GizmoOperation::Scale);

        NativeOverlayDrawList overlay_drawlist;
        const glm::vec2 clip_min(active_panel->pos.x, active_panel->pos.y);
        const glm::vec2 clip_max(clip_min.x + active_panel->size.x, clip_min.y + active_panel->size.y);
        overlay_drawlist.PushClipRect(clip_min, clip_max, true);
        const auto& frame_input = viewer_->getWindowManager()->frameInput();
        const NativeGizmoInput gizmo_input = nativeGizmoInputFromFrame(frame_input);
        const bool snap_modifier = nativeControlModifierDown(frame_input);

        bool gizmo_changed = false;
        bool is_using = false;
        glm::mat4 delta_matrix(1.0f);
        bool bounds_result_valid = false;
        glm::vec3 bounds_result_center_world(0.0f);
        glm::vec3 bounds_result_local_size(0.0f);
        ScaleGizmoResult scale_result;
        const bool scale_gizmo_has_priority = use_bounds && (isScaleGizmoHovered() || isScaleGizmoActive());

        if (use_bounds) {
            const glm::vec3 safe_world_scale = glm::max(world_scale, glm::vec3(1e-6f));
            BoundsGizmoConfig bounds_config;
            bounds_config.id = panelGizmoId(CROPBOX_GIZMO_ID_BASE, active_panel->panel);
            bounds_config.viewport_pos = active_panel->pos;
            bounds_config.viewport_size = active_panel->size;
            bounds_config.view = view;
            bounds_config.projection = projection;
            bounds_config.center_world = glm::vec3(gizmo_matrix[3]);
            bounds_config.orientation_world = userFacingLocalRotation(gizmo_matrix);
            bounds_config.half_extents_world = gizmo_ops::extractScale(gizmo_matrix) * 0.5f;
            bounds_config.min_half_extents_world = safe_world_scale * (MIN_GIZMO_SCALE * 0.5f);
            bounds_config.draw_list = &overlay_drawlist;
            bounds_config.input = gizmo_input;
            bounds_config.input_enabled = !scale_gizmo_has_priority;
            bounds_config.snap = snap_modifier;
            bounds_config.snap_ratio = SCALE_SNAP_RATIO;

            const auto bounds_result = drawBoundsGizmo(bounds_config);
            is_using = bounds_result.active;
            gizmo_changed = bounds_result.changed;
            if (bounds_result.active) {
                const glm::mat3 box_rotation = gizmo_ops::extractRotation(gizmo_matrix);
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
        } else if (gizmo_op == GizmoOperation::Translate) {
            TranslationGizmoConfig translation_config;
            translation_config.id = panelGizmoId(CROPBOX_GIZMO_ID_BASE, active_panel->panel);
            translation_config.viewport_pos = active_panel->pos;
            translation_config.viewport_size = active_panel->size;
            translation_config.view = view;
            translation_config.projection = projection;
            translation_config.pivot_world = glm::vec3(gizmo_matrix[3]);
            translation_config.orientation_world =
                gizmo_local_aligned ? userFacingLocalRotation(gizmo_matrix) : glm::mat3(1.0f);
            translation_config.draw_list = &overlay_drawlist;
            translation_config.input = gizmo_input;
            translation_config.snap = snap_modifier;
            translation_config.snap_units = TRANSLATE_SNAP_UNITS;

            const auto translation_result = drawTranslationGizmo(translation_config);
            is_using = translation_result.active;
            gizmo_changed = translation_result.changed;
            delta_matrix = glm::translate(glm::mat4(1.0f), translation_result.delta_translation);
            if (translation_result.active) {
                const glm::vec3 translated_pivot = glm::vec3(gizmo_matrix[3]) + translation_result.delta_translation;
                gizmo_matrix[3] = glm::vec4(translated_pivot, 1.0f);
            }
            if (translation_result.hovered || translation_result.active) {
                guiFocusState().want_capture_mouse = true;
            }
        } else if (gizmo_op == GizmoOperation::Rotate) {
            RotationGizmoConfig rotation_config;
            rotation_config.id = panelGizmoId(CROPBOX_GIZMO_ID_BASE, active_panel->panel);
            rotation_config.viewport_pos = active_panel->pos;
            rotation_config.viewport_size = active_panel->size;
            rotation_config.view = view;
            rotation_config.projection = projection;
            rotation_config.pivot_world = glm::vec3(gizmo_matrix[3]);
            rotation_config.orientation_world =
                gizmo_local_aligned ? userFacingLocalRotation(gizmo_matrix) : glm::mat3(1.0f);
            rotation_config.draw_list = &overlay_drawlist;
            rotation_config.input = gizmo_input;
            rotation_config.snap = snap_modifier;
            rotation_config.snap_degrees = ROTATION_SNAP_DEGREES;

            const auto rotation_result = drawRotationGizmo(rotation_config);
            is_using = rotation_result.active;
            gizmo_changed = rotation_result.changed;
            delta_matrix = glm::mat4(rotation_result.delta_rotation);
            if (rotation_result.hovered || rotation_result.active) {
                guiFocusState().want_capture_mouse = true;
            }
        }

        if (use_bounds) {
            ScaleGizmoConfig scale_config;
            scale_config.id = panelGizmoId(CROPBOX_GIZMO_ID_BASE, active_panel->panel);
            scale_config.viewport_pos = active_panel->pos;
            scale_config.viewport_size = active_panel->size;
            scale_config.view = view;
            scale_config.projection = projection;
            scale_config.pivot_world = transform_gizmo_pivot_world;
            scale_config.orientation_world = userFacingLocalRotation(gizmo_matrix);
            scale_config.draw_list = &overlay_drawlist;
            scale_config.input = gizmo_input;
            scale_config.input_enabled = !isBoundsGizmoActive();
            scale_config.snap = snap_modifier;
            scale_config.snap_ratio = SCALE_SNAP_RATIO;

            scale_result = drawScaleGizmo(scale_config);
            is_using = is_using || scale_result.active;
            gizmo_changed = gizmo_changed || scale_result.changed;
            if (scale_result.changed) {
                delta_matrix = glm::scale(glm::mat4(1.0f), scale_result.delta_scale);
                gizmo_matrix[0] *= scale_result.delta_scale.x;
                gizmo_matrix[1] *= scale_result.delta_scale.y;
                gizmo_matrix[2] *= scale_result.delta_scale.z;
            }
            if (scale_result.hovered || scale_result.active) {
                guiFocusState().want_capture_mouse = true;
            }
        }

        if (is_using && !cropbox_gizmo_active_) {
            cropbox_gizmo_active_ = true;
            cropbox_node_name_ = cropbox_node->name;
            cropbox_data_before_drag_ = *cropbox_node->cropbox;
            cropbox_transform_before_drag_ = scene_manager->getNodeTransform(cropbox_node->name);

            gizmo_context_ = gizmo_ops::captureCropBox(
                scene_manager->getScene(),
                cropbox_node->name,
                pivot_world,
                local_pivot,
                transform_space_,
                pivot_mode_);
        }

        if (gizmo_changed && gizmo_context_.isActive()) {
            auto& scene = scene_manager->getScene();

            if (gizmo_op == GizmoOperation::Rotate) {
                const glm::mat3 delta_rot = gizmo_ops::extractRotation(delta_matrix);
                gizmo_ops::applyRotation(gizmo_context_, scene, delta_rot);
            } else if (gizmo_op == GizmoOperation::Scale) {
                glm::vec3 new_size;
                glm::vec3 new_pivot_world;
                if (bounds_result_valid) {
                    new_size = bounds_result_local_size;
                    new_pivot_world = bounds_result_center_world;
                } else {
                    new_size = glm::max(
                        gizmo_ops::extractScale(gizmo_matrix) / glm::max(world_scale, glm::vec3(1e-6f)),
                        glm::vec3(MIN_GIZMO_SCALE));
                    new_pivot_world = glm::vec3(gizmo_matrix[3]);
                }
                gizmo_ops::applyBoundsScale(gizmo_context_, scene, new_size);
                gizmo_ops::applyTranslation(gizmo_context_, scene, new_pivot_world);
            } else {
                const glm::vec3 new_pivot_world(gizmo_matrix[3]);
                gizmo_ops::applyTranslation(gizmo_context_, scene, new_pivot_world);
            }

            render_manager->markDirty(DirtyFlag::SPLATS | DirtyFlag::OVERLAY);
        }

        if (!is_using && cropbox_gizmo_active_) {
            cropbox_gizmo_active_ = false;
            gizmo_context_.reset();

            auto* node = scene_manager->getScene().getMutableNode(cropbox_node_name_);
            if (node && node->cropbox) {
                auto entry = std::make_unique<op::CropBoxUndoEntry>(
                    *scene_manager,
                    render_manager,
                    cropbox_node_name_,
                    cropbox_data_before_drag_,
                    cropbox_transform_before_drag_,
                    settings.show_crop_box,
                    settings.use_crop_box);
                if (entry->hasChanges()) {
                    op::undoHistory().push(std::move(entry));

                    using namespace lfs::core::events;
                    ui::CropBoxChanged{
                        .min_bounds = node->cropbox->min,
                        .max_bounds = node->cropbox->max,
                        .enabled = settings.use_crop_box}
                        .emit();
                }
            }
        }

        if (cropbox_gizmo_active_) {
            render_manager->setCropboxGizmoState(
                true, cropbox_node->cropbox->min, cropbox_node->cropbox->max,
                scene_coords::nodeVisualizerWorldTransform(scene_manager->getScene(), cropbox_id));
        } else {
            render_manager->setCropboxGizmoActive(false);
        }

        overlay_drawlist.PopClipRect();
    }

    void GizmoManager::renderEllipsoidGizmo(const UIContext& ctx, const ViewportLayout& viewport) {
        auto* const render_manager = ctx.viewer->getRenderingManager();
        auto* const scene_manager = ctx.viewer->getSceneManager();
        if (!render_manager || !scene_manager)
            return;

        const auto& settings = render_manager->getSettings();
        if (!settings.show_ellipsoid)
            return;

        core::NodeId ellipsoid_id = core::NULL_NODE;
        const core::SceneNode* ellipsoid_node = nullptr;

        if (scene_manager->getSelectedNodeType() == core::NodeType::ELLIPSOID) {
            ellipsoid_id = scene_manager->getSelectedNodeEllipsoidId();
        }

        if (ellipsoid_id == core::NULL_NODE)
            return;

        ellipsoid_node = scene_manager->getScene().getNodeById(ellipsoid_id);
        if (!ellipsoid_node || !ellipsoid_node->visible || !ellipsoid_node->ellipsoid)
            return;
        if (!scene_manager->getScene().isNodeEffectivelyVisible(ellipsoid_id))
            return;

        const auto active_panel = resolveActiveGizmoPanel(ctx.viewer, viewport);
        if (!active_panel || !active_panel->valid())
            return;

        auto& vp = *active_panel->viewport;
        const glm::mat4 view = vp.getViewMatrix();
        const glm::ivec2 vp_size(static_cast<int>(active_panel->size.x), static_cast<int>(active_panel->size.y));
        const glm::mat4 projection = lfs::rendering::createProjectionMatrixFromFocal(
            vp_size, settings.focal_length_mm, settings.orthographic, settings.ortho_scale);

        const glm::vec3 radii = ellipsoid_node->ellipsoid->radii;
        const glm::mat4 world_transform = scene_coords::nodeVisualizerWorldTransform(scene_manager->getScene(), ellipsoid_id);

        const glm::vec3 world_scale = gizmo_ops::extractScale(world_transform);
        const glm::mat3 rotation = gizmo_ops::extractRotation(world_transform);
        const glm::vec3 translation = gizmo_ops::extractTranslation(world_transform);

        const bool use_world_space = (transform_space_ == TransformSpace::World);
        const GizmoOperation gizmo_op = current_operation_;

        const glm::vec3 local_pivot(0.0f);
        const glm::vec3 pivot_world = translation;

        const bool gizmo_local_aligned = (gizmo_op == GizmoOperation::Scale) || !use_world_space;
        glm::vec3 transform_gizmo_pivot_world = pivot_world;
        glm::mat4 gizmo_matrix;
        if (ellipsoid_gizmo_active_ && gizmo_context_.isActive()) {
            const auto& target = gizmo_context_.targets[0];
            const glm::vec3 current_radii = target.radii * gizmo_context_.cumulative_scale;
            const glm::mat3 current_rotation = gizmo_context_.cumulative_rotation * target.rotation;
            const glm::vec3 current_pivot = gizmo_context_.pivot_world + gizmo_context_.cumulative_translation;
            transform_gizmo_pivot_world = current_pivot;

            gizmo_matrix = gizmo_ops::computeGizmoMatrix(
                current_pivot, current_rotation, current_radii * world_scale,
                gizmo_context_.use_world_space, gizmo_op == GizmoOperation::Scale);
        } else {
            const glm::vec3 scaled_radii = radii * world_scale;
            gizmo_matrix = glm::translate(glm::mat4(1.0f), pivot_world);
            if (gizmo_local_aligned) {
                gizmo_matrix = gizmo_matrix * glm::mat4(rotation);
            }
            gizmo_matrix = glm::scale(gizmo_matrix, scaled_radii);
        }

        const bool use_bounds = (gizmo_op == GizmoOperation::Scale);

        NativeOverlayDrawList overlay_drawlist;
        const glm::vec2 clip_min(active_panel->pos.x, active_panel->pos.y);
        const glm::vec2 clip_max(clip_min.x + active_panel->size.x, clip_min.y + active_panel->size.y);
        overlay_drawlist.PushClipRect(clip_min, clip_max, true);
        const auto& frame_input = viewer_->getWindowManager()->frameInput();
        const NativeGizmoInput gizmo_input = nativeGizmoInputFromFrame(frame_input);
        const bool snap_modifier = nativeControlModifierDown(frame_input);

        bool gizmo_changed = false;
        bool is_using = false;
        glm::mat4 delta_matrix(1.0f);
        bool bounds_result_valid = false;
        glm::vec3 bounds_result_center_world(0.0f);
        glm::vec3 bounds_result_radii(0.0f);
        ScaleGizmoResult scale_result;
        const bool scale_gizmo_has_priority = use_bounds && (isScaleGizmoHovered() || isScaleGizmoActive());

        if (use_bounds) {
            const glm::vec3 safe_world_scale = glm::max(world_scale, glm::vec3(1e-6f));
            BoundsGizmoConfig bounds_config;
            bounds_config.id = panelGizmoId(ELLIPSOID_GIZMO_ID_BASE, active_panel->panel);
            bounds_config.viewport_pos = active_panel->pos;
            bounds_config.viewport_size = active_panel->size;
            bounds_config.view = view;
            bounds_config.projection = projection;
            bounds_config.center_world = glm::vec3(gizmo_matrix[3]);
            bounds_config.orientation_world = userFacingLocalRotation(gizmo_matrix);
            bounds_config.half_extents_world = gizmo_ops::extractScale(gizmo_matrix);
            bounds_config.min_half_extents_world = safe_world_scale * MIN_GIZMO_SCALE;
            bounds_config.draw_list = &overlay_drawlist;
            bounds_config.input = gizmo_input;
            bounds_config.input_enabled = !scale_gizmo_has_priority;
            bounds_config.snap = snap_modifier;
            bounds_config.snap_ratio = SCALE_SNAP_RATIO;

            const auto bounds_result = drawBoundsGizmo(bounds_config);
            is_using = bounds_result.active;
            gizmo_changed = bounds_result.changed;
            if (bounds_result.active) {
                const glm::mat3 box_rotation = gizmo_ops::extractRotation(gizmo_matrix);
                gizmo_matrix[3] = glm::vec4(bounds_result.center_world, 1.0f);
                gizmo_matrix[0] = glm::vec4(box_rotation[0] * bounds_result.half_extents_world.x, 0.0f);
                gizmo_matrix[1] = glm::vec4(box_rotation[1] * bounds_result.half_extents_world.y, 0.0f);
                gizmo_matrix[2] = glm::vec4(box_rotation[2] * bounds_result.half_extents_world.z, 0.0f);
            }
            if (bounds_result.changed) {
                bounds_result_valid = true;
                bounds_result_center_world = bounds_result.center_world;
                bounds_result_radii =
                    glm::max(bounds_result.half_extents_world / safe_world_scale, glm::vec3(MIN_GIZMO_SCALE));
            }
            if (bounds_result.hovered || bounds_result.active) {
                guiFocusState().want_capture_mouse = true;
            }
        } else if (gizmo_op == GizmoOperation::Translate) {
            TranslationGizmoConfig translation_config;
            translation_config.id = panelGizmoId(ELLIPSOID_GIZMO_ID_BASE, active_panel->panel);
            translation_config.viewport_pos = active_panel->pos;
            translation_config.viewport_size = active_panel->size;
            translation_config.view = view;
            translation_config.projection = projection;
            translation_config.pivot_world = glm::vec3(gizmo_matrix[3]);
            translation_config.orientation_world =
                gizmo_local_aligned ? userFacingLocalRotation(gizmo_matrix) : glm::mat3(1.0f);
            translation_config.draw_list = &overlay_drawlist;
            translation_config.input = gizmo_input;
            translation_config.snap = snap_modifier;
            translation_config.snap_units = TRANSLATE_SNAP_UNITS;

            const auto translation_result = drawTranslationGizmo(translation_config);
            is_using = translation_result.active;
            gizmo_changed = translation_result.changed;
            delta_matrix = glm::translate(glm::mat4(1.0f), translation_result.delta_translation);
            if (translation_result.active) {
                const glm::vec3 translated_pivot = glm::vec3(gizmo_matrix[3]) + translation_result.delta_translation;
                gizmo_matrix[3] = glm::vec4(translated_pivot, 1.0f);
            }
            if (translation_result.hovered || translation_result.active) {
                guiFocusState().want_capture_mouse = true;
            }
        } else if (gizmo_op == GizmoOperation::Rotate) {
            RotationGizmoConfig rotation_config;
            rotation_config.id = panelGizmoId(ELLIPSOID_GIZMO_ID_BASE, active_panel->panel);
            rotation_config.viewport_pos = active_panel->pos;
            rotation_config.viewport_size = active_panel->size;
            rotation_config.view = view;
            rotation_config.projection = projection;
            rotation_config.pivot_world = glm::vec3(gizmo_matrix[3]);
            rotation_config.orientation_world =
                gizmo_local_aligned ? userFacingLocalRotation(gizmo_matrix) : glm::mat3(1.0f);
            rotation_config.draw_list = &overlay_drawlist;
            rotation_config.input = gizmo_input;
            rotation_config.snap = snap_modifier;
            rotation_config.snap_degrees = ROTATION_SNAP_DEGREES;

            const auto rotation_result = drawRotationGizmo(rotation_config);
            is_using = rotation_result.active;
            gizmo_changed = rotation_result.changed;
            delta_matrix = glm::mat4(rotation_result.delta_rotation);
            if (rotation_result.hovered || rotation_result.active) {
                guiFocusState().want_capture_mouse = true;
            }
        }

        if (use_bounds) {
            ScaleGizmoConfig scale_config;
            scale_config.id = panelGizmoId(ELLIPSOID_GIZMO_ID_BASE, active_panel->panel);
            scale_config.viewport_pos = active_panel->pos;
            scale_config.viewport_size = active_panel->size;
            scale_config.view = view;
            scale_config.projection = projection;
            scale_config.pivot_world = transform_gizmo_pivot_world;
            scale_config.orientation_world = userFacingLocalRotation(gizmo_matrix);
            scale_config.draw_list = &overlay_drawlist;
            scale_config.input = gizmo_input;
            scale_config.input_enabled = !isBoundsGizmoActive();
            scale_config.snap = snap_modifier;
            scale_config.snap_ratio = SCALE_SNAP_RATIO;

            scale_result = drawScaleGizmo(scale_config);
            is_using = is_using || scale_result.active;
            gizmo_changed = gizmo_changed || scale_result.changed;
            if (scale_result.changed) {
                delta_matrix = glm::scale(glm::mat4(1.0f), scale_result.delta_scale);
                gizmo_matrix[0] *= scale_result.delta_scale.x;
                gizmo_matrix[1] *= scale_result.delta_scale.y;
                gizmo_matrix[2] *= scale_result.delta_scale.z;
            }
            if (scale_result.hovered || scale_result.active) {
                guiFocusState().want_capture_mouse = true;
            }
        }

        if (is_using && !ellipsoid_gizmo_active_) {
            ellipsoid_gizmo_active_ = true;
            ellipsoid_node_name_ = ellipsoid_node->name;
            ellipsoid_data_before_drag_ = *ellipsoid_node->ellipsoid;
            ellipsoid_transform_before_drag_ = scene_manager->getNodeTransform(ellipsoid_node->name);

            gizmo_context_ = gizmo_ops::captureEllipsoid(
                scene_manager->getScene(),
                ellipsoid_node->name,
                pivot_world,
                local_pivot,
                transform_space_,
                pivot_mode_);
        }

        if (gizmo_changed && gizmo_context_.isActive()) {
            auto& scene = scene_manager->getScene();

            if (gizmo_op == GizmoOperation::Rotate) {
                const glm::mat3 delta_rot = gizmo_ops::extractRotation(delta_matrix);
                gizmo_ops::applyRotation(gizmo_context_, scene, delta_rot);
            } else if (gizmo_op == GizmoOperation::Scale) {
                glm::vec3 new_radii;
                glm::vec3 new_pivot_world;
                if (bounds_result_valid) {
                    new_radii = bounds_result_radii;
                    new_pivot_world = bounds_result_center_world;
                } else {
                    new_radii = glm::max(
                        gizmo_ops::extractScale(gizmo_matrix) / glm::max(world_scale, glm::vec3(1e-6f)),
                        glm::vec3(MIN_GIZMO_SCALE));
                    new_pivot_world = glm::vec3(gizmo_matrix[3]);
                }
                gizmo_ops::applyBoundsScale(gizmo_context_, scene, new_radii);
                gizmo_ops::applyTranslation(gizmo_context_, scene, new_pivot_world);
            } else {
                const glm::vec3 new_pivot_world(gizmo_matrix[3]);
                gizmo_ops::applyTranslation(gizmo_context_, scene, new_pivot_world);
            }

            render_manager->markDirty(DirtyFlag::SPLATS | DirtyFlag::OVERLAY);
        }

        if (!is_using && ellipsoid_gizmo_active_) {
            ellipsoid_gizmo_active_ = false;
            gizmo_context_.reset();

            auto* node = scene_manager->getScene().getMutableNode(ellipsoid_node_name_);
            if (node && node->ellipsoid) {
                auto entry = std::make_unique<op::EllipsoidUndoEntry>(
                    *scene_manager,
                    render_manager,
                    ellipsoid_node_name_,
                    ellipsoid_data_before_drag_,
                    ellipsoid_transform_before_drag_,
                    settings.show_ellipsoid,
                    settings.use_ellipsoid);
                if (entry->hasChanges()) {
                    op::undoHistory().push(std::move(entry));

                    using namespace lfs::core::events;
                    ui::EllipsoidChanged{
                        .radii = node->ellipsoid->radii,
                        .enabled = settings.use_ellipsoid}
                        .emit();
                }
            }
        }

        if (ellipsoid_gizmo_active_) {
            const glm::mat4 current_world_transform =
                scene_coords::nodeVisualizerWorldTransform(scene_manager->getScene(), ellipsoid_id);
            render_manager->setEllipsoidGizmoState(true, ellipsoid_node->ellipsoid->radii,
                                                   current_world_transform);
        } else {
            render_manager->setEllipsoidGizmoActive(false);
        }

        overlay_drawlist.PopClipRect();
    }

    void GizmoManager::renderViewportGizmo(const ViewportLayout& viewport) {
        if (!show_viewport_gizmo_ || viewport.size.x <= 0 || viewport.size.y <= 0)
            return;

        auto* rendering_manager = viewer_->getRenderingManager();
        if (!rendering_manager)
            return;

        const glm::vec2 vp_pos(viewport.pos.x, viewport.pos.y);
        const glm::vec2 vp_size(viewport.size.x, viewport.size.y);
        auto panels = collectViewportGizmoPanels(viewer_, vp_pos, vp_size);
        if (panels.empty()) {
            return;
        }

        const auto find_panel = [&](const SplitViewPanelId panel_id) -> ViewportGizmoPanelTarget* {
            for (auto& panel : panels) {
                if (panel.panel == panel_id) {
                    return &panel;
                }
            }
            return nullptr;
        };

        const auto& frame_input = viewer_->getWindowManager()->frameInput();
        const float mouse_x = frame_input.mouse_x;
        const float mouse_y = frame_input.mouse_y;

        const bool ui_wants_mouse = guiFocusState().want_capture_mouse;
        int hovered_axis = -1;
        ViewportGizmoPanelTarget* hovered_panel = nullptr;
        if (!ui_wants_mouse) {
            for (auto& panel : panels) {
                const float gizmo_x = panel.pos.x + panel.size.x - VIEWPORT_GIZMO_SIZE - VIEWPORT_GIZMO_MARGIN_X;
                const float gizmo_y = panel.pos.y + VIEWPORT_GIZMO_MARGIN_Y;
                const bool mouse_in_gizmo = mouse_x >= gizmo_x &&
                                            mouse_x <= gizmo_x + VIEWPORT_GIZMO_SIZE &&
                                            mouse_y >= gizmo_y &&
                                            mouse_y <= gizmo_y + VIEWPORT_GIZMO_SIZE;
                if (!mouse_in_gizmo) {
                    continue;
                }

                if (const auto layout = buildViewportGizmoLayout(
                        panel, VIEWPORT_GIZMO_SIZE, VIEWPORT_GIZMO_MARGIN_X, VIEWPORT_GIZMO_MARGIN_Y)) {
                    hovered_axis = hitTestViewportGizmoLayout(*layout, glm::vec2(mouse_x, mouse_y));
                }
                hovered_panel = &panel;
                break;
            }
        }
        if (!ui_wants_mouse) {
            const glm::vec2 capture_mouse_pos(mouse_x, mouse_y);
            const float time = static_cast<float>(SDL_GetTicks()) / 1000.0f;

            if (frame_input.mouse_clicked[0] && hovered_panel) {
                if (auto* const input_controller = viewer_->getInputController()) {
                    input_controller->setFocusedSplitPanel(hovered_panel->panel);
                } else {
                    rendering_manager->setFocusedSplitPanel(hovered_panel->panel);
                }

                auto& active_viewport = *hovered_panel->viewport;
                if (hovered_axis >= 0 && hovered_axis <= 5) {
                    const int axis = hovered_axis % 3;
                    const bool negative = hovered_axis >= 3;
                    active_viewport.camera.setAxisAlignedView(axis, negative);
                    rendering_manager->setGridPlaneForPanel(hovered_panel->panel, axis);
                    rendering_manager->markDirty(DirtyFlag::CAMERA);
                } else {
                    viewport_gizmo_dragging_ = true;
                    viewport_gizmo_active_panel_ = hovered_panel->panel;
                    active_viewport.camera.startRotateAroundCenter(capture_mouse_pos, time);
                    if (SDL_Window* const window = viewer_->getWindow()) {
                        float fx, fy;
                        SDL_GetMouseState(&fx, &fy);
                        gizmo_drag_start_cursor_ = {fx, fy};
                        SDL_SetWindowRelativeMouseMode(window, true);
                    }
                }
            }

            if (viewport_gizmo_dragging_) {
                if (auto* const active_panel = find_panel(viewport_gizmo_active_panel_);
                    active_panel && frame_input.mouse_down[0]) {
                    if (const auto* const input_controller = viewer_->getInputController();
                        input_controller &&
                        input_controller->cameraNavigationMode() ==
                            InputController::CameraNavigationMode::Trackball) {
                        active_panel->viewport->camera.updateTrackballRotateAroundCenter(capture_mouse_pos, time);
                    } else {
                        active_panel->viewport->camera.updateRotateAroundCenter(capture_mouse_pos, time);
                    }
                    rendering_manager->markDirty(DirtyFlag::CAMERA);
                } else {
                    if (auto* const active_panel = find_panel(viewport_gizmo_active_panel_)) {
                        active_panel->viewport->camera.endRotateAroundCenter();
                        if (const auto* const input_controller = viewer_->getInputController();
                            input_controller && input_controller->cameraViewSnapEnabled()) {
                            constexpr float kAxisSnapAngleDegrees = 10.0f;
                            int snapped_axis = -1;
                            if (active_panel->viewport->camera.snapToNearestAxisView(
                                    kAxisSnapAngleDegrees, &snapped_axis, nullptr)) {
                                rendering_manager->setGridPlaneForPanel(active_panel->panel, snapped_axis);
                            }
                        }
                        rendering_manager->markDirty(DirtyFlag::CAMERA);
                    }
                    viewport_gizmo_dragging_ = false;

                    if (SDL_Window* const window = viewer_->getWindow()) {
                        SDL_SetWindowRelativeMouseMode(window, false);
                        SDL_WarpMouseInWindow(window,
                                              static_cast<float>(gizmo_drag_start_cursor_.x),
                                              static_cast<float>(gizmo_drag_start_cursor_.y));
                    }
                }
            }
        }
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
        rm->markDirty(DirtyFlag::SPLATS | DirtyFlag::OVERLAY);
    }

    void GizmoManager::deactivateAllTools() {
        python::cancel_active_operator();
        if (auto* const t = viewer_->getBrushTool())
            t->setEnabled(false);
        if (auto* const t = viewer_->getAlignTool())
            t->setEnabled(false);

        if (auto* const sm = viewer_->getSceneManager()) {
            sm->applyDeleted();
        }

        auto& editor = viewer_->getEditorContext();
        editor.setActiveTool(ToolType::None);
        current_operation_ = GizmoOperation::Translate;
    }

    void GizmoManager::setSelectionSubMode(SelectionSubMode mode) {
        selection_mode_ = mode;

        if (auto* rm = viewer_->getRenderingManager()) {
            rm->setSelectionPreviewMode(toSelectionPreviewMode(mode));
        }
    }

    bool GizmoManager::isPositionInViewportGizmo(const double x, const double y) const {
        if (!show_viewport_gizmo_)
            return false;

        const auto vp_pos = viewer_->getGuiManager()->getViewportPos();
        const auto vp_size = viewer_->getGuiManager()->getViewportSize();
        const auto panels = collectViewportGizmoPanels(viewer_, vp_pos, vp_size);
        for (const auto& panel : panels) {
            const float gizmo_x = panel.pos.x + panel.size.x - VIEWPORT_GIZMO_SIZE - VIEWPORT_GIZMO_MARGIN_X;
            const float gizmo_y = panel.pos.y + VIEWPORT_GIZMO_MARGIN_Y;
            if (x >= gizmo_x && x <= gizmo_x + VIEWPORT_GIZMO_SIZE &&
                y >= gizmo_y && y <= gizmo_y + VIEWPORT_GIZMO_SIZE) {
                return true;
            }
        }
        return false;
    }

    ToolType GizmoManager::getCurrentToolMode() const {
        return viewer_->getEditorContext().getActiveTool();
    }

    void GizmoManager::openPieMenu(ImVec2 cursor_pos) {
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

    void GizmoManager::onPieMenuMouseMove(ImVec2 pos) {
        pie_menu_.onMouseMove(pos);
    }

    void GizmoManager::onPieMenuClick(ImVec2 pos) {
        pie_menu_.onMouseClick(pos);
        handlePieMenuSelection();
    }

    void GizmoManager::renderPieMenu() {
        if (!pie_menu_.isOpen())
            return;

        auto* drawlist = ImGui::GetForegroundDrawList();
        pie_menu_.draw(drawlist);
    }

    void GizmoManager::handlePieMenuSelection() {
        if (!pie_menu_.hasSelection())
            return;

        const auto& tool_id = pie_menu_.getSelectedId();

        if (!tool_id.empty()) {
            if (tool_id == "builtin.cropbox" || tool_id == "builtin.ellipsoid") {
                addCropObject(tool_id == "builtin.cropbox");
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

    void GizmoManager::addCropObject(bool is_cropbox) {
        auto* sm = viewer_->getSceneManager();
        if (!sm)
            return;

        const auto selected = sm->getSelectedNodeNames();
        for (const auto& name : selected) {
            const auto* node = sm->getScene().getNode(name);
            if (!node)
                continue;

            if (is_cropbox) {
                if (node->type != core::NodeType::SPLAT && node->type != core::NodeType::POINTCLOUD)
                    continue;

                auto cropbox_id = cap::ensureCropBox(*sm, viewer_->getRenderingManager(), node->id);
                if (!cropbox_id) {
                    LOG_WARN("Failed to add cropbox for '{}': {}", name, cropbox_id.error());
                    continue;
                }

                if (const auto* cropbox = sm->getScene().getNodeById(*cropbox_id)) {
                    if (auto result = cap::selectNode(*sm, cropbox->name); !result) {
                        LOG_WARN("Failed to select cropbox '{}': {}", cropbox->name, result.error());
                    }
                }
            } else {
                if (node->type != core::NodeType::SPLAT)
                    continue;
                lfs::core::events::cmd::AddCropEllipsoid{.node_name = name}.emit();
            }

            sm->syncCropBoxToRenderSettings();
            current_operation_ = GizmoOperation::Translate;
            return;
        }
    }

    void GizmoManager::handleCropAction(const std::string& action_id) {
        using namespace lfs::core::events;

        auto* sm = viewer_->getSceneManager();
        const bool is_cropbox = sm && sm->getSelectedNodeType() == core::NodeType::CROPBOX;

        if (action_id == "crop.translate") {
            current_operation_ = GizmoOperation::Translate;
        } else if (action_id == "crop.rotate") {
            current_operation_ = GizmoOperation::Rotate;
        } else if (action_id == "crop.scale") {
            current_operation_ = GizmoOperation::Scale;
        } else if (action_id == "crop.apply") {
            if (is_cropbox)
                cmd::ApplyCropBox{}.emit();
            else
                cmd::ApplyEllipsoid{}.emit();
        } else if (action_id == "crop.fit") {
            if (is_cropbox) {
                auto cropbox_id = cap::resolveCropBoxId(*sm, std::nullopt);
                if (!cropbox_id)
                    return;
                if (auto result = cap::fitCropBoxToParent(*sm, viewer_->getRenderingManager(), *cropbox_id, false);
                    !result) {
                    LOG_WARN("Failed to fit cropbox: {}", result.error());
                }
            } else {
                cmd::FitEllipsoidToScene{.use_percentile = false}.emit();
            }
        } else if (action_id == "crop.fit_trim") {
            if (is_cropbox) {
                auto cropbox_id = cap::resolveCropBoxId(*sm, std::nullopt);
                if (!cropbox_id)
                    return;
                if (auto result = cap::fitCropBoxToParent(*sm, viewer_->getRenderingManager(), *cropbox_id, true);
                    !result) {
                    LOG_WARN("Failed to trim-fit cropbox: {}", result.error());
                }
            } else {
                cmd::FitEllipsoidToScene{.use_percentile = true}.emit();
            }
        } else if (action_id == "crop.invert") {
            if (!is_cropbox)
                return;
            cmd::ToggleCropInverse{}.emit();
        } else if (action_id == "crop.reset") {
            if (is_cropbox) {
                auto cropbox_id = cap::resolveCropBoxId(*sm, std::nullopt);
                if (!cropbox_id)
                    return;
                if (auto result = cap::resetCropBox(*sm, viewer_->getRenderingManager(), *cropbox_id); !result) {
                    LOG_WARN("Failed to reset cropbox: {}", result.error());
                }
            } else {
                cmd::ResetEllipsoid{}.emit();
            }
        } else if (action_id == "crop.delete") {
            if (!sm)
                return;
            const core::NodeId id = is_cropbox ? sm->getSelectedNodeCropBoxId()
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

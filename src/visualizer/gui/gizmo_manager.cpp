/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

// clang-format off
#include <glad/glad.h>
// clang-format on

#include "gui/gizmo_manager.hpp"
#include "core/events.hpp"
#include "core/logger.hpp"
#include "core/scene.hpp"
#include "gui/gui_manager.hpp"
#include "gui/ui_widgets.hpp"
#include "operator/operator_id.hpp"
#include "operator/operator_registry.hpp"
#include "python/python_runtime.hpp"
#include "rendering/rendering.hpp"
#include "rendering/rendering_manager.hpp"
#include "scene/scene_manager.hpp"
#include "theme/theme.hpp"
#include "tools/align_tool.hpp"
#include "tools/brush_tool.hpp"
#include "tools/selection_tool.hpp"
#include "tools/unified_tool_registry.hpp"
#include "visualizer_impl.hpp"
#include <GLFW/glfw3.h>
#include <array>
#include <glm/gtc/type_ptr.hpp>
#include <unordered_set>

namespace lfs::vis::gui {

    using ToolType = lfs::vis::ToolType;

    constexpr float GIZMO_AXIS_LIMIT = 0.0001f;
    constexpr float MIN_GIZMO_SCALE = 0.001f;

    namespace {
        inline glm::mat3 extractRotation(const glm::mat4& m) {
            return glm::mat3(glm::normalize(glm::vec3(m[0])), glm::normalize(glm::vec3(m[1])),
                             glm::normalize(glm::vec3(m[2])));
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
        });

        ui::NodeDeselected::when([this](const auto&) {
            python::cancel_active_operator();
            if (auto* const t = viewer_->getBrushTool())
                t->setEnabled(false);
            if (auto* const t = viewer_->getAlignTool())
                t->setEnabled(false);
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

            static constexpr std::array<const char*, 8> TOOL_IDS = {
                nullptr, "builtin.select", "builtin.translate", "builtin.rotate",
                "builtin.scale", "builtin.brush", "builtin.align", "builtin.mirror"};
            auto& registry = UnifiedToolRegistry::instance();
            const auto idx = static_cast<size_t>(tool);
            if (idx < TOOL_IDS.size() && TOOL_IDS[idx]) {
                registry.setActiveTool(TOOL_IDS[idx]);
            } else {
                registry.clearActiveTool();
            }

            switch (tool) {
            case ToolType::Translate:
                current_operation_ = ImGuizmo::TRANSLATE;
                LOG_INFO("SetToolbarTool: TRANSLATE");
                break;
            case ToolType::Rotate:
                current_operation_ = ImGuizmo::ROTATE;
                LOG_INFO("SetToolbarTool: ROTATE");
                break;
            case ToolType::Scale:
                current_operation_ = ImGuizmo::SCALE;
                LOG_INFO("SetToolbarTool: SCALE");
                break;
            case ToolType::Selection:
                break;
            default:
                LOG_INFO("SetToolbarTool: tool_mode={}", e.tool_mode);
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

            const glm::mat4 world_transform = sm->getScene().getWorldTransform(cropbox_id);

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

            const glm::mat4 world_transform = sm->getScene().getWorldTransform(ellipsoid_id);
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

            const core::NodeId cropbox_id = sm->getSelectedNodeCropBoxId();
            if (cropbox_id == core::NULL_NODE)
                return;

            auto* node = sm->getScene().getMutableNode(sm->getScene().getNodeById(cropbox_id)->name);
            if (!node || !node->cropbox)
                return;

            node->cropbox->inverse = !node->cropbox->inverse;
            sm->getScene().invalidateCache();
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
                    node_gizmo_operation_ = ImGuizmo::TRANSLATE;
                    current_operation_ = ImGuizmo::TRANSLATE;
                } else if (gizmo_type == "rotate") {
                    node_gizmo_operation_ = ImGuizmo::ROTATE;
                    current_operation_ = ImGuizmo::ROTATE;
                } else if (gizmo_type == "scale") {
                    node_gizmo_operation_ = ImGuizmo::SCALE;
                    current_operation_ = ImGuizmo::SCALE;
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
                    auto mode = lfs::rendering::SelectionMode::Centers;
                    switch (selection_mode_) {
                    case SelectionSubMode::Centers: mode = lfs::rendering::SelectionMode::Centers; break;
                    case SelectionSubMode::Rectangle: mode = lfs::rendering::SelectionMode::Rectangle; break;
                    case SelectionSubMode::Polygon: mode = lfs::rendering::SelectionMode::Polygon; break;
                    case SelectionSubMode::Lasso: mode = lfs::rendering::SelectionMode::Lasso; break;
                    case SelectionSubMode::Rings: mode = lfs::rendering::SelectionMode::Rings; break;
                    }
                    rm->setSelectionMode(mode);

                    if (selection_mode_ != previous_selection_mode_) {
                        if (selection_tool)
                            selection_tool->onSelectionModeChanged();

                        if (selection_mode_ == SelectionSubMode::Rings) {
                            auto settings = rm->getSettings();
                            settings.show_rings = true;
                            settings.show_center_markers = false;
                            rm->updateSettings(settings);
                        }
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
        if (selected_type == core::NodeType::CROPBOX || selected_type == core::NodeType::ELLIPSOID)
            return;

        const auto& scene = scene_manager->getScene();
        const auto selected_names = scene_manager->getSelectedNodeNames();
        bool any_visible = false;
        for (const auto& name : selected_names) {
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
        const bool is_multi_selection = (selected_names.size() > 1);

        auto& vp = ctx.viewer->getViewport();
        const glm::mat4 view = vp.getViewMatrix();
        const glm::ivec2 vp_size(static_cast<int>(viewport.size.x), static_cast<int>(viewport.size.y));
        const glm::mat4 projection = lfs::rendering::createProjectionMatrixFromFocal(
            vp_size, settings.focal_length_mm, settings.orthographic, settings.ortho_scale);

        const bool use_world_space = (transform_space_ == TransformSpace::World) || is_multi_selection;

        const glm::vec3 local_pivot = (pivot_mode_ == PivotMode::Origin)
                                          ? glm::vec3(0.0f)
                                          : scene_manager->getSelectionCenter();

        const glm::vec3 gizmo_position = node_gizmo_active_
                                             ? gizmo_pivot_
                                             : (is_multi_selection
                                                    ? scene_manager->getSelectionWorldCenter()
                                                    : glm::vec3(scene_manager->getSelectedNodeWorldTransform() *
                                                                glm::vec4(local_pivot, 1.0f)));

        glm::mat4 gizmo_matrix(1.0f);
        gizmo_matrix[3] = glm::vec4(gizmo_position, 1.0f);

        if (!is_multi_selection && !use_world_space) {
            const glm::mat3 rotation_scale(scene_manager->getSelectedNodeWorldTransform());
            gizmo_matrix[0] = glm::vec4(rotation_scale[0], 0.0f);
            gizmo_matrix[1] = glm::vec4(rotation_scale[1], 0.0f);
            gizmo_matrix[2] = glm::vec4(rotation_scale[2], 0.0f);
        }

        ImGuizmo::SetOrthographic(settings.orthographic);
        ImGuizmo::SetRect(viewport.pos.x, viewport.pos.y, viewport.size.x, viewport.size.y);
        ImGuizmo::SetAxisLimit(GIZMO_AXIS_LIMIT);
        ImGuizmo::SetPlaneLimit(GIZMO_AXIS_LIMIT);

        const bool is_using = ImGuizmo::IsUsing();

        if (!is_using) {
            node_hovered_axis_ = ImGuizmo::IsOver(ImGuizmo::TRANSLATE_X) ||
                                 ImGuizmo::IsOver(ImGuizmo::TRANSLATE_Y) ||
                                 ImGuizmo::IsOver(ImGuizmo::TRANSLATE_Z);
            ImGuizmo::SetAxisMask(false, false, false);
        } else {
            ImGuizmo::SetAxisMask(node_hovered_axis_, node_hovered_axis_, node_hovered_axis_);
        }

        const bool modal_open = ImGui::IsPopupOpen("", ImGuiPopupFlags_AnyPopupId | ImGuiPopupFlags_AnyPopupLevel);
        ImDrawList* overlay_drawlist = modal_open ? ImGui::GetBackgroundDrawList() : ImGui::GetForegroundDrawList();
        const ImVec2 clip_min(viewport.pos.x, viewport.pos.y);
        const ImVec2 clip_max(clip_min.x + viewport.size.x, clip_min.y + viewport.size.y);
        overlay_drawlist->PushClipRect(clip_min, clip_max, true);
        ImGuizmo::SetDrawlist(overlay_drawlist);

        const ImGuizmo::MODE gizmo_mode = use_world_space ? ImGuizmo::WORLD : ImGuizmo::LOCAL;

        glm::mat4 delta_matrix;
        const bool gizmo_changed = ImGuizmo::Manipulate(
            glm::value_ptr(view), glm::value_ptr(projection),
            node_gizmo_operation_, gizmo_mode,
            glm::value_ptr(gizmo_matrix), glm::value_ptr(delta_matrix), nullptr);

        if (node_gizmo_operation_ == ImGuizmo::ROTATE) {
            const glm::vec4 clip_pos = projection * view * glm::vec4(gizmo_position, 1.0f);
            if (clip_pos.w > 0.0f) {
                const glm::vec2 ndc(clip_pos.x / clip_pos.w, clip_pos.y / clip_pos.w);
                const ImVec2 screen_pos(viewport.pos.x + (ndc.x * 0.5f + 0.5f) * viewport.size.x,
                                        viewport.pos.y + (-ndc.y * 0.5f + 0.5f) * viewport.size.y);
                constexpr float PIVOT_RADIUS = 4.0f;
                constexpr ImU32 PIVOT_COLOR = IM_COL32(255, 255, 255, 200);
                constexpr ImU32 PIVOT_OUTLINE = IM_COL32(0, 0, 0, 200);
                overlay_drawlist->AddCircleFilled(screen_pos, PIVOT_RADIUS + 1.0f, PIVOT_OUTLINE);
                overlay_drawlist->AddCircleFilled(screen_pos, PIVOT_RADIUS, PIVOT_COLOR);
            }
        }

        if (is_using && !node_gizmo_active_) {
            node_gizmo_active_ = true;
            gizmo_pivot_ = gizmo_position;
            gizmo_cumulative_rotation_ = glm::mat3(1.0f);
            gizmo_cumulative_scale_ = glm::vec3(1.0f);

            std::unordered_set<core::NodeId> selected_ids;
            for (const auto& name : selected_names) {
                if (const auto* node = scene.getNode(name)) {
                    selected_ids.insert(node->id);
                }
            }

            node_gizmo_node_names_.clear();
            for (const auto& name : selected_names) {
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
            node_original_world_positions_.clear();
            node_parent_world_inverses_.clear();
            node_original_rotations_.clear();
            node_original_scales_.clear();

            for (const auto& name : node_gizmo_node_names_) {
                const auto* node = scene.getNode(name);
                if (!node)
                    continue;

                const glm::mat4 world_t = scene.getWorldTransform(node->id);
                const glm::mat4 local_t = node->local_transform.get();
                node_transforms_before_drag_.push_back(local_t);
                node_original_rotations_.push_back(extractRotation(local_t));
                node_original_scales_.push_back(extractScale(local_t));

                glm::mat4 parent_world(1.0f);
                if (node->parent_id != core::NULL_NODE) {
                    parent_world = scene.getWorldTransform(node->parent_id);
                }

                node_original_world_positions_.emplace_back(world_t[3]);
                node_parent_world_inverses_.push_back(glm::inverse(parent_world));
            }
        }

        if (gizmo_changed && is_using) {
            if (is_multi_selection) {
                if (node_gizmo_operation_ == ImGuizmo::TRANSLATE) {
                    const glm::vec3 new_gizmo_pos(gizmo_matrix[3]);
                    const glm::vec3 delta = new_gizmo_pos - gizmo_pivot_;

                    for (size_t i = 0; i < node_gizmo_node_names_.size(); ++i) {
                        const glm::mat4& original_local = node_transforms_before_drag_[i];
                        const glm::vec3& original_world_pos = node_original_world_positions_[i];
                        const glm::mat4& parent_inv = node_parent_world_inverses_[i];

                        const glm::vec3 new_world_pos = original_world_pos + delta;
                        const glm::vec3 new_local_pos = glm::vec3(parent_inv * glm::vec4(new_world_pos, 1.0f));

                        glm::mat4 new_transform = original_local;
                        new_transform[3] = glm::vec4(new_local_pos, 1.0f);
                        scene_manager->setNodeTransform(node_gizmo_node_names_[i], new_transform);
                    }
                } else if (node_gizmo_operation_ == ImGuizmo::ROTATE) {
                    const glm::mat3 delta_rot = extractRotation(delta_matrix);
                    gizmo_cumulative_rotation_ = delta_rot * gizmo_cumulative_rotation_;

                    for (size_t i = 0; i < node_gizmo_node_names_.size(); ++i) {
                        const glm::vec3& original_world_pos = node_original_world_positions_[i];
                        const glm::mat4& parent_inv = node_parent_world_inverses_[i];
                        const glm::mat3& original_rot = node_original_rotations_[i];
                        const glm::vec3& original_scale = node_original_scales_[i];

                        const glm::vec3 offset = original_world_pos - gizmo_pivot_;
                        const glm::vec3 rotated_offset = gizmo_cumulative_rotation_ * offset;
                        const glm::vec3 new_world_pos = gizmo_pivot_ + rotated_offset;
                        const glm::vec3 new_local_pos = glm::vec3(parent_inv * glm::vec4(new_world_pos, 1.0f));

                        const glm::mat3 parent_rot = extractRotation(glm::inverse(parent_inv));
                        const glm::mat3 parent_rot_inv = glm::transpose(parent_rot);
                        const glm::mat3 local_delta_rot = parent_rot_inv * gizmo_cumulative_rotation_ * parent_rot;
                        const glm::mat3 new_rot = local_delta_rot * original_rot;

                        glm::mat4 new_transform(1.0f);
                        new_transform[0] = glm::vec4(new_rot[0] * original_scale.x, 0.0f);
                        new_transform[1] = glm::vec4(new_rot[1] * original_scale.y, 0.0f);
                        new_transform[2] = glm::vec4(new_rot[2] * original_scale.z, 0.0f);
                        new_transform[3] = glm::vec4(new_local_pos, 1.0f);
                        scene_manager->setNodeTransform(node_gizmo_node_names_[i], new_transform);
                    }
                } else if (node_gizmo_operation_ == ImGuizmo::SCALE) {
                    gizmo_cumulative_scale_ *= extractScale(delta_matrix);

                    const glm::mat3 world_scale(gizmo_cumulative_scale_.x, 0.0f, 0.0f,
                                                0.0f, gizmo_cumulative_scale_.y, 0.0f,
                                                0.0f, 0.0f, gizmo_cumulative_scale_.z);

                    for (size_t i = 0; i < node_gizmo_node_names_.size(); ++i) {
                        const glm::vec3& original_world_pos = node_original_world_positions_[i];
                        const glm::mat4& parent_inv = node_parent_world_inverses_[i];
                        const glm::mat3& original_rot = node_original_rotations_[i];
                        const glm::vec3& original_scale = node_original_scales_[i];

                        const glm::vec3 offset = original_world_pos - gizmo_pivot_;
                        const glm::vec3 new_world_pos = gizmo_pivot_ + offset * gizmo_cumulative_scale_;
                        const glm::vec3 new_local_pos = glm::vec3(parent_inv * glm::vec4(new_world_pos, 1.0f));

                        const glm::mat3 parent_rot_inv = extractRotation(parent_inv);
                        const glm::mat3 parent_rot = glm::transpose(parent_rot_inv);
                        const glm::mat3 local_scale = parent_rot_inv * world_scale * parent_rot;

                        const glm::mat3 original_rs(original_rot[0] * original_scale.x,
                                                    original_rot[1] * original_scale.y,
                                                    original_rot[2] * original_scale.z);
                        const glm::mat3 new_rs = local_scale * original_rs;

                        const glm::mat4 new_transform(glm::vec4(new_rs[0], 0.0f),
                                                      glm::vec4(new_rs[1], 0.0f),
                                                      glm::vec4(new_rs[2], 0.0f),
                                                      glm::vec4(new_local_pos, 1.0f));
                        scene_manager->setNodeTransform(node_gizmo_node_names_[i], new_transform);
                    }
                }
            } else {
                const glm::mat4 node_transform = scene_manager->getSelectedNodeTransform();
                const glm::vec3 new_gizmo_pos_world = glm::vec3(gizmo_matrix[3]);

                const auto& sm_scene = scene_manager->getScene();
                const auto* node = sm_scene.getNode(*selected_names.begin());
                const glm::mat4 parent_world_inv = (node && node->parent_id != core::NULL_NODE)
                                                       ? glm::inverse(sm_scene.getWorldTransform(node->parent_id))
                                                       : glm::mat4(1.0f);
                const glm::vec3 new_gizmo_pos = glm::vec3(parent_world_inv * glm::vec4(new_gizmo_pos_world, 1.0f));

                glm::mat4 new_transform;
                if (use_world_space) {
                    const glm::mat3 old_rs(node_transform);
                    const glm::mat3 delta_rs(delta_matrix);
                    const glm::mat3 parent_rot_inv = extractRotation(parent_world_inv);
                    const glm::mat3 parent_rot = glm::transpose(parent_rot_inv);
                    const glm::mat3 local_delta = parent_rot_inv * delta_rs * parent_rot;
                    const glm::mat3 new_rs = local_delta * old_rs;
                    new_transform = glm::mat4(new_rs);
                    new_transform[3] = glm::vec4(new_gizmo_pos - new_rs * local_pivot, 1.0f);
                } else {
                    const glm::mat3 new_rs(gizmo_matrix);
                    new_transform = gizmo_matrix;
                    new_transform[3] = glm::vec4(new_gizmo_pos - new_rs * local_pivot, 1.0f);
                }
                scene_manager->setSelectedNodeTransform(new_transform);
            }
        }

        if (!is_using && node_gizmo_active_) {
            node_gizmo_active_ = false;
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
            for (const auto& name : selected_names) {
                const auto* node = scene.getNode(name);
                if (!node || node->type != core::NodeType::SPLAT)
                    continue;

                const core::NodeId cropbox_id = scene.getCropBoxForSplat(node->id);
                if (cropbox_id != core::NULL_NODE) {
                    const auto* cropbox_node = scene.getNodeById(cropbox_id);
                    if (cropbox_node && cropbox_node->cropbox) {
                        const glm::mat4 cropbox_world = scene.getWorldTransform(cropbox_id);
                        render_manager->setCropboxGizmoState(true, cropbox_node->cropbox->min,
                                                             cropbox_node->cropbox->max, cropbox_world);
                    }
                }

                const core::NodeId ellipsoid_id = scene.getEllipsoidForSplat(node->id);
                if (ellipsoid_id != core::NULL_NODE) {
                    const auto* ellipsoid_node = scene.getNodeById(ellipsoid_id);
                    if (ellipsoid_node && ellipsoid_node->ellipsoid) {
                        const glm::mat4 ellipsoid_world = scene.getWorldTransform(ellipsoid_id);
                        render_manager->setEllipsoidGizmoState(true, ellipsoid_node->ellipsoid->radii,
                                                               ellipsoid_world);
                    }
                }
            }
        }

        overlay_drawlist->PopClipRect();
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

        auto& vp = ctx.viewer->getViewport();
        const glm::mat4 view = vp.getViewMatrix();
        const glm::ivec2 vp_size(static_cast<int>(viewport.size.x), static_cast<int>(viewport.size.y));
        const glm::mat4 projection = lfs::rendering::createProjectionMatrixFromFocal(
            vp_size, settings.focal_length_mm, settings.orthographic, settings.ortho_scale);

        const glm::vec3 cropbox_min = cropbox_node->cropbox->min;
        const glm::vec3 cropbox_max = cropbox_node->cropbox->max;
        const glm::mat4 world_transform = scene_manager->getScene().getWorldTransform(cropbox_id);

        const glm::vec3 local_size = cropbox_max - cropbox_min;
        const glm::vec3 world_scale = gizmo_ops::extractScale(world_transform);
        const glm::mat3 rotation = gizmo_ops::extractRotation(world_transform);
        const glm::vec3 translation = gizmo_ops::extractTranslation(world_transform);

        const bool use_world_space = (transform_space_ == TransformSpace::World);
        const ImGuizmo::OPERATION gizmo_op = current_operation_;

        const glm::vec3 local_pivot = gizmo_ops::computeLocalPivot(
            scene_manager->getScene(), cropbox_id,
            pivot_mode_, GizmoTargetType::CropBox);
        const glm::vec3 pivot_world = translation + rotation * (local_pivot * world_scale);

        const bool gizmo_local_aligned = (gizmo_op == ImGuizmo::SCALE) || !use_world_space;
        glm::mat4 gizmo_matrix;
        if (cropbox_gizmo_active_ && gizmo_context_.isActive()) {
            const auto& target = gizmo_context_.targets[0];
            const glm::vec3 original_size = target.bounds_max - target.bounds_min;
            const glm::vec3 current_size = original_size * gizmo_context_.cumulative_scale;
            const glm::mat3 current_rotation = gizmo_context_.cumulative_rotation * target.rotation;
            const glm::vec3 current_pivot = gizmo_context_.pivot_world + gizmo_context_.cumulative_translation;

            gizmo_matrix = gizmo_ops::computeGizmoMatrix(
                current_pivot, current_rotation, current_size * world_scale,
                gizmo_context_.use_world_space, gizmo_op == ImGuizmo::SCALE);
        } else {
            const glm::vec3 scaled_size = local_size * world_scale;
            gizmo_matrix = glm::translate(glm::mat4(1.0f), pivot_world);
            if (gizmo_local_aligned) {
                gizmo_matrix = gizmo_matrix * glm::mat4(rotation);
            }
            gizmo_matrix = glm::scale(gizmo_matrix, scaled_size);
        }

        ImGuizmo::SetOrthographic(settings.orthographic);
        ImGuizmo::SetRect(viewport.pos.x, viewport.pos.y, viewport.size.x, viewport.size.y);
        ImGuizmo::SetAxisLimit(GIZMO_AXIS_LIMIT);
        ImGuizmo::SetPlaneLimit(GIZMO_AXIS_LIMIT);

        static const float local_bounds[6] = {-0.5f, -0.5f, -0.5f, 0.5f, 0.5f, 0.5f};
        const bool use_bounds = (gizmo_op == ImGuizmo::SCALE);
        const ImGuizmo::OPERATION effective_op = use_bounds ? ImGuizmo::BOUNDS : gizmo_op;
        const float* bounds_ptr = use_bounds ? local_bounds : nullptr;

        {
            const bool is_using = ImGuizmo::IsUsing();
            if (!is_using) {
                cropbox_hovered_axis_ = ImGuizmo::IsOver(ImGuizmo::TRANSLATE_X) ||
                                        ImGuizmo::IsOver(ImGuizmo::TRANSLATE_Y) ||
                                        ImGuizmo::IsOver(ImGuizmo::TRANSLATE_Z) ||
                                        ImGuizmo::IsOver(ImGuizmo::BOUNDS);
                ImGuizmo::SetAxisMask(false, false, false);
            } else {
                ImGuizmo::SetAxisMask(cropbox_hovered_axis_, cropbox_hovered_axis_, cropbox_hovered_axis_);
            }
        }

        const bool modal_open = ImGui::IsPopupOpen("", ImGuiPopupFlags_AnyPopupId | ImGuiPopupFlags_AnyPopupLevel);
        ImDrawList* overlay_drawlist = modal_open ? ImGui::GetBackgroundDrawList() : ImGui::GetForegroundDrawList();
        const ImVec2 clip_min(viewport.pos.x, viewport.pos.y);
        const ImVec2 clip_max(clip_min.x + viewport.size.x, clip_min.y + viewport.size.y);
        overlay_drawlist->PushClipRect(clip_min, clip_max, true);
        ImGuizmo::SetDrawlist(overlay_drawlist);

        glm::mat4 delta_matrix;
        const ImGuizmo::MODE gizmo_mode = gizmo_local_aligned ? ImGuizmo::LOCAL : ImGuizmo::WORLD;

        const bool gizmo_changed = ImGuizmo::Manipulate(
            glm::value_ptr(view), glm::value_ptr(projection),
            effective_op, gizmo_mode, glm::value_ptr(gizmo_matrix),
            glm::value_ptr(delta_matrix), nullptr, bounds_ptr);

        const bool is_using = ImGuizmo::IsUsing();

        if (is_using && !cropbox_gizmo_active_) {
            cropbox_gizmo_active_ = true;
            cropbox_node_name_ = cropbox_node->name;

            gizmo_context_ = gizmo_ops::captureCropBox(
                scene_manager->getScene(),
                cropbox_node->name,
                pivot_world,
                local_pivot,
                transform_space_,
                pivot_mode_,
                gizmo_op);
        }

        if (gizmo_changed && gizmo_context_.isActive()) {
            auto& scene = scene_manager->getScene();

            if (gizmo_op == ImGuizmo::ROTATE) {
                const glm::mat3 delta_rot = gizmo_ops::extractRotation(delta_matrix);
                gizmo_ops::applyRotation(gizmo_context_, scene, delta_rot);
            } else if (gizmo_op == ImGuizmo::SCALE) {
                float mat_trans[3], mat_rot[3], mat_scale[3];
                ImGuizmo::DecomposeMatrixToComponents(glm::value_ptr(gizmo_matrix), mat_trans, mat_rot, mat_scale);
                const glm::vec3 new_size = glm::max(
                    glm::vec3(mat_scale[0], mat_scale[1], mat_scale[2]) / world_scale,
                    glm::vec3(MIN_GIZMO_SCALE));
                gizmo_ops::applyBoundsScale(gizmo_context_, scene, new_size);

                const glm::vec3 new_pivot_world(mat_trans[0], mat_trans[1], mat_trans[2]);
                gizmo_ops::applyTranslation(gizmo_context_, scene, new_pivot_world);
            } else {
                const glm::vec3 new_pivot_world(gizmo_matrix[3]);
                gizmo_ops::applyTranslation(gizmo_context_, scene, new_pivot_world);
            }

            render_manager->markDirty();
        }

        if (!is_using && cropbox_gizmo_active_) {
            cropbox_gizmo_active_ = false;
            gizmo_context_.reset();

            auto* node = scene_manager->getScene().getMutableNode(cropbox_node_name_);
            if (node && node->cropbox) {
                using namespace lfs::core::events;
                ui::CropBoxChanged{
                    .min_bounds = node->cropbox->min,
                    .max_bounds = node->cropbox->max,
                    .enabled = settings.use_crop_box}
                    .emit();
            }
        }

        if (cropbox_gizmo_active_) {
            render_manager->setCropboxGizmoState(
                true, cropbox_node->cropbox->min, cropbox_node->cropbox->max,
                scene_manager->getScene().getWorldTransform(cropbox_id));
        } else {
            render_manager->setCropboxGizmoActive(false);
        }

        overlay_drawlist->PopClipRect();
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

        auto& vp = ctx.viewer->getViewport();
        const glm::mat4 view = vp.getViewMatrix();
        const glm::ivec2 vp_size(static_cast<int>(viewport.size.x), static_cast<int>(viewport.size.y));
        const glm::mat4 projection = lfs::rendering::createProjectionMatrixFromFocal(
            vp_size, settings.focal_length_mm, settings.orthographic, settings.ortho_scale);

        const glm::vec3 radii = ellipsoid_node->ellipsoid->radii;
        const glm::mat4 world_transform = scene_manager->getScene().getWorldTransform(ellipsoid_id);

        const glm::vec3 world_scale = gizmo_ops::extractScale(world_transform);
        const glm::mat3 rotation = gizmo_ops::extractRotation(world_transform);
        const glm::vec3 translation = gizmo_ops::extractTranslation(world_transform);

        const bool use_world_space = (transform_space_ == TransformSpace::World);
        const ImGuizmo::OPERATION gizmo_op = current_operation_;

        const glm::vec3 local_pivot(0.0f);
        const glm::vec3 pivot_world = translation;

        const bool gizmo_local_aligned = (gizmo_op == ImGuizmo::SCALE) || !use_world_space;
        glm::mat4 gizmo_matrix;
        if (ellipsoid_gizmo_active_ && gizmo_context_.isActive()) {
            const auto& target = gizmo_context_.targets[0];
            const glm::vec3 current_radii = target.radii * gizmo_context_.cumulative_scale;
            const glm::mat3 current_rotation = gizmo_context_.cumulative_rotation * target.rotation;
            const glm::vec3 current_pivot = gizmo_context_.pivot_world + gizmo_context_.cumulative_translation;

            gizmo_matrix = gizmo_ops::computeGizmoMatrix(
                current_pivot, current_rotation, current_radii * world_scale,
                gizmo_context_.use_world_space, gizmo_op == ImGuizmo::SCALE);
        } else {
            const glm::vec3 scaled_radii = radii * world_scale;
            gizmo_matrix = glm::translate(glm::mat4(1.0f), pivot_world);
            if (gizmo_local_aligned) {
                gizmo_matrix = gizmo_matrix * glm::mat4(rotation);
            }
            gizmo_matrix = glm::scale(gizmo_matrix, scaled_radii);
        }

        ImGuizmo::SetOrthographic(settings.orthographic);
        ImGuizmo::SetRect(viewport.pos.x, viewport.pos.y, viewport.size.x, viewport.size.y);
        ImGuizmo::SetAxisLimit(GIZMO_AXIS_LIMIT);
        ImGuizmo::SetPlaneLimit(GIZMO_AXIS_LIMIT);

        static const float local_bounds[6] = {-1.0f, -1.0f, -1.0f, 1.0f, 1.0f, 1.0f};
        const bool use_bounds = (gizmo_op == ImGuizmo::SCALE);
        const ImGuizmo::OPERATION effective_op = use_bounds ? ImGuizmo::BOUNDS : gizmo_op;
        const float* bounds_ptr = use_bounds ? local_bounds : nullptr;

        {
            const bool is_using = ImGuizmo::IsUsing();
            if (!is_using) {
                ellipsoid_hovered_axis_ = ImGuizmo::IsOver(ImGuizmo::TRANSLATE_X) ||
                                          ImGuizmo::IsOver(ImGuizmo::TRANSLATE_Y) ||
                                          ImGuizmo::IsOver(ImGuizmo::TRANSLATE_Z) ||
                                          ImGuizmo::IsOver(ImGuizmo::BOUNDS);
                ImGuizmo::SetAxisMask(false, false, false);
            } else {
                ImGuizmo::SetAxisMask(ellipsoid_hovered_axis_, ellipsoid_hovered_axis_, ellipsoid_hovered_axis_);
            }
        }

        const bool modal_open = ImGui::IsPopupOpen("", ImGuiPopupFlags_AnyPopupId | ImGuiPopupFlags_AnyPopupLevel);
        ImDrawList* overlay_drawlist = modal_open ? ImGui::GetBackgroundDrawList() : ImGui::GetForegroundDrawList();
        const ImVec2 clip_min(viewport.pos.x, viewport.pos.y);
        const ImVec2 clip_max(clip_min.x + viewport.size.x, clip_min.y + viewport.size.y);
        overlay_drawlist->PushClipRect(clip_min, clip_max, true);
        ImGuizmo::SetDrawlist(overlay_drawlist);

        glm::mat4 delta_matrix;
        const ImGuizmo::MODE gizmo_mode = gizmo_local_aligned ? ImGuizmo::LOCAL : ImGuizmo::WORLD;

        const bool gizmo_changed = ImGuizmo::Manipulate(
            glm::value_ptr(view), glm::value_ptr(projection),
            effective_op, gizmo_mode, glm::value_ptr(gizmo_matrix),
            glm::value_ptr(delta_matrix), nullptr, bounds_ptr);

        const bool is_using = ImGuizmo::IsUsing();

        if (is_using && !ellipsoid_gizmo_active_) {
            ellipsoid_gizmo_active_ = true;
            ellipsoid_node_name_ = ellipsoid_node->name;

            gizmo_context_ = gizmo_ops::captureEllipsoid(
                scene_manager->getScene(),
                ellipsoid_node->name,
                pivot_world,
                local_pivot,
                transform_space_,
                pivot_mode_,
                gizmo_op);
        }

        if (gizmo_changed && gizmo_context_.isActive()) {
            auto& scene = scene_manager->getScene();

            if (gizmo_op == ImGuizmo::ROTATE) {
                const glm::mat3 delta_rot = gizmo_ops::extractRotation(delta_matrix);
                gizmo_ops::applyRotation(gizmo_context_, scene, delta_rot);
            } else if (gizmo_op == ImGuizmo::SCALE) {
                float mat_trans[3], mat_rot[3], mat_scale[3];
                ImGuizmo::DecomposeMatrixToComponents(glm::value_ptr(gizmo_matrix), mat_trans, mat_rot, mat_scale);
                const glm::vec3 new_radii = glm::max(
                    glm::vec3(mat_scale[0], mat_scale[1], mat_scale[2]) / world_scale,
                    glm::vec3(MIN_GIZMO_SCALE));
                gizmo_ops::applyBoundsScale(gizmo_context_, scene, new_radii);

                const glm::vec3 new_pivot_world(mat_trans[0], mat_trans[1], mat_trans[2]);
                gizmo_ops::applyTranslation(gizmo_context_, scene, new_pivot_world);
            } else {
                const glm::vec3 new_pivot_world(gizmo_matrix[3]);
                gizmo_ops::applyTranslation(gizmo_context_, scene, new_pivot_world);
            }

            render_manager->markDirty();
        }

        if (!is_using && ellipsoid_gizmo_active_) {
            ellipsoid_gizmo_active_ = false;
            gizmo_context_.reset();

            auto* node = scene_manager->getScene().getMutableNode(ellipsoid_node_name_);
            if (node && node->ellipsoid) {
                using namespace lfs::core::events;
                ui::EllipsoidChanged{
                    .radii = node->ellipsoid->radii,
                    .enabled = settings.use_ellipsoid}
                    .emit();
            }
        }

        if (ellipsoid_gizmo_active_) {
            const glm::mat4 current_world_transform = scene_manager->getScene().getWorldTransform(ellipsoid_id);
            render_manager->setEllipsoidGizmoState(true, ellipsoid_node->ellipsoid->radii,
                                                   current_world_transform);
        } else {
            render_manager->setEllipsoidGizmoActive(false);
        }

        overlay_drawlist->PopClipRect();
    }

    void GizmoManager::renderCropGizmoMiniToolbar(const UIContext&) {
        const auto vp_pos = viewer_->getGuiManager()->getViewportPos();
        const auto vp_size = viewer_->getGuiManager()->getViewportSize();

        constexpr float MARGIN_X = 10.0f;
        constexpr float MARGIN_BOTTOM = 100.0f;
        constexpr int BUTTON_COUNT = 3;
        constexpr ImGuiWindowFlags WINDOW_FLAGS =
            ImGuiWindowFlags_NoTitleBar | ImGuiWindowFlags_NoResize | ImGuiWindowFlags_NoMove |
            ImGuiWindowFlags_NoScrollbar | ImGuiWindowFlags_NoCollapse | ImGuiWindowFlags_NoSavedSettings;

        const auto& t = theme();
        const float scale = lfs::python::get_shared_dpi_scale();
        const float btn_size = t.sizes.toolbar_button_size * scale;
        const float spacing = t.sizes.toolbar_spacing * scale;
        const float padding = t.sizes.toolbar_padding * scale;
        const float toolbar_width = BUTTON_COUNT * btn_size + (BUTTON_COUNT - 1) * spacing + 2.0f * padding;
        const float toolbar_height = btn_size + 2.0f * padding;
        const float toolbar_x = vp_pos.x + MARGIN_X * scale;
        const float toolbar_y = vp_pos.y + vp_size.y - MARGIN_BOTTOM * scale;

        widgets::DrawWindowShadow({toolbar_x, toolbar_y}, {toolbar_width, toolbar_height}, t.sizes.window_rounding);
        ImGui::SetNextWindowPos({toolbar_x, toolbar_y});
        ImGui::SetNextWindowSize({toolbar_width, toolbar_height});

        ImGui::PushStyleVar(ImGuiStyleVar_WindowRounding, t.sizes.window_rounding);
        ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, {padding, padding});
        ImGui::PushStyleVar(ImGuiStyleVar_ItemSpacing, {spacing, 0.0f});
        ImGui::PushStyleVar(ImGuiStyleVar_FramePadding, {0.0f, 0.0f});
        ImGui::PushStyleColor(ImGuiCol_WindowBg, t.subtoolbar_background());

        if (ImGui::Begin("##CropGizmoMiniToolbar", nullptr, WINDOW_FLAGS)) {
            const ImVec2 btn_sz(btn_size, btn_size);

            const auto button = [&](const char* id, const ImGuizmo::OPERATION op, const char* label, const char* tip) {
                if (widgets::IconButton(id, 0, btn_sz, current_operation_ == op, label)) {
                    current_operation_ = op;
                }
                if (ImGui::IsItemHovered()) {
                    widgets::SetThemedTooltip("%s", tip);
                }
            };

            button("##mini_t", ImGuizmo::TRANSLATE, "T", "Translate (T)");
            ImGui::SameLine();
            button("##mini_r", ImGuizmo::ROTATE, "R", "Rotate (R)");
            ImGui::SameLine();
            button("##mini_s", ImGuizmo::SCALE, "S", "Scale (S)");
        }
        ImGui::End();

        ImGui::PopStyleColor();
        ImGui::PopStyleVar(4);
    }

    void GizmoManager::renderViewportGizmo(const ViewportLayout& viewport) {
        if (!show_viewport_gizmo_ || viewport.size.x <= 0 || viewport.size.y <= 0)
            return;

        auto* rendering_manager = viewer_->getRenderingManager();
        if (!rendering_manager)
            return;

        auto* const engine = rendering_manager->getRenderingEngine();
        if (!engine)
            return;

        auto& vp = viewer_->getViewport();
        const glm::vec2 vp_pos(viewport.pos.x, viewport.pos.y);
        const glm::vec2 vp_size(viewport.size.x, viewport.size.y);

        const float gizmo_x = vp_pos.x + vp_size.x - VIEWPORT_GIZMO_SIZE - VIEWPORT_GIZMO_MARGIN_X;
        const float gizmo_y = vp_pos.y + VIEWPORT_GIZMO_MARGIN_Y;

        const ImVec2 mouse = ImGui::GetMousePos();
        const bool mouse_in_gizmo = mouse.x >= gizmo_x && mouse.x <= gizmo_x + VIEWPORT_GIZMO_SIZE &&
                                    mouse.y >= gizmo_y && mouse.y <= gizmo_y + VIEWPORT_GIZMO_SIZE;

        const int hovered_axis = engine->hitTestViewportGizmo(glm::vec2(mouse.x, mouse.y), vp_pos, vp_size);
        engine->setViewportGizmoHover(hovered_axis);

        if (!ImGui::GetIO().WantCaptureMouse) {
            const glm::vec2 capture_mouse_pos(mouse.x, mouse.y);
            const float time = static_cast<float>(ImGui::GetTime());

            if (ImGui::IsMouseClicked(ImGuiMouseButton_Left) && mouse_in_gizmo) {
                if (hovered_axis >= 0 && hovered_axis <= 5) {
                    const int axis = hovered_axis % 3;
                    const bool negative = hovered_axis >= 3;
                    const glm::mat3 rotation = engine->getAxisViewRotation(axis, negative);
                    const float dist = glm::length(vp.camera.pivot - vp.camera.t);

                    vp.camera.pivot = glm::vec3(0.0f);
                    vp.camera.R = rotation;
                    vp.camera.t = -rotation[2] * dist;

                    const auto& settings = rendering_manager->getSettings();
                    lfs::core::events::ui::GridSettingsChanged{
                        .enabled = settings.show_grid,
                        .plane = axis,
                        .opacity = settings.grid_opacity}
                        .emit();

                    rendering_manager->markDirty();
                } else {
                    viewport_gizmo_dragging_ = true;
                    vp.camera.startRotateAroundCenter(capture_mouse_pos, time);
                    if (GLFWwindow* const window = glfwGetCurrentContext()) {
                        glfwGetCursorPos(window, &gizmo_drag_start_cursor_.x, &gizmo_drag_start_cursor_.y);
                        glfwSetInputMode(window, GLFW_CURSOR, GLFW_CURSOR_DISABLED);
                    }
                }
            }

            if (viewport_gizmo_dragging_) {
                if (ImGui::IsMouseDown(ImGuiMouseButton_Left)) {
                    vp.camera.updateRotateAroundCenter(capture_mouse_pos, time);
                    rendering_manager->markDirty();
                } else {
                    vp.camera.endRotateAroundCenter();
                    viewport_gizmo_dragging_ = false;

                    if (GLFWwindow* const window = glfwGetCurrentContext()) {
                        glfwSetInputMode(window, GLFW_CURSOR, GLFW_CURSOR_NORMAL);
                        glfwSetCursorPos(window, gizmo_drag_start_cursor_.x, gizmo_drag_start_cursor_.y);
                    }
                }
            }
        }

        if (auto result = engine->renderViewportGizmo(vp.getRotationMatrix(), vp_pos, vp_size); !result) {
            LOG_WARN("Failed to render viewport gizmo: {}", result.error());
        }

        if (viewport_gizmo_dragging_) {
            const float center_x = gizmo_x + VIEWPORT_GIZMO_SIZE * 0.5f;
            const float center_y = gizmo_y + VIEWPORT_GIZMO_SIZE * 0.5f;
            constexpr float OVERLAY_RADIUS = VIEWPORT_GIZMO_SIZE * 0.46f;
            ImGui::GetBackgroundDrawList()->AddCircleFilled(
                ImVec2(center_x, center_y), OVERLAY_RADIUS,
                toU32WithAlpha(theme().overlay.text_dim, 0.2f), 32);
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
        rm->markDirty();
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
        current_operation_ = ImGuizmo::TRANSLATE;
    }

    void GizmoManager::setSelectionSubMode(SelectionSubMode mode) {
        selection_mode_ = mode;

        if (auto* rm = viewer_->getRenderingManager()) {
            lfs::rendering::SelectionMode rm_mode = lfs::rendering::SelectionMode::Centers;
            switch (mode) {
            case SelectionSubMode::Centers: rm_mode = lfs::rendering::SelectionMode::Centers; break;
            case SelectionSubMode::Rectangle: rm_mode = lfs::rendering::SelectionMode::Rectangle; break;
            case SelectionSubMode::Polygon: rm_mode = lfs::rendering::SelectionMode::Polygon; break;
            case SelectionSubMode::Lasso: rm_mode = lfs::rendering::SelectionMode::Lasso; break;
            case SelectionSubMode::Rings: rm_mode = lfs::rendering::SelectionMode::Rings; break;
            }
            rm->setSelectionMode(rm_mode);
        }
    }

    bool GizmoManager::isPositionInViewportGizmo(const double x, const double y) const {
        if (!show_viewport_gizmo_)
            return false;

        const auto vp_pos = viewer_->getGuiManager()->getViewportPos();
        const auto vp_size = viewer_->getGuiManager()->getViewportSize();

        const float gizmo_x = vp_pos.x + vp_size.x - VIEWPORT_GIZMO_SIZE - VIEWPORT_GIZMO_MARGIN_X;
        const float gizmo_y = vp_pos.y + VIEWPORT_GIZMO_MARGIN_Y;

        return x >= gizmo_x && x <= gizmo_x + VIEWPORT_GIZMO_SIZE &&
               y >= gizmo_y && y <= gizmo_y + VIEWPORT_GIZMO_SIZE;
    }

    ToolType GizmoManager::getCurrentToolMode() const {
        return viewer_->getEditorContext().getActiveTool();
    }

} // namespace lfs::vis::gui

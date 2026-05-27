/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "core/editor_context.hpp"
#include "scene/scene_manager.hpp"
#include "visualizer/app_store.hpp"
#include "training/training_manager.hpp"
#include "visualizer/gui_capabilities.hpp"

#include <string_view>

namespace lfs::vis {

    namespace {
        [[nodiscard]] std::string formatTransformSelectionError(const bool has_editable,
                                                                const bool found_locked,
                                                                const bool found_untransformable) {
            if (found_locked && found_untransformable)
                return "selection contains locked or unsupported nodes";
            if (found_locked)
                return has_editable ? "selection contains locked nodes" : "selection is locked";
            if (found_untransformable)
                return has_editable ? "selection contains unsupported nodes" : "select parent node";
            return "No transform targets provided";
        }

        [[nodiscard]] std::string_view activeToolId(const ToolType tool) noexcept {
            switch (tool) {
            case ToolType::Selection: return "builtin.select";
            case ToolType::Translate: return "builtin.translate";
            case ToolType::Rotate: return "builtin.rotate";
            case ToolType::Scale: return "builtin.scale";
            case ToolType::Mirror: return "builtin.mirror";
            case ToolType::Brush: return "builtin.brush";
            case ToolType::Align: return "builtin.align";
            case ToolType::None:
            default: return {};
            }
        }
    } // namespace

    void EditorContext::update(const SceneManager* scene_manager, const TrainerManager* trainer_manager) {
        if (!scene_manager) {
            mode_ = EditorMode::EMPTY;
            has_selection_ = false;
            has_gaussians_ = false;
            has_editable_transform_selection_ = false;
            has_splat_selection_ = false;
            has_editable_splat_selection_ = false;
            transform_selection_error_.clear();
            selected_node_type_ = core::NodeType::SPLAT;
            return;
        }

        // Determine mode based on training state
        if (trainer_manager) {
            if (trainer_manager->isRunning()) {
                mode_ = EditorMode::TRAINING;
            } else if (trainer_manager->isPaused()) {
                mode_ = EditorMode::PAUSED;
            } else if (trainer_manager->isFinished()) {
                mode_ = EditorMode::FINISHED;
            } else if (scene_manager->hasDataset()) {
                const auto* model = scene_manager->getScene().getTrainingModel();
                mode_ = model ? EditorMode::VIEWING_SPLATS : EditorMode::PRE_TRAINING;
            } else if (scene_manager->getScene().getNodeCount() > 0) {
                mode_ = EditorMode::VIEWING_SPLATS;
            } else {
                mode_ = EditorMode::EMPTY;
            }
        } else {
            if (scene_manager->hasDataset()) {
                const auto* model = scene_manager->getScene().getTrainingModel();
                mode_ = model ? EditorMode::VIEWING_SPLATS : EditorMode::PRE_TRAINING;
            } else if (scene_manager->getScene().getNodeCount() > 0) {
                mode_ = EditorMode::VIEWING_SPLATS;
            } else {
                mode_ = EditorMode::EMPTY;
            }
        }

        // Update selection state
        const auto selected_node_names = scene_manager->getSelectedNodeNames();
        has_selection_ = !selected_node_names.empty();
        has_editable_transform_selection_ = false;
        has_splat_selection_ = false;
        has_editable_splat_selection_ = false;
        transform_selection_error_.clear();
        selected_node_type_ = core::NodeType::SPLAT;

        if (has_selection_) {
            const auto& scene = scene_manager->getScene();
            bool found_locked = false;
            bool found_untransformable = false;
            bool has_editable_transform_target = false;
            bool selected_type_initialized = false;

            for (const auto& name : selected_node_names) {
                const auto* const node = scene.getNode(name);
                if (!node)
                    continue;

                if (!selected_type_initialized) {
                    selected_node_type_ = node->type;
                    selected_type_initialized = true;
                }

                const bool locked = static_cast<bool>(node->locked);
                const bool transformable = cap::isTransformableNodeType(node->type);
                if (!transformable) {
                    found_untransformable = true;
                } else if (locked) {
                    found_locked = true;
                } else {
                    has_editable_transform_target = true;
                }

                if (node->type == core::NodeType::SPLAT) {
                    has_splat_selection_ = true;
                    if (!locked)
                        has_editable_splat_selection_ = true;
                }
            }

            has_editable_transform_selection_ =
                has_editable_transform_target && !found_locked && !found_untransformable;
            if (!has_editable_transform_selection_) {
                transform_selection_error_ =
                    formatTransformSelectionError(has_editable_transform_target,
                                                  found_locked,
                                                  found_untransformable);
            }
        }

        has_gaussians_ = (mode_ == EditorMode::VIEWING_SPLATS ||
                          mode_ == EditorMode::TRAINING ||
                          mode_ == EditorMode::PAUSED ||
                          mode_ == EditorMode::FINISHED);
    }

    bool EditorContext::canTransformSelectedNode() const {
        return has_selection_ && !isToolsDisabled() && has_editable_transform_selection_;
    }

    bool EditorContext::canSelectGaussians() const {
        return has_gaussians_ && !isToolsDisabled();
    }

    bool EditorContext::isToolAvailable(const ToolType tool) const {
        if (isToolsDisabled())
            return false;
        if (!has_selection_ && tool != ToolType::None)
            return false;

        switch (tool) {
        case ToolType::None:
            return true;
        case ToolType::Selection:
        case ToolType::Brush:
            return has_gaussians_;
        case ToolType::Mirror:
            return has_gaussians_ && has_editable_splat_selection_;
        case ToolType::Translate:
        case ToolType::Rotate:
        case ToolType::Scale:
            return canTransformSelectedNode();
        case ToolType::Align:
            return has_editable_splat_selection_;
        }
        return false;
    }

    const char* EditorContext::getToolUnavailableReason(const ToolType tool) const {
        if (isToolsDisabled())
            return "switch to edit mode first";
        if (!has_selection_ && tool != ToolType::None)
            return "no node selected";

        switch (tool) {
        case ToolType::None:
            return nullptr;
        case ToolType::Selection:
            return has_gaussians_ ? nullptr : "no gaussians";
        case ToolType::Brush:
            return has_gaussians_ ? nullptr : "no gaussians";
        case ToolType::Mirror:
            if (!has_gaussians_)
                return "no gaussians";
            if (has_editable_splat_selection_)
                return nullptr;
            if (has_splat_selection_)
                return "selection is locked";
            return "select PLY node";
        case ToolType::Translate:
        case ToolType::Rotate:
        case ToolType::Scale:
            if (canTransformSelectedNode())
                return nullptr;
            return transform_selection_error_.empty() ? "select parent node" : transform_selection_error_.c_str();
        case ToolType::Align:
            if (has_editable_splat_selection_)
                return nullptr;
            if (has_splat_selection_)
                return "selection is locked";
            return "select PLY node";
        }
        return nullptr;
    }

    void EditorContext::setActiveTool(const ToolType tool) {
        if (isToolAvailable(tool)) {
            active_tool_ = tool;
            clearActiveOperator();
        }
    }

    void EditorContext::validateActiveTool() {
        if (!isToolAvailable(active_tool_)) {
            active_tool_ = ToolType::None;
            app_store().active_tool.set(std::string{});
        }
    }

    void EditorContext::setActiveOperator(const std::string& id, const std::string& gizmo_type) {
        LOG_DEBUG("EditorContext::setActiveOperator: id='{}', gizmo_type='{}'", id, gizmo_type);
        active_operator_id_ = id;
        gizmo_type_ = gizmo_type;
        app_store().active_tool.set(active_operator_id_);
        LOG_DEBUG("EditorContext::setActiveOperator: active_operator_id_='{}', hasActive={}",
                  active_operator_id_, !active_operator_id_.empty());
    }

    void EditorContext::clearActiveOperator() {
        LOG_DEBUG("EditorContext::clearActiveOperator: was '{}'", active_operator_id_);
        active_operator_id_.clear();
        gizmo_type_.clear();
        app_store().active_tool.set(std::string(activeToolId(active_tool_)));
    }

    void EditorContext::setGizmoType(const std::string& type) { gizmo_type_ = type; }

    void EditorContext::clearGizmo() { gizmo_type_.clear(); }

} // namespace lfs::vis

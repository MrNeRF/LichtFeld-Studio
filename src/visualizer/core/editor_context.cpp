/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "core/editor_context.hpp"
#include "scene/scene_manager.hpp"
#include "training/training_manager.hpp"
#include "visualizer/gui_capabilities.hpp"

namespace lfs::vis {

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
        has_selection_ = scene_manager->hasSelectedNode();
        selected_node_type_ = has_selection_ ? scene_manager->getSelectedNodeType() : core::NodeType::SPLAT;
        has_editable_transform_selection_ = false;
        has_splat_selection_ = false;
        has_editable_splat_selection_ = false;
        transform_selection_error_.clear();

        if (has_selection_) {
            const auto transform_selection = cap::resolveEditableTransformSelection(
                *scene_manager, std::nullopt, cap::TransformTargetPolicy::RequireAllEditable);
            has_editable_transform_selection_ = transform_selection.has_value();
            if (!transform_selection)
                transform_selection_error_ = transform_selection.error();

            const auto& scene = scene_manager->getScene();
            for (const auto& name : scene_manager->getSelectedNodeNames()) {
                const auto* const node = scene.getNode(name);
                if (!node)
                    continue;

                const bool locked = static_cast<bool>(node->locked);

                if (node->type == core::NodeType::SPLAT) {
                    has_splat_selection_ = true;
                    if (!locked)
                        has_editable_splat_selection_ = true;
                }
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
        }
    }

    void EditorContext::setActiveOperator(const std::string& id, const std::string& gizmo_type) {
        LOG_DEBUG("EditorContext::setActiveOperator: id='{}', gizmo_type='{}'", id, gizmo_type);
        active_operator_id_ = id;
        gizmo_type_ = gizmo_type;
        LOG_DEBUG("EditorContext::setActiveOperator: active_operator_id_='{}', hasActive={}",
                  active_operator_id_, !active_operator_id_.empty());
    }

    void EditorContext::clearActiveOperator() {
        LOG_DEBUG("EditorContext::clearActiveOperator: was '{}'", active_operator_id_);
        active_operator_id_.clear();
        gizmo_type_.clear();
    }

    void EditorContext::setGizmoType(const std::string& type) { gizmo_type_ = type; }

    void EditorContext::clearGizmo() { gizmo_type_.clear(); }

} // namespace lfs::vis

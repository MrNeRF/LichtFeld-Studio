/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "core/editor_context.hpp"
#include "scene/scene_manager.hpp"
#include "training/training_manager.hpp"

namespace lfs::vis {

    void EditorContext::update(const SceneManager* scene_manager, const TrainerManager* trainer_manager) {
        if (!scene_manager) {
            mode_ = EditorMode::EMPTY;
            has_selection_ = false;
            has_gaussians_ = false;
            selected_node_type_ = NodeType::SPLAT;
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
        selected_node_type_ = has_selection_ ? scene_manager->getSelectedNodeType() : NodeType::SPLAT;

        has_gaussians_ = (mode_ == EditorMode::VIEWING_SPLATS ||
                          mode_ == EditorMode::TRAINING ||
                          mode_ == EditorMode::PAUSED ||
                          mode_ == EditorMode::FINISHED);
    }

    bool EditorContext::isTransformableNodeType(const NodeType type) {
        // Only SPLAT (PLY files) and DATASET root can be transformed
        return type == NodeType::DATASET || type == NodeType::SPLAT;
    }

    bool EditorContext::canTransformSelectedNode() const {
        return has_selection_ && !isToolsDisabled() && isTransformableNodeType(selected_node_type_);
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
        case ToolType::Mirror:
            return has_gaussians_;
        case ToolType::Translate:
        case ToolType::Rotate:
        case ToolType::Scale:
            return canTransformSelectedNode();
        case ToolType::Align:
            return selected_node_type_ == NodeType::SPLAT;
        case ToolType::CropBox:
            return has_selection_;
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
        case ToolType::Brush:
        case ToolType::Mirror:
            return has_gaussians_ ? nullptr : "no gaussians";
        case ToolType::Translate:
        case ToolType::Rotate:
        case ToolType::Scale:
            return isTransformableNodeType(selected_node_type_) ? nullptr : "select parent node";
        case ToolType::Align:
            return selected_node_type_ == NodeType::SPLAT ? nullptr : "select PLY node";
        case ToolType::CropBox:
            return nullptr;
        }
        return nullptr;
    }

    void EditorContext::setActiveTool(const ToolType tool) {
        if (isToolAvailable(tool)) {
            active_tool_ = tool;
        }
    }

    void EditorContext::validateActiveTool() {
        if (!isToolAvailable(active_tool_)) {
            active_tool_ = ToolType::None;
        }
    }

} // namespace lfs::vis

/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include "scene/scene.hpp"
#include <cstdint>

namespace lfs::vis {

    class SceneManager;
    class TrainerManager;

    // Application editing mode
    enum class EditorMode : uint8_t {
        EMPTY,
        VIEWING_SPLATS,
        PRE_TRAINING,
        TRAINING,
        PAUSED,
        FINISHED
    };

    // Tool types for toolbar
    enum class ToolType : uint8_t {
        None,
        Selection,
        Translate,
        Rotate,
        Scale,
        Mirror,
        Brush,
        Align,
        CropBox
    };

    // Centralized editor state - single source of truth for tool availability
    class EditorContext {
    public:
        EditorContext() = default;

        // Update state from managers (call once per frame)
        void update(const SceneManager* scene_manager, const TrainerManager* trainer_manager);

        // Mode queries
        [[nodiscard]] EditorMode getMode() const { return mode_; }
        [[nodiscard]] bool isPreTraining() const { return mode_ == EditorMode::PRE_TRAINING; }
        [[nodiscard]] bool isTraining() const { return mode_ == EditorMode::TRAINING; }
        [[nodiscard]] bool isTrainingOrPaused() const {
            return mode_ == EditorMode::TRAINING || mode_ == EditorMode::PAUSED;
        }
        [[nodiscard]] bool isFinished() const { return mode_ == EditorMode::FINISHED; }
        [[nodiscard]] bool isToolsDisabled() const {
            return mode_ == EditorMode::TRAINING || mode_ == EditorMode::PAUSED || mode_ == EditorMode::FINISHED;
        }
        [[nodiscard]] bool isEmpty() const { return mode_ == EditorMode::EMPTY; }

        // Selection queries
        [[nodiscard]] bool hasSelection() const { return has_selection_; }
        [[nodiscard]] NodeType getSelectedNodeType() const { return selected_node_type_; }

        // Tool availability
        [[nodiscard]] bool isToolAvailable(ToolType tool) const;
        [[nodiscard]] const char* getToolUnavailableReason(ToolType tool) const;

        // Capability queries
        [[nodiscard]] bool canTransformSelectedNode() const;
        [[nodiscard]] bool canSelectGaussians() const;
        [[nodiscard]] bool hasGaussians() const { return has_gaussians_; }
        [[nodiscard]] bool forcePointCloudMode() const { return mode_ == EditorMode::PRE_TRAINING; }

        // Active tool management
        void setActiveTool(ToolType tool);
        [[nodiscard]] ToolType getActiveTool() const { return active_tool_; }
        void validateActiveTool();

    private:
        EditorMode mode_ = EditorMode::EMPTY;
        NodeType selected_node_type_ = NodeType::SPLAT;
        ToolType active_tool_ = ToolType::None;
        bool has_selection_ = false;
        bool has_gaussians_ = false;

        [[nodiscard]] static bool isTransformableNodeType(NodeType type);
    };

} // namespace lfs::vis

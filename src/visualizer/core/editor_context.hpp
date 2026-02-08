/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include "core/export.hpp"
#include "core/modal_event.hpp"
#include "core/operator_callbacks.hpp"
#include "core/scene.hpp"
#include <cstdint>
#include <string>

namespace lfs::vis {

    using ModalEvent = lfs::core::ModalEvent;

    // Use callback types from core
    using CancelOperatorCallback = lfs::core::CancelOperatorCallback;
    using InvokeOperatorCallback = lfs::core::InvokeOperatorCallback;
    using ModalEventCallback = lfs::core::ModalEventCallback;

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
        Align
    };

    // Selection sub-modes (for selection tool)
    enum class SelectionSubMode : uint8_t {
        Centers,
        Rectangle,
        Polygon,
        Lasso,
        Rings
    };

    // Transform coordinate space
    enum class TransformSpace : uint8_t {
        Local,
        World
    };

    // Pivot point for transforms
    enum class PivotMode : uint8_t {
        Origin,      // (0,0,0) in local space
        BoundsCenter // Bounding box center
    };

    // Centralized editor state - single source of truth for tool availability
    class LFS_VIS_API EditorContext : public lfs::core::IOperatorCallbacks {
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
        [[nodiscard]] core::NodeType getSelectedNodeType() const { return selected_node_type_; }

        // Tool availability
        [[nodiscard]] bool isToolAvailable(ToolType tool) const;
        [[nodiscard]] const char* getToolUnavailableReason(ToolType tool) const;

        // Capability queries
        [[nodiscard]] bool canTransformSelectedNode() const;
        [[nodiscard]] bool canSelectGaussians() const;
        [[nodiscard]] bool hasGaussians() const { return has_gaussians_; }
        [[nodiscard]] bool forcePointCloudMode() const { return mode_ == EditorMode::PRE_TRAINING; }

        // Active tool management (legacy - will be removed)
        void setActiveTool(ToolType tool);
        [[nodiscard]] ToolType getActiveTool() const { return active_tool_; }
        void validateActiveTool();

        // String-based operator system (Blender-style)
        void setActiveOperator(const std::string& id, const std::string& gizmo_type);
        void clearActiveOperator();
        [[nodiscard]] const std::string& getActiveOperator() const { return active_operator_id_; }
        [[nodiscard]] const std::string& getGizmoType() const { return gizmo_type_; }
        [[nodiscard]] bool hasActiveOperator() const { return !active_operator_id_.empty(); }

        void setGizmoType(const std::string& type);
        void clearGizmo();

        // Python operator callbacks (stored here to avoid static var issues across shared libs)
        // Implements IOperatorCallbacks interface
        void setCancelOperatorCallback(CancelOperatorCallback cb) override { cancel_operator_cb_ = std::move(cb); }
        void setInvokeOperatorCallback(InvokeOperatorCallback cb) override { invoke_operator_cb_ = std::move(cb); }
        void setModalEventCallback(ModalEventCallback cb) override { modal_event_cb_ = std::move(cb); }

        [[nodiscard]] bool hasCancelOperatorCallback() const override { return cancel_operator_cb_ != nullptr; }
        [[nodiscard]] bool hasInvokeOperatorCallback() const override { return invoke_operator_cb_ != nullptr; }
        [[nodiscard]] bool hasModalEventCallback() const override { return modal_event_cb_ != nullptr; }

        void cancelOperator() const override {
            if (cancel_operator_cb_)
                cancel_operator_cb_();
        }
        bool invokeOperator(const char* id) const override { return invoke_operator_cb_ ? invoke_operator_cb_(id) : false; }
        bool dispatchModalEvent(const ModalEvent& evt) const override { return modal_event_cb_ ? modal_event_cb_(evt) : false; }

    private:
        EditorMode mode_ = EditorMode::EMPTY;
        core::NodeType selected_node_type_ = core::NodeType::SPLAT;
        ToolType active_tool_ = ToolType::None;
        bool has_selection_ = false;
        bool has_gaussians_ = false;

        // String-based operator system
        std::string active_operator_id_;
        std::string gizmo_type_;

        // Python operator callbacks
        CancelOperatorCallback cancel_operator_cb_;
        InvokeOperatorCallback invoke_operator_cb_;
        ModalEventCallback modal_event_cb_;

        [[nodiscard]] static bool isTransformableNodeType(core::NodeType type);
    };

} // namespace lfs::vis

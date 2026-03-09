/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "edit_ops.hpp"
#include "core/services.hpp"
#include "operation/undo_history.hpp"
#include "operator/operator_registry.hpp"
#include "rendering/dirty_flags.hpp"
#include "rendering/rendering_manager.hpp"
#include "scene/scene_manager.hpp"

namespace lfs::vis::op {

    const OperatorDescriptor UndoOperator::DESCRIPTOR = {
        .builtin_id = BuiltinOp::Undo,
        .python_class_id = {},
        .label = "Undo",
        .description = "Undo the last action",
        .icon = "undo",
        .shortcut = "Ctrl+Z",
        .flags = OperatorFlags::REGISTER,
        .source = OperatorSource::CPP,
        .poll_deps = PollDependency::NONE,
    };

    bool UndoOperator::poll(const OperatorContext& /*ctx*/) const {
        return undoHistory().canUndo();
    }

    OperatorResult UndoOperator::invoke(OperatorContext& /*ctx*/, OperatorProperties& /*props*/) {
        undoHistory().undo();
        if (auto* rm = services().renderingOrNull()) {
            rm->markDirty(DirtyFlag::ALL);
        }
        return OperatorResult::FINISHED;
    }

    const OperatorDescriptor RedoOperator::DESCRIPTOR = {
        .builtin_id = BuiltinOp::Redo,
        .python_class_id = {},
        .label = "Redo",
        .description = "Redo the last undone action",
        .icon = "redo",
        .shortcut = "Ctrl+Shift+Z",
        .flags = OperatorFlags::REGISTER,
        .source = OperatorSource::CPP,
        .poll_deps = PollDependency::NONE,
    };

    bool RedoOperator::poll(const OperatorContext& /*ctx*/) const {
        return undoHistory().canRedo();
    }

    OperatorResult RedoOperator::invoke(OperatorContext& /*ctx*/, OperatorProperties& /*props*/) {
        undoHistory().redo();
        if (auto* rm = services().renderingOrNull()) {
            rm->markDirty(DirtyFlag::ALL);
        }
        return OperatorResult::FINISHED;
    }

    const OperatorDescriptor DeleteOperator::DESCRIPTOR = {
        .builtin_id = BuiltinOp::Delete,
        .python_class_id = {},
        .label = "Delete",
        .description = "Delete selected nodes",
        .icon = "delete",
        .shortcut = "Delete",
        .flags = OperatorFlags::REGISTER | OperatorFlags::UNDO,
        .source = OperatorSource::CPP,
        .poll_deps = PollDependency::SELECTION,
    };

    bool DeleteOperator::poll(const OperatorContext& ctx) const {
        return ctx.hasSelection();
    }

    OperatorResult DeleteOperator::invoke(OperatorContext& ctx, OperatorProperties& /*props*/) {
        const auto nodes = ctx.selectedNodes();
        if (nodes.empty()) {
            return OperatorResult::CANCELLED;
        }

        for (const auto& name : nodes) {
            ctx.scene().removePLY(name, false);
        }

        return OperatorResult::FINISHED;
    }

    void registerEditOperators() {
        operators().registerOperator(BuiltinOp::Undo, UndoOperator::DESCRIPTOR,
                                     [] { return std::make_unique<UndoOperator>(); });
        operators().registerOperator(BuiltinOp::Redo, RedoOperator::DESCRIPTOR,
                                     [] { return std::make_unique<RedoOperator>(); });
        operators().registerOperator(BuiltinOp::Delete, DeleteOperator::DESCRIPTOR,
                                     [] { return std::make_unique<DeleteOperator>(); });
    }

    void unregisterEditOperators() {
        operators().unregisterOperator(BuiltinOp::Undo);
        operators().unregisterOperator(BuiltinOp::Redo);
        operators().unregisterOperator(BuiltinOp::Delete);
    }

} // namespace lfs::vis::op

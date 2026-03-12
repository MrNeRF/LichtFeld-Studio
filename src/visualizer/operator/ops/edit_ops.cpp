/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "edit_ops.hpp"
#include "core/services.hpp"
#include "operation/undo_history.hpp"
#include "operator/operator_registry.hpp"
#include "operator/property_schema.hpp"
#include "rendering/dirty_flags.hpp"
#include "rendering/rendering_manager.hpp"
#include "scene/scene_manager.hpp"
#include <algorithm>

namespace lfs::vis::op {

    namespace {

        std::vector<std::string> resolve_delete_targets(const OperatorContext& ctx, const OperatorProperties* props) {
            if (props) {
                if (const auto name = props->get<std::string>("name"); name && !name->empty()) {
                    return {*name};
                }
            }
            return ctx.selectedNodes();
        }

    } // namespace

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

    bool UndoOperator::poll(const OperatorContext& /*ctx*/, const OperatorProperties* /*props*/) const {
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

    bool RedoOperator::poll(const OperatorContext& /*ctx*/, const OperatorProperties* /*props*/) const {
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

    bool DeleteOperator::poll(const OperatorContext& ctx, const OperatorProperties* props) const {
        const auto targets = resolve_delete_targets(ctx, props);
        if (targets.empty()) {
            return false;
        }

        const auto& scene = ctx.scene().getScene();
        return std::ranges::all_of(targets, [&scene](const std::string& name) {
            return scene.getNode(name) != nullptr;
        });
    }

    OperatorResult DeleteOperator::invoke(OperatorContext& ctx, OperatorProperties& props) {
        const auto nodes = resolve_delete_targets(ctx, &props);
        if (nodes.empty()) {
            return OperatorResult::CANCELLED;
        }

        const bool keep_children = props.get_or<bool>("keep_children", false);
        props.set("resolved_node_names", nodes);
        props.set("keep_children", keep_children);
        for (const auto& name : nodes) {
            ctx.scene().removePLY(name, keep_children);
        }

        return OperatorResult::FINISHED;
    }

    void registerEditOperators() {
        const auto make_schema = [](std::string name, std::string description, PropertyType type) {
            PropertySchema schema{};
            schema.name = std::move(name);
            schema.description = std::move(description);
            schema.type = type;
            return schema;
        };

        propertySchemas().registerSchema(
            UndoOperator::DESCRIPTOR.id(),
            {});
        propertySchemas().registerSchema(
            RedoOperator::DESCRIPTOR.id(),
            {});
        propertySchemas().registerSchema(
            DeleteOperator::DESCRIPTOR.id(),
            {
                make_schema("name", "Optional node name; defaults to the current selected node(s)",
                            PropertyType::STRING),
                make_schema("keep_children",
                            "Keep the children and reparent them to the removed node's parent",
                            PropertyType::BOOL),
            });
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
        propertySchemas().unregisterSchema(UndoOperator::DESCRIPTOR.id());
        propertySchemas().unregisterSchema(RedoOperator::DESCRIPTOR.id());
        propertySchemas().unregisterSchema(DeleteOperator::DESCRIPTOR.id());
    }

} // namespace lfs::vis::op

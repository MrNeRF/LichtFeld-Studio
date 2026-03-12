/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "transform_ops.hpp"
#include "operation/undo_entry.hpp"
#include "operation/undo_history.hpp"
#include "operator/operator_registry.hpp"
#include "operator/property_schema.hpp"
#include "visualizer/gui_capabilities.hpp"

namespace lfs::vis::op {

    namespace {

        std::expected<std::vector<std::string>, std::string> resolve_transform_targets_from_props(
            const OperatorContext& ctx,
            const OperatorProperties* props) {
            std::optional<std::string> requested_node;
            if (props) {
                if (const auto node = props->get<std::string>("node"); node && !node->empty()) {
                    requested_node = *node;
                }
            }
            return cap::resolveTransformTargets(ctx.scene(), requested_node);
        }

    } // namespace

    const OperatorDescriptor TransformSetOperator::DESCRIPTOR = {
        .builtin_id = BuiltinOp::TransformSet,
        .python_class_id = {},
        .label = "Set Transform",
        .description = "Set absolute transform values",
        .icon = "",
        .shortcut = "",
        .flags = OperatorFlags::REGISTER | OperatorFlags::UNDO,
        .source = OperatorSource::CPP,
    };

    bool TransformSetOperator::poll(const OperatorContext& ctx, const OperatorProperties* props) const {
        if (!props) {
            return ctx.hasSelection();
        }

        const auto targets = resolve_transform_targets_from_props(ctx, props);
        if (!targets) {
            return false;
        }

        return props->has("translation") || props->has("rotation") || props->has("scale");
    }

    OperatorResult TransformSetOperator::invoke(OperatorContext& ctx, OperatorProperties& props) {
        const auto nodes = resolve_transform_targets_from_props(ctx, &props);
        if (!nodes) {
            return OperatorResult::CANCELLED;
        }

        const auto translation = props.has("translation") ? std::optional(props.get_or<glm::vec3>("translation", glm::vec3(0.0f)))
                                                          : std::nullopt;
        const auto rotation = props.has("rotation") ? std::optional(props.get_or<glm::vec3>("rotation", glm::vec3(0.0f)))
                                                    : std::nullopt;
        const auto scale = props.has("scale") ? std::optional(props.get_or<glm::vec3>("scale", glm::vec3(1.0f)))
                                              : std::nullopt;
        props.set("resolved_node_names", *nodes);
        const auto result = cap::setTransform(
            ctx.scene(), *nodes, translation, rotation, scale, "transform.set");
        return result ? OperatorResult::FINISHED : OperatorResult::CANCELLED;
    }

    const OperatorDescriptor TransformTranslateOperator::DESCRIPTOR = {
        .builtin_id = BuiltinOp::TransformTranslate,
        .python_class_id = {},
        .label = "Translate",
        .description = "Move selected nodes",
        .icon = "translate",
        .shortcut = "G",
        .flags = OperatorFlags::REGISTER | OperatorFlags::UNDO,
        .source = OperatorSource::CPP,
    };

    bool TransformTranslateOperator::poll(const OperatorContext& ctx, const OperatorProperties* props) const {
        if (!props) {
            return ctx.hasSelection();
        }

        const auto targets = resolve_transform_targets_from_props(ctx, props);
        return targets && props->has("value");
    }

    OperatorResult TransformTranslateOperator::invoke(OperatorContext& ctx, OperatorProperties& props) {
        const auto nodes = resolve_transform_targets_from_props(ctx, &props);
        if (!nodes || !props.has("value")) {
            return OperatorResult::CANCELLED;
        }

        const auto value = props.get_or<glm::vec3>("value", glm::vec3(0.0f));
        props.set("resolved_node_names", *nodes);
        const auto result = cap::translateNodes(ctx.scene(), *nodes, value, "transform.translate");
        return result ? OperatorResult::FINISHED : OperatorResult::CANCELLED;
    }

    const OperatorDescriptor TransformRotateOperator::DESCRIPTOR = {
        .builtin_id = BuiltinOp::TransformRotate,
        .python_class_id = {},
        .label = "Rotate",
        .description = "Rotate selected nodes",
        .icon = "rotate",
        .shortcut = "R",
        .flags = OperatorFlags::REGISTER | OperatorFlags::UNDO,
        .source = OperatorSource::CPP,
    };

    bool TransformRotateOperator::poll(const OperatorContext& ctx, const OperatorProperties* props) const {
        if (!props) {
            return ctx.hasSelection();
        }

        const auto targets = resolve_transform_targets_from_props(ctx, props);
        return targets && props->has("value");
    }

    OperatorResult TransformRotateOperator::invoke(OperatorContext& ctx, OperatorProperties& props) {
        const auto nodes = resolve_transform_targets_from_props(ctx, &props);
        if (!nodes || !props.has("value")) {
            return OperatorResult::CANCELLED;
        }

        const auto value = props.get_or<glm::vec3>("value", glm::vec3(0.0f));
        props.set("resolved_node_names", *nodes);
        const auto result = cap::rotateNodes(ctx.scene(), *nodes, value, "transform.rotate");
        return result ? OperatorResult::FINISHED : OperatorResult::CANCELLED;
    }

    const OperatorDescriptor TransformScaleOperator::DESCRIPTOR = {
        .builtin_id = BuiltinOp::TransformScale,
        .python_class_id = {},
        .label = "Scale",
        .description = "Scale selected nodes",
        .icon = "scale",
        .shortcut = "S",
        .flags = OperatorFlags::REGISTER | OperatorFlags::UNDO,
        .source = OperatorSource::CPP,
    };

    bool TransformScaleOperator::poll(const OperatorContext& ctx, const OperatorProperties* props) const {
        if (!props) {
            return ctx.hasSelection();
        }

        const auto targets = resolve_transform_targets_from_props(ctx, props);
        return targets && props->has("value");
    }

    OperatorResult TransformScaleOperator::invoke(OperatorContext& ctx, OperatorProperties& props) {
        const auto nodes = resolve_transform_targets_from_props(ctx, &props);
        if (!nodes || !props.has("value")) {
            return OperatorResult::CANCELLED;
        }

        const auto value = props.get_or<glm::vec3>("value", glm::vec3(1.0f));
        props.set("resolved_node_names", *nodes);
        const auto result = cap::scaleNodes(ctx.scene(), *nodes, value, "transform.scale");
        return result ? OperatorResult::FINISHED : OperatorResult::CANCELLED;
    }

    const OperatorDescriptor TransformApplyBatchOperator::DESCRIPTOR = {
        .builtin_id = BuiltinOp::TransformApplyBatch,
        .python_class_id = {},
        .label = "Apply Batch Transform",
        .description = "Apply pre-computed transforms with undo support",
        .icon = "",
        .shortcut = "",
        .flags = OperatorFlags::UNDO,
        .source = OperatorSource::CPP,
    };

    bool TransformApplyBatchOperator::poll(const OperatorContext& /*ctx*/,
                                           const OperatorProperties* /*props*/) const {
        return true;
    }

    OperatorResult TransformApplyBatchOperator::invoke(OperatorContext& ctx, OperatorProperties& props) {
        auto node_names = props.get<std::vector<std::string>>("node_names");
        auto old_transforms = props.get<std::vector<glm::mat4>>("old_transforms");
        if (!node_names || !old_transforms || node_names->empty()) {
            return OperatorResult::CANCELLED;
        }

        auto entry = std::make_unique<SceneSnapshot>(ctx.scene(), "transform.batch");
        if (!entry->captureTransformsBefore(*node_names, *old_transforms)) {
            return OperatorResult::CANCELLED;
        }
        entry->captureAfter();
        undoHistory().push(std::move(entry));

        return OperatorResult::FINISHED;
    }

    void registerTransformOperators() {
        const auto make_schema = [](std::string name, std::string description, PropertyType type,
                                    std::optional<int> size = std::nullopt) {
            PropertySchema schema{};
            schema.name = std::move(name);
            schema.description = std::move(description);
            schema.type = type;
            schema.size = size;
            return schema;
        };

        propertySchemas().registerSchema(
            TransformSetOperator::DESCRIPTOR.id(),
            {
                make_schema("node", "Optional node name; defaults to the current selected node(s)",
                            PropertyType::STRING),
                make_schema("translation", "Optional XYZ translation", PropertyType::FLOAT_VECTOR, 3),
                make_schema("rotation", "Optional XYZ Euler rotation in radians",
                            PropertyType::FLOAT_VECTOR, 3),
                make_schema("scale", "Optional XYZ scale", PropertyType::FLOAT_VECTOR, 3),
            });
        propertySchemas().registerSchema(
            TransformTranslateOperator::DESCRIPTOR.id(),
            {
                make_schema("node", "Optional node name; defaults to the current selected node(s)",
                            PropertyType::STRING),
                make_schema("value", "XYZ translation delta", PropertyType::FLOAT_VECTOR, 3),
            });
        propertySchemas().registerSchema(
            TransformRotateOperator::DESCRIPTOR.id(),
            {
                make_schema("node", "Optional node name; defaults to the current selected node(s)",
                            PropertyType::STRING),
                make_schema("value", "XYZ Euler delta in radians", PropertyType::FLOAT_VECTOR, 3),
            });
        propertySchemas().registerSchema(
            TransformScaleOperator::DESCRIPTOR.id(),
            {
                make_schema("node", "Optional node name; defaults to the current selected node(s)",
                            PropertyType::STRING),
                make_schema("value", "XYZ scale multiplier", PropertyType::FLOAT_VECTOR, 3),
            });
        propertySchemas().registerSchema(
            TransformApplyBatchOperator::DESCRIPTOR.id(),
            {});
        operators().registerOperator(BuiltinOp::TransformSet, TransformSetOperator::DESCRIPTOR,
                                     [] { return std::make_unique<TransformSetOperator>(); });
        operators().registerOperator(BuiltinOp::TransformTranslate, TransformTranslateOperator::DESCRIPTOR,
                                     [] { return std::make_unique<TransformTranslateOperator>(); });
        operators().registerOperator(BuiltinOp::TransformRotate, TransformRotateOperator::DESCRIPTOR,
                                     [] { return std::make_unique<TransformRotateOperator>(); });
        operators().registerOperator(BuiltinOp::TransformScale, TransformScaleOperator::DESCRIPTOR,
                                     [] { return std::make_unique<TransformScaleOperator>(); });
        operators().registerOperator(BuiltinOp::TransformApplyBatch, TransformApplyBatchOperator::DESCRIPTOR,
                                     [] { return std::make_unique<TransformApplyBatchOperator>(); });
    }

    void unregisterTransformOperators() {
        operators().unregisterOperator(BuiltinOp::TransformSet);
        operators().unregisterOperator(BuiltinOp::TransformTranslate);
        operators().unregisterOperator(BuiltinOp::TransformRotate);
        operators().unregisterOperator(BuiltinOp::TransformScale);
        operators().unregisterOperator(BuiltinOp::TransformApplyBatch);
        propertySchemas().unregisterSchema(TransformSetOperator::DESCRIPTOR.id());
        propertySchemas().unregisterSchema(TransformTranslateOperator::DESCRIPTOR.id());
        propertySchemas().unregisterSchema(TransformRotateOperator::DESCRIPTOR.id());
        propertySchemas().unregisterSchema(TransformScaleOperator::DESCRIPTOR.id());
        propertySchemas().unregisterSchema(TransformApplyBatchOperator::DESCRIPTOR.id());
    }

} // namespace lfs::vis::op

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

        std::optional<std::string> requested_transform_node(const OperatorProperties* props) {
            if (!props)
                return std::nullopt;
            if (const auto node = props->get<std::string>("node"); node && !node->empty())
                return *node;
            return std::nullopt;
        }

        std::expected<cap::ResolvedTransformTargets, std::string> resolve_transform_targets_from_props(
            const OperatorContext& ctx,
            const OperatorProperties* props) {
            return cap::resolveEditableTransformSelection(
                ctx.scene(), requested_transform_node(props), cap::TransformTargetPolicy::AllowEditableSubset);
        }

        bool poll_transform_operator(const OperatorContext& ctx,
                                     const OperatorProperties* props,
                                     const bool require_value = true) {
            const auto targets = resolve_transform_targets_from_props(ctx, props);
            return targets && (!require_value || !props || props->has("value"));
        }

        template <typename ApplyFn>
        OperatorResult invoke_value_transform_operator(
            OperatorContext& ctx,
            OperatorProperties& props,
            const glm::vec3 default_value,
            const std::string_view undo_label,
            ApplyFn&& apply) {
            const auto nodes = resolve_transform_targets_from_props(ctx, &props);
            if (!nodes || !props.has("value"))
                return OperatorResult::CANCELLED;

            const auto value = props.get_or<glm::vec3>("value", default_value);
            props.set("resolved_node_names", nodes->node_names);
            const auto result = apply(ctx.scene(), nodes->node_names, value, undo_label);
            return result ? OperatorResult::FINISHED : OperatorResult::CANCELLED;
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
        if (!props)
            return poll_transform_operator(ctx, nullptr, false);
        if (!poll_transform_operator(ctx, props, false))
            return false;

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
        props.set("resolved_node_names", nodes->node_names);
        const auto result = cap::setTransform(
            ctx.scene(), nodes->node_names, translation, rotation, scale, "transform.set");
        return result ? OperatorResult::FINISHED : OperatorResult::CANCELLED;
    }

    const OperatorDescriptor TransformTranslateOperator::DESCRIPTOR = {
        .builtin_id = BuiltinOp::TransformTranslate,
        .python_class_id = {},
        .label = "Translate",
        .description = "Move selected nodes",
        .icon = "translate",
        .shortcut = "",
        .flags = OperatorFlags::REGISTER | OperatorFlags::UNDO,
        .source = OperatorSource::CPP,
    };

    bool TransformTranslateOperator::poll(const OperatorContext& ctx, const OperatorProperties* props) const {
        return poll_transform_operator(ctx, props);
    }

    OperatorResult TransformTranslateOperator::invoke(OperatorContext& ctx, OperatorProperties& props) {
        return invoke_value_transform_operator(
            ctx, props, glm::vec3(0.0f), "transform.translate",
            [](SceneManager& scene, const std::vector<std::string>& nodes, const glm::vec3& value, const std::string_view undo_label) {
                return cap::translateNodes(scene, nodes, value, undo_label);
            });
    }

    const OperatorDescriptor TransformRotateOperator::DESCRIPTOR = {
        .builtin_id = BuiltinOp::TransformRotate,
        .python_class_id = {},
        .label = "Rotate",
        .description = "Rotate selected nodes",
        .icon = "rotate",
        .shortcut = "",
        .flags = OperatorFlags::REGISTER | OperatorFlags::UNDO,
        .source = OperatorSource::CPP,
    };

    bool TransformRotateOperator::poll(const OperatorContext& ctx, const OperatorProperties* props) const {
        return poll_transform_operator(ctx, props);
    }

    OperatorResult TransformRotateOperator::invoke(OperatorContext& ctx, OperatorProperties& props) {
        return invoke_value_transform_operator(
            ctx, props, glm::vec3(0.0f), "transform.rotate",
            [](SceneManager& scene, const std::vector<std::string>& nodes, const glm::vec3& value, const std::string_view undo_label) {
                return cap::rotateNodes(scene, nodes, value, undo_label);
            });
    }

    const OperatorDescriptor TransformScaleOperator::DESCRIPTOR = {
        .builtin_id = BuiltinOp::TransformScale,
        .python_class_id = {},
        .label = "Scale",
        .description = "Scale selected nodes",
        .icon = "scale",
        .shortcut = "",
        .flags = OperatorFlags::REGISTER | OperatorFlags::UNDO,
        .source = OperatorSource::CPP,
    };

    bool TransformScaleOperator::poll(const OperatorContext& ctx, const OperatorProperties* props) const {
        return poll_transform_operator(ctx, props);
    }

    OperatorResult TransformScaleOperator::invoke(OperatorContext& ctx, OperatorProperties& props) {
        return invoke_value_transform_operator(
            ctx, props, glm::vec3(1.0f), "transform.scale",
            [](SceneManager& scene, const std::vector<std::string>& nodes, const glm::vec3& value, const std::string_view undo_label) {
                return cap::scaleNodes(scene, nodes, value, undo_label);
            });
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
        pushSceneSnapshotIfChanged(std::move(entry));

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
                make_schema("translation", "Optional visualizer-world XYZ translation", PropertyType::FLOAT_VECTOR, 3),
                make_schema("rotation", "Optional visualizer-world XYZ Euler rotation in radians",
                            PropertyType::FLOAT_VECTOR, 3),
                make_schema("scale", "Optional visualizer-world XYZ scale", PropertyType::FLOAT_VECTOR, 3),
            });
        propertySchemas().registerSchema(
            TransformTranslateOperator::DESCRIPTOR.id(),
            {
                make_schema("node", "Optional node name; defaults to the current selected node(s)",
                            PropertyType::STRING),
                make_schema("value", "Visualizer-world XYZ translation delta", PropertyType::FLOAT_VECTOR, 3),
            });
        propertySchemas().registerSchema(
            TransformRotateOperator::DESCRIPTOR.id(),
            {
                make_schema("node", "Optional node name; defaults to the current selected node(s)",
                            PropertyType::STRING),
                make_schema("value", "Visualizer-world XYZ Euler delta in radians", PropertyType::FLOAT_VECTOR, 3),
            });
        propertySchemas().registerSchema(
            TransformScaleOperator::DESCRIPTOR.id(),
            {
                make_schema("node", "Optional node name; defaults to the current selected node(s)",
                            PropertyType::STRING),
                make_schema("value", "Visualizer-world XYZ scale multiplier", PropertyType::FLOAT_VECTOR, 3),
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

/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#define GLM_ENABLE_EXPERIMENTAL

#include "transform_ops.hpp"
#include "operation/undo_entry.hpp"
#include "operation/undo_history.hpp"
#include "operator/operator_registry.hpp"
#include "scene/scene_manager.hpp"
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtx/euler_angles.hpp>

namespace lfs::vis::op {

    namespace {

        struct TransformComponents {
            glm::vec3 translation{0.0f};
            glm::vec3 rotation{0.0f};
            glm::vec3 scale{1.0f};
        };

        TransformComponents decompose(const glm::mat4& m) {
            TransformComponents result;
            result.translation = glm::vec3(m[3]);

            glm::vec3 col0 = glm::vec3(m[0]);
            glm::vec3 col1 = glm::vec3(m[1]);
            glm::vec3 col2 = glm::vec3(m[2]);

            result.scale.x = glm::length(col0);
            result.scale.y = glm::length(col1);
            result.scale.z = glm::length(col2);

            if (result.scale.x > 0.0f)
                col0 /= result.scale.x;
            if (result.scale.y > 0.0f)
                col1 /= result.scale.y;
            if (result.scale.z > 0.0f)
                col2 /= result.scale.z;

            const glm::mat3 rot(col0, col1, col2);
            glm::extractEulerAngleXYZ(glm::mat4(rot), result.rotation.x, result.rotation.y, result.rotation.z);

            return result;
        }

        glm::mat4 compose(const TransformComponents& c) {
            const glm::mat4 t = glm::translate(glm::mat4(1.0f), c.translation);
            const glm::mat4 r = glm::eulerAngleXYZ(c.rotation.x, c.rotation.y, c.rotation.z);
            const glm::mat4 s = glm::scale(glm::mat4(1.0f), c.scale);
            return t * r * s;
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

    bool TransformSetOperator::poll(const OperatorContext& ctx) const {
        return ctx.hasSelection();
    }

    OperatorResult TransformSetOperator::invoke(OperatorContext& ctx, OperatorProperties& props) {
        const auto nodes = ctx.selectedNodes();
        if (nodes.empty()) {
            return OperatorResult::CANCELLED;
        }

        auto entry = std::make_unique<SceneSnapshot>(ctx.scene(), "transform.set");
        entry->captureTransforms(nodes);

        const auto translation = props.get_or<glm::vec3>("translation", glm::vec3(0.0f));
        const auto rotation = props.get_or<glm::vec3>("rotation", glm::vec3(0.0f));
        const auto scale = props.get_or<glm::vec3>("scale", glm::vec3(1.0f));
        const glm::mat4 new_transform = compose({translation, rotation, scale});

        for (const auto& name : nodes) {
            ctx.scene().setNodeTransform(name, new_transform);
        }

        entry->captureAfter();
        undoHistory().push(std::move(entry));

        return OperatorResult::FINISHED;
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

    bool TransformTranslateOperator::poll(const OperatorContext& ctx) const {
        return ctx.hasSelection();
    }

    OperatorResult TransformTranslateOperator::invoke(OperatorContext& ctx, OperatorProperties& props) {
        const auto nodes = ctx.selectedNodes();
        if (nodes.empty()) {
            return OperatorResult::CANCELLED;
        }

        auto entry = std::make_unique<SceneSnapshot>(ctx.scene(), "transform.translate");
        entry->captureTransforms(nodes);

        const auto value = props.get_or<glm::vec3>("value", glm::vec3(0.0f));

        for (const auto& name : nodes) {
            glm::mat4 transform = ctx.scene().getNodeTransform(name);
            transform[3] += glm::vec4(value, 0.0f);
            ctx.scene().setNodeTransform(name, transform);
        }

        entry->captureAfter();
        undoHistory().push(std::move(entry));

        return OperatorResult::FINISHED;
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

    bool TransformRotateOperator::poll(const OperatorContext& ctx) const {
        return ctx.hasSelection();
    }

    OperatorResult TransformRotateOperator::invoke(OperatorContext& ctx, OperatorProperties& props) {
        const auto nodes = ctx.selectedNodes();
        if (nodes.empty()) {
            return OperatorResult::CANCELLED;
        }

        auto entry = std::make_unique<SceneSnapshot>(ctx.scene(), "transform.rotate");
        entry->captureTransforms(nodes);

        const auto value = props.get_or<glm::vec3>("value", glm::vec3(0.0f));
        const glm::mat4 rotation_delta = glm::eulerAngleXYZ(value.x, value.y, value.z);

        for (const auto& name : nodes) {
            auto components = decompose(ctx.scene().getNodeTransform(name));
            const glm::mat4 current_rotation =
                glm::eulerAngleXYZ(components.rotation.x, components.rotation.y, components.rotation.z);
            const glm::mat4 new_rotation = rotation_delta * current_rotation;
            glm::extractEulerAngleXYZ(new_rotation, components.rotation.x, components.rotation.y, components.rotation.z);
            ctx.scene().setNodeTransform(name, compose(components));
        }

        entry->captureAfter();
        undoHistory().push(std::move(entry));

        return OperatorResult::FINISHED;
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

    bool TransformScaleOperator::poll(const OperatorContext& ctx) const {
        return ctx.hasSelection();
    }

    OperatorResult TransformScaleOperator::invoke(OperatorContext& ctx, OperatorProperties& props) {
        const auto nodes = ctx.selectedNodes();
        if (nodes.empty()) {
            return OperatorResult::CANCELLED;
        }

        auto entry = std::make_unique<SceneSnapshot>(ctx.scene(), "transform.scale");
        entry->captureTransforms(nodes);

        const auto value = props.get_or<glm::vec3>("value", glm::vec3(1.0f));

        for (const auto& name : nodes) {
            auto components = decompose(ctx.scene().getNodeTransform(name));
            components.scale *= value;
            ctx.scene().setNodeTransform(name, compose(components));
        }

        entry->captureAfter();
        undoHistory().push(std::move(entry));

        return OperatorResult::FINISHED;
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

    bool TransformApplyBatchOperator::poll(const OperatorContext& /*ctx*/) const {
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
    }

} // namespace lfs::vis::op

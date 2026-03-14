/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "transform_ops.hpp"
#include "core/logger.hpp"
#include "core/scene.hpp"
#include "scene/scene_manager.hpp"
#include "visualizer/gui_capabilities.hpp"
#include <glm/gtc/matrix_transform.hpp>

namespace lfs::vis::op {

    namespace {

        std::expected<std::vector<std::string>, std::string> resolve_editable_selected_transform_targets(
            SceneManager& scene) {
            return cap::resolveEditableTransformTargets(scene, std::nullopt);
        }

        glm::vec3 resolve_transform_pivot(SceneManager& scene,
                                          const OperatorProperties& props,
                                          const std::vector<std::string>& targets) {
            if (const auto pivot = props.get<glm::vec3>("pivot"))
                return *pivot;
            return cap::getTransformTargetsCenter(scene, targets).value_or(glm::vec3(0.0f));
        }

    } // namespace

    OperationResult TransformTranslate::execute(SceneManager& scene,
                                                const OperatorProperties& props,
                                                const std::any& /*input*/) {
        auto selected = resolve_editable_selected_transform_targets(scene);
        if (!selected) {
            return OperationResult::failure(selected.error());
        }

        auto delta = props.get_or<glm::vec3>("delta", glm::vec3(0.0f));

        for (const auto& name : *selected) {
            auto transform = scene.getNodeTransform(name);
            transform = glm::translate(transform, delta);
            scene.setNodeTransform(name, transform);
        }

        return OperationResult::success();
    }

    bool TransformTranslate::poll(SceneManager& scene) const {
        return resolve_editable_selected_transform_targets(scene).has_value();
    }

    OperationResult TransformRotate::execute(SceneManager& scene,
                                             const OperatorProperties& props,
                                             const std::any& /*input*/) {
        auto selected = resolve_editable_selected_transform_targets(scene);
        if (!selected) {
            return OperationResult::failure(selected.error());
        }

        auto axis = props.get_or<glm::vec3>("axis", glm::vec3(0.0f, 1.0f, 0.0f));
        auto angle = props.get_or<float>("angle", 0.0f);
        const auto pivot = resolve_transform_pivot(scene, props, *selected);

        for (const auto& name : *selected) {
            auto transform = scene.getNodeTransform(name);
            transform = glm::translate(transform, pivot);
            transform = glm::rotate(transform, glm::radians(angle), axis);
            transform = glm::translate(transform, -pivot);
            scene.setNodeTransform(name, transform);
        }

        return OperationResult::success();
    }

    bool TransformRotate::poll(SceneManager& scene) const {
        return resolve_editable_selected_transform_targets(scene).has_value();
    }

    OperationResult TransformScale::execute(SceneManager& scene,
                                            const OperatorProperties& props,
                                            const std::any& /*input*/) {
        auto selected = resolve_editable_selected_transform_targets(scene);
        if (!selected) {
            return OperationResult::failure(selected.error());
        }

        auto scale = props.get_or<glm::vec3>("scale", glm::vec3(1.0f));
        const auto pivot = resolve_transform_pivot(scene, props, *selected);

        for (const auto& name : *selected) {
            auto transform = scene.getNodeTransform(name);
            transform = glm::translate(transform, pivot);
            transform = glm::scale(transform, scale);
            transform = glm::translate(transform, -pivot);
            scene.setNodeTransform(name, transform);
        }

        return OperationResult::success();
    }

    bool TransformScale::poll(SceneManager& scene) const {
        return resolve_editable_selected_transform_targets(scene).has_value();
    }

    OperationResult TransformSet::execute(SceneManager& scene,
                                          const OperatorProperties& props,
                                          const std::any& /*input*/) {
        auto selected = resolve_editable_selected_transform_targets(scene);
        if (!selected) {
            return OperationResult::failure(selected.error());
        }

        auto transform = props.get_or<glm::mat4>("transform", glm::mat4(1.0f));

        for (const auto& name : *selected) {
            scene.setNodeTransform(name, transform);
        }

        return OperationResult::success();
    }

    bool TransformSet::poll(SceneManager& scene) const {
        return resolve_editable_selected_transform_targets(scene).has_value();
    }

} // namespace lfs::vis::op

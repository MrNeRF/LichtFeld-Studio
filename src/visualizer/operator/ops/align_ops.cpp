/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "align_ops.hpp"
#include "core/services.hpp"
#include "gui/gui_manager.hpp"
#include "operation/undo_entry.hpp"
#include "operation/undo_history.hpp"
#include "operator/operator_registry.hpp"
#include "rendering/rendering_manager.hpp"
#include "scene/scene_manager.hpp"
#include "visualizer_impl.hpp"
#include <GLFW/glfw3.h>
#include <glm/gtc/matrix_transform.hpp>

namespace lfs::vis::op {

    const OperatorDescriptor AlignPickPointOperator::DESCRIPTOR = {
        .builtin_id = BuiltinOp::AlignPickPoint,
        .python_class_id = {},
        .label = "Align to Ground",
        .description = "Pick 3 points to define ground plane",
        .icon = "align",
        .shortcut = "",
        .flags = OperatorFlags::REGISTER | OperatorFlags::UNDO,
        .source = OperatorSource::CPP,
        .poll_deps = PollDependency::SCENE,
    };

    bool AlignPickPointOperator::poll(const OperatorContext& ctx) const {
        return ctx.scene().getScene().getTotalGaussianCount() > 0;
    }

    OperatorResult AlignPickPointOperator::invoke(OperatorContext& ctx, OperatorProperties& props) {
        picked_points_.clear();
        transforms_before_.clear();
        services().clearAlignPickedPoints();

        const auto x = props.get_or<double>("x", 0.0);
        const auto y = props.get_or<double>("y", 0.0);

        const glm::vec3 world_pos = unprojectScreenPoint(x, y);
        if (world_pos.x <= -1e9f) {
            return OperatorResult::CANCELLED;
        }

        picked_points_.push_back(world_pos);
        services().setAlignPickedPoints(picked_points_);
        captureTransformsBefore(ctx);

        if (services().renderingOrNull()) {
            services().renderingOrNull()->markDirty();
        }

        return OperatorResult::RUNNING_MODAL;
    }

    OperatorResult AlignPickPointOperator::modal(OperatorContext& ctx, OperatorProperties& /*props*/) {
        const auto* event = ctx.event();
        if (!event) {
            return OperatorResult::RUNNING_MODAL;
        }

        if (event->type == ModalEvent::Type::MOUSE_BUTTON) {
            const auto* mb = event->as<MouseButtonEvent>();
            if (!mb || mb->action != GLFW_PRESS) {
                return OperatorResult::RUNNING_MODAL;
            }

            if (mb->button == GLFW_MOUSE_BUTTON_RIGHT) {
                return OperatorResult::CANCELLED;
            }

            if (mb->button == GLFW_MOUSE_BUTTON_LEFT) {
                const glm::vec3 world_pos = unprojectScreenPoint(mb->position.x, mb->position.y);
                if (world_pos.x <= -1e9f) {
                    return OperatorResult::RUNNING_MODAL;
                }

                picked_points_.push_back(world_pos);
                services().setAlignPickedPoints(picked_points_);

                if (services().renderingOrNull()) {
                    services().renderingOrNull()->markDirty();
                }

                if (picked_points_.size() == 3) {
                    applyAlignment(ctx);
                    services().clearAlignPickedPoints();
                    return OperatorResult::FINISHED;
                }
            }
        }

        if (event->type == ModalEvent::Type::KEY) {
            const auto* ke = event->as<KeyEvent>();
            if (ke && ke->key == GLFW_KEY_ESCAPE && ke->action == GLFW_PRESS) {
                return OperatorResult::CANCELLED;
            }
        }

        return OperatorResult::RUNNING_MODAL;
    }

    void AlignPickPointOperator::cancel(OperatorContext& /*ctx*/) {
        picked_points_.clear();
        transforms_before_.clear();
        services().clearAlignPickedPoints();
        if (services().renderingOrNull()) {
            services().renderingOrNull()->markDirty();
        }
    }

    glm::vec3 AlignPickPointOperator::unprojectScreenPoint(double x, double y) const {
        constexpr glm::vec3 INVALID_POS(-1e10f);

        auto* rm = services().renderingOrNull();
        auto* gm = services().guiOrNull();
        if (!rm || !gm || !gm->getViewer()) {
            return INVALID_POS;
        }

        const float depth = rm->getDepthAtPixel(static_cast<int>(x), static_cast<int>(y));
        if (depth < 0.0f) {
            return INVALID_POS;
        }

        const auto& viewport = gm->getViewer()->getViewport();
        return viewport.unprojectPixel(static_cast<float>(x), static_cast<float>(y), depth,
                                       rm->getFocalLengthMm());
    }

    void AlignPickPointOperator::captureTransformsBefore(const OperatorContext& ctx) {
        transforms_before_.clear();
        auto& scene = ctx.scene().getScene();
        for (const auto* node : scene.getNodes()) {
            transforms_before_.emplace_back(node->name, node->local_transform);
        }
    }

    void AlignPickPointOperator::applyAlignment(OperatorContext& ctx) {
        if (picked_points_.size() != 3) {
            return;
        }

        std::vector<std::string> node_names;
        auto& scene = ctx.scene().getScene();
        for (const auto* node : scene.getNodes()) {
            node_names.push_back(node->name);
        }

        auto entry = std::make_unique<SceneSnapshot>(ctx.scene(), "transform.align");
        entry->captureTransforms(node_names);

        const glm::vec3& p0 = picked_points_[0];
        const glm::vec3& p1 = picked_points_[1];
        const glm::vec3& p2 = picked_points_[2];

        const glm::vec3 v01 = p1 - p0;
        const glm::vec3 v02 = p2 - p0;
        glm::vec3 normal = glm::normalize(glm::cross(v01, v02));
        const glm::vec3 center = (p0 + p1 + p2) / 3.0f;

        if (normal.y > 0.0f) {
            normal = -normal;
        }

        constexpr glm::vec3 kTargetUp(0.0f, -1.0f, 0.0f);
        const glm::vec3 axis = glm::cross(normal, kTargetUp);
        const float axis_len = glm::length(axis);

        glm::mat4 rotation(1.0f);
        if (axis_len > 1e-6f) {
            const float angle = acos(glm::clamp(glm::dot(normal, kTargetUp), -1.0f, 1.0f));
            rotation = glm::rotate(glm::mat4(1.0f), angle, glm::normalize(axis));
        } else if (glm::dot(normal, kTargetUp) < 0.0f) {
            rotation = glm::rotate(glm::mat4(1.0f), glm::pi<float>(), glm::vec3(1.0f, 0.0f, 0.0f));
        }

        const glm::mat4 to_origin = glm::translate(glm::mat4(1.0f), -center);
        const glm::mat4 from_origin = glm::translate(glm::mat4(1.0f), glm::vec3(center.x, 0.0f, center.z));
        const glm::mat4 transform = from_origin * rotation * to_origin;

        for (const auto* node : scene.getNodes()) {
            ctx.scene().setNodeTransform(node->name, transform * node->local_transform);
        }

        entry->captureAfter();
        undoHistory().push(std::move(entry));

        if (services().renderingOrNull()) {
            services().renderingOrNull()->markDirty();
        }
    }

    void registerAlignOperators() {
        operators().registerOperator(BuiltinOp::AlignPickPoint, AlignPickPointOperator::DESCRIPTOR,
                                     [] { return std::make_unique<AlignPickPointOperator>(); });
    }

    void unregisterAlignOperators() {
        operators().unregisterOperator(BuiltinOp::AlignPickPoint);
    }

} // namespace lfs::vis::op

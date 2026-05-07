/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "align_ops.hpp"
#include "core/services.hpp"
#include "gui/gui_manager.hpp"
#include "input/key_codes.hpp"
#include "operation/undo_entry.hpp"
#include "operation/undo_history.hpp"
#include "operator/operator_registry.hpp"
#include "rendering/rendering_manager.hpp"
#include "scene/scene_manager.hpp"
#include "visualizer/scene_coordinate_utils.hpp"
#include "visualizer_impl.hpp"
#include <algorithm>
#include <glm/gtc/matrix_transform.hpp>
#include <unordered_set>

namespace lfs::vis::op {

    namespace {
        [[nodiscard]] bool isAlignTransformTarget(const core::SceneNode& node) {
            switch (node.type) {
            case core::NodeType::SPLAT:
            case core::NodeType::POINTCLOUD:
            case core::NodeType::GROUP:
            case core::NodeType::DATASET:
            case core::NodeType::MESH:
                return true;
            default:
                return false;
            }
        }

        [[nodiscard]] core::NodeId resolveAlignTargetId(const core::Scene& scene,
                                                        const core::SceneNode& node) {
            if ((node.type == core::NodeType::CROPBOX ||
                 node.type == core::NodeType::ELLIPSOID) &&
                node.parent_id != core::NULL_NODE) {
                const auto* const parent = scene.getNodeById(node.parent_id);
                if (parent && isAlignTransformTarget(*parent)) {
                    return parent->id;
                }
            }

            return isAlignTransformTarget(node) ? node.id : core::NULL_NODE;
        }

        [[nodiscard]] bool hasTargetAncestor(const core::Scene& scene,
                                             const core::NodeId node_id,
                                             const std::unordered_set<core::NodeId>& target_ids) {
            const auto* node = scene.getNodeById(node_id);
            while (node && node->parent_id != core::NULL_NODE) {
                if (target_ids.contains(node->parent_id)) {
                    return true;
                }
                node = scene.getNodeById(node->parent_id);
            }
            return false;
        }

        [[nodiscard]] std::vector<core::NodeId> resolveAlignmentTargets(const OperatorContext& ctx) {
            const auto& scene = ctx.scene().getScene();
            const auto selected_names = ctx.selectedNodes();

            std::vector<core::NodeId> target_ids;
            std::unordered_set<core::NodeId> seen;

            if (!selected_names.empty()) {
                for (const auto& name : selected_names) {
                    const auto* const node = scene.getNode(name);
                    if (!node) {
                        continue;
                    }

                    const core::NodeId target_id = resolveAlignTargetId(scene, *node);
                    if (target_id != core::NULL_NODE && seen.insert(target_id).second) {
                        target_ids.push_back(target_id);
                    }
                }
            } else {
                for (const auto node_id : scene.getRootNodes()) {
                    const auto* const node = scene.getNodeById(node_id);
                    if (node && isAlignTransformTarget(*node) && seen.insert(node_id).second) {
                        target_ids.push_back(node_id);
                    }
                }
            }

            std::vector<core::NodeId> top_level_targets;
            top_level_targets.reserve(target_ids.size());
            for (const core::NodeId target_id : target_ids) {
                if (!hasTargetAncestor(scene, target_id, seen)) {
                    top_level_targets.push_back(target_id);
                }
            }
            return top_level_targets;
        }

        [[nodiscard]] bool isTargetOrDescendant(const core::Scene& scene,
                                                const core::NodeId node_id,
                                                const std::unordered_set<core::NodeId>& target_ids) {
            const auto* node = scene.getNodeById(node_id);
            while (node) {
                if (target_ids.contains(node->id)) {
                    return true;
                }
                node = node->parent_id != core::NULL_NODE ? scene.getNodeById(node->parent_id) : nullptr;
            }
            return false;
        }

        [[nodiscard]] std::vector<bool> buildAlignmentTargetNodeMask(const core::Scene& scene,
                                                                     const std::vector<core::NodeId>& target_ids) {
            if (target_ids.empty()) {
                return {};
            }

            const std::unordered_set<core::NodeId> target_set(target_ids.begin(), target_ids.end());
            std::vector<bool> mask;
            for (const auto* const node : scene.getNodes()) {
                if (node && node->model && scene.isNodeEffectivelyVisible(node->id)) {
                    mask.push_back(isTargetOrDescendant(scene, node->id, target_set));
                }
            }
            return mask;
        }

        [[nodiscard]] bool hasVisibleAlignmentTarget(const std::vector<bool>& mask) {
            return std::any_of(mask.begin(), mask.end(), [](const bool enabled) {
                return enabled;
            });
        }
    } // namespace

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

    bool AlignPickPointOperator::poll(const OperatorContext& ctx,
                                      const OperatorProperties* /*props*/) const {
        return ctx.scene().getScene().getTotalGaussianCount() > 0;
    }

    OperatorResult AlignPickPointOperator::invoke(OperatorContext& ctx, OperatorProperties& props) {
        picked_points_.clear();
        services().clearAlignPickedPoints();
        pick_button_ = props.get_or<int>("button", static_cast<int>(lfs::vis::input::AppMouseButton::LEFT));

        const auto x = props.get_or<double>("x", 0.0);
        const auto y = props.get_or<double>("y", 0.0);

        const glm::vec3 world_pos = unprojectScreenPoint(ctx, x, y);
        if (!Viewport::isValidWorldPosition(world_pos)) {
            return OperatorResult::CANCELLED;
        }

        picked_points_.push_back(world_pos);
        services().setAlignPickedPoints(picked_points_);

        if (services().renderingOrNull()) {
            services().renderingOrNull()->markDirty(DirtyFlag::OVERLAY);
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
            if (!mb || mb->action != lfs::vis::input::ACTION_PRESS) {
                return OperatorResult::RUNNING_MODAL;
            }

            if (mb->button == static_cast<int>(lfs::vis::input::AppMouseButton::RIGHT) &&
                mb->button != pick_button_) {
                return OperatorResult::CANCELLED;
            }

            if (mb->button == pick_button_) {
                const glm::vec3 world_pos = unprojectScreenPoint(ctx, mb->position.x, mb->position.y);
                if (!Viewport::isValidWorldPosition(world_pos)) {
                    return OperatorResult::RUNNING_MODAL;
                }

                picked_points_.push_back(world_pos);
                services().setAlignPickedPoints(picked_points_);

                if (services().renderingOrNull()) {
                    services().renderingOrNull()->markDirty(DirtyFlag::OVERLAY);
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
            if (ke && ke->action == lfs::vis::input::ACTION_PRESS) {
                return OperatorResult::PASS_THROUGH;
            }
        }

        return OperatorResult::RUNNING_MODAL;
    }

    void AlignPickPointOperator::cancel(OperatorContext& /*ctx*/) {
        picked_points_.clear();
        services().clearAlignPickedPoints();
        if (services().renderingOrNull()) {
            services().renderingOrNull()->markDirty(DirtyFlag::OVERLAY);
        }
    }

    glm::vec3 AlignPickPointOperator::unprojectScreenPoint(const OperatorContext& ctx,
                                                           const double x,
                                                           const double y) const {
        auto* rm = services().renderingOrNull();
        auto* gm = services().guiOrNull();
        if (!rm || !gm || !gm->getViewer()) {
            return glm::vec3(Viewport::INVALID_WORLD_POS);
        }

        const auto viewport_pos = gm->getViewportPos();
        const auto viewport_size = gm->getViewportSize();

        const auto panel_info = rm->resolveViewerPanel(
            gm->getViewer()->getViewport(),
            viewport_pos,
            viewport_size,
            glm::vec2(static_cast<float>(x), static_cast<float>(y)));
        if (!panel_info || !panel_info->valid()) {
            return glm::vec3(Viewport::INVALID_WORLD_POS);
        }

        const float scale_x = static_cast<float>(panel_info->render_width) / panel_info->width;
        const float scale_y = static_cast<float>(panel_info->render_height) / panel_info->height;
        const float render_x = (static_cast<float>(x) - panel_info->x) * scale_x;
        const float render_y = (static_cast<float>(y) - panel_info->y) * scale_y;

        Viewport projection_viewport = *panel_info->viewport;
        projection_viewport.windowSize = {panel_info->render_width, panel_info->render_height};

        float depth = -1.0f;
        if (ctx.hasSelection()) {
            const auto target_ids = resolveAlignmentTargets(ctx);
            const auto target_mask = buildAlignmentTargetNodeMask(ctx.scene().getScene(), target_ids);
            if (!hasVisibleAlignmentTarget(target_mask)) {
                return glm::vec3(Viewport::INVALID_WORLD_POS);
            }
            depth = rm->renderDepthAtPixelForNodeMask(
                &ctx.scene(),
                projection_viewport,
                {panel_info->render_width, panel_info->render_height},
                static_cast<int>(render_x),
                static_cast<int>(render_y),
                target_mask);
        } else {
            depth = rm->getDepthAtPixel(
                static_cast<int>(render_x),
                static_cast<int>(render_y),
                panel_info->panel);
        }
        if (depth <= 0.0f) {
            return glm::vec3(Viewport::INVALID_WORLD_POS);
        }

        return projection_viewport.unprojectPixel(
            render_x,
            render_y,
            depth,
            rm->getFocalLengthMm());
    }

    void AlignPickPointOperator::applyAlignment(OperatorContext& ctx) {
        if (picked_points_.size() != 3) {
            return;
        }

        const auto target_ids = resolveAlignmentTargets(ctx);
        if (target_ids.empty()) {
            LOG_WARN("3-point alignment has no transformable selected target");
            return;
        }

        auto& scene = ctx.scene().getScene();
        std::vector<std::string> node_names;
        node_names.reserve(target_ids.size());
        for (const auto node_id : target_ids) {
            const auto* const node = scene.getNodeById(node_id);
            if (node) {
                node_names.push_back(node->name);
            }
        }

        auto entry = std::make_unique<SceneSnapshot>(ctx.scene(), "transform.align");
        entry->captureTransforms(node_names);

        const glm::vec3& p0 = picked_points_[0];
        const glm::vec3& p1 = picked_points_[1];
        const glm::vec3& p2 = picked_points_[2];

        const glm::vec3 v01 = p1 - p0;
        const glm::vec3 v02 = p2 - p0;
        const glm::vec3 cross_v = glm::cross(v01, v02);
        const float cross_len = glm::length(cross_v);
        if (cross_len <= 1e-6f) {
            return;
        }
        glm::vec3 normal = cross_v / cross_len;
        const glm::vec3 center = (p0 + p1 + p2) / 3.0f;

        constexpr glm::vec3 kTargetUp(0.0f, 1.0f, 0.0f);
        if (glm::dot(normal, kTargetUp) < 0.0f) {
            normal = -normal;
        }

        const glm::vec3 axis = glm::cross(normal, kTargetUp);
        const float axis_len = glm::length(axis);

        glm::mat4 rotation(1.0f);
        if (axis_len > 1e-6f) {
            const float angle = acos(glm::clamp(glm::dot(normal, kTargetUp), -1.0f, 1.0f));
            rotation = glm::rotate(glm::mat4(1.0f), angle, glm::normalize(axis));
        }

        const glm::mat4 to_origin = glm::translate(glm::mat4(1.0f), -center);
        const glm::mat4 from_origin =
            glm::translate(glm::mat4(1.0f), center - glm::dot(center, kTargetUp) * kTargetUp);
        const glm::mat4 visualizer_transform = from_origin * rotation * to_origin;

        for (const auto node_id : target_ids) {
            const auto* const node = scene.getNodeById(node_id);
            if (!node) {
                continue;
            }

            const glm::mat4 old_visualizer_world = vis::scene_coords::nodeVisualizerWorldTransform(scene, node_id);
            const glm::mat4 new_visualizer_world = visualizer_transform * old_visualizer_world;
            const auto new_local =
                vis::scene_coords::nodeLocalTransformFromVisualizerWorld(scene, node_id, new_visualizer_world);
            if (!new_local) {
                continue;
            }

            ctx.scene().setNodeTransform(node->name, *new_local);
        }

        entry->captureAfter();
        pushSceneSnapshotIfChanged(std::move(entry));

        if (services().renderingOrNull()) {
            services().renderingOrNull()->markDirty(DirtyFlag::SPLATS | DirtyFlag::MESH | DirtyFlag::OVERLAY);
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

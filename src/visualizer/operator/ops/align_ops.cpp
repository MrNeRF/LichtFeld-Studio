/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "align_ops.hpp"
#include "core/event_bridge/localization_manager.hpp"
#include "core/services.hpp"
#include "gui/gui_manager.hpp"
#include "gui/string_keys.hpp"
#include "input/key_codes.hpp"
#include "operation/undo_entry.hpp"
#include "operation/undo_history.hpp"
#include "operator/operator_registry.hpp"
#include "rendering/coordinate_conventions.hpp"
#include "rendering/rendering_manager.hpp"
#include "scene/scene_manager.hpp"
#include "visualizer/scene_coordinate_utils.hpp"
#include "visualizer_impl.hpp"
#include <algorithm>
#include <cmath>
#include <glm/gtc/matrix_transform.hpp>
#include <unordered_set>

namespace lfs::vis::op {

    namespace {
        constexpr double kClickDragThresholdPx = 4.0;
        constexpr double kNearExistingPointPx = 6.0;

        [[nodiscard]] bool isAlignTransformTarget(const core::SceneNode& node) {
            switch (node.type) {
            case core::NodeType::SPLAT:
            case core::NodeType::POINTCLOUD:
            case core::NodeType::GROUP:
            case core::NodeType::PLY_SEQUENCE:
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

        void faceNormalTowardCamera(glm::vec3& normal, const glm::vec3& center, const glm::vec3& camera_pos) {
            if (glm::dot(normal, camera_pos - center) < 0.0f) {
                normal = -normal;
            }
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

    OperatorResult AlignPickPointOperator::invoke(OperatorContext& /*ctx*/, OperatorProperties& props) {
        picked_points_.clear();
        pick_panel_.reset();
        press_active_ = false;
        press_button_ = -1;
        services().clearAlignPickedPoints();
        pick_button_ = props.get_or<int>("button", static_cast<int>(lfs::vis::input::AppMouseButton::LEFT));

        // First press enters the modal; the point is placed on release if the
        // gesture is a click (see modal click-vs-drag handling).
        press_active_ = true;
        press_button_ = pick_button_;
        press_pos_ = {
            props.get_or<double>("x", 0.0),
            props.get_or<double>("y", 0.0),
        };

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

        if (event->type == ModalEvent::Type::MOUSE_MOVE ||
            event->type == ModalEvent::Type::MOUSE_SCROLL) {
            return OperatorResult::PASS_THROUGH;
        }

        if (event->type == ModalEvent::Type::MOUSE_BUTTON) {
            const auto* mb = event->as<MouseButtonEvent>();
            if (!mb) {
                return OperatorResult::RUNNING_MODAL;
            }

            const bool is_pick_button = mb->button == pick_button_;
            const bool is_right_button =
                mb->button == static_cast<int>(lfs::vis::input::AppMouseButton::RIGHT);

            if (mb->action == lfs::vis::input::ACTION_PRESS) {
                if (is_pick_button || is_right_button) {
                    press_active_ = true;
                    press_button_ = mb->button;
                    press_pos_ = mb->position;
                }
                // Let camera navigation own press so orbit/pan can begin.
                return OperatorResult::PASS_THROUGH;
            }

            if (mb->action != lfs::vis::input::ACTION_RELEASE) {
                return OperatorResult::PASS_THROUGH;
            }

            if (!press_active_ || press_button_ != mb->button) {
                return OperatorResult::PASS_THROUGH;
            }

            press_active_ = false;
            const double move_dist = glm::length(mb->position - press_pos_);
            if (move_dist > kClickDragThresholdPx) {
                return OperatorResult::PASS_THROUGH;
            }

            if (is_right_button && !is_pick_button) {
                removeLastPoint();
                return OperatorResult::RUNNING_MODAL;
            }

            if (is_pick_button) {
                if (picked_points_.size() >= 3) {
                    return OperatorResult::RUNNING_MODAL;
                }
                tryPlacePoint(ctx, press_pos_.x, press_pos_.y);
                return OperatorResult::RUNNING_MODAL;
            }

            return OperatorResult::PASS_THROUGH;
        }

        if (event->type == ModalEvent::Type::KEY) {
            const auto* ke = event->as<KeyEvent>();
            if (!ke || ke->action != lfs::vis::input::ACTION_PRESS) {
                return OperatorResult::PASS_THROUGH;
            }

            if (ke->key == lfs::vis::input::KEY_ESCAPE) {
                return OperatorResult::CANCELLED;
            }

            if (ke->key == lfs::vis::input::KEY_BACKSPACE) {
                removeLastPoint();
                return OperatorResult::RUNNING_MODAL;
            }

            if (ke->key == lfs::vis::input::KEY_ENTER ||
                ke->key == lfs::vis::input::KEY_KP_ENTER) {
                if (picked_points_.size() != 3) {
                    return OperatorResult::RUNNING_MODAL;
                }
                if (applyAlignment(ctx)) {
                    picked_points_.clear();
                    pick_panel_.reset();
                    services().clearAlignPickedPoints();
                    return OperatorResult::FINISHED;
                }
                return OperatorResult::RUNNING_MODAL;
            }

            return OperatorResult::PASS_THROUGH;
        }

        return OperatorResult::PASS_THROUGH;
    }

    void AlignPickPointOperator::cancel(OperatorContext& /*ctx*/) {
        picked_points_.clear();
        pick_panel_.reset();
        press_active_ = false;
        services().clearAlignPickedPoints();
        if (services().renderingOrNull()) {
            services().renderingOrNull()->markDirty(DirtyFlag::OVERLAY);
        }
    }

    void AlignPickPointOperator::setStatus(const char* locale_key, const double duration_seconds) const {
        services().setAlignStatusMessage(LOC(locale_key), duration_seconds);
        if (services().renderingOrNull()) {
            services().renderingOrNull()->markDirty(DirtyFlag::OVERLAY);
        }
    }

    void AlignPickPointOperator::syncPickedPointsToServices() {
        services().setAlignPickedPoints(picked_points_);
        if (services().renderingOrNull()) {
            services().renderingOrNull()->markDirty(DirtyFlag::OVERLAY);
        }
    }

    void AlignPickPointOperator::removeLastPoint() {
        if (picked_points_.empty()) {
            return;
        }
        picked_points_.pop_back();
        if (picked_points_.empty()) {
            pick_panel_.reset();
        }
        syncPickedPointsToServices();
    }

    bool AlignPickPointOperator::tryPlacePoint(OperatorContext& ctx, const double x, const double y) {
        if (picked_points_.size() >= 3) {
            return false;
        }

        if (isNearExistingPoint(x, y)) {
            return false;
        }

        SplitViewPanelId panel = SplitViewPanelId::Left;
        const glm::vec3 world_pos = unprojectScreenPoint(ctx, x, y, &panel);
        if (!Viewport::isValidWorldPosition(world_pos)) {
            setStatus(lichtfeld::Strings::Align::STATUS_NO_SURFACE);
            return false;
        }

        if (!pick_panel_) {
            pick_panel_ = panel;
        }

        picked_points_.push_back(world_pos);
        syncPickedPointsToServices();
        return true;
    }

    bool AlignPickPointOperator::isNearExistingPoint(const double x, const double y) const {
        if (picked_points_.empty()) {
            return false;
        }

        auto* rm = services().renderingOrNull();
        auto* gm = services().guiOrNull();
        if (!rm || !gm || !gm->getViewer()) {
            return false;
        }

        const auto viewport_pos = gm->getViewportPos();
        const auto viewport_size = gm->getViewportSize();
        const auto panel_info = rm->resolveViewerPanel(
            gm->getViewer()->getViewport(),
            viewport_pos,
            viewport_size,
            glm::vec2(static_cast<float>(x), static_cast<float>(y)));
        if (!panel_info || !panel_info->valid()) {
            return false;
        }

        const auto render_settings = rm->getSettings();
        Viewport projection_viewport = *panel_info->viewport;
        projection_viewport.windowSize = {panel_info->render_width, panel_info->render_height};

        const float screen_scale_x = panel_info->width / static_cast<float>(std::max(panel_info->render_width, 1));
        const float screen_scale_y = panel_info->height / static_cast<float>(std::max(panel_info->render_height, 1));
        const glm::vec2 click_screen(static_cast<float>(x), static_cast<float>(y));

        for (const auto& point : picked_points_) {
            const auto projected = lfs::rendering::projectWorldPoint(
                projection_viewport.camera.R,
                projection_viewport.camera.t,
                projection_viewport.windowSize,
                point,
                render_settings.focal_length_mm,
                render_settings.orthographic,
                render_settings.ortho_scale);
            if (!projected) {
                continue;
            }
            const glm::vec2 screen_pos{
                panel_info->x + projected->x * screen_scale_x,
                panel_info->y + projected->y * screen_scale_y,
            };
            if (glm::length(screen_pos - click_screen) <= static_cast<float>(kNearExistingPointPx)) {
                return true;
            }
        }
        return false;
    }

    glm::vec3 AlignPickPointOperator::resolvePickPanelCameraPosition() const {
        auto* rm = services().renderingOrNull();
        auto* gm = services().guiOrNull();
        if (!rm || !gm || !gm->getViewer()) {
            return glm::vec3(0.0f);
        }

        const auto viewport_pos = gm->getViewportPos();
        const auto viewport_size = gm->getViewportSize();
        const auto panel_info = rm->resolveViewerPanel(
            gm->getViewer()->getViewport(),
            viewport_pos,
            viewport_size,
            std::nullopt,
            pick_panel_);
        if (!panel_info || !panel_info->valid() || !panel_info->viewport) {
            return gm->getViewer()->getViewport().camera.t;
        }
        return panel_info->viewport->camera.t;
    }

    glm::vec3 AlignPickPointOperator::unprojectScreenPoint(const OperatorContext& ctx,
                                                           const double x,
                                                           const double y,
                                                           SplitViewPanelId* out_panel) const {
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

        if (out_panel) {
            *out_panel = panel_info->panel;
        }

        const float scale_x = static_cast<float>(panel_info->render_width) / panel_info->width;
        const float scale_y = static_cast<float>(panel_info->render_height) / panel_info->height;
        const float render_x = (static_cast<float>(x) - panel_info->x) * scale_x;
        const float render_y = (static_cast<float>(y) - panel_info->y) * scale_y;

        Viewport projection_viewport = *panel_info->viewport;
        projection_viewport.windowSize = {panel_info->render_width, panel_info->render_height};
        const auto render_settings = rm->getSettings();

        const int depth_x = static_cast<int>(render_x);
        const int depth_y = static_cast<int>(render_y);

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
                depth_x,
                depth_y,
                target_mask);
            if (depth <= 0.0f) {
                depth = rm->getDepthAtPixel(depth_x, depth_y, panel_info->panel);
            }
        } else {
            depth = rm->getDepthAtPixel(depth_x, depth_y, panel_info->panel);
        }
        if (depth <= 0.0f) {
            return glm::vec3(Viewport::INVALID_WORLD_POS);
        }

        return projection_viewport.unprojectPixel(
            render_x,
            render_y,
            depth,
            render_settings.focal_length_mm,
            render_settings.orthographic,
            render_settings.ortho_scale);
    }

    bool AlignPickPointOperator::applyAlignment(OperatorContext& ctx) {
        if (picked_points_.size() != 3) {
            return false;
        }

        const auto target_ids = resolveAlignmentTargets(ctx);
        if (target_ids.empty()) {
            LOG_WARN("3-point alignment has no transformable selected target");
            setStatus(lichtfeld::Strings::Align::STATUS_NO_TARGET, 2.0);
            return false;
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
            setStatus(lichtfeld::Strings::Align::STATUS_COLINEAR, 2.0);
            return false;
        }
        glm::vec3 normal = cross_v / cross_len;
        const glm::vec3 center = (p0 + p1 + p2) / 3.0f;

        constexpr glm::vec3 kTargetUp(0.0f, 1.0f, 0.0f);
        const glm::vec3 camera_pos = resolvePickPanelCameraPosition();
        faceNormalTowardCamera(normal, center, camera_pos);

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
        return true;
    }

    void registerAlignOperators() {
        operators().registerOperator(BuiltinOp::AlignPickPoint, AlignPickPointOperator::DESCRIPTOR,
                                     [] { return std::make_unique<AlignPickPointOperator>(); });
    }

    void unregisterAlignOperators() {
        operators().unregisterOperator(BuiltinOp::AlignPickPoint);
    }

} // namespace lfs::vis::op

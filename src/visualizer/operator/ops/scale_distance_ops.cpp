/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "scale_distance_ops.hpp"
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
#include "rendering/point_cloud_vulkan_renderer.hpp"
#include "rendering/rendering_manager.hpp"
#include "scene/scene_manager.hpp"
#include "visualizer/gui_capabilities.hpp"
#include "visualizer/scene_coordinate_utils.hpp"
#include "visualizer_impl.hpp"
#include <algorithm>
#include <cmath>
#include <glm/gtc/matrix_transform.hpp>
#include <limits>
#include <unordered_set>
#include <utility>

namespace lfs::vis::op {

    namespace {
        constexpr double kClickDragThresholdPx = 4.0;
        constexpr double kMarkerHitRadiusPx = 8.0;

        [[nodiscard]] core::NodeId resolveScaleTargetId(const core::Scene& scene,
                                                        const core::SceneNode& node) {
            core::NodeId target_id = core::NULL_NODE;
            if ((node.type == core::NodeType::CROPBOX ||
                 node.type == core::NodeType::ELLIPSOID) &&
                node.parent_id != core::NULL_NODE) {
                const auto* const parent = scene.getNodeById(node.parent_id);
                if (parent && cap::isAlignTransformTargetType(parent->type)) {
                    target_id = parent->id;
                }
            }
            if (target_id == core::NULL_NODE && cap::isAlignTransformTargetType(node.type)) {
                target_id = node.id;
            }
            if (target_id == core::NULL_NODE) {
                return core::NULL_NODE;
            }
            const auto* const target = scene.getNodeById(target_id);
            if (!target || static_cast<bool>(target->locked)) {
                return core::NULL_NODE;
            }
            return target_id;
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

        [[nodiscard]] std::vector<core::NodeId> resolveScaleTargets(const OperatorContext& ctx) {
            const auto& scene = ctx.scene().getScene();
            const auto selected_names = ctx.selectedNodes();

            std::vector<core::NodeId> target_ids;
            std::unordered_set<core::NodeId> seen;

            for (const auto& name : selected_names) {
                const auto* node = scene.getNode(name);
                if (!node)
                    continue;
                const auto target_id = resolveScaleTargetId(scene, *node);
                if (target_id != core::NULL_NODE && seen.insert(target_id).second)
                    target_ids.push_back(target_id);
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

    } // namespace

    const OperatorDescriptor ScaleDistancePickPointOperator::DESCRIPTOR = {
        .builtin_id = BuiltinOp::ScaleDistancePickPoint,
        .python_class_id = {},
        .label = "Scale by Distance",
        .description = "Pick 2 points to define scale distance",
        .icon = "bounds",
        .shortcut = "",
        .flags = OperatorFlags::REGISTER,
        .source = OperatorSource::CPP,
        .poll_deps = PollDependency::SCENE | PollDependency::SELECTION,
    };

    bool ScaleDistancePickPointOperator::poll(const OperatorContext& ctx,
                                              const OperatorProperties* /*props*/) const {
        return !resolveScaleTargets(ctx).empty();
    }

    OperatorResult ScaleDistancePickPointOperator::invoke(OperatorContext& /*ctx*/, OperatorProperties& props) {
        clearAllPoints();
        press_active_ = false;
        press_button_ = -1;
        press_point_index_.reset();
        drag_active_ = false;
        (void)services().takeScaleDistanceUiAction();
        pick_button_ = props.get_or<int>("button", static_cast<int>(lfs::vis::input::AppMouseButton::LEFT));

        press_active_ = true;
        press_button_ = pick_button_;
        press_pos_ = {
            props.get_or<double>("x", 0.0),
            props.get_or<double>("y", 0.0),
        };

        if (services().renderingOrNull()) {
            services().renderingOrNull()->markDirty(DirtyFlag::OVERLAY, lfs::vis::FrameReason::Overlay);
        }

        return OperatorResult::RUNNING_MODAL;
    }

    OperatorResult ScaleDistancePickPointOperator::handlePendingUiAction(OperatorContext& ctx) {
        float target_distance = 0.0f;
        switch (services().takeScaleDistanceUiAction(&target_distance)) {
        case Services::ScaleDistanceUiAction::Apply:
            if (picked_points_.size() != 2) {
                return OperatorResult::RUNNING_MODAL;
            }
            if (applyScale(ctx, target_distance)) {
                clearAllPoints();
                return OperatorResult::FINISHED;
            }
            return OperatorResult::RUNNING_MODAL;

        case Services::ScaleDistanceUiAction::Clear:
            clearAllPoints();
            return OperatorResult::RUNNING_MODAL;

        default:
            return OperatorResult::RUNNING_MODAL;
        }
    }

    OperatorResult ScaleDistancePickPointOperator::modal(OperatorContext& ctx, OperatorProperties& /*props*/) {
        resolvePendingPoint();

        OperatorResult ui_result = OperatorResult::RUNNING_MODAL;
        if (services().hasScaleDistanceUiAction()) {
            ui_result = handlePendingUiAction(ctx);
            if (ui_result != OperatorResult::RUNNING_MODAL) {
                return ui_result;
            }
        }

        const auto* event = ctx.event();
        if (!event) {
            return OperatorResult::RUNNING_MODAL;
        }

        if (event->type == ModalEvent::Type::NONE) {
            return ui_result;
        }

        if (event->type == ModalEvent::Type::MOUSE_MOVE) {
            const auto* mm = event->as<MouseMoveEvent>();
            if (!mm) {
                return OperatorResult::PASS_THROUGH;
            }

            if (event->over_gui && !drag_active_) {
                return OperatorResult::PASS_THROUGH;
            }

            if (!drag_active_ && press_active_ && press_point_index_ &&
                press_button_ == pick_button_) {
                if (glm::length(mm->position - press_pos_) > kClickDragThresholdPx) {
                    drag_active_ = true;
                    selected_point_ = press_point_index_;
                    services().setScaleDistanceSelectedPoint(selected_point_);
                }
            }

            if (drag_active_ && press_point_index_ &&
                static_cast<size_t>(*press_point_index_) < picked_points_.size()) {
                (void)applyPointSample(sampleScreenPoint(mm->position), press_point_index_);
                return OperatorResult::RUNNING_MODAL;
            }
            return OperatorResult::PASS_THROUGH;
        }

        if (event->type == ModalEvent::Type::MOUSE_SCROLL) {
            return OperatorResult::PASS_THROUGH;
        }

        if (event->type == ModalEvent::Type::MOUSE_BUTTON) {
            const auto* mb = event->as<MouseButtonEvent>();
            if (!mb) {
                return OperatorResult::RUNNING_MODAL;
            }

            const bool is_pick_button = mb->button == pick_button_;

            if (mb->action == lfs::vis::input::ACTION_PRESS) {
                if (event->over_gui) {
                    return OperatorResult::PASS_THROUGH;
                }
                if (!is_pick_button) {
                    return OperatorResult::PASS_THROUGH;
                }
                press_active_ = true;
                press_button_ = mb->button;
                press_pos_ = mb->position;
                drag_active_ = false;
                press_point_index_ = hitTestPoint(mb->position.x, mb->position.y);
                return OperatorResult::RUNNING_MODAL;
            }

            if (mb->action == lfs::vis::input::ACTION_RELEASE) {
                if (press_active_ && mb->button == press_button_) {
                    const bool was_drag = drag_active_;
                    const auto hit_index = press_point_index_;
                    press_active_ = false;
                    press_button_ = -1;
                    press_point_index_.reset();
                    drag_active_ = false;

                    if (was_drag) {
                        return OperatorResult::RUNNING_MODAL;
                    }

                    if (hit_index) {
                        selected_point_ = hit_index;
                        services().setScaleDistanceSelectedPoint(selected_point_);
                        return OperatorResult::RUNNING_MODAL;
                    }

                    if (tryPlacePoint(mb->position.x, mb->position.y)) {
                        return OperatorResult::RUNNING_MODAL;
                    }
                }
                return OperatorResult::PASS_THROUGH;
            }
        }

        // Keys arrive as key map actions, so every control follows the user's bindings.
        if (event->type == ModalEvent::Type::ACTION) {
            const auto* action = event->as<ActionEvent>();
            if (!action) {
                return OperatorResult::PASS_THROUGH;
            }

            switch (action->action) {
            case lfs::vis::input::Action::CANCEL_POLYGON:
                if (selected_point_) {
                    selected_point_.reset();
                    services().clearScaleDistanceSelectedPoint();
                } else if (!picked_points_.empty()) {
                    clearAllPoints();
                } else {
                    cancel(ctx);
                    return OperatorResult::CANCELLED;
                }
                return OperatorResult::RUNNING_MODAL;

            case lfs::vis::input::Action::REMOVE_LAST_POINT:
                removeLastPoint();
                return OperatorResult::RUNNING_MODAL;

            case lfs::vis::input::Action::DELETE_SELECTED:
                if (selected_point_) {
                    removeSelectedPoint();
                } else {
                    removeLastPoint();
                }
                return OperatorResult::RUNNING_MODAL;

            default:
                break;
            }
            return OperatorResult::PASS_THROUGH;
        }

        return OperatorResult::PASS_THROUGH;
    }

    void ScaleDistancePickPointOperator::cancel(OperatorContext& /*ctx*/) {
        clearAllPoints();
        press_active_ = false;
        press_button_ = -1;
        press_point_index_.reset();
        drag_active_ = false;
    }

    void ScaleDistancePickPointOperator::setStatus(const char* locale_key, const double duration_seconds) const {
        if (!locale_key) {
            services().clearScaleDistanceStatusMessage();
            return;
        }
        services().setScaleDistanceStatusMessage(LOC(locale_key), duration_seconds);
    }

    void ScaleDistancePickPointOperator::syncPickedPointsToServices() {
        services().setScaleDistancePickedPoints(picked_points_);
        if (auto* rm = services().renderingOrNull()) {
            rm->markDirty(DirtyFlag::OVERLAY, lfs::vis::FrameReason::Overlay);
        }
    }

    void ScaleDistancePickPointOperator::clearAllPoints() {
        picked_points_.clear();
        selected_point_.reset();
        pending_pick_.reset();
        services().clearScaleDistanceSelectedPoint();
        services().clearScaleDistancePickedPoints();
        if (auto* rm = services().renderingOrNull()) {
            rm->markDirty(DirtyFlag::OVERLAY, lfs::vis::FrameReason::Overlay);
        }
    }

    void ScaleDistancePickPointOperator::removeLastPoint() {
        pending_pick_.reset();
        press_active_ = false;
        press_button_ = -1;
        press_point_index_.reset();
        drag_active_ = false;
        if (picked_points_.empty()) {
            return;
        }
        picked_points_.pop_back();
        if (selected_point_ && static_cast<size_t>(*selected_point_) >= picked_points_.size()) {
            selected_point_.reset();
            services().clearScaleDistanceSelectedPoint();
        }
        syncPickedPointsToServices();
    }

    void ScaleDistancePickPointOperator::removeSelectedPoint() {
        pending_pick_.reset();
        press_active_ = false;
        press_button_ = -1;
        press_point_index_.reset();
        drag_active_ = false;
        if (!selected_point_ || static_cast<size_t>(*selected_point_) >= picked_points_.size()) {
            return;
        }
        picked_points_.erase(picked_points_.begin() + *selected_point_);
        selected_point_.reset();
        services().clearScaleDistanceSelectedPoint();
        syncPickedPointsToServices();
    }

    bool ScaleDistancePickPointOperator::tryPlacePoint(const double x, const double y) {
        if (picked_points_.size() >= 2) {
            selected_point_.reset();
            services().clearScaleDistanceSelectedPoint();
            return false;
        }
        return requestPoint({.screen_position = {x, y}, .point_index = std::nullopt});
    }

    bool ScaleDistancePickPointOperator::requestPoint(const PendingPick& pick) {
        pending_pick_ = pick;
        resolvePendingPoint();
        return pending_pick_.has_value();
    }

    bool ScaleDistancePickPointOperator::applyPointSample(const PointSample& sample, std::optional<int> index) {
        if (sample.pending || !Viewport::isValidWorldPosition(sample.world)) {
            return false;
        }
        pick_panel_ = sample.panel;
        if (index) {
            if (*index < 0 || static_cast<size_t>(*index) >= picked_points_.size()) {
                return false;
            }
            picked_points_[static_cast<size_t>(*index)] = sample.world;
        } else {
            if (picked_points_.size() >= 2) {
                return false;
            }
            picked_points_.push_back(sample.world);
        }
        syncPickedPointsToServices();
        return true;
    }

    void ScaleDistancePickPointOperator::resolvePendingPoint() {
        if (!pending_pick_) {
            return;
        }
        const auto sample = sampleScreenPoint(pending_pick_->screen_position);
        if (sample.pending) {
            return;
        }
        const auto pick = *pending_pick_;
        pending_pick_.reset();
        if (!applyPointSample(sample, pick.point_index)) {
            setStatus(lichtfeld::Strings::ScaleDistance::STATUS_NO_SURFACE, 1.5);
        }
    }

    std::optional<int> ScaleDistancePickPointOperator::hitTestPoint(const double x, const double y) const {
        if (picked_points_.empty()) {
            return std::nullopt;
        }
        auto* rm = services().renderingOrNull();
        auto* gm = services().guiOrNull();
        if (!rm || !gm || !gm->getViewer()) {
            return std::nullopt;
        }

        const auto viewport_pos = gm->getViewportPos();
        const auto viewport_size = gm->getViewportSize();
        const glm::vec2 click_screen(static_cast<float>(x), static_cast<float>(y));
        const auto panel_info = rm->resolveViewerPanel(
            gm->getViewer()->getViewport(),
            viewport_pos,
            viewport_size,
            click_screen);
        if (!panel_info || !panel_info->valid()) {
            return std::nullopt;
        }

        const float screen_scale_x = panel_info->width / static_cast<float>(std::max(panel_info->render_width, 1));
        const float screen_scale_y = panel_info->height / static_cast<float>(std::max(panel_info->render_height, 1));

        Viewport projection_viewport = *panel_info->viewport;
        projection_viewport.windowSize = {panel_info->render_width, panel_info->render_height};
        const auto render_settings = rm->getSettings();
        const float ortho_scale = alignPanelOrthoScale(projection_viewport, render_settings);

        std::optional<int> best;
        float best_dist = std::numeric_limits<float>::max();

        for (size_t i = 0; i < picked_points_.size(); ++i) {
            const auto projected = rendering::projectWorldPoint(
                projection_viewport.camera.R,
                projection_viewport.camera.t,
                projection_viewport.windowSize,
                picked_points_[i],
                render_settings.focal_length_mm,
                render_settings.orthographic,
                ortho_scale);
            if (!projected) {
                continue;
            }
            const glm::vec2 screen_pos{
                panel_info->x + projected->x * screen_scale_x,
                panel_info->y + projected->y * screen_scale_y,
            };
            const float marker_radius = alignMarkerScreenRadius(
                picked_points_[i],
                projection_viewport.getViewMatrix(),
                projection_viewport.getProjectionMatrix(render_settings.focal_length_mm),
                static_cast<float>(projection_viewport.windowSize.y),
                render_settings.orthographic,
                ortho_scale,
                screen_scale_x,
                screen_scale_y);
            const float hit_radius = std::max(static_cast<float>(kMarkerHitRadiusPx), marker_radius);
            const float dist = glm::length(screen_pos - click_screen);
            if (dist <= hit_radius && dist < best_dist) {
                best_dist = dist;
                best = static_cast<int>(i);
            }
        }
        return best;
    }

    ScaleDistancePickPointOperator::PointSample ScaleDistancePickPointOperator::sampleScreenPoint(const glm::dvec2 position) const {
        auto* rm = services().renderingOrNull();
        auto* gm = services().guiOrNull();
        if (!rm || !gm || !gm->getViewer()) {
            return {};
        }

        const auto viewport_pos = gm->getViewportPos();
        const auto viewport_size = gm->getViewportSize();

        const auto panel_info = rm->resolveViewerPanel(
            gm->getViewer()->getViewport(),
            viewport_pos,
            viewport_size,
            glm::vec2(static_cast<float>(position.x), static_cast<float>(position.y)));
        if (!panel_info || !panel_info->valid()) {
            return {};
        }

        const float scale_x = static_cast<float>(panel_info->render_width) / panel_info->width;
        const float scale_y = static_cast<float>(panel_info->render_height) / panel_info->height;
        const float render_x = (static_cast<float>(position.x) - panel_info->x) * scale_x;
        const float render_y = (static_cast<float>(position.y) - panel_info->y) * scale_y;

        Viewport projection_viewport = *panel_info->viewport;
        projection_viewport.windowSize = {panel_info->render_width, panel_info->render_height};
        const auto render_settings = rm->getSettings();

        const int depth_x = static_cast<int>(render_x);
        const int depth_y = static_cast<int>(render_y);

        const float depth = rm->getDepthAtPixel(depth_x, depth_y, panel_info->panel, true);
        if (depth == PointCloudVulkanRenderer::kDepthSamplePending)
            return {.panel = panel_info->panel, .pending = true};
        if (!std::isfinite(depth) || depth <= 0.0f || depth >= 1e9f) {
            return {};
        }

        return {.world = projection_viewport.unprojectPixel(
                    render_x,
                    render_y,
                    depth,
                    render_settings.focal_length_mm,
                    render_settings.orthographic,
                    alignPanelOrthoScale(projection_viewport, render_settings)),
                .panel = panel_info->panel};
    }

    bool ScaleDistancePickPointOperator::applyScale(OperatorContext& ctx, const float target_distance) {
        if (picked_points_.size() != 2) {
            return false;
        }

        const auto transform = computeScaleDistanceTransform(picked_points_[0], picked_points_[1], target_distance);
        if (!transform) {
            setStatus(lichtfeld::Strings::ScaleDistance::STATUS_INVALID_DISTANCE, 2.0);
            return false;
        }

        const auto target_ids = resolveScaleTargets(ctx);
        if (target_ids.empty()) {
            setStatus(lichtfeld::Strings::ScaleDistance::STATUS_NO_TARGET, 2.0);
            return false;
        }

        auto& scene = ctx.scene().getScene();
        std::vector<glm::mat4> originals;
        std::vector<std::string> names;

        for (const auto id : target_ids) {
            const auto* node = scene.getNodeById(id);
            if (!node || node->locked) {
                continue;
            }
            names.push_back(node->name);
            originals.push_back(node->transform());
        }

        if (names.empty()) {
            return false;
        }

        auto entry = std::make_unique<SceneSnapshot>(ctx.scene(), "transform.scale_distance");
        if (!entry->captureTransformsBefore(names, originals)) {
            return false;
        }

        for (const auto id : target_ids) {
            const auto* node = scene.getNodeById(id);
            if (!node || node->locked) {
                continue;
            }
            const glm::mat4 world_orig = vis::scene_coords::nodeVisualizerWorldTransform(scene, id);
            const glm::mat4 world_new = *transform * world_orig;
            const auto local = vis::scene_coords::nodeLocalTransformFromVisualizerWorld(scene, id, world_new);
            if (local) {
                ctx.scene().setNodeTransform(node->name, *local);
            }
        }

        entry->captureAfter();
        pushSceneSnapshotIfChanged(std::move(entry));

        clearAllPoints();

        if (auto* rm = services().renderingOrNull()) {
            rm->markDirty(DirtyFlag::SPLATS | DirtyFlag::MESH | DirtyFlag::OVERLAY, lfs::vis::FrameReason::SceneChange);
        }

        return true;
    }

    std::optional<glm::mat4> computeScaleDistanceTransform(
        const glm::vec3& p0, const glm::vec3& p1, const float target_distance) {
        if (!std::isfinite(target_distance) || target_distance <= 1e-6f) {
            return std::nullopt;
        }

        const float measured = glm::length(p1 - p0);
        if (!std::isfinite(measured) || measured <= 1e-6f) {
            return std::nullopt;
        }

        const float factor = target_distance / measured;
        if (!std::isfinite(factor) || factor <= 1e-6f) {
            return std::nullopt;
        }

        const glm::vec3 pivot = (p0 + p1) * 0.5f;
        const glm::mat4 to_pivot = glm::translate(glm::mat4(1.0f), -pivot);
        const glm::mat4 scale = glm::scale(glm::mat4(1.0f), glm::vec3(factor));
        const glm::mat4 from_pivot = glm::translate(glm::mat4(1.0f), pivot);

        return from_pivot * scale * to_pivot;
    }

    void registerScaleDistanceOperators() {
        operators().registerOperator(BuiltinOp::ScaleDistancePickPoint, ScaleDistancePickPointOperator::DESCRIPTOR,
                                     [] { return std::make_unique<ScaleDistancePickPointOperator>(); });
    }

    void unregisterScaleDistanceOperators() {
        operators().unregisterOperator(BuiltinOp::ScaleDistancePickPoint);
    }

} // namespace lfs::vis::op

/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "point_pick_operator.hpp"
#include "core/event_bridge/localization_manager.hpp"
#include "core/services.hpp"
#include "gui/gui_manager.hpp"
#include "input/key_codes.hpp"
#include "rendering/coordinate_conventions.hpp"
#include "rendering/point_cloud_vulkan_renderer.hpp"
#include "rendering/rendering_manager.hpp"
#include "scene/scene_manager.hpp"
#include "visualizer/gui_capabilities.hpp"
#include "visualizer_impl.hpp"
#include <algorithm>
#include <cmath>
#include <limits>
#include <unordered_set>
#include <utility>

namespace lfs::vis::op {

    namespace {
        constexpr double kClickDragThresholdPx = 4.0;
        constexpr double kMarkerHitRadiusPx = 8.0;

    } // namespace

    [[nodiscard]] core::NodeId resolvePointPickTargetId(const core::Scene& scene,
                                                        const core::SceneNode& node, const bool include_crop_polygon) {
        core::NodeId target_id = core::NULL_NODE;
        if ((node.type == core::NodeType::CROPBOX ||
             node.type == core::NodeType::ELLIPSOID ||
             (include_crop_polygon && node.type == core::NodeType::CROP_POLYGON)) &&
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

    namespace {
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

    } // namespace

    std::vector<core::NodeId> resolvePointPickTargets(const OperatorContext& ctx, const bool include_crop_polygon) {
        const auto& scene = ctx.scene().getScene();
        const auto selected_names = ctx.selectedNodes();

        std::vector<core::NodeId> target_ids;
        std::unordered_set<core::NodeId> seen;

        for (const auto& name : selected_names) {
            const auto* node = scene.getNode(name);
            if (!node)
                continue;
            const auto target_id = resolvePointPickTargetId(scene, *node, include_crop_polygon);
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

    OperatorResult PointPickOperator::invoke(OperatorContext& /*ctx*/, OperatorProperties& props) {
        clearAllPoints();
        press_active_ = false;
        press_button_ = -1;
        press_point_index_.reset();
        drag_active_ = false;
        (void)state_.takeUiAction();
        pick_button_ = props.get_or<int>("button", static_cast<int>(lfs::vis::input::AppMouseButton::LEFT));

        // First press enters the modal; the point is placed on release if the
        // gesture is a click (see modal click-vs-drag handling).
        press_active_ = true;
        press_button_ = pick_button_;
        press_pos_ = {
            props.get_or<double>("x", 0.0),
            props.get_or<double>("y", 0.0),
        };

        markOverlayDirty();

        return OperatorResult::RUNNING_MODAL;
    }

    OperatorResult PointPickOperator::modal(OperatorContext& ctx, OperatorProperties& /*props*/) {
        resolvePendingPoint();
        if (const auto result = prepareModal(ctx)) {
            return *result;
        }

        OperatorResult ui_result = OperatorResult::RUNNING_MODAL;
        if (state_.hasUiAction()) {
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

        // Navigation remains available while previewing; point editing uses the original scene.
        if (pointEditingBlocked() &&
            (event->type == ModalEvent::Type::MOUSE_MOVE ||
             event->type == ModalEvent::Type::MOUSE_BUTTON)) {
            return OperatorResult::PASS_THROUGH;
        }

        if (event->type == ModalEvent::Type::MOUSE_MOVE) {
            const auto* mm = event->as<MouseMoveEvent>();
            if (!mm) {
                return OperatorResult::PASS_THROUGH;
            }

            if (event->over_gui && !drag_active_) {
                return OperatorResult::PASS_THROUGH;
            }

            // Promote a marker press into a surface drag once past click threshold.
            if (!drag_active_ && press_active_ && press_point_index_ &&
                press_button_ == pick_button_) {
                if (glm::length(mm->position - press_pos_) > kClickDragThresholdPx) {
                    drag_active_ = true;
                    selected_point_ = press_point_index_;
                    state_.setSelectedPoint(selected_point_);
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
                if (is_pick_button) {
                    press_active_ = true;
                    press_button_ = mb->button;
                    press_pos_ = mb->position;
                    press_point_index_.reset();
                    drag_active_ = false;

                    press_point_index_ = hitTestPoint(mb->position.x, mb->position.y);
                    if (press_point_index_) {
                        selected_point_ = press_point_index_;
                        state_.setSelectedPoint(selected_point_);
                        if (services().renderingOrNull()) {
                            services().renderingOrNull()->markDirty(DirtyFlag::OVERLAY, lfs::vis::FrameReason::Overlay);
                        }
                    }
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
            const bool was_dragging = drag_active_;
            drag_active_ = false;

            if (is_pick_button && press_point_index_ &&
                (was_dragging || move_dist > kClickDragThresholdPx)) {
                (void)requestPoint({mb->position, press_point_index_});
                press_point_index_.reset();
                return OperatorResult::RUNNING_MODAL;
            }

            if (move_dist > kClickDragThresholdPx) {
                press_point_index_.reset();
                return OperatorResult::PASS_THROUGH;
            }

            if (is_pick_button) {
                // Click on existing marker: select only.
                if (press_point_index_) {
                    selected_point_ = press_point_index_;
                    state_.setSelectedPoint(selected_point_);
                    if (services().renderingOrNull()) {
                        services().renderingOrNull()->markDirty(DirtyFlag::OVERLAY, lfs::vis::FrameReason::Overlay);
                    }
                    press_point_index_.reset();
                    return OperatorResult::RUNNING_MODAL;
                }

                // Click near a marker (release may differ slightly): select.
                if (const auto hit = hitTestPoint(press_pos_.x, press_pos_.y)) {
                    selected_point_ = hit;
                    state_.setSelectedPoint(selected_point_);
                    if (services().renderingOrNull()) {
                        services().renderingOrNull()->markDirty(DirtyFlag::OVERLAY, lfs::vis::FrameReason::Overlay);
                    }
                    return OperatorResult::RUNNING_MODAL;
                }

                if (picked_points_.size() >= max_points_) {
                    return OperatorResult::RUNNING_MODAL;
                }
                (void)tryPlacePoint(press_pos_.x, press_pos_.y);
                return OperatorResult::RUNNING_MODAL;
            }

            return OperatorResult::PASS_THROUGH;
        }

        // Keys arrive as key map actions, so every control follows the user's bindings.
        if (event->type == ModalEvent::Type::ACTION) {
            const auto* action = event->as<ActionEvent>();
            if (!action) {
                return OperatorResult::PASS_THROUGH;
            }

            switch (action->action) {
            case lfs::vis::input::Action::CANCEL_POLYGON:
                return OperatorResult::CANCELLED;

            case lfs::vis::input::Action::REMOVE_LAST_POINT:
                beforePointEdit(ctx);
                removeLastPoint();
                return OperatorResult::RUNNING_MODAL;

            case lfs::vis::input::Action::DELETE_SELECTED:
                beforePointEdit(ctx);
                removeSelectedPoint();
                return OperatorResult::RUNNING_MODAL;

            default:
                return handleToolAction(ctx, action->action);
            }
        }

        return OperatorResult::PASS_THROUGH;
    }

    OperatorResult PointPickOperator::handlePendingUiAction(OperatorContext& ctx) {
        float value = 0.0f;
        const auto action = state_.takeUiAction(&value);
        switch (action) {
        case PointPickState::UiAction::Apply:
            if (applyPickedPoints(ctx, value)) {
                clearAllPoints();
                return OperatorResult::FINISHED;
            }
            break;
        case PointPickState::UiAction::Clear:
            beforePointEdit(ctx);
            clearAllPoints();
            break;
        default:
            handleToolUiAction(ctx, action);
            break;
        }
        return OperatorResult::RUNNING_MODAL;
    }

    void PointPickOperator::cancel(OperatorContext& ctx) {
        beforePointEdit(ctx);
        clearAllPoints();
        press_active_ = false;
        drag_active_ = false;
        markOverlayDirty();
    }

    void PointPickOperator::markOverlayDirty() const {
        if (auto* rm = services().renderingOrNull()) {
            rm->markDirty(DirtyFlag::OVERLAY, lfs::vis::FrameReason::Overlay);
        }
    }

    void PointPickOperator::setStatus(const char* locale_key, const double duration_seconds) const {
        state_.setStatusMessage(LOC(locale_key), duration_seconds);
        markOverlayDirty();
    }

    void PointPickOperator::syncPickedPointsToServices() {
        state_.setPickedPoints(picked_points_);
        state_.setSelectedPoint(selected_point_);
        markOverlayDirty();
    }

    void PointPickOperator::clearAllPoints() {
        pending_pick_.reset();
        press_active_ = false;
        picked_points_.clear();
        selected_point_.reset();
        pick_panel_.reset();
        press_point_index_.reset();
        drag_active_ = false;
        state_.clearPickedPoints();
        markOverlayDirty();
    }

    void PointPickOperator::removeLastPoint() {
        pending_pick_.reset();
        press_active_ = false;
        press_point_index_.reset();
        drag_active_ = false;
        if (picked_points_.empty()) {
            return;
        }
        const int last = static_cast<int>(picked_points_.size()) - 1;
        if (selected_point_ && *selected_point_ >= last) {
            selected_point_.reset();
        }
        picked_points_.pop_back();
        if (picked_points_.empty()) {
            pick_panel_.reset();
            selected_point_.reset();
        }
        syncPickedPointsToServices();
    }

    void PointPickOperator::removeSelectedPoint() {
        pending_pick_.reset();
        press_active_ = false;
        press_point_index_.reset();
        drag_active_ = false;
        if (!selected_point_) {
            return;
        }
        const int idx = *selected_point_;
        if (idx < 0 || static_cast<size_t>(idx) >= picked_points_.size()) {
            selected_point_.reset();
            state_.clearSelectedPoint();
            return;
        }
        picked_points_.erase(picked_points_.begin() + idx);
        selected_point_.reset();
        if (picked_points_.empty()) {
            pick_panel_.reset();
        }
        syncPickedPointsToServices();
    }

    bool PointPickOperator::tryPlacePoint(const double x, const double y) {
        if (picked_points_.size() >= max_points_) {
            return false;
        }

        if (hitTestPoint(x, y)) {
            return false;
        }

        return requestPoint({{x, y}, std::nullopt});
    }

    bool PointPickOperator::requestPoint(const PendingPick& pick) {
        pending_pick_.reset();
        const auto sample = sampleScreenPoint(pick.screen_position);
        if (sample.pending) {
            pending_pick_ = pick;
            return false;
        }
        if (!Viewport::isValidWorldPosition(sample.world)) {
            setStatus(no_surface_key_);
            return false;
        }
        return applyPointSample(sample, pick.point_index);
    }

    bool PointPickOperator::applyPointSample(const PointSample& sample, std::optional<int> index) {
        if (!Viewport::isValidWorldPosition(sample.world))
            return false;
        if (index) {
            if (*index < 0 || static_cast<size_t>(*index) >= picked_points_.size())
                return false;
            picked_points_[*index] = sample.world;
        } else {
            if (picked_points_.size() >= max_points_)
                return false;
            picked_points_.push_back(sample.world);
            selected_point_ = static_cast<int>(picked_points_.size()) - 1;
        }
        if (!pick_panel_)
            pick_panel_ = sample.panel;
        pointSampleApplied();
        syncPickedPointsToServices();
        return true;
    }

    void PointPickOperator::resolvePendingPoint() {
        if (const auto pick = std::exchange(pending_pick_, std::nullopt))
            (void)requestPoint(*pick);
    }

    std::optional<int> PointPickOperator::hitTestPoint(const double x, const double y) const {
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
        const auto panel_info = rm->resolveViewerPanel(
            gm->getViewer()->getViewport(),
            viewport_pos,
            viewport_size,
            glm::vec2(static_cast<float>(x), static_cast<float>(y)));
        if (!panel_info || !panel_info->valid()) {
            return std::nullopt;
        }

        const auto render_settings = rm->getSettings();
        Viewport projection_viewport = *panel_info->viewport;
        projection_viewport.windowSize = {panel_info->render_width, panel_info->render_height};

        const float screen_scale_x = panel_info->width / static_cast<float>(std::max(panel_info->render_width, 1));
        const float screen_scale_y = panel_info->height / static_cast<float>(std::max(panel_info->render_height, 1));
        const glm::vec2 click_screen(static_cast<float>(x), static_cast<float>(y));
        const float ortho_scale = pointPickPanelOrthoScale(projection_viewport, render_settings);

        std::optional<int> best;
        float best_dist = std::numeric_limits<float>::max();

        for (size_t i = 0; i < picked_points_.size(); ++i) {
            const auto projected = lfs::rendering::projectWorldPoint(
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
            const float marker_radius = pointPickMarkerScreenRadius(
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

    PointPickOperator::PointSample PointPickOperator::sampleScreenPoint(const glm::dvec2 position) const {
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

        // Hover, placement and marker dragging all follow the displayed surface.
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
                    pointPickPanelOrthoScale(projection_viewport, render_settings)),
                .panel = panel_info->panel};
    }

    float pointPickMarkerScreenRadius(const glm::vec3& world_pos,
                                      const glm::mat4& view,
                                      const glm::mat4& projection,
                                      const float window_height,
                                      const bool orthographic,
                                      const float ortho_scale,
                                      const float screen_scale_x,
                                      const float screen_scale_y) {
        const glm::vec4 view_pos = view * glm::vec4(world_pos, 1.0f);
        const float depth = -view_pos.z;
        if (depth <= 0.0f) {
            return 0.0f;
        }

        float radius_render = 0.0f;
        if (orthographic) {
            if (!std::isfinite(ortho_scale) || ortho_scale <= 0.0f) {
                return 0.0f;
            }
            radius_render = kPointPickMarkerWorldRadius * ortho_scale;
        } else {
            radius_render = (kPointPickMarkerWorldRadius * projection[1][1] * window_height) / (2.0f * depth);
        }
        return glm::clamp(radius_render * glm::min(screen_scale_x, screen_scale_y), 5.0f, 50.0f);
    }

} // namespace lfs::vis::op

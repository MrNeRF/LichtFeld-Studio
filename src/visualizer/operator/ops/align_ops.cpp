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
#include "visualizer/gui_capabilities.hpp"
#include "visualizer/scene_coordinate_utils.hpp"
#include "visualizer_impl.hpp"
#include <algorithm>
#include <cmath>
#include <glm/gtc/constants.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <limits>
#include <unordered_set>

namespace lfs::vis::op {

    namespace {
        constexpr double kClickDragThresholdPx = 4.0;
        constexpr double kMarkerHitRadiusPx = 8.0;

        [[nodiscard]] core::NodeId resolveAlignTargetId(const core::Scene& scene,
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

        [[nodiscard]] std::vector<core::NodeId> resolveAlignmentTargets(const OperatorContext& ctx) {
            const auto& scene = ctx.scene().getScene();
            const auto selected_names = ctx.selectedNodes();

            std::vector<core::NodeId> target_ids;
            std::unordered_set<core::NodeId> seen;

            for (const auto& name : selected_names) {
                const auto* node = scene.getNode(name);
                if (!node)
                    continue;
                const auto target_id = resolveAlignTargetId(scene, *node);
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

    const OperatorDescriptor AlignPickPointOperator::DESCRIPTOR = {
        .builtin_id = BuiltinOp::AlignPickPoint,
        .python_class_id = {},
        .label = "Align to Ground",
        .description = "Pick 3 points to define ground plane",
        .icon = "align",
        .shortcut = "",
        .flags = OperatorFlags::REGISTER,
        .source = OperatorSource::CPP,
        .poll_deps = PollDependency::SCENE,
    };

    bool AlignPickPointOperator::poll(const OperatorContext& ctx,
                                      const OperatorProperties* /*props*/) const {
        return ctx.scene().getScene().getTotalGaussianCount() > 0;
    }

    OperatorResult AlignPickPointOperator::invoke(OperatorContext& /*ctx*/, OperatorProperties& props) {
        clearAllPoints();
        press_active_ = false;
        press_button_ = -1;
        press_point_index_.reset();
        drag_active_ = false;
        (void)services().takeAlignUiAction();
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

    OperatorResult AlignPickPointOperator::handlePendingUiAction(OperatorContext& ctx) {
        switch (services().takeAlignUiAction()) {
        case Services::AlignUiAction::Apply:
            if (picked_points_.size() != 3 || !pointsAreNonDegenerate(picked_points_)) {
                return OperatorResult::RUNNING_MODAL;
            }
            if (applyAlignment(ctx)) {
                clearAllPoints();
                return OperatorResult::FINISHED;
            }
            return OperatorResult::RUNNING_MODAL;
        case Services::AlignUiAction::TogglePreview:
            if (services().getAlignPreviewEnabled()) {
                restorePreview(ctx);
            } else {
                (void)updatePreview(ctx);
            }
            return OperatorResult::RUNNING_MODAL;
        case Services::AlignUiAction::RefreshPreview:
            if (services().getAlignPreviewEnabled()) {
                (void)updatePreview(ctx);
            }
            return OperatorResult::RUNNING_MODAL;
        case Services::AlignUiAction::Clear:
            restorePreview(ctx);
            clearAllPoints();
            return OperatorResult::RUNNING_MODAL;
        case Services::AlignUiAction::None:
        default:
            return OperatorResult::RUNNING_MODAL;
        }
    }

    OperatorResult AlignPickPointOperator::modal(OperatorContext& ctx, OperatorProperties& /*props*/) {
        if (!preview_targets_.empty()) {
            const auto targets = resolveAlignmentTargets(ctx);
            if (targets.size() != preview_targets_.size() ||
                std::any_of(preview_targets_.begin(), preview_targets_.end(), [&](const auto& saved) {
                    const auto* node = ctx.scene().getScene().getNodeById(saved.id);
                    return !node || node->uuid != saved.uuid ||
                           std::find(targets.begin(), targets.end(), saved.id) == targets.end();
                })) {
                restorePreview(ctx);
                clearAllPoints();
                (void)services().takeAlignUiAction();
                setStatus("align.status_targets_changed", 2.0);
                return OperatorResult::PASS_THROUGH;
            }
        }

        OperatorResult ui_result = OperatorResult::RUNNING_MODAL;
        if (services().hasAlignUiAction()) {
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
        if (services().getAlignPreviewEnabled() &&
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
                    services().setAlignSelectedPoint(selected_point_);
                }
            }

            if (drag_active_ && press_point_index_ &&
                static_cast<size_t>(*press_point_index_) < picked_points_.size()) {
                SplitViewPanelId panel = SplitViewPanelId::Left;
                const glm::vec3 world_pos =
                    unprojectScreenPoint(mm->position.x, mm->position.y, &panel);
                if (Viewport::isValidWorldPosition(world_pos)) {
                    picked_points_[static_cast<size_t>(*press_point_index_)] = world_pos;
                    if (!pick_panel_) {
                        pick_panel_ = panel;
                    }
                    services().setAlignCameraPosition(resolvePickPanelCameraPosition());
                    syncPickedPointsToServices();
                }
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
                        services().setAlignSelectedPoint(selected_point_);
                        if (services().renderingOrNull()) {
                            services().renderingOrNull()->markDirty(DirtyFlag::OVERLAY);
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
                const int idx = *press_point_index_;
                if (idx >= 0 && static_cast<size_t>(idx) < picked_points_.size()) {
                    SplitViewPanelId panel = SplitViewPanelId::Left;
                    const glm::vec3 world_pos =
                        unprojectScreenPoint(mb->position.x, mb->position.y, &panel);
                    if (Viewport::isValidWorldPosition(world_pos)) {
                        picked_points_[static_cast<size_t>(idx)] = world_pos;
                        if (!pick_panel_) {
                            pick_panel_ = panel;
                        }
                        services().setAlignCameraPosition(resolvePickPanelCameraPosition());
                        syncPickedPointsToServices();
                    }
                }
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
                    services().setAlignSelectedPoint(selected_point_);
                    if (services().renderingOrNull()) {
                        services().renderingOrNull()->markDirty(DirtyFlag::OVERLAY);
                    }
                    press_point_index_.reset();
                    return OperatorResult::RUNNING_MODAL;
                }

                // Click near a marker (release may differ slightly): select.
                if (const auto hit = hitTestPoint(press_pos_.x, press_pos_.y)) {
                    selected_point_ = hit;
                    services().setAlignSelectedPoint(selected_point_);
                    if (services().renderingOrNull()) {
                        services().renderingOrNull()->markDirty(DirtyFlag::OVERLAY);
                    }
                    return OperatorResult::RUNNING_MODAL;
                }

                if (picked_points_.size() >= 3) {
                    return OperatorResult::RUNNING_MODAL;
                }
                (void)tryPlacePoint(press_pos_.x, press_pos_.y);
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
                restorePreview(ctx);
                removeLastPoint();
                return OperatorResult::RUNNING_MODAL;
            }

            if (ke->key == lfs::vis::input::KEY_DELETE) {
                restorePreview(ctx);
                removeSelectedPoint();
                return OperatorResult::RUNNING_MODAL;
            }

            if (ke->key == lfs::vis::input::KEY_ENTER ||
                ke->key == lfs::vis::input::KEY_KP_ENTER) {
                if (picked_points_.size() != 3) {
                    return OperatorResult::RUNNING_MODAL;
                }
                if (applyAlignment(ctx)) {
                    clearAllPoints();
                    return OperatorResult::FINISHED;
                }
                return OperatorResult::RUNNING_MODAL;
            }

            return OperatorResult::PASS_THROUGH;
        }

        return OperatorResult::PASS_THROUGH;
    }

    void AlignPickPointOperator::cancel(OperatorContext& ctx) {
        restorePreview(ctx);
        clearAllPoints();
        press_active_ = false;
        drag_active_ = false;
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
        services().setAlignSelectedPoint(selected_point_);
        if (services().renderingOrNull()) {
            services().renderingOrNull()->markDirty(DirtyFlag::OVERLAY);
        }
    }

    void AlignPickPointOperator::clearAllPoints() {
        press_active_ = false;
        picked_points_.clear();
        selected_point_.reset();
        pick_panel_.reset();
        press_point_index_.reset();
        drag_active_ = false;
        services().clearAlignPickedPoints();
        if (services().renderingOrNull()) {
            services().renderingOrNull()->markDirty(DirtyFlag::OVERLAY);
        }
    }

    void AlignPickPointOperator::removeLastPoint() {
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

    void AlignPickPointOperator::removeSelectedPoint() {
        press_active_ = false;
        press_point_index_.reset();
        drag_active_ = false;
        if (!selected_point_) {
            return;
        }
        const int idx = *selected_point_;
        if (idx < 0 || static_cast<size_t>(idx) >= picked_points_.size()) {
            selected_point_.reset();
            services().clearAlignSelectedPoint();
            return;
        }
        picked_points_.erase(picked_points_.begin() + idx);
        selected_point_.reset();
        if (picked_points_.empty()) {
            pick_panel_.reset();
        }
        syncPickedPointsToServices();
    }

    bool AlignPickPointOperator::tryPlacePoint(const double x, const double y) {
        if (picked_points_.size() >= 3) {
            return false;
        }

        if (hitTestPoint(x, y)) {
            return false;
        }

        SplitViewPanelId panel = SplitViewPanelId::Left;
        const glm::vec3 world_pos = unprojectScreenPoint(x, y, &panel);
        if (!Viewport::isValidWorldPosition(world_pos)) {
            setStatus(lichtfeld::Strings::Align::STATUS_NO_SURFACE);
            return false;
        }

        if (!pick_panel_) {
            pick_panel_ = panel;
        }

        services().setAlignCameraPosition(resolvePickPanelCameraPosition());
        picked_points_.push_back(world_pos);
        selected_point_ = static_cast<int>(picked_points_.size()) - 1;
        syncPickedPointsToServices();
        return true;
    }

    std::optional<int> AlignPickPointOperator::hitTestPoint(const double x, const double y) const {
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
                render_settings.ortho_scale);
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
                render_settings.ortho_scale,
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

    glm::vec3 AlignPickPointOperator::unprojectScreenPoint(const double x,
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

        // Hover, placement and marker dragging all follow the displayed surface.
        const float depth = rm->getDepthAtPixel(depth_x, depth_y, panel_info->panel);
        if (!std::isfinite(depth) || depth <= 0.0f || depth >= 1e9f) {
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

    void AlignPickPointOperator::restorePreview(OperatorContext& ctx) {
        auto& scene = ctx.scene().getScene();
        for (const auto& saved : preview_targets_) {
            if (auto* node = scene.getNodeById(saved.id); node && node->uuid == saved.uuid) {
                // Rollback must also restore a target that was locked during preview.
                node->local_transform.set(saved.local, false);
            }
        }
        scene.invalidateTransformCache();
        preview_targets_.clear();
        preview_snap_world_.reset();
        services().setAlignPreviewEnabled(false);
        syncPickedPointsToServices();
        if (auto* rm = services().renderingOrNull()) {
            rm->markDirty(DirtyFlag::SPLATS | DirtyFlag::MESH | DirtyFlag::OVERLAY);
        }
    }

    bool AlignPickPointOperator::updatePreview(OperatorContext& ctx) {
        if (!pointsAreNonDegenerate(picked_points_)) {
            return false;
        }
        auto& scene = ctx.scene().getScene();
        if (preview_targets_.empty()) {
            for (const auto id : resolveAlignmentTargets(ctx)) {
                const auto* node = scene.getNodeById(id);
                preview_targets_.push_back({id, node->uuid, node->transform(),
                                            vis::scene_coords::nodeVisualizerWorldTransform(scene, id)});
            }
            if (preview_targets_.empty()) {
                setStatus(lichtfeld::Strings::Align::STATUS_NO_TARGET, 2.0);
                return false;
            }
            preview_snap_world_ = resolveAlignSnapTargetWorld(ctx.scene());
            preview_camera_ = services().getAlignCameraPosition();
        }
        AlignTransformInputs inputs{picked_points_[0], picked_points_[1], picked_points_[2], preview_camera_, std::nullopt, false};
        if (services().getAlignAxisSnapEnabled()) {
            inputs.snap_node_world = preview_snap_world_;
        }
        inputs.edge_to_world_x = services().getAlignEdgeToAxisEnabled();
        const auto transform = computeAlignTransform(inputs);
        if (!transform) {
            restorePreview(ctx);
            return false;
        }
        for (const auto& saved : preview_targets_) {
            const auto* node = scene.getNodeById(saved.id);
            if (!node || node->uuid != saved.uuid) {
                restorePreview(ctx);
                return false;
            }
            const auto local = vis::scene_coords::nodeLocalTransformFromVisualizerWorld(
                scene, saved.id, *transform * saved.world);
            if (!local) {
                restorePreview(ctx);
                return false;
            }
            ctx.scene().setNodeTransform(node->name, *local);
        }
        auto displayed_points = picked_points_;
        for (auto& point : displayed_points) {
            point = glm::vec3(*transform * glm::vec4(point, 1.0f));
        }
        services().setAlignPickedPoints(std::move(displayed_points));
        services().setAlignPreviewEnabled(true);
        press_active_ = false;
        press_point_index_.reset();
        drag_active_ = false;
        if (auto* rm = services().renderingOrNull()) {
            rm->markDirty(DirtyFlag::SPLATS | DirtyFlag::MESH | DirtyFlag::OVERLAY);
        }
        return true;
    }

    bool AlignPickPointOperator::applyAlignment(OperatorContext& ctx) {
        if (!services().getAlignPreviewEnabled() && !updatePreview(ctx)) {
            return false;
        }

        auto& scene = ctx.scene().getScene();
        std::vector<glm::mat4> originals;
        std::vector<std::string> names;
        for (const auto& saved : preview_targets_) {
            const auto* node = scene.getNodeById(saved.id);
            if (!node || node->uuid != saved.uuid || node->locked) {
                restorePreview(ctx);
                return false;
            }
            names.push_back(node->name);
            originals.push_back(saved.local);
        }
        auto entry = std::make_unique<SceneSnapshot>(ctx.scene(), "transform.align");
        if (!entry->captureTransformsBefore(names, originals)) {
            restorePreview(ctx);
            return false;
        }
        entry->captureAfter();
        pushSceneSnapshotIfChanged(std::move(entry));
        preview_targets_.clear();
        preview_snap_world_.reset();
        services().setAlignPreviewEnabled(false);
        return true;
    }

    void faceNormalTowardCamera(glm::vec3& normal, const glm::vec3& center, const glm::vec3& camera_pos) {
        if (glm::dot(normal, camera_pos - center) < 0.0f) {
            normal = -normal;
        }
    }

    bool pointsAreNonDegenerate(const std::vector<glm::vec3>& points) {
        if (points.size() != 3) {
            return false;
        }
        const glm::vec3 cross_v = glm::cross(points[1] - points[0], points[2] - points[0]);
        const float cross_len = glm::length(cross_v);
        return std::isfinite(cross_len) && cross_len > 1e-6f;
    }

    bool snapAlignNormalToNodeAxes(glm::vec3& normal,
                                   const glm::mat4& node_world,
                                   const float max_degrees) {
        const float max_rad = glm::radians(max_degrees);
        const float min_cos = std::cos(max_rad);

        glm::vec3 best_axis(0.0f);
        float best_abs_dot = -1.0f;
        int best_sign = 1;

        for (int axis = 0; axis < 3; ++axis) {
            glm::vec3 col(node_world[axis]);
            const float len = glm::length(col);
            if (len <= 1e-6f) {
                continue;
            }
            col /= len;
            const float d = glm::dot(normal, col);
            const float abs_d = std::abs(d);
            if (abs_d > best_abs_dot) {
                best_abs_dot = abs_d;
                best_axis = col;
                best_sign = d >= 0.0f ? 1 : -1;
            }
        }

        if (best_abs_dot < min_cos || best_abs_dot < 0.0f) {
            return false;
        }

        normal = best_axis * static_cast<float>(best_sign);
        return true;
    }

    std::optional<glm::mat4> computeAlignTransform(const AlignTransformInputs& in) {
        const glm::vec3 v01 = in.p1 - in.p0;
        const glm::vec3 v02 = in.p2 - in.p0;
        const glm::vec3 cross_v = glm::cross(v01, v02);
        const float cross_len = glm::length(cross_v);
        if (!std::isfinite(cross_len) || cross_len <= 1e-6f) {
            return std::nullopt;
        }

        glm::vec3 normal = cross_v / cross_len;
        const glm::vec3 center = (in.p0 + in.p1 + in.p2) / 3.0f;
        faceNormalTowardCamera(normal, center, in.camera_pos);

        if (in.snap_node_world) {
            (void)snapAlignNormalToNodeAxes(normal, *in.snap_node_world);
        }

        constexpr glm::vec3 kTargetUp(0.0f, 1.0f, 0.0f);
        glm::mat4 rotation(1.0f);
        const float up_dot = glm::dot(normal, kTargetUp);
        const glm::vec3 axis = glm::cross(normal, kTargetUp);
        const float axis_len = glm::length(axis);
        if (axis_len > 1e-6f) {
            const float angle = std::atan2(axis_len, up_dot);
            rotation = glm::rotate(glm::mat4(1.0f), angle, axis / axis_len);
        } else if (up_dot < 0.0f) {
            rotation = glm::rotate(glm::mat4(1.0f), glm::pi<float>(), glm::vec3(1.0f, 0.0f, 0.0f));
        }

        if (in.edge_to_world_x) {
            rotation = alignEdgeToWorldXRotation(rotation, in.p0, in.p1) * rotation;
        }

        const glm::mat4 to_origin = glm::translate(glm::mat4(1.0f), -center);
        const glm::mat4 from_origin =
            glm::translate(glm::mat4(1.0f), center - glm::dot(center, kTargetUp) * kTargetUp);
        return from_origin * rotation * to_origin;
    }

    std::optional<glm::mat4> resolveAlignSnapTargetWorld(const SceneManager& scene_manager) {
        const auto& scene = scene_manager.getScene();
        for (const auto& name : scene_manager.getSelectedNodeNames()) {
            const auto* const node = scene.getNode(name);
            if (!node) {
                continue;
            }
            const core::NodeId target_id = resolveAlignTargetId(scene, *node);
            if (target_id == core::NULL_NODE) {
                continue;
            }
            return vis::scene_coords::nodeVisualizerWorldTransform(scene, target_id);
        }
        return std::nullopt;
    }

    float alignMarkerScreenRadius(const glm::vec3& world_pos,
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
            radius_render = kAlignMarkerWorldRadius * ortho_scale;
        } else {
            radius_render = (kAlignMarkerWorldRadius * projection[1][1] * window_height) / (2.0f * depth);
        }
        return glm::clamp(radius_render * glm::min(screen_scale_x, screen_scale_y), 5.0f, 50.0f);
    }

    glm::mat4 alignEdgeToWorldXRotation(const glm::mat4& up_rotation,
                                        const glm::vec3& p0,
                                        const glm::vec3& p1) {
        const glm::vec3 rotated_edge = glm::mat3(up_rotation) * (p1 - p0);
        const glm::vec2 xz(rotated_edge.x, rotated_edge.z);
        const float len = glm::length(xz);
        if (len <= 1e-6f) {
            return glm::mat4(1.0f);
        }
        // glm::rotate around +Y: x' = x cos θ + z sin θ, z' = -x sin θ + z cos θ.
        // θ = atan2(z, x) maps (x, z) onto (+len, 0).
        const float yaw = std::atan2(xz.y, xz.x);
        return glm::rotate(glm::mat4(1.0f), yaw, glm::vec3(0.0f, 1.0f, 0.0f));
    }

    void registerAlignOperators() {
        operators().registerOperator(BuiltinOp::AlignPickPoint, AlignPickPointOperator::DESCRIPTOR,
                                     [] { return std::make_unique<AlignPickPointOperator>(); });
    }

    void unregisterAlignOperators() {
        operators().unregisterOperator(BuiltinOp::AlignPickPoint);
    }

} // namespace lfs::vis::op

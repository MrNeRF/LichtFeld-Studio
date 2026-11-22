/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "align_ops.hpp"
#include "core/event_bridge/localization_manager.hpp"
#include "core/services.hpp"
#include "gui/gui_manager.hpp"
#include "gui/string_keys.hpp"
#include "operation/undo_entry.hpp"
#include "operation/undo_history.hpp"
#include "operator/operator_registry.hpp"
#include "rendering/rendering_manager.hpp"
#include "rendering/viewport_request_builder.hpp"
#include "scene/scene_manager.hpp"
#include "visualizer/scene_coordinate_utils.hpp"
#include "visualizer_impl.hpp"
#include <algorithm>
#include <cmath>
#include <glm/gtc/constants.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/quaternion.hpp>
#include <limits>

namespace lfs::vis::op {

    const OperatorDescriptor AlignPickPointOperator::DESCRIPTOR = {
        .builtin_id = BuiltinOp::AlignPickPoint,
        .python_class_id = {},
        .label = "Align to Ground",
        .description = "Pick 3 points to define ground plane",
        .icon = "align",
        .shortcut = "",
        .flags = OperatorFlags::REGISTER,
        .source = OperatorSource::CPP,
        .poll_deps = PollDependency::SCENE | PollDependency::SELECTION,
    };

    AlignPickPointOperator::AlignPickPointOperator()
        : PointPickOperator(services().alignPointPick(), 3, lichtfeld::Strings::Align::STATUS_NO_SURFACE) {}

    namespace {
        std::array<const core::SceneNode*, 2> comparisonPair(const core::Scene& scene, size_t offset) {
            const auto left = resolvePlyComparisonDepthSample(scene, offset, SplitViewPanelId::Left);
            const auto right = resolvePlyComparisonDepthSample(scene, offset, SplitViewPanelId::Right);
            if (!left.node || !right.node || !left.model || !right.model || scene.isNodeEffectivelyLocked(right.node->id))
                return {};
            for (auto* node = left.node; node && node->parent_id != core::NULL_NODE;) {
                node = scene.getNodeById(node->parent_id);
                if (node == right.node)
                    return {};
            }
            return {left.node, right.node};
        }
        std::array<const core::SceneNode*, 2> sessionPair(
            const core::Scene& scene, const RenderingManager::ModelAlignmentView& view) {
            const auto* reference = scene.getNodeByUuid(view.nodes[0]);
            const auto* mobile = scene.getNodeByUuid(view.nodes[1]);
            if (!reference || !mobile || !scene.isNodeEffectivelyVisible(reference->id) ||
                !scene.isNodeEffectivelyVisible(mobile->id) || scene.isNodeEffectivelyLocked(mobile->id))
                return {};
            return {reference, mobile};
        }
    } // namespace

    bool canAlignComparisonPair(const core::Scene& scene, const size_t offset) {
        return comparisonPair(scene, offset)[0] != nullptr;
    }

    OperatorResult AlignPickPointOperator::invoke(OperatorContext& ctx, OperatorProperties& props) {
        pair_.reset();
        max_points_ = 3;
        const auto* rm = services().renderingOrNull();
        const auto settings = rm ? rm->getSettings() : RenderSettings{};
        const auto& scene = ctx.scene().getScene();
        const auto* alignment = rm ? rm->modelAlignmentView() : nullptr;
        model_alignment_ = alignment != nullptr;
        if (alignment || splitViewUsesPLYComparison(settings.split_view_mode)) {
            const auto [reference, mobile] = alignment ? sessionPair(scene, *alignment)
                                                       : comparisonPair(scene, settings.split_view_offset);
            if (!reference || !mobile)
                return OperatorResult::CANCELLED;
            pair_ = Pair{{reference->uuid, mobile->uuid},
                         {vis::scene_coords::nodeVisualizerWorldTransform(scene, reference->id),
                          vis::scene_coords::nodeVisualizerWorldTransform(scene, mobile->id)}};
            max_points_ = std::numeric_limits<int>::max();
        }
        return PointPickOperator::invoke(ctx, props);
    }

    void AlignPickPointOperator::endModelAlignment() {
        if (!model_alignment_)
            return;
        model_alignment_ = false;
        auto* gui = services().guiOrNull();
        auto* renderer = services().renderingOrNull();
        if (renderer && gui && gui->getViewer())
            renderer->endModelAlignment(gui->getViewer()->getViewport());
    }

    void AlignPickPointOperator::cancel(OperatorContext& ctx) {
        PointPickOperator::cancel(ctx);
        endModelAlignment();
    }

    bool AlignPickPointOperator::acceptsPointSample(const PointSample& sample, std::optional<int> index) const {
        if (!pair_)
            return true;
        if (editablePairs()) {
            const bool incomplete = picked_points_.size() % 2 != 0;
            if (!index)
                return !incomplete || sample.panel == (pendingPairOnRight() ? SplitViewPanelId::Left : SplitViewPanelId::Right);
            if (incomplete && static_cast<size_t>(*index) + 1 == picked_points_.size())
                return sample.panel == (pendingPairOnRight() ? SplitViewPanelId::Right : SplitViewPanelId::Left);
        }
        const bool right = index.value_or(static_cast<int>(picked_points_.size())) % 2 != 0;
        return sample.panel == (right ? SplitViewPanelId::Right : SplitViewPanelId::Left);
    }

    void AlignPickPointOperator::showPointMenu(const glm::dvec2 position, const int point) {
        auto* gui = services().guiOrNull();
        if (!gui)
            return;
        gui->globalContextMenu().request(
            {{.label = LOC("align.delete_pair"), .action = "delete_pair"}},
            static_cast<float>(position.x), static_cast<float>(position.y),
            [point](std::string_view) {
                auto& state = services().alignPointPick();
                auto& registry = operators();
                if (registry.activeModalId() != to_string(BuiltinOp::AlignPickPoint) ||
                    state.getSelectedPoint() != point)
                    return;
                state.requestUiAction(PointPickState::UiAction::DeleteSelected);
                (void)registry.dispatchModalEvent({});
            });
    }

    std::optional<glm::mat4> AlignPickPointOperator::computePairPreviewTransform() const {
        if (!hasEnoughAlignmentPairs(picked_points_.size()))
            return std::nullopt;
        std::vector<glm::dvec3> source, target;
        source.reserve(picked_points_.size() / 2);
        target.reserve(picked_points_.size() / 2);
        for (size_t i = 0; i < picked_points_.size(); i += 2) {
            target.emplace_back(picked_points_[i]);
            source.emplace_back(picked_points_[i + 1]);
        }
        const auto fit = computePairAlignment(source, target, true);
        return fit ? std::optional<glm::mat4>(glm::mat4(*fit)) : std::nullopt;
    }

    std::optional<OperatorResult> AlignPickPointOperator::prepareModal(OperatorContext& ctx) {
        const auto* rm = services().renderingOrNull();
        const auto settings = rm ? rm->getSettings() : RenderSettings{};
        const bool comparing = splitViewUsesPLYComparison(settings.split_view_mode);
        if (pair_) {
            const auto& scene = ctx.scene().getScene();
            const auto* alignment = rm ? rm->modelAlignmentView() : nullptr;
            if (model_alignment_ && (!alignment || !rm->isIndependentSplitViewActive()))
                return OperatorResult::CANCELLED;
            const auto [reference, mobile] = alignment   ? sessionPair(scene, *alignment)
                                             : comparing ? comparisonPair(scene, settings.split_view_offset)
                                                         : std::array<const core::SceneNode*, 2>{};
            if (!reference || !mobile)
                return OperatorResult::CANCELLED;
            const std::array world{vis::scene_coords::nodeVisualizerWorldTransform(scene, reference->id),
                                   vis::scene_coords::nodeVisualizerWorldTransform(scene, mobile->id)};
            if (reference->uuid != pair_->ids[0] || mobile->uuid != pair_->ids[1]) {
                if (!model_alignment_ || !picked_points_.empty())
                    return OperatorResult::CANCELLED;
                clearAllPoints();
                pair_ = Pair{{reference->uuid, mobile->uuid}, world};
            }
            auto expected = pair_->world;
            if (!preview_targets_.empty())
                expected[1] = preview_targets_.front().preview_world.value_or(expected[1]);
            if (world != expected) {
                const bool had_points = !picked_points_.empty();
                restorePreview(ctx);
                clearAllPoints();
                (void)services().alignPointPick().takeUiAction();
                pair_->world = {world[0], vis::scene_coords::nodeVisualizerWorldTransform(scene, mobile->id)};
                if (had_points)
                    setStatus("align.pair_moved", 3.0);
                return OperatorResult::RUNNING_MODAL;
            }
            return std::nullopt;
        }
        if (comparing)
            return OperatorResult::CANCELLED;
        if (!preview_targets_.empty()) {
            const auto targets = resolvePointPickTargets(ctx, true);
            if (targets.size() != preview_targets_.size() ||
                std::any_of(preview_targets_.begin(), preview_targets_.end(), [&](const auto& saved) {
                    const auto* node = ctx.scene().getScene().getNodeById(saved.id);
                    return !node || node->uuid != saved.uuid ||
                           std::find(targets.begin(), targets.end(), saved.id) == targets.end();
                })) {
                restorePreview(ctx);
                clearAllPoints();
                (void)services().alignPointPick().takeUiAction();
                setStatus("align.status_targets_changed", 2.0);
                return OperatorResult::PASS_THROUGH;
            }
        }

        return std::nullopt;
    }

    bool AlignPickPointOperator::pointEditingBlocked() const {
        if (model_alignment_) {
            const auto* renderer = services().renderingOrNull();
            const auto* view = renderer ? renderer->modelAlignmentView() : nullptr;
            return view && view->preview.has_value();
        }
        return services().getAlignPreviewEnabled();
    }

    void AlignPickPointOperator::beforePointEdit(OperatorContext& ctx) {
        restorePreview(ctx);
    }

    void AlignPickPointOperator::pointSampleApplied() {
        services().setAlignCameraPosition(resolvePickPanelCameraPosition());
    }

    bool AlignPickPointOperator::applyPickedPoints(OperatorContext& ctx, float /*value*/) {
        return applyAlignment(ctx);
    }

    void AlignPickPointOperator::handleToolUiAction(OperatorContext& ctx, const PointPickState::UiAction action) {
        if (action == PointPickState::UiAction::TogglePreview) {
            if (pointEditingBlocked())
                restorePreview(ctx);
            else
                (void)updatePreview(ctx);
        } else if (action == PointPickState::UiAction::RefreshPreview && pointEditingBlocked()) {
            (void)updatePreview(ctx);
        }
    }

    OperatorResult AlignPickPointOperator::handleToolAction(OperatorContext& ctx, const input::Action action) {
        if (model_alignment_ && action == input::Action::UNDO) {
            beforePointEdit(ctx);
            undoPointEdit();
            return OperatorResult::RUNNING_MODAL;
        }
        if (action != input::Action::APPLY_ALIGNMENT)
            return OperatorResult::PASS_THROUGH;
        if (applyPickedPoints(ctx, 0.0f)) {
            clearAllPoints();
            return OperatorResult::FINISHED;
        }
        return OperatorResult::RUNNING_MODAL;
    }

    bool AlignPickPointOperator::poll(const OperatorContext& ctx,
                                      const OperatorProperties* /*props*/) const {
        const auto* rm = services().renderingOrNull();
        const auto settings = rm ? rm->getSettings() : RenderSettings{};
        if (const auto* alignment = rm ? rm->modelAlignmentView() : nullptr)
            return sessionPair(ctx.scene().getScene(), *alignment)[0] != nullptr;
        return splitViewUsesPLYComparison(settings.split_view_mode)
                   ? canAlignComparisonPair(ctx.scene().getScene(), settings.split_view_offset)
                   : !resolvePointPickTargets(ctx, true).empty();
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

    void AlignPickPointOperator::restorePreview(OperatorContext& ctx) {
        if (model_alignment_) {
            auto* gui = services().guiOrNull();
            auto* renderer = services().renderingOrNull();
            if (renderer && gui && gui->getViewer())
                renderer->setModelAlignmentPreview(gui->getViewer()->getViewport(), std::nullopt);
            return;
        }
        auto& scene = ctx.scene().getScene();
        for (const auto& saved : preview_targets_) {
            if (auto* node = scene.getNodeById(saved.id); node && node->uuid == saved.uuid &&
                                                          (!pair_ || !saved.preview_local || node->transform() == *saved.preview_local)) {
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
            rm->markDirty(DirtyFlag::SPLATS | DirtyFlag::MESH | DirtyFlag::OVERLAY, lfs::vis::FrameReason::SceneChange);
        }
    }

    bool AlignPickPointOperator::updatePreview(OperatorContext& ctx) {
        if (model_alignment_) {
            const auto transform = computePairPreviewTransform();
            auto* gui = services().guiOrNull();
            auto* renderer = services().renderingOrNull();
            if (!transform || !renderer || !gui || !gui->getViewer()) {
                setStatus("align.pair_degenerate", 3.0);
                return false;
            }
            renderer->setModelAlignmentPreview(gui->getViewer()->getViewport(), transform);
            publishPairPreviewFeedback(*transform);
            return true;
        }
        if (!pair_ && !pointsAreNonDegenerate(picked_points_)) {
            return false;
        }
        auto& scene = ctx.scene().getScene();
        if (preview_targets_.empty()) {
            std::vector<core::NodeId> targets;
            if (pair_) {
                const auto* mobile = scene.getNodeByUuid(pair_->ids[1]);
                if (!mobile || scene.isNodeEffectivelyLocked(mobile->id))
                    return false;
                targets = {mobile->id};
            } else
                targets = resolvePointPickTargets(ctx, true);
            for (const auto id : targets) {
                const auto* node = scene.getNodeById(id);
                preview_targets_.push_back({id, node->uuid, node->transform(), vis::scene_coords::nodeVisualizerWorldTransform(scene, id), {}, {}});
            }
            if (preview_targets_.empty()) {
                setStatus(lichtfeld::Strings::Align::STATUS_NO_TARGET, 2.0);
                return false;
            }
            preview_snap_world_ = resolveAlignSnapTargetWorld(ctx.scene());
            preview_camera_ = services().getAlignCameraPosition();
        }
        const auto transform = [&]() -> std::optional<glm::mat4> {
            if (pair_)
                return computePairPreviewTransform();
            AlignTransformInputs inputs{picked_points_[0], picked_points_[1], picked_points_[2], preview_camera_, std::nullopt, false};
            if (services().getAlignAxisSnapEnabled()) {
                inputs.snap_node_world = preview_snap_world_;
            }
            inputs.edge_to_world_x = services().getAlignEdgeToAxisEnabled();
            return computeAlignTransform(inputs);
        }();
        if (!transform) {
            restorePreview(ctx);
            if (pair_) {
                const auto count = picked_points_.size();
                setStatus(hasEnoughAlignmentPairs(count) ? "align.pair_degenerate"
                          : count >= 6                   ? "align.pair_finish"
                                                         : "align.pair_incomplete",
                          3.0);
            }
            return false;
        }
        for (auto& saved : preview_targets_) {
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
            if (!ctx.scene().setNodeTransform(node->name, *local)) {
                restorePreview(ctx);
                return false;
            }
            if (pair_) {
                saved.preview_local = *local;
                saved.preview_world = vis::scene_coords::nodeVisualizerWorldTransform(scene, saved.id);
            }
        }
        auto displayed_points = picked_points_;
        for (size_t i = pair_ ? 1 : 0; i < displayed_points.size(); i += pair_ ? 2 : 1) {
            displayed_points[i] = glm::vec3(*transform * glm::vec4(displayed_points[i], 1.0f));
        }
        services().alignPointPick().setPickedPoints(std::move(displayed_points));
        if (pair_)
            publishPairPreviewFeedback(*transform);
        services().setAlignPreviewEnabled(true);
        press_active_ = false;
        press_point_index_.reset();
        drag_active_ = false;
        if (auto* rm = services().renderingOrNull()) {
            rm->markDirty(DirtyFlag::SPLATS | DirtyFlag::MESH | DirtyFlag::OVERLAY, lfs::vis::FrameReason::SceneChange);
        }
        return true;
    }

    void AlignPickPointOperator::publishPairPreviewFeedback(const glm::mat4& transform) {
        const auto& points = picked_points_;
        double squared_error = 0.0;
        double largest_squared_error = 0.0;
        double squared_coordinate_scale = 0.0;
        size_t largest_error_index = 0;
        for (size_t i = 0; i < points.size(); i += 2) {
            const glm::dvec3 reference(points[i]);
            const glm::dvec3 mobile(transform * glm::vec4(points[i + 1], 1.0f));
            const auto delta = reference - mobile;
            const double error = glm::dot(delta, delta);
            squared_error += error;
            squared_coordinate_scale = std::max({squared_coordinate_scale,
                                                 glm::dot(reference, reference), glm::dot(mobile, mobile)});
            if (error > largest_squared_error) {
                largest_squared_error = error;
                largest_error_index = i;
            }
        }
        const auto pair_count = points.size() / 2;
        const double mean_squared_error = squared_error / static_cast<double>(pair_count);
        // Suppress ties and residuals at the precision of the viewport coordinates.
        constexpr double float_tolerance = 8.0 * std::numeric_limits<float>::epsilon();
        const bool distinct = pair_count > 3 && largest_squared_error >
                                                    std::max(2.25 * mean_squared_error, float_tolerance * float_tolerance * squared_coordinate_scale);
        const auto scale = glm::length(glm::dvec3(transform[0]));
        const auto rms = std::sqrt(mean_squared_error);
        services().setAlignPreviewFeedback({
            .focus = distinct ? std::optional{static_cast<int>(largest_error_index)} : std::nullopt,
            .summary = distinct ? LOCF("align.pair_fit_worst", scale, rms, largest_error_index / 2 + 1)
                                : LOCF("align.pair_fit", scale, rms),
        });
    }

    bool AlignPickPointOperator::applyAlignment(OperatorContext& ctx) {
        auto& scene = ctx.scene().getScene();
        std::vector<glm::mat4> originals;
        std::vector<std::string> names;
        std::optional<glm::mat4> model_alignment_local;
        if (model_alignment_) {
            const auto* mobile = scene.getNodeByUuid(pair_->ids[1]);
            const auto fit = computePairPreviewTransform();
            if (!mobile || !fit || scene.isNodeEffectivelyLocked(mobile->id))
                return false;
            model_alignment_local = vis::scene_coords::nodeLocalTransformFromVisualizerWorld(
                scene, mobile->id, *fit * pair_->world[1]);
            if (!model_alignment_local)
                return false;
            names.push_back(mobile->name);
            originals.push_back(mobile->transform());
        } else {
            if (!services().getAlignPreviewEnabled() && !updatePreview(ctx))
                return false;
            for (const auto& saved : preview_targets_) {
                const auto* node = scene.getNodeById(saved.id);
                if (!node || node->uuid != saved.uuid || scene.isNodeEffectivelyLocked(saved.id)) {
                    restorePreview(ctx);
                    return false;
                }
                names.push_back(node->name);
                originals.push_back(saved.local);
            }
        }

        auto entry = std::make_unique<SceneSnapshot>(ctx.scene(), "transform.align");
        if (!entry->captureTransformsBefore(names, originals)) {
            restorePreview(ctx);
            return false;
        }
        if (model_alignment_local && !ctx.scene().setNodeTransform(names.front(), *model_alignment_local))
            return false;
        entry->captureAfter();
        pushSceneSnapshotIfChanged(std::move(entry));
        if (model_alignment_) {
            restorePreview(ctx);
            endModelAlignment();
        }
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

    std::optional<glm::dmat4> computePairAlignment(
        std::span<const glm::dvec3> source, std::span<const glm::dvec3> reference, bool adjust_scale) {
        if (source.size() < 3 || source.size() != reference.size())
            return std::nullopt;

        glm::dvec3 source_center(0.0), reference_center(0.0);
        for (size_t i = 0; i < source.size(); ++i) {
            source_center += source[i];
            reference_center += reference[i];
        }
        source_center /= static_cast<double>(source.size());
        reference_center /= static_cast<double>(reference.size());

        const auto has_plane = [](std::span<const glm::dvec3> points, glm::dvec3 center) {
            // A triangle whose height is tiny relative to its extent has an
            // unstable orientation for landmarks picked from rendered depth.
            constexpr double min_relative_height = 1e-3;
            glm::dvec3 axis(0.0);
            double radius_squared = 0.0;
            for (const auto point : points) {
                const auto offset = point - center;
                const double squared = glm::dot(offset, offset);
                if (!std::isfinite(squared))
                    return false;
                if (squared > radius_squared) {
                    axis = offset;
                    radius_squared = squared;
                }
            }
            if (radius_squared <= 0.0)
                return false;
            for (const auto point : points) {
                const auto cross = glm::cross(axis, point - center);
                if (glm::length(cross) / radius_squared > min_relative_height)
                    return true;
            }
            return false;
        };
        if (!has_plane(source, source_center) || !has_plane(reference, reference_center))
            return std::nullopt;

        glm::dmat3 covariance(0.0);
        double variance = 0.0;
        for (size_t i = 0; i < source.size(); ++i) {
            const auto from = source[i] - source_center;
            const auto to = reference[i] - reference_center;
            covariance += glm::outerProduct(from, to);
            variance += glm::dot(from, from);
        }
        double magnitude = 0.0;
        for (int i = 0; i < 3; ++i)
            magnitude = std::max(magnitude, glm::length(covariance[i]));
        if (!(magnitude > 0.0) || !std::isfinite(magnitude) || !std::isfinite(variance))
            return std::nullopt;
        covariance /= magnitude;

        // Horn's quaternion matrix for the source-to-reference rotation.
        const double xx = covariance[0][0], xy = covariance[1][0], xz = covariance[2][0];
        const double yx = covariance[0][1], yy = covariance[1][1], yz = covariance[2][1];
        const double zx = covariance[0][2], zy = covariance[1][2], zz = covariance[2][2];
        glm::dmat4 symmetric(
            xx + yy + zz, yz - zy, zx - xz, xy - yx,
            yz - zy, xx - yy - zz, xy + yx, zx + xz,
            zx - xz, xy + yx, -xx + yy - zz, yz + zy,
            xy - yx, zx + xz, yz + zy, -xx - yy + zz);
        glm::dmat4 eigenvectors(1.0);
        for (int sweep = 0; sweep < 12; ++sweep) {
            for (int p = 0; p < 4; ++p) {
                for (int q = p + 1; q < 4; ++q) {
                    if (std::abs(symmetric[q][p]) <= 1e-14)
                        continue;
                    const double angle = 0.5 * std::atan2(
                                                   2.0 * symmetric[q][p], symmetric[p][p] - symmetric[q][q]);
                    glm::dmat4 jacobi(1.0);
                    jacobi[p][p] = jacobi[q][q] = std::cos(angle);
                    jacobi[p][q] = std::sin(angle);
                    jacobi[q][p] = -jacobi[p][q];
                    symmetric = glm::transpose(jacobi) * symmetric * jacobi;
                    eigenvectors *= jacobi;
                }
            }
        }
        int best = 0;
        for (int i = 1; i < 4; ++i) {
            if (symmetric[i][i] > symmetric[best][best])
                best = i;
        }
        for (int i = 0; i < 4; ++i) {
            if (i != best && symmetric[best][best] - symmetric[i][i] <= 1e-10)
                return std::nullopt;
        }
        const auto q = eigenvectors[best];
        const auto rotation = glm::mat3_cast(glm::dquat(q[0], q[1], q[2], q[3]));
        const double scale = adjust_scale ? symmetric[best][best] * magnitude / variance : 1.0;
        if (!(scale > 0.0) || !std::isfinite(scale))
            return std::nullopt;
        const glm::dmat3 linear = scale * rotation;
        const auto translation = reference_center - linear * source_center;
        if (!std::isfinite(glm::length(translation)))
            return std::nullopt;
        glm::dmat4 result(linear);
        result[3] = glm::dvec4(translation, 1.0);
        return result;
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
            const core::NodeId target_id = resolvePointPickTargetId(scene, *node, true);
            if (target_id == core::NULL_NODE) {
                continue;
            }
            return vis::scene_coords::nodeVisualizerWorldTransform(scene, target_id);
        }
        return std::nullopt;
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

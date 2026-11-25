/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "model_alignment_ops.hpp"
#include "core/services.hpp"
#include "gui/gui_manager.hpp"
#include "operation/undo_entry.hpp"
#include "operation/undo_history.hpp"
#include "rendering/rendering_manager.hpp"
#include "rendering/viewport_request_builder.hpp"
#include "scene/scene_manager.hpp"
#include "visualizer/app_store.hpp"
#include "visualizer/scene_coordinate_utils.hpp"
#include "visualizer_impl.hpp"
#include <algorithm>
#include <cmath>
#include <glm/gtc/constants.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/quaternion.hpp>
#include <limits>

namespace lfs::vis::op {

    ModelAlignmentOperator* ModelAlignmentOperator::active_ = nullptr;
    std::mutex ModelAlignmentOperator::published_session_mutex_;
    std::weak_ptr<const ModelAlignmentOperator::Session> ModelAlignmentOperator::published_session_;

    std::shared_ptr<const ModelAlignmentOperator::Session> ModelAlignmentOperator::savedViewSession() {
        const std::lock_guard lock(published_session_mutex_);
        return published_session_.lock();
    }

    void ModelAlignmentOperator::publishSavedView() {
        const std::lock_guard lock(published_session_mutex_);
        published_session_ = session_;
    }
    const OperatorDescriptor ModelAlignmentOperator::DESCRIPTOR = {
        .builtin_id = BuiltinOp::ModelAlignment,
        .label = "Align Models",
        .description = "Align two models using matching points",
        .icon = "align",
        .flags = OperatorFlags::REGISTER,
        .source = OperatorSource::CPP,
        .poll_deps = PollDependency::SCENE | PollDependency::SELECTION,
    };
    ModelAlignmentOperator::ModelAlignmentOperator()
        : PointPickOperator(point_state_, std::numeric_limits<int>::max(), "align.status_no_surface"),
          point_state_(publish_align_state_generation, true) {}
    ModelAlignmentOperator::~ModelAlignmentOperator() { end(); }
    namespace {
        std::array<const core::SceneNode*, 2> selectedPair(const OperatorContext& ctx, const OperatorProperties* props) {
            auto names = ctx.scene().getSelectedNodeNames();
            if (names.size() != 2)
                return {};
            const auto& scene = ctx.scene().getScene();
            // Use scene order for callers without explicit roles.
            for (const auto* node : scene.getNodes()) {
                if (node->name == names[0])
                    break;
                if (node->name == names[1]) {
                    std::swap(names[0], names[1]);
                    break;
                }
            }
            const auto reference = props ? props->get_or<std::string>("reference", names[0]) : names[0];
            const auto moving = props ? props->get_or<std::string>("moving", names[1]) : names[1];
            if (reference == moving || std::find(names.begin(), names.end(), reference) == names.end() ||
                std::find(names.begin(), names.end(), moving) == names.end())
                return {};
            return {scene.getNode(reference), scene.getNode(moving)};
        }
    } // namespace
    bool ModelAlignmentOperator::poll(const OperatorContext& ctx, const OperatorProperties* props) const {
        const auto& scene = ctx.scene().getScene();
        const auto [reference, mobile] = session_
                                             ? std::array{scene.getNodeByUuid(session_->nodes[0]), scene.getNodeByUuid(session_->nodes[1])}
                                             : selectedPair(ctx, props);
        return reference && mobile && canBegin(scene, *reference, *mobile);
    }
    OperatorResult ModelAlignmentOperator::invoke(OperatorContext& ctx, OperatorProperties& props) {
        if (!session_) {
            auto* gui = services().guiOrNull();
            if (!gui || !gui->getViewer())
                return OperatorResult::CANCELLED;
            const auto& scene = ctx.scene().getScene();
            const auto [reference, mobile] = selectedPair(ctx, &props);
            if (!reference || !mobile || !begin(gui->getViewer()->getViewport(), scene, *reference, *mobile))
                return OperatorResult::CANCELLED;
        }
        edits_.clear();
        feedback_ = {};
        return PointPickOperator::invoke(ctx, props);
    }
    void ModelAlignmentOperator::cancel(OperatorContext& ctx) {
        edits_.clear();
        PointPickOperator::cancel(ctx);
        end();
    }
    bool ModelAlignmentOperator::acceptsPointSample(const PointSample& sample, std::optional<int> index) const {
        const bool incomplete = picked_points_.size() % 2 != 0;
        if (!index)
            return !incomplete || sample.panel == (pendingPairOnRight() ? SplitViewPanelId::Left : SplitViewPanelId::Right);
        if (incomplete && static_cast<size_t>(*index) + 1 == picked_points_.size())
            return sample.panel == (pendingPairOnRight() ? SplitViewPanelId::Right : SplitViewPanelId::Left);
        const bool right = *index % 2 != 0;
        return sample.panel == (right ? SplitViewPanelId::Right : SplitViewPanelId::Left);
    }

    std::optional<glm::mat4> ModelAlignmentOperator::computePairPreviewTransform() const {
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

    void ModelAlignmentOperator::reportPairFitFailure() const {
        const auto count = picked_points_.size();
        setStatus(hasEnoughAlignmentPairs(count) ? "align.pair_degenerate"
                  : count % 2 != 0               ? "align.pair_finish"
                                                 : "align.pair_incomplete",
                  3.0);
    }

    void ModelAlignmentOperator::publishPairPreviewFeedback(const glm::mat4& transform) {
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
        feedback_ = {
            .focus = distinct ? std::optional{static_cast<int>(largest_error_index)} : std::nullopt,
            .summary = distinct ? LOCF("align.pair_fit_worst", scale, rms, largest_error_index / 2 + 1)
                                : LOCF("align.pair_fit", scale, rms),
        };
    }

    void ModelAlignmentOperator::pointsWillChange() {
        edits_.push_back({picked_points_, selected_point_, pick_panel_, pair_edit_});
    }
    void ModelAlignmentOperator::undoPointEdit() {
        if (edits_.empty())
            return;
        auto edit = std::move(edits_.back());
        edits_.pop_back();
        clearAllPoints();
        picked_points_ = std::move(edit.points);
        selected_point_ = edit.selected;
        pick_panel_ = edit.panel;
        pair_edit_ = std::move(edit.pair);
        syncPickedPointsToServices();
    }
    void ModelAlignmentOperator::appendPickedPoint(const PointSample& sample) {
        const bool completes_right_first = pair_edit_.pending_right;
        if (picked_points_.size() % 2 == 0) {
            size_t slot = 0;
            while (std::find(pair_edit_.colors.begin(), pair_edit_.colors.end(), slot) != pair_edit_.colors.end())
                ++slot;
            pair_edit_.colors.push_back(slot);
        }
        PointPickOperator::appendPickedPoint(sample);
        if (completes_right_first) {
            std::iter_swap(picked_points_.end() - 2, picked_points_.end() - 1);
            --*selected_point_;
        }
        pair_edit_.pending_right = picked_points_.size() % 2 != 0 && sample.panel == SplitViewPanelId::Right;
    }
    void ModelAlignmentOperator::erasePickedPoint(size_t index, bool selected) {
        const size_t first = selected ? index - index % 2 : index;
        if (selected || index % 2 == 0) {
            const auto pair = first / 2;
            if (pair < pair_edit_.colors.size())
                pair_edit_.colors.erase(pair_edit_.colors.begin() + pair);
        }
        if (!selected || first + 1 == picked_points_.size())
            pair_edit_.pending_right = false;
        const auto last = std::min(picked_points_.size(), first + (selected ? 2u : 1u));
        picked_points_.erase(picked_points_.begin() + first, picked_points_.begin() + last);
    }
    std::optional<OperatorResult> ModelAlignmentOperator::prepareModal(OperatorContext& ctx) {
        auto* renderer = services().renderingOrNull();
        if (!session_ || !renderer || !renderer->isIndependentSplitViewActive())
            return OperatorResult::CANCELLED;
        const auto& scene = ctx.scene().getScene();
        const auto* reference = scene.getNodeByUuid(session_->nodes[0]);
        const auto* mobile = scene.getNodeByUuid(session_->nodes[1]);
        if (!reference || !mobile || !canBegin(scene, *reference, *mobile))
            return OperatorResult::CANCELLED;
        const std::array world{scene_coords::nodeVisualizerWorldTransform(scene, reference->id),
                               scene_coords::nodeVisualizerWorldTransform(scene, mobile->id)};
        if (world != session_->world) {
            const bool had_points = !picked_points_.empty();
            restorePreview(ctx);
            edits_.clear();
            clearAllPoints();
            (void)point_state_.takeUiAction();
            session_->world = world;
            if (had_points)
                setStatus("align.pair_moved", 3.0);
            return OperatorResult::RUNNING_MODAL;
        }
        return std::nullopt;
    }
    bool ModelAlignmentOperator::pointEditingBlocked() const { return session_ && session_->preview.has_value(); }
    void ModelAlignmentOperator::beforePointEdit(OperatorContext& ctx) { restorePreview(ctx); }
    bool ModelAlignmentOperator::applyPickedPoints(OperatorContext& ctx, float) { return applyAlignment(ctx); }
    void ModelAlignmentOperator::restorePreview(OperatorContext&) {
        setPreview(std::nullopt);
        feedback_ = {};
    }
    bool ModelAlignmentOperator::updatePreview(OperatorContext&) {
        const auto transform = computePairPreviewTransform();
        if (!transform || !session_) {
            reportPairFitFailure();
            return false;
        }
        setPreview(transform);
        publishPairPreviewFeedback(*transform);
        return true;
    }
    void ModelAlignmentOperator::handleToolUiAction(OperatorContext& ctx, const PointPickState::UiAction action) {
        if (action == PointPickState::UiAction::TogglePreview) {
            if (pointEditingBlocked())
                restorePreview(ctx);
            else
                (void)updatePreview(ctx);
        } else if (action == PointPickState::UiAction::RefreshPreview && pointEditingBlocked()) {
            (void)updatePreview(ctx);
        }
    }

    OperatorResult ModelAlignmentOperator::handleToolAction(OperatorContext& ctx, const input::Action action) {
        if (blocksSceneAction(action))
            return OperatorResult::RUNNING_MODAL;
        if (action == input::Action::REDO)
            return OperatorResult::RUNNING_MODAL;
        if (action == input::Action::UNDO) {
            if (pointEditingBlocked())
                restorePreview(ctx);
            else
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

    bool ModelAlignmentOperator::applyAlignment(OperatorContext& ctx) {
        if (!session_)
            return false;
        auto& scene = ctx.scene().getScene();
        const auto* mobile = scene.getNodeByUuid(session_->nodes[1]);
        const auto fit = computePairPreviewTransform();
        if (!fit) {
            reportPairFitFailure();
            return false;
        }
        if (!mobile || scene.isNodeEffectivelyLocked(mobile->id)) {
            setStatus("align.status_no_target");
            return false;
        }
        const auto local = scene_coords::nodeLocalTransformFromVisualizerWorld(scene, mobile->id, *fit * session_->world[1]);
        if (!local) {
            setStatus("align.status_no_target");
            return false;
        }
        auto entry = std::make_unique<SceneSnapshot>(ctx.scene(), "transform.align");
        if (!entry->captureTransformsBefore({mobile->name}, {mobile->transform()}))
            return false;
        if (!ctx.scene().setNodeTransform(mobile->name, *local))
            return false;
        entry->captureAfter();
        pushSceneSnapshotIfChanged(std::move(entry));
        end();
        return true;
    }
    bool ModelAlignmentOperator::canBegin(
        const core::Scene& scene, const core::SceneNode& reference, const core::SceneNode& mobile) {
        if (reference.type != core::NodeType::SPLAT ||
            mobile.type != core::NodeType::SPLAT || !reference.model || !mobile.model ||
            reference.model->size() == 0 || mobile.model->size() == 0 ||
            !scene.isNodeEffectivelyVisible(reference.id) || !scene.isNodeEffectivelyVisible(mobile.id) ||
            scene.isNodeEffectivelyLocked(mobile.id))
            return false;
        for (auto* parent = &reference; parent; parent = scene.getNodeById(parent->parent_id)) {
            if (parent->id == mobile.id)
                return false;
        }

        return true;
    }

    bool ModelAlignmentOperator::begin(Viewport& primary, const core::Scene& scene,
                                       const core::SceneNode& reference, const core::SceneNode& mobile) {
        auto* renderer = services().renderingOrNull();
        if (session_ || active_ || !renderer || !canBegin(scene, reference, mobile))
            return false;
        const auto settings = renderer->getSettings();
        session_ = std::make_shared<Session>(Session{
            .nodes = {reference.uuid, mobile.uuid},
            .world = {scene_coords::nodeVisualizerWorldTransform(scene, reference.id), scene_coords::nodeVisualizerWorldTransform(scene, mobile.id)},
            .previous_mode = settings.split_view_mode,
            .previous_position = settings.split_position,
            .previous_offset = settings.split_view_offset,
            .previous_focus = renderer->getFocusedSplitPanel(),
            .primary_viewport = &primary,
            .previous_primary = primary,
            .previous_secondary = renderer->projectSecondaryViewport(),
        });
        active_ = this;
        publishSavedView();
        renderer->restoreSplitViewMode(SplitViewMode::IndependentDual, primary,
                                       splitViewUsesIndependentPanels(settings.split_view_mode) ? std::nullopt : std::optional{0.5f});
        auto& secondary = renderer->projectSecondaryViewport();
        primary.ortho_scale_override = primary.ortho_scale_override.value_or(settings.ortho_scale);
        secondary.ortho_scale_override = secondary.ortho_scale_override.value_or(settings.ortho_scale);
        renderer->setFocusedSplitPanel(SplitViewPanelId::Left);
        publishView();
        return true;
    }
    namespace {
        void restoreAlignmentCamera(Viewport& target, const Viewport& saved) {
            target.camera = saved.camera;
            target.ortho_scale_override = saved.ortho_scale_override;
        }
    } // namespace
    void ModelAlignmentOperator::end() {
        if (!session_)
            return;
        const auto saved = session_;
        session_.reset();
        active_ = nullptr;
        if (auto* renderer = services().renderingOrNull()) {
            renderer->setIndependentPanelOverrides(std::nullopt);
            auto& primary = *saved->primary_viewport;
            renderer->restoreSplitViewMode(saved->previous_mode, primary, saved->previous_position, saved->previous_offset);
            restoreAlignmentCamera(primary, saved->previous_primary);
            restoreAlignmentCamera(renderer->projectSecondaryViewport(), saved->previous_secondary);
            renderer->setFocusedSplitPanel(saved->previous_focus);
        }
        {
            const std::lock_guard lock(published_session_mutex_);
            published_session_.reset();
        }
        publish_align_state_generation();
    }
    void ModelAlignmentOperator::publishView() {
        auto* renderer = services().renderingOrNull();
        if (!renderer || !session_)
            return;
        RenderingManager::IndependentSplitOverrides overrides{
            .exact_surface_picking = !session_->preview,
        };
        for (size_t i = 0; i < overrides.panels.size(); ++i) {
            auto& panel = overrides.panels[i];
            if (session_->composite && splitViewPanelIndex(*session_->composite) == i)
                panel.visible_nodes = {session_->nodes[0], session_->nodes[1]};
            else
                panel.visible_nodes = {session_->nodes[i]};
            if (session_->preview)
                panel.transforms.emplace_back(session_->nodes[1], *session_->preview);
        }
        renderer->setIndependentPanelOverrides(std::move(overrides));
        publish_align_state_generation();
    }
    void ModelAlignmentOperator::setPreview(std::optional<glm::mat4> transform) {
        auto* renderer = services().renderingOrNull();
        if (!renderer || !session_)
            return;
        auto& primary = *session_->primary_viewport;

        auto& view = *session_;
        auto& right = renderer->projectSecondaryViewport();
        if (transform && !view.preview) {
            view.free_views = std::array{primary, right};
            const float scale = glm::length(glm::vec3((*transform)[0]));
            const glm::mat3 rotation = glm::mat3(*transform) / scale;
            right.camera.t = glm::vec3(*transform * glm::vec4(right.camera.t, 1.0f));
            right.camera.pivot = glm::vec3(*transform * glm::vec4(right.camera.pivot, 1.0f));
            right.camera.R = rotation * right.camera.R;
            if (right.ortho_scale_override)
                *right.ortho_scale_override /= scale;
            view.composite = SplitViewPanelId::Right;
        } else if (!transform && view.preview) {
            if (view.free_views) {
                restoreAlignmentCamera(primary, (*view.free_views)[0]);
                restoreAlignmentCamera(right, (*view.free_views)[1]);
            }
            view.free_views.reset();
            view.composite.reset();
            view.linked = false;
        }
        view.preview = std::move(transform);
        publishView();
    }

    bool ModelAlignmentOperator::canSwapRoles(const core::Scene& scene) const {
        auto* renderer = services().renderingOrNull();
        if (!renderer || !session_)
            return false;
        if (session_->preview || !picked_points_.empty())
            return false;
        const auto& nodes = session_->nodes;
        const auto* reference = scene.getNodeByUuid(nodes[0]);
        const auto* mobile = scene.getNodeByUuid(nodes[1]);
        return reference && mobile && canBegin(scene, *mobile, *reference);
    }

    ModelAlignmentState ModelAlignmentOperator::state() const {
        ModelAlignmentState result;
        if (!session_)
            return result;
        result.active = true;
        result.preview = session_->preview.has_value();
        result.can_apply = hasEnoughAlignmentPairs(picked_points_.size());
        result.linked = session_->linked;
        result.display_mode = !session_->composite ? ModelAlignmentDisplayMode::Separate : *session_->composite == SplitViewPanelId::Left ? ModelAlignmentDisplayMode::OverlayLeft
                                                                                                                                          : ModelAlignmentDisplayMode::OverlayRight;
        const auto* manager = services().sceneOrNull();
        result.can_swap_roles = manager && canSwapRoles(manager->getScene());
        result.can_delete_pair = !session_->preview && point_state_.getSelectedPoint().has_value();
        result.can_undo = session_->preview.has_value() || canUndoPointEdit();
        return result;
    }
    void ModelAlignmentOperator::swapRoles(const core::Scene& scene) {
        auto* renderer = services().renderingOrNull();
        if (!renderer || !session_)
            return;
        auto& primary = *session_->primary_viewport;

        if (!canSwapRoles(scene))
            return;
        std::swap(session_->nodes[0], session_->nodes[1]);
        std::swap(session_->world[0], session_->world[1]);
        auto& right = renderer->projectSecondaryViewport();
        std::swap(primary.camera, right.camera);
        std::swap(primary.ortho_scale_override, right.ortho_scale_override);
        publishView();
    }

    void ModelAlignmentOperator::setComposite(std::optional<SplitViewPanelId> panel) {
        if (!session_ || !session_->preview || session_->composite == panel)
            return;
        session_->composite = panel;
        publishView();
    }

    void ModelAlignmentOperator::toggleCameraLink() {
        auto* renderer = services().renderingOrNull();
        if (!renderer || !session_)
            return;
        auto& primary = *session_->primary_viewport;

        if (!session_ || !session_->preview)
            return;
        auto& view = *session_;
        auto& right = renderer->projectSecondaryViewport();
        view.linked = !view.linked;
        if (view.linked)
            syncCamera(renderer->getFocusedSplitPanel() == SplitViewPanelId::Right ? right : primary);
        publishView();
    }

    void ModelAlignmentOperator::syncCamera(const Viewport& source) {
        auto* renderer = services().renderingOrNull();
        if (!renderer || !session_)
            return;
        auto& primary = *session_->primary_viewport;

        if (!session_ || !session_->linked)
            return;
        auto& right = renderer->projectSecondaryViewport();
        auto& target = &source == &right ? primary : right;
        target.camera.t = source.camera.t;
        target.camera.pivot = source.camera.pivot;
        target.camera.R = source.camera.R;
        target.ortho_scale_override = source.ortho_scale_override;
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
} // namespace lfs::vis::op

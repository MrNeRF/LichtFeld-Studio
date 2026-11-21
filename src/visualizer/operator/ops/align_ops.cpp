/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "align_ops.hpp"
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
        std::array<const core::SceneNode*, 2> comparisonPair(const OperatorContext& ctx) {
            auto* rm = services().renderingOrNull();
            if (!rm)
                return {};
            const auto settings = rm->getSettings();
            if (!splitViewUsesPLYComparison(settings.split_view_mode))
                return {};
            const auto& scene = ctx.scene().getScene();
            const auto left = resolvePlyComparisonDepthSample(scene, settings.split_view_offset, SplitViewPanelId::Left);
            const auto right = resolvePlyComparisonDepthSample(scene, settings.split_view_offset, SplitViewPanelId::Right);
            if (!left.model || !right.model)
                return {};
            for (auto* node = left.node; node && node->parent_id != core::NULL_NODE;) {
                node = scene.getNodeById(node->parent_id);
                if (node == right.node)
                    return {};
            }
            return {left.node, right.node};
        }
    } // namespace

    OperatorResult AlignPickPointOperator::invoke(OperatorContext& ctx, OperatorProperties& props) {
        pair_.reset();
        max_points_ = 3;
        const auto [reference, mobile] = comparisonPair(ctx);
        if (reference && mobile && !ctx.scene().getScene().isNodeEffectivelyLocked(mobile->id)) {
            pair_ = std::array{reference->uuid, mobile->uuid};
            max_points_ = 6;
        }
        return PointPickOperator::invoke(ctx, props);
    }

    bool AlignPickPointOperator::acceptsPointSample(const PointSample& sample, std::optional<int> index) const {
        if (!pair_)
            return true;
        const bool right = index.value_or(static_cast<int>(picked_points_.size())) % 2 != 0;
        return sample.panel == (right ? SplitViewPanelId::Right : SplitViewPanelId::Left);
    }

    std::optional<glm::mat4> AlignPickPointOperator::computePairPreviewTransform() const {
        if (picked_points_.size() != 6)
            return std::nullopt;
        std::array<glm::dvec3, 3> source, target;
        for (size_t i = 0; i < source.size(); ++i) {
            target[i] = picked_points_[2 * i];
            source[i] = picked_points_[2 * i + 1];
        }
        const auto fit = computePairAlignment(source, target, true);
        return fit ? std::optional<glm::mat4>(glm::mat4(*fit)) : std::nullopt;
    }

    std::optional<OperatorResult> AlignPickPointOperator::prepareModal(OperatorContext& ctx) {
        if (pair_) {
            const auto [reference, mobile] = comparisonPair(ctx);
            if (!reference || !mobile || ctx.scene().getScene().isNodeEffectivelyLocked(mobile->id) ||
                reference->uuid != (*pair_)[0] || mobile->uuid != (*pair_)[1]) {
                restorePreview(ctx);
                clearAllPoints();
                return OperatorResult::CANCELLED;
            }
        }
        const auto* rm = services().renderingOrNull();
        if (!pair_ && rm && splitViewUsesPLYComparison(rm->getSettings().split_view_mode)) {
            restorePreview(ctx);
            clearAllPoints();
            return OperatorResult::CANCELLED;
        }
        if (!preview_targets_.empty()) {
            const auto targets = pair_ ? std::vector{comparisonPair(ctx)[1]->id} : resolvePointPickTargets(ctx, true);
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
            if (services().getAlignPreviewEnabled())
                restorePreview(ctx);
            else
                (void)updatePreview(ctx);
        } else if (action == PointPickState::UiAction::RefreshPreview && services().getAlignPreviewEnabled()) {
            (void)updatePreview(ctx);
        }
    }

    OperatorResult AlignPickPointOperator::handleToolAction(OperatorContext& ctx, const input::Action action) {
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
        const auto [reference, mobile] = comparisonPair(ctx);
        const auto* rm = services().renderingOrNull();
        return rm && splitViewUsesPLYComparison(rm->getSettings().split_view_mode)
                   ? reference && mobile && !ctx.scene().getScene().isNodeEffectivelyLocked(mobile->id)
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
            rm->markDirty(DirtyFlag::SPLATS | DirtyFlag::MESH | DirtyFlag::OVERLAY, lfs::vis::FrameReason::SceneChange);
        }
    }

    bool AlignPickPointOperator::updatePreview(OperatorContext& ctx) {
        if (!pair_ && !pointsAreNonDegenerate(picked_points_)) {
            return false;
        }
        auto& scene = ctx.scene().getScene();
        if (preview_targets_.empty()) {
            const auto [reference, mobile] = comparisonPair(ctx);
            if (pair_ && (!reference || !mobile || reference->uuid != (*pair_)[0] ||
                          mobile->uuid != (*pair_)[1] || scene.isNodeEffectivelyLocked(mobile->id)))
                return false;
            const auto targets = pair_ ? std::vector{mobile->id} : resolvePointPickTargets(ctx, true);
            for (const auto id : targets) {
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
                setStatus(picked_points_.size() == 6 ? "align.pair_degenerate" : "align.pair_incomplete", 3.0);
            }
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
            if (!ctx.scene().setNodeTransform(node->name, *local)) {
                restorePreview(ctx);
                return false;
            }
        }
        auto displayed_points = picked_points_;
        for (size_t i = pair_ ? 1 : 0; i < displayed_points.size(); i += pair_ ? 2 : 1) {
            displayed_points[i] = glm::vec3(*transform * glm::vec4(displayed_points[i], 1.0f));
        }
        services().alignPointPick().setPickedPoints(std::move(displayed_points));
        services().setAlignPreviewEnabled(true);
        press_active_ = false;
        press_point_index_.reset();
        drag_active_ = false;
        if (auto* rm = services().renderingOrNull()) {
            rm->markDirty(DirtyFlag::SPLATS | DirtyFlag::MESH | DirtyFlag::OVERLAY, lfs::vis::FrameReason::SceneChange);
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
            if (!node || node->uuid != saved.uuid || scene.isNodeEffectivelyLocked(saved.id)) {
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
                if (glm::length(cross) / radius_squared > 1e-6)
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

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
#include "scene/scene_manager.hpp"
#include "visualizer/scene_coordinate_utils.hpp"
#include "visualizer_impl.hpp"
#include <algorithm>
#include <cmath>
#include <glm/gtc/constants.hpp>
#include <glm/gtc/matrix_transform.hpp>

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

    std::optional<OperatorResult> AlignPickPointOperator::prepareModal(OperatorContext& ctx) {
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
        return services().getAlignPreviewEnabled();
    }

    void AlignPickPointOperator::beforePointEdit(OperatorContext& ctx) {
        restorePreview(ctx);
    }

    void AlignPickPointOperator::pointSampleApplied() {
        services().setAlignCameraPosition(resolvePickPanelCameraPosition());
    }

    bool AlignPickPointOperator::applyPickedPoints(OperatorContext& ctx, float /*value*/) {
        return picked_points_.size() == 3 && pointsAreNonDegenerate(picked_points_) && applyAlignment(ctx);
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
        if (picked_points_.size() == 3 && applyAlignment(ctx)) {
            clearAllPoints();
            return OperatorResult::FINISHED;
        }
        return OperatorResult::RUNNING_MODAL;
    }

    bool AlignPickPointOperator::poll(const OperatorContext& ctx,
                                      const OperatorProperties* /*props*/) const {
        return !resolvePointPickTargets(ctx, true).empty();
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
        if (!pointsAreNonDegenerate(picked_points_)) {
            return false;
        }
        auto& scene = ctx.scene().getScene();
        if (preview_targets_.empty()) {
            for (const auto id : resolvePointPickTargets(ctx, true)) {
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

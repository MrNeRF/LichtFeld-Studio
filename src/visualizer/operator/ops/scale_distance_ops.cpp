/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "scale_distance_ops.hpp"
#include "core/services.hpp"
#include "gui/string_keys.hpp"
#include "operation/undo_entry.hpp"
#include "operation/undo_history.hpp"
#include "operator/operator_registry.hpp"
#include "rendering/rendering_manager.hpp"
#include "scene/scene_manager.hpp"
#include "visualizer/scene_coordinate_utils.hpp"
#include <cmath>
#include <glm/gtc/matrix_transform.hpp>
#include <utility>

namespace lfs::vis::op {

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

    ScaleDistancePickPointOperator::ScaleDistancePickPointOperator()
        : PointPickOperator(services().scaleDistancePointPick(), 2, lichtfeld::Strings::ScaleDistance::STATUS_NO_SURFACE) {}

    bool ScaleDistancePickPointOperator::poll(const OperatorContext& ctx,
                                              const OperatorProperties* /*props*/) const {
        return !resolvePointPickTargets(ctx, false).empty();
    }

    bool ScaleDistancePickPointOperator::applyPickedPoints(OperatorContext& ctx, const float target_distance) {
        if (picked_points_.size() != 2) {
            return false;
        }

        const auto transform = computeScaleDistanceTransform(picked_points_[0], picked_points_[1], target_distance);
        if (!transform) {
            setStatus(lichtfeld::Strings::ScaleDistance::STATUS_INVALID_DISTANCE, 2.0);
            return false;
        }

        const auto target_ids = resolvePointPickTargets(ctx, false);
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

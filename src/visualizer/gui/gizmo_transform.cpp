/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "gui/gizmo_transform.hpp"
#include "visualizer/scene_coordinate_utils.hpp"
#include <cassert>
#include <glm/gtc/matrix_transform.hpp>

namespace lfs::vis::gui {

    void GizmoTransformContext::reset() {
        target_names.clear();
        targets.clear();
        pivot_world = glm::vec3(0.0f);
        pivot_local = glm::vec3(0.0f);
        cumulative_rotation = glm::mat3(1.0f);
        cumulative_scale = glm::vec3(1.0f);
        cumulative_translation = glm::vec3(0.0f);
    }

    namespace gizmo_ops {

        glm::mat3 extractRotation(const glm::mat4& m) {
            return glm::mat3(
                glm::normalize(glm::vec3(m[0])),
                glm::normalize(glm::vec3(m[1])),
                glm::normalize(glm::vec3(m[2])));
        }

        glm::vec3 extractScale(const glm::mat4& m) {
            return glm::vec3(
                glm::length(glm::vec3(m[0])),
                glm::length(glm::vec3(m[1])),
                glm::length(glm::vec3(m[2])));
        }

        glm::vec3 extractTranslation(const glm::mat4& m) {
            return glm::vec3(m[3]);
        }

        void setNodeVisualizerWorldTransform(core::Scene& scene,
                                             const std::string& name,
                                             const glm::mat4& visualizer_world_transform) {
            const auto* const node = scene.getNode(name);
            if (!node) {
                return;
            }
            if (const auto local_transform =
                    scene_coords::nodeLocalTransformFromVisualizerWorld(scene, node->id, visualizer_world_transform)) {
                scene.setNodeTransform(name, *local_transform);
            }
        }

        glm::mat4 computeGizmoMatrix(
            const glm::vec3& pivot_world,
            const glm::mat3& rotation,
            const glm::vec3& scale,
            bool use_world_space,
            bool is_scale_operation) {

            glm::mat4 gizmo_matrix = glm::translate(glm::mat4(1.0f), pivot_world);

            // For scale operations: always use local axes so scale corresponds to local dimensions
            // For rotate/translate: respect world/local setting
            const bool include_rotation = is_scale_operation || !use_world_space;

            if (include_rotation) {
                gizmo_matrix = gizmo_matrix * glm::mat4(rotation);
            }

            gizmo_matrix = glm::scale(gizmo_matrix, scale);

            return gizmo_matrix;
        }

        GizmoTransformContext captureCropBox(
            const core::Scene& scene,
            const std::string& name,
            const glm::vec3& pivot_world,
            const glm::vec3& pivot_local,
            TransformSpace space,
            PivotMode pivot_mode) {

            GizmoTransformContext ctx;
            ctx.type = GizmoTargetType::CropBox;
            ctx.target_names.push_back(name);
            ctx.pivot_world = pivot_world;
            ctx.pivot_local = pivot_local;
            ctx.use_world_space = (space == TransformSpace::World);
            ctx.pivot_mode = pivot_mode;

            const auto* node = scene.getNode(name);
            if (!node || !node->cropbox)
                return ctx;

            GizmoTransformContext::TargetState state;
            state.name = name;

            const glm::mat4 world_transform = scene_coords::nodeVisualizerWorldTransform(scene, node->id);
            state.visualizer_world_transform = world_transform;
            state.rotation = extractRotation(world_transform);

            state.bounds_min = node->cropbox->min;
            state.bounds_max = node->cropbox->max;

            ctx.targets.push_back(state);
            return ctx;
        }

        GizmoTransformContext captureEllipsoid(
            const core::Scene& scene,
            const std::string& name,
            const glm::vec3& pivot_world,
            const glm::vec3& pivot_local,
            TransformSpace space,
            PivotMode pivot_mode) {

            GizmoTransformContext ctx;
            ctx.type = GizmoTargetType::Ellipsoid;
            ctx.target_names.push_back(name);
            ctx.pivot_world = pivot_world;
            ctx.pivot_local = pivot_local;
            ctx.use_world_space = (space == TransformSpace::World);
            ctx.pivot_mode = pivot_mode;

            const auto* node = scene.getNode(name);
            if (!node || !node->ellipsoid)
                return ctx;

            GizmoTransformContext::TargetState state;
            state.name = name;

            const glm::mat4 world_transform = scene_coords::nodeVisualizerWorldTransform(scene, node->id);
            state.visualizer_world_transform = world_transform;
            state.rotation = extractRotation(world_transform);

            state.radii = node->ellipsoid->radii;

            ctx.targets.push_back(state);
            return ctx;
        }

        void applyTranslation(
            GizmoTransformContext& ctx,
            core::Scene& scene,
            const glm::vec3& new_pivot_world) {

            const glm::vec3 delta = new_pivot_world - ctx.pivot_world;
            ctx.cumulative_translation = delta;
            const glm::mat4 world_delta = glm::translate(glm::mat4(1.0f), delta);

            for (const auto& target : ctx.targets) {
                const glm::mat4 new_world_transform = world_delta * target.visualizer_world_transform;
                setNodeVisualizerWorldTransform(scene, target.name, new_world_transform);
            }

            scene.invalidateCache();
        }

        void applyRotation(
            GizmoTransformContext& ctx,
            core::Scene& scene,
            const glm::mat3& delta_rotation) {

            // Accumulate rotation in world space
            ctx.cumulative_rotation = delta_rotation * ctx.cumulative_rotation;
            const glm::mat4 world_delta = glm::translate(glm::mat4(1.0f), ctx.pivot_world) *
                                          glm::mat4(ctx.cumulative_rotation) *
                                          glm::translate(glm::mat4(1.0f), -ctx.pivot_world);

            for (const auto& target : ctx.targets) {
                const glm::mat4 new_world_transform = world_delta * target.visualizer_world_transform;
                setNodeVisualizerWorldTransform(scene, target.name, new_world_transform);
            }

            scene.invalidateCache();
        }

        void applyBoundsScale(
            GizmoTransformContext& ctx,
            core::Scene& scene,
            const glm::vec3& new_size) {

            assert(ctx.targets.size() == 1);
            const auto& target = ctx.targets[0];

            auto* node = scene.getMutableNode(target.name);
            if (!node)
                return;

            if (ctx.type == GizmoTargetType::CropBox && node->cropbox) {
                const glm::vec3 original_size = target.bounds_max - target.bounds_min;
                ctx.cumulative_scale = new_size / original_size;

                const glm::vec3 original_center = (target.bounds_min + target.bounds_max) * 0.5f;
                const glm::vec3 half_size = new_size * 0.5f;
                node->cropbox->min = original_center - half_size;
                node->cropbox->max = original_center + half_size;
            } else if (ctx.type == GizmoTargetType::Ellipsoid && node->ellipsoid) {
                ctx.cumulative_scale = new_size / target.radii;

                node->ellipsoid->radii = new_size;
            }

            scene.invalidateCache();
        }

    } // namespace gizmo_ops

} // namespace lfs::vis::gui

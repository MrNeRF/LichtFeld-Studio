/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "gui/gizmo_transform.hpp"
#include "visualizer/scene_coordinate_utils.hpp"
#include <algorithm>
#include <cassert>
#include <glm/gtc/matrix_transform.hpp>
#include <limits>
#include <unordered_set>

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

        std::vector<std::string> topLevelTransformTargets(
            const core::Scene& scene,
            const std::vector<std::string>& target_names) {
            std::unordered_set<core::NodeId> selected_ids;
            selected_ids.reserve(target_names.size());
            for (const auto& name : target_names) {
                if (const auto* node = scene.getNode(name)) {
                    selected_ids.insert(node->id);
                }
            }

            std::vector<std::string> top_level_names;
            top_level_names.reserve(target_names.size());
            for (const auto& name : target_names) {
                const auto* node = scene.getNode(name);
                if (!node)
                    continue;

                bool ancestor_selected = false;
                for (core::NodeId check_id = node->parent_id; check_id != core::NULL_NODE;) {
                    if (selected_ids.contains(check_id)) {
                        ancestor_selected = true;
                        break;
                    }
                    const auto* parent = scene.getNodeById(check_id);
                    check_id = parent ? parent->parent_id : core::NULL_NODE;
                }

                if (!ancestor_selected) {
                    top_level_names.push_back(name);
                }
            }

            return top_level_names;
        }

        enum class TransformDeltaComposition {
            SharedSelectionWorld,
            IndividualLocal,
        };

        std::vector<NodeLocalTransformResult> computeNodeLocalTransforms(
            const core::Scene& scene,
            const std::vector<std::string>& target_names,
            const std::vector<glm::mat4>& original_visualizer_world_transforms,
            const glm::mat4& visualizer_delta,
            const TransformDeltaComposition composition) {
            std::vector<NodeLocalTransformResult> results;
            assert(target_names.size() == original_visualizer_world_transforms.size());
            if (target_names.size() != original_visualizer_world_transforms.size()) {
                return results;
            }
            const size_t count = target_names.size();
            results.reserve(count);

            for (size_t i = 0; i < count; ++i) {
                const glm::mat4 next_visualizer_world = composition == TransformDeltaComposition::SharedSelectionWorld
                                                            ? visualizer_delta * original_visualizer_world_transforms[i]
                                                            : original_visualizer_world_transforms[i] * visualizer_delta;
                if (const auto local_transform =
                        scene_coords::nodeLocalTransformFromVisualizerWorld(
                            scene, target_names[i], next_visualizer_world)) {
                    results.push_back(NodeLocalTransformResult{
                        .name = target_names[i],
                        .local_transform = *local_transform,
                    });
                }
            }

            return results;
        }

        std::vector<NodeLocalTransformResult> computeNodeSharedSelectionLocalTransforms(
            const core::Scene& scene,
            const std::vector<std::string>& target_names,
            const std::vector<glm::mat4>& original_visualizer_world_transforms,
            const glm::mat4& visualizer_world_delta) {
            return computeNodeLocalTransforms(
                scene,
                target_names,
                original_visualizer_world_transforms,
                visualizer_world_delta,
                TransformDeltaComposition::SharedSelectionWorld);
        }

        std::vector<NodeLocalTransformResult> computeNodeIndividualLocalTransforms(
            const core::Scene& scene,
            const std::vector<std::string>& target_names,
            const std::vector<glm::mat4>& original_visualizer_world_transforms,
            const glm::mat4& local_visualizer_delta) {
            return computeNodeLocalTransforms(
                scene,
                target_names,
                original_visualizer_world_transforms,
                local_visualizer_delta,
                TransformDeltaComposition::IndividualLocal);
        }

        bool computeCombinedVisualizerWorldBounds(
            const core::Scene& scene,
            const std::vector<std::string>& target_names,
            glm::vec3& out_min,
            glm::vec3& out_max) {
            bool has_bounds = false;
            out_min = glm::vec3(std::numeric_limits<float>::max());
            out_max = glm::vec3(std::numeric_limits<float>::lowest());

            for (const auto& name : target_names) {
                const auto* node = scene.getNode(name);
                if (!node)
                    continue;

                glm::vec3 local_min, local_max;
                const bool node_has_bounds = scene.getNodeBounds(node->id, local_min, local_max);
                const glm::mat4 visualizer_world = scene_coords::nodeVisualizerWorldTransform(scene, node->id);

                if (!node_has_bounds) {
                    const glm::vec3 origin = glm::vec3(visualizer_world[3]);
                    out_min = glm::min(out_min, origin);
                    out_max = glm::max(out_max, origin);
                    has_bounds = true;
                    continue;
                }

                for (int x = 0; x < 2; ++x) {
                    for (int y = 0; y < 2; ++y) {
                        for (int z = 0; z < 2; ++z) {
                            const glm::vec3 corner(
                                x ? local_max.x : local_min.x,
                                y ? local_max.y : local_min.y,
                                z ? local_max.z : local_min.z);
                            const glm::vec3 world_corner =
                                glm::vec3(visualizer_world * glm::vec4(corner, 1.0f));
                            out_min = glm::min(out_min, world_corner);
                            out_max = glm::max(out_max, world_corner);
                        }
                    }
                }
                has_bounds = true;
            }

            return has_bounds;
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

/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#define GLM_ENABLE_EXPERIMENTAL

#include "visualizer/gui_capabilities.hpp"

#include "core/events.hpp"
#include "core/splat_data_transform.hpp"
#include "operation/undo_entry.hpp"
#include "operation/undo_history.hpp"
#include "rendering/rendering_manager.hpp"
#include "scene/scene_manager.hpp"

#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtx/euler_angles.hpp>
#include <memory>

namespace lfs::vis::cap {

    namespace {

        core::NodeId find_attached_child_node(const core::Scene& scene,
                                              const core::NodeId parent_id,
                                              const core::NodeType type) {
            if (parent_id == core::NULL_NODE)
                return core::NULL_NODE;

            const auto* const parent = scene.getNodeById(parent_id);
            if (!parent)
                return core::NULL_NODE;

            for (const core::NodeId child_id : parent->children) {
                const auto* const child = scene.getNodeById(child_id);
                if (child && child->type == type)
                    return child_id;
            }

            return core::NULL_NODE;
        }

    } // namespace

    TransformComponents decomposeTransform(const glm::mat4& matrix) {
        TransformComponents result;
        result.translation = glm::vec3(matrix[3]);

        glm::vec3 col0 = glm::vec3(matrix[0]);
        glm::vec3 col1 = glm::vec3(matrix[1]);
        glm::vec3 col2 = glm::vec3(matrix[2]);

        result.scale.x = glm::length(col0);
        result.scale.y = glm::length(col1);
        result.scale.z = glm::length(col2);

        if (result.scale.x > 0.0f)
            col0 /= result.scale.x;
        if (result.scale.y > 0.0f)
            col1 /= result.scale.y;
        if (result.scale.z > 0.0f)
            col2 /= result.scale.z;

        const glm::mat3 rotation_matrix(col0, col1, col2);
        glm::extractEulerAngleXYZ(glm::mat4(rotation_matrix), result.rotation.x, result.rotation.y, result.rotation.z);

        return result;
    }

    glm::mat4 composeTransform(const TransformComponents& components) {
        const glm::mat4 translation = glm::translate(glm::mat4(1.0f), components.translation);
        const glm::mat4 rotation = glm::eulerAngleXYZ(components.rotation.x, components.rotation.y, components.rotation.z);
        const glm::mat4 scale = glm::scale(glm::mat4(1.0f), components.scale);
        return translation * rotation * scale;
    }

    SelectionSnapshot getSelectionSnapshot(const core::Scene& scene, const int max_indices) {
        SelectionSnapshot snapshot;

        const auto mask = scene.getSelectionMask();
        if (!mask)
            return snapshot;

        auto mask_vec = mask->to_vector_uint8();
        for (size_t i = 0; i < mask_vec.size(); ++i) {
            if (mask_vec[i] == 0)
                continue;

            ++snapshot.selected_count;
            if (static_cast<int>(snapshot.indices.size()) < max_indices)
                snapshot.indices.push_back(static_cast<int64_t>(i));
        }

        snapshot.truncated = snapshot.selected_count > static_cast<int64_t>(snapshot.indices.size());
        return snapshot;
    }

    std::expected<void, std::string> clearGaussianSelection(SceneManager& scene_manager) {
        scene_manager.deselectAllGaussians();
        return {};
    }

    std::expected<void, std::string> clearNodeSelection(SceneManager& scene_manager) {
        scene_manager.clearSelection();
        return {};
    }

    std::expected<void, std::string> selectNode(SceneManager& scene_manager,
                                                const std::string& name,
                                                const std::string_view mode) {
        if (!scene_manager.getScene().getNode(name))
            return std::unexpected("Node not found: " + name);

        if (mode == "add") {
            scene_manager.addToSelection(name);
            return {};
        }
        if (mode != "replace")
            return std::unexpected("Unsupported node selection mode: " + std::string(mode));

        scene_manager.selectNode(name);
        return {};
    }

    std::expected<void, std::string> selectNodes(SceneManager& scene_manager,
                                                 const std::vector<std::string>& names,
                                                 const std::string_view mode) {
        for (const auto& name : names) {
            if (!scene_manager.getScene().getNode(name))
                return std::unexpected("Node not found: " + name);
        }

        if (mode == "add") {
            for (const auto& name : names)
                scene_manager.addToSelection(name);
            return {};
        }
        if (mode != "replace")
            return std::unexpected("Unsupported node selection mode: " + std::string(mode));

        scene_manager.selectNodes(names);
        return {};
    }

    std::expected<std::vector<std::string>, std::string> resolveTransformTargets(
        const SceneManager& scene_manager,
        const std::optional<std::string>& requested_node) {
        if (requested_node) {
            if (!scene_manager.getScene().getNode(*requested_node))
                return std::unexpected("Node not found: " + *requested_node);
            return std::vector<std::string>{*requested_node};
        }

        auto names = scene_manager.getSelectedNodeNames();
        if (names.empty())
            return std::unexpected("No node specified and no node selected");
        return names;
    }

    std::expected<void, std::string> setTransform(SceneManager& scene_manager,
                                                  const std::vector<std::string>& targets,
                                                  const std::optional<glm::vec3>& translation,
                                                  const std::optional<glm::vec3>& rotation,
                                                  const std::optional<glm::vec3>& scale,
                                                  const std::string_view undo_label) {
        if (targets.empty())
            return std::unexpected("No transform targets provided");
        if (!translation && !rotation && !scale)
            return std::unexpected("At least one transform component must be provided");

        auto entry = std::make_unique<vis::op::SceneSnapshot>(scene_manager, std::string(undo_label));
        entry->captureTransforms(targets);

        for (const auto& name : targets) {
            auto components = decomposeTransform(scene_manager.getScene().getNodeTransform(name));
            if (translation)
                components.translation = *translation;
            if (rotation)
                components.rotation = *rotation;
            if (scale)
                components.scale = *scale;
            scene_manager.setNodeTransform(name, composeTransform(components));
        }

        entry->captureAfter();
        vis::op::undoHistory().push(std::move(entry));
        return {};
    }

    std::expected<void, std::string> setTransformMatrix(SceneManager& scene_manager,
                                                        const std::vector<std::string>& targets,
                                                        const glm::mat4& transform,
                                                        const std::string_view undo_label) {
        if (targets.empty())
            return std::unexpected("No transform targets provided");

        auto entry = std::make_unique<vis::op::SceneSnapshot>(scene_manager, std::string(undo_label));
        entry->captureTransforms(targets);

        for (const auto& name : targets)
            scene_manager.setNodeTransform(name, transform);

        entry->captureAfter();
        vis::op::undoHistory().push(std::move(entry));
        return {};
    }

    std::expected<void, std::string> translateNodes(SceneManager& scene_manager,
                                                    const std::vector<std::string>& targets,
                                                    const glm::vec3& value,
                                                    const std::string_view undo_label) {
        if (targets.empty())
            return std::unexpected("No transform targets provided");

        auto entry = std::make_unique<vis::op::SceneSnapshot>(scene_manager, std::string(undo_label));
        entry->captureTransforms(targets);

        for (const auto& name : targets) {
            auto transform = scene_manager.getScene().getNodeTransform(name);
            transform[3] += glm::vec4(value, 0.0f);
            scene_manager.setNodeTransform(name, transform);
        }

        entry->captureAfter();
        vis::op::undoHistory().push(std::move(entry));
        return {};
    }

    std::expected<void, std::string> rotateNodes(SceneManager& scene_manager,
                                                 const std::vector<std::string>& targets,
                                                 const glm::vec3& value,
                                                 const std::string_view undo_label) {
        if (targets.empty())
            return std::unexpected("No transform targets provided");

        auto entry = std::make_unique<vis::op::SceneSnapshot>(scene_manager, std::string(undo_label));
        entry->captureTransforms(targets);

        const glm::mat4 rotation_delta = glm::eulerAngleXYZ(value.x, value.y, value.z);
        for (const auto& name : targets) {
            auto components = decomposeTransform(scene_manager.getScene().getNodeTransform(name));
            const glm::mat4 current_rotation =
                glm::eulerAngleXYZ(components.rotation.x, components.rotation.y, components.rotation.z);
            const glm::mat4 new_rotation = rotation_delta * current_rotation;
            glm::extractEulerAngleXYZ(new_rotation, components.rotation.x, components.rotation.y, components.rotation.z);
            scene_manager.setNodeTransform(name, composeTransform(components));
        }

        entry->captureAfter();
        vis::op::undoHistory().push(std::move(entry));
        return {};
    }

    std::expected<void, std::string> scaleNodes(SceneManager& scene_manager,
                                                const std::vector<std::string>& targets,
                                                const glm::vec3& value,
                                                const std::string_view undo_label) {
        if (targets.empty())
            return std::unexpected("No transform targets provided");

        auto entry = std::make_unique<vis::op::SceneSnapshot>(scene_manager, std::string(undo_label));
        entry->captureTransforms(targets);

        for (const auto& name : targets) {
            auto components = decomposeTransform(scene_manager.getScene().getNodeTransform(name));
            components.scale *= value;
            scene_manager.setNodeTransform(name, composeTransform(components));
        }

        entry->captureAfter();
        vis::op::undoHistory().push(std::move(entry));
        return {};
    }

    std::expected<core::NodeId, std::string> resolveCropBoxParentId(const SceneManager& scene_manager,
                                                                    const std::optional<std::string>& requested_node) {
        const auto& scene = scene_manager.getScene();
        const auto resolve = [&scene](const core::SceneNode* node) -> std::expected<core::NodeId, std::string> {
            if (!node)
                return std::unexpected("Node not found");
            if (node->type == core::NodeType::CROPBOX)
                return node->parent_id;
            if (node->type == core::NodeType::SPLAT || node->type == core::NodeType::POINTCLOUD)
                return node->id;
            return std::unexpected("Crop boxes can only target splat or pointcloud nodes");
        };

        if (requested_node)
            return resolve(scene.getNode(*requested_node));

        const auto selected_name = scene_manager.getSelectedNodeName();
        if (selected_name.empty())
            return std::unexpected("No node specified and no node selected");
        return resolve(scene.getNode(selected_name));
    }

    std::expected<core::NodeId, std::string> resolveCropBoxId(const SceneManager& scene_manager,
                                                              const std::optional<std::string>& requested_node) {
        const auto& scene = scene_manager.getScene();
        if (requested_node) {
            const auto* const node = scene.getNode(*requested_node);
            if (!node)
                return std::unexpected("Node not found: " + *requested_node);

            if (node->type == core::NodeType::CROPBOX && node->cropbox)
                return node->id;

            if (const core::NodeId attached_cropbox = find_attached_child_node(scene, node->id, core::NodeType::CROPBOX);
                attached_cropbox != core::NULL_NODE)
                return attached_cropbox;

            if (node->type == core::NodeType::SPLAT || node->type == core::NodeType::POINTCLOUD) {
                const core::NodeId cropbox_id = scene.getCropBoxForSplat(node->id);
                if (cropbox_id == core::NULL_NODE)
                    return std::unexpected("Node has no crop box: " + *requested_node);
                return cropbox_id;
            }

            return std::unexpected("Node does not reference a crop box: " + *requested_node);
        }

        const core::NodeId cropbox_id = scene_manager.getSelectedNodeCropBoxId();
        if (cropbox_id == core::NULL_NODE)
            return std::unexpected("No crop box specified and no crop box selected");
        return cropbox_id;
    }

    std::expected<core::NodeId, std::string> ensureCropBox(SceneManager& scene_manager,
                                                           RenderingManager* rendering_manager,
                                                           const core::NodeId parent_id) {
        auto& scene = scene_manager.getScene();
        const auto* const parent = scene.getNodeById(parent_id);
        if (!parent)
            return std::unexpected("Target node not found");

        if (parent->type != core::NodeType::SPLAT && parent->type != core::NodeType::POINTCLOUD)
            return std::unexpected("Crop boxes can only be attached to splat or pointcloud nodes");

        if (const core::NodeId existing = scene.getCropBoxForSplat(parent_id); existing != core::NULL_NODE) {
            if (rendering_manager) {
                auto settings = rendering_manager->getSettings();
                settings.show_crop_box = true;
                rendering_manager->updateSettings(settings);
            }
            return existing;
        }

        const std::string cropbox_name = parent->name + "_cropbox";
        const core::NodeId cropbox_id = scene.addCropBox(cropbox_name, parent_id);
        if (cropbox_id == core::NULL_NODE)
            return std::unexpected("Failed to create crop box for node: " + parent->name);

        core::CropBoxData data;
        glm::vec3 min_bounds, max_bounds;
        if (scene.getNodeBounds(parent_id, min_bounds, max_bounds)) {
            data.min = min_bounds;
            data.max = max_bounds;
        }
        data.enabled = true;
        scene.setCropBoxData(cropbox_id, data);

        if (const auto* const cropbox_node = scene.getNodeById(cropbox_id)) {
            core::events::state::PLYAdded{
                .name = cropbox_node->name,
                .node_gaussians = 0,
                .total_gaussians = scene.getTotalGaussianCount(),
                .is_visible = cropbox_node->visible,
                .parent_name = parent->name,
                .is_group = false,
                .node_type = static_cast<int>(core::NodeType::CROPBOX)}
                .emit();
        }

        if (rendering_manager) {
            auto settings = rendering_manager->getSettings();
            settings.show_crop_box = true;
            rendering_manager->updateSettings(settings);
        }

        return cropbox_id;
    }

    std::expected<void, std::string> updateCropBox(SceneManager& scene_manager,
                                                   RenderingManager* rendering_manager,
                                                   const core::NodeId cropbox_id,
                                                   const CropBoxUpdate& update) {
        auto& scene = scene_manager.getScene();
        const auto* const cropbox_node = scene.getNodeById(cropbox_id);
        if (!cropbox_node || !cropbox_node->cropbox)
            return std::unexpected("Invalid crop box target");

        const auto before_data = *cropbox_node->cropbox;
        const auto before_transform = scene_manager.getNodeTransform(cropbox_node->name);

        auto updated_data = before_data;
        auto updated_components = decomposeTransform(before_transform);

        bool cropbox_changed = false;
        bool transform_changed = false;

        if (update.min_bounds) {
            updated_data.min = *update.min_bounds;
            cropbox_changed = true;
        }
        if (update.max_bounds) {
            updated_data.max = *update.max_bounds;
            cropbox_changed = true;
        }
        if (update.has_inverse) {
            updated_data.inverse = update.inverse;
            cropbox_changed = true;
        }
        if (update.has_enabled) {
            updated_data.enabled = update.enabled;
            cropbox_changed = true;
        }
        if (update.translation) {
            updated_components.translation = *update.translation;
            transform_changed = true;
        }
        if (update.rotation) {
            updated_components.rotation = *update.rotation;
            transform_changed = true;
        }
        if (update.scale) {
            updated_components.scale = *update.scale;
            transform_changed = true;
        }

        if (cropbox_changed)
            scene.setCropBoxData(cropbox_id, updated_data);
        if (transform_changed)
            scene_manager.setNodeTransform(cropbox_node->name, composeTransform(updated_components));

        if (rendering_manager && (cropbox_changed || transform_changed))
            rendering_manager->markDirty(vis::DirtyFlag::SPLATS | vis::DirtyFlag::OVERLAY);

        if (rendering_manager && (update.has_show || update.has_use)) {
            auto settings = rendering_manager->getSettings();
            if (update.has_show)
                settings.show_crop_box = update.show;
            if (update.has_use)
                settings.use_crop_box = update.use;
            rendering_manager->updateSettings(settings);
        }

        if (cropbox_changed || transform_changed) {
            auto entry = std::make_unique<vis::op::CropBoxUndoEntry>(
                scene_manager, cropbox_node->name, before_data, before_transform);
            if (entry->hasChanges())
                vis::op::undoHistory().push(std::move(entry));
        }

        return {};
    }

    std::expected<void, std::string> fitCropBoxToParent(SceneManager& scene_manager,
                                                        RenderingManager* rendering_manager,
                                                        const core::NodeId cropbox_id,
                                                        const bool use_percentile) {
        auto& scene = scene_manager.getScene();
        const auto* const cropbox_node = scene.getNodeById(cropbox_id);
        if (!cropbox_node || cropbox_node->type != core::NodeType::CROPBOX || !cropbox_node->cropbox)
            return std::unexpected("Invalid crop box target");

        const auto* const parent = scene.getNodeById(cropbox_node->parent_id);
        if (!parent)
            return std::unexpected("Crop box parent not found");

        glm::vec3 min_bounds, max_bounds;
        bool bounds_valid = false;
        if (parent->type == core::NodeType::SPLAT && parent->model && parent->model->size() > 0) {
            bounds_valid = core::compute_bounds(*parent->model, min_bounds, max_bounds, 0.0f, use_percentile);
        } else if (parent->type == core::NodeType::POINTCLOUD && parent->point_cloud && parent->point_cloud->size() > 0) {
            bounds_valid = core::compute_bounds(*parent->point_cloud, min_bounds, max_bounds, 0.0f, use_percentile);
        }

        if (!bounds_valid)
            return std::unexpected("Cannot compute bounds for node: " + parent->name);

        const auto before_data = *cropbox_node->cropbox;
        const auto before_transform = scene_manager.getNodeTransform(cropbox_node->name);

        const glm::vec3 center = (min_bounds + max_bounds) * 0.5f;
        const glm::vec3 half_size = (max_bounds - min_bounds) * 0.5f;

        auto updated_data = before_data;
        updated_data.min = -half_size;
        updated_data.max = half_size;
        scene.setCropBoxData(cropbox_id, updated_data);
        scene.setNodeTransform(cropbox_node->name, glm::translate(glm::mat4(1.0f), center));

        if (rendering_manager)
            rendering_manager->markDirty(vis::DirtyFlag::SPLATS | vis::DirtyFlag::OVERLAY);

        auto entry = std::make_unique<vis::op::CropBoxUndoEntry>(
            scene_manager, cropbox_node->name, before_data, before_transform);
        if (entry->hasChanges())
            vis::op::undoHistory().push(std::move(entry));

        return {};
    }

    std::expected<void, std::string> resetCropBox(SceneManager& scene_manager,
                                                  RenderingManager* rendering_manager,
                                                  const core::NodeId cropbox_id) {
        auto& scene = scene_manager.getScene();
        const auto* const cropbox_node = scene.getNodeById(cropbox_id);
        if (!cropbox_node || cropbox_node->type != core::NodeType::CROPBOX || !cropbox_node->cropbox)
            return std::unexpected("Invalid crop box target");

        const auto before_data = *cropbox_node->cropbox;
        const auto before_transform = scene_manager.getNodeTransform(cropbox_node->name);

        auto reset_data = before_data;
        reset_data.min = glm::vec3(-1.0f);
        reset_data.max = glm::vec3(1.0f);
        reset_data.inverse = false;
        scene.setCropBoxData(cropbox_id, reset_data);
        scene.setNodeTransform(cropbox_node->name, glm::mat4(1.0f));

        if (rendering_manager) {
            auto settings = rendering_manager->getSettings();
            settings.use_crop_box = false;
            rendering_manager->updateSettings(settings);
            rendering_manager->markDirty(vis::DirtyFlag::SPLATS | vis::DirtyFlag::OVERLAY);
        }

        auto entry = std::make_unique<vis::op::CropBoxUndoEntry>(
            scene_manager, cropbox_node->name, before_data, before_transform);
        if (entry->hasChanges())
            vis::op::undoHistory().push(std::move(entry));

        return {};
    }

    std::expected<core::NodeId, std::string> resolveEllipsoidParentId(const SceneManager& scene_manager,
                                                                      const std::optional<std::string>& requested_node) {
        const auto& scene = scene_manager.getScene();
        const auto resolve = [&scene](const core::SceneNode* node) -> std::expected<core::NodeId, std::string> {
            if (!node)
                return std::unexpected("Node not found");
            if (node->type == core::NodeType::ELLIPSOID)
                return node->parent_id;
            if (node->type == core::NodeType::SPLAT || node->type == core::NodeType::POINTCLOUD)
                return node->id;
            return std::unexpected("Ellipsoids can only target splat or pointcloud nodes");
        };

        if (requested_node)
            return resolve(scene.getNode(*requested_node));

        const auto selected_name = scene_manager.getSelectedNodeName();
        if (selected_name.empty())
            return std::unexpected("No node specified and no node selected");
        return resolve(scene.getNode(selected_name));
    }

    std::expected<core::NodeId, std::string> resolveEllipsoidId(const SceneManager& scene_manager,
                                                                const std::optional<std::string>& requested_node) {
        const auto& scene = scene_manager.getScene();
        if (requested_node) {
            const auto* const node = scene.getNode(*requested_node);
            if (!node)
                return std::unexpected("Node not found: " + *requested_node);

            if (node->type == core::NodeType::ELLIPSOID && node->ellipsoid)
                return node->id;

            if (const core::NodeId attached_ellipsoid =
                    find_attached_child_node(scene, node->id, core::NodeType::ELLIPSOID);
                attached_ellipsoid != core::NULL_NODE)
                return attached_ellipsoid;

            if (node->type == core::NodeType::SPLAT || node->type == core::NodeType::POINTCLOUD) {
                const core::NodeId ellipsoid_id = scene.getEllipsoidForSplat(node->id);
                if (ellipsoid_id == core::NULL_NODE)
                    return std::unexpected("Node has no ellipsoid: " + *requested_node);
                return ellipsoid_id;
            }

            return std::unexpected("Node does not reference an ellipsoid: " + *requested_node);
        }

        const core::NodeId ellipsoid_id = scene_manager.getSelectedNodeEllipsoidId();
        if (ellipsoid_id == core::NULL_NODE)
            return std::unexpected("No ellipsoid specified and no ellipsoid selected");
        return ellipsoid_id;
    }

    std::expected<core::NodeId, std::string> ensureEllipsoid(SceneManager& scene_manager,
                                                             RenderingManager* rendering_manager,
                                                             const core::NodeId parent_id) {
        constexpr float CIRCUMSCRIBE_FACTOR = 1.732050808f; // sqrt(3)

        auto& scene = scene_manager.getScene();
        const auto* const parent = scene.getNodeById(parent_id);
        if (!parent)
            return std::unexpected("Target node not found");

        if (parent->type != core::NodeType::SPLAT && parent->type != core::NodeType::POINTCLOUD)
            return std::unexpected("Ellipsoids can only be attached to splat or pointcloud nodes");

        if (const core::NodeId existing = scene.getEllipsoidForSplat(parent_id); existing != core::NULL_NODE) {
            if (rendering_manager) {
                auto settings = rendering_manager->getSettings();
                settings.show_ellipsoid = true;
                rendering_manager->updateSettings(settings);
            }
            return existing;
        }

        const std::string ellipsoid_name = parent->name + "_ellipsoid";
        const core::NodeId ellipsoid_id = scene.addEllipsoid(ellipsoid_name, parent_id);
        if (ellipsoid_id == core::NULL_NODE)
            return std::unexpected("Failed to create ellipsoid for node: " + parent->name);

        core::EllipsoidData data;
        glm::vec3 min_bounds, max_bounds;
        if (scene.getNodeBounds(parent_id, min_bounds, max_bounds)) {
            const glm::vec3 center = (min_bounds + max_bounds) * 0.5f;
            const glm::vec3 half_size = (max_bounds - min_bounds) * 0.5f;
            data.radii = half_size * CIRCUMSCRIBE_FACTOR;
            scene.setNodeTransform(ellipsoid_name, glm::translate(glm::mat4(1.0f), center));
        }
        data.enabled = true;
        scene.setEllipsoidData(ellipsoid_id, data);

        if (const auto* const ellipsoid_node = scene.getNodeById(ellipsoid_id)) {
            core::events::state::PLYAdded{
                .name = ellipsoid_node->name,
                .node_gaussians = 0,
                .total_gaussians = scene.getTotalGaussianCount(),
                .is_visible = ellipsoid_node->visible,
                .parent_name = parent->name,
                .is_group = false,
                .node_type = static_cast<int>(core::NodeType::ELLIPSOID)}
                .emit();
        }

        if (rendering_manager) {
            auto settings = rendering_manager->getSettings();
            settings.show_ellipsoid = true;
            rendering_manager->updateSettings(settings);
        }

        scene.notifyMutation(core::Scene::MutationType::MODEL_CHANGED);
        return ellipsoid_id;
    }

    std::expected<void, std::string> updateEllipsoid(SceneManager& scene_manager,
                                                     RenderingManager* rendering_manager,
                                                     const core::NodeId ellipsoid_id,
                                                     const EllipsoidUpdate& update) {
        auto& scene = scene_manager.getScene();
        const auto* const ellipsoid_node = scene.getNodeById(ellipsoid_id);
        if (!ellipsoid_node || !ellipsoid_node->ellipsoid)
            return std::unexpected("Invalid ellipsoid target");

        const auto before_data = *ellipsoid_node->ellipsoid;
        const auto before_transform = scene_manager.getNodeTransform(ellipsoid_node->name);

        auto updated_data = before_data;
        auto updated_components = decomposeTransform(before_transform);

        bool ellipsoid_changed = false;
        bool transform_changed = false;

        if (update.radii) {
            updated_data.radii = glm::max(*update.radii, glm::vec3(1e-4f));
            ellipsoid_changed = true;
        }
        if (update.has_inverse) {
            updated_data.inverse = update.inverse;
            ellipsoid_changed = true;
        }
        if (update.has_enabled) {
            updated_data.enabled = update.enabled;
            ellipsoid_changed = true;
        }
        if (update.translation) {
            updated_components.translation = *update.translation;
            transform_changed = true;
        }
        if (update.rotation) {
            updated_components.rotation = *update.rotation;
            transform_changed = true;
        }
        if (update.scale) {
            updated_components.scale = *update.scale;
            transform_changed = true;
        }

        if (ellipsoid_changed)
            scene.setEllipsoidData(ellipsoid_id, updated_data);
        if (transform_changed)
            scene_manager.setNodeTransform(ellipsoid_node->name, composeTransform(updated_components));

        if (rendering_manager && (ellipsoid_changed || transform_changed))
            rendering_manager->markDirty(vis::DirtyFlag::SPLATS | vis::DirtyFlag::OVERLAY);

        if (rendering_manager && (update.has_show || update.has_use)) {
            auto settings = rendering_manager->getSettings();
            if (update.has_show)
                settings.show_ellipsoid = update.show;
            if (update.has_use)
                settings.use_ellipsoid = update.use;
            rendering_manager->updateSettings(settings);
        }

        if (ellipsoid_changed)
            scene.notifyMutation(core::Scene::MutationType::MODEL_CHANGED);

        if (ellipsoid_changed || transform_changed) {
            auto entry = std::make_unique<vis::op::EllipsoidUndoEntry>(
                scene_manager, ellipsoid_node->name, before_data, before_transform);
            if (entry->hasChanges())
                vis::op::undoHistory().push(std::move(entry));
        }

        return {};
    }

    std::expected<void, std::string> fitEllipsoidToParent(SceneManager& scene_manager,
                                                          RenderingManager* rendering_manager,
                                                          const core::NodeId ellipsoid_id,
                                                          const bool use_percentile) {
        constexpr float CIRCUMSCRIBE_FACTOR = 1.732050808f; // sqrt(3)

        auto& scene = scene_manager.getScene();
        const auto* const ellipsoid_node = scene.getNodeById(ellipsoid_id);
        if (!ellipsoid_node || ellipsoid_node->type != core::NodeType::ELLIPSOID || !ellipsoid_node->ellipsoid)
            return std::unexpected("Invalid ellipsoid target");

        const auto* const parent = scene.getNodeById(ellipsoid_node->parent_id);
        if (!parent)
            return std::unexpected("Ellipsoid parent not found");

        glm::vec3 min_bounds, max_bounds;
        bool bounds_valid = false;
        if (parent->type == core::NodeType::SPLAT && parent->model && parent->model->size() > 0) {
            bounds_valid = core::compute_bounds(*parent->model, min_bounds, max_bounds, 0.0f, use_percentile);
        } else if (parent->type == core::NodeType::POINTCLOUD && parent->point_cloud && parent->point_cloud->size() > 0) {
            bounds_valid = core::compute_bounds(*parent->point_cloud, min_bounds, max_bounds, 0.0f, use_percentile);
        }

        if (!bounds_valid)
            return std::unexpected("Cannot compute bounds for node: " + parent->name);

        const auto before_data = *ellipsoid_node->ellipsoid;
        const auto before_transform = scene_manager.getNodeTransform(ellipsoid_node->name);

        auto updated_data = before_data;
        updated_data.radii = glm::max((max_bounds - min_bounds) * 0.5f * CIRCUMSCRIBE_FACTOR, glm::vec3(1e-4f));
        scene.setEllipsoidData(ellipsoid_id, updated_data);
        scene_manager.setNodeTransform(
            ellipsoid_node->name,
            glm::translate(glm::mat4(1.0f), (min_bounds + max_bounds) * 0.5f));

        if (rendering_manager)
            rendering_manager->markDirty(vis::DirtyFlag::SPLATS | vis::DirtyFlag::OVERLAY);

        scene.notifyMutation(core::Scene::MutationType::MODEL_CHANGED);

        auto entry = std::make_unique<vis::op::EllipsoidUndoEntry>(
            scene_manager, ellipsoid_node->name, before_data, before_transform);
        if (entry->hasChanges())
            vis::op::undoHistory().push(std::move(entry));

        return {};
    }

    std::expected<void, std::string> resetEllipsoid(SceneManager& scene_manager,
                                                    RenderingManager* rendering_manager,
                                                    const core::NodeId ellipsoid_id) {
        auto& scene = scene_manager.getScene();
        const auto* const ellipsoid_node = scene.getNodeById(ellipsoid_id);
        if (!ellipsoid_node || ellipsoid_node->type != core::NodeType::ELLIPSOID || !ellipsoid_node->ellipsoid)
            return std::unexpected("Invalid ellipsoid target");

        const auto before_data = *ellipsoid_node->ellipsoid;
        const auto before_transform = scene_manager.getNodeTransform(ellipsoid_node->name);

        auto reset_data = before_data;
        reset_data.radii = glm::vec3(1.0f);
        reset_data.inverse = false;
        scene.setEllipsoidData(ellipsoid_id, reset_data);
        scene_manager.setNodeTransform(ellipsoid_node->name, glm::mat4(1.0f));

        if (rendering_manager) {
            auto settings = rendering_manager->getSettings();
            settings.use_ellipsoid = false;
            rendering_manager->updateSettings(settings);
            rendering_manager->markDirty(vis::DirtyFlag::SPLATS | vis::DirtyFlag::OVERLAY);
        }

        scene.notifyMutation(core::Scene::MutationType::MODEL_CHANGED);

        auto entry = std::make_unique<vis::op::EllipsoidUndoEntry>(
            scene_manager, ellipsoid_node->name, before_data, before_transform);
        if (entry->hasChanges())
            vis::op::undoHistory().push(std::move(entry));

        return {};
    }

} // namespace lfs::vis::cap

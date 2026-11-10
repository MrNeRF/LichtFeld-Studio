/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "training_cropbox_mask.hpp"

#include "core/assert.hpp"
#include "core/camera.hpp"
#include "core/logger.hpp"
#include "core/scene.hpp"
#include "core/splat_data.hpp"
#include "core/splat_data_transform.hpp"

#include <glm/gtc/matrix_inverse.hpp>

#include <algorithm>
#include <cmath>
#include <unordered_set>

namespace lfs::training {

    namespace {
        std::optional<core::Tensor> region_remove_mask(
            const core::Tensor& means,
            const TrainingCropBoxGeometry& geometry,
            const core::Tensor* const deleted_mask) {
            if (geometry.is_polygon()) {
                return compute_crop_polygon_remove_mask(
                    means, geometry.polygon, geometry.max.z, geometry.model_to_cropbox, geometry.inverse,
                    deleted_mask);
            }
            if (geometry.is_ellipsoid()) {
                return compute_ellipsoid_remove_mask(
                    means, geometry.ellipsoid_radii, geometry.model_to_cropbox, geometry.inverse, deleted_mask);
            }
            return compute_cropbox_remove_mask(
                means, geometry.min, geometry.max, geometry.model_to_cropbox, geometry.inverse, deleted_mask);
        }

        const core::SceneNode* resolve_roi_target_node(const core::Scene& scene) {
            if (const auto* training_node = scene.getNodeByUuid(scene.getTrainingModelNodeUuid());
                training_node && scene.getRegionOfInterestData(training_node->id)) {
                return training_node;
            }
            if (const auto* pc_node = scene.getDatasetPointCloudNode();
                pc_node && scene.getRegionOfInterestData(pc_node->id)) {
                return pc_node;
            }
            for (const auto* node : scene.getNodes()) {
                if (node &&
                    (node->type == core::NodeType::SPLAT || node->type == core::NodeType::POINTCLOUD) &&
                    node->region_of_interest && node->region_of_interest->enabled) {
                    return node;
                }
            }
            return nullptr;
        }
    } // namespace

    std::optional<TrainingCropBoxGeometry> resolve_training_cropbox_geom(
        const core::Scene& scene) {
        const auto* training_node = resolve_roi_target_node(scene);
        if (!training_node) {
            return std::nullopt;
        }

        const auto* roi = scene.getRegionOfInterestData(training_node->id);
        if (!roi || !roi->enabled) {
            return std::nullopt;
        }
        const glm::mat4 world_to_cropbox = glm::inverse(
            scene.getWorldTransform(training_node->id) * roi->local_transform);
        if (roi->shape == core::RegionOfInterestShape::Polygon) {
            if (!core::crop_polygon::isValidOutline(roi->vertices) ||
                !core::crop_polygon::isValidHalfDepth(roi->half_depth)) {
                return std::nullopt;
            }
            const glm::vec2 extents = core::crop_polygon::outlineHalfExtents(roi->vertices);
            return TrainingCropBoxGeometry{.min = glm::vec3(-extents, -roi->half_depth),
                                           .max = glm::vec3(extents, roi->half_depth),
                                           .world_to_cropbox = world_to_cropbox,
                                           .model_to_cropbox = world_to_cropbox * scene.getWorldTransform(training_node->id),
                                           .inverse = roi->inverse,
                                           .polygon = roi->vertices,
                                           .crop_model_after_training = roi->crop_model_after_training};
        }
        if (roi->shape == core::RegionOfInterestShape::Ellipsoid) {
            return TrainingCropBoxGeometry{.min = -roi->radii,
                                           .max = roi->radii,
                                           .world_to_cropbox = world_to_cropbox,
                                           .model_to_cropbox = world_to_cropbox * scene.getWorldTransform(training_node->id),
                                           .inverse = roi->inverse,
                                           .ellipsoid_radii = roi->radii,
                                           .ellipsoid = true,
                                           .crop_model_after_training = roi->crop_model_after_training};
        }
        return TrainingCropBoxGeometry{
            .min = roi->min,
            .max = roi->max,
            .world_to_cropbox = world_to_cropbox,
            .model_to_cropbox =
                world_to_cropbox * scene.getWorldTransform(training_node->id),
            .inverse = roi->inverse,
            .crop_model_after_training = roi->crop_model_after_training};
    }

    bool training_crop_region_contains(
        const TrainingCropBoxGeometry& geometry,
        const glm::vec3& local_point) {
        if (!glm::all(glm::greaterThanEqual(local_point, geometry.min)) ||
            !glm::all(glm::lessThanEqual(local_point, geometry.max))) {
            return false;
        }
        if (geometry.is_ellipsoid()) {
            const glm::vec3 normalized = local_point / geometry.ellipsoid_radii;
            return glm::dot(normalized, normalized) <= 1.0f;
        }
        return !geometry.is_polygon() || core::crop_polygon::containsPoint(geometry.polygon, glm::vec2(local_point));
    }

    std::array<std::array<float, 3>, 8> training_cropbox_model_corners(
        const TrainingCropBoxGeometry& geometry) {
        const glm::mat4 cropbox_to_model = glm::inverse(geometry.model_to_cropbox);
        std::array<std::array<float, 3>, 8> corners{};
        for (size_t corner = 0; corner < corners.size(); ++corner) {
            const glm::vec3 local{corner & 1 ? geometry.max.x : geometry.min.x,
                                  corner & 2 ? geometry.max.y : geometry.min.y,
                                  corner & 4 ? geometry.max.z : geometry.min.z};
            const glm::vec4 model = cropbox_to_model * glm::vec4(local, 1.0f);
            corners[corner] = {model.x, model.y, model.z};
        }
        return corners;
    }

    std::optional<TrainingCropBoxGeometry> resolve_training_cropbox_loss_geom(
        const core::Scene& scene,
        const float outside_weight) {
        LFS_ASSERT_MSG(
            std::isfinite(outside_weight) &&
                outside_weight >= 0.0f &&
                outside_weight <= 1.0f,
            "crop box loss weight must be finite and within [0, 1]");
        if (outside_weight == 1.0f) {
            return std::nullopt;
        }
        return resolve_training_cropbox_geom(scene);
    }

    std::optional<core::Tensor> compute_cropbox_remove_mask(
        const core::Tensor& means,
        const glm::vec3& crop_min,
        const glm::vec3& crop_max,
        const glm::mat4& points_to_cropbox,
        const bool inverse,
        const core::Tensor* const deleted_mask) {
        if (!means.is_valid()) {
            return std::nullopt;
        }
        LFS_ASSERT_MSG(means.dtype() == core::DataType::Float32, "crop box means must be Float32");
        LFS_ASSERT_MSG(means.ndim() == 2, "crop box means must be a 2D tensor");
        LFS_ASSERT_MSG(means.size(1) >= 3, "crop box means must have at least three columns");
        if (means.size(0) == 0) {
            return std::nullopt;
        }

        auto inside_mask =
            core::compute_cropbox_mask(means, crop_min, crop_max, points_to_cropbox);
        if (!inside_mask.is_valid() || inside_mask.numel() == 0) {
            return std::nullopt;
        }

        auto remove_mask = inverse ? inside_mask : inside_mask.logical_not();
        if (deleted_mask) {
            LFS_ASSERT_MSG(deleted_mask->is_valid(), "crop box deleted mask must be valid");
            LFS_ASSERT_MSG(
                deleted_mask->numel() == remove_mask.numel(),
                "crop box deleted mask must match the means row count");
            remove_mask = remove_mask.logical_and(deleted_mask->logical_not());
        }

        return remove_mask;
    }

    std::optional<core::Tensor> compute_crop_polygon_remove_mask(
        const core::Tensor& means,
        const std::vector<glm::vec2>& vertices,
        const float half_depth,
        const glm::mat4& points_to_polygon,
        const bool inverse,
        const core::Tensor* const deleted_mask) {
        if (!means.is_valid()) {
            return std::nullopt;
        }
        LFS_ASSERT_MSG(means.dtype() == core::DataType::Float32, "crop polygon means must be Float32");
        LFS_ASSERT_MSG(means.ndim() == 2, "crop polygon means must be a 2D tensor");
        LFS_ASSERT_MSG(means.size(1) >= 3, "crop polygon means must have at least three columns");
        if (means.size(0) == 0) {
            return std::nullopt;
        }

        const auto inside_mask = core::compute_crop_polygon_mask(means, vertices, half_depth, points_to_polygon);
        auto remove_mask = inverse ? inside_mask : inside_mask.logical_not();
        if (deleted_mask) {
            LFS_ASSERT_MSG(deleted_mask->is_valid(), "crop polygon deleted mask must be valid");
            LFS_ASSERT_MSG(
                deleted_mask->numel() == remove_mask.numel(),
                "crop polygon deleted mask must match the means row count");
            remove_mask = remove_mask.logical_and(deleted_mask->logical_not());
        }

        return remove_mask;
    }

    std::optional<core::Tensor> compute_ellipsoid_remove_mask(
        const core::Tensor& means,
        const glm::vec3& radii,
        const glm::mat4& points_to_ellipsoid,
        const bool inverse,
        const core::Tensor* const deleted_mask) {
        if (!means.is_valid() || means.size(0) == 0)
            return std::nullopt;
        const auto device = means.device();
        const auto transform = core::Tensor::from_vector(
            {points_to_ellipsoid[0][0], points_to_ellipsoid[1][0], points_to_ellipsoid[2][0], points_to_ellipsoid[3][0],
             points_to_ellipsoid[0][1], points_to_ellipsoid[1][1], points_to_ellipsoid[2][1], points_to_ellipsoid[3][1],
             points_to_ellipsoid[0][2], points_to_ellipsoid[1][2], points_to_ellipsoid[2][2], points_to_ellipsoid[3][2],
             points_to_ellipsoid[0][3], points_to_ellipsoid[1][3], points_to_ellipsoid[2][3], points_to_ellipsoid[3][3]},
            {4, 4}, device);
        const auto local = transform.mm(means.cat(core::Tensor::ones({means.size(0), 1}, device), 1).t()).t();
        const auto x = local.slice(1, 0, 1).squeeze(1) / radii.x;
        const auto y = local.slice(1, 1, 2).squeeze(1) / radii.y;
        const auto z = local.slice(1, 2, 3).squeeze(1) / radii.z;
        auto remove = inverse ? (x * x + y * y + z * z <= 1.0f)
                              : (x * x + y * y + z * z > 1.0f);
        if (deleted_mask)
            remove = remove.logical_and(deleted_mask->logical_not());
        return remove;
    }

    std::optional<core::Tensor> compute_training_cropbox_remove_mask(
        const core::Scene& scene,
        const core::SplatData& model) {
        const auto geometry = resolve_training_cropbox_geom(scene);
        if (!geometry) {
            return std::nullopt;
        }

        const auto& deleted = model.deleted();
        const core::Tensor* const deleted_mask =
            model.has_deleted_mask() && deleted.is_valid() ? &deleted : nullptr;

        return region_remove_mask(model.means(), *geometry, deleted_mask);
    }

    std::optional<core::Tensor> training_cropbox_crop_mask(
        const core::SplatData& model,
        const TrainingCropBoxGeometry& geometry) {
        return region_remove_mask(model.means(), geometry, nullptr);
    }

    std::optional<core::SplatData> crop_copy_to_training_cropbox(
        const core::SplatData& model,
        const TrainingCropBoxGeometry& geometry) {
        // Deleted rows outside the region need no copy: a model cropped by soft deletion (MCMC) exports as it is.
        const core::Tensor* const deleted = model.deleted_mask_matches_size() ? &model.deleted() : nullptr;
        const auto removed = region_remove_mask(model.means(), geometry, deleted);
        if (!removed || removed->sum_scalar() == 0)
            return std::nullopt;
        auto keep = removed->logical_not();
        if (deleted)
            keep = keep.logical_and(deleted->logical_not());
        return core::extract_by_mask(model, keep);
    }

    namespace {
        struct RoiCameraSnapshot {
            glm::vec3 center{0.0F};
            float focal_x = 0.0F;
            float focal_y = 0.0F;
            float center_x = 0.0F;
            float center_y = 0.0F;
            float width = 0.0F;
            float height = 0.0F;
            glm::mat4 world_to_camera{1.0F};
            bool pose_valid = false;
        };

        [[nodiscard]] std::vector<glm::vec3> roi_camera_coverage_points(
            const core::Scene& scene, const core::SceneNode& target,
            const core::RegionOfInterestData& roi) {
            std::vector<glm::vec3> points;
            const glm::mat4 to_world = scene.getWorldTransform(target.id) * roi.local_transform;
            const auto append = [&](const glm::vec3 local) {
                points.push_back(glm::vec3(to_world * glm::vec4(local, 1.0F)));
            };
            switch (roi.shape) {
            case core::RegionOfInterestShape::Box:
                points.reserve(8);
                for (const float x : {roi.min.x, roi.max.x})
                    for (const float y : {roi.min.y, roi.max.y})
                        for (const float z : {roi.min.z, roi.max.z})
                            append({x, y, z});
                break;
            case core::RegionOfInterestShape::Ellipsoid:
                points.reserve(26);
                for (const float x : {-1.0F, 0.0F, 1.0F})
                    for (const float y : {-1.0F, 0.0F, 1.0F})
                        for (const float z : {-1.0F, 0.0F, 1.0F}) {
                            const glm::vec3 direction{x, y, z};
                            if (glm::dot(direction, direction) > 0.0F)
                                append(glm::normalize(direction) * roi.radii);
                        }
                break;
            case core::RegionOfInterestShape::Polygon:
                points.reserve(roi.vertices.size() * 2);
                for (const auto& vertex : roi.vertices) {
                    append({vertex.x, vertex.y, -roi.half_depth});
                    append({vertex.x, vertex.y, roi.half_depth});
                }
                break;
            }
            return points;
        }

        [[nodiscard]] std::optional<RoiCameraSnapshot> make_roi_camera_snapshot(
            const core::Scene& scene, const core::SceneNode& node) {
            if (!node.camera || node.camera->camera_model_type() != core::CameraModelType::PINHOLE)
                return std::nullopt;
            RoiCameraSnapshot snapshot;
            const auto rotation = node.camera->R().cpu().contiguous().to_vector();
            const auto translation = node.camera->T().cpu().contiguous().to_vector();
            snapshot.focal_x = node.camera->focal_x();
            snapshot.focal_y = node.camera->focal_y();
            snapshot.center_x = node.camera->center_x();
            snapshot.center_y = node.camera->center_y();
            snapshot.width = static_cast<float>(node.camera->camera_width());
            snapshot.height = static_cast<float>(node.camera->camera_height());
            if (rotation.size() < 9 || translation.size() < 3 || snapshot.focal_x <= 0.0F ||
                snapshot.focal_y <= 0.0F || snapshot.width <= 0.0F || snapshot.height <= 0.0F) {
                return std::nullopt;
            }
            glm::mat4 local_world_to_camera(1.0F);
            for (int row = 0; row < 3; ++row) {
                for (int column = 0; column < 3; ++column)
                    local_world_to_camera[column][row] = rotation[row * 3 + column];
                local_world_to_camera[3][row] = translation[row];
            }
            const glm::mat4 camera_to_world = scene.getWorldTransform(node.id) * glm::inverse(local_world_to_camera);
            snapshot.world_to_camera = glm::inverse(camera_to_world);
            snapshot.center = glm::vec3(camera_to_world[3]);
            snapshot.pose_valid = std::isfinite(snapshot.center.x) && std::isfinite(snapshot.center.y) &&
                                  std::isfinite(snapshot.center.z);
            return snapshot;
        }

        [[nodiscard]] float roi_camera_crop_coverage(
            const RoiCameraSnapshot& camera,
            const std::vector<glm::vec3>& points,
            const bool center_is_inside) {
            if (!camera.pose_valid || points.empty())
                return 0.0F;
            if (center_is_inside)
                return 100.0F;
            constexpr float near_z = 1.e-6F;
            float min_x = camera.width, max_x = 0.0F, min_y = camera.height, max_y = 0.0F;
            bool projected = false;
            for (const auto& point : points) {
                const glm::vec3 p = glm::vec3(camera.world_to_camera * glm::vec4(point, 1.0F));
                if (p.z < near_z)
                    continue;
                const float x = camera.focal_x * p.x / p.z + camera.center_x;
                const float y = camera.focal_y * p.y / p.z + camera.center_y;
                min_x = std::min(min_x, x);
                max_x = std::max(max_x, x);
                min_y = std::min(min_y, y);
                max_y = std::max(max_y, y);
                projected = true;
            }
            if (!projected)
                return 0.0F;
            const float width = std::max(0.0F, std::min(camera.width, max_x) - std::max(0.0F, min_x));
            const float height = std::max(0.0F, std::min(camera.height, max_y) - std::max(0.0F, min_y));
            return 100.0F * width * height / std::max(camera.width * camera.height, 1.e-6F);
        }
    } // namespace

    bool select_relevant_roi_cameras(core::Scene& scene) {
        const auto* target = resolve_roi_target_node(scene);
        const auto* roi = target ? scene.getRegionOfInterestData(target->id) : nullptr;
        if (!target || !roi || !roi->enabled || !roi->select_relevant_cameras)
            return false;
        const auto geometry = resolve_training_cropbox_geom(scene);
        const auto points = roi_camera_coverage_points(scene, *target, *roi);
        if (!geometry || points.empty()) {
            LOG_WARN("ROI camera selection skipped: ROI geometry is invalid");
            return false;
        }

        std::vector<std::pair<core::NodeId, RoiCameraSnapshot>> cameras;
        for (const auto* node : scene.getNodes()) {
            if (!node || node->type != core::NodeType::CAMERA || !node->camera)
                continue;
            const auto snapshot = make_roi_camera_snapshot(scene, *node);
            if (!snapshot) {
                LOG_WARN("ROI camera selection skipped: all training cameras must use a valid pinhole camera");
                return false;
            }
            cameras.emplace_back(node->id, *snapshot);
        }
        if (cameras.empty()) {
            LOG_WARN("ROI camera selection skipped: no cameras are available");
            return false;
        }

        std::unordered_set<core::NodeId> selected;
        const float threshold = std::clamp(roi->camera_coverage_percent, 0.0F, 100.0F);
        for (const auto& [id, camera] : cameras) {
            const glm::vec3 local_center = glm::vec3(
                geometry->world_to_cropbox * glm::vec4(camera.center, 1.0F));
            float coverage = roi_camera_crop_coverage(
                camera, points, training_crop_region_contains(*geometry, local_center));
            if (roi->inverse)
                coverage = 100.0F - coverage;
            if (coverage > 0.0F && coverage >= threshold)
                selected.insert(id);
        }
        if (selected.empty()) {
            LOG_WARN("ROI camera selection found no camera meeting the {:.1f}% coverage threshold; preserving the existing camera selection", threshold);
            return false;
        }
        static_cast<void>(scene.setCameraTrainingSelection(selected));
        return true;
    }

    void add_manual_crop_from_roi(core::Scene& scene) {
        const auto training_id = scene.getTrainingModelNodeId();
        const auto* training_node = scene.getNodeById(training_id);
        const auto* roi = scene.getRegionOfInterestData(training_id);
        if (!training_node || !roi || !roi->enabled || roi->crop_model_after_training)
            return;

        // Never repurpose a user's existing Crop helper. The fallback is
        // only an optional, disabled manual action after successful ROI training.
        switch (roi->shape) {
        case core::RegionOfInterestShape::Box: {
            if (scene.getCropBoxForSplat(training_id) != core::NULL_NODE)
                return;
            const auto id = scene.addCropBox(training_node->name + "_roi_crop", training_id);
            auto* data = scene.getCropBoxData(id);
            const auto* node = scene.getNodeById(id);
            if (!data || !node)
                return;
            data->min = roi->min;
            data->max = roi->max;
            data->inverse = roi->inverse;
            data->enabled = false;
            scene.setNodeTransform(node->id, roi->local_transform);
            scene.notifyMutation(core::Scene::MutationType::TRANSFORM_CHANGED);
            return;
        }
        case core::RegionOfInterestShape::Ellipsoid: {
            if (scene.getEllipsoidForSplat(training_id) != core::NULL_NODE)
                return;
            const auto id = scene.addEllipsoid(training_node->name + "_roi_ellipsoid", training_id);
            auto* data = scene.getEllipsoidData(id);
            const auto* node = scene.getNodeById(id);
            if (!data || !node)
                return;
            data->radii = roi->radii;
            data->inverse = roi->inverse;
            data->enabled = false;
            scene.setNodeTransform(node->id, roi->local_transform);
            scene.notifyMutation(core::Scene::MutationType::TRANSFORM_CHANGED);
            return;
        }
        case core::RegionOfInterestShape::Polygon: {
            if (scene.getCropPolygonForSplat(training_id) != core::NULL_NODE ||
                !core::crop_polygon::isValidOutline(roi->vertices) ||
                !core::crop_polygon::isValidHalfDepth(roi->half_depth))
                return;
            core::CropPolygonData data;
            data.vertices = roi->vertices;
            data.half_depth = roi->half_depth;
            data.inverse = roi->inverse;
            data.enabled = false;
            static_cast<void>(scene.addCropPolygon(
                training_node->name + "_roi_polygon", training_id, std::move(data), roi->local_transform));
            return;
        }
        }
    }

} // namespace lfs::training

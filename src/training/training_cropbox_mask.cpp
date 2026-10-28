/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "training_cropbox_mask.hpp"

#include "core/assert.hpp"
#include "core/scene.hpp"
#include "core/splat_data.hpp"
#include "core/splat_data_transform.hpp"

#include <glm/gtc/matrix_inverse.hpp>

#include <cmath>

namespace lfs::training {

    namespace {
        std::optional<TrainingCropBoxGeometry> resolve_training_crop_polygon_geom(
            const core::Scene& scene,
            const core::NodeId training_node_id) {
            const core::NodeId polygon_id = scene.getCropPolygonForSplat(training_node_id);
            const auto* polygon = scene.getCropPolygonData(polygon_id);
            if (!polygon || !polygon->enabled || !core::crop_polygon::isValidOutline(polygon->vertices) ||
                !core::crop_polygon::isValidHalfDepth(polygon->half_depth)) {
                return std::nullopt;
            }

            const glm::vec2 extents = core::crop_polygon::outlineHalfExtents(polygon->vertices);
            const glm::mat4 world_to_polygon = glm::inverse(scene.getWorldTransform(polygon_id));
            return TrainingCropBoxGeometry{
                .min = glm::vec3(-extents, -polygon->half_depth),
                .max = glm::vec3(extents, polygon->half_depth),
                .world_to_cropbox = world_to_polygon,
                .model_to_cropbox = world_to_polygon * scene.getWorldTransform(training_node_id),
                .inverse = polygon->inverse,
                .polygon = polygon->vertices};
        }

        std::optional<core::Tensor> region_remove_mask(
            const core::Tensor& means,
            const TrainingCropBoxGeometry& geometry,
            const core::Tensor* const deleted_mask) {
            if (geometry.is_polygon()) {
                return compute_crop_polygon_remove_mask(
                    means, geometry.polygon, geometry.max.z, geometry.model_to_cropbox, geometry.inverse,
                    deleted_mask);
            }
            return compute_cropbox_remove_mask(
                means, geometry.min, geometry.max, geometry.model_to_cropbox, geometry.inverse, deleted_mask);
        }
    } // namespace

    std::optional<TrainingCropBoxGeometry> resolve_training_cropbox_geom(
        const core::Scene& scene) {
        const auto* training_node = scene.getNodeByUuid(scene.getTrainingModelNodeUuid());
        if (!training_node) {
            return std::nullopt;
        }

        const core::NodeId cropbox_id = scene.getCropBoxForSplat(training_node->id);
        if (cropbox_id == core::NULL_NODE) {
            return resolve_training_crop_polygon_geom(scene, training_node->id);
        }

        const auto* cropbox = scene.getCropBoxData(cropbox_id);
        if (!cropbox || !cropbox->enabled) {
            return std::nullopt;
        }

        const glm::mat4 world_to_cropbox =
            glm::inverse(scene.getWorldTransform(cropbox_id));
        return TrainingCropBoxGeometry{
            .min = cropbox->min,
            .max = cropbox->max,
            .world_to_cropbox = world_to_cropbox,
            .model_to_cropbox =
                world_to_cropbox * scene.getWorldTransform(training_node->id),
            .inverse = cropbox->inverse};
    }

    bool training_crop_region_contains(
        const TrainingCropBoxGeometry& geometry,
        const glm::vec3& local_point) {
        if (!glm::all(glm::greaterThanEqual(local_point, geometry.min)) ||
            !glm::all(glm::lessThanEqual(local_point, geometry.max))) {
            return false;
        }
        return !geometry.is_polygon() ||
               core::crop_polygon::containsPoint(geometry.polygon, glm::vec2(local_point));
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

} // namespace lfs::training

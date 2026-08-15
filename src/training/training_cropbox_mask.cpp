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

    std::optional<TrainingCropBoxGeometry> resolve_training_cropbox_geom(
        const core::Scene& scene) {
        const auto* training_node = scene.getNodeByUuid(scene.getTrainingModelNodeUuid());
        if (!training_node) {
            return std::nullopt;
        }

        const core::NodeId cropbox_id = scene.getCropBoxForSplat(training_node->id);
        if (cropbox_id == core::NULL_NODE) {
            return std::nullopt;
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

        return compute_cropbox_remove_mask(
            model.means(),
            geometry->min,
            geometry->max,
            geometry->model_to_cropbox,
            geometry->inverse,
            deleted_mask);
    }

} // namespace lfs::training

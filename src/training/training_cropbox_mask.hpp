/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include "core/tensor.hpp"

#include <glm/glm.hpp>
#include <optional>

namespace lfs::core {
    class Scene;
    class SplatData;
} // namespace lfs::core

namespace lfs::training {

    struct TrainingCropBoxGeometry {
        glm::vec3 min;
        glm::vec3 max;
        glm::mat4 world_to_cropbox;
        glm::mat4 model_to_cropbox;
        bool inverse = false;
    };

    [[nodiscard]] std::optional<TrainingCropBoxGeometry> resolve_training_cropbox_geom(
        const core::Scene& scene);

    [[nodiscard]] std::optional<TrainingCropBoxGeometry> resolve_training_cropbox_loss_geom(
        const core::Scene& scene,
        float outside_weight);

    [[nodiscard]] std::optional<core::Tensor> compute_cropbox_remove_mask(
        const core::Tensor& means,
        const glm::vec3& crop_min,
        const glm::vec3& crop_max,
        const glm::mat4& points_to_cropbox,
        bool inverse,
        const core::Tensor* deleted_mask = nullptr);

    [[nodiscard]] std::optional<core::Tensor> compute_training_cropbox_remove_mask(
        const core::Scene& scene,
        const core::SplatData& model);

} // namespace lfs::training

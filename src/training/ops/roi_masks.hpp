/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#pragma once
#include "lfs/training/ops/types.hpp"
#include <glm/glm.hpp>

namespace lfs::training {
    struct RoiBox {
        glm::mat4 model_to_cropbox{1.0f};
        glm::vec3 min{0.0f}, max{0.0f};
        bool inverse = false;
    };

    // Largest stretch of the model-to-box transform: box units per model unit.
    [[nodiscard]] float box_units_per_model_unit(const glm::mat4& model_to_cropbox);

    // Marks splats that no listed camera sees inside, in front of or behind the box (outside
    // it when inverse). means [N,3] model frame; scaling_raw [N,3] log scales or empty, which
    // grows the box by each splat's 3-sigma footprint; deleted [N] bool or empty; cameras
    // [M,3] camera centres in box space. rejected and, when valid, in_roi are [N] bool:
    // in_roi marks centres inside the ROI (outside the box when inverse). Deleted rows stay false.
    void roi_reject_mask(gpu_ops::In means, gpu_ops::In scaling_raw, gpu_ops::In deleted,
                         gpu_ops::In cameras_in_box, const RoiBox& box,
                         gpu_ops::Out rejected, gpu_ops::Out in_roi);
} // namespace lfs::training

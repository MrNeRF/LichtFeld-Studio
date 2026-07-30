/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include "core/tensor.hpp"

namespace lfs::training::losses {

    [[nodiscard]] lfs::core::Tensor compose_pixel_loss_weights(
        const lfs::core::Tensor& user_weight,
        const lfs::core::Tensor& roi_weight);

    struct MaskOpacityPenalty {
        lfs::core::Tensor loss;
        lfs::core::Tensor grad_alpha;
    };

    [[nodiscard]] MaskOpacityPenalty compute_mask_opacity_penalty(
        const lfs::core::Tensor& alpha,
        const lfs::core::Tensor& penalty_weight,
        const lfs::core::Tensor& pixel_weight,
        float scale);

} // namespace lfs::training::losses

/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include "core/assert.hpp"
#include "training/losses/mask_loss.hpp"

#include <cmath>

namespace lfs::training::losses::test_reference {

    [[nodiscard]] inline lfs::core::Tensor compose_pixel_loss_weights_reference(
        const lfs::core::Tensor& user_weight,
        const lfs::core::Tensor& roi_weight) {
        if (!user_weight.is_valid()) {
            return roi_weight;
        }
        if (!roi_weight.is_valid()) {
            return user_weight;
        }

        LFS_ASSERT_MSG(
            user_weight.ndim() == 2 && roi_weight.ndim() == 2,
            "pixel loss weights must be 2D");
        LFS_ASSERT_MSG(
            user_weight.shape() == roi_weight.shape(),
            "user and ROI pixel loss weights must have matching shapes");
        LFS_ASSERT_MSG(
            roi_weight.dtype() == lfs::core::DataType::Float32,
            "ROI pixel loss weight must be Float32");

        const auto user_float =
            user_weight.dtype() == lfs::core::DataType::UInt8 ||
                    user_weight.dtype() == lfs::core::DataType::Bool
                ? user_weight.gt(0).to(lfs::core::DataType::Float32)
                : user_weight;
        LFS_ASSERT_MSG(
            user_float.dtype() == lfs::core::DataType::Float32,
            "user pixel loss weight must be Bool, UInt8, or Float32");
        return user_float * roi_weight;
    }

    [[nodiscard]] inline MaskOpacityPenalty compute_mask_opacity_penalty_reference(
        const lfs::core::Tensor& alpha,
        const lfs::core::Tensor& penalty_weight,
        const lfs::core::Tensor& pixel_weight,
        const float scale) {
        LFS_ASSERT_MSG(
            alpha.is_valid() && penalty_weight.is_valid(),
            "mask opacity penalty inputs must be valid");
        LFS_ASSERT_MSG(
            alpha.dtype() == lfs::core::DataType::Float32 &&
                penalty_weight.dtype() == lfs::core::DataType::Float32,
            "mask opacity penalty inputs must be Float32");
        LFS_ASSERT_MSG(
            alpha.shape() == penalty_weight.shape(),
            "mask opacity alpha and penalty weights must have matching shapes");
        LFS_ASSERT_MSG(
            std::isfinite(scale) && scale >= 0.0f,
            "mask opacity penalty scale must be finite and nonnegative");

        auto effective_weight = penalty_weight;
        if (pixel_weight.is_valid()) {
            LFS_ASSERT_MSG(
                pixel_weight.dtype() == lfs::core::DataType::Float32 &&
                    pixel_weight.shape() == alpha.shape(),
                "mask opacity pixel weight must be matching Float32");
            effective_weight = effective_weight * pixel_weight;
        }

        const float gradient_scale = scale / static_cast<float>(alpha.numel());
        return MaskOpacityPenalty{
            .loss = (alpha * effective_weight).mean() * scale,
            .grad_alpha = effective_weight * gradient_scale};
    }

} // namespace lfs::training::losses::test_reference

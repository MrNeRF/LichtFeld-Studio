/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include "core/export.hpp"
#include "core/tensor.hpp"

namespace lfs::training::kernels {

    inline constexpr int GRADIENT_LOSS_START_STEP = 2000;
    inline constexpr float GRADIENT_LOSS_EPSILON = 0.001f;

    struct LFS_CUDA_API GradientResidualWorkspace {
        lfs::core::Tensor coefficients;
        lfs::core::Tensor partial;
        lfs::core::Tensor totals;
        lfs::core::Tensor loss;
        void ensure_size(size_t pixels);
    };

    LFS_CUDA_API lfs::core::Tensor gradient_residual_loss_gradient(
        const lfs::core::Tensor& image,
        const lfs::core::Tensor& target,
        const lfs::core::Tensor& pixel_weight,
        lfs::core::Tensor& gradient,
        float weight,
        GradientResidualWorkspace& workspace);

} // namespace lfs::training::kernels

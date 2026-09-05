/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once
#include "core/error.hpp"
#include "core/tensor.hpp"
namespace lfs::training::losses {
    struct EffectiveRankRegularization {
        struct Params {
            float erank_weight = 0.01f;
            float thin_scale_weight = 1.0f;
            float epsilon = 1e-7f;
        };
        // Scalar loss stays on CUDA; gradients accumulate into grad_log_scales.
        static lfs::Result<core::Tensor> forward(const core::Tensor& log_scales,
                                                 core::Tensor& grad_log_scales,
                                                 const Params& params,
                                                 const core::Tensor& active = {},
                                                 const core::Tensor& frozen = {},
                                                 size_t active_count = 0);
    };
} // namespace lfs::training::losses

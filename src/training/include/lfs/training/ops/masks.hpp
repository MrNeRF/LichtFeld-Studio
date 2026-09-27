/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include "lfs/training/ops/domain_types.hpp"
#include "lfs/training/ops/types.hpp"

namespace lfs::training::kernels {
    /// SegmentAndIgnore band bounds for Float32 masks in [0,1].
    /// Midpoints between adjacent eight-bit levels preserve their classification
    /// despite floating-point error in the normalization by 255.
    inline constexpr float kMaskKeepMin = 250.5f / 255.0f;
    inline constexpr float kMaskSegmentMin = 127.5f / 255.0f;
} // namespace lfs::training::kernels

namespace lfs::gpu_ops {
    struct MaskOps {
        void (*photometric_weight)(In mask, In roi, Out weight, MaskPhotoMode);
        void (*opacity_penalty)(In alpha, In mask, In roi, Out grad_alpha,
                                Out reduction_temp, Out loss, MaskOpacityMode, float power, float scale);
        void (*alpha_consistency)(In alpha, In mask, In roi, Out grad_alpha,
                                  Out reduction_temp, Out loss, float weight);
    };
} // namespace lfs::gpu_ops

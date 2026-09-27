/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include "lfs/training/ops/domain_types.hpp"
#include "lfs/training/ops/types.hpp"

namespace lfs::gpu_ops {
    struct MaskOps {
        void (*photometric_weight)(In mask, In roi, Out weight, MaskPhotoMode);
        void (*opacity_penalty)(In alpha, In mask, In roi, Out grad_alpha,
                                Out reduction_temp, Out loss, MaskOpacityMode, float power, float scale);
        void (*alpha_consistency)(In alpha, In mask, In roi, Out grad_alpha,
                                  Out reduction_temp, Out loss, float weight);
    };
} // namespace lfs::gpu_ops

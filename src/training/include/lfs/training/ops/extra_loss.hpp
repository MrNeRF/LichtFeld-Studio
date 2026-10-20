/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include "lfs/training/ops/types.hpp"

namespace lfs::gpu_ops {
    enum class Regularizer { Scale,
                             Opacity };

    struct ExtraLossOps {
        void (*regularize)(In raw_parameter, Out gradient, Out loss, Out reduction_temp,
                           Regularizer, float weight);
    };
} // namespace lfs::gpu_ops

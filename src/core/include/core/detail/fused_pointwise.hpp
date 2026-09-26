/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#pragma once

#include <cstdint>

namespace lfs::core::tensor_ops {

    static constexpr int FUSED_POINTWISE_MAX_OPS = 16;

    struct FusedPointwiseOp {
        uint8_t kind = 0;
        float scalar = 0.0f;
        const float* rhs = nullptr;
    };

    struct FusedPointwiseOpChain {
        FusedPointwiseOp ops[FUSED_POINTWISE_MAX_OPS];
        int num_ops = 0;
    };

} // namespace lfs::core::tensor_ops

/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include <cuda_runtime.h>

namespace lfs::core {

    // Evaluate from the raw logit: sigmoid(x) * (1 - sigmoid(x)) loses the
    // positive-tail gradient as soon as sigmoid(x) rounds to one. The absolute
    // value also avoids overflowing exp(-x) for large negative logits.
    // Related precision loss: Barron et al., "A Fix for Softmax and Cross-Entropy",
    // Section 2, https://jonbarron.info/softmaxfix/
    __device__ __forceinline__ float sigmoid_derivative_from_logit(const float x) {
        const float e = expf(-fabsf(x));
        const float denominator = 1.0f + e;
        return e / (denominator * denominator);
    }

} // namespace lfs::core

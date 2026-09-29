/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include "core/nn/ops.hpp"

namespace lfs::core::nn {
    struct RGBConvParams {
        std::array<float, 3> shift, scale;
        bool official_scaling;
    };
    struct PoolReduceParams {
        int y0, y1, x0, x1;
        float inverse_count;
    };

    // Optional per-model execution binding. The model owns all buffers and views.
    struct LpipsDispatch {
        void (*weight_taps)(const Tensor&, Tensor&);
        void (*rgb_conv)(const Tensor&, const Tensor&, const Tensor&, Tensor&, const RGBConvParams&);
        void (*convolution)(const Tensor&, const Tensor&, const Tensor&, const Tensor&,
                            Tensor&, Tensor&, const Conv2dParams&);
        void (*pool_reduce)(const Tensor&, const Tensor&, const Tensor&, Tensor&, Tensor&, Tensor&,
                            const PoolReduceParams&);
        // Run exclusively on a worker queue to overlap independent CPU metrics.
        bool prefer_independent_queue = false;
    };

} // namespace lfs::core::nn

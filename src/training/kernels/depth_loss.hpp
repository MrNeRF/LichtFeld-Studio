/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include <cstddef>
#include <cuda_runtime.h>

namespace lfs::training::kernels {

    enum class DepthLossMode : int {
        PearsonAbs = 0,
        AdaptiveWarpedL1 = 1,
    };

    [[nodiscard]] size_t depth_loss_partial_count(size_t num_pixels);

    void launch_depth_loss(
        const float* rendered_depth_accum,
        const float* rendered_alpha_accum,
        const float* target_depth,
        float* grad_depth,
        float* loss_out,
        float* partial_sums,
        size_t num_pixels,
        float weight,
        DepthLossMode mode,
        cudaStream_t stream = nullptr);

} // namespace lfs::training::kernels

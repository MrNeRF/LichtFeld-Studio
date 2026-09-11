/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include <cstdint>
#include <cuda_runtime.h>

namespace gsplat_lfs {

    void launch_relocation_kernel(
        float* opacities,      // [N] - modified in-place
        float* scales,         // [N, 3] - modified in-place
        const int32_t* ratios, // [N] - integer split counts
        const float* binoms,   // [n_max, n_max]
        int64_t N,
        int32_t n_max,
        float min_opacity,
        cudaStream_t stream = nullptr);

    void launch_add_noise_kernel(
        float* raw_opacities, // [N] - modified in-place
        float* raw_scales,    // [N, 3] - modified in-place
        float* raw_quats,     // [N, 4] - modified in-place
        const float* noise,   // [N, 3]
        float* means,         // [N, 3] - modified in-place
        int64_t N,
        float current_lr,
        cudaStream_t stream = nullptr);

} // namespace gsplat_lfs

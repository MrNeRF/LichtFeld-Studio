/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "Relocation.h"
#include "Common.h"
#include "Ops.h"

#include <cuda_runtime.h>

namespace gsplat_lfs {

    void relocation(
        float* opacities,
        float* scales,
        const int32_t* ratios,
        const float* binoms,
        int64_t N,
        int32_t n_max,
        float min_opacity,
        cudaStream_t stream) {
        gsplat_lfs::debug_validate_cuda_pointer(opacities, "opacities");
        gsplat_lfs::debug_validate_cuda_pointer(scales, "scales");
        gsplat_lfs::debug_validate_cuda_pointer(ratios, "ratios");
        gsplat_lfs::debug_validate_cuda_pointer(binoms, "binoms");

        if (N == 0) {
            return;
        }

        launch_relocation_kernel(
            opacities, scales, ratios, binoms,
            N, n_max, min_opacity, stream);
    }

    void add_noise(
        float* raw_opacities,
        float* raw_scales,
        float* raw_quats,
        const float* noise,
        float* means,
        int64_t N,
        float current_lr,
        cudaStream_t stream) {
        gsplat_lfs::debug_validate_cuda_pointer(raw_opacities, "raw_opacities");
        gsplat_lfs::debug_validate_cuda_pointer(raw_scales, "raw_scales");
        gsplat_lfs::debug_validate_cuda_pointer(raw_quats, "raw_quats");
        gsplat_lfs::debug_validate_cuda_pointer(noise, "noise");
        gsplat_lfs::debug_validate_cuda_pointer(means, "means");

        if (N == 0) {
            return;
        }

        launch_add_noise_kernel(
            raw_opacities, raw_scales, raw_quats,
            noise, means, N, current_lr, stream);
    }

} // namespace gsplat_lfs

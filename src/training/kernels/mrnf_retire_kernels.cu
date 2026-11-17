/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "core/cuda_error.hpp"
#include "kernel_stream.hpp"
#include "mrnf_kernels.hpp"
#include <cuda_runtime.h>

namespace lfs::training::mrnf_strategy {

    __global__ void blend_peak_update_kernel(float* __restrict__ step_peak, float* __restrict__ peak,
                                             float* __restrict__ renders, const float peak_decay, const size_t N) {
        const size_t idx = threadIdx.x + blockIdx.x * static_cast<size_t>(blockDim.x);
        if (idx >= N)
            return;
        const float step = step_peak[idx];
        step_peak[idx] = 0.0f;
        if (step > 0.0f)
            renders[idx] += 1.0f;
        peak[idx] = fmaxf(peak[idx] * peak_decay, step);
    }
    void launch_blend_peak_update(float* step_peak, float* peak, float* renders, const float peak_decay,
                                  const size_t N, void* stream) {
        if (N == 0)
            return;
        constexpr int threads = 256;
        const int blocks = static_cast<int>((N + threads - 1) / threads);
        cudaStream_t s = resolve_stream(stream);
        blend_peak_update_kernel<<<blocks, threads, 0, s>>>(step_peak, peak, renders, peak_decay, N);
        LFS_CUDA_LAUNCH_CHECK(s, "training.mrnf.blend_peak_update");
    }
    __global__ void mrnf_covered_factor_kernel(
        const float* __restrict__ peak,
        const float* __restrict__ renders,
        const float* __restrict__ raw_opacities,
        const float* __restrict__ log_scales,
        float grace,
        float keep_reach,
        float log_size_floor,
        size_t N,
        float* __restrict__ factor) {

        const size_t idx = threadIdx.x + blockIdx.x * static_cast<size_t>(blockDim.x);
        if (idx >= N)
            return;
        float r = 1.0f; // within the grace: unknown, treated as visible
        if (renders[idx] >= grace) {
            const float opacity = fmaxf(1.0f / (1.0f + expf(-raw_opacities[idx])), 1e-6f);
            const float p = peak[idx];
            // Explicit at and above the opacity: the fast-math division need not give exactly 1 there.
            r = p >= opacity ? 1.0f : fminf(fmaxf(p / opacity, 0.0f), 1.0f);
        }
        const float largest = fmaxf(fmaxf(log_scales[idx * 3 + 0], log_scales[idx * 3 + 1]), log_scales[idx * 3 + 2]);
        factor[idx] = r < keep_reach && largest >= log_size_floor ? r : 1.0f;
    }
    void launch_mrnf_covered_factor(
        const float* peak,
        const float* renders,
        const float* raw_opacities,
        const float* log_scales,
        float grace,
        float keep_reach,
        float log_size_floor,
        size_t N,
        float* factor,
        void* stream) {

        if (N == 0)
            return;
        constexpr int threads = 256;
        const int blocks = static_cast<int>((N + threads - 1) / threads);
        cudaStream_t s = resolve_stream(stream);
        mrnf_covered_factor_kernel<<<blocks, threads, 0, s>>>(peak, renders, raw_opacities, log_scales, grace,
                                                              keep_reach, log_size_floor, N, factor);
        LFS_CUDA_LAUNCH_CHECK(s, "training.mrnf.covered_factor");
    }

} // namespace lfs::training::mrnf_strategy

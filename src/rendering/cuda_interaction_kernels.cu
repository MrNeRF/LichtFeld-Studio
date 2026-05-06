/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include <cuda_runtime.h>

namespace lfs {

    constexpr float SH_C0 = 0.28209479177387814f;

    __global__ void adjustSaturationKernel(
        float* __restrict__ sh0,
        const float* __restrict__ screen_positions,
        const float cursor_x,
        const float cursor_y,
        const float cursor_radius_sq,
        const float saturation_delta,
        const int num_gaussians) {

        const int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx >= num_gaussians)
            return;

        const float dx = screen_positions[idx * 2 + 0] - cursor_x;
        const float dy = screen_positions[idx * 2 + 1] - cursor_y;
        if (dx * dx + dy * dy > cursor_radius_sq)
            return;

        const float r = SH_C0 * sh0[idx * 3 + 0] + 0.5f;
        const float g = SH_C0 * sh0[idx * 3 + 1] + 0.5f;
        const float b = SH_C0 * sh0[idx * 3 + 2] + 0.5f;

        const float lum = 0.2126f * r + 0.7152f * g + 0.0722f * b;

        const float factor = 1.0f + saturation_delta;
        const float new_r = fmaxf(0.0f, fminf(1.0f, lum + factor * (r - lum)));
        const float new_g = fmaxf(0.0f, fminf(1.0f, lum + factor * (g - lum)));
        const float new_b = fmaxf(0.0f, fminf(1.0f, lum + factor * (b - lum)));

        sh0[idx * 3 + 0] = (new_r - 0.5f) / SH_C0;
        sh0[idx * 3 + 1] = (new_g - 0.5f) / SH_C0;
        sh0[idx * 3 + 2] = (new_b - 0.5f) / SH_C0;
    }

    void launchAdjustSaturation(
        float* sh0,
        const float* screen_positions,
        const float cursor_x,
        const float cursor_y,
        const float cursor_radius,
        const float saturation_delta,
        const int num_gaussians,
        cudaStream_t stream) {

        if (num_gaussians <= 0)
            return;

        constexpr int threads = 256;
        const int blocks = (num_gaussians + threads - 1) / threads;
        adjustSaturationKernel<<<blocks, threads, 0, stream>>>(
            sh0, screen_positions, cursor_x, cursor_y,
            cursor_radius * cursor_radius, saturation_delta, num_gaussians);
    }

} // namespace lfs

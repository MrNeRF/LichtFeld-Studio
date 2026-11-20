/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "core/assert.hpp"
#include "core/cuda_error.hpp"
#include "kernel_stream.hpp"
#include "mrnf_kernels.hpp"
#include <cuda_runtime.h>

namespace lfs::training::mrnf_strategy {

    namespace {

        // Exponent applied to each view's per-splat error sum before it enters the window, so a large
        // footprint cannot outweigh a high error per pixel.
        constexpr float REFINE_ERROR_COMPRESS_VIEW_POWER = 0.2f;

    } // namespace

    __global__ void fold_need_kernel(float* __restrict__ error, const float error_scale, const float error_max,
                                     const float power, const size_t N) {
        const size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
        if (i >= N)
            return;
        const float e = error[i];
        error[i] = e > 0.0f ? powf(fminf(e / error_scale, error_max), power) : 0.0f;
    }

    void launch_fold_need(float* error_inout, const float error_scale, const float error_max, const float power,
                          const size_t N, void* stream) {
        if (N == 0)
            return;
        LFS_ASSERT_MSG(error_scale > 0.0f, "need error scale must be positive");
        constexpr int threads = 256;
        const int blocks = static_cast<int>((N + threads - 1) / threads);
        cudaStream_t s = resolve_stream(stream);
        fold_need_kernel<<<blocks, threads, 0, s>>>(error_inout, error_scale, error_max, power, N);
        LFS_CUDA_LAUNCH_CHECK(s, "training.mrnf.fold_need");
    }

    __global__ void fold_densification_error_view_sum_and_zero_kernel(
        float* __restrict__ refine_weight,
        float* __restrict__ densification_info,
        size_t N,
        float* __restrict__ coverage_views) {

        const size_t idx = threadIdx.x + blockIdx.x * static_cast<size_t>(blockDim.x);
        if (idx >= N)
            return;

        const float coverage = densification_info[idx];
        if (coverage_views != nullptr && coverage > 0.0f)
            coverage_views[idx] += 1.0f;
        const float error = densification_info[N + idx];
        const float s = error > 0.0f ? powf(error, REFINE_ERROR_COMPRESS_VIEW_POWER) : 0.0f;
        refine_weight[idx] += s;
        if (s > 0.0f)
            densification_info[3 * N + idx] += 1.0f;
        densification_info[2 * N + idx] += coverage;
        densification_info[idx] = 0.0f;
        densification_info[N + idx] = 0.0f;
    }

    void launch_fold_densification_error_view_sum_and_zero(
        float* refine_weight,
        float* densification_info,
        size_t N,
        float* coverage_views,
        void* stream) {
        if (N == 0)
            return;
        constexpr int threads = 256;
        const int blocks = static_cast<int>((N + threads - 1) / threads);
        cudaStream_t s = resolve_stream(stream);
        fold_densification_error_view_sum_and_zero_kernel<<<blocks, threads, 0, s>>>(
            refine_weight, densification_info, N, coverage_views);
        LFS_CUDA_LAUNCH_CHECK(s, "training.mrnf.fold_densification_error_view_sum_and_zero");
    }

    __global__ void refine_error_view_mean_kernel(
        float* __restrict__ refine_weight,
        const float* __restrict__ densification_info,
        size_t N) {

        const size_t idx = threadIdx.x + blockIdx.x * static_cast<size_t>(blockDim.x);
        if (idx >= N)
            return;
        refine_weight[idx] /= fmaxf(densification_info[3 * N + idx], 1.0f);
    }

    void launch_refine_error_view_mean(
        float* refine_weight,
        const float* densification_info,
        size_t N,
        void* stream) {
        if (N == 0)
            return;
        constexpr int threads = 256;
        const int blocks = static_cast<int>((N + threads - 1) / threads);
        cudaStream_t s = resolve_stream(stream);
        refine_error_view_mean_kernel<<<blocks, threads, 0, s>>>(refine_weight, densification_info, N);
        LFS_CUDA_LAUNCH_CHECK(s, "training.mrnf.error_view_mean");
    }

} // namespace lfs::training::mrnf_strategy

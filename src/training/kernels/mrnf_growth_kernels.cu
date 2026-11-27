/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "core/assert.hpp"
#include "core/cuda_error.hpp"
#include "kernel_stream.hpp"
#include "mrnf_kernels.hpp"
#include <cub/device/device_radix_sort.cuh>
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

    __global__ void __launch_bounds__(256, 6) fold_step_stats_sparse_kernel(
        const uint32_t* __restrict__ visible_indices,
        const size_t n_visible,
        float* __restrict__ refine_weight,
        float* __restrict__ densification_info,
        float* __restrict__ coverage_views,
        const bool* __restrict__ frozen_mask,
        const size_t frozen_n,
        float* __restrict__ step_peak,
        float* __restrict__ peak,
        float* __restrict__ renders,
        const float inv_peak_scale,
        const size_t N) {
        const size_t k = threadIdx.x + blockIdx.x * static_cast<size_t>(blockDim.x);
        if (k >= n_visible)
            return;
        const uint32_t prim = visible_indices[k];
        if (prim >= N)
            return;

        const bool frozen = (frozen_mask != nullptr && prim < frozen_n && frozen_mask[prim]);
        if (densification_info != nullptr && refine_weight != nullptr) {
            const float coverage = densification_info[prim];
            const float error = densification_info[N + prim];
            if (coverage != 0.0f)
                densification_info[prim] = 0.0f;
            if (error != 0.0f)
                densification_info[N + prim] = 0.0f;
            if (coverage_views != nullptr && coverage > 0.0f)
                coverage_views[prim] += 1.0f;
            if (error > 0.0f) {
                const float s = powf(error, REFINE_ERROR_COMPRESS_VIEW_POWER);
                if (!frozen)
                    refine_weight[prim] += s;
                if (s > 0.0f)
                    densification_info[3 * N + prim] += 1.0f;
            }
            if (frozen) {
                refine_weight[prim] = 0.0f;
                densification_info[2 * N + prim] = 0.0f;
            } else if (coverage != 0.0f) {
                densification_info[2 * N + prim] += coverage;
            }
        }
        if (step_peak != nullptr && peak != nullptr && renders != nullptr) {
            const float step = step_peak[prim];
            if (step != 0.0f) {
                step_peak[prim] = 0.0f;
                if (step > 0.0f) {
                    renders[prim] += 1.0f;
                    const float scaled_step = step * inv_peak_scale;
                    if (scaled_step > peak[prim])
                        peak[prim] = scaled_step;
                }
            }
        }
    }

    void launch_fold_step_stats_sparse(
        const uint32_t* visible_indices,
        const size_t n_visible,
        float* refine_weight,
        float* densification_info,
        float* coverage_views,
        const bool* frozen_mask,
        const size_t frozen_n,
        float* step_peak,
        float* peak,
        float* renders,
        const float inv_peak_scale,
        const size_t N,
        void* stream) {
        if (N == 0 || n_visible == 0 || visible_indices == nullptr)
            return;
        constexpr int threads = 256;
        const int blocks = static_cast<int>((n_visible + threads - 1) / threads);
        cudaStream_t s = resolve_stream(stream);
        fold_step_stats_sparse_kernel<<<blocks, threads, 0, s>>>(
            visible_indices, n_visible, refine_weight, densification_info, coverage_views,
            frozen_mask, frozen_n, step_peak, peak, renders, inv_peak_scale, N);
        LFS_CUDA_LAUNCH_CHECK(s, "training.mrnf.fold_step_stats_sparse");
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

    size_t sort_int64_temp_bytes(const size_t count) {
        if (count <= 1)
            return 0;
        size_t temp_bytes = 0;
        LFS_CUDA_CHECK_MSG(
            cub::DeviceRadixSort::SortKeys(
                nullptr, temp_bytes, static_cast<const int64_t*>(nullptr),
                static_cast<int64_t*>(nullptr), count),
            "training.mrnf.sort_int64 temp query");
        return temp_bytes;
    }

    void launch_sort_int64(
        const int64_t* in,
        int64_t* out,
        const size_t count,
        void* temp_storage,
        size_t temp_bytes,
        void* stream) {
        if (count == 0)
            return;
        cudaStream_t s = resolve_stream(stream);
        if (count == 1) {
            if (in != out) {
                LFS_CUDA_CHECK_MSG(
                    cudaMemcpyAsync(out, in, sizeof(int64_t), cudaMemcpyDeviceToDevice, s),
                    "training.mrnf.sort_int64 single copy");
            }
            return;
        }
        LFS_CUDA_CHECK_MSG(
            cub::DeviceRadixSort::SortKeys(temp_storage, temp_bytes, in, out, count, 0, 64, s),
            "training.mrnf.sort_int64");
    }

} // namespace lfs::training::mrnf_strategy

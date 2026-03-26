/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "core/tensor/internal/tensor_generic_ops.cuh"
#include "lfs_kernels.hpp"
#include <algorithm>
#include <cassert>
#include <cub/cub.cuh>
#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <thrust/copy.h>
#include <thrust/count.h>
#include <thrust/device_ptr.h>
#include <thrust/execution_policy.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/sequence.h>
#include <thrust/sort.h>

namespace lfs::training::lfs_strategy {

    namespace {

        __device__ __forceinline__ float d_sigmoid(float x) {
            return 1.0f / (1.0f + expf(-x));
        }

        __device__ __forceinline__ float d_logit(float p) {
            return logf(p / (1.0f - p));
        }

        struct positive_weight {
            __host__ __device__ bool operator()(const float w) const {
                return w > 0.0f;
            }
        };

    } // namespace

    __global__ void lfs_noise_injection_kernel(
        float* __restrict__ means,
        const float* __restrict__ raw_opacities,
        const float* __restrict__ vis_count,
        float lr_mean,
        float noise_weight,
        float median_scale,
        size_t N,
        uint64_t seed) {

        const size_t idx = threadIdx.x + blockIdx.x * static_cast<size_t>(blockDim.x);
        if (idx >= N)
            return;

        if (vis_count[idx] <= 0.0f)
            return;

        const float inv_opac = 1.0f - d_sigmoid(raw_opacities[idx]);
        float weight = powf(fmaxf(inv_opac, 0.0f), 150.0f);
        weight *= lr_mean * noise_weight;

        if (weight < 1e-12f)
            return;

        curandState rng;
        curand_init(seed, idx, 0, &rng);

        for (int d = 0; d < 3; ++d) {
            const float noise = curand_normal(&rng) * weight;
            const float clamped_noise = fminf(fmaxf(noise, -median_scale), median_scale);
            means[idx * 3 + d] += clamped_noise;
        }
    }

    void launch_lfs_noise_injection(
        float* means,
        const float* raw_opacities,
        const float* vis_count,
        float lr_mean,
        float noise_weight,
        float median_scale,
        size_t N,
        uint64_t seed,
        void* stream) {

        if (N == 0)
            return;

        constexpr int threads = 256;
        const int blocks = static_cast<int>((N + threads - 1) / threads);
        cudaStream_t s = stream ? static_cast<cudaStream_t>(stream) : nullptr;

        lfs_noise_injection_kernel<<<blocks, threads, 0, s>>>(
            means, raw_opacities, vis_count,
            lr_mean, noise_weight, median_scale, N, seed);
    }

    __global__ void lfs_decay_kernel(
        float* __restrict__ raw_opacities,
        float* __restrict__ log_scales,
        float opacity_decay,
        float scale_decay,
        float train_t,
        size_t N) {

        const size_t idx = threadIdx.x + blockIdx.x * static_cast<size_t>(blockDim.x);
        if (idx >= N)
            return;

        const float t_shrink = 1.0f - train_t;

        float opac = d_sigmoid(raw_opacities[idx]) - opacity_decay * t_shrink;
        opac = fminf(fmaxf(opac, 1e-12f), 1.0f - 1e-12f);
        raw_opacities[idx] = d_logit(opac);

        const float decay_factor = 1.0f - scale_decay * t_shrink;
        for (int d = 0; d < 3; ++d) {
            const float scale = expf(log_scales[idx * 3 + d]) * decay_factor;
            log_scales[idx * 3 + d] = logf(fmaxf(scale, 1e-12f));
        }
    }

    void launch_lfs_decay(
        float* raw_opacities,
        float* log_scales,
        float opacity_decay,
        float scale_decay,
        float train_t,
        size_t N,
        void* stream) {

        if (N == 0)
            return;

        constexpr int threads = 256;
        const int blocks = static_cast<int>((N + threads - 1) / threads);
        cudaStream_t s = stream ? static_cast<cudaStream_t>(stream) : nullptr;

        lfs_decay_kernel<<<blocks, threads, 0, s>>>(
            raw_opacities, log_scales, opacity_decay, scale_decay, train_t, N);
    }

    __global__ void elementwise_add_inplace_kernel(
        float* __restrict__ a,
        const float* __restrict__ b,
        size_t N) {
        const size_t idx = threadIdx.x + blockIdx.x * static_cast<size_t>(blockDim.x);
        if (idx < N)
            a[idx] += b[idx];
    }

    void launch_elementwise_add_inplace(
        float* a,
        const float* b,
        size_t N,
        void* stream) {
        if (N == 0)
            return;
        constexpr int threads = 256;
        const int blocks = static_cast<int>((N + threads - 1) / threads);
        cudaStream_t s = stream ? static_cast<cudaStream_t>(stream) : nullptr;
        elementwise_add_inplace_kernel<<<blocks, threads, 0, s>>>(a, b, N);
    }

    __global__ void extract_axis_kernel(
        const float* __restrict__ means,
        float* __restrict__ output,
        int axis,
        size_t N) {
        const size_t idx = threadIdx.x + blockIdx.x * static_cast<size_t>(blockDim.x);
        if (idx < N)
            output[idx] = means[idx * 3 + axis];
    }

    void launch_percentile_bounds(
        const float* means,
        size_t N,
        float percentile,
        LFSBounds* bounds,
        void* stream) {

        assert(N > 0);
        assert(bounds != nullptr);

        cudaStream_t s = stream ? static_cast<cudaStream_t>(stream) : nullptr;

        const float low_pct = (1.0f - percentile) / 2.0f;
        const float high_pct = 1.0f - low_pct;
        const size_t low_idx = static_cast<size_t>(low_pct * static_cast<float>(N - 1));
        const size_t high_idx = static_cast<size_t>(high_pct * static_cast<float>(N - 1));

        const int n_int = static_cast<int>(N);

        float* d_input = nullptr;
        float* d_sorted = nullptr;
        cudaMallocAsync(&d_input, N * sizeof(float), s);
        cudaMallocAsync(&d_sorted, N * sizeof(float), s);

        size_t temp_bytes = 0;
        cub::DeviceRadixSort::SortKeys(nullptr, temp_bytes, d_input, d_sorted, n_int, 0, 32, s);
        char* d_temp = nullptr;
        cudaMallocAsync(&d_temp, temp_bytes, s);

        constexpr int threads = 256;
        const int blocks = static_cast<int>((N + threads - 1) / threads);

        float h_low, h_high;
        float extents[3], centers[3];

        for (int axis = 0; axis < 3; ++axis) {
            extract_axis_kernel<<<blocks, threads, 0, s>>>(means, d_input, axis, N);
            cub::DeviceRadixSort::SortKeys(d_temp, temp_bytes, d_input, d_sorted, n_int, 0, 32, s);
            cudaMemcpyAsync(&h_low, d_sorted + low_idx, sizeof(float), cudaMemcpyDeviceToHost, s);
            cudaMemcpyAsync(&h_high, d_sorted + high_idx, sizeof(float), cudaMemcpyDeviceToHost, s);
            cudaStreamSynchronize(s);

            centers[axis] = (h_low + h_high) * 0.5f;
            extents[axis] = (h_high - h_low) * 0.5f;
        }

        cudaFreeAsync(d_input, s);
        cudaFreeAsync(d_sorted, s);
        cudaFreeAsync(d_temp, s);

        for (int i = 0; i < 3; ++i) {
            bounds->center[i] = centers[i];
            bounds->extent[i] = extents[i];
        }

        float sorted_ext[3] = {extents[0], extents[1], extents[2]};
        std::sort(sorted_ext, sorted_ext + 3);
        bounds->median_size = sorted_ext[1] * 2.0f;
        bounds->max_extent = sorted_ext[2];
    }

    __global__ void gumbel_key_kernel(
        const float* __restrict__ weights,
        float* __restrict__ keys,
        size_t N,
        uint64_t seed) {

        const size_t idx = threadIdx.x + blockIdx.x * static_cast<size_t>(blockDim.x);
        if (idx >= N)
            return;

        const float w = weights[idx];
        if (w <= 0.0f) {
            keys[idx] = -1e30f;
            return;
        }

        curandState rng;
        curand_init(seed, idx, 0, &rng);
        float u = curand_uniform(&rng);
        u = fmaxf(u, 1e-10f);
        u = fminf(u, 1.0f - 1e-7f);

        keys[idx] = -logf(-logf(u)) + logf(w);
    }

    __global__ void gumbel_key_for_indices_kernel(
        const float* __restrict__ weights,
        const int64_t* __restrict__ indices,
        float* __restrict__ keys,
        size_t N,
        uint64_t seed) {

        const size_t idx = threadIdx.x + blockIdx.x * static_cast<size_t>(blockDim.x);
        if (idx >= N)
            return;

        const int64_t src_idx = indices[idx];
        const float w = weights[src_idx];

        curandState rng;
        curand_init(seed, idx, 0, &rng);
        float u = curand_uniform(&rng);
        u = fmaxf(u, 1e-10f);
        u = fminf(u, 1.0f - 1e-7f);

        keys[idx] = -logf(-logf(u)) + logf(w);
    }

    void launch_gumbel_topk(
        const float* weights,
        size_t N,
        size_t K,
        uint64_t seed,
        int64_t* output_indices,
        void* stream) {

        assert(K <= N);
        if (K == 0)
            return;

        cudaStream_t s = stream ? static_cast<cudaStream_t>(stream) : nullptr;

        if (K == N) {
            auto out_ptr = thrust::device_pointer_cast(output_indices);
            lfs::core::tensor_ops::run_with_thrust_policy(s, [&](auto policy) {
                thrust::sequence(policy, out_ptr, out_ptr + N);
            });
            return;
        }

        auto weights_ptr = thrust::device_pointer_cast(weights);
        size_t active_count = 0;
        lfs::core::tensor_ops::run_with_thrust_policy(s, [&](auto policy) {
            active_count = static_cast<size_t>(
                thrust::count_if(policy, weights_ptr, weights_ptr + N, positive_weight{}));
        });

        const bool compact_active = active_count >= K && active_count < N;
        const size_t sort_count = compact_active ? active_count : N;

        float* d_keys = nullptr;
        int64_t* d_indices = nullptr;
        float* d_keys_sorted = nullptr;
        int64_t* d_indices_sorted = nullptr;
        cudaMallocAsync(&d_keys, sort_count * sizeof(float), s);
        cudaMallocAsync(&d_indices, sort_count * sizeof(int64_t), s);
        cudaMallocAsync(&d_keys_sorted, sort_count * sizeof(float), s);
        cudaMallocAsync(&d_indices_sorted, sort_count * sizeof(int64_t), s);

        constexpr int threads = 256;
        const int blocks = static_cast<int>((sort_count + threads - 1) / threads);

        if (compact_active) {
            auto indices_ptr = thrust::device_pointer_cast(d_indices);
            auto counting_begin = thrust::make_counting_iterator<int64_t>(0);
            lfs::core::tensor_ops::run_with_thrust_policy(s, [&](auto policy) {
                thrust::copy_if(
                    policy,
                    counting_begin,
                    counting_begin + static_cast<std::ptrdiff_t>(N),
                    weights_ptr,
                    indices_ptr,
                    positive_weight{});
            });
            gumbel_key_for_indices_kernel<<<blocks, threads, 0, s>>>(
                weights, d_indices, d_keys, sort_count, seed);
        } else {
            gumbel_key_kernel<<<blocks, threads, 0, s>>>(weights, d_keys, sort_count, seed);
            auto indices_ptr = thrust::device_pointer_cast(d_indices);
            lfs::core::tensor_ops::run_with_thrust_policy(s, [&](auto policy) {
                thrust::sequence(policy, indices_ptr, indices_ptr + static_cast<std::ptrdiff_t>(sort_count));
            });
        }

        const int sort_count_int = static_cast<int>(sort_count);
        size_t temp_bytes = 0;
        cub::DeviceRadixSort::SortPairsDescending(
            nullptr, temp_bytes,
            d_keys, d_keys_sorted,
            d_indices, d_indices_sorted,
            sort_count_int, 0, 32, s);
        char* d_temp = nullptr;
        cudaMallocAsync(&d_temp, temp_bytes, s);
        cub::DeviceRadixSort::SortPairsDescending(
            d_temp, temp_bytes,
            d_keys, d_keys_sorted,
            d_indices, d_indices_sorted,
            sort_count_int, 0, 32, s);

        cudaMemcpyAsync(output_indices, d_indices_sorted, K * sizeof(int64_t), cudaMemcpyDeviceToDevice, s);

        cudaFreeAsync(d_temp, s);
        cudaFreeAsync(d_keys, s);
        cudaFreeAsync(d_indices, s);
        cudaFreeAsync(d_keys_sorted, s);
        cudaFreeAsync(d_indices_sorted, s);
    }

} // namespace lfs::training::lfs_strategy

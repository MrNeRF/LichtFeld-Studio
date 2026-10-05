/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "core/cuda_error.hpp"
#include "core/tensor/internal/tensor_generic_ops.cuh"
#include "lfs/cuda_scratch.hpp"
#include "lfs/training/refine_scratch.hpp"
#include "mrnf_kernels.hpp"
#include <algorithm>
#include <cmath>
#include <cub/cub.cuh>
#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <limits>
#include <thrust/copy.h>
#include <thrust/count.h>
#include <thrust/device_ptr.h>
#include <thrust/execution_policy.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/sequence.h>
#include <thrust/sort.h>

#include "kernel_stream.hpp"

namespace lfs::training::mrnf_strategy {

    namespace {

        __device__ __forceinline__ float d_sigmoid(float x) {
            return 1.0f / (1.0f + expf(-x));
        }

        __device__ __forceinline__ float d_logit(float p) {
            if (isnan(p))
                return p;
            // The upper bound must be representable below 1 in float32.
            p = fminf(fmaxf(p, 1e-12f), nextafterf(1.0f, 0.0f));
            return logf(p / (1.0f - p));
        }

        struct positive_weight {
            __host__ __device__ bool operator()(const float w) const {
                return w > 0.0f;
            }
        };

    } // namespace

    __global__ void prune_bounds_or_kernel(
        const float* __restrict__ means,
        const float* __restrict__ max_log_scales,
        bool* __restrict__ prune_mask,
        size_t N,
        float center_x,
        float center_y,
        float center_z,
        float max_allowed,
        float log_max_allowed) {
        const size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
        if (i >= N)
            return;

        const float dx = fabsf(means[3 * i] - center_x);
        const float dy = fabsf(means[3 * i + 1] - center_y);
        const float dz = fabsf(means[3 * i + 2] - center_z);
        const bool distance_exceeds = !isnan(dx) && !isnan(dy) && !isnan(dz) &&
                                      fmaxf(dx, fmaxf(dy, dz)) > max_allowed;
        prune_mask[i] = prune_mask[i] || max_log_scales[i] > log_max_allowed || distance_exceeds;
    }

    void launch_prune_bounds_or(
        const float* means,
        const float* max_log_scales,
        bool* prune_mask,
        size_t N,
        const float* center,
        float max_allowed,
        float log_max_allowed,
        void* stream) {
        if (N == 0)
            return;
        constexpr int threads = 256;
        const int blocks = static_cast<int>((N + threads - 1) / threads);
        cudaStream_t s = resolve_stream(stream);
        prune_bounds_or_kernel<<<blocks, threads, 0, s>>>(
            means, max_log_scales, prune_mask, N,
            center[0], center[1], center[2], max_allowed, log_max_allowed);
        LFS_CUDA_LAUNCH_CHECK(s, "training.mrnf.prune_bounds_or");
    }

    __global__ void replace_parent_weights_kernel(
        const float* __restrict__ opacities,
        const float* __restrict__ visibility,
        const bool* __restrict__ active_mask,
        const bool* __restrict__ trainable_mask,
        const float* __restrict__ edge_guidance,
        float* __restrict__ output,
        size_t N) {
        const size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
        if (i >= N)
            return;
        float weight = opacities[i] * static_cast<float>(visibility[i] > 0.0f);
        if (active_mask)
            weight = weight * static_cast<float>(active_mask[i]);
        if (trainable_mask)
            weight = weight * static_cast<float>(trainable_mask[i]);
        if (edge_guidance)
            weight = weight * edge_guidance[i];
        output[i] = weight;
    }

    void launch_replace_parent_weights(
        const float* opacities,
        const float* visibility,
        const bool* active_mask,
        const bool* trainable_mask,
        const float* edge_guidance,
        float* output,
        size_t N,
        void* stream) {
        if (N == 0)
            return;
        constexpr int threads = 256;
        const int blocks = static_cast<int>((N + threads - 1) / threads);
        cudaStream_t s = resolve_stream(stream);
        replace_parent_weights_kernel<<<blocks, threads, 0, s>>>(
            opacities, visibility, active_mask, trainable_mask, edge_guidance, output, N);
        LFS_CUDA_LAUNCH_CHECK(s, "training.mrnf.replace_parent_weights");
    }

    __global__ void mrnf_noise_injection_kernel(
        float* __restrict__ means,
        const float* __restrict__ raw_opacities,
        const float* __restrict__ vis_count,
        const bool* __restrict__ frozen_mask,
        size_t frozen_mask_size,
        float lr_mean,
        float noise_weight,
        float median_scale,
        size_t N,
        uint64_t seed) {

        const size_t idx = threadIdx.x + blockIdx.x * static_cast<size_t>(blockDim.x);
        if (idx >= N)
            return;
        if (frozen_mask != nullptr && idx < frozen_mask_size && frozen_mask[idx])
            return;

        if (vis_count[idx] <= 0.0f)
            return;

        const float inv_opac = 1.0f - d_sigmoid(raw_opacities[idx]);
        float weight = powf(fmaxf(inv_opac, 0.0f), 150.0f);
        weight *= lr_mean * noise_weight;

        if (weight < 1e-12f)
            return;

        curandStatePhilox4_32_10_t rng;
        curand_init(seed, idx, 0, &rng);
        const float4 n = curand_normal4(&rng);
        const float noise_xyz[3] = {n.x, n.y, n.z};

        for (int d = 0; d < 3; ++d) {
            const float noise = noise_xyz[d] * weight;
            const float clamped_noise = fminf(fmaxf(noise, -median_scale), median_scale);
            means[idx * 3 + d] += clamped_noise;
        }
    }

    void launch_mrnf_noise_injection(
        float* means,
        const float* raw_opacities,
        const float* vis_count,
        const bool* frozen_mask,
        size_t frozen_mask_size,
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
        cudaStream_t s = resolve_stream(stream);

        mrnf_noise_injection_kernel<<<blocks, threads, 0, s>>>(
            means, raw_opacities, vis_count,
            frozen_mask, frozen_mask_size,
            lr_mean, noise_weight, median_scale, N, seed);
        LFS_CUDA_LAUNCH_CHECK(s, "training.mrnf.noise_injection");
    }

    __global__ void mrnf_decay_kernel(
        float* __restrict__ raw_opacities,
        float* __restrict__ log_scales,
        const bool* __restrict__ frozen_mask,
        size_t frozen_mask_size,
        float opacity_decay,
        float scale_decay,
        float train_t,
        size_t N) {

        const size_t idx = threadIdx.x + blockIdx.x * static_cast<size_t>(blockDim.x);
        if (idx >= N)
            return;
        if (frozen_mask != nullptr && idx < frozen_mask_size && frozen_mask[idx])
            return;

        const float t_shrink = 1.0f - train_t;

        const float opacity_delta = opacity_decay * t_shrink;
        // A sigmoid/logit round trip loses finite saturated logits even when
        // decay is disabled. Still repair infinities from older checkpoints.
        if (opacity_delta != 0.0f || isinf(raw_opacities[idx])) {
            raw_opacities[idx] = d_logit(d_sigmoid(raw_opacities[idx]) - opacity_delta);
        }

        const float decay_factor = 1.0f - scale_decay * t_shrink;
        for (int d = 0; d < 3; ++d) {
            const float scale = expf(log_scales[idx * 3 + d]) * decay_factor;
            log_scales[idx * 3 + d] = logf(fmaxf(scale, 1e-12f));
        }
    }

    void launch_mrnf_decay(
        float* raw_opacities,
        float* log_scales,
        const bool* frozen_mask,
        size_t frozen_mask_size,
        float opacity_decay,
        float scale_decay,
        float train_t,
        size_t N,
        void* stream) {

        if (N == 0)
            return;

        constexpr int threads = 256;
        const int blocks = static_cast<int>((N + threads - 1) / threads);
        cudaStream_t s = resolve_stream(stream);

        mrnf_decay_kernel<<<blocks, threads, 0, s>>>(
            raw_opacities, log_scales, frozen_mask, frozen_mask_size,
            opacity_decay, scale_decay, train_t, N);
        LFS_CUDA_LAUNCH_CHECK(s, "training.mrnf.decay");
    }

    __global__ void fold_densification_error_and_zero_kernel(
        float* __restrict__ refine_weight_max,
        float* __restrict__ densification_info,
        size_t N) {

        const size_t idx = threadIdx.x + blockIdx.x * static_cast<size_t>(blockDim.x);
        if (idx >= N)
            return;

        refine_weight_max[idx] = fmaxf(refine_weight_max[idx], densification_info[N + idx]);
        densification_info[N + idx] = 0.0f;
    }

    void launch_fold_densification_error_and_zero(
        float* refine_weight_max,
        float* densification_info,
        size_t N,
        void* stream) {
        if (N == 0)
            return;
        constexpr int threads = 256;
        const int blocks = static_cast<int>((N + threads - 1) / threads);
        cudaStream_t s = resolve_stream(stream);
        fold_densification_error_and_zero_kernel<<<blocks, threads, 0, s>>>(
            refine_weight_max, densification_info, N);
        LFS_CUDA_LAUNCH_CHECK(s, "training.mrnf.fold_densification_error_and_zero");
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
        MRNFBounds* bounds,
        void* stream) {

        LFS_ASSERT(N > 0);
        LFS_ASSERT(bounds != nullptr);
        LFS_ASSERT_MSG(std::isfinite(percentile) && percentile >= 0.0f && percentile <= 1.0f,
                       "MRNF bounds percentile must be finite and within [0, 1]");
        LFS_ASSERT_MSG(N <= static_cast<size_t>(std::numeric_limits<int>::max()),
                       "MRNF percentile input exceeds CUB's int item-count limit");

        cudaStream_t s = resolve_stream(stream);

        const float low_pct = (1.0f - percentile) / 2.0f;
        const float high_pct = 1.0f - low_pct;
        const size_t low_idx = static_cast<size_t>(low_pct * static_cast<float>(N - 1));
        const size_t high_idx = static_cast<size_t>(high_pct * static_cast<float>(N - 1));

        const int n_int = static_cast<int>(N);

        const size_t values_bytes = cuda_scratch::checked_bytes(
            N, sizeof(float), "MRNF percentile values");
        cuda_scratch::DeviceBuffer input_buffer(values_bytes, s, "mrnf.percentile.input");
        cuda_scratch::DeviceBuffer sorted_buffer(values_bytes, s, "mrnf.percentile.sorted");
        auto* d_input = input_buffer.as<float>();
        auto* d_sorted = sorted_buffer.as<float>();

        cuda_scratch::CubWorkspace cub_workspace(
            "cub::DeviceRadixSort::SortKeys", s,
            [&](void* workspace, size_t& workspace_bytes) {
                return cub::DeviceRadixSort::SortKeys(
                    workspace, workspace_bytes, d_input, d_sorted, n_int, 0, 32, s);
            });

        constexpr int threads = 256;
        const int blocks = static_cast<int>((N + threads - 1) / threads);

        float h_low, h_high;
        float extents[3], centers[3];

        for (int axis = 0; axis < 3; ++axis) {
            extract_axis_kernel<<<blocks, threads, 0, s>>>(means, d_input, axis, N);
            LFS_CUDA_CHECK_MSG(cudaGetLastError(), "MRNF percentile axis extraction");
            cub_workspace.run([&](void* workspace, size_t& workspace_bytes) {
                return cub::DeviceRadixSort::SortKeys(
                    workspace, workspace_bytes, d_input, d_sorted, n_int, 0, 32, s);
            });
            LFS_CUDA_CHECK_MSG(
                cudaMemcpyAsync(&h_low, d_sorted + low_idx, sizeof(float), cudaMemcpyDeviceToHost, s),
                "MRNF percentile low readback");
            LFS_CUDA_CHECK_MSG(
                cudaMemcpyAsync(&h_high, d_sorted + high_idx, sizeof(float), cudaMemcpyDeviceToHost, s),
                "MRNF percentile high readback");
            LFS_CUDA_CHECK_MSG(cudaStreamSynchronize(s), "MRNF percentile stream sync");

            centers[axis] = (h_low + h_high) * 0.5f;
            extents[axis] = (h_high - h_low) * 0.5f;
        }

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

        curandStatePhilox4_32_10_t rng;
        curand_init(seed, idx, 0, &rng);
        float u = curand_uniform(&rng);
        u = fmaxf(u, 1e-10f);
        u = fminf(u, 1.0f - 1e-7f);

        keys[idx] = -logf(-logf(u)) + logf(w);
    }

    __global__ void gumbel_key_for_indices_kernel(
        const float* __restrict__ weights,
        const uint32_t* __restrict__ indices,
        float* __restrict__ keys,
        size_t N,
        uint64_t seed) {

        const size_t idx = threadIdx.x + blockIdx.x * static_cast<size_t>(blockDim.x);
        if (idx >= N)
            return;

        const uint32_t src_idx = indices[idx];
        const float w = weights[src_idx];

        curandStatePhilox4_32_10_t rng;
        curand_init(seed, idx, 0, &rng);
        float u = curand_uniform(&rng);
        u = fmaxf(u, 1e-10f);
        u = fminf(u, 1.0f - 1e-7f);

        keys[idx] = -logf(-logf(u)) + logf(w);
    }

    __global__ void widen_gumbel_indices_kernel(
        const uint32_t* __restrict__ input,
        int64_t* __restrict__ output,
        size_t N) {
        const size_t idx = threadIdx.x + blockIdx.x * static_cast<size_t>(blockDim.x);
        if (idx < N) {
            output[idx] = static_cast<int64_t>(input[idx]);
        }
    }

    void launch_gumbel_topk(
        const float* weights,
        size_t N,
        size_t K,
        uint64_t seed,
        int64_t* output_indices,
        void* stream,
        bool compact_sparse,
        GumbelTopKScratch* scratch,
        size_t known_nnz) {

        LFS_ASSERT(K <= N);
        if (K == 0)
            return;

        LFS_ASSERT_MSG(N <= static_cast<size_t>(std::numeric_limits<uint32_t>::max()),
                       "MRNF Gumbel input exceeds the uint32 payload contract");
        LFS_ASSERT_MSG(N <= static_cast<size_t>(std::numeric_limits<int>::max()),
                       "MRNF Gumbel input exceeds CUB's int item-count limit");

        cudaStream_t s = resolve_stream(stream);

        if (K == N) {
            auto out_ptr = thrust::device_pointer_cast(output_indices);
            // Caller may consume output_indices via a Tensor whose home stream
            // is not `s` (scratch allocated earlier). Keep the host wait.
            thrust::sequence(thrust::cuda::par.on(s), out_ptr, out_ptr + N);
            return;
        }

        auto weights_ptr = thrust::device_pointer_cast(weights);
        size_t active_count = N;
        if (compact_sparse) {
            if (known_nnz > 0) {
                LFS_ASSERT_MSG(known_nnz <= N,
                               lfs::core::detail::format_cuda_safe(
                                   "Gumbel known_nnz must be <= N (known_nnz={}, N={})",
                                   known_nnz, N));
                active_count = known_nnz;
            } else {
                active_count = static_cast<size_t>(
                    thrust::count_if(thrust::cuda::par.on(s), weights_ptr, weights_ptr + N,
                                     positive_weight{}));
            }
        }

        const bool compact_active = compact_sparse && active_count >= K && active_count < N;
        const size_t sort_count = compact_active ? active_count : N;

        float* d_keys = nullptr;
        uint32_t* d_indices = nullptr;
        float* d_keys_sorted = nullptr;
        uint32_t* d_indices_sorted = nullptr;

        cuda_scratch::DeviceBuffer keys_buffer;
        cuda_scratch::DeviceBuffer indices_buffer;
        cuda_scratch::DeviceBuffer sorted_keys_buffer;
        cuda_scratch::DeviceBuffer sorted_indices_buffer;

        if (scratch) {
            scratch->ensure_n(sort_count, lfs::core::Device::CUDA);
            LFS_ASSERT_MSG(scratch->n_capacity >= sort_count,
                           lfs::core::detail::format_cuda_safe(
                               "Gumbel scratch n_capacity must be >= sort_count (cap={}, sort_count={})",
                               scratch->n_capacity, sort_count));
            LFS_ASSERT_MSG(scratch->keys.is_valid() && scratch->keys.ptr<float>() != nullptr,
                           "Gumbel scratch keys buffer must be a non-null CUDA f32 tensor");
            LFS_ASSERT_MSG(scratch->indices.is_valid() && scratch->indices.ptr<uint32_t>() != nullptr,
                           "Gumbel scratch indices buffer must be a non-null CUDA uint32 tensor");
            LFS_ASSERT_MSG(
                scratch->keys_sorted.is_valid() && scratch->keys_sorted.ptr<float>() != nullptr,
                "Gumbel scratch sorted-keys buffer must be a non-null CUDA f32 tensor");
            LFS_ASSERT_MSG(
                scratch->indices_sorted.is_valid() &&
                    scratch->indices_sorted.ptr<uint32_t>() != nullptr,
                "Gumbel scratch sorted-indices buffer must be a non-null CUDA uint32 tensor");
            d_keys = scratch->keys.ptr<float>();
            d_indices = scratch->indices.ptr<uint32_t>();
            d_keys_sorted = scratch->keys_sorted.ptr<float>();
            d_indices_sorted = scratch->indices_sorted.ptr<uint32_t>();
        } else {
            const size_t keys_bytes = cuda_scratch::checked_bytes(
                sort_count, sizeof(float), "MRNF Gumbel keys");
            const size_t indices_bytes = cuda_scratch::checked_bytes(
                sort_count, sizeof(uint32_t), "MRNF Gumbel indices");
            keys_buffer = cuda_scratch::DeviceBuffer(keys_bytes, s, "mrnf.gumbel.keys");
            indices_buffer = cuda_scratch::DeviceBuffer(indices_bytes, s, "mrnf.gumbel.indices");
            sorted_keys_buffer = cuda_scratch::DeviceBuffer(keys_bytes, s, "mrnf.gumbel.keys_sorted");
            sorted_indices_buffer = cuda_scratch::DeviceBuffer(
                indices_bytes, s, "mrnf.gumbel.indices_sorted");
            d_keys = keys_buffer.as<float>();
            d_indices = indices_buffer.as<uint32_t>();
            d_keys_sorted = sorted_keys_buffer.as<float>();
            d_indices_sorted = sorted_indices_buffer.as<uint32_t>();
        }

        constexpr int threads = 256;
        const int blocks = static_cast<int>((sort_count + threads - 1) / threads);

        if (compact_active) {
            auto indices_ptr = thrust::device_pointer_cast(d_indices);
            auto counting_begin = thrust::make_counting_iterator<uint32_t>(0);
            thrust::copy_if(
                thrust::cuda::par.on(s),
                counting_begin,
                counting_begin + static_cast<std::ptrdiff_t>(N),
                weights_ptr,
                indices_ptr,
                positive_weight{});
            gumbel_key_for_indices_kernel<<<blocks, threads, 0, s>>>(
                weights, d_indices, d_keys, sort_count, seed);
            LFS_CUDA_LAUNCH_CHECK(s, "training.mrnf.gumbel_keys_indices");
        } else {
            gumbel_key_kernel<<<blocks, threads, 0, s>>>(weights, d_keys, sort_count, seed);
            LFS_CUDA_LAUNCH_CHECK(s, "training.mrnf.gumbel_keys");
            auto indices_ptr = thrust::device_pointer_cast(d_indices);
            lfs::core::tensor_ops::run_with_thrust_policy(s, [&](auto policy) {
                thrust::sequence(policy, indices_ptr, indices_ptr + static_cast<std::ptrdiff_t>(sort_count));
            });
        }
        LFS_CUDA_CHECK_MSG(cudaGetLastError(), "MRNF Gumbel key generation");

        const int sort_count_int = static_cast<int>(sort_count);
        auto sort_pairs = [&](void* workspace, size_t& workspace_bytes) {
            return cub::DeviceRadixSort::SortPairsDescending(
                workspace, workspace_bytes,
                d_keys, d_keys_sorted,
                d_indices, d_indices_sorted,
                sort_count_int, 0, 32, s);
        };

        if (scratch) {
            size_t workspace_bytes = 0;
            LFS_CUDA_CHECK_MSG(sort_pairs(nullptr, workspace_bytes),
                               "MRNF Gumbel CUB workspace query");
            scratch->ensure_cub(workspace_bytes, lfs::core::Device::CUDA);
            LFS_ASSERT_MSG(workspace_bytes == 0 ||
                               (scratch->cub.is_valid() && scratch->cub_bytes >= workspace_bytes &&
                                scratch->cub.data_ptr() != nullptr),
                           lfs::core::detail::format_cuda_safe(
                               "Gumbel CUB workspace must cover queried bytes (have={}, need={})",
                               scratch->cub_bytes, workspace_bytes));
            LFS_CUDA_CHECK_MSG(
                sort_pairs(workspace_bytes == 0 ? nullptr : scratch->cub.data_ptr(),
                           workspace_bytes),
                "MRNF Gumbel CUB sort");
        } else {
            cuda_scratch::CubWorkspace cub_workspace(
                "cub::DeviceRadixSort::SortPairsDescending", s, sort_pairs);
            cub_workspace.run(sort_pairs);
        }

        const int output_blocks = static_cast<int>((K + threads - 1) / threads);
        widen_gumbel_indices_kernel<<<output_blocks, threads, 0, s>>>(
            d_indices_sorted, output_indices, K);
        LFS_CUDA_LAUNCH_CHECK(s, "MRNF Gumbel uint32 output widening");
    }

} // namespace lfs::training::mrnf_strategy

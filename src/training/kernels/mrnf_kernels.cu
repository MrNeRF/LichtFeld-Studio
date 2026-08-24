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
            return logf(p / (1.0f - p));
        }

        struct positive_weight {
            __host__ __device__ bool operator()(const float w) const {
                return w > 0.0f;
            }
        };

    } // namespace

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
        const bool* __restrict__ far_mask,
        size_t far_mask_size,
        float opacity_decay,
        float scale_decay,
        float far_decay_scale,
        float train_t,
        size_t N) {

        const size_t idx = threadIdx.x + blockIdx.x * static_cast<size_t>(blockDim.x);
        if (idx >= N)
            return;
        if (frozen_mask != nullptr && idx < frozen_mask_size && frozen_mask[idx])
            return;

        float opac_decay = opacity_decay;
        float scl_decay = scale_decay;
        if (far_mask != nullptr && idx < far_mask_size && far_mask[idx]) {
            opac_decay *= far_decay_scale;
            scl_decay *= far_decay_scale;
        }

        const float t_shrink = 1.0f - train_t;

        float opac = d_sigmoid(raw_opacities[idx]) - opac_decay * t_shrink;
        opac = fminf(fmaxf(opac, 1e-12f), 1.0f - 1e-12f);
        raw_opacities[idx] = d_logit(opac);

        const float decay_factor = 1.0f - scl_decay * t_shrink;
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
        const bool* far_mask,
        size_t far_mask_size,
        float opacity_decay,
        float scale_decay,
        float far_decay_scale,
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
            far_mask, far_mask_size, opacity_decay, scale_decay, far_decay_scale,
            train_t, N);
        LFS_CUDA_LAUNCH_CHECK(s, "training.mrnf.decay");
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
        cudaStream_t s = resolve_stream(stream);
        elementwise_add_inplace_kernel<<<blocks, threads, 0, s>>>(a, b, N);
        LFS_CUDA_LAUNCH_CHECK(s, "training.mrnf.elementwise_add");
    }

    __global__ void fold_densification_and_zero_kernel(
        float* __restrict__ vis_count,
        float* __restrict__ refine_weight_max,
        float* __restrict__ densification_info,
        size_t N,
        size_t n_rows,
        float* __restrict__ ratio_max,
        float ratio_pow) {

        const size_t idx = threadIdx.x + blockIdx.x * static_cast<size_t>(blockDim.x);
        if (idx >= N)
            return;

        const float vis = densification_info[idx];
        const float err = densification_info[N + idx];
        vis_count[idx] += vis;
        refine_weight_max[idx] = fmaxf(refine_weight_max[idx], err);
        if (ratio_max != nullptr) {
            const float ratio = (vis >= 0.05f) ? (ratio_pow > 0.0f ? (err / powf(vis, ratio_pow)) : (err / vis)) : 0.0f;
            ratio_max[idx] = fmaxf(ratio_max[idx], ratio);
        }
        for (size_t row = 0; row < n_rows; ++row) {
            densification_info[row * N + idx] = 0.f;
        }
    }

    void launch_fold_densification_and_zero(
        float* vis_count,
        float* refine_weight_max,
        float* densification_info,
        size_t N,
        void* stream,
        size_t n_rows,
        float* ratio_max,
        float ratio_pow) {
        if (N == 0)
            return;
        const size_t rows = n_rows >= 2 ? n_rows : 2;
        constexpr int threads = 256;
        const int blocks = static_cast<int>((N + threads - 1) / threads);
        cudaStream_t s = resolve_stream(stream);
        fold_densification_and_zero_kernel<<<blocks, threads, 0, s>>>(
            vis_count, refine_weight_max, densification_info, N, rows, ratio_max, ratio_pow);
        LFS_CUDA_LAUNCH_CHECK(s, "training.mrnf.fold_densification_and_zero");
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

    __global__ void geomean_extent_kernel(
        const float* __restrict__ scaling_raw,
        float* __restrict__ extents,
        size_t N) {
        const size_t idx = threadIdx.x + blockIdx.x * static_cast<size_t>(blockDim.x);
        if (idx >= N)
            return;
        const float s0 = scaling_raw[idx * 3 + 0];
        const float s1 = scaling_raw[idx * 3 + 1];
        const float s2 = scaling_raw[idx * 3 + 2];
        const float g = expf((s0 + s1 + s2) * (1.0f / 3.0f));
        extents[idx] = (isfinite(g) && g > 0.0f) ? g : 0.0f;
    }

    __global__ void fill_pos_inf_kernel(float* data, size_t n) {
        const size_t i = threadIdx.x + blockIdx.x * static_cast<size_t>(blockDim.x);
        if (i < n)
            data[i] = INFINITY;
    }

    __global__ void store_positive_median_kernel(
        const float* __restrict__ sorted,
        const int* __restrict__ count,
        float* __restrict__ out) {
        const int c = *count;
        if (c <= 0) {
            *out = 0.0f;
            return;
        }
        const float m = sorted[c / 2];
        *out = (isfinite(m) && m > 0.0f) ? m : 0.0f;
    }

    void launch_median_geomean_extent(
        const float* scaling_raw,
        size_t N,
        float* out_median,
        bool* out_valid,
        void* stream) {

        LFS_ASSERT(out_median != nullptr);
        LFS_ASSERT(out_valid != nullptr);
        *out_median = 0.0f;
        *out_valid = false;
        if (N == 0 || scaling_raw == nullptr)
            return;
        LFS_ASSERT_MSG(N <= static_cast<size_t>(std::numeric_limits<int>::max()),
                       "MRNF median-extent input exceeds CUB's int item-count limit");

        cudaStream_t s = resolve_stream(stream);
        const int n_int = static_cast<int>(N);
        constexpr int threads = 256;
        const int blocks = static_cast<int>((N + threads - 1) / threads);

        const size_t values_bytes = cuda_scratch::checked_bytes(
            N, sizeof(float), "MRNF median geomean extents");
        cuda_scratch::DeviceBuffer extents_buf(values_bytes, s, "mrnf.median_extent.input");
        cuda_scratch::DeviceBuffer selected_buf(values_bytes, s, "mrnf.median_extent.selected");
        cuda_scratch::DeviceBuffer sorted_buf(values_bytes, s, "mrnf.median_extent.sorted");
        cuda_scratch::DeviceBuffer count_buf(sizeof(int), s, "mrnf.median_extent.count");
        cuda_scratch::DeviceBuffer median_buf(sizeof(float), s, "mrnf.median_extent.scalar");

        auto* d_ext = extents_buf.as<float>();
        auto* d_sel = selected_buf.as<float>();
        auto* d_sorted = sorted_buf.as<float>();
        auto* d_count = count_buf.as<int>();
        auto* d_median = median_buf.as<float>();
        LFS_CUDA_CHECK_MSG(cudaMemsetAsync(d_count, 0, sizeof(int), s),
                           "MRNF median-extent count zero");
        LFS_CUDA_CHECK_MSG(cudaMemsetAsync(d_median, 0, sizeof(float), s),
                           "MRNF median-extent scalar zero");

        geomean_extent_kernel<<<blocks, threads, 0, s>>>(scaling_raw, d_ext, N);
        LFS_CUDA_LAUNCH_CHECK(s, "training.mrnf.geomean_extent");
        fill_pos_inf_kernel<<<blocks, threads, 0, s>>>(d_sel, N);
        LFS_CUDA_LAUNCH_CHECK(s, "training.mrnf.median_extent_fill_inf");

        // Named predicate, kept from the PR #1840 Windows fix: this CUB call sits inside a
        // host lambda, where nvcc+MSVC rejected a braced temporary. Do not inline it back.
        const positive_weight positive_pred;
        auto select_op = [&](void* workspace, size_t& workspace_bytes) {
            return cub::DeviceSelect::If(
                workspace, workspace_bytes, d_ext, d_sel, d_count,
                n_int, positive_pred, s);
        };
        size_t select_bytes = 0;
        LFS_CUDA_CHECK_MSG(select_op(nullptr, select_bytes), "MRNF median-extent select size");
        if (select_bytes > 0) {
            cuda_scratch::CubWorkspace select_ws(
                "cub::DeviceSelect::If", s, select_op);
            select_ws.run(select_op);
        } else {
            LFS_CUDA_CHECK_MSG(select_op(nullptr, select_bytes), "MRNF median-extent select");
        }

        auto sort_op = [&](void* workspace, size_t& workspace_bytes) {
            return cub::DeviceRadixSort::SortKeys(
                workspace, workspace_bytes, d_sel, d_sorted, n_int, 0, 32, s);
        };
        size_t sort_bytes = 0;
        LFS_CUDA_CHECK_MSG(sort_op(nullptr, sort_bytes), "MRNF median-extent sort size");
        if (sort_bytes > 0) {
            cuda_scratch::CubWorkspace sort_ws(
                "cub::DeviceRadixSort::SortKeys", s, sort_op);
            sort_ws.run(sort_op);
        } else {
            LFS_CUDA_CHECK_MSG(sort_op(nullptr, sort_bytes), "MRNF median-extent sort");
        }

        store_positive_median_kernel<<<1, 1, 0, s>>>(d_sorted, d_count, d_median);
        LFS_CUDA_LAUNCH_CHECK(s, "training.mrnf.store_positive_median");
        LFS_CUDA_CHECK_MSG(
            cudaMemcpyAsync(out_median, d_median, sizeof(float), cudaMemcpyDeviceToHost, s),
            "MRNF median-extent readback");
        LFS_CUDA_CHECK_MSG(cudaStreamSynchronize(s), "MRNF median-extent stream sync");
        *out_valid = std::isfinite(*out_median) && *out_median > 0.0f;
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
        const int64_t* __restrict__ indices,
        float* __restrict__ keys,
        size_t N,
        uint64_t seed) {

        const size_t idx = threadIdx.x + blockIdx.x * static_cast<size_t>(blockDim.x);
        if (idx >= N)
            return;

        const int64_t src_idx = indices[idx];
        const float w = weights[src_idx];

        curandStatePhilox4_32_10_t rng;
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
        void* stream,
        bool compact_sparse,
        GumbelTopKScratch* scratch,
        size_t known_nnz) {

        LFS_ASSERT(K <= N);
        if (K == 0)
            return;

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
        int64_t* d_indices = nullptr;
        float* d_keys_sorted = nullptr;
        int64_t* d_indices_sorted = nullptr;

        cuda_scratch::DeviceBuffer keys_buffer;
        cuda_scratch::DeviceBuffer indices_buffer;
        cuda_scratch::DeviceBuffer sorted_keys_buffer;
        cuda_scratch::DeviceBuffer sorted_indices_buffer;

        if (scratch) {
            scratch->ensure_n(N, lfs::core::Device::CUDA);
            LFS_ASSERT_MSG(scratch->n_capacity >= sort_count,
                           lfs::core::detail::format_cuda_safe(
                               "Gumbel scratch n_capacity must be >= sort_count (cap={}, sort_count={})",
                               scratch->n_capacity, sort_count));
            LFS_ASSERT_MSG(scratch->keys.is_valid() && scratch->keys.ptr<float>() != nullptr,
                           "Gumbel scratch keys buffer must be a non-null CUDA f32 tensor");
            LFS_ASSERT_MSG(scratch->indices.is_valid() && scratch->indices.ptr<int64_t>() != nullptr,
                           "Gumbel scratch indices buffer must be a non-null CUDA i64 tensor");
            LFS_ASSERT_MSG(
                scratch->keys_sorted.is_valid() && scratch->keys_sorted.ptr<float>() != nullptr,
                "Gumbel scratch sorted-keys buffer must be a non-null CUDA f32 tensor");
            LFS_ASSERT_MSG(
                scratch->indices_sorted.is_valid() &&
                    scratch->indices_sorted.ptr<int64_t>() != nullptr,
                "Gumbel scratch sorted-indices buffer must be a non-null CUDA i64 tensor");
            d_keys = scratch->keys.ptr<float>();
            d_indices = scratch->indices.ptr<int64_t>();
            d_keys_sorted = scratch->keys_sorted.ptr<float>();
            d_indices_sorted = scratch->indices_sorted.ptr<int64_t>();
        } else {
            const size_t keys_bytes = cuda_scratch::checked_bytes(
                sort_count, sizeof(float), "MRNF Gumbel keys");
            const size_t indices_bytes = cuda_scratch::checked_bytes(
                sort_count, sizeof(int64_t), "MRNF Gumbel indices");
            keys_buffer = cuda_scratch::DeviceBuffer(keys_bytes, s, "mrnf.gumbel.keys");
            indices_buffer = cuda_scratch::DeviceBuffer(indices_bytes, s, "mrnf.gumbel.indices");
            sorted_keys_buffer = cuda_scratch::DeviceBuffer(keys_bytes, s, "mrnf.gumbel.keys_sorted");
            sorted_indices_buffer = cuda_scratch::DeviceBuffer(
                indices_bytes, s, "mrnf.gumbel.indices_sorted");
            d_keys = keys_buffer.as<float>();
            d_indices = indices_buffer.as<int64_t>();
            d_keys_sorted = sorted_keys_buffer.as<float>();
            d_indices_sorted = sorted_indices_buffer.as<int64_t>();
        }

        constexpr int threads = 256;
        const int blocks = static_cast<int>((sort_count + threads - 1) / threads);

        if (compact_active) {
            auto indices_ptr = thrust::device_pointer_cast(d_indices);
            auto counting_begin = thrust::make_counting_iterator<int64_t>(0);
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

        LFS_CUDA_CHECK_MSG(
            cudaMemcpyAsync(
                output_indices, d_indices_sorted,
                cuda_scratch::checked_bytes(K, sizeof(int64_t), "MRNF Gumbel output"),
                cudaMemcpyDeviceToDevice, s),
            "MRNF Gumbel output copy");
    }

    __global__ void project_visible_centers_kernel(
        const float* __restrict__ means,
        const float* __restrict__ w2c,
        float fx,
        float fy,
        float cx,
        float cy,
        int width,
        int height,
        float near_plane,
        float* __restrict__ means2d,
        float* __restrict__ radii,
        size_t N) {

        const size_t idx = threadIdx.x + blockIdx.x * static_cast<size_t>(blockDim.x);
        if (idx >= N)
            return;

        const float x = means[idx * 3 + 0];
        const float y = means[idx * 3 + 1];
        const float z = means[idx * 3 + 2];
        const float cam_x = w2c[0] * x + w2c[1] * y + w2c[2] * z + w2c[3];
        const float cam_y = w2c[4] * x + w2c[5] * y + w2c[6] * z + w2c[7];
        const float cam_z = w2c[8] * x + w2c[9] * y + w2c[10] * z + w2c[11];
        if (!(cam_z > near_plane) || !isfinite(cam_x) || !isfinite(cam_y) || !isfinite(cam_z)) {
            means2d[idx * 2 + 0] = 0.0f;
            means2d[idx * 2 + 1] = 0.0f;
            radii[idx] = 0.0f;
            return;
        }

        const float px = fx * (cam_x / cam_z) + cx;
        const float py = fy * (cam_y / cam_z) + cy;
        means2d[idx * 2 + 0] = px;
        means2d[idx * 2 + 1] = py;
        const bool in_image = px >= 0.0f && py >= 0.0f &&
                              px < static_cast<float>(width) &&
                              py < static_cast<float>(height);
        radii[idx] = in_image ? 1.0f : 0.0f;
    }

    void launch_project_visible_centers(
        const float* means,
        const float* w2c,
        float fx,
        float fy,
        float cx,
        float cy,
        int width,
        int height,
        float near_plane,
        float* means2d,
        float* radii,
        size_t N,
        void* stream) {

        if (N == 0)
            return;
        LFS_ASSERT(means != nullptr);
        LFS_ASSERT(w2c != nullptr);
        LFS_ASSERT(means2d != nullptr);
        LFS_ASSERT(radii != nullptr);
        LFS_ASSERT(width > 0);
        LFS_ASSERT(height > 0);

        constexpr int threads = 256;
        const int blocks = static_cast<int>((N + threads - 1) / threads);
        cudaStream_t s = resolve_stream(stream);
        project_visible_centers_kernel<<<blocks, threads, 0, s>>>(
            means, w2c, fx, fy, cx, cy, width, height, near_plane, means2d, radii, N);
        LFS_CUDA_LAUNCH_CHECK(s, "training.mrnf.project_visible_centers");
    }

    __global__ void gather_center_error_kernel(
        const float* __restrict__ means2d,
        const float* __restrict__ radii,
        const float* __restrict__ error,
        int width,
        int height,
        float* __restrict__ scores,
        size_t N) {

        const size_t idx = threadIdx.x + blockIdx.x * static_cast<size_t>(blockDim.x);
        if (idx >= N)
            return;

        if (radii[idx] <= 0.0f) {
            scores[idx] = 0.0f;
            return;
        }

        int x = static_cast<int>(floorf(means2d[idx * 2 + 0]));
        int y = static_cast<int>(floorf(means2d[idx * 2 + 1]));
        x = max(0, min(x, width - 1));
        y = max(0, min(y, height - 1));
        scores[idx] = error[static_cast<size_t>(y) * static_cast<size_t>(width) + static_cast<size_t>(x)];
    }

    void launch_gather_center_error(
        const float* means2d,
        const float* radii,
        const float* error,
        int width,
        int height,
        float* scores,
        size_t N,
        void* stream) {

        if (N == 0)
            return;
        LFS_ASSERT(means2d != nullptr);
        LFS_ASSERT(radii != nullptr);
        LFS_ASSERT(error != nullptr);
        LFS_ASSERT(scores != nullptr);
        LFS_ASSERT(width > 0);
        LFS_ASSERT(height > 0);

        constexpr int threads = 256;
        const int blocks = static_cast<int>((N + threads - 1) / threads);
        cudaStream_t s = resolve_stream(stream);
        gather_center_error_kernel<<<blocks, threads, 0, s>>>(
            means2d, radii, error, width, height, scores, N);
        LFS_CUDA_LAUNCH_CHECK(s, "training.mrnf.gather_center_error");
    }

    __global__ void far_field_mask_kernel(
        const float* __restrict__ means,
        float centroid_x,
        float centroid_y,
        float centroid_z,
        float far_radius_sq,
        bool* __restrict__ far_out,
        size_t N) {

        const size_t idx = threadIdx.x + blockIdx.x * static_cast<size_t>(blockDim.x);
        if (idx >= N)
            return;

        const float dx = means[idx * 3 + 0] - centroid_x;
        const float dy = means[idx * 3 + 1] - centroid_y;
        const float dz = means[idx * 3 + 2] - centroid_z;
        far_out[idx] = (dx * dx + dy * dy + dz * dz) > far_radius_sq;
    }

    void launch_far_field_mask(
        const float* means,
        float centroid_x,
        float centroid_y,
        float centroid_z,
        float far_radius,
        bool* far_out,
        size_t N,
        void* stream) {

        if (N == 0)
            return;
        LFS_ASSERT(means != nullptr);
        LFS_ASSERT(far_out != nullptr);

        constexpr int threads = 256;
        const int blocks = static_cast<int>((N + threads - 1) / threads);
        cudaStream_t s = resolve_stream(stream);
        const float far_radius_sq = far_radius * far_radius;
        far_field_mask_kernel<<<blocks, threads, 0, s>>>(
            means, centroid_x, centroid_y, centroid_z, far_radius_sq, far_out, N);
        LFS_CUDA_LAUNCH_CHECK(s, "training.mrnf.far_field_mask");
    }

    __global__ void mean_abs_error_hw_kernel(
        const float* __restrict__ pred,
        const float* __restrict__ target,
        int channels,
        size_t hw,
        float* __restrict__ out_hw) {

        const size_t idx = threadIdx.x + blockIdx.x * static_cast<size_t>(blockDim.x);
        if (idx >= hw)
            return;

        float sum = 0.0f;
        for (int c = 0; c < channels; ++c) {
            const size_t pix = static_cast<size_t>(c) * hw + idx;
            sum += fabsf(pred[pix] - target[pix]);
        }
        out_hw[idx] = sum / static_cast<float>(channels);
    }

    void launch_mean_abs_error_hw(
        const float* pred,
        const float* target,
        int channels,
        int height,
        int width,
        float* out_hw,
        void* stream) {

        LFS_ASSERT(pred != nullptr);
        LFS_ASSERT(target != nullptr);
        LFS_ASSERT(out_hw != nullptr);
        LFS_ASSERT(channels > 0);
        LFS_ASSERT(height > 0);
        LFS_ASSERT(width > 0);

        const size_t hw = static_cast<size_t>(height) * static_cast<size_t>(width);
        constexpr int threads = 256;
        const int blocks = static_cast<int>((hw + threads - 1) / threads);
        cudaStream_t s = resolve_stream(stream);
        mean_abs_error_hw_kernel<<<blocks, threads, 0, s>>>(
            pred, target, channels, hw, out_hw);
        LFS_CUDA_LAUNCH_CHECK(s, "training.mrnf.mean_abs_error_hw");
    }

    __global__ void seed_weights_from_error_alpha_kernel(
        const float* __restrict__ error_hw,
        const float* __restrict__ alpha,
        float* __restrict__ out_weights,
        size_t hw) {

        const size_t idx = threadIdx.x + blockIdx.x * static_cast<size_t>(blockDim.x);
        if (idx >= hw)
            return;
        out_weights[idx] = error_hw[idx] * (1.0f - alpha[idx]);
    }

    void launch_seed_weights_from_error_alpha(
        const float* error_hw,
        const float* alpha,
        float* out_weights,
        size_t hw,
        void* stream) {

        if (hw == 0)
            return;
        LFS_ASSERT(error_hw != nullptr);
        LFS_ASSERT(alpha != nullptr);
        LFS_ASSERT(out_weights != nullptr);

        constexpr int threads = 256;
        const int blocks = static_cast<int>((hw + threads - 1) / threads);
        cudaStream_t s = resolve_stream(stream);
        seed_weights_from_error_alpha_kernel<<<blocks, threads, 0, s>>>(
            error_hw, alpha, out_weights, hw);
        LFS_CUDA_LAUNCH_CHECK(s, "training.mrnf.seed_weights");
    }

    __global__ void gather_seed_payloads_chw_kernel(
        const int64_t* __restrict__ pixel_indices,
        size_t K,
        size_t hw,
        const float* __restrict__ target,
        int channels,
        const float* __restrict__ alpha,
        const float* __restrict__ depth,
        float* __restrict__ out_rgb,
        float* __restrict__ out_alpha,
        float* __restrict__ out_depth) {

        const size_t idx = threadIdx.x + blockIdx.x * static_cast<size_t>(blockDim.x);
        if (idx >= K)
            return;

        const int64_t pix64 = pixel_indices[idx];
        const size_t pix = (pix64 >= 0) ? static_cast<size_t>(pix64) : 0;
        const size_t clamped = (pix < hw) ? pix : 0;
        const int use_c = channels > 0 ? channels : 1;
        for (int c = 0; c < 3; ++c) {
            const int src_c = (c < use_c) ? c : (use_c - 1);
            out_rgb[idx * 3 + static_cast<size_t>(c)] =
                target[static_cast<size_t>(src_c) * hw + clamped];
        }
        out_alpha[idx] = alpha[clamped];
        out_depth[idx] = depth ? depth[clamped] : 0.0f;
    }

    void launch_gather_seed_payloads(
        const int64_t* pixel_indices,
        size_t K,
        size_t hw,
        const float* target,
        int channels,
        const float* alpha,
        const float* depth,
        float* out_rgb,
        float* out_alpha,
        float* out_depth,
        void* stream) {

        if (K == 0)
            return;
        LFS_ASSERT(pixel_indices != nullptr);
        LFS_ASSERT(target != nullptr);
        LFS_ASSERT(alpha != nullptr);
        LFS_ASSERT(out_rgb != nullptr);
        LFS_ASSERT(out_alpha != nullptr);
        LFS_ASSERT(out_depth != nullptr);
        LFS_ASSERT(hw > 0);
        LFS_ASSERT(channels > 0);

        constexpr int threads = 256;
        const int blocks = static_cast<int>((K + threads - 1) / threads);
        cudaStream_t s = resolve_stream(stream);
        gather_seed_payloads_chw_kernel<<<blocks, threads, 0, s>>>(
            pixel_indices, K, hw, target, channels, alpha, depth,
            out_rgb, out_alpha, out_depth);
        LFS_CUDA_LAUNCH_CHECK(s, "training.mrnf.gather_seed_payloads");
    }

    void launch_sorted_median(
        const float* values,
        size_t n,
        float* out_median,
        lfs::training::PositiveMedianScratch* scratch,
        void* stream) {

        LFS_ASSERT(out_median != nullptr);
        *out_median = 0.0f;
        if (n == 0 || values == nullptr)
            return;
        LFS_ASSERT_MSG(n <= static_cast<size_t>(std::numeric_limits<int>::max()),
                       "MRNF sorted-median input exceeds CUB's int item-count limit");
        LFS_ASSERT_MSG(scratch != nullptr, "MRNF sorted-median requires PositiveMedianScratch");

        cudaStream_t s = resolve_stream(stream);
        const int n_int = static_cast<int>(n);
        scratch->ensure_n(n, lfs::core::Device::CUDA);
        LFS_ASSERT_MSG(scratch->n_capacity >= n &&
                           scratch->selected.is_valid() &&
                           scratch->sorted.is_valid(),
                       "MRNF sorted-median scratch must cover n");

        float* d_selected = scratch->selected.ptr<float>();
        float* d_sorted = scratch->sorted.ptr<float>();
        LFS_CUDA_CHECK_MSG(
            cudaMemcpyAsync(d_selected, values, n * sizeof(float), cudaMemcpyDeviceToDevice, s),
            "MRNF sorted-median copy");

        auto sort_op = [&](void* workspace, size_t& workspace_bytes) {
            return cub::DeviceRadixSort::SortKeys(
                workspace, workspace_bytes, d_selected, d_sorted,
                n_int, 0, static_cast<int>(sizeof(float) * 8), s);
        };
        size_t sort_bytes = 0;
        LFS_CUDA_CHECK_MSG(sort_op(nullptr, sort_bytes), "MRNF sorted-median sort size");
        scratch->ensure_temps(0, sort_bytes, lfs::core::Device::CUDA);
        if (sort_bytes > 0) {
            LFS_ASSERT_MSG(scratch->sort_temp.is_valid() &&
                               scratch->sort_temp_bytes >= sort_bytes &&
                               scratch->sort_temp.data_ptr() != nullptr,
                           "MRNF sorted-median sort temp must cover queried bytes");
            void* ws = scratch->sort_temp.data_ptr();
            LFS_CUDA_CHECK_MSG(sort_op(ws, sort_bytes), "MRNF sorted-median sort");
        } else {
            LFS_CUDA_CHECK_MSG(sort_op(nullptr, sort_bytes), "MRNF sorted-median sort");
        }

        LFS_CUDA_CHECK_MSG(
            cudaMemcpyAsync(out_median, d_sorted + (n / 2), sizeof(float),
                            cudaMemcpyDeviceToHost, s),
            "MRNF sorted-median readback");
        LFS_CUDA_CHECK_MSG(cudaStreamSynchronize(s), "MRNF sorted-median stream sync");
        if (!std::isfinite(*out_median))
            *out_median = 0.0f;
    }

    __global__ void apply_explore_starvation_weights_kernel(
        float* __restrict__ weights,
        const float* __restrict__ vis_count,
        size_t n,
        float median_vis) {

        const size_t idx = threadIdx.x + blockIdx.x * static_cast<size_t>(blockDim.x);
        if (idx >= n)
            return;
        const float vis_i = vis_count[idx];
        if (vis_i == 0.0f) {
            weights[idx] = 0.0f;
            return;
        }
        const float denom = fmaxf(median_vis, 1.19209290e-07f);
        const float starved = fminf(fmaxf(1.0f - vis_i / denom, 0.0f), 1.0f);
        const float term = powf(starved, kStarvGamma);
        weights[idx] *= (kStarvEps + term);
    }

    void launch_apply_explore_starvation_weights(
        float* weights,
        const float* vis_count,
        size_t n,
        float median_vis,
        void* stream) {

        if (n == 0)
            return;
        LFS_ASSERT(weights != nullptr);
        LFS_ASSERT(vis_count != nullptr);

        constexpr int threads = 256;
        const int blocks = static_cast<int>((n + threads - 1) / threads);
        cudaStream_t s = resolve_stream(stream);
        apply_explore_starvation_weights_kernel<<<blocks, threads, 0, s>>>(
            weights, vis_count, n, median_vis);
        LFS_CUDA_LAUNCH_CHECK(s, "training.mrnf.explore_starvation_weights");
    }

} // namespace lfs::training::mrnf_strategy

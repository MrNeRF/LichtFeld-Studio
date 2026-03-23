/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "lfs_kernels.hpp"
#include <algorithm>
#include <cassert>
#include <cub/cub.cuh>
#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <thrust/execution_policy.h>
#include <thrust/sequence.h>

namespace lfs::training::lfs_strategy {

    namespace {

        __device__ __forceinline__ float d_sigmoid(float x) {
            return 1.0f / (1.0f + expf(-x));
        }

        __device__ __forceinline__ float d_logit(float p) {
            return logf(p / (1.0f - p));
        }

        __device__ __forceinline__ void d_quat_rotate(
            float qw, float qx, float qy, float qz,
            float vx, float vy, float vz,
            float& ox, float& oy, float& oz) {
            const float tx = 2.0f * (qy * vz - qz * vy);
            const float ty = 2.0f * (qz * vx - qx * vz);
            const float tz = 2.0f * (qx * vy - qy * vx);
            ox = vx + qw * tx + (qy * tz - qz * ty);
            oy = vy + qw * ty + (qz * tx - qx * tz);
            oz = vz + qw * tz + (qx * ty - qy * tx);
        }

    } // namespace

    __global__ void lfs_split_kernel(
        const int64_t* __restrict__ split_indices,
        float* __restrict__ means,
        float* __restrict__ log_scales,
        float* __restrict__ raw_opacities,
        const float* __restrict__ rotations,
        const float* __restrict__ sh0,
        const float* __restrict__ shN,
        float* __restrict__ child_means,
        float* __restrict__ child_log_scales,
        float* __restrict__ child_raw_opacities,
        float* __restrict__ child_rotations,
        float* __restrict__ child_sh0,
        float* __restrict__ child_shN,
        size_t K,
        size_t sh_rest) {

        const size_t idx = threadIdx.x + blockIdx.x * static_cast<size_t>(blockDim.x);
        if (idx >= K)
            return;

        const int64_t si = split_indices[idx];

        const float orig_mean0 = means[si * 3 + 0];
        const float orig_mean1 = means[si * 3 + 1];
        const float orig_mean2 = means[si * 3 + 2];

        // Match the fixed IGS+ split rule: opacity *= 0.6 in opacity space.
        const float opac = d_sigmoid(raw_opacities[si]);
        const float new_opac = fminf(fmaxf(opac * 0.6f, 1e-7f), 1.0f - 1e-7f);
        const float new_raw_opac = d_logit(new_opac);

        const float ls0 = log_scales[si * 3 + 0];
        const float ls1 = log_scales[si * 3 + 1];
        const float ls2 = log_scales[si * 3 + 2];
        const float s0 = expf(ls0), s1 = expf(ls1), s2 = expf(ls2);

        // Match IGS+: longest axis shrinks by 0.5, others by 0.85.
        float new_ls0 = ls0 + logf(0.85f);
        float new_ls1 = ls1 + logf(0.85f);
        float new_ls2 = ls2 + logf(0.85f);
        float local_offset[3] = {0.0f, 0.0f, 0.0f};
        float offset_magnitude = 0.5f * s0;

        if (s0 >= s1 && s0 >= s2) {
            new_ls0 = ls0 + logf(0.5f);
            offset_magnitude = 0.5f * s0;
            local_offset[0] = offset_magnitude;
        } else if (s1 >= s2) {
            new_ls1 = ls1 + logf(0.5f);
            offset_magnitude = 0.5f * s1;
            local_offset[1] = offset_magnitude;
        } else {
            new_ls2 = ls2 + logf(0.5f);
            offset_magnitude = 0.5f * s2;
            local_offset[2] = offset_magnitude;
        }

        // Rotate offset by quaternion
        float qw = rotations[si * 4 + 0];
        float qx = rotations[si * 4 + 1];
        float qy = rotations[si * 4 + 2];
        float qz = rotations[si * 4 + 3];
        const float qnorm = rsqrtf(fmaxf(qw * qw + qx * qx + qy * qy + qz * qz, 1e-8f));
        qw *= qnorm;
        qx *= qnorm;
        qy *= qnorm;
        qz *= qnorm;

        float ox, oy, oz;
        d_quat_rotate(qw, qx, qy, qz, local_offset[0], local_offset[1], local_offset[2], ox, oy, oz);

        // Modify parent in-place using the same orientation as IGS+.
        means[si * 3 + 0] = orig_mean0 + ox;
        means[si * 3 + 1] = orig_mean1 + oy;
        means[si * 3 + 2] = orig_mean2 + oz;
        log_scales[si * 3 + 0] = new_ls0;
        log_scales[si * 3 + 1] = new_ls1;
        log_scales[si * 3 + 2] = new_ls2;
        raw_opacities[si] = new_raw_opac;

        // Output child
        child_means[idx * 3 + 0] = orig_mean0 - ox;
        child_means[idx * 3 + 1] = orig_mean1 - oy;
        child_means[idx * 3 + 2] = orig_mean2 - oz;
        child_log_scales[idx * 3 + 0] = new_ls0;
        child_log_scales[idx * 3 + 1] = new_ls1;
        child_log_scales[idx * 3 + 2] = new_ls2;
        child_raw_opacities[idx] = new_raw_opac;

        child_rotations[idx * 4 + 0] = rotations[si * 4 + 0];
        child_rotations[idx * 4 + 1] = rotations[si * 4 + 1];
        child_rotations[idx * 4 + 2] = rotations[si * 4 + 2];
        child_rotations[idx * 4 + 3] = rotations[si * 4 + 3];

        child_sh0[idx * 3 + 0] = sh0[si * 3 + 0];
        child_sh0[idx * 3 + 1] = sh0[si * 3 + 1];
        child_sh0[idx * 3 + 2] = sh0[si * 3 + 2];

        for (size_t j = 0; j < sh_rest * 3; ++j) {
            child_shN[idx * sh_rest * 3 + j] = shN[si * sh_rest * 3 + j];
        }
    }

    void launch_lfs_split_inplace(
        const int64_t* split_indices,
        float* means,
        float* log_scales,
        float* raw_opacities,
        const float* rotations,
        const float* sh0,
        const float* shN,
        float* child_means,
        float* child_log_scales,
        float* child_raw_opacities,
        float* child_rotations,
        float* child_sh0,
        float* child_shN,
        size_t K,
        size_t sh_rest,
        void* stream) {

        if (K == 0)
            return;

        constexpr int threads = 256;
        const int blocks = static_cast<int>((K + threads - 1) / threads);
        cudaStream_t s = stream ? static_cast<cudaStream_t>(stream) : nullptr;

        lfs_split_kernel<<<blocks, threads, 0, s>>>(
            split_indices, means, log_scales, raw_opacities,
            rotations, sh0, shN,
            child_means, child_log_scales, child_raw_opacities,
            child_rotations, child_sh0, child_shN,
            K, sh_rest);
    }

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

    __global__ void int_to_int64_kernel(
        const int* __restrict__ src,
        int64_t* __restrict__ dst,
        size_t n) {
        const size_t i = threadIdx.x + blockIdx.x * static_cast<size_t>(blockDim.x);
        if (i < n)
            dst[i] = static_cast<int64_t>(src[i]);
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

        float* d_keys = nullptr;
        cudaMallocAsync(&d_keys, N * sizeof(float), s);

        constexpr int threads = 256;
        const int blocks = static_cast<int>((N + threads - 1) / threads);
        gumbel_key_kernel<<<blocks, threads, 0, s>>>(weights, d_keys, N, seed);

        int* d_indices = nullptr;
        float* d_keys_sorted = nullptr;
        int* d_indices_sorted = nullptr;
        cudaMallocAsync(&d_indices, N * sizeof(int), s);
        cudaMallocAsync(&d_keys_sorted, N * sizeof(float), s);
        cudaMallocAsync(&d_indices_sorted, N * sizeof(int), s);

        thrust::sequence(thrust::cuda::par.on(s), d_indices, d_indices + N);

        const int n_int = static_cast<int>(N);
        size_t temp_bytes = 0;
        cub::DeviceRadixSort::SortPairsDescending(
            nullptr, temp_bytes,
            d_keys, d_keys_sorted,
            d_indices, d_indices_sorted,
            n_int, 0, 32, s);
        char* d_temp = nullptr;
        cudaMallocAsync(&d_temp, temp_bytes, s);
        cub::DeviceRadixSort::SortPairsDescending(
            d_temp, temp_bytes,
            d_keys, d_keys_sorted,
            d_indices, d_indices_sorted,
            n_int, 0, 32, s);

        const int conv_blocks = static_cast<int>((K + threads - 1) / threads);
        int_to_int64_kernel<<<conv_blocks, threads, 0, s>>>(d_indices_sorted, output_indices, K);

        cudaFreeAsync(d_temp, s);
        cudaFreeAsync(d_keys, s);
        cudaFreeAsync(d_indices, s);
        cudaFreeAsync(d_keys_sorted, s);
        cudaFreeAsync(d_indices_sorted, s);
    }

    __global__ void sobel_edge_map_kernel(
        const float* __restrict__ image,
        float* __restrict__ edge_map,
        int C, int H, int W,
        float floor_value) {

        const int x = threadIdx.x + blockIdx.x * blockDim.x;
        const int y = threadIdx.y + blockIdx.y * blockDim.y;
        if (x >= W || y >= H)
            return;

        auto sample = [&](int row, int col) {
            row = max(0, min(row, H - 1));
            col = max(0, min(col, W - 1));
            float lum = 0.0f;
            for (int c = 0; c < C; ++c)
                lum += image[c * H * W + row * W + col];
            return lum / static_cast<float>(C);
        };

        const float tl = sample(y - 1, x - 1);
        const float tc = sample(y - 1, x);
        const float tr = sample(y - 1, x + 1);
        const float ml = sample(y, x - 1);
        const float mr = sample(y, x + 1);
        const float bl = sample(y + 1, x - 1);
        const float bc = sample(y + 1, x);
        const float br = sample(y + 1, x + 1);

        const float gx = -tl + tr - 2.0f * ml + 2.0f * mr - bl + br;
        const float gy = -tl - 2.0f * tc - tr + bl + 2.0f * bc + br;

        const float edge = fminf(sqrtf(gx * gx + gy * gy), 1.0f);
        edge_map[y * W + x] = fmaxf(edge, floor_value);
    }

    void launch_sobel_edge_map(
        const float* image,
        float* edge_map,
        int C, int H, int W,
        float floor_value,
        void* stream) {

        assert(image != nullptr);
        assert(edge_map != nullptr);
        assert(C > 0 && H > 0 && W > 0);

        const dim3 block(16, 16);
        const dim3 grid((W + block.x - 1) / block.x, (H + block.y - 1) / block.y);
        cudaStream_t s = stream ? static_cast<cudaStream_t>(stream) : nullptr;

        sobel_edge_map_kernel<<<grid, block, 0, s>>>(image, edge_map, C, H, W, floor_value);
    }

} // namespace lfs::training::lfs_strategy

/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later
 *
 * Optimized scalar reduction kernels using two-stage grid-stride patterns.
 * - float4 vectorized loads for memory bandwidth
 * - Warp-level reductions via block_reduce_sum/min/max
 * - GPU-aware grid sizing for full SM utilization
 */

#include "core/cuda_error.hpp"
#include "core/tensor/backend/cuda/kernels/tensor_ops.hpp"
#include "core/tensor/backend/cuda/kernels/warp_reduce.cuh"
#include "internal/gpu_config.hpp"
#include <cuda_runtime.h>

namespace lfs::core::tensor_ops {

    // Stage 2: aggregate partial results (reused by all reductions)
    __global__ void reduce_partials_sum(const float* __restrict__ partials, float* __restrict__ result, int n) {
        float sum = 0.0f;
        for (int i = threadIdx.x; i < n; i += blockDim.x) {
            sum += partials[i];
        }
        sum = warp_ops::block_reduce_sum(sum);
        if (threadIdx.x == 0)
            *result = sum;
    }

    // ============================================================================
    // DOT PRODUCT
    // ============================================================================

    __global__ void dot_stage1(const float* __restrict__ a, const float* __restrict__ b,
                               float* __restrict__ partials, size_t n) {
        const size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
        const size_t stride = blockDim.x * gridDim.x;
        float sum = 0.0f;

        for (size_t i = tid * 4; i < n; i += stride * 4) {
            if (i + 3 < n) {
                float4 aa = reinterpret_cast<const float4*>(a)[i / 4];
                float4 bb = reinterpret_cast<const float4*>(b)[i / 4];
                sum += aa.x * bb.x + aa.y * bb.y + aa.z * bb.z + aa.w * bb.w;
            } else {
                for (size_t j = i; j < n; ++j)
                    sum += a[j] * b[j];
            }
        }

        sum = warp_ops::block_reduce_sum(sum);
        if (threadIdx.x == 0)
            partials[blockIdx.x] = sum;
    }

    __global__ void dot_small(const float* __restrict__ a, const float* __restrict__ b,
                              float* __restrict__ result, int n) {
        float sum = 0.0f;
        for (int i = threadIdx.x * 4; i < n; i += blockDim.x * 4) {
            if (i + 3 < n) {
                float4 aa = reinterpret_cast<const float4*>(a)[i / 4];
                float4 bb = reinterpret_cast<const float4*>(b)[i / 4];
                sum += aa.x * bb.x + aa.y * bb.y + aa.z * bb.z + aa.w * bb.w;
            } else {
                for (int j = i; j < n && j < i + 4; ++j)
                    sum += a[j] * b[j];
            }
        }
        sum = warp_ops::block_reduce_sum(sum);
        if (threadIdx.x == 0)
            *result = sum;
    }

    void launch_dot_product(const float* a, const float* b, float* result, size_t n, cudaStream_t stream) {
        if (n == 0) {
            LFS_CUDA_CHECK(cudaMemsetAsync(result, 0, sizeof(float), stream));
            return;
        }

        constexpr int BLOCK = 256;
        if (n < 100000) {
            dot_small<<<1, BLOCK, 0, stream>>>(a, b, result, static_cast<int>(n));
            LFS_CUDA_LAUNCH_CHECK(stream, "tensor.dot.dot_small");
            return;
        }

        const int grid = GPUConfig::get().optimal_grid_size(BLOCK);
        float* partials = nullptr;
        LFS_CUDA_CHECK(cudaMallocAsync(&partials, grid * sizeof(float), stream));
        dot_stage1<<<grid, BLOCK, 0, stream>>>(a, b, partials, n);
        reduce_partials_sum<<<1, BLOCK, 0, stream>>>(partials, result, grid);
        LFS_CUDA_CHECK(cudaFreeAsync(partials, stream));
    }

    // ============================================================================
    // COUNT NONZERO (multi-block + float4, fully device-side)
    // ============================================================================

    // Small-n path: single block, float4 loads when aligned.
    __global__ void count_nonzero_float_small(const float* __restrict__ data,
                                              size_t* __restrict__ result, size_t n) {
        unsigned int count = 0u;
        for (size_t i = threadIdx.x * 4; i < n; i += blockDim.x * 4) {
            if (i + 3 < n) {
                float4 v = reinterpret_cast<const float4*>(data)[i / 4];
                count += (v.x != 0.0f) + (v.y != 0.0f) + (v.z != 0.0f) + (v.w != 0.0f);
            } else {
                for (size_t j = i; j < n; ++j)
                    count += (data[j] != 0.0f);
            }
        }
        count = warp_ops::block_reduce_sum(count);
        if (threadIdx.x == 0)
            *result = static_cast<size_t>(count);
    }

    // Stage1: SM-capped grid-stride, float4, write per-block partial count.
    __global__ void count_nonzero_float_stage1(const float* __restrict__ data,
                                               unsigned int* __restrict__ partials, size_t n) {
        const size_t tid = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
        const size_t stride = static_cast<size_t>(blockDim.x) * gridDim.x;
        unsigned int count = 0u;
        for (size_t i = tid * 4; i < n; i += stride * 4) {
            if (i + 3 < n) {
                float4 v = reinterpret_cast<const float4*>(data)[i / 4];
                count += (v.x != 0.0f) + (v.y != 0.0f) + (v.z != 0.0f) + (v.w != 0.0f);
            } else {
                for (size_t j = i; j < n; ++j)
                    count += (data[j] != 0.0f);
            }
        }
        count = warp_ops::block_reduce_sum(count);
        if (threadIdx.x == 0)
            partials[blockIdx.x] = count;
    }

    __global__ void count_nonzero_bool_small(const unsigned char* __restrict__ data,
                                             size_t* __restrict__ result, size_t n) {
        unsigned int count = 0u;
        for (size_t i = threadIdx.x; i < n; i += blockDim.x)
            count += (data[i] != 0);
        count = warp_ops::block_reduce_sum(count);
        if (threadIdx.x == 0)
            *result = static_cast<size_t>(count);
    }

    __global__ void count_nonzero_bool_stage1(const unsigned char* __restrict__ data,
                                              unsigned int* __restrict__ partials, size_t n) {
        const size_t tid = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
        const size_t stride = static_cast<size_t>(blockDim.x) * gridDim.x;
        unsigned int count = 0u;
        for (size_t i = tid; i < n; i += stride)
            count += (data[i] != 0);
        count = warp_ops::block_reduce_sum(count);
        if (threadIdx.x == 0)
            partials[blockIdx.x] = count;
    }

    __global__ void count_partials_to_size_t(const unsigned int* __restrict__ partials,
                                             size_t* __restrict__ result, int n) {
        unsigned long long sum = 0ull;
        for (int i = threadIdx.x; i < n; i += blockDim.x)
            sum += partials[i];
        sum = warp_ops::block_reduce_sum(sum);
        if (threadIdx.x == 0)
            *result = static_cast<size_t>(sum);
    }

    void launch_count_nonzero_scalar_float(const float* data, size_t* result, size_t n,
                                           cudaStream_t stream) {
        if (n == 0) {
            LFS_CUDA_CHECK(cudaMemsetAsync(result, 0, sizeof(size_t), stream));
            return;
        }
        constexpr int BLOCK = 256;
        // Single-block is enough under ~100k; multi-block for large masks.
        if (n < 100000) {
            count_nonzero_float_small<<<1, BLOCK, 0, stream>>>(data, result, n);
            LFS_CUDA_LAUNCH_CHECK(stream, "tensor.dot.count_nonzero_float_small");
            return;
        }
        const int grid = GPUConfig::get().optimal_grid_size(BLOCK);
        unsigned int* partials = nullptr;
        LFS_CUDA_CHECK(cudaMallocAsync(&partials, static_cast<size_t>(grid) * sizeof(unsigned int), stream));
        count_nonzero_float_stage1<<<grid, BLOCK, 0, stream>>>(data, partials, n);
        LFS_CUDA_LAUNCH_CHECK(stream, "tensor.dot.count_nonzero_float_stage1");
        count_partials_to_size_t<<<1, BLOCK, 0, stream>>>(partials, result, grid);
        LFS_CUDA_LAUNCH_CHECK(stream, "tensor.dot.count_nonzero_float_stage2");
        LFS_CUDA_CHECK(cudaFreeAsync(partials, stream));
    }

    void launch_count_nonzero_scalar_bool(const unsigned char* data, size_t* result, size_t n,
                                          cudaStream_t stream) {
        if (n == 0) {
            LFS_CUDA_CHECK(cudaMemsetAsync(result, 0, sizeof(size_t), stream));
            return;
        }
        constexpr int BLOCK = 256;
        if (n < 100000) {
            count_nonzero_bool_small<<<1, BLOCK, 0, stream>>>(data, result, n);
            LFS_CUDA_LAUNCH_CHECK(stream, "tensor.dot.count_nonzero_bool_small");
            return;
        }
        const int grid = GPUConfig::get().optimal_grid_size(BLOCK);
        unsigned int* partials = nullptr;
        LFS_CUDA_CHECK(cudaMallocAsync(&partials, static_cast<size_t>(grid) * sizeof(unsigned int), stream));
        count_nonzero_bool_stage1<<<grid, BLOCK, 0, stream>>>(data, partials, n);
        LFS_CUDA_LAUNCH_CHECK(stream, "tensor.dot.count_nonzero_bool_stage1");
        count_partials_to_size_t<<<1, BLOCK, 0, stream>>>(partials, result, grid);
        LFS_CUDA_LAUNCH_CHECK(stream, "tensor.dot.count_nonzero_bool_stage2");
        LFS_CUDA_CHECK(cudaFreeAsync(partials, stream));
    }

} // namespace lfs::core::tensor_ops

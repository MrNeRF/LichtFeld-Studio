/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later
 *
 * Optimized scalar reduction kernels using two-stage grid-stride patterns.
 * - float4 vectorized loads for memory bandwidth
 * - Warp-level reductions via block_reduce_sum/min/max
 * - GPU-aware grid sizing for full SM utilization
 */

#include "core/cuda_error.hpp"
#include "internal/gpu_config.hpp"
#include "internal/tensor_ops.hpp"
#include "internal/warp_reduce.cuh"
#include <cfloat>
#include <cuda_runtime.h>

namespace lfs::core::tensor_ops {

    // Functors for templated reductions
    struct identity_op {
        __device__ float operator()(float x) const { return x; }
    };

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
    // UNARY REDUCTIONS (sum)
    // ============================================================================

    template <typename Op>
    __global__ void unary_stage1(const float* __restrict__ data, float* __restrict__ partials, size_t n, Op op) {
        const size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
        const size_t stride = blockDim.x * gridDim.x;
        float sum = 0.0f;

        for (size_t i = tid * 4; i < n; i += stride * 4) {
            if (i + 3 < n) {
                float4 v = reinterpret_cast<const float4*>(data)[i / 4];
                sum += op(v.x) + op(v.y) + op(v.z) + op(v.w);
            } else {
                for (size_t j = i; j < n; ++j)
                    sum += op(data[j]);
            }
        }

        sum = warp_ops::block_reduce_sum(sum);
        if (threadIdx.x == 0)
            partials[blockIdx.x] = sum;
    }

    template <typename Op>
    __global__ void unary_small(const float* __restrict__ data, float* __restrict__ result, int n, Op op) {
        float sum = 0.0f;
        for (int i = threadIdx.x * 4; i < n; i += blockDim.x * 4) {
            if (i + 3 < n) {
                float4 v = reinterpret_cast<const float4*>(data)[i / 4];
                sum += op(v.x) + op(v.y) + op(v.z) + op(v.w);
            } else {
                for (int j = i; j < n && j < i + 4; ++j)
                    sum += op(data[j]);
            }
        }
        sum = warp_ops::block_reduce_sum(sum);
        if (threadIdx.x == 0)
            *result = sum;
    }

    void launch_sum_scalar(const float* data, float* result, size_t n, cudaStream_t stream) {
        if (n == 0) {
            LFS_CUDA_CHECK(cudaMemsetAsync(result, 0, sizeof(float), stream));
            return;
        }

        constexpr int BLOCK = 256;
        if (n < 100000) {
            unary_small<<<1, BLOCK, 0, stream>>>(data, result, static_cast<int>(n), identity_op{});
            LFS_CUDA_LAUNCH_CHECK(stream, "tensor.dot.sum_scalar");
            return;
        }

        const int grid = GPUConfig::get().optimal_grid_size(BLOCK);
        float* partials = nullptr;
        LFS_CUDA_CHECK(cudaMallocAsync(&partials, grid * sizeof(float), stream));
        unary_stage1<<<grid, BLOCK, 0, stream>>>(data, partials, n, identity_op{});
        reduce_partials_sum<<<1, BLOCK, 0, stream>>>(partials, result, grid);
        LFS_CUDA_CHECK(cudaFreeAsync(partials, stream));
    }

} // namespace lfs::core::tensor_ops

/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later
 *
 * Optimized scalar reduction kernels using two-stage grid-stride patterns.
 * - float4 vectorized loads for memory bandwidth
 * - Warp-level reductions via block_reduce_sum/min/max
 * - GPU-aware grid sizing for full SM utilization
 */

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
    struct abs_op {
        __device__ float operator()(float x) const { return fabsf(x); }
    };
    struct square_op {
        __device__ float operator()(float x) const { return x * x; }
    };
    struct max_op {
        __device__ float operator()(float a, float b) const { return fmaxf(a, b); }
    };
    struct min_op {
        __device__ float operator()(float a, float b) const { return fminf(a, b); }
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

    __global__ void reduce_partials_max(const float* __restrict__ partials, float* __restrict__ result, int n) {
        float val = -FLT_MAX;
        for (int i = threadIdx.x; i < n; i += blockDim.x) {
            val = fmaxf(val, partials[i]);
        }
        val = warp_ops::block_reduce_max(val);
        if (threadIdx.x == 0)
            *result = val;
    }

    __global__ void reduce_partials_min(const float* __restrict__ partials, float* __restrict__ result, int n) {
        float val = FLT_MAX;
        for (int i = threadIdx.x; i < n; i += blockDim.x) {
            val = fminf(val, partials[i]);
        }
        val = warp_ops::block_reduce_min(val);
        if (threadIdx.x == 0)
            *result = val;
    }

    // Helper kernels for in-place operations
    __global__ void sqrt_inplace(float* r) { *r = sqrtf(*r); }
    __global__ void div_inplace(float* r, float inv_n) { *r *= inv_n; }

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
            cudaMemsetAsync(result, 0, sizeof(float), stream);
            return;
        }

        constexpr int BLOCK = 256;
        if (n < 100000) {
            dot_small<<<1, BLOCK, 0, stream>>>(a, b, result, static_cast<int>(n));
            return;
        }

        const int grid = GPUConfig::get().optimal_grid_size(BLOCK);
        float* partials = nullptr;
        cudaMallocAsync(&partials, grid * sizeof(float), stream);
        dot_stage1<<<grid, BLOCK, 0, stream>>>(a, b, partials, n);
        reduce_partials_sum<<<1, BLOCK, 0, stream>>>(partials, result, grid);
        cudaFreeAsync(partials, stream);
    }

    // ============================================================================
    // UNARY REDUCTIONS (sum, l1_norm, l2_norm)
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
            cudaMemsetAsync(result, 0, sizeof(float), stream);
            return;
        }

        constexpr int BLOCK = 256;
        if (n < 100000) {
            unary_small<<<1, BLOCK, 0, stream>>>(data, result, static_cast<int>(n), identity_op{});
            return;
        }

        const int grid = GPUConfig::get().optimal_grid_size(BLOCK);
        float* partials = nullptr;
        cudaMallocAsync(&partials, grid * sizeof(float), stream);
        unary_stage1<<<grid, BLOCK, 0, stream>>>(data, partials, n, identity_op{});
        reduce_partials_sum<<<1, BLOCK, 0, stream>>>(partials, result, grid);
        cudaFreeAsync(partials, stream);
    }

    void launch_mean_scalar(const float* data, float* result, size_t n, cudaStream_t stream) {
        if (n == 0) {
            cudaMemsetAsync(result, 0, sizeof(float), stream);
            return;
        }
        launch_sum_scalar(data, result, n, stream);
        div_inplace<<<1, 1, 0, stream>>>(result, 1.0f / static_cast<float>(n));
    }

    void launch_l1_norm(const float* data, float* result, size_t n, cudaStream_t stream) {
        if (n == 0) {
            cudaMemsetAsync(result, 0, sizeof(float), stream);
            return;
        }

        constexpr int BLOCK = 256;
        if (n < 100000) {
            unary_small<<<1, BLOCK, 0, stream>>>(data, result, static_cast<int>(n), abs_op{});
            return;
        }

        const int grid = GPUConfig::get().optimal_grid_size(BLOCK);
        float* partials = nullptr;
        cudaMallocAsync(&partials, grid * sizeof(float), stream);
        unary_stage1<<<grid, BLOCK, 0, stream>>>(data, partials, n, abs_op{});
        reduce_partials_sum<<<1, BLOCK, 0, stream>>>(partials, result, grid);
        cudaFreeAsync(partials, stream);
    }

    void launch_l2_norm(const float* data, float* result, size_t n, cudaStream_t stream) {
        if (n == 0) {
            cudaMemsetAsync(result, 0, sizeof(float), stream);
            return;
        }

        constexpr int BLOCK = 256;
        if (n < 100000) {
            unary_small<<<1, BLOCK, 0, stream>>>(data, result, static_cast<int>(n), square_op{});
            sqrt_inplace<<<1, 1, 0, stream>>>(result);
            return;
        }

        const int grid = GPUConfig::get().optimal_grid_size(BLOCK);
        float* partials = nullptr;
        cudaMallocAsync(&partials, grid * sizeof(float), stream);
        unary_stage1<<<grid, BLOCK, 0, stream>>>(data, partials, n, square_op{});
        reduce_partials_sum<<<1, BLOCK, 0, stream>>>(partials, result, grid);
        sqrt_inplace<<<1, 1, 0, stream>>>(result);
        cudaFreeAsync(partials, stream);
    }

    // ============================================================================
    // MIN/MAX REDUCTIONS
    // ============================================================================

    template <typename Op>
    __global__ void minmax_stage1(const float* __restrict__ data, float* __restrict__ partials,
                                  size_t n, float init, Op op) {
        const size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
        const size_t stride = blockDim.x * gridDim.x;
        float val = init;

        for (size_t i = tid * 4; i < n; i += stride * 4) {
            if (i + 3 < n) {
                float4 v = reinterpret_cast<const float4*>(data)[i / 4];
                val = op(val, op(op(v.x, v.y), op(v.z, v.w)));
            } else {
                for (size_t j = i; j < n; ++j)
                    val = op(val, data[j]);
            }
        }

        if (init == -FLT_MAX)
            val = warp_ops::block_reduce_max(val);
        else
            val = warp_ops::block_reduce_min(val);

        if (threadIdx.x == 0)
            partials[blockIdx.x] = val;
    }

    template <typename Op>
    __global__ void minmax_small(const float* __restrict__ data, float* __restrict__ result,
                                 int n, float init, Op op) {
        float val = init;
        for (int i = threadIdx.x * 4; i < n; i += blockDim.x * 4) {
            if (i + 3 < n) {
                float4 v = reinterpret_cast<const float4*>(data)[i / 4];
                val = op(val, op(op(v.x, v.y), op(v.z, v.w)));
            } else {
                for (int j = i; j < n && j < i + 4; ++j)
                    val = op(val, data[j]);
            }
        }

        if (init == -FLT_MAX)
            val = warp_ops::block_reduce_max(val);
        else
            val = warp_ops::block_reduce_min(val);

        if (threadIdx.x == 0)
            *result = val;
    }

    void launch_max_scalar(const float* data, float* result, size_t n, cudaStream_t stream) {
        if (n == 0) {
            launch_fill(result, 1, -FLT_MAX, stream); // Empty array sentinel
            return;
        }

        constexpr int BLOCK = 256;
        if (n < 100000) {
            minmax_small<<<1, BLOCK, 0, stream>>>(data, result, static_cast<int>(n), -FLT_MAX, max_op{});
            return;
        }

        const int grid = GPUConfig::get().optimal_grid_size(BLOCK);
        float* partials = nullptr;
        cudaMallocAsync(&partials, grid * sizeof(float), stream);
        minmax_stage1<<<grid, BLOCK, 0, stream>>>(data, partials, n, -FLT_MAX, max_op{});
        reduce_partials_max<<<1, BLOCK, 0, stream>>>(partials, result, grid);
        cudaFreeAsync(partials, stream);
    }

    void launch_min_scalar(const float* data, float* result, size_t n, cudaStream_t stream) {
        if (n == 0) {
            launch_fill(result, 1, FLT_MAX, stream); // Empty array sentinel
            return;
        }

        constexpr int BLOCK = 256;
        if (n < 100000) {
            minmax_small<<<1, BLOCK, 0, stream>>>(data, result, static_cast<int>(n), FLT_MAX, min_op{});
            return;
        }

        const int grid = GPUConfig::get().optimal_grid_size(BLOCK);
        float* partials = nullptr;
        cudaMallocAsync(&partials, grid * sizeof(float), stream);
        minmax_stage1<<<grid, BLOCK, 0, stream>>>(data, partials, n, FLT_MAX, min_op{});
        reduce_partials_min<<<1, BLOCK, 0, stream>>>(partials, result, grid);
        cudaFreeAsync(partials, stream);
    }

    // ============================================================================
    // COUNT NONZERO
    // ============================================================================

    __global__ void count_nonzero_float(const float* data, size_t* result, int n) {
        float count = 0.0f;
        for (int i = threadIdx.x * 2; i < n; i += blockDim.x * 2) {
            if (i + 1 < n) {
                float2 v = *(float2*)&data[i];
                count += (v.x != 0.0f) + (v.y != 0.0f);
            } else if (i < n) {
                count += (data[i] != 0.0f);
            }
        }
        count = warp_ops::block_reduce_sum(count);
        if (threadIdx.x == 0)
            *result = static_cast<size_t>(count);
    }

    __global__ void count_nonzero_bool(const unsigned char* data, size_t* result, int n) {
        float count = 0.0f;
        for (int i = threadIdx.x; i < n; i += blockDim.x) {
            count += (data[i] != 0);
        }
        count = warp_ops::block_reduce_sum(count);
        if (threadIdx.x == 0)
            *result = static_cast<size_t>(count);
    }

    void launch_count_nonzero_scalar_float(const float* data, size_t* result, size_t n, cudaStream_t stream) {
        count_nonzero_float<<<1, 256, 0, stream>>>(data, result, static_cast<int>(n));
    }

    void launch_count_nonzero_scalar_bool(const unsigned char* data, size_t* result, size_t n, cudaStream_t stream) {
        count_nonzero_bool<<<1, 256, 0, stream>>>(data, result, static_cast<int>(n));
    }

} // namespace lfs::core::tensor_ops

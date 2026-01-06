/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "core/tensor_debug.hpp"
#include <cmath>
#include <cuda_runtime.h>
#include <limits>

namespace lfs::core::debug {

    namespace {
        constexpr int BLOCK_SIZE = 256;
        constexpr int WARP_SIZE = 32;

        // Warp-level reduction for min
        __device__ float warp_reduce_min(float val) {
            for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
                val = fminf(val, __shfl_down_sync(0xffffffff, val, offset));
            }
            return val;
        }

        // Warp-level reduction for max
        __device__ float warp_reduce_max(float val) {
            for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
                val = fmaxf(val, __shfl_down_sync(0xffffffff, val, offset));
            }
            return val;
        }

        // Warp-level reduction for sum
        __device__ float warp_reduce_sum(float val) {
            for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
                val += __shfl_down_sync(0xffffffff, val, offset);
            }
            return val;
        }

        // Warp-level reduction for count
        __device__ unsigned int warp_reduce_sum_uint(unsigned int val) {
            for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
                val += __shfl_down_sync(0xffffffff, val, offset);
            }
            return val;
        }

        struct ValidationResult {
            unsigned int nan_count;
            unsigned int inf_count;
            float min_val;
            float max_val;
            float sum;
        };

        __global__ void validate_tensor_kernel(const float* __restrict__ data,
                                               size_t n,
                                               ValidationResult* __restrict__ result) {
            __shared__ unsigned int s_nan_count[BLOCK_SIZE / WARP_SIZE];
            __shared__ unsigned int s_inf_count[BLOCK_SIZE / WARP_SIZE];
            __shared__ float s_min[BLOCK_SIZE / WARP_SIZE];
            __shared__ float s_max[BLOCK_SIZE / WARP_SIZE];
            __shared__ float s_sum[BLOCK_SIZE / WARP_SIZE];

            const int tid = threadIdx.x;
            const int warp_id = tid / WARP_SIZE;
            const int lane_id = tid % WARP_SIZE;
            const size_t idx = blockIdx.x * blockDim.x + tid;
            const size_t stride = blockDim.x * gridDim.x;

            unsigned int local_nan = 0;
            unsigned int local_inf = 0;
            float local_min = CUDA_INFINITY;
            float local_max = -CUDA_INFINITY;
            float local_sum = 0.0f;

            for (size_t i = idx; i < n; i += stride) {
                const float val = data[i];
                if (isnan(val)) {
                    ++local_nan;
                } else if (isinf(val)) {
                    ++local_inf;
                } else {
                    local_min = fminf(local_min, val);
                    local_max = fmaxf(local_max, val);
                    local_sum += val;
                }
            }

            // Warp-level reduction
            local_nan = warp_reduce_sum_uint(local_nan);
            local_inf = warp_reduce_sum_uint(local_inf);
            local_min = warp_reduce_min(local_min);
            local_max = warp_reduce_max(local_max);
            local_sum = warp_reduce_sum(local_sum);

            // Store warp results to shared memory
            if (lane_id == 0) {
                s_nan_count[warp_id] = local_nan;
                s_inf_count[warp_id] = local_inf;
                s_min[warp_id] = local_min;
                s_max[warp_id] = local_max;
                s_sum[warp_id] = local_sum;
            }
            __syncthreads();

            // Final reduction by first warp
            if (warp_id == 0) {
                const int num_warps = BLOCK_SIZE / WARP_SIZE;
                local_nan = lane_id < num_warps ? s_nan_count[lane_id] : 0;
                local_inf = lane_id < num_warps ? s_inf_count[lane_id] : 0;
                local_min = lane_id < num_warps ? s_min[lane_id] : CUDA_INFINITY;
                local_max = lane_id < num_warps ? s_max[lane_id] : -CUDA_INFINITY;
                local_sum = lane_id < num_warps ? s_sum[lane_id] : 0.0f;

                local_nan = warp_reduce_sum_uint(local_nan);
                local_inf = warp_reduce_sum_uint(local_inf);
                local_min = warp_reduce_min(local_min);
                local_max = warp_reduce_max(local_max);
                local_sum = warp_reduce_sum(local_sum);

                if (lane_id == 0) {
                    atomicAdd(&result->nan_count, local_nan);
                    atomicAdd(&result->inf_count, local_inf);
                    // For min/max, we need atomicMin/Max which don't exist for float
                    // Use atomic CAS instead
                    float old = result->min_val;
                    while (local_min < old) {
                        old = atomicCAS((unsigned int*)&result->min_val,
                                        __float_as_uint(old),
                                        __float_as_uint(local_min));
                        old = __uint_as_float(__float_as_uint(old));
                    }
                    old = result->max_val;
                    while (local_max > old) {
                        old = atomicCAS((unsigned int*)&result->max_val,
                                        __float_as_uint(old),
                                        __float_as_uint(local_max));
                        old = __uint_as_float(__float_as_uint(old));
                    }
                    atomicAdd(&result->sum, local_sum);
                }
            }
        }
    } // namespace

    TensorValidation validate_tensor_gpu_impl(const float* data, size_t n) {
        TensorValidation result;

        if (n == 0) {
            return result;
        }

        // Allocate result on device
        ValidationResult* d_result;
        cudaMalloc(&d_result, sizeof(ValidationResult));

        ValidationResult init = {0, 0, CUDA_INFINITY, -CUDA_INFINITY, 0.0f};
        cudaMemcpy(d_result, &init, sizeof(ValidationResult), cudaMemcpyHostToDevice);

        // Launch kernel
        const int num_blocks = std::min(256, static_cast<int>((n + BLOCK_SIZE - 1) / BLOCK_SIZE));
        validate_tensor_kernel<<<num_blocks, BLOCK_SIZE>>>(data, n, d_result);

        // Copy result back
        ValidationResult h_result;
        cudaMemcpy(&h_result, d_result, sizeof(ValidationResult), cudaMemcpyDeviceToHost);
        cudaFree(d_result);

        // Convert to TensorValidation
        result.nan_count = h_result.nan_count;
        result.inf_count = h_result.inf_count;
        result.has_nan = h_result.nan_count > 0;
        result.has_inf = h_result.inf_count > 0;
        result.min_val = h_result.min_val;
        result.max_val = h_result.max_val;

        const size_t valid_count = n - result.nan_count - result.inf_count;
        result.mean_val = valid_count > 0 ? h_result.sum / valid_count : 0.0f;

        return result;
    }

} // namespace lfs::core::debug

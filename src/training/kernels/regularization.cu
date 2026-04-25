/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "lfs/core/warp_reduce.cuh"
#include "lfs/kernels/regularization.cuh"

namespace lfs::training::kernels {

    // =============================================================================
    // SCALE REGULARIZATION (exp-based)
    // =============================================================================

    /**
     * Fused kernel: computes exp(x), accumulates gradient, and sums for loss
     * OPTIMIZED: Uses warp-level reductions (5-10× faster than CUB!)
     */
    __global__ void fused_scale_regularization_kernel(
        const float* __restrict__ params,
        float* __restrict__ param_grads,
        float* __restrict__ partial_sums,
        size_t n,
        float grad_scale) { // weight / n

        float local_sum = 0.0f;

        // Grid-stride loop for coalesced memory access
        for (size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
             idx < n;
             idx += blockDim.x * gridDim.x) {

            const float x = params[idx];
            const float exp_x = expf(x); // exp(scaling_raw)

            // Accumulate for loss
            local_sum += exp_x;

            if (param_grads != nullptr) {
                atomicAdd(&param_grads[idx], grad_scale * exp_x);
            }
        }

        // Block-level warp reduction (tiny-cuda-nn style - much faster!)
        local_sum = lfs::core::warp_ops::block_reduce_sum(local_sum);

        if (threadIdx.x == 0) {
            partial_sums[blockIdx.x] = local_sum;
        }
    }

    /**
     * Final reduction kernel
     * OPTIMIZED: Uses warp-level reductions
     */
    __global__ void final_scale_reduce_kernel(
        const float* __restrict__ partial_sums,
        float* __restrict__ result,
        int num_blocks,
        float weight,
        size_t n) {

        // Grid-stride loop to handle more than blockDim.x partial sums
        float sum = 0.0f;
        for (int i = threadIdx.x; i < num_blocks; i += blockDim.x) {
            sum += partial_sums[i];
        }

        // Block-level warp reduction
        sum = lfs::core::warp_ops::block_reduce_sum(sum);

        if (threadIdx.x == 0) {
            result[0] = weight * (sum / static_cast<float>(n)); // loss = weight * mean
        }
    }

    void launch_fused_scale_regularization(
        const float* params,
        float* param_grads,
        float* loss_out,
        float* temp_buffer,
        size_t n,
        float weight,
        cudaStream_t stream) {

        if (n == 0 || weight == 0.0f) {
            return;
        }

        const int block_size = 256;
        const int num_blocks = std::min((n + block_size - 1) / block_size, size_t(1024));

        float grad_scale = weight / static_cast<float>(n);

        // Launch fused kernel
        fused_scale_regularization_kernel<<<num_blocks, block_size, 0, stream>>>(
            params, param_grads, temp_buffer, n, grad_scale);

        // Launch final reduction
        final_scale_reduce_kernel<<<1, block_size, 0, stream>>>(
            temp_buffer, loss_out, num_blocks, weight, n);
    }

    // =============================================================================
    // OPACITY REGULARIZATION (sigmoid-based)
    // =============================================================================

    /**
     * Fused kernel: computes sigmoid(x), accumulates gradient, and sums for loss
     * OPTIMIZED: Uses warp-level reductions
     */
    __global__ void fused_opacity_regularization_kernel(
        const float* __restrict__ params,
        float* __restrict__ param_grads,
        float* __restrict__ partial_sums,
        size_t n,
        float grad_scale) { // weight / n

        float local_sum = 0.0f;

        // Grid-stride loop for coalesced memory access
        for (size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
             idx < n;
             idx += blockDim.x * gridDim.x) {

            const float x = params[idx];
            const float sigmoid_x = 1.0f / (1.0f + expf(-x)); // sigmoid(opacity_raw)

            // Accumulate for loss
            local_sum += sigmoid_x;

            if (param_grads != nullptr) {
                atomicAdd(&param_grads[idx], grad_scale * sigmoid_x * (1.0f - sigmoid_x));
            }
        }

        // Block-level warp reduction (tiny-cuda-nn style - much faster!)
        local_sum = lfs::core::warp_ops::block_reduce_sum(local_sum);

        if (threadIdx.x == 0) {
            partial_sums[blockIdx.x] = local_sum;
        }
    }

    /**
     * Final reduction kernel
     * OPTIMIZED: Uses warp-level reductions
     */
    __global__ void final_opacity_reduce_kernel(
        const float* __restrict__ partial_sums,
        float* __restrict__ result,
        int num_blocks,
        float weight,
        size_t n) {

        // Grid-stride loop to handle more than blockDim.x partial sums
        float sum = 0.0f;
        for (int i = threadIdx.x; i < num_blocks; i += blockDim.x) {
            sum += partial_sums[i];
        }

        // Block-level warp reduction
        sum = lfs::core::warp_ops::block_reduce_sum(sum);

        if (threadIdx.x == 0) {
            result[0] = weight * (sum / static_cast<float>(n)); // loss = weight * mean
        }
    }

    void launch_fused_opacity_regularization(
        const float* params,
        float* param_grads,
        float* loss_out,
        float* temp_buffer,
        size_t n,
        float weight,
        cudaStream_t stream) {

        if (n == 0 || weight == 0.0f) {
            return;
        }

        const int block_size = 256;
        const int num_blocks = std::min((n + block_size - 1) / block_size, size_t(1024));

        float grad_scale = weight / static_cast<float>(n);

        // Launch fused kernel
        fused_opacity_regularization_kernel<<<num_blocks, block_size, 0, stream>>>(
            params, param_grads, temp_buffer, n, grad_scale);

        // Launch final reduction
        final_opacity_reduce_kernel<<<1, block_size, 0, stream>>>(
            temp_buffer, loss_out, num_blocks, weight, n);
    }

} // namespace lfs::training::kernels

/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "kernels/regularization.cuh"
#include <cub/cub.cuh>
#include <cuda_runtime.h>

namespace gs {
    namespace regularization {

        // =============================================================================
        // EXP REGULARIZATION (for scaling)
        // =============================================================================

        /**
         * Fused kernel: computes exp(x), accumulates gradient, and sums for loss
         * This is much more efficient than separate exp() and gradient operations
         */
        __global__ void exp_l1_regularization_kernel(
            const float* __restrict__ params,      // Input: raw parameters
            float* __restrict__ param_grads,       // Output: accumulated gradients
            float* __restrict__ block_sums,        // Output: partial sums
            const int n,
            const float grad_scale) {               // weight / n

            // CUB block reduction
            typedef cub::BlockReduce<float, 256> BlockReduce;
            __shared__ typename BlockReduce::TempStorage temp_storage;

            const int tid = blockIdx.x * blockDim.x + threadIdx.x;
            const int stride = gridDim.x * blockDim.x;

            float local_sum = 0.0f;

            // Grid-stride loop
            for (int i = tid; i < n; i += stride) {
                const float x = params[i];
                const float exp_x = expf(x);  // exp(scaling_raw)

                // Accumulate to loss sum
                local_sum += exp_x;

                // Accumulate gradient: ∂L/∂scaling_raw = grad_scale * exp(scaling_raw)
                atomicAdd(&param_grads[i], grad_scale * exp_x);
            }

            // Block reduction
            float block_sum = BlockReduce(temp_storage).Sum(local_sum);

            // First thread writes block sum
            if (threadIdx.x == 0) {
                block_sums[blockIdx.x] = block_sum;
            }
        }

        /**
         * Reduce block sums to final scalar
         */
        __global__ void reduce_block_sums_kernel(
            const float* __restrict__ block_sums,
            float* __restrict__ output,
            const int num_blocks,
            const float weight,
            const int n) {

            typedef cub::BlockReduce<float, 256> BlockReduce;
            __shared__ typename BlockReduce::TempStorage temp_storage;

            const int tid = threadIdx.x;
            float local_sum = 0.0f;

            // Each thread processes multiple blocks if necessary
            for (int i = tid; i < num_blocks; i += blockDim.x) {
                local_sum += block_sums[i];
            }

            // Block reduction
            float total_sum = BlockReduce(temp_storage).Sum(local_sum);

            // First thread writes final result
            if (tid == 0) {
                output[0] = weight * (total_sum / static_cast<float>(n)); // loss = weight * mean
            }
        }

        float compute_exp_l1_regularization_with_grad_cuda(
            torch::Tensor& scaling_raw,
            float weight) {

            // Skip if weight is zero
            if (weight == 0.0f) {
                return 0.0f;
            }

            TORCH_CHECK(scaling_raw.is_cuda(), "scaling_raw must be a CUDA tensor");
            TORCH_CHECK(scaling_raw.dtype() == torch::kFloat32, "scaling_raw must be float32");
            TORCH_CHECK(scaling_raw.requires_grad(), "scaling_raw must require gradients");
            TORCH_CHECK(scaling_raw.dim() == 2, "scaling_raw must be 2D [N, 3], got ", scaling_raw.dim(), "D");
            TORCH_CHECK(scaling_raw.size(1) == 3, "scaling_raw must have 3 columns, got ", scaling_raw.size(1));

            const int n = scaling_raw.numel();  // Total elements N * 3
            if (n == 0) {
                return 0.0f;
            }

            // Ensure gradient tensor exists
            if (!scaling_raw.grad().defined()) {
                scaling_raw.mutable_grad() = torch::zeros_like(scaling_raw);
            }

            const float* params_ptr = scaling_raw.data_ptr<float>();
            float* param_grads_ptr = scaling_raw.grad().data_ptr<float>();

            // Configure kernel launch
            const int threads = 256;
            const int blocks = std::min((n + threads - 1) / threads, 1024);

            // Allocate temporary storage for block sums
            auto block_sums = torch::empty({blocks}, torch::TensorOptions()
                .dtype(torch::kFloat32)
                .device(scaling_raw.device()));
            float* block_sums_ptr = block_sums.data_ptr<float>();

            // Allocate output for loss value
            auto loss_tensor = torch::empty({1}, torch::TensorOptions()
                .dtype(torch::kFloat32)
                .device(scaling_raw.device()));
            float* loss_ptr = loss_tensor.data_ptr<float>();

            // Launch first kernel: compute exp, accumulate gradients, and partial sums
            const float grad_scale = weight / static_cast<float>(n);
            exp_l1_regularization_kernel<<<blocks, threads>>>(
                params_ptr,
                param_grads_ptr,
                block_sums_ptr,
                n,
                grad_scale);

            // Check for errors
            cudaError_t err = cudaGetLastError();
            TORCH_CHECK(err == cudaSuccess, "exp_l1_regularization_kernel launch failed: ",
                       cudaGetErrorString(err));

            // Launch second kernel: reduce block sums to scalar
            reduce_block_sums_kernel<<<1, threads>>>(
                block_sums_ptr,
                loss_ptr,
                blocks,
                weight,
                n);

            err = cudaGetLastError();
            TORCH_CHECK(err == cudaSuccess, "reduce_block_sums_kernel launch failed: ",
                       cudaGetErrorString(err));

            // Copy result back to CPU
            float loss_value;
            cudaMemcpy(&loss_value, loss_ptr, sizeof(float), cudaMemcpyDeviceToHost);

            return loss_value;
        }

        // =============================================================================
        // SIGMOID REGULARIZATION (for opacity)
        // =============================================================================

        /**
         * Fused kernel: computes sigmoid(x), accumulates gradient, and sums for loss
         */
        __global__ void sigmoid_l1_regularization_kernel(
            const float* __restrict__ params,      // Input: raw parameters [N, 1]
            float* __restrict__ param_grads,       // Output: accumulated gradients [N, 1]
            float* __restrict__ block_sums,        // Output: partial sums
            const int n,
            const float grad_scale) {               // weight / n

            // CUB block reduction
            typedef cub::BlockReduce<float, 256> BlockReduce;
            __shared__ typename BlockReduce::TempStorage temp_storage;

            const int tid = blockIdx.x * blockDim.x + threadIdx.x;
            const int stride = gridDim.x * blockDim.x;

            float local_sum = 0.0f;

            // Grid-stride loop
            for (int i = tid; i < n; i += stride) {
                const float x = params[i];
                const float sigmoid_x = 1.0f / (1.0f + expf(-x));  // sigmoid(opacity_raw)

                // Accumulate to loss sum
                local_sum += sigmoid_x;

                // Compute sigmoid derivative: σ(x) * (1 - σ(x))
                const float sigmoid_deriv = sigmoid_x * (1.0f - sigmoid_x);

                // Accumulate gradient: ∂L/∂opacity_raw = grad_scale * sigmoid'(x)
                atomicAdd(&param_grads[i], grad_scale * sigmoid_deriv);
            }

            // Block reduction
            float block_sum = BlockReduce(temp_storage).Sum(local_sum);

            // First thread writes block sum
            if (threadIdx.x == 0) {
                block_sums[blockIdx.x] = block_sum;
            }
        }

        float compute_sigmoid_l1_regularization_with_grad_cuda(
            torch::Tensor& opacity_raw,
            float weight) {

            // Skip if weight is zero
            if (weight == 0.0f) {
                return 0.0f;
            }

            TORCH_CHECK(opacity_raw.is_cuda(), "opacity_raw must be a CUDA tensor");
            TORCH_CHECK(opacity_raw.dtype() == torch::kFloat32, "opacity_raw must be float32");
            TORCH_CHECK(opacity_raw.requires_grad(), "opacity_raw must require gradients");
            TORCH_CHECK(opacity_raw.dim() == 2, "opacity_raw must be 2D [N, 1], got ", opacity_raw.dim(), "D");
            TORCH_CHECK(opacity_raw.size(1) == 1, "opacity_raw must have shape [N, 1], got [", opacity_raw.size(0), ", ", opacity_raw.size(1), "]");

            const int n = opacity_raw.numel();
            if (n == 0) {
                return 0.0f;
            }

            // Ensure gradient tensor exists
            if (!opacity_raw.grad().defined()) {
                opacity_raw.mutable_grad() = torch::zeros_like(opacity_raw);
            }

            const float* params_ptr = opacity_raw.data_ptr<float>();
            float* param_grads_ptr = opacity_raw.grad().data_ptr<float>();

            // Configure kernel launch
            const int threads = 256;
            const int blocks = std::min((n + threads - 1) / threads, 1024);

            // Allocate temporary storage for block sums
            auto block_sums = torch::empty({blocks}, torch::TensorOptions()
                .dtype(torch::kFloat32)
                .device(opacity_raw.device()));
            float* block_sums_ptr = block_sums.data_ptr<float>();

            // Allocate output for loss value
            auto loss_tensor = torch::empty({1}, torch::TensorOptions()
                .dtype(torch::kFloat32)
                .device(opacity_raw.device()));
            float* loss_ptr = loss_tensor.data_ptr<float>();

            // Launch first kernel: compute sigmoid, accumulate gradients, and partial sums
            const float grad_scale = weight / static_cast<float>(n);
            sigmoid_l1_regularization_kernel<<<blocks, threads>>>(
                params_ptr,
                param_grads_ptr,
                block_sums_ptr,
                n,
                grad_scale);

            // Check for errors
            cudaError_t err = cudaGetLastError();
            TORCH_CHECK(err == cudaSuccess, "sigmoid_l1_regularization_kernel launch failed: ",
                       cudaGetErrorString(err));

            // Launch second kernel: reduce block sums to scalar
            reduce_block_sums_kernel<<<1, threads>>>(
                block_sums_ptr,
                loss_ptr,
                blocks,
                weight,
                n);

            err = cudaGetLastError();
            TORCH_CHECK(err == cudaSuccess, "reduce_block_sums_kernel launch failed: ",
                       cudaGetErrorString(err));

            // Copy result back to CPU
            float loss_value;
            cudaMemcpy(&loss_value, loss_ptr, sizeof(float), cudaMemcpyDeviceToHost);

            return loss_value;
        }

    } // namespace regularization
} // namespace gs

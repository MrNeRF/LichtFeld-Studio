/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "core/cuda_error.hpp"
#include "training/kernels/ppisp_controller.cuh"
namespace lfs::training::kernels {
    constexpr int BLOCK_SIZE = 256;
    constexpr int TILE_SIZE = 16;
    __global__ void relu_backward_kernel(const float* grad, const float* input, float* out, const int n) {
        const int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx < n) {
            out[idx] = (input[idx] > 0.0f) ? grad[idx] : 0.0f;
        }
    }

    void launch_relu_backward(const float* grad, const float* input, float* out, const int n,
                              cudaStream_t stream) {
        const int blocks = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;
        relu_backward_kernel<<<blocks, BLOCK_SIZE, 0, stream>>>(grad, input, out, n);
        LFS_CUDA_LAUNCH_CHECK(stream, "training.ppisp_controller_pool.relu_backward");
    }

    __global__ void outer_product_accumulate_kernel(const float* a, const float* b, float* c,
                                                    const int m, const int n, const float scale) {
        const int i = blockIdx.y * blockDim.y + threadIdx.y;
        const int j = blockIdx.x * blockDim.x + threadIdx.x;
        if (i < m && j < n) {
            c[i * n + j] += scale * a[i] * b[j];
        }
    }

    void launch_outer_product_accumulate(const float* a, const float* b, float* c, const int m, const int n,
                                         const float scale, cudaStream_t stream) {
        const dim3 block(TILE_SIZE, TILE_SIZE);
        const dim3 grid((n + TILE_SIZE - 1) / TILE_SIZE, (m + TILE_SIZE - 1) / TILE_SIZE);
        outer_product_accumulate_kernel<<<grid, block, 0, stream>>>(a, b, c, m, n, scale);
        LFS_CUDA_LAUNCH_CHECK(stream, "training.ppisp_controller_pool.outer_product");
    }

    __global__ void bias_grad_accumulate_kernel(const float* grad, float* bias_grad, const int n) {
        const int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx < n) {
            bias_grad[idx] += grad[idx];
        }
    }

    void launch_bias_grad_accumulate(const float* grad, float* bias_grad, const int n, cudaStream_t stream) {
        const int blocks = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;
        bias_grad_accumulate_kernel<<<blocks, BLOCK_SIZE, 0, stream>>>(grad, bias_grad, n);
        LFS_CUDA_LAUNCH_CHECK(stream, "training.ppisp_controller_pool.bias_grad");
    }

} // namespace lfs::training::kernels

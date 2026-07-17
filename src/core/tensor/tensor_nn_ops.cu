/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "core/cuda_error.hpp"
#include "internal/tensor_functors.hpp"
#include "internal/tensor_nn_ops.hpp"
#include <cassert>
#include <cuda_runtime.h>
#include <limits>

namespace lfs::core::tensor_ops {

    namespace {
        constexpr int BLOCK_SIZE = 256;
    }

    __global__ void max_pool2d_kernel(const float* __restrict__ input,
                                      float* __restrict__ output,
                                      int N, int C, int H_in, int W_in,
                                      int H_out, int W_out,
                                      int kernel, int stride, int padding) {
        const int total = N * C * H_out * W_out;

        for (int idx = blockIdx.x * blockDim.x + threadIdx.x; idx < total;
             idx += blockDim.x * gridDim.x) {
            const int w_out = idx % W_out;
            const int h_out = (idx / W_out) % H_out;
            const int c = (idx / (W_out * H_out)) % C;
            const int n = idx / (W_out * H_out * C);

            const int h_start = h_out * stride - padding;
            const int w_start = w_out * stride - padding;

            float max_val = -std::numeric_limits<float>::infinity();

            for (int kh = 0; kh < kernel; ++kh) {
                const int h_in = h_start + kh;
                if (h_in < 0 || h_in >= H_in)
                    continue;

                for (int kw = 0; kw < kernel; ++kw) {
                    const int w_in = w_start + kw;
                    if (w_in < 0 || w_in >= W_in)
                        continue;

                    const int input_idx = ((n * C + c) * H_in + h_in) * W_in + w_in;
                    max_val = ops::max_reduce_op{}(max_val, input[input_idx]);
                }
            }

            output[idx] = max_val;
        }
    }

    __global__ void adaptive_avg_pool2d_kernel(const float* __restrict__ input,
                                               float* __restrict__ output,
                                               int N, int C, int H_in, int W_in,
                                               int H_out, int W_out) {
        const int total = N * C * H_out * W_out;

        for (int idx = blockIdx.x * blockDim.x + threadIdx.x; idx < total;
             idx += blockDim.x * gridDim.x) {
            const int w_out = idx % W_out;
            const int h_out = (idx / W_out) % H_out;
            const int c = (idx / (W_out * H_out)) % C;
            const int n = idx / (W_out * H_out * C);

            const int h_start = (h_out * H_in) / H_out;
            const int h_end = ((h_out + 1) * H_in + H_out - 1) / H_out;
            const int w_start = (w_out * W_in) / W_out;
            const int w_end = ((w_out + 1) * W_in + W_out - 1) / W_out;

            float sum = 0.0f;
            int count = 0;

            for (int h = h_start; h < h_end; ++h) {
                for (int w = w_start; w < w_end; ++w) {
                    const int input_idx = ((n * C + c) * H_in + h) * W_in + w;
                    sum += input[input_idx];
                    ++count;
                }
            }

            output[idx] = count > 0 ? sum / static_cast<float>(count) : 0.0f;
        }
    }

    void launch_max_pool2d(const float* input, float* output,
                           int N, int C, int H_in, int W_in,
                           int H_out, int W_out,
                           int kernel_size, int stride, int padding,
                           cudaStream_t stream) {
        const int total = N * C * H_out * W_out;
        if (total == 0)
            return;

        const int num_blocks = (total + BLOCK_SIZE - 1) / BLOCK_SIZE;
        max_pool2d_kernel<<<num_blocks, BLOCK_SIZE, 0, stream>>>(
            input, output, N, C, H_in, W_in, H_out, W_out, kernel_size, stride, padding);
        LFS_CUDA_LAUNCH_CHECK(stream, "tensor.nn.max_pool2d");
    }

    void launch_adaptive_avg_pool2d(const float* input, float* output,
                                    int N, int C, int H_in, int W_in,
                                    int H_out, int W_out,
                                    cudaStream_t stream) {
        const int total = N * C * H_out * W_out;
        if (total == 0)
            return;

        const int num_blocks = (total + BLOCK_SIZE - 1) / BLOCK_SIZE;
        adaptive_avg_pool2d_kernel<<<num_blocks, BLOCK_SIZE, 0, stream>>>(
            input, output, N, C, H_in, W_in, H_out, W_out);
        LFS_CUDA_LAUNCH_CHECK(stream, "tensor.nn.adaptive_avg_pool2d");
    }

    __global__ void bias_relu_kernel(const float* __restrict__ input,
                                     const float* __restrict__ bias,
                                     float* __restrict__ output,
                                     int total_elements, int channels, int spatial_size) {
        for (int idx = blockIdx.x * blockDim.x + threadIdx.x; idx < total_elements;
             idx += blockDim.x * gridDim.x) {
            const int c = (idx / spatial_size) % channels;
            const float val = input[idx] + bias[c];
            output[idx] = fmaxf(val, 0.0f);
        }
    }

    __global__ void bias_add_kernel(const float* __restrict__ input,
                                    const float* __restrict__ bias,
                                    float* __restrict__ output,
                                    int total_elements, int channels, int spatial_size) {
        for (int idx = blockIdx.x * blockDim.x + threadIdx.x; idx < total_elements;
             idx += blockDim.x * gridDim.x) {
            const int c = (idx / spatial_size) % channels;
            output[idx] = input[idx] + bias[c];
        }
    }

    void launch_bias_relu(const float* input, const float* bias, float* output,
                          int total_elements, int channels, int spatial_size,
                          cudaStream_t stream) {
        if (total_elements == 0)
            return;
        const int num_blocks = (total_elements + BLOCK_SIZE - 1) / BLOCK_SIZE;
        bias_relu_kernel<<<num_blocks, BLOCK_SIZE, 0, stream>>>(
            input, bias, output, total_elements, channels, spatial_size);
        LFS_CUDA_LAUNCH_CHECK(stream, "tensor.nn.bias_relu");
    }

    void launch_bias_add(const float* input, const float* bias, float* output,
                         int total_elements, int channels, int spatial_size,
                         cudaStream_t stream) {
        if (total_elements == 0)
            return;
        const int num_blocks = (total_elements + BLOCK_SIZE - 1) / BLOCK_SIZE;
        bias_add_kernel<<<num_blocks, BLOCK_SIZE, 0, stream>>>(
            input, bias, output, total_elements, channels, spatial_size);
        LFS_CUDA_LAUNCH_CHECK(stream, "tensor.nn.bias_add");
    }

    __global__ void relu_kernel(const float* __restrict__ input,
                                float* __restrict__ output, int n) {
        for (int idx = blockIdx.x * blockDim.x + threadIdx.x; idx < n;
             idx += blockDim.x * gridDim.x) {
            output[idx] = fmaxf(input[idx], 0.0f);
        }
    }

    void launch_relu(const float* input, float* output, int n, cudaStream_t stream) {
        if (n == 0)
            return;
        const int num_blocks = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;
        relu_kernel<<<num_blocks, BLOCK_SIZE, 0, stream>>>(input, output, n);
        LFS_CUDA_LAUNCH_CHECK(stream, "tensor.nn.relu");
    }

} // namespace lfs::core::tensor_ops

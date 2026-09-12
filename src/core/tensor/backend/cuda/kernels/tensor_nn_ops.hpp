/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include <cstddef>
#include <cuda_runtime.h>

namespace lfs::core::tensor_ops {

    // MaxPool2d: window-based max pooling [N,C,H,W] -> [N,C,H_out,W_out]
    // kernel_size: pooling window size
    // stride: step between windows (default = kernel_size)
    // padding: zero-padding added to input
    void launch_max_pool2d(const float* input, float* output,
                           int N, int C, int H_in, int W_in,
                           int H_out, int W_out,
                           int kernel_size, int stride, int padding,
                           cudaStream_t stream = nullptr);

    // AdaptiveAvgPool2d: pools to fixed output size regardless of input size
    // Adaptive window sizes computed per output position
    void launch_adaptive_avg_pool2d(const float* input, float* output,
                                    int N, int C, int H_in, int W_in,
                                    int H_out, int W_out,
                                    cudaStream_t stream = nullptr);

    // Fused bias + relu: output = relu(input + bias)
    // input: [N, C] or [N, C, H, W], bias: [C], output: same as input
    void launch_bias_relu(const float* input, const float* bias, float* output,
                          int total_elements, int channels, int spatial_size,
                          cudaStream_t stream = nullptr);

    // Fused bias only: output = input + bias
    void launch_bias_add(const float* input, const float* bias, float* output,
                         int total_elements, int channels, int spatial_size,
                         cudaStream_t stream = nullptr);

    // ReLU: output = max(0, input)
    void launch_relu(const float* input, float* output, int n,
                     cudaStream_t stream = nullptr);

} // namespace lfs::core::tensor_ops

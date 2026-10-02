/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once
#include <cuda_runtime.h>
namespace lfs::training::kernels {
    void launch_relu_backward(const float* grad, const float* input, float* out, int n, cudaStream_t stream);
    void launch_outer_product_accumulate(const float* a, const float* b, float* c, int m, int n, float scale, cudaStream_t stream);
    void launch_bias_grad_accumulate(const float* grad, float* bias_grad, int n, cudaStream_t stream);
} // namespace lfs::training::kernels

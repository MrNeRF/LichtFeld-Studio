/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#pragma once

#include "core/nn/ops.hpp"

#include <array>
#include <cstddef>

// Neural-network ops on a backend's dedicated kernels (Metal), for backends
// that report them through available(). Operands are contiguous and share
// one dtype and backend; the callers validate them.
namespace lfs::core::nn::dedicated {
    bool available(const Tensor& tensor);
    // out[b][m][n] = act(a[b][m] . w + bias[n]) * scale[n] + residual[b][m][n],
    // with w stored [k][n] or, with trans_b, [n][k], and shared by every batch
    // unless batched_b.
    Tensor linear(const Tensor& a, const Tensor& w, const Tensor* bias, const Tensor* scale, const Tensor* residual,
                  const TensorShape& output_shape, std::size_t batch, std::size_t m, std::size_t n, std::size_t k,
                  bool trans_b, bool batched_b, Activation activation);
    // Layer norm over the last dimension, or RMS norm without a bias.
    Tensor norm(const Tensor& input, const Tensor& weight, const Tensor* bias, float eps);
    // [B, H, N, d] attention; the mask strides step batch, head, query and key.
    Tensor attention(const Tensor& q, const Tensor& k, const Tensor& v, const Tensor* mask, float scale,
                     const std::array<long long, 4>& mask_strides);
    // NCHW convolution with OIHW weights, or with transpose the transposed
    // convolution with [C_in][C_out / groups][kh][kw] weights.
    Tensor conv2d(const Tensor& input, const Tensor& weight, const Tensor* bias, const Conv2dParams& params,
                  int out_h, int out_w, bool transpose);
} // namespace lfs::core::nn::dedicated

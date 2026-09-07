/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#pragma once

#include "core/nn/ops.hpp"

namespace lfs::core::nn::vulkan {
    Tensor gemm(const Tensor& a, const Tensor& b, bool trans_b, const Tensor* bias,
                Activation activation, const Tensor* residual = nullptr, const Tensor* scale = nullptr);
    Tensor norm(const Tensor& input, const Tensor& weight, const Tensor* bias, float eps);
    Tensor softmax(const Tensor& input, const Tensor* mask);
    Tensor attention(const Tensor& q, const Tensor& k, const Tensor& v, const Tensor* mask, float scale);
    Tensor activate(const Tensor& input, Activation activation);
    Tensor conv(const Tensor& input, const Tensor& weight, const Tensor* bias,
                const Conv2dParams& params, bool transpose = false);
    Tensor resize(const Tensor& input, int height, int width, ResizeMode mode, CoordTransform coord);
    Tensor pool(const Tensor& input, int kh, int kw, int sh, int sw, int ph, int pw,
                bool average = false, bool count_include_pad = true);
    Tensor window_partition(const Tensor& input, int window);
    Tensor window_unpartition(const Tensor& input, int window, int height, int width);
    std::array<Tensor, 3> split_qkv(const Tensor& input, int heads);
    Tensor merge_heads(const Tensor& input);
    Tensor fourier_pe(const Tensor& coords, const Tensor& gaussian);
    Tensor grid(const Tensor& like, int height, int width, float u0, float u1, float v0, float v1);
} // namespace lfs::core::nn::vulkan

/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#pragma once

#include "core/export.hpp"
#include "core/tensor.hpp"

#include <array>
#include <cstddef>
#include <optional>
#include <utility>

namespace lfs::core::nn {

    // Epilogue applied to GEMM/linear results after bias (if any).
    // GeluErf is the exact 0.5*x*(1+erf(x/sqrt(2))) form; GeluTanh is the
    // BERT/GPT tanh approximation.
    enum class Activation : int {
        None = 0,
        Relu = 1,
        GeluTanh = 2,
        GeluErf = 3,
        Silu = 4,
    };

    enum class GELUApprox : int {
        Erf = 0,
        Tanh = 1,
    };

    enum class ResizeMode : int {
        Nearest = 0,
        Bilinear = 1,
        Cubic = 2,
    };

    enum class ConvPadMode : int {
        Zeros = 0,
        Replicate = 1,
    };

    // Matches ONNX Resize coordinate_transformation_mode for the subset we
    // implement: half_pixel, asymmetric, align_corners.
    enum class CoordTransform : int {
        HalfPixel = 0,
        Asymmetric = 1,
        AlignCorners = 2,
    };

    struct Conv2dParams {
        int stride_h = 1;
        int stride_w = 1;
        int pad_h = 0;
        int pad_w = 0;
        int dilation_h = 1;
        int dilation_w = 1;
        int groups = 1;
        int output_pad_h = 0;
        int output_pad_w = 0;
        ConvPadMode pad_mode = ConvPadMode::Zeros;
    };

    // C = op(A) @ op(B) [+ bias] [+ activation].
    // A, B, C are fp32 or fp16 (fp16 uses tensor cores with fp32 accumulate).
    // Leading dimensions of A/B collapse into a GEMM batch. trans_b defaults
    // to true so C = A @ Bᵀ matches linear (weight stored as [N, K]).
    // residual, if given, must match the [..., M, N] output and is added after
    // (acc [+ bias] [+ activation]) * scale. scale, if given, has N elements.
    [[nodiscard]] LFS_CORE_API Tensor gemm(const Tensor& a, const Tensor& b,
                                           bool trans_a = false, bool trans_b = true,
                                           const Tensor* bias = nullptr,
                                           Activation activation = Activation::None,
                                           const Tensor* residual = nullptr,
                                           const Tensor* scale = nullptr);

    // y = x @ Wᵀ [+ bias] [+ activation]. x is [..., K], W is [N, K], bias [N].
    [[nodiscard]] LFS_CORE_API Tensor linear(const Tensor& input, const Tensor& weight,
                                             const Tensor* bias = nullptr,
                                             Activation activation = Activation::None);

    // Batched GEMM. a is [B, M, K], b is [B, N, K] if trans_b else [B, K, N].
    // A rank-2 b is broadcast across the batch.
    [[nodiscard]] LFS_CORE_API Tensor bmm(const Tensor& a, const Tensor& b,
                                          bool trans_a = false, bool trans_b = false,
                                          const Tensor* bias = nullptr,
                                          Activation activation = Activation::None);

    // LayerNorm over the last dimension: y = (x - mean) / sqrt(var + eps) * weight + bias.
    [[nodiscard]] LFS_CORE_API Tensor layer_norm(const Tensor& input, const Tensor& weight,
                                                 const Tensor& bias, float eps = 1e-5f);

    // RMSNorm over the last dimension: y = x / sqrt(mean(x²) + eps) * weight.
    [[nodiscard]] LFS_CORE_API Tensor rms_norm(const Tensor& input, const Tensor& weight,
                                               float eps = 1e-6f);

    // Softmax over the last dimension. mask, if present, is added before the
    // softmax (broadcastable to input).
    [[nodiscard]] LFS_CORE_API Tensor softmax(const Tensor& input, const Tensor* mask = nullptr);

    // Scaled dot-product attention on [B, H, N, d] (or [B, H, Nq, d] vs [B, H, Nk, d]).
    // Online-softmax tiling; does not materialise the N×N score matrix.
    // scale <= 0 selects 1/sqrt(d). mask is additive and broadcastable to [B, H, Nq, Nk].
    [[nodiscard]] LFS_CORE_API Tensor attention(const Tensor& query, const Tensor& key,
                                                const Tensor& value, const Tensor* mask = nullptr,
                                                float scale = 0.0f);

    // Fold sequence dim N into windows: [B, H, N, d] -> [B * n_windows, H, window, d],
    // padding N up to a multiple of window with zeros.
    [[nodiscard]] LFS_CORE_API Tensor window_partition(const Tensor& input, int window_size);

    // Inverse of window_partition. original_n is the unpadded sequence length.
    [[nodiscard]] LFS_CORE_API Tensor window_unpartition(const Tensor& windows, int window_size,
                                                         int original_n);

    [[nodiscard]] LFS_CORE_API std::pair<int, int> conv2d_output_hw(
        int height, int width, int kernel_h, int kernel_w, const Conv2dParams& params);

    [[nodiscard]] LFS_CORE_API std::pair<int, int> conv_transpose2d_output_hw(
        int height, int width, int kernel_h, int kernel_w, const Conv2dParams& params);

    [[nodiscard]] LFS_CORE_API std::size_t conv2d_workspace_bytes(
        const TensorShape& input_shape, const TensorShape& weight_shape,
        const Conv2dParams& params, DataType dtype);

    [[nodiscard]] LFS_CORE_API std::size_t conv_transpose2d_workspace_bytes(
        const TensorShape& input_shape, const TensorShape& weight_shape,
        const Conv2dParams& params, DataType dtype);

    // NCHW conv. weight is [C_out, C_in/groups, kH, kW]. workspace, if given, must
    // be a CUDA tensor of at least conv2d_workspace_bytes(); otherwise one is
    // allocated for this call.
    [[nodiscard]] LFS_CORE_API Tensor conv2d(const Tensor& input, const Tensor& weight,
                                             const Tensor* bias, const Conv2dParams& params,
                                             Tensor* workspace = nullptr);

    [[nodiscard]] LFS_CORE_API Tensor conv_transpose2d(const Tensor& input, const Tensor& weight,
                                                       const Tensor* bias,
                                                       const Conv2dParams& params,
                                                       Tensor* workspace = nullptr);

    // NCHW resize. nearest uses floor(x_orig) after the coordinate transform
    // (ONNX nearest_mode=floor). bilinear is the ONNX linear mode. cubic is
    // the ONNX cubic kernel with cubic_coeff_a = -0.75.
    [[nodiscard]] LFS_CORE_API Tensor resize2d(const Tensor& input, int out_h, int out_w,
                                               ResizeMode mode, CoordTransform coord);

    [[nodiscard]] LFS_CORE_API Tensor max_pool2d(const Tensor& input, int kernel_h, int kernel_w,
                                                 int stride_h, int stride_w, int pad_h, int pad_w);

    // count_include_pad matches ONNX AveragePool (true) vs PyTorch (false).
    [[nodiscard]] LFS_CORE_API Tensor avg_pool2d(const Tensor& input, int kernel_h, int kernel_w,
                                                 int stride_h, int stride_w, int pad_h, int pad_w,
                                                 bool count_include_pad = true);

    [[nodiscard]] LFS_CORE_API Tensor gelu(const Tensor& input,
                                           GELUApprox approx = GELUApprox::Erf);
    [[nodiscard]] LFS_CORE_API Tensor silu(const Tensor& input);
    [[nodiscard]] LFS_CORE_API Tensor relu(const Tensor& input);

    [[nodiscard]] LFS_CORE_API Tensor cast(const Tensor& input, DataType dtype);

    // qkv is [B, S, 3*H*D] or [B, S, 3, H, D]. Returns Q,K,V each [B, H, S, D].
    [[nodiscard]] LFS_CORE_API std::array<Tensor, 3> split_qkv(const Tensor& qkv, int heads);

    // context [B, H, S, D] -> [B, S, H*D].
    [[nodiscard]] LFS_CORE_API Tensor merge_heads(const Tensor& context);

    // MoGe-2 unit-circle UV encoding as NCHW [1, 2, H, W].
    [[nodiscard]] LFS_CORE_API Tensor uv_grid(int height, int width, float aspect, DataType dtype,
                                              Device device, cudaStream_t stream);

    // y = x + hidden * gamma with gamma broadcast on the last dimension.
    [[nodiscard]] LFS_CORE_API Tensor residual_scale(const Tensor& x, const Tensor& hidden,
                                                     const Tensor& gamma);

} // namespace lfs::core::nn

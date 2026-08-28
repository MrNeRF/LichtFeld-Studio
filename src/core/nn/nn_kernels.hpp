/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#pragma once

#include "core/tensor_fwd.hpp"

#include <cstddef>
#include <cuda_runtime.h>

namespace lfs::core::nn::kernels {

    // Activation integers match lfs::core::nn::Activation.
    // Coord/resize integers match CoordTransform / ResizeMode.

    // trans_c stores C as [N, M] (column-major / NCHW) instead of [M, N].
    // Bias is always along N (added to every M row of a given column).
    // residual, if non-null, is [M, N] row-major added after scale.
    // scale, if non-null, is [N] multiplied after bias/activation.
    // scatter_h > 0 selects a 2x2 stride-2 conv-transpose store: M = Nimg*H*W,
    // N = Cout*4, C is NCHW [Cout, 2H, 2W].
    void gemm(const void* a, const void* b, void* c, int m, int n, int k,
              long long stride_a, long long stride_b, long long stride_c, int batch,
              bool trans_a, bool trans_b, const void* bias, int activation,
              DataType dtype, cudaStream_t stream, bool trans_c = false,
              const void* residual = nullptr, const void* scale = nullptr,
              int scatter_h = 0, int scatter_w = 0);

    // Implicit GEMM conv: A is gathered from NCHW input (never materialised).
    // Output is NCHW. Weight is OIHW. Bias, if non-null, is [C_out].
    void conv2d_implicit(const void* input, const void* weight, const void* bias, void* output,
                         int n, int cin, int h, int w, int cout, int kh, int kw, int out_h,
                         int out_w, int stride_h, int stride_w, int pad_h, int pad_w, int dil_h,
                         int dil_w, int pad_mode, DataType dtype, cudaStream_t stream);

    void layer_norm(const void* x, const void* weight, const void* bias, void* y,
                    int rows, int cols, float eps, DataType dtype, cudaStream_t stream);

    void rms_norm(const void* x, const void* weight, void* y, int rows, int cols,
                  float eps, DataType dtype, cudaStream_t stream);

    void softmax(const void* x, const void* mask, void* y, int rows, int cols,
                 long long mask_stride_row, long long mask_stride_col, bool has_mask,
                 DataType dtype, cudaStream_t stream);

    void attention(const void* q, const void* k, const void* v, const void* mask, void* o,
                   int batch, int heads, int n_q, int n_k, int d, float scale,
                   long long mask_sb, long long mask_sh, long long mask_sq, long long mask_sk,
                   bool has_mask, DataType dtype, cudaStream_t stream);

    void im2col(const void* input, void* col, int n, int c, int h, int w, int k_h, int k_w,
                int out_h, int out_w, int stride_h, int stride_w, int pad_h, int pad_w,
                int dil_h, int dil_w, int c_start, int c_count, int pad_mode, DataType dtype,
                cudaStream_t stream);

    void col2im(const void* col, void* output, int n, int c, int h, int w, int k_h, int k_w,
                int in_h, int in_w, int stride_h, int stride_w, int pad_h, int pad_w,
                int dil_h, int dil_w, int c_start, int c_count, DataType dtype,
                cudaStream_t stream);

    void resize2d(const void* input, void* output, int n, int c, int in_h, int in_w,
                  int out_h, int out_w, int mode, int coord, DataType dtype,
                  cudaStream_t stream);

    void max_pool2d(const void* input, void* output, int n, int c, int in_h, int in_w,
                    int out_h, int out_w, int k_h, int k_w, int stride_h, int stride_w,
                    int pad_h, int pad_w, DataType dtype, cudaStream_t stream);

    void avg_pool2d(const void* input, void* output, int n, int c, int in_h, int in_w,
                    int out_h, int out_w, int k_h, int k_w, int stride_h, int stride_w,
                    int pad_h, int pad_w, bool count_include_pad, DataType dtype,
                    cudaStream_t stream);

    void gelu(const void* x, void* y, std::size_t n, int approx, DataType dtype,
              cudaStream_t stream);

    void silu(const void* x, void* y, std::size_t n, DataType dtype, cudaStream_t stream);

    void relu(const void* x, void* y, std::size_t n, DataType dtype, cudaStream_t stream);

    void sigmoid(const void* x, void* y, std::size_t n, DataType dtype, cudaStream_t stream);

    void window_partition_2d(const void* input, void* output, int batch, int height, int width,
                             int channels, int window, int n_h, int n_w, DataType dtype,
                             cudaStream_t stream);

    void window_unpartition_2d(const void* windows, void* output, int batch, int height, int width,
                               int channels, int window, int n_h, int n_w, DataType dtype,
                               cudaStream_t stream);

    void fourier_pe_coords(const void* coords, const void* gaussian, void* output, int count,
                           int feats, DataType dtype, cudaStream_t stream);

    void fourier_pe_grid(const void* gaussian, void* output, int height, int width, int feats,
                         DataType dtype, cudaStream_t stream);

    void channel_bias(void* nchw, const void* bias, int n, int c, int spatial, DataType dtype,
                      cudaStream_t stream);

    // qkv is [B, S, 3, H, D] packed; q/k/v are [B, H, S, D].
    void split_qkv(const void* qkv, void* q, void* k, void* v, int batch, int seq, int heads,
                   int d, DataType dtype, cudaStream_t stream);

    // qkv is [B, H, W, 3*heads*d] packed as [3, heads, d] on the last dim.
    // Writes Q/K/V as windowed [B*nH*nW, heads, window*window, d], zero-filling
    // the bottom/right pad so H/W need not be multiples of `window`.
    void split_qkv_window_2d(const void* qkv, void* q, void* k, void* v, int batch, int height,
                             int width, int heads, int d, int window, int n_h, int n_w,
                             const void* bias, DataType dtype, cudaStream_t stream);

    // context [B, H, S, D] -> packed [B, S, H, D].
    void merge_heads(const void* context, void* packed, int batch, int heads, int seq, int d,
                     DataType dtype, cudaStream_t stream);

    // Inverse of split_qkv_window_2d on the Q layout: [B*nH*nW, heads, ws*ws, d]
    // -> [B, orig_h, orig_w, heads*d], cropping pad.
    void merge_heads_unwindow_2d(const void* context, void* packed, int batch, int orig_h,
                                 int orig_w, int heads, int d, int window, int n_h, int n_w,
                                 DataType dtype, cudaStream_t stream);

    // 2x2 stride-2 max-pool over the spatial sequence of [B, heads, H*W, D].
    void max_pool_heads_2d(const void* input, void* output, int batch, int heads, int height,
                           int width, int d, DataType dtype, cudaStream_t stream);

    // 2x2 stride-2 max-pool on BHWC [B, H, W, C] -> [B, H/2, W/2, C].
    void max_pool2d_bhwc(const void* input, void* output, int batch, int height, int width,
                         int channels, DataType dtype, cudaStream_t stream);

    void uv_grid(void* output, int height, int width, float u0, float u1, float v0, float v1,
                 DataType dtype, cudaStream_t stream);

    // y = x + hidden * gamma, gamma has `cols` elements.
    void residual_scale(const void* x, const void* hidden, const void* gamma, void* y, int rows,
                        int cols, DataType dtype, cudaStream_t stream);

} // namespace lfs::core::nn::kernels

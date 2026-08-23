/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#pragma once

#include "core/tensor_fwd.hpp"

#include <cstddef>
#include <cuda_runtime.h>

namespace lfs::core::nn::kernels {

    // Activation integers match lfs::core::nn::Activation.
    // Coord/resize integers match CoordTransform / ResizeMode.

    void gemm(const void* a, const void* b, void* c, int m, int n, int k,
              long long stride_a, long long stride_b, long long stride_c, int batch,
              bool trans_a, bool trans_b, const void* bias, int activation,
              DataType dtype, cudaStream_t stream);

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
                int dil_h, int dil_w, int c_start, int c_count, DataType dtype,
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

    void channel_bias(void* nchw, const void* bias, int n, int c, int spatial, DataType dtype,
                      cudaStream_t stream);

} // namespace lfs::core::nn::kernels

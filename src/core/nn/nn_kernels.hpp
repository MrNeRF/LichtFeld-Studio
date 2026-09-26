/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#pragma once

#include "core/cuda_types.hpp"
#include "core/tensor_fwd.hpp"

#include <cstddef>

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
    // Output is NCHW. Weight is OIHW. Bias, if non-null, is [C_out]. The
    // tensor-core path needs a tap-major weight copy: either a prebuilt
    // weight_taps, or weight_scratch of conv2d_weight_scratch_bytes() to build
    // one per call. With neither the WMMA path runs.
    void conv2d_implicit(const void* input, const void* weight, const void* weight_taps,
                         const void* bias, void* output, void* weight_scratch, int n, int cin,
                         int h, int w, int cout, int kh, int kw, int out_h, int out_w,
                         int stride_h, int stride_w, int pad_h, int pad_w, int dil_h, int dil_w,
                         int pad_mode, int activation, DataType dtype, cudaStream_t stream);

    std::size_t conv2d_weight_scratch_bytes(int cout, int cin, DataType dtype);

    // sm_80+ tensor-core path for the 3x3 implicit conv on tap-major
    // ([9][C_out][C_in]) fp16 weights built by conv3x3_weight_taps from OIHW.
    bool conv3x3_mma_available();
    void conv3x3_weight_taps(const void* weight, void* weight_taps, int cout, int cin,
                             cudaStream_t stream);
    void conv2d_implicit_3x3_mma(const void* input, const void* weight_taps, const void* bias,
                                 void* output, int n, int cin, int h, int w, int cout, int out_h,
                                 int out_w, int pad_mode, int activation, cudaStream_t stream);

    // LPIPS layer 1: fp32 RGB [N][3][H][W], per-channel ((2x-1 | x) - shift) / scale
    // folded into the gather, 3x3 zero-padded conv with 64 fp16 OIHW filters,
    // bias and ReLU, fp16 NCHW output.
    void lpips_rgb_conv3x3(const float* input, const void* weight, const void* bias, void* output,
                           const float* shift, const float* scale, bool official_scaling, int n,
                           int h, int w, cudaStream_t stream);

    // LPIPS block tail: channel-normalised, lin-weighted squared distance of two
    // fp16 NCHW feature maps accumulated into *result (caller zeroes it), with
    // an optional fused 2x2 stride-2 max pool of both maps into pooled_x/y.
    void lpips_pool_reduce(const void* x, const void* y, const void* lin_weight, float* result,
                           void* pooled_x, void* pooled_y, int n, int channels, int h, int w,
                           int interior_y0, int interior_y1, int interior_x0, int interior_x1,
                           float inv_count, cudaStream_t stream);

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

    // ---- Dense matching (RoMa v1 and v2) ------------------------------------

    void grid_sample_bhwc(const void* input, const float* grid, void* output, int channels,
                          int h_in, int w_in, int h_out, int w_out, DataType dtype,
                          cudaStream_t stream);

    // Correlation of f_a against a (2r+1)^2 window of f_b around warp, all
    // channel-last. Output is [h * w, (2r+1)^2].
    void local_correlation(const void* f_a_nhwc, const void* f_b_nhwc, const float* warp, void* out,
                           int channels, int h, int w, int radius, DataType dtype,
                           cudaStream_t stream);

    // [h*w, 2] float32 grid of pixel centres in [-1, 1], (x, y) order.
    void normalized_grid(float* out, int h, int w, cudaStream_t stream);

    // NCHW [1, 2, h, w] of (warp - grid) scaled per axis.
    void displacement_input(const float* warp, void* out, int h, int w, float sx, float sy,
                            DataType dtype, cudaStream_t stream);

    // Depthwise channel-last convolution, stride 1, same padding, kernel 3 or
    // 5. Weights are tap-major [kernel * kernel, channels].
    void depthwise_conv2d_nhwc(const void* input, const void* weight, const void* bias,
                               void* output, int channels, int h, int w, int kernel, int activation,
                               DataType dtype, cudaStream_t stream);

    // Separable antialiased bicubic resize (float32 NCHW) matching
    // torch.nn.functional.interpolate(..., mode="bicubic", antialias=True,
    // align_corners=False). workspace holds n * c * in_h * out_w floats.
    void resize_bicubic_aa(const float* input, float* output, float* workspace, int n, int c,
                           int in_h, int in_w, int out_h, int out_w, cudaStream_t stream);

    // dst[c * rows + r] = src[r * cols + c].
    // Same resize reading an interleaved [in_h, in_w, C] image (uint8 or
    // float32) and writing planar [C, out_h, out_w] float32 in [0, 1].
    // workspace holds channels * in_h * out_w floats.
    void resize_bicubic_aa_hwc(const void* input, float* output, float* workspace, int in_h,
                               int in_w, int channels, int out_h, int out_w, DataType src_dtype,
                               cudaStream_t stream);

    void transpose2d(const void* src, void* dst, int rows, int cols, DataType dtype,
                     cudaStream_t stream);

    // (x - mean) / stddev per channel on NCHW with 3 channels.
    void normalize_image(const void* src, void* dst, int pixels, const float* mean,
                         const float* stddev, DataType src_dtype, DataType dst_dtype,
                         cudaStream_t stream);

    // ---- RoMa v1 -----------------------------------------------------------

    // L2-normalises each row into a float32 destination.
    void l2_normalize_rows(const void* src, float* dst, int rows, int cols, DataType dtype,
                           cudaStream_t stream);

    // In place: exp((x - 1) / temperature) over a Gram matrix of normalised rows.
    void cosine_kernel_inplace(float* gram, long long count, float temperature,
                               cudaStream_t stream);

    void add_diagonal(float* matrix, int n, float sigma, cudaStream_t stream);

    // Conjugate gradients on [rows, cols] with independent right-hand sides.
    void column_dot(const float* a, const float* b, float* out, int rows, int cols,
                    cudaStream_t stream);
    void cg_step(float* x, float* r, const float* p, const float* ap, const float* rs,
                 const float* pap, int rows, int cols, cudaStream_t stream);
    void cg_direction(float* p, const float* r, const float* rs_new, const float* rs_old, int rows,
                      int cols, cudaStream_t stream);

    void fourier_cos(const float* src, float* dst, long long count, cudaStream_t stream);

    // Soft-argmax over a `side` x `side` class grid; writes [pixels, 2].
    // `row_stride` is the width of one logit row, which is wider than the class
    // count when the row carries a trailing certainty column.
    void cls_to_flow(const void* logits, float* flow, int pixels, int classes, int row_stride,
                     int side, DataType dtype, cudaStream_t stream);

    void v1_refiner_update(const float* prev_flow, const float* prev_cert, const void* delta,
                           float* flow, float* cert, int pixels, float step, DataType dtype,
                           cudaStream_t stream);

    void attenuate_certainty(const float* cert, const float* low, float* out, int pixels,
                             cudaStream_t stream);

} // namespace lfs::core::nn::kernels

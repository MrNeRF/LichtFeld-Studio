/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "core/cuda_error.hpp"
#include "nn_device.cuh"
#include "nn_kernels.hpp"

#include <algorithm>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <float.h>

namespace lfs::core::nn::kernels {
    namespace {

        __global__ void im2col_kernel(const void* __restrict__ input, void* __restrict__ col,
                                      int n, int c, int h, int w, int k_h, int k_w, int out_h,
                                      int out_w, int stride_h, int stride_w, int pad_h, int pad_w,
                                      int dil_h, int dil_w, int c_start, int c_count, int pad_mode,
                                      bool is_half) {
            const int m = n * out_h * out_w;
            const int kk = c_count * k_h * k_w;
            const int total = m * kk;
            for (int idx = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x); idx < total;
                 idx += static_cast<int>(blockDim.x * gridDim.x)) {
                const int row = idx / kk;
                const int col_i = idx % kk;
                const int ow = row % out_w;
                const int oh = (row / out_w) % out_h;
                const int ni = row / (out_w * out_h);
                const int kw = col_i % k_w;
                const int kh = (col_i / k_w) % k_h;
                const int ci = col_i / (k_w * k_h);
                const int ih = oh * stride_h - pad_h + kh * dil_h;
                const int iw = ow * stride_w - pad_w + kw * dil_w;
                const int ic = c_start + ci;
                float v = 0.0f;
                int yh = ih;
                int xw = iw;
                if (pad_mode == 1) {
                    yh = yh < 0 ? 0 : (yh >= h ? h - 1 : yh);
                    xw = xw < 0 ? 0 : (xw >= w ? w - 1 : xw);
                }
                if (yh >= 0 && yh < h && xw >= 0 && xw < w && ic >= 0 && ic < c) {
                    const long long in_i =
                        (((static_cast<long long>(ni) * c + ic) * h + yh) * w + xw);
                    v = device::ld_strided(input, in_i, is_half);
                }
                device::st_strided(col, idx, v, is_half);
            }
        }

        __global__ void col2im_kernel(const void* __restrict__ col, void* __restrict__ output,
                                      int n, int c, int h, int w, int k_h, int k_w, int in_h,
                                      int in_w, int stride_h, int stride_w, int pad_h, int pad_w,
                                      int dil_h, int dil_w, int c_start, int c_count, bool is_half) {
            const int total = n * c_count * h * w;
            for (int idx = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x); idx < total;
                 idx += static_cast<int>(blockDim.x * gridDim.x)) {
                const int ow = idx % w;
                const int oh = (idx / w) % h;
                const int ci = (idx / (w * h)) % c_count;
                const int ni = idx / (w * h * c_count);
                const int oc = c_start + ci;
                float sum = 0.0f;
                for (int kh = 0; kh < k_h; ++kh) {
                    const int ih_num = oh + pad_h - kh * dil_h;
                    if (ih_num < 0 || (stride_h > 0 && ih_num % stride_h != 0)) {
                        continue;
                    }
                    const int ih = stride_h ? ih_num / stride_h : 0;
                    if (ih < 0 || ih >= in_h) {
                        continue;
                    }
                    for (int kw = 0; kw < k_w; ++kw) {
                        const int iw_num = ow + pad_w - kw * dil_w;
                        if (iw_num < 0 || (stride_w > 0 && iw_num % stride_w != 0)) {
                            continue;
                        }
                        const int iw = stride_w ? iw_num / stride_w : 0;
                        if (iw < 0 || iw >= in_w) {
                            continue;
                        }
                        const int row = ni * in_h * in_w + ih * in_w + iw;
                        const int col_i = (ci * k_h + kh) * k_w + kw;
                        const int kk = c_count * k_h * k_w;
                        const long long col_idx = static_cast<long long>(row) * kk + col_i;
                        sum += device::ld_strided(col, col_idx, is_half);
                    }
                }
                const long long out_i =
                    (((static_cast<long long>(ni) * c + oc) * h + oh) * w + ow);
                device::st_strided(output, out_i, sum, is_half);
            }
        }

        __global__ void max_pool_kernel(const void* __restrict__ input, void* __restrict__ output,
                                        int n, int c, int in_h, int in_w, int out_h, int out_w,
                                        int k_h, int k_w, int stride_h, int stride_w, int pad_h,
                                        int pad_w, bool is_half) {
            const int total = n * c * out_h * out_w;
            for (int idx = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x); idx < total;
                 idx += static_cast<int>(blockDim.x * gridDim.x)) {
                const int ow = idx % out_w;
                const int oh = (idx / out_w) % out_h;
                const int ch = (idx / (out_w * out_h)) % c;
                const int ni = idx / (out_w * out_h * c);
                const int h0 = oh * stride_h - pad_h;
                const int w0 = ow * stride_w - pad_w;
                float best = -FLT_MAX;
                for (int kh = 0; kh < k_h; ++kh) {
                    const int ih = h0 + kh;
                    if (ih < 0 || ih >= in_h) {
                        continue;
                    }
                    for (int kw = 0; kw < k_w; ++kw) {
                        const int iw = w0 + kw;
                        if (iw < 0 || iw >= in_w) {
                            continue;
                        }
                        const long long in_i =
                            (((static_cast<long long>(ni) * c + ch) * in_h + ih) * in_w + iw);
                        best = fmaxf(best, device::ld_strided(input, in_i, is_half));
                    }
                }
                device::st_strided(output, idx, best, is_half);
            }
        }

        __global__ void avg_pool_kernel(const void* __restrict__ input, void* __restrict__ output,
                                        int n, int c, int in_h, int in_w, int out_h, int out_w,
                                        int k_h, int k_w, int stride_h, int stride_w, int pad_h,
                                        int pad_w, bool count_include_pad, bool is_half) {
            const int total = n * c * out_h * out_w;
            for (int idx = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x); idx < total;
                 idx += static_cast<int>(blockDim.x * gridDim.x)) {
                const int ow = idx % out_w;
                const int oh = (idx / out_w) % out_h;
                const int ch = (idx / (out_w * out_h)) % c;
                const int ni = idx / (out_w * out_h * c);
                const int h0 = oh * stride_h - pad_h;
                const int w0 = ow * stride_w - pad_w;
                float sum = 0.0f;
                int count = 0;
                for (int kh = 0; kh < k_h; ++kh) {
                    const int ih = h0 + kh;
                    for (int kw = 0; kw < k_w; ++kw) {
                        const int iw = w0 + kw;
                        if (ih < 0 || ih >= in_h || iw < 0 || iw >= in_w) {
                            if (count_include_pad) {
                                ++count;
                            }
                            continue;
                        }
                        const long long in_i =
                            (((static_cast<long long>(ni) * c + ch) * in_h + ih) * in_w + iw);
                        sum += device::ld_strided(input, in_i, is_half);
                        ++count;
                    }
                }
                const float v = count > 0 ? sum / static_cast<float>(count) : 0.0f;
                device::st_strided(output, idx, v, is_half);
            }
        }

        int grid_for(const int total) {
            const int block = 256;
            const int max_blocks = 2048;
            return std::min(max_blocks, (total + block - 1) / block);
        }

    } // namespace

    void im2col(const void* input, void* col, int n, int c, int h, int w, int k_h, int k_w,
                int out_h, int out_w, int stride_h, int stride_w, int pad_h, int pad_w,
                int dil_h, int dil_w, int c_start, int c_count, int pad_mode, DataType dtype,
                cudaStream_t stream) {
        const int total = n * out_h * out_w * c_count * k_h * k_w;
        if (total <= 0) {
            return;
        }
        im2col_kernel<<<grid_for(total), 256, 0, stream>>>(
            input, col, n, c, h, w, k_h, k_w, out_h, out_w, stride_h, stride_w, pad_h, pad_w,
            dil_h, dil_w, c_start, c_count, pad_mode, dtype == DataType::Float16);
        LFS_CUDA_LAUNCH_CHECK(stream, "nn.conv.im2col");
    }

    void col2im(const void* col, void* output, int n, int c, int h, int w, int k_h, int k_w,
                int in_h, int in_w, int stride_h, int stride_w, int pad_h, int pad_w,
                int dil_h, int dil_w, int c_start, int c_count, DataType dtype,
                cudaStream_t stream) {
        const int total = n * c_count * h * w;
        if (total <= 0) {
            return;
        }
        col2im_kernel<<<grid_for(total), 256, 0, stream>>>(
            col, output, n, c, h, w, k_h, k_w, in_h, in_w, stride_h, stride_w, pad_h, pad_w,
            dil_h, dil_w, c_start, c_count, dtype == DataType::Float16);
        LFS_CUDA_LAUNCH_CHECK(stream, "nn.conv.col2im");
    }

    void max_pool2d(const void* input, void* output, int n, int c, int in_h, int in_w,
                    int out_h, int out_w, int k_h, int k_w, int stride_h, int stride_w,
                    int pad_h, int pad_w, DataType dtype, cudaStream_t stream) {
        const int total = n * c * out_h * out_w;
        if (total <= 0) {
            return;
        }
        max_pool_kernel<<<grid_for(total), 256, 0, stream>>>(
            input, output, n, c, in_h, in_w, out_h, out_w, k_h, k_w, stride_h, stride_w, pad_h,
            pad_w, dtype == DataType::Float16);
        LFS_CUDA_LAUNCH_CHECK(stream, "nn.pool.max2d");
    }

    void avg_pool2d(const void* input, void* output, int n, int c, int in_h, int in_w,
                    int out_h, int out_w, int k_h, int k_w, int stride_h, int stride_w,
                    int pad_h, int pad_w, bool count_include_pad, DataType dtype,
                    cudaStream_t stream) {
        const int total = n * c * out_h * out_w;
        if (total <= 0) {
            return;
        }
        avg_pool_kernel<<<grid_for(total), 256, 0, stream>>>(
            input, output, n, c, in_h, in_w, out_h, out_w, k_h, k_w, stride_h, stride_w, pad_h,
            pad_w, count_include_pad, dtype == DataType::Float16);
        LFS_CUDA_LAUNCH_CHECK(stream, "nn.pool.avg2d");
    }

} // namespace lfs::core::nn::kernels

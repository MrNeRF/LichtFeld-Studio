/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "core/cuda_error.hpp"
#include "nn_device.cuh"
#include "nn_kernels.hpp"
#include "nn_nvtx.hpp"

#include <algorithm>
#include <cstdint>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

namespace lfs::core::nn::kernels {
    namespace {

        __device__ __forceinline__ float warp_sum(float v) {
#pragma unroll
            for (int offset = 16; offset > 0; offset >>= 1) {
                v += __shfl_xor_sync(0xffffffffu, v, offset);
            }
            return v;
        }

        __global__ void layer_norm_kernel(const void* __restrict__ x, const void* __restrict__ weight,
                                          const void* __restrict__ bias, void* __restrict__ y,
                                          int rows, int cols, float eps, bool is_half) {
            const int row = static_cast<int>(blockIdx.x);
            if (row >= rows) {
                return;
            }
            const int tid = static_cast<int>(threadIdx.x);
            const int nthreads = static_cast<int>(blockDim.x);
            const long long base = static_cast<long long>(row) * cols;

            float sum = 0.0f;
            float sumsq = 0.0f;
            for (int c = tid; c < cols; c += nthreads) {
                const float v = device::ld_strided(x, base + c, is_half);
                sum += v;
                sumsq += v * v;
            }
            __shared__ float red[32];
            float w = warp_sum(sum);
            if ((tid & 31) == 0) {
                red[tid / 32] = w;
            }
            __syncthreads();
            if (tid < 32) {
                const float v = (tid < (nthreads + 31) / 32) ? red[tid] : 0.0f;
                w = warp_sum(v);
                if (tid == 0) {
                    red[0] = w;
                }
            }
            __syncthreads();
            const float mean = red[0] / static_cast<float>(cols);

            w = warp_sum(sumsq);
            if ((tid & 31) == 0) {
                red[tid / 32] = w;
            }
            __syncthreads();
            if (tid < 32) {
                const float v = (tid < (nthreads + 31) / 32) ? red[tid] : 0.0f;
                w = warp_sum(v);
                if (tid == 0) {
                    red[0] = w;
                }
            }
            __syncthreads();
            const float var = red[0] / static_cast<float>(cols) - mean * mean;
            const float inv = rsqrtf(var + eps);

            for (int c = tid; c < cols; c += nthreads) {
                const float v = device::ld_strided(x, base + c, is_half);
                const float g = device::ld_strided(weight, c, is_half);
                const float b = device::ld_strided(bias, c, is_half);
                device::st_strided(y, base + c, (v - mean) * inv * g + b, is_half);
            }
        }

        __global__ void rms_norm_kernel(const void* __restrict__ x, const void* __restrict__ weight,
                                        void* __restrict__ y, int rows, int cols, float eps,
                                        bool is_half) {
            const int row = static_cast<int>(blockIdx.x);
            if (row >= rows) {
                return;
            }
            const int tid = static_cast<int>(threadIdx.x);
            const int nthreads = static_cast<int>(blockDim.x);
            const long long base = static_cast<long long>(row) * cols;

            float sumsq = 0.0f;
            for (int c = tid; c < cols; c += nthreads) {
                const float v = device::ld_strided(x, base + c, is_half);
                sumsq += v * v;
            }
            __shared__ float red[32];
            float w = warp_sum(sumsq);
            if ((tid & 31) == 0) {
                red[tid / 32] = w;
            }
            __syncthreads();
            if (tid < 32) {
                const float v = (tid < (nthreads + 31) / 32) ? red[tid] : 0.0f;
                w = warp_sum(v);
                if (tid == 0) {
                    red[0] = w;
                }
            }
            __syncthreads();
            const float inv = rsqrtf(red[0] / static_cast<float>(cols) + eps);

            for (int c = tid; c < cols; c += nthreads) {
                const float v = device::ld_strided(x, base + c, is_half);
                const float g = device::ld_strided(weight, c, is_half);
                device::st_strided(y, base + c, v * inv * g, is_half);
            }
        }

        __global__ void unary_kernel(const void* __restrict__ x, void* __restrict__ y,
                                     std::size_t n, int kind, bool is_half) {
            for (std::size_t i = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
                 i < n; i += static_cast<std::size_t>(blockDim.x) * gridDim.x) {
                const float v = device::ld_strided(x, static_cast<long long>(i), is_half);
                float o = v;
                switch (kind) {
                case 0:
                    o = device::gelu_erf(v);
                    break;
                case 1:
                    o = device::gelu_tanh(v);
                    break;
                case 2:
                    o = device::silu(v);
                    break;
                case 3:
                    o = fmaxf(v, 0.0f);
                    break;
                default:
                    break;
                }
                device::st_strided(y, static_cast<long long>(i), o, is_half);
            }
        }

        __device__ __forceinline__ float sample_nchw(const void* input, int n, int c, int h, int w,
                                                     int ni, int ch, float y, float x, int mode,
                                                     bool is_half) {
            if (mode == 0) {
                int iy = static_cast<int>(floorf(y));
                int ix = static_cast<int>(floorf(x));
                iy = iy < 0 ? 0 : (iy >= h ? h - 1 : iy);
                ix = ix < 0 ? 0 : (ix >= w ? w - 1 : ix);
                const long long idx =
                    (((static_cast<long long>(ni) * c + ch) * h + iy) * w + ix);
                return device::ld_strided(input, idx, is_half);
            }
            const int y0 = static_cast<int>(floorf(y));
            const int x0 = static_cast<int>(floorf(x));
            const int y1 = y0 + 1;
            const int x1 = x0 + 1;
            const float wy = y - static_cast<float>(y0);
            const float wx = x - static_cast<float>(x0);
            auto at = [&](int yy, int xx) {
                yy = yy < 0 ? 0 : (yy >= h ? h - 1 : yy);
                xx = xx < 0 ? 0 : (xx >= w ? w - 1 : xx);
                const long long idx =
                    (((static_cast<long long>(ni) * c + ch) * h + yy) * w + xx);
                return device::ld_strided(input, idx, is_half);
            };
            if (mode == 2) {
                const int y_base = static_cast<int>(floorf(y));
                const int x_base = static_cast<int>(floorf(x));
                const float fy = y - static_cast<float>(y_base);
                const float fx = x - static_cast<float>(x_base);
                auto cubic = [](float t) {
                    const float a = -0.75f;
                    const float abs_t = fabsf(t);
                    if (abs_t <= 1.0f) {
                        return ((a + 2.0f) * abs_t - (a + 3.0f)) * abs_t * abs_t + 1.0f;
                    }
                    if (abs_t < 2.0f) {
                        return ((a * abs_t - 5.0f * a) * abs_t + 8.0f * a) * abs_t - 4.0f * a;
                    }
                    return 0.0f;
                };
                float wy[4];
                float wx[4];
#pragma unroll
                for (int i = 0; i < 4; ++i) {
                    wy[i] = cubic(fy - static_cast<float>(i - 1));
                    wx[i] = cubic(fx - static_cast<float>(i - 1));
                }
                float acc = 0.0f;
#pragma unroll
                for (int i = 0; i < 4; ++i) {
                    float row = 0.0f;
#pragma unroll
                    for (int j = 0; j < 4; ++j) {
                        row += at(y_base + i - 1, x_base + j - 1) * wx[j];
                    }
                    acc += row * wy[i];
                }
                return acc;
            }
            auto at_pair = [&](int yy, int xx0, int xx1, float& a, float& b) {
                yy = yy < 0 ? 0 : (yy >= h ? h - 1 : yy);
                xx0 = xx0 < 0 ? 0 : (xx0 >= w ? w - 1 : xx0);
                xx1 = xx1 < 0 ? 0 : (xx1 >= w ? w - 1 : xx1);
                const long long row =
                    ((static_cast<long long>(ni) * c + ch) * h + yy) * w;
                if (is_half && xx1 == xx0 + 1) {
                    const auto* ptr = static_cast<const __half*>(input) + row + xx0;
                    if ((reinterpret_cast<uintptr_t>(ptr) & 3u) == 0) {
                        const auto hv = *reinterpret_cast<const __half2*>(ptr);
                        a = __half2float(hv.x);
                        b = __half2float(hv.y);
                        return;
                    }
                }
                a = device::ld_strided(input, row + xx0, is_half);
                b = device::ld_strided(input, row + xx1, is_half);
            };
            float v00, v01, v10, v11;
            at_pair(y0, x0, x1, v00, v01);
            at_pair(y1, x0, x1, v10, v11);
            const float v0 = v00 * (1.0f - wx) + v01 * wx;
            const float v1 = v10 * (1.0f - wx) + v11 * wx;
            return v0 * (1.0f - wy) + v1 * wy;
        }

        __global__ void resize2d_kernel(const void* __restrict__ input, void* __restrict__ output,
                                        int n, int c, int in_h, int in_w, int out_h, int out_w,
                                        int mode, int coord, bool is_half) {
            const int total = n * c * out_h * out_w;
            for (int idx = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x); idx < total;
                 idx += static_cast<int>(blockDim.x * gridDim.x)) {
                const int ow = idx % out_w;
                const int oh = (idx / out_w) % out_h;
                const int ch = (idx / (out_w * out_h)) % c;
                const int ni = idx / (out_w * out_h * c);
                const float fy = device::resize_coord(oh, in_h, out_h, coord);
                const float fx = device::resize_coord(ow, in_w, out_w, coord);
                const float v = sample_nchw(input, n, c, in_h, in_w, ni, ch, fy, fx, mode, is_half);
                device::st_strided(output, idx, v, is_half);
            }
        }

        int unary_grid(const std::size_t n) {
            const int block = 256;
            const int max_blocks = 2048;
            const int blocks = static_cast<int>((n + static_cast<std::size_t>(block) - 1) /
                                                static_cast<std::size_t>(block));
            return std::min(max_blocks, std::max(blocks, 1));
        }

    } // namespace

    void layer_norm(const void* x, const void* weight, const void* bias, void* y, int rows,
                    int cols, float eps, DataType dtype, cudaStream_t stream) {
        NvtxRange nvtx("nn.op/layer_norm");
        if (rows <= 0 || cols <= 0) {
            return;
        }
        const int threads = cols >= 256 ? 256 : 128;
        layer_norm_kernel<<<rows, threads, 0, stream>>>(
            x, weight, bias, y, rows, cols, eps, dtype == DataType::Float16);
        LFS_CUDA_LAUNCH_CHECK(stream, "nn.norm.layer");
    }

    void rms_norm(const void* x, const void* weight, void* y, int rows, int cols, float eps,
                  DataType dtype, cudaStream_t stream) {
        if (rows <= 0 || cols <= 0) {
            return;
        }
        const int threads = cols >= 256 ? 256 : 128;
        rms_norm_kernel<<<rows, threads, 0, stream>>>(
            x, weight, y, rows, cols, eps, dtype == DataType::Float16);
        LFS_CUDA_LAUNCH_CHECK(stream, "nn.norm.rms");
    }

    void gelu(const void* x, void* y, std::size_t n, int approx, DataType dtype,
              cudaStream_t stream) {
        NvtxRange nvtx("nn.op/gelu");
        if (n == 0) {
            return;
        }
        unary_kernel<<<unary_grid(n), 256, 0, stream>>>(
            x, y, n, approx == 0 ? 0 : 1, dtype == DataType::Float16);
        LFS_CUDA_LAUNCH_CHECK(stream, "nn.act.gelu");
    }

    void silu(const void* x, void* y, std::size_t n, DataType dtype, cudaStream_t stream) {
        if (n == 0) {
            return;
        }
        unary_kernel<<<unary_grid(n), 256, 0, stream>>>(x, y, n, 2, dtype == DataType::Float16);
        LFS_CUDA_LAUNCH_CHECK(stream, "nn.act.silu");
    }

    void relu(const void* x, void* y, std::size_t n, DataType dtype, cudaStream_t stream) {
        NvtxRange nvtx("nn.op/relu");
        if (n == 0) {
            return;
        }
        unary_kernel<<<unary_grid(n), 256, 0, stream>>>(x, y, n, 3, dtype == DataType::Float16);
        LFS_CUDA_LAUNCH_CHECK(stream, "nn.act.relu");
    }

    __global__ void channel_bias_kernel(void* __restrict__ nchw, const void* __restrict__ bias,
                                        int n, int c, int spatial, bool is_half) {
        const int total = n * c * spatial;
        for (int idx = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x); idx < total;
             idx += static_cast<int>(blockDim.x * gridDim.x)) {
            const int ch = (idx / spatial) % c;
            const float v = device::ld_strided(nchw, idx, is_half) +
                            device::ld_strided(bias, ch, is_half);
            device::st_strided(nchw, idx, v, is_half);
        }
    }

    void channel_bias(void* nchw, const void* bias, int n, int c, int spatial, DataType dtype,
                      cudaStream_t stream) {
        NvtxRange nvtx("nn.op/channel_bias");
        const int total = n * c * spatial;
        if (total <= 0) {
            return;
        }
        const int block = 256;
        const int grid = std::min(2048, (total + block - 1) / block);
        channel_bias_kernel<<<grid, block, 0, stream>>>(nchw, bias, n, c, spatial,
                                                        dtype == DataType::Float16);
        LFS_CUDA_LAUNCH_CHECK(stream, "nn.bias.channel");
    }

    void resize2d(const void* input, void* output, int n, int c, int in_h, int in_w, int out_h,
                  int out_w, int mode, int coord, DataType dtype, cudaStream_t stream) {
        NvtxRange nvtx("nn.op/resize2d");
        const int total = n * c * out_h * out_w;
        if (total <= 0) {
            return;
        }
        const int block = 256;
        const int grid = std::min(2048, (total + block - 1) / block);
        resize2d_kernel<<<grid, block, 0, stream>>>(
            input, output, n, c, in_h, in_w, out_h, out_w, mode, coord,
            dtype == DataType::Float16);
        LFS_CUDA_LAUNCH_CHECK(stream, "nn.resize2d");
    }

    __global__ void split_qkv_kernel(const void* __restrict__ qkv, void* __restrict__ q,
                                     void* __restrict__ k, void* __restrict__ v, int batch,
                                     int seq, int heads, int d, bool is_half) {
        const int total = batch * seq * heads * d;
        for (int idx = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x); idx < total;
             idx += static_cast<int>(blockDim.x * gridDim.x)) {
            const int dd = idx % d;
            const int h = (idx / d) % heads;
            const int s = (idx / (d * heads)) % seq;
            const int b = idx / (d * heads * seq);
            const long long src_q =
                (((((static_cast<long long>(b) * seq + s) * 3 + 0) * heads + h) * d) + dd);
            const long long src_k =
                (((((static_cast<long long>(b) * seq + s) * 3 + 1) * heads + h) * d) + dd);
            const long long src_v =
                (((((static_cast<long long>(b) * seq + s) * 3 + 2) * heads + h) * d) + dd);
            const long long dst =
                ((((static_cast<long long>(b) * heads + h) * seq + s) * d) + dd);
            device::st_strided(q, dst, device::ld_strided(qkv, src_q, is_half), is_half);
            device::st_strided(k, dst, device::ld_strided(qkv, src_k, is_half), is_half);
            device::st_strided(v, dst, device::ld_strided(qkv, src_v, is_half), is_half);
        }
    }

    __global__ void merge_heads_kernel(const void* __restrict__ context, void* __restrict__ packed,
                                       int batch, int heads, int seq, int d, bool is_half) {
        const int total = batch * seq * heads * d;
        for (int idx = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x); idx < total;
             idx += static_cast<int>(blockDim.x * gridDim.x)) {
            const int dd = idx % d;
            const int h = (idx / d) % heads;
            const int s = (idx / (d * heads)) % seq;
            const int b = idx / (d * heads * seq);
            const long long src =
                ((((static_cast<long long>(b) * heads + h) * seq + s) * d) + dd);
            const long long dst =
                ((((static_cast<long long>(b) * seq + s) * heads + h) * d) + dd);
            device::st_strided(packed, dst, device::ld_strided(context, src, is_half), is_half);
        }
    }

    __global__ void uv_grid_kernel(void* __restrict__ output, int height, int width, float u0,
                                   float u1, float v0, float v1, bool is_half) {
        const int total = height * width;
        for (int idx = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x); idx < total;
             idx += static_cast<int>(blockDim.x * gridDim.x)) {
            const int x = idx % width;
            const int y = idx / width;
            const float uu = (width == 1)
                                 ? u0
                                 : u0 + (u1 - u0) * static_cast<float>(x) /
                                            static_cast<float>(width - 1);
            const float vv = (height == 1)
                                 ? v0
                                 : v0 + (v1 - v0) * static_cast<float>(y) /
                                            static_cast<float>(height - 1);
            device::st_strided(output, idx, uu, is_half);
            device::st_strided(output, static_cast<long long>(total) + idx, vv, is_half);
        }
    }

    __global__ void residual_scale_kernel(const void* __restrict__ x, const void* __restrict__ hidden,
                                          const void* __restrict__ gamma, void* __restrict__ y,
                                          int rows, int cols, bool is_half) {
        const int total = rows * cols;
        for (int idx = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x); idx < total;
             idx += static_cast<int>(blockDim.x * gridDim.x)) {
            const int c = idx % cols;
            const float xv = device::ld_strided(x, idx, is_half);
            const float hv = device::ld_strided(hidden, idx, is_half);
            const float g = device::ld_strided(gamma, c, is_half);
            device::st_strided(y, idx, xv + hv * g, is_half);
        }
    }

    void split_qkv(const void* qkv, void* q, void* k, void* v, int batch, int seq, int heads,
                   int d, DataType dtype, cudaStream_t stream) {
        NvtxRange nvtx("nn.op/split_qkv");
        const int total = batch * seq * heads * d;
        if (total <= 0) {
            return;
        }
        const int block = 256;
        const int grid = std::min(2048, (total + block - 1) / block);
        split_qkv_kernel<<<grid, block, 0, stream>>>(qkv, q, k, v, batch, seq, heads, d,
                                                     dtype == DataType::Float16);
        LFS_CUDA_LAUNCH_CHECK(stream, "nn.split_qkv");
    }

    void merge_heads(const void* context, void* packed, int batch, int heads, int seq, int d,
                     DataType dtype, cudaStream_t stream) {
        NvtxRange nvtx("nn.op/merge_heads");
        const int total = batch * seq * heads * d;
        if (total <= 0) {
            return;
        }
        const int block = 256;
        const int grid = std::min(2048, (total + block - 1) / block);
        merge_heads_kernel<<<grid, block, 0, stream>>>(context, packed, batch, heads, seq, d,
                                                       dtype == DataType::Float16);
        LFS_CUDA_LAUNCH_CHECK(stream, "nn.merge_heads");
    }

    void uv_grid(void* output, int height, int width, float u0, float u1, float v0, float v1,
                 DataType dtype, cudaStream_t stream) {
        NvtxRange nvtx("nn.op/uv_grid");
        const int total = height * width;
        if (total <= 0) {
            return;
        }
        const int block = 256;
        const int grid = std::min(2048, (total + block - 1) / block);
        uv_grid_kernel<<<grid, block, 0, stream>>>(output, height, width, u0, u1, v0, v1,
                                                   dtype == DataType::Float16);
        LFS_CUDA_LAUNCH_CHECK(stream, "nn.uv_grid");
    }

    void residual_scale(const void* x, const void* hidden, const void* gamma, void* y, int rows,
                        int cols, DataType dtype, cudaStream_t stream) {
        NvtxRange nvtx("nn.op/residual_scale");
        const int total = rows * cols;
        if (total <= 0) {
            return;
        }
        const int block = 256;
        const int grid = std::min(2048, (total + block - 1) / block);
        residual_scale_kernel<<<grid, block, 0, stream>>>(x, hidden, gamma, y, rows, cols,
                                                          dtype == DataType::Float16);
        LFS_CUDA_LAUNCH_CHECK(stream, "nn.residual_scale");
    }

} // namespace lfs::core::nn::kernels

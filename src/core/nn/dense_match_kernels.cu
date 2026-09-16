/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "core/assert.hpp"
#include "core/cuda_error.hpp"
#include "nn_device.cuh"
#include "nn_kernels.hpp"
#include "nn_nvtx.hpp"

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>

namespace lfs::core::nn::kernels {
    namespace {

        constexpr float kPi = 3.14159265358979323846f;

        __device__ __forceinline__ float warp_reduce(float v) {
#pragma unroll
            for (int offset = 16; offset > 0; offset >>= 1) {
                v += __shfl_down_sync(0xffffffffu, v, offset);
            }
            return v;
        }

        // PyTorch grid_sample, align_corners=false: pixel centres at (i + 0.5).
        __device__ __forceinline__ float unnormalize(float coord, int size) {
            return ((coord + 1.0f) * static_cast<float>(size) - 1.0f) * 0.5f;
        }

        struct Bilinear {
            int x0;
            int y0;
            float wx1;
            float wy1;
        };

        __device__ __forceinline__ Bilinear bilinear_of(float gx, float gy, int w, int h) {
            const float fx = unnormalize(gx, w);
            const float fy = unnormalize(gy, h);
            const float x0f = floorf(fx);
            const float y0f = floorf(fy);
            Bilinear b;
            b.x0 = static_cast<int>(x0f);
            b.y0 = static_cast<int>(y0f);
            b.wx1 = fx - x0f;
            b.wy1 = fy - y0f;
            return b;
        }

        __device__ __forceinline__ float fetch_nchw(const void* src, int c, int y, int x, int h,
                                                    int w, int channels, bool is_half) {
            if (x < 0 || x >= w || y < 0 || y >= h) {
                return 0.0f;
            }
            const long long idx =
                (static_cast<long long>(c) * h + y) * w + x;
            return device::ld_strided(src, idx, is_half);
        }

        __global__ void grid_sample_bhwc_kernel(const void* __restrict__ input,
                                                const float* __restrict__ grid,
                                                void* __restrict__ output, int channels, int h_in,
                                                int w_in, int h_out, int w_out, bool is_half) {
            const int pixel = blockIdx.x;
            if (pixel >= h_out * w_out) {
                return;
            }
            const float gx = grid[2 * pixel];
            const float gy = grid[2 * pixel + 1];
            const Bilinear b = bilinear_of(gx, gy, w_in, h_in);
            const float wx0 = 1.0f - b.wx1;
            const float wy0 = 1.0f - b.wy1;
            const bool y0_ok = b.y0 >= 0 && b.y0 < h_in;
            const bool y1_ok = b.y0 + 1 >= 0 && b.y0 + 1 < h_in;
            const bool x0_ok = b.x0 >= 0 && b.x0 < w_in;
            const bool x1_ok = b.x0 + 1 >= 0 && b.x0 + 1 < w_in;
            const long long row0 = (static_cast<long long>(b.y0) * w_in + b.x0) * channels;
            const long long row_stride = static_cast<long long>(w_in) * channels;
            for (int c = threadIdx.x; c < channels; c += blockDim.x) {
                float v = 0.0f;
                if (y0_ok && x0_ok)
                    v += wy0 * wx0 * device::ld_strided(input, row0 + c, is_half);
                if (y0_ok && x1_ok)
                    v += wy0 * b.wx1 * device::ld_strided(input, row0 + channels + c, is_half);
                if (y1_ok && x0_ok)
                    v += b.wy1 * wx0 * device::ld_strided(input, row0 + row_stride + c, is_half);
                if (y1_ok && x1_ok)
                    v += b.wy1 * b.wx1 *
                         device::ld_strided(input, row0 + row_stride + channels + c, is_half);
                device::st_strided(output, static_cast<long long>(pixel) * channels + c, v,
                                   is_half);
            }
        }

        // Local correlation. The 2r+1 window offsets are exactly one pixel apart
        // in normalised coordinates, so all (2r+1)^2 samples share the same
        // bilinear weights: correlate f_a against the (2r+2)^2 integer-offset
        // neighbourhood once, then blend.
        __global__ void local_correlation_kernel(const void* __restrict__ f_a_nhwc,
                                                 const void* __restrict__ f_b_nhwc,
                                                 const float* __restrict__ warp,
                                                 void* __restrict__ out, int channels, int h, int w,
                                                 int radius, bool is_half) {
            extern __shared__ float smem[];
            const int warps_per_block = blockDim.y;
            const int lane = threadIdx.x;
            const int warp_id = threadIdx.y;
            const int pixel = blockIdx.x * warps_per_block + warp_id;
            const int span = 2 * radius + 2; // integer offsets -radius .. radius+1
            const int k_span = 2 * radius + 1;
            float* corr = smem + warp_id * span * span;
            if (pixel >= h * w) {
                return;
            }
            const float gx = warp[2 * pixel];
            const float gy = warp[2 * pixel + 1];
            const Bilinear b = bilinear_of(gx, gy, w, h);
            const long long a_base = static_cast<long long>(pixel) * channels;
            const float inv = rsqrtf(static_cast<float>(channels));

            for (int p = 0; p < span * span; ++p) {
                const int dy = p / span - radius;
                const int dx = p % span - radius;
                const int yy = b.y0 + dy;
                const int xx = b.x0 + dx;
                float acc = 0.0f;
                if (yy >= 0 && yy < h && xx >= 0 && xx < w) {
                    const long long base = (static_cast<long long>(yy) * w + xx) * channels;
                    for (int c = lane; c < channels; c += 32) {
                        acc += device::ld_strided(f_a_nhwc, a_base + c, is_half) *
                               device::ld_strided(f_b_nhwc, base + c, is_half);
                    }
                }
                acc = warp_reduce(acc);
                if (lane == 0) {
                    corr[p] = acc * inv;
                }
            }
            __syncwarp();
            const float wx0 = 1.0f - b.wx1;
            const float wy0 = 1.0f - b.wy1;
            for (int k = lane; k < k_span * k_span; k += 32) {
                const int ky = k / k_span;
                const int kx = k % k_span;
                const int p = ky * span + kx;
                const float v = (corr[p] * wx0 + corr[p + 1] * b.wx1) * wy0 +
                                (corr[p + span] * wx0 + corr[p + span + 1] * b.wx1) * b.wy1;
                device::st_strided(out, static_cast<long long>(pixel) * k_span * k_span + k, v,
                                   is_half);
            }
        }

        __global__ void normalized_grid_kernel(float* __restrict__ out, int h, int w) {
            const int idx = blockIdx.x * blockDim.x + threadIdx.x;
            if (idx >= h * w) {
                return;
            }
            const int y = idx / w;
            const int x = idx - y * w;
            out[2 * idx] = 2.0f * ((static_cast<float>(x) + 0.5f) / static_cast<float>(w)) - 1.0f;
            out[2 * idx + 1] = 2.0f * ((static_cast<float>(y) + 0.5f) / static_cast<float>(h)) - 1.0f;
        }

        __global__ void displacement_kernel(const float* __restrict__ warp, void* __restrict__ out,
                                            int h, int w, float sx, float sy, bool is_half) {
            const int idx = blockIdx.x * blockDim.x + threadIdx.x;
            if (idx >= h * w) {
                return;
            }
            const int y = idx / w;
            const int x = idx - y * w;
            const float gx = 2.0f * ((static_cast<float>(x) + 0.5f) / static_cast<float>(w)) - 1.0f;
            const float gy = 2.0f * ((static_cast<float>(y) + 0.5f) / static_cast<float>(h)) - 1.0f;
            device::st_strided(out, 2 * idx, (warp[2 * idx] - gx) * sx, is_half);
            device::st_strided(out, 2 * idx + 1, (warp[2 * idx + 1] - gy) * sy, is_half);
        }

        __device__ __forceinline__ float softplus(float x) {
            // log1p(exp(x)) with the standard large-x guard.
            return x > 20.0f ? x : log1pf(expf(x));
        }

        // Channel-last depthwise convolution. Each thread owns one (x, channel
        // pair) column and slides a 5x5 register window down `rows` outputs, so
        // steady state costs five coalesced half2 loads per output instead of
        // twenty-five. Weights are tap-major [K*K, C].
        template <int K, int ROWS>
        __global__ void depthwise_nhwc_kernel(const __half2* __restrict__ input,
                                              const __half2* __restrict__ weight,
                                              const __half2* __restrict__ bias,
                                              __half2* __restrict__ output, int pairs, int h, int w,
                                              bool relu) {
            constexpr int kPad = K / 2;
            const int c = blockIdx.x * blockDim.x + threadIdx.x;
            const int x = blockIdx.y * blockDim.y + threadIdx.y;
            const int y0 = blockIdx.z * ROWS;
            if (c >= pairs || x >= w) {
                return;
            }
            const long long row_stride = static_cast<long long>(w) * pairs;
            float2 wr[K * K];
#pragma unroll
            for (int t = 0; t < K * K; ++t) {
                wr[t] = __half22float2(weight[static_cast<long long>(t) * pairs + c]);
            }
            const float2 b = bias != nullptr ? __half22float2(bias[c]) : make_float2(0.0f, 0.0f);

            auto fetch = [&](int y, int x_off) -> float2 {
                const int sx = x + x_off;
                if (y < 0 || y >= h || sx < 0 || sx >= w) {
                    return make_float2(0.0f, 0.0f);
                }
                return __half22float2(
                    input[static_cast<long long>(y) * row_stride + static_cast<long long>(sx) * pairs + c]);
            };

            float2 win[K][K];
#pragma unroll
            for (int i = 0; i < K; ++i) {
#pragma unroll
                for (int j = 0; j < K; ++j) {
                    win[i][j] = fetch(y0 - kPad + i, j - kPad);
                }
            }
#pragma unroll 1
            for (int r = 0; r < ROWS; ++r) {
                const int y = y0 + r;
                if (y >= h) {
                    return;
                }
                float ax = b.x;
                float ay = b.y;
#pragma unroll
                for (int i = 0; i < K; ++i) {
#pragma unroll
                    for (int j = 0; j < K; ++j) {
                        const float2 t = win[i][j];
                        const float2 f = wr[i * K + j];
                        ax += t.x * f.x;
                        ay += t.y * f.y;
                    }
                }
                if (relu) {
                    ax = fmaxf(ax, 0.0f);
                    ay = fmaxf(ay, 0.0f);
                }
                output[static_cast<long long>(y) * row_stride + static_cast<long long>(x) * pairs + c] =
                    __floats2half2_rn(ax, ay);
#pragma unroll
                for (int i = 0; i < K - 1; ++i) {
#pragma unroll
                    for (int j = 0; j < K; ++j) {
                        win[i][j] = win[i + 1][j];
                    }
                }
#pragma unroll
                for (int j = 0; j < K; ++j) {
                    win[K - 1][j] = fetch(y + 1 + kPad, j - kPad);
                }
            }
        }

        // Separable antialiased bicubic resize, matching
        // torch.nn.functional.interpolate(mode="bicubic", antialias=True,
        // align_corners=False).
        __device__ __forceinline__ float cubic_weight(float x) {
            // PIL-compatible bicubic; torch's antialiased resize uses a = -0.5
            // here, not the -0.75 of its plain bicubic path.
            constexpr float a = -0.5f;
            x = fabsf(x);
            if (x < 1.0f) {
                return ((a + 2.0f) * x - (a + 3.0f)) * x * x + 1.0f;
            }
            if (x < 2.0f) {
                return ((x - 5.0f) * x + 8.0f) * x * a - 4.0f * a;
            }
            return 0.0f;
        }

        struct AaSpan {
            int lo;
            int hi;
            float centre;
            float inv_scale;
        };

        __device__ __forceinline__ AaSpan aa_span(int out_idx, int in_len, int out_len) {
            const float scale = static_cast<float>(in_len) / static_cast<float>(out_len);
            const float support = scale >= 1.0f ? 2.0f * scale : 2.0f;
            AaSpan s;
            s.inv_scale = scale >= 1.0f ? 1.0f / scale : 1.0f;
            s.centre = (static_cast<float>(out_idx) + 0.5f) * scale;
            s.lo = static_cast<int>(s.centre - support + 0.5f);
            s.lo = s.lo < 0 ? 0 : s.lo;
            s.hi = static_cast<int>(s.centre + support + 0.5f);
            s.hi = s.hi > in_len ? in_len : s.hi;
            return s;
        }

        // Horizontal pass straight off an interleaved HWC image, with the
        // uint8-to-unit-float conversion folded in: [in_h, in_w, C] (uint8 or
        // float32) -> [C, in_h, out_w] float32.
        __global__ void resize_aa_x_hwc_kernel(const void* __restrict__ input,
                                               float* __restrict__ output, int in_h, int in_w,
                                               int channels, int out_w, float gain, bool is_u8) {
            const int out_x = blockIdx.x * blockDim.x + threadIdx.x;
            const int y = blockIdx.y;
            const int c = blockIdx.z;
            if (out_x >= out_w || y >= in_h || c >= channels) {
                return;
            }
            const AaSpan s = aa_span(out_x, in_w, out_w);
            const long long row = static_cast<long long>(y) * in_w * channels + c;
            float total = 0.0f;
            float acc = 0.0f;
            for (int i = s.lo; i < s.hi; ++i) {
                const float wgt = cubic_weight((static_cast<float>(i) + 0.5f - s.centre) * s.inv_scale);
                const long long idx = row + static_cast<long long>(i) * channels;
                const float v = is_u8 ? static_cast<float>(static_cast<const unsigned char*>(input)[idx])
                                      : static_cast<const float*>(input)[idx];
                total += wgt;
                acc += wgt * v;
            }
            output[(static_cast<long long>(c) * in_h + y) * out_w + out_x] =
                total != 0.0f ? acc * gain / total : 0.0f;
        }

        // Horizontal pass: [planes, in_h, in_w] -> [planes, in_h, out_w].
        __global__ void resize_aa_x_kernel(const float* __restrict__ input,
                                           float* __restrict__ output, int rows, int in_w,
                                           int out_w) {
            const int out_x = blockIdx.x * blockDim.x + threadIdx.x;
            const int row = blockIdx.y;
            if (out_x >= out_w || row >= rows) {
                return;
            }
            const AaSpan s = aa_span(out_x, in_w, out_w);
            const float* src = input + static_cast<long long>(row) * in_w;
            float total = 0.0f;
            float acc = 0.0f;
            for (int i = s.lo; i < s.hi; ++i) {
                const float wgt = cubic_weight((static_cast<float>(i) + 0.5f - s.centre) * s.inv_scale);
                total += wgt;
                acc += wgt * src[i];
            }
            output[static_cast<long long>(row) * out_w + out_x] = total != 0.0f ? acc / total : 0.0f;
        }

        // Vertical pass: [planes, in_h, out_w] -> [planes, out_h, out_w].
        __global__ void resize_aa_y_kernel(const float* __restrict__ input,
                                           float* __restrict__ output, int planes, int in_h,
                                           int out_h, int out_w) {
            const int out_x = blockIdx.x * blockDim.x + threadIdx.x;
            const int out_y = blockIdx.y;
            const int plane = blockIdx.z;
            if (out_x >= out_w || out_y >= out_h || plane >= planes) {
                return;
            }
            const AaSpan s = aa_span(out_y, in_h, out_h);
            const float* src = input + static_cast<long long>(plane) * in_h * out_w + out_x;
            float total = 0.0f;
            float acc = 0.0f;
            for (int i = s.lo; i < s.hi; ++i) {
                const float wgt = cubic_weight((static_cast<float>(i) + 0.5f - s.centre) * s.inv_scale);
                total += wgt;
                acc += wgt * src[static_cast<long long>(i) * out_w];
            }
            output[(static_cast<long long>(plane) * out_h + out_y) * out_w + out_x] =
                total != 0.0f ? acc / total : 0.0f;
        }

        __global__ void transpose2d_kernel(const void* __restrict__ src, void* __restrict__ dst,
                                           int rows, int cols, bool is_half) {
            const long long idx = static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x;
            if (idx >= static_cast<long long>(rows) * cols) {
                return;
            }
            const int r = static_cast<int>(idx / cols);
            const int c = static_cast<int>(idx - static_cast<long long>(r) * cols);
            device::st_strided(dst, static_cast<long long>(c) * rows + r,
                               device::ld_strided(src, idx, is_half), is_half);
        }

        __global__ void normalize_image_kernel(const void* __restrict__ src, void* __restrict__ dst,
                                               int pixels, float m0, float m1, float m2, float s0,
                                               float s1, float s2, bool src_half, bool dst_half) {
            const int idx = blockIdx.x * blockDim.x + threadIdx.x;
            if (idx >= pixels) {
                return;
            }
            const float mean[3] = {m0, m1, m2};
            const float inv[3] = {1.0f / s0, 1.0f / s1, 1.0f / s2};
#pragma unroll
            for (int c = 0; c < 3; ++c) {
                const long long i = static_cast<long long>(c) * pixels + idx;
                device::st_strided(dst, i,
                                   (device::ld_strided(src, i, src_half) - mean[c]) * inv[c],
                                   dst_half);
            }
        }

        // ---- RoMa v1 ------------------------------------------------------

        // exp((cosine_similarity - 1) / temperature) in place on a Gram matrix
        // whose rows were produced from L2-normalised features.
        __global__ void cos_kernel_kernel(float* __restrict__ gram, long long count,
                                          float inv_temp) {
            const long long idx = static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x;
            if (idx >= count) {
                return;
            }
            gram[idx] = __expf((gram[idx] - 1.0f) * inv_temp);
        }

        // Adds sigma to the diagonal of an n x n matrix.
        __global__ void add_diagonal_kernel(float* __restrict__ matrix, int n, float sigma) {
            const int i = blockIdx.x * blockDim.x + threadIdx.x;
            if (i < n) {
                matrix[static_cast<long long>(i) * n + i] += sigma;
            }
        }

        // L2-normalises each row of [rows, cols] into a float32 destination.
        __global__ void l2_normalize_rows_kernel(const void* __restrict__ src,
                                                 float* __restrict__ dst, int rows, int cols,
                                                 bool is_half) {
            const int row = blockIdx.x;
            if (row >= rows) {
                return;
            }
            const long long base = static_cast<long long>(row) * cols;
            float acc = 0.0f;
            for (int c = threadIdx.x; c < cols; c += blockDim.x) {
                const float v = device::ld_strided(src, base + c, is_half);
                acc += v * v;
            }
            __shared__ float red[32];
            float w = acc;
#pragma unroll
            for (int offset = 16; offset > 0; offset >>= 1) {
                w += __shfl_xor_sync(0xffffffffu, w, offset);
            }
            if ((threadIdx.x & 31) == 0) {
                red[threadIdx.x / 32] = w;
            }
            __syncthreads();
            if (threadIdx.x < 32) {
                const int warps = (blockDim.x + 31) / 32;
                float v = threadIdx.x < warps ? red[threadIdx.x] : 0.0f;
#pragma unroll
                for (int offset = 16; offset > 0; offset >>= 1) {
                    v += __shfl_xor_sync(0xffffffffu, v, offset);
                }
                if (threadIdx.x == 0) {
                    red[0] = v;
                }
            }
            __syncthreads();
            const float inv = 1.0f / (sqrtf(red[0]) + 1e-6f);
            for (int c = threadIdx.x; c < cols; c += blockDim.x) {
                dst[base + c] = device::ld_strided(src, base + c, is_half) * inv;
            }
        }

        // Per-column dot products of two [rows, cols] float32 matrices.
        __global__ void column_dot_kernel(const float* __restrict__ a, const float* __restrict__ b,
                                          float* __restrict__ out, int rows, int cols) {
            const int col = blockIdx.x;
            if (col >= cols) {
                return;
            }
            float acc = 0.0f;
            for (int r = threadIdx.x; r < rows; r += blockDim.x) {
                const long long idx = static_cast<long long>(r) * cols + col;
                acc += a[idx] * b[idx];
            }
            __shared__ float red[32];
#pragma unroll
            for (int offset = 16; offset > 0; offset >>= 1) {
                acc += __shfl_xor_sync(0xffffffffu, acc, offset);
            }
            if ((threadIdx.x & 31) == 0) {
                red[threadIdx.x / 32] = acc;
            }
            __syncthreads();
            if (threadIdx.x == 0) {
                const int warps = (blockDim.x + 31) / 32;
                float v = 0.0f;
                for (int i = 0; i < warps; ++i) {
                    v += red[i];
                }
                out[col] = v;
            }
        }

        // One conjugate-gradient update pass over [rows, cols] with per-column
        // step sizes: x += alpha * p, r -= alpha * Ap.
        __global__ void cg_step_kernel(float* __restrict__ x, float* __restrict__ r,
                                       const float* __restrict__ p, const float* __restrict__ ap,
                                       const float* __restrict__ rs,
                                       const float* __restrict__ pap, int rows, int cols) {
            const long long idx = static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x;
            if (idx >= static_cast<long long>(rows) * cols) {
                return;
            }
            const int col = static_cast<int>(idx % cols);
            const float alpha = rs[col] / (pap[col] + 1e-30f);
            x[idx] += alpha * p[idx];
            r[idx] -= alpha * ap[idx];
        }

        // p = r + (rs_new / rs_old) * p.
        __global__ void cg_direction_kernel(float* __restrict__ p, const float* __restrict__ r,
                                            const float* __restrict__ rs_new,
                                            const float* __restrict__ rs_old, int rows, int cols) {
            const long long idx = static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x;
            if (idx >= static_cast<long long>(rows) * cols) {
                return;
            }
            const int col = static_cast<int>(idx % cols);
            const float beta = rs_new[col] / (rs_old[col] + 1e-30f);
            p[idx] = r[idx] + beta * p[idx];
        }

        // cos(8 * pi * x), the Fourier basis the Gaussian process regresses onto.
        __global__ void fourier_cos_kernel(const float* __restrict__ src, float* __restrict__ dst,
                                           long long count) {
            const long long idx = static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x;
            if (idx < count) {
                dst[idx] = __cosf(8.0f * 3.14159265358979323846f * src[idx]);
            }
        }

        // Soft-argmax of the coarse classifier: softmax over `classes`, then a
        // weighted mean of the peak and its four neighbours on the class grid.
        __global__ void cls_to_flow_kernel(const void* __restrict__ logits, float* __restrict__ flow,
                                           int pixels, int classes, int row_stride, int side,
                                           bool is_half) {
            const int pixel = blockIdx.x;
            if (pixel >= pixels) {
                return;
            }
            const long long base = static_cast<long long>(pixel) * row_stride;
            float best = -1e30f;
            int mode = 0;
            for (int c = threadIdx.x; c < classes; c += blockDim.x) {
                const float v = device::ld_strided(logits, base + c, is_half);
                if (v > best) {
                    best = v;
                    mode = c;
                }
            }
            __shared__ float red_val[32];
            __shared__ int red_idx[32];
            const int lane = threadIdx.x & 31;
#pragma unroll
            for (int offset = 16; offset > 0; offset >>= 1) {
                const float other = __shfl_xor_sync(0xffffffffu, best, offset);
                const int other_idx = __shfl_xor_sync(0xffffffffu, mode, offset);
                if (other > best || (other == best && other_idx < mode)) {
                    best = other;
                    mode = other_idx;
                }
            }
            if (lane == 0) {
                red_val[threadIdx.x / 32] = best;
                red_idx[threadIdx.x / 32] = mode;
            }
            __syncthreads();
            __shared__ float sum_exp;
            __shared__ int peak;
            if (threadIdx.x == 0) {
                const int warps = (blockDim.x + 31) / 32;
                float v = red_val[0];
                int m = red_idx[0];
                for (int i = 1; i < warps; ++i) {
                    if (red_val[i] > v || (red_val[i] == v && red_idx[i] < m)) {
                        v = red_val[i];
                        m = red_idx[i];
                    }
                }
                peak = m;
            }
            __syncthreads();
            // softmax denominator, shifted by the peak logit for stability
            const float peak_logit = device::ld_strided(logits, base + peak, is_half);
            float acc = 0.0f;
            for (int c = threadIdx.x; c < classes; c += blockDim.x) {
                acc += __expf(device::ld_strided(logits, base + c, is_half) - peak_logit);
            }
#pragma unroll
            for (int offset = 16; offset > 0; offset >>= 1) {
                acc += __shfl_xor_sync(0xffffffffu, acc, offset);
            }
            if (lane == 0) {
                red_val[threadIdx.x / 32] = acc;
            }
            __syncthreads();
            if (threadIdx.x == 0) {
                const int warps = (blockDim.x + 31) / 32;
                float v = 0.0f;
                for (int i = 0; i < warps; ++i) {
                    v += red_val[i];
                }
                sum_exp = v;
            }
            __syncthreads();
            if (threadIdx.x != 0) {
                return;
            }
            const int offsets[5] = {peak - 1, peak, peak + 1, peak - side, peak + side};
            float wx = 0.0f;
            float wy = 0.0f;
            float total = 0.0f;
            for (int i = 0; i < 5; ++i) {
                int c = offsets[i];
                c = c < 0 ? 0 : (c >= classes ? classes - 1 : c);
                const float prob =
                    __expf(device::ld_strided(logits, base + c, is_half) - peak_logit) / sum_exp;
                const int gy = c / side;
                const int gx = c - gy * side;
                wx += prob * (2.0f * ((static_cast<float>(gx) + 0.5f) / side) - 1.0f);
                wy += prob * (2.0f * ((static_cast<float>(gy) + 0.5f) / side) - 1.0f);
                total += prob;
            }
            flow[2 * pixel] = wx / total;
            flow[2 * pixel + 1] = wy / total;
        }

        // RoMa v1 refiner epilogue: the displacement is relative to the finest
        // grid, and the certainty accumulates across scales.
        __global__ void v1_refiner_update_kernel(const float* __restrict__ prev_flow,
                                                 const float* __restrict__ prev_cert,
                                                 const void* __restrict__ delta, float* __restrict__ flow,
                                                 float* __restrict__ cert, int pixels, float step,
                                                 bool is_half) {
            const int idx = blockIdx.x * blockDim.x + threadIdx.x;
            if (idx >= pixels) {
                return;
            }
            const long long base = static_cast<long long>(idx) * 3;
            flow[2 * idx] = prev_flow[2 * idx] + device::ld_strided(delta, base, is_half) * step;
            flow[2 * idx + 1] =
                prev_flow[2 * idx + 1] + device::ld_strided(delta, base + 1, is_half) * step;
            cert[idx] = prev_cert[idx] + device::ld_strided(delta, base + 2, is_half);
        }

        // certainty -= 0.5 * low where low is negative, then sigmoid.
        __global__ void attenuate_certainty_kernel(const float* __restrict__ cert,
                                                   const float* __restrict__ low,
                                                   float* __restrict__ out, int pixels) {
            const int idx = blockIdx.x * blockDim.x + threadIdx.x;
            if (idx >= pixels) {
                return;
            }
            const float l = low[idx];
            const float v = cert[idx] - (l < 0.0f ? 0.5f * l : 0.0f);
            out[idx] = 1.0f / (1.0f + __expf(-v));
        }

    } // namespace

    void grid_sample_bhwc(const void* input, const float* grid, void* output, int channels,
                          int h_in, int w_in, int h_out, int w_out, DataType dtype,
                          cudaStream_t stream) {
        NvtxRange nvtx("nn/grid_sample_bhwc");
        const int threads = std::min(256, ((channels + 31) / 32) * 32);
        grid_sample_bhwc_kernel<<<h_out * w_out, threads, 0, stream>>>(
            input, grid, output, channels, h_in, w_in, h_out, w_out, dtype == DataType::Float16);
        LFS_CUDA_CHECK(cudaGetLastError());
    }

    void local_correlation(const void* f_a_nhwc, const void* f_b_nhwc, const float* warp, void* out,
                           int channels, int h, int w, int radius, DataType dtype,
                           cudaStream_t stream) {
        NvtxRange nvtx("nn/local_correlation");
        const int span = 2 * radius + 2;
        const int warps = 4;
        const dim3 block(32, warps);
        const int blocks = (h * w + warps - 1) / warps;
        const std::size_t smem = static_cast<std::size_t>(warps) * span * span * sizeof(float);
        local_correlation_kernel<<<blocks, block, smem, stream>>>(
            f_a_nhwc, f_b_nhwc, warp, out, channels, h, w, radius, dtype == DataType::Float16);
        LFS_CUDA_CHECK(cudaGetLastError());
    }

    void normalized_grid(float* out, int h, int w, cudaStream_t stream) {
        const int threads = 256;
        normalized_grid_kernel<<<(h * w + threads - 1) / threads, threads, 0, stream>>>(out, h, w);
        LFS_CUDA_CHECK(cudaGetLastError());
    }

    void displacement_input(const float* warp, void* out, int h, int w, float sx, float sy,
                            DataType dtype, cudaStream_t stream) {
        const int threads = 256;
        displacement_kernel<<<(h * w + threads - 1) / threads, threads, 0, stream>>>(
            warp, out, h, w, sx, sy, dtype == DataType::Float16);
        LFS_CUDA_CHECK(cudaGetLastError());
    }

    void depthwise_conv2d_nhwc(const void* input, const void* weight, const void* bias,
                               void* output, int channels, int h, int w, int kernel, int activation,
                               DataType dtype, cudaStream_t stream) {
        NvtxRange nvtx("nn/depthwise_conv2d");
        LFS_ASSERT_MSG(dtype == DataType::Float16 && channels % 2 == 0,
                       "channel-last depthwise needs float16 with an even channel count");
        constexpr int kRows = 8;
        const int pairs = channels / 2;
        const int tx = std::min(pairs, 32);
        const int ty = std::max(1, 128 / tx);
        const dim3 block(tx, ty);
        const dim3 grid((pairs + tx - 1) / tx, (w + ty - 1) / ty, (h + kRows - 1) / kRows);
        const bool use_relu = activation == 1;
        const auto* in = static_cast<const __half2*>(input);
        const auto* wt = static_cast<const __half2*>(weight);
        const auto* bs = static_cast<const __half2*>(bias);
        auto* out = static_cast<__half2*>(output);
        if (kernel == 5) {
            depthwise_nhwc_kernel<5, kRows><<<grid, block, 0, stream>>>(in, wt, bs, out, pairs, h, w,
                                                                        use_relu);
            LFS_CUDA_CHECK(cudaGetLastError());
        } else if (kernel == 3) {
            depthwise_nhwc_kernel<3, kRows><<<grid, block, 0, stream>>>(in, wt, bs, out, pairs, h, w,
                                                                        use_relu);
            LFS_CUDA_CHECK(cudaGetLastError());
        }
    }

    void resize_bicubic_aa(const float* input, float* output, float* workspace, int n, int c,
                           int in_h, int in_w, int out_h, int out_w, cudaStream_t stream) {
        NvtxRange nvtx("nn/resize_bicubic_aa");
        const int planes = n * c;
        constexpr int kThreads = 128;
        {
            const dim3 grid((out_w + kThreads - 1) / kThreads, planes * in_h);
            resize_aa_x_kernel<<<grid, kThreads, 0, stream>>>(input, workspace, planes * in_h, in_w,
                                                              out_w);
            LFS_CUDA_CHECK(cudaGetLastError());
        }
        {
            const dim3 grid((out_w + kThreads - 1) / kThreads, out_h, planes);
            resize_aa_y_kernel<<<grid, kThreads, 0, stream>>>(workspace, output, planes, in_h,
                                                              out_h, out_w);
        }
        LFS_CUDA_CHECK(cudaGetLastError());
    }

    void resize_bicubic_aa_hwc(const void* input, float* output, float* workspace, int in_h,
                               int in_w, int channels, int out_h, int out_w, DataType src_dtype,
                               cudaStream_t stream) {
        NvtxRange nvtx("nn/resize_bicubic_aa_hwc");
        constexpr int kThreads = 128;
        const bool is_u8 = src_dtype == DataType::UInt8;
        const float gain = is_u8 ? 1.0f / 255.0f : 1.0f;
        {
            const dim3 grid((out_w + kThreads - 1) / kThreads, in_h, channels);
            resize_aa_x_hwc_kernel<<<grid, kThreads, 0, stream>>>(input, workspace, in_h, in_w,
                                                                  channels, out_w, gain, is_u8);
            LFS_CUDA_CHECK(cudaGetLastError());
        }
        {
            const dim3 grid((out_w + kThreads - 1) / kThreads, out_h, channels);
            resize_aa_y_kernel<<<grid, kThreads, 0, stream>>>(workspace, output, channels, in_h,
                                                              out_h, out_w);
        }
        LFS_CUDA_CHECK(cudaGetLastError());
    }

    void transpose2d(const void* src, void* dst, int rows, int cols, DataType dtype,
                     cudaStream_t stream) {
        const int threads = 256;
        const long long n = static_cast<long long>(rows) * cols;
        transpose2d_kernel<<<static_cast<int>((n + threads - 1) / threads), threads, 0, stream>>>(
            src, dst, rows, cols, dtype == DataType::Float16);
        LFS_CUDA_CHECK(cudaGetLastError());
    }

    void normalize_image(const void* src, void* dst, int pixels, const float* mean,
                         const float* stddev, DataType src_dtype, DataType dst_dtype,
                         cudaStream_t stream) {
        const int threads = 256;
        normalize_image_kernel<<<(pixels + threads - 1) / threads, threads, 0, stream>>>(
            src, dst, pixels, mean[0], mean[1], mean[2], stddev[0], stddev[1], stddev[2],
            src_dtype == DataType::Float16, dst_dtype == DataType::Float16);
        LFS_CUDA_CHECK(cudaGetLastError());
    }

    // ---- RoMa v1 ----------------------------------------------------------

    void l2_normalize_rows(const void* src, float* dst, int rows, int cols, DataType dtype,
                           cudaStream_t stream) {
        const int threads = std::min(256, ((cols + 31) / 32) * 32);
        l2_normalize_rows_kernel<<<rows, threads, 0, stream>>>(src, dst, rows, cols,
                                                               dtype == DataType::Float16);
        LFS_CUDA_CHECK(cudaGetLastError());
    }

    void cosine_kernel_inplace(float* gram, long long count, float temperature,
                               cudaStream_t stream) {
        const int threads = 256;
        cos_kernel_kernel<<<static_cast<int>((count + threads - 1) / threads), threads, 0, stream>>>(
            gram, count, 1.0f / temperature);
        LFS_CUDA_CHECK(cudaGetLastError());
    }

    void add_diagonal(float* matrix, int n, float sigma, cudaStream_t stream) {
        const int threads = 256;
        add_diagonal_kernel<<<(n + threads - 1) / threads, threads, 0, stream>>>(matrix, n, sigma);
        LFS_CUDA_CHECK(cudaGetLastError());
    }

    void column_dot(const float* a, const float* b, float* out, int rows, int cols,
                    cudaStream_t stream) {
        column_dot_kernel<<<cols, 256, 0, stream>>>(a, b, out, rows, cols);
        LFS_CUDA_CHECK(cudaGetLastError());
    }

    void cg_step(float* x, float* r, const float* p, const float* ap, const float* rs,
                 const float* pap, int rows, int cols, cudaStream_t stream) {
        const int threads = 256;
        const long long n = static_cast<long long>(rows) * cols;
        cg_step_kernel<<<static_cast<int>((n + threads - 1) / threads), threads, 0, stream>>>(
            x, r, p, ap, rs, pap, rows, cols);
        LFS_CUDA_CHECK(cudaGetLastError());
    }

    void cg_direction(float* p, const float* r, const float* rs_new, const float* rs_old, int rows,
                      int cols, cudaStream_t stream) {
        const int threads = 256;
        const long long n = static_cast<long long>(rows) * cols;
        cg_direction_kernel<<<static_cast<int>((n + threads - 1) / threads), threads, 0, stream>>>(
            p, r, rs_new, rs_old, rows, cols);
        LFS_CUDA_CHECK(cudaGetLastError());
    }

    void fourier_cos(const float* src, float* dst, long long count, cudaStream_t stream) {
        const int threads = 256;
        fourier_cos_kernel<<<static_cast<int>((count + threads - 1) / threads), threads, 0,
                             stream>>>(src, dst, count);
        LFS_CUDA_CHECK(cudaGetLastError());
    }

    void cls_to_flow(const void* logits, float* flow, int pixels, int classes, int row_stride,
                     int side, DataType dtype, cudaStream_t stream) {
        NvtxRange nvtx("nn/cls_to_flow");
        cls_to_flow_kernel<<<pixels, 256, 0, stream>>>(logits, flow, pixels, classes, row_stride,
                                                       side, dtype == DataType::Float16);
        LFS_CUDA_CHECK(cudaGetLastError());
    }

    void v1_refiner_update(const float* prev_flow, const float* prev_cert, const void* delta,
                           float* flow, float* cert, int pixels, float step, DataType dtype,
                           cudaStream_t stream) {
        const int threads = 256;
        v1_refiner_update_kernel<<<(pixels + threads - 1) / threads, threads, 0, stream>>>(
            prev_flow, prev_cert, delta, flow, cert, pixels, step, dtype == DataType::Float16);
        LFS_CUDA_CHECK(cudaGetLastError());
    }

    void attenuate_certainty(const float* cert, const float* low, float* out, int pixels,
                             cudaStream_t stream) {
        const int threads = 256;
        attenuate_certainty_kernel<<<(pixels + threads - 1) / threads, threads, 0, stream>>>(
            cert, low, out, pixels);
        LFS_CUDA_CHECK(cudaGetLastError());
    }

} // namespace lfs::core::nn::kernels

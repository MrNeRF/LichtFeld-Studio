/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "core/assert.hpp"
#include "core/cuda_error.hpp"
#include "nn_kernels.hpp"

#include <algorithm>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <mma.h>

namespace lfs::core::nn::kernels {
    namespace {

        constexpr int kRgbTileH = 8;
        constexpr int kRgbTileW = 16;
        constexpr int kRgbTileM = kRgbTileH * kRgbTileW;
        constexpr int kRgbHaloH = kRgbTileH + 2;
        constexpr int kRgbHaloW = kRgbTileW + 2;
        constexpr int kRgbCout = 64;
        constexpr int kRgbK = 27;
        constexpr int kRgbKPad = 32;
        constexpr int kRgbPitch = kRgbKPad + 8;
        constexpr int kRgbCPitch = kRgbTileM + 4;
        constexpr int kRgbThreads = 256;
        constexpr int kRgbTileHalves = ((3 * kRgbHaloH * kRgbHaloW + 15) / 16) * 16;
        constexpr int kRgbTileBytes =
            (kRgbTileHalves + kRgbTileM * kRgbPitch + kRgbCout * kRgbPitch) * 2;
        constexpr int kRgbCBytes = kRgbCout * kRgbCPitch * 4;
        constexpr int kRgbSmemBytes = kRgbTileBytes > kRgbCBytes ? kRgbTileBytes : kRgbCBytes;

        // First VGG layer with the input scaling folded into the gather: the fp32
        // image is scaled per channel and converted to fp16 exactly as the
        // unfused path does, then a K=27 (padded to 32) WMMA product yields all
        // 64 output channels of a 8x16 pixel tile. Memory-bound on the output, so
        // WMMA keeps it a single sm_70+ kernel.
        __global__ void __launch_bounds__(kRgbThreads)
            lpips_rgb_conv3x3_kernel(const float* __restrict__ X, const __half* __restrict__ W,
                                     const __half* __restrict__ bias, __half* __restrict__ Y,
                                     const float3 shift, const float3 scale,
                                     const bool official, const int h, const int w) {
            __shared__ __align__(128) unsigned char smem_raw[kRgbSmemBytes];
            auto* tile = reinterpret_cast<__half*>(smem_raw);
            auto* As = tile + kRgbTileHalves;
            auto* Bs = As + kRgbTileM * kRgbPitch;
            auto* Cs = reinterpret_cast<float*>(smem_raw);

            const int tid = static_cast<int>(threadIdx.x);
            const int ni = static_cast<int>(blockIdx.z);
            const int tiles_w = (w + kRgbTileW - 1) / kRgbTileW;
            const int oh0 = (static_cast<int>(blockIdx.y) / tiles_w) * kRgbTileH;
            const int ow0 = (static_cast<int>(blockIdx.y) % tiles_w) * kRgbTileW;
            const long long plane = static_cast<long long>(h) * w;
            const float* Xn = X + static_cast<long long>(ni) * 3 * plane;

            for (int i = tid; i < 3 * kRgbHaloH * kRgbHaloW; i += kRgbThreads) {
                const int c = i / (kRgbHaloH * kRgbHaloW);
                const int rem = i - c * (kRgbHaloH * kRgbHaloW);
                const int hr = rem / kRgbHaloW;
                const int hc = rem - hr * kRgbHaloW;
                const int ih = oh0 - 1 + hr;
                const int iw = ow0 - 1 + hc;
                float v = 0.0f;
                if (static_cast<unsigned>(ih) < static_cast<unsigned>(h) &&
                    static_cast<unsigned>(iw) < static_cast<unsigned>(w)) {
                    const float s = c == 0 ? shift.x : (c == 1 ? shift.y : shift.z);
                    const float d = c == 0 ? scale.x : (c == 1 ? scale.y : scale.z);
                    float x = Xn[c * plane + static_cast<long long>(ih) * w + iw];
                    if (official) {
                        x = x * 2.0f - 1.0f;
                    }
                    v = (x - s) / d;
                }
                tile[i] = __float2half_rn(v);
            }
            for (int i = tid; i < kRgbCout * kRgbKPad; i += kRgbThreads) {
                const int n = i / kRgbKPad;
                const int k = i - n * kRgbKPad;
                Bs[n * kRgbPitch + k] = k < kRgbK ? W[n * kRgbK + k] : __float2half_rn(0.0f);
            }
            __syncthreads();
            for (int i = tid; i < kRgbTileM * kRgbKPad; i += kRgbThreads) {
                const int m = i / kRgbKPad;
                const int k = i - m * kRgbKPad;
                __half v = __float2half_rn(0.0f);
                if (k < kRgbK) {
                    const int ic = k / 9;
                    const int kh = (k - ic * 9) / 3;
                    const int kw = k - ic * 9 - kh * 3;
                    const int r = m / kRgbTileW + kh;
                    const int c = m - (m / kRgbTileW) * kRgbTileW + kw;
                    v = tile[(ic * kRgbHaloH + r) * kRgbHaloW + c];
                }
                As[m * kRgbPitch + k] = v;
            }
            __syncthreads();

            using namespace nvcuda::wmma;
            const int warp = tid >> 5;
            const int warp_n = warp >> 2;
            const int warp_m = warp & 3;
            fragment<accumulator, 16, 16, 16, float> c_frag[2][2];
#pragma unroll
            for (int i = 0; i < 2; ++i) {
#pragma unroll
                for (int j = 0; j < 2; ++j) {
                    fill_fragment(c_frag[i][j], 0.0f);
                }
            }
#pragma unroll
            for (int kk = 0; kk < kRgbKPad; kk += 16) {
                fragment<matrix_a, 16, 16, 16, __half, row_major> a_frag[2];
                fragment<matrix_b, 16, 16, 16, __half, col_major> b_frag[2];
#pragma unroll
                for (int i = 0; i < 2; ++i) {
                    load_matrix_sync(a_frag[i], Bs + (warp_n * 32 + i * 16) * kRgbPitch + kk,
                                     kRgbPitch);
                    load_matrix_sync(b_frag[i], As + (warp_m * 32 + i * 16) * kRgbPitch + kk,
                                     kRgbPitch);
                }
#pragma unroll
                for (int i = 0; i < 2; ++i) {
#pragma unroll
                    for (int j = 0; j < 2; ++j) {
                        mma_sync(c_frag[i][j], a_frag[i], b_frag[j], c_frag[i][j]);
                    }
                }
            }
            __syncthreads();
#pragma unroll
            for (int i = 0; i < 2; ++i) {
#pragma unroll
                for (int j = 0; j < 2; ++j) {
                    store_matrix_sync(Cs + (warp_n * 32 + i * 16) * kRgbCPitch + warp_m * 32 + j * 16,
                                      c_frag[i][j], kRgbCPitch, mem_row_major);
                }
            }
            __syncthreads();

            __half* Yn = Y + static_cast<long long>(ni) * kRgbCout * plane;
            for (int i = tid; i < kRgbCout * kRgbTileM; i += kRgbThreads) {
                const int n = i / kRgbTileM;
                const int m = i - n * kRgbTileM;
                const int oh = oh0 + m / kRgbTileW;
                const int ow = ow0 + (m - (m / kRgbTileW) * kRgbTileW);
                if (oh < h && ow < w) {
                    const float v = fmaxf(Cs[n * kRgbCPitch + m] + __half2float(bias[n]), 0.0f);
                    Yn[n * plane + static_cast<long long>(oh) * w + ow] = __float2half_rn(v);
                }
            }
        }

        constexpr int kReduceWarps = 8;
        constexpr int kReduceThreads = kReduceWarps * 32;

        // One block covers a 32-column x 2-row strip of both feature maps for all
        // channels, warps splitting the channel range. Pass 1 accumulates the
        // per-pixel squared norms (and writes the 2x2 max pool), pass 2 re-reads
        // the strip from L2 for the lin-weighted normalised squared distance.
        template <bool kPool>
        __global__ void __launch_bounds__(kReduceThreads)
            lpips_pool_reduce_kernel(const __half* __restrict__ x, const __half* __restrict__ y,
                                     const __half* __restrict__ lin, float* __restrict__ result,
                                     __half* __restrict__ pooled_x, __half* __restrict__ pooled_y,
                                     const int channels, const int h, const int w,
                                     const int interior_y0, const int interior_y1,
                                     const int interior_x0, const int interior_x1,
                                     const float inv_count) {
            __shared__ float partial[kReduceWarps][4][32];
            __shared__ float block_score[kReduceWarps];
            const int tid = static_cast<int>(threadIdx.x);
            const int lane = tid & 31;
            const int warp = tid >> 5;
            const int ni = static_cast<int>(blockIdx.z);
            const int r = static_cast<int>(blockIdx.y);
            const int col = static_cast<int>(blockIdx.x) * 32 + lane;
            const int row0 = 2 * r;
            const bool col_ok = col < w;
            const bool row1_ok = row0 + 1 < h;
            const bool score0 = row0 >= interior_y0 && row0 < interior_y1 && col >= interior_x0 &&
                                col < interior_x1;
            const bool score1 = row0 + 1 >= interior_y0 && row0 + 1 < interior_y1 &&
                                col >= interior_x0 && col < interior_x1;
            const long long plane = static_cast<long long>(h) * w;
            const __half* xb = x + static_cast<long long>(ni) * channels * plane;
            const __half* yb = y + static_cast<long long>(ni) * channels * plane;
            const int c_begin = warp * channels / kReduceWarps;
            const int c_end = (warp + 1) * channels / kReduceWarps;
            const int out_h = h / 2;
            const int out_w = w / 2;
            const bool pool_ok = kPool && (lane & 1) == 0 && col + 1 < w && row1_ok;

            float sx0 = 0.0f, sx1 = 0.0f, sy0 = 0.0f, sy1 = 0.0f;
            for (int c = c_begin; c < c_end; ++c) {
                const long long base = c * plane + static_cast<long long>(row0) * w + col;
                const float x0 = col_ok ? __half2float(xb[base]) : 0.0f;
                const float x1 = (col_ok && row1_ok) ? __half2float(xb[base + w]) : 0.0f;
                const float y0 = col_ok ? __half2float(yb[base]) : 0.0f;
                const float y1 = (col_ok && row1_ok) ? __half2float(yb[base + w]) : 0.0f;
                sx0 += x0 * x0;
                sx1 += x1 * x1;
                sy0 += y0 * y0;
                sy1 += y1 * y1;
                if constexpr (kPool) {
                    float mx = fmaxf(x0, x1);
                    float my = fmaxf(y0, y1);
                    mx = fmaxf(mx, __shfl_xor_sync(0xffffffffu, mx, 1));
                    my = fmaxf(my, __shfl_xor_sync(0xffffffffu, my, 1));
                    if (pool_ok) {
                        const long long o = (static_cast<long long>(ni) * channels + c) * out_h * out_w +
                                            static_cast<long long>(r) * out_w + (col >> 1);
                        pooled_x[o] = __float2half_rn(mx);
                        pooled_y[o] = __float2half_rn(my);
                    }
                }
            }
            partial[warp][0][lane] = sx0;
            partial[warp][1][lane] = sx1;
            partial[warp][2][lane] = sy0;
            partial[warp][3][lane] = sy1;
            __syncthreads();
            float tx0 = 0.0f, tx1 = 0.0f, ty0 = 0.0f, ty1 = 0.0f;
#pragma unroll
            for (int k = 0; k < kReduceWarps; ++k) {
                tx0 += partial[k][0][lane];
                tx1 += partial[k][1][lane];
                ty0 += partial[k][2][lane];
                ty1 += partial[k][3][lane];
            }
            const float ix0 = rsqrtf(tx0 + 1.0e-10f);
            const float ix1 = rsqrtf(tx1 + 1.0e-10f);
            const float iy0 = rsqrtf(ty0 + 1.0e-10f);
            const float iy1 = rsqrtf(ty1 + 1.0e-10f);

            float score = 0.0f;
            for (int c = c_begin; c < c_end; ++c) {
                const long long base = c * plane + static_cast<long long>(row0) * w + col;
                const float x0 = col_ok ? __half2float(xb[base]) : 0.0f;
                const float x1 = (col_ok && row1_ok) ? __half2float(xb[base + w]) : 0.0f;
                const float y0 = col_ok ? __half2float(yb[base]) : 0.0f;
                const float y1 = (col_ok && row1_ok) ? __half2float(yb[base + w]) : 0.0f;
                const float wv = __half2float(lin[c]);
                const float d0 = x0 * ix0 - y0 * iy0;
                const float d1 = x1 * ix1 - y1 * iy1;
                if (score0)
                    score += wv * d0 * d0;
                if (score1)
                    score += wv * d1 * d1;
            }
#pragma unroll
            for (int offset = 16; offset > 0; offset >>= 1) {
                score += __shfl_xor_sync(0xffffffffu, score, offset);
            }
            if (lane == 0) {
                block_score[warp] = score;
            }
            __syncthreads();
            if (tid == 0) {
                float total = 0.0f;
#pragma unroll
                for (int k = 0; k < kReduceWarps; ++k) {
                    total += block_score[k];
                }
                atomicAdd(result, total * inv_count);
            }
        }

    } // namespace

    void lpips_rgb_conv3x3(const float* input, const void* weight, const void* bias, void* output,
                           const float* shift, const float* scale, const bool official_scaling,
                           const int n, const int h, const int w, const cudaStream_t stream) {
        if (n <= 0 || h <= 0 || w <= 0) {
            return;
        }
        const int tiles = ((w + kRgbTileW - 1) / kRgbTileW) * ((h + kRgbTileH - 1) / kRgbTileH);
        dim3 grid(1, tiles, n);
        lpips_rgb_conv3x3_kernel<<<grid, kRgbThreads, 0, stream>>>(
            input, static_cast<const __half*>(weight), static_cast<const __half*>(bias),
            static_cast<__half*>(output), make_float3(shift[0], shift[1], shift[2]),
            make_float3(scale[0], scale[1], scale[2]), official_scaling, h, w);
        LFS_CUDA_LAUNCH_CHECK(stream, "nn.lpips.rgb_conv3x3");
    }

    void lpips_pool_reduce(const void* x, const void* y, const void* lin_weight, float* result,
                           void* pooled_x, void* pooled_y, const int n, const int channels,
                           const int h, const int w, const int interior_y0, const int interior_y1,
                           const int interior_x0, const int interior_x1, const float inv_count,
                           const cudaStream_t stream) {
        if (n <= 0 || channels <= 0 || h <= 0 || w <= 0) {
            return;
        }
        LFS_ASSERT_MSG(channels % kReduceWarps == 0, "lpips_pool_reduce needs channels % 8 == 0");
        dim3 grid((w + 31) / 32, (h + 1) / 2, n);
        auto* xs = static_cast<const __half*>(x);
        auto* ys = static_cast<const __half*>(y);
        auto* lin = static_cast<const __half*>(lin_weight);
        if (pooled_x != nullptr && pooled_y != nullptr) {
            lpips_pool_reduce_kernel<true><<<grid, kReduceThreads, 0, stream>>>(
                xs, ys, lin, result, static_cast<__half*>(pooled_x), static_cast<__half*>(pooled_y),
                channels, h, w, interior_y0, interior_y1, interior_x0, interior_x1, inv_count);
            LFS_CUDA_LAUNCH_CHECK(stream, "nn.lpips.pool_reduce");
        } else {
            lpips_pool_reduce_kernel<false><<<grid, kReduceThreads, 0, stream>>>(
                xs, ys, lin, result, nullptr, nullptr, channels, h, w, interior_y0, interior_y1,
                interior_x0, interior_x1, inv_count);
            LFS_CUDA_LAUNCH_CHECK(stream, "nn.lpips.pool_reduce");
        }
    }

} // namespace lfs::core::nn::kernels

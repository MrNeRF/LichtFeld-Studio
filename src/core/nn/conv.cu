/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "core/assert.hpp"
#include "core/cuda_error.hpp"
#include "core/logger.hpp"
#include "nn_device.cuh"
#include "nn_kernels.hpp"
#include "nn_nvtx.hpp"

#include <algorithm>
#include <atomic>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <float.h>
#include <mma.h>
#include <mutex>

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

        __device__ __forceinline__ __half load_im2col_nchw(const __half* X, int ni, int cin,
                                                           int h, int w, int m_idx, int k_idx,
                                                           int out_w, int pad_h, int pad_w,
                                                           int pad_mode, int spatial, int kdim) {
            if (m_idx >= spatial || k_idx >= kdim) {
                return __float2half(0.0f);
            }
            const int ow = m_idx % out_w;
            const int oh = m_idx / out_w;
            const int kw = k_idx % 3;
            const int kh = (k_idx / 3) % 3;
            const int ic = k_idx / 9;
            int ih = oh - pad_h + kh;
            int iw = ow - pad_w + kw;
            if (pad_mode == 1) {
                ih = ih < 0 ? 0 : (ih >= h ? h - 1 : ih);
                iw = iw < 0 ? 0 : (iw >= w ? w - 1 : iw);
            }
            if (static_cast<unsigned>(ih) >= static_cast<unsigned>(h) ||
                static_cast<unsigned>(iw) >= static_cast<unsigned>(w) ||
                static_cast<unsigned>(ic) >= static_cast<unsigned>(cin)) {
                return __float2half(0.0f);
            }
            const long long in_i =
                ((static_cast<long long>(ni) * cin + ic) * h + ih) * w + iw;
            return X[in_i];
        }

        __device__ __forceinline__ __half load_nchw_pad(const __half* X, int ni, int cin, int h,
                                                        int w, int ic, int ih, int iw, int pad_mode) {
            if (pad_mode == 1) {
                ih = ih < 0 ? 0 : (ih >= h ? h - 1 : ih);
                iw = iw < 0 ? 0 : (iw >= w ? w - 1 : iw);
            }
            if (static_cast<unsigned>(ih) >= static_cast<unsigned>(h) ||
                static_cast<unsigned>(iw) >= static_cast<unsigned>(w) ||
                static_cast<unsigned>(ic) >= static_cast<unsigned>(cin)) {
                return __float2half(0.0f);
            }
            return X[((static_cast<long long>(ni) * cin + ic) * h + ih) * w + iw];
        }

        __device__ __forceinline__ float load_im2col_nchw_f32(
            const float* X, int ni, int cin, int h, int w, int m_idx, int k_idx,
            int out_w, int pad_h, int pad_w, int pad_mode, int spatial, int kdim) {
            if (m_idx >= spatial || k_idx >= kdim) {
                return 0.0f;
            }
            const int ow = m_idx % out_w;
            const int oh = m_idx / out_w;
            const int kw = k_idx % 3;
            const int kh = (k_idx / 3) % 3;
            const int ic = k_idx / 9;
            int ih = oh - pad_h + kh;
            int iw = ow - pad_w + kw;
            if (pad_mode == 1) {
                ih = ih < 0 ? 0 : (ih >= h ? h - 1 : ih);
                iw = iw < 0 ? 0 : (iw >= w ? w - 1 : iw);
            }
            if (static_cast<unsigned>(ih) >= static_cast<unsigned>(h) ||
                static_cast<unsigned>(iw) >= static_cast<unsigned>(w) ||
                static_cast<unsigned>(ic) >= static_cast<unsigned>(cin)) {
                return 0.0f;
            }
            return X[((static_cast<long long>(ni) * cin + ic) * h + ih) * w + iw];
        }

        template <bool kHalf>
        __device__ __forceinline__ float load_simt_3x3(const void* X, int ni, int cin, int h,
                                                       int w, int m_idx, int k_idx, int out_w,
                                                       int pad_h, int pad_w, int pad_mode,
                                                       int spatial, int kdim) {
            if constexpr (kHalf) {
                return __half2float(load_im2col_nchw(
                    static_cast<const __half*>(X), ni, cin, h, w, m_idx, k_idx, out_w,
                    pad_h, pad_w, pad_mode, spatial, kdim));
            } else {
                return load_im2col_nchw_f32(
                    static_cast<const float*>(X), ni, cin, h, w, m_idx, k_idx, out_w,
                    pad_h, pad_w, pad_mode, spatial, kdim);
            }
        }

        template <bool kHalf>
        __device__ __forceinline__ float load_simt_weight(const void* W, long long index) {
            if constexpr (kHalf) {
                return __half2float(static_cast<const __half*>(W)[index]);
            } else {
                return static_cast<const float*>(W)[index];
            }
        }

        template <bool kHalf>
        __device__ __forceinline__ void store_simt_output(void* Y, long long index, float value) {
            if constexpr (kHalf) {
                static_cast<__half*>(Y)[index] = __float2half_rn(value);
            } else {
                static_cast<float*>(Y)[index] = value;
            }
        }

        // SIMT implicit GEMM for 3x3 NCHW convolution. Each 256-thread block
        // computes a 64x64 output tile with a 4x4 register tile per thread.
        // The A operand is gathered directly from NCHW and no im2col workspace is used.
        template <bool kHalf>
        __global__ __launch_bounds__(256) void conv2d_implicit_simt_kernel(
            const void* __restrict__ X, const void* __restrict__ W,
            const void* __restrict__ bias, void* __restrict__ Y,
            int n, int cin, int h, int w, int cout, int out_h, int out_w,
            int pad_h, int pad_w, int pad_mode, int activation) {
            constexpr int BM = 64;
            constexpr int BN = 64;
            constexpr int BK = 8;
            const int tid = static_cast<int>(threadIdx.x);
            const int ni = static_cast<int>(blockIdx.z);
            const int block_row = static_cast<int>(blockIdx.y) * BM;
            const int block_col = static_cast<int>(blockIdx.x) * BN;
            const int row0 = (tid / 16) * 4;
            const int col0 = (tid % 16) * 4;
            const int spatial = out_h * out_w;
            const int kdim = cin * 9;
            __shared__ float As[BM][BK];
            __shared__ float Bs[BK][BN];
            float acc[4][4] = {};

            for (int k0 = 0; k0 < kdim; k0 += BK) {
                for (int index = tid; index < BM * BK; index += 256) {
                    const int r = index / BK;
                    const int k = index % BK;
                    As[r][k] = load_simt_3x3<kHalf>(
                        X, ni, cin, h, w, block_row + r, k0 + k, out_w,
                        pad_h, pad_w, pad_mode, spatial, kdim);
                }
                for (int index = tid; index < BK * BN; index += 256) {
                    const int k = index / BN;
                    const int c = index % BN;
                    const int gc = block_col + c;
                    Bs[k][c] = gc < cout && k0 + k < kdim
                                   ? load_simt_weight<kHalf>(W, static_cast<long long>(gc) * kdim + k0 + k)
                                   : 0.0f;
                }
                __syncthreads();
#pragma unroll
                for (int k = 0; k < BK; ++k) {
#pragma unroll
                    for (int r = 0; r < 4; ++r) {
#pragma unroll
                        for (int c = 0; c < 4; ++c) {
                            acc[r][c] += As[row0 + r][k] * Bs[k][col0 + c];
                        }
                    }
                }
                __syncthreads();
            }

#pragma unroll
            for (int r = 0; r < 4; ++r) {
#pragma unroll
                for (int c = 0; c < 4; ++c) {
                    const int gr = block_row + row0 + r;
                    const int gc = block_col + col0 + c;
                    if (ni < n && gr < spatial && gc < cout) {
                        float value = acc[r][c];
                        if (bias)
                            value += load_simt_weight<kHalf>(bias, gc);
                        value = device::apply_activation(value, activation);
                        store_simt_output<kHalf>(
                            Y, static_cast<long long>(ni) * cout * spatial + static_cast<long long>(gc) * spatial + gr,
                            value);
                    }
                }
            }
        }

        // 128xBN implicit 3x3. Contiguous row tiles keep a 3x130x32 halo in smem
        // so each IC chunk is loaded once and reused across the 9 spatial offsets.
        // BN=128 halves halo refetches on wide (Cout>=128) layers.
        template <int BN>
        __global__ void __launch_bounds__(256, 2)
            hgemm_implicit_3x3_halo_kernel(const __half* __restrict__ X, const __half* __restrict__ W,
                                           __half* __restrict__ Y, const __half* __restrict__ bias,
                                           int n, int cin, int h, int w, int cout, int out_h,
                                           int out_w, int pad_h, int pad_w, int pad_mode,
                                           int activation) {
            constexpr int BM = 128;
            constexpr int BK = 32;
            constexpr int kNFrags = BN / 16;
            const int ni = static_cast<int>(blockIdx.z);
            if (ni >= n) {
                return;
            }
            const int spatial = out_h * out_w;
            const int kdim = cin * 9;
            const int tid = static_cast<int>(threadIdx.x);
            const int n_tw = (out_w + BM - 1) / BM;
            const int tile = static_cast<int>(blockIdx.y);
            const int oh0 = tile / n_tw;
            const int tw = tile % n_tw;
            const int ow0 = tw * BM;
            const int block_col = static_cast<int>(blockIdx.x) * BN;
            if (oh0 >= out_h) {
                return;
            }
            Y += static_cast<long long>(ni) * cout * spatial;
            const int tile_m = out_w - ow0 < BM ? out_w - ow0 : BM;

#if __CUDA_ARCH__ >= 700
            using namespace nvcuda::wmma;
            constexpr int WM = 16;
            constexpr int WN = 16;
            constexpr int WK = 16;
            const int warp_id = tid / 32;
            const int warp_m = warp_id / 2;
            const int warp_n = warp_id % 2;

            __shared__ __align__(16) __half Bs[BN][BK];
            __shared__ __align__(16) __half Xs[3][BM + 2][BK];

            fragment<matrix_a, WM, WN, WK, __half, row_major> a_frag[2];
            fragment<matrix_b, WM, WN, WK, __half, col_major> b_frag[2];
            fragment<accumulator, WM, WN, WK, float> c_frag[2][kNFrags / 2];
#pragma unroll
            for (int i = 0; i < 2; ++i) {
#pragma unroll
                for (int j = 0; j < kNFrags / 2; ++j) {
                    fill_fragment(c_frag[i][j], 0.0f);
                }
            }

            for (int ic0 = 0; ic0 < cin; ic0 += BK) {
                for (int i = tid; i < 3 * 2 * BK; i += 256) {
                    const int ic_l = i % BK;
                    const int edge = (i / BK) % 2;
                    const int rh = i / (BK * 2);
                    const int iw_l = edge ? (BM + 1) : 0;
                    Xs[rh][iw_l][ic_l] = load_nchw_pad(X, ni, cin, h, w, ic0 + ic_l,
                                                       oh0 - 1 + rh, ow0 - 1 + iw_l, pad_mode);
                }
                constexpr int nvec = BM / 8;
                for (int i = tid; i < 3 * BK * nvec; i += 256) {
                    const int vec = i % nvec;
                    const int ic_l = (i / nvec) % BK;
                    const int rh = i / (nvec * BK);
                    const int iw0 = vec * 8;
                    const int ic = ic0 + ic_l;
                    int yh = oh0 - 1 + rh;
                    int xw = ow0 + iw0;
                    if (pad_mode == 1) {
                        yh = yh < 0 ? 0 : (yh >= h ? h - 1 : yh);
                    }
                    if (static_cast<unsigned>(ic) < static_cast<unsigned>(cin) &&
                        static_cast<unsigned>(yh) < static_cast<unsigned>(h) &&
                        static_cast<unsigned>(xw) < static_cast<unsigned>(w) && xw + 7 < w) {
                        const __half* src =
                            X + ((static_cast<long long>(ni) * cin + ic) * h + yh) * w + xw;
                        if ((reinterpret_cast<uintptr_t>(src) & 15u) == 0) {
                            const uint4 packed = *reinterpret_cast<const uint4*>(src);
                            const auto* hv = reinterpret_cast<const __half*>(&packed);
#pragma unroll
                            for (int t = 0; t < 8; ++t) {
                                Xs[rh][1 + iw0 + t][ic_l] = hv[t];
                            }
                        } else {
#pragma unroll
                            for (int t = 0; t < 8; ++t) {
                                Xs[rh][1 + iw0 + t][ic_l] = src[t];
                            }
                        }
                    } else {
#pragma unroll
                        for (int t = 0; t < 8; ++t) {
                            Xs[rh][1 + iw0 + t][ic_l] = load_nchw_pad(
                                X, ni, cin, h, w, ic, oh0 - 1 + rh, ow0 + iw0 + t, pad_mode);
                        }
                    }
                }
                __syncthreads();
#pragma unroll
                for (int kh = 0; kh < 3; ++kh) {
#pragma unroll
                    for (int kw = 0; kw < 3; ++kw) {
                        for (int i = tid; i < BN * BK; i += 256) {
                            const int r = i / BK;
                            const int c = i % BK;
                            const int gc = block_col + r;
                            const int ic = ic0 + c;
                            __half v = __float2half(0.0f);
                            if (gc < cout && ic < cin) {
                                v = W[static_cast<long long>(gc) * kdim + ic * 9 + kh * 3 + kw];
                            }
                            Bs[r][c] = v;
                        }
                        __syncthreads();
                        const int am = warp_m * 32;
                        const int bn = warp_n * (BN / 2);
#pragma unroll
                        for (int kk = 0; kk < BK; kk += WK) {
                            load_matrix_sync(a_frag[0], &Xs[kh][kw + am][kk], BK);
                            load_matrix_sync(a_frag[1], &Xs[kh][kw + am + 16][kk], BK);
#pragma unroll
                            for (int fj = 0; fj < kNFrags / 2; fj += 2) {
                                load_matrix_sync(b_frag[0], &Bs[bn + fj * 16][kk], BK);
                                load_matrix_sync(b_frag[1], &Bs[bn + fj * 16 + 16][kk], BK);
                                mma_sync(c_frag[0][fj], a_frag[0], b_frag[0], c_frag[0][fj]);
                                mma_sync(c_frag[0][fj + 1], a_frag[0], b_frag[1], c_frag[0][fj + 1]);
                                mma_sync(c_frag[1][fj], a_frag[1], b_frag[0], c_frag[1][fj]);
                                mma_sync(c_frag[1][fj + 1], a_frag[1], b_frag[1], c_frag[1][fj + 1]);
                            }
                        }
                        __syncthreads();
                    }
                }
            }

            const int am = warp_m * 32;
            const int bn = warp_n * (BN / 2);
            const int lane = tid % 32;
            const int base_row = lane / 4;
            const int base_col = (lane % 4) * 2;
#pragma unroll
            for (int fi = 0; fi < 2; ++fi) {
#pragma unroll
                for (int fj = 0; fj < kNFrags / 2; ++fj) {
#pragma unroll
                    for (int i = 0; i < 8; ++i) {
                        const int r = am + fi * 16 + base_row + ((i >> 1) & 1) * 8;
                        const int c = bn + fj * 16 + base_col + (i & 1) + ((i & 4) ? 8 : 0);
                        const int gc = block_col + c;
                        if (r < tile_m && gc < cout) {
                            const int gr = oh0 * out_w + ow0 + r;
                            float v = c_frag[fi][fj].x[i];
                            if (bias) {
                                v += __half2float(bias[gc]);
                            }
                            v = device::apply_activation(v, activation);
                            Y[static_cast<long long>(gc) * spatial + gr] = __float2half_rn(v);
                        }
                    }
                }
            }
#else
            (void)oh0;
            (void)ow0;
            (void)tile_m;
            (void)kdim;
            (void)pad_h;
            (void)pad_w;
            (void)pad_mode;
            (void)spatial;
            (void)kNFrags;
            (void)activation;
#endif
        }

        // Implicit 3x3 GEMM. M = OH*OW, N = Cout, K = Cin*9. NT WMMA, NCHW store.
        template <int BM, int BN, int nthreads>
        __global__ void __launch_bounds__(nthreads)
            hgemm_implicit_3x3_kernel(const __half* __restrict__ X, const __half* __restrict__ W,
                                      __half* __restrict__ Y, const __half* __restrict__ bias,
                                      int n, int cin, int h, int w, int cout, int out_h, int out_w,
                                      int pad_h, int pad_w, int pad_mode, int activation) {
            const int ni = static_cast<int>(blockIdx.z);
            if (ni >= n) {
                return;
            }
            const int spatial = out_h * out_w;
            const int kdim = cin * 9;
            const int tid = static_cast<int>(threadIdx.x);
            const int block_row = static_cast<int>(blockIdx.y) * BM;
            const int block_col = static_cast<int>(blockIdx.x) * BN;
            Y += static_cast<long long>(ni) * cout * spatial;

#if __CUDA_ARCH__ >= 700
            using namespace nvcuda::wmma;
            constexpr int BK = 32;
            constexpr int WM = 16;
            constexpr int WN = 16;
            constexpr int WK = 16;

            const int warp_id = tid / 32;
            const int warp_m = warp_id / 2;
            const int warp_n = warp_id % 2;

            __shared__ __half As[BM][BK];
            __shared__ __half Bs[BN][BK];
            __shared__ float Cs[BM][BN];

            fragment<matrix_a, WM, WN, WK, __half, row_major> a_frag[2];
            fragment<matrix_b, WM, WN, WK, __half, col_major> b_frag[2];
            fragment<accumulator, WM, WN, WK, float> c_frag[2][2];
#pragma unroll
            for (int i = 0; i < 2; ++i) {
#pragma unroll
                for (int j = 0; j < 2; ++j) {
                    fill_fragment(c_frag[i][j], 0.0f);
                }
            }

            for (int k0 = 0; k0 < kdim; k0 += BK) {
                for (int i = tid; i < BM * BK; i += nthreads) {
                    const int r = i / BK;
                    const int c = i % BK;
                    As[r][c] = load_im2col_nchw(X, ni, cin, h, w, block_row + r, k0 + c, out_w,
                                                pad_h, pad_w, pad_mode, spatial, kdim);
                }
                for (int i = tid; i < BN * BK; i += nthreads) {
                    const int r = i / BK;
                    const int c = i % BK;
                    const int gc = block_col + r;
                    const int kk = k0 + c;
                    __half v = __float2half(0.0f);
                    if (gc < cout && kk < kdim) {
                        v = W[static_cast<long long>(gc) * kdim + kk];
                    }
                    Bs[r][c] = v;
                }
                __syncthreads();

#pragma unroll
                for (int kk = 0; kk < BK; kk += WK) {
                    const int am = warp_m * 32;
                    const int bn = warp_n * 32;
                    load_matrix_sync(a_frag[0], &As[am][kk], BK);
                    load_matrix_sync(a_frag[1], &As[am + 16][kk], BK);
                    load_matrix_sync(b_frag[0], &Bs[bn][kk], BK);
                    load_matrix_sync(b_frag[1], &Bs[bn + 16][kk], BK);
                    mma_sync(c_frag[0][0], a_frag[0], b_frag[0], c_frag[0][0]);
                    mma_sync(c_frag[0][1], a_frag[0], b_frag[1], c_frag[0][1]);
                    mma_sync(c_frag[1][0], a_frag[1], b_frag[0], c_frag[1][0]);
                    mma_sync(c_frag[1][1], a_frag[1], b_frag[1], c_frag[1][1]);
                }
                __syncthreads();
            }

            const int am = warp_m * 32;
            const int bn = warp_n * 32;
            store_matrix_sync(&Cs[am][bn], c_frag[0][0], BN, mem_row_major);
            store_matrix_sync(&Cs[am][bn + 16], c_frag[0][1], BN, mem_row_major);
            store_matrix_sync(&Cs[am + 16][bn], c_frag[1][0], BN, mem_row_major);
            store_matrix_sync(&Cs[am + 16][bn + 16], c_frag[1][1], BN, mem_row_major);
            __syncthreads();

            for (int i = tid; i < BM * BN; i += nthreads) {
                const int c = i / BM;
                const int r = i % BM;
                const int gr = block_row + r;
                const int gc = block_col + c;
                if (gr < spatial && gc < cout) {
                    float v = Cs[r][c];
                    if (bias) {
                        v += __half2float(bias[gc]);
                    }
                    v = device::apply_activation(v, activation);
                    Y[static_cast<long long>(gc) * spatial + gr] = __float2half_rn(v);
                }
            }
#else
            for (int i = tid; i < BM * BN; i += nthreads) {
                const int r = i / BN;
                const int c = i % BN;
                const int gr = block_row + r;
                const int gc = block_col + c;
                if (gr >= spatial || gc >= cout) {
                    continue;
                }
                float acc = 0.0f;
                for (int kk = 0; kk < kdim; ++kk) {
                    const float av = __half2float(load_im2col_nchw(
                        X, ni, cin, h, w, gr, kk, out_w, pad_h, pad_w, pad_mode, spatial, kdim));
                    const float bv = __half2float(W[static_cast<long long>(gc) * kdim + kk]);
                    acc += av * bv;
                }
                if (bias) {
                    acc += __half2float(bias[gc]);
                }
                acc = device::apply_activation(acc, activation);
                Y[static_cast<long long>(gc) * spatial + gr] = __float2half_rn(acc);
            }
#endif
        }

    } // namespace

    void im2col(const void* input, void* col, int n, int c, int h, int w, int k_h, int k_w,
                int out_h, int out_w, int stride_h, int stride_w, int pad_h, int pad_w,
                int dil_h, int dil_w, int c_start, int c_count, int pad_mode, DataType dtype,
                cudaStream_t stream) {
        NvtxRange nvtx("nn.op/im2col");
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
        NvtxRange nvtx("nn.op/col2im");
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

    std::size_t conv2d_weight_scratch_bytes(const int cout, const int cin, const DataType dtype) {
        if (dtype != DataType::Float16 || cin % 8 != 0 || !conv3x3_mma_available()) {
            return 0;
        }
        return static_cast<std::size_t>(9) * static_cast<std::size_t>(cout) *
               static_cast<std::size_t>(cin) * sizeof(__half);
    }

    void conv2d_implicit(const void* input, const void* weight, const void* weight_taps,
                         const void* bias, void* output, void* weight_scratch, int n, int cin,
                         int h, int w, int cout, int kh, int kw, int out_h, int out_w,
                         int stride_h, int stride_w, int pad_h, int pad_w, int dil_h, int dil_w,
                         int pad_mode, int activation, DataType dtype, cudaStream_t stream) {
        NvtxRange nvtx("nn.op/conv_implicit");
        (void)stride_h;
        (void)stride_w;
        (void)dil_h;
        (void)dil_w;
        if (n <= 0 || cout <= 0 || out_h <= 0 || out_w <= 0) {
            return;
        }
        LFS_ASSERT_MSG(dtype == DataType::Float16 || dtype == DataType::Float32,
                       "implicit conv requires float16 or float32");
        LFS_ASSERT_MSG(kh == 3 && kw == 3, "implicit conv requires a 3x3 kernel");
        const int spatial = out_h * out_w;
        const int kdim = cin * 9;
        if (dtype == DataType::Float32) {
            dim3 block(256);
            dim3 grid((cout + 63) / 64, (spatial + 63) / 64, n);
            conv2d_implicit_simt_kernel<false><<<grid, block, 0, stream>>>(
                input, weight, bias, output, n, cin, h, w, cout, out_h, out_w,
                pad_h, pad_w, pad_mode, activation);
            LFS_CUDA_LAUNCH_CHECK(stream, "nn.conv.implicit_3x3_f32");
            return;
        }
        auto* x = static_cast<const __half*>(input);
        auto* wt = static_cast<const __half*>(weight);
        auto* y = static_cast<__half*>(output);
        auto* b = static_cast<const __half*>(bias);
        const char* const forced_path = std::getenv("LFS_LPIPS_CONV");
        const bool force_simt = forced_path && std::strcmp(forced_path, "simt") == 0;
        const bool force_wmma = forced_path && std::strcmp(forced_path, "wmma") == 0;
        const bool force_mma = forced_path && std::strcmp(forced_path, "mma") == 0;
        auto launch_simt = [&]() {
            dim3 block(256);
            dim3 grid((cout + 63) / 64, (spatial + 63) / 64, n);
            conv2d_implicit_simt_kernel<true><<<grid, block, 0, stream>>>(
                x, wt, b, y, n, cin, h, w, cout, out_h, out_w, pad_h, pad_w, pad_mode, activation);
            LFS_CUDA_LAUNCH_CHECK(stream, "nn.conv.implicit_3x3_simt_f16");
        };
        auto launch_wmma = [&]() {
            const bool large = spatial >= 96 && kdim >= 32;
            if (large) {
                constexpr int BM = 128;
                dim3 block(256);
                const int n_tw = (out_w + BM - 1) / BM;
                if (cout >= 128) {
                    constexpr int BN = 128;
                    dim3 grid((cout + BN - 1) / BN, n_tw * out_h, n);
                    hgemm_implicit_3x3_halo_kernel<BN><<<grid, block, 0, stream>>>(
                        x, wt, y, b, n, cin, h, w, cout, out_h, out_w, pad_h, pad_w, pad_mode,
                        activation);
                    LFS_CUDA_LAUNCH_CHECK(stream, "nn.conv.implicit_3x3_halo128");
                } else {
                    constexpr int BN = 64;
                    dim3 grid((cout + BN - 1) / BN, n_tw * out_h, n);
                    hgemm_implicit_3x3_halo_kernel<BN><<<grid, block, 0, stream>>>(
                        x, wt, y, b, n, cin, h, w, cout, out_h, out_w, pad_h, pad_w, pad_mode,
                        activation);
                    LFS_CUDA_LAUNCH_CHECK(stream, "nn.conv.implicit_3x3_halo");
                }
                return;
            }
            constexpr int BM = 64, BN = 64;
            dim3 block(128);
            dim3 grid((cout + BN - 1) / BN, (spatial + BM - 1) / BM, n);
            hgemm_implicit_3x3_kernel<BM, BN, 128><<<grid, block, 0, stream>>>(
                x, wt, y, b, n, cin, h, w, cout, out_h, out_w, pad_h, pad_w, pad_mode, activation);
            LFS_CUDA_LAUNCH_CHECK(stream, "nn.conv.implicit_3x3_64x64");
        };
        if (force_simt) {
            launch_simt();
            return;
        }
        const bool auto75 = forced_path && std::strcmp(forced_path, "auto75") == 0;
        int device = 0;
        int major = 0;
        LFS_CUDA_CHECK(cudaGetDevice(&device));
        LFS_CUDA_CHECK(cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, device));
        static std::once_flag auto_once[16];
        static std::atomic<int> auto_choice[16] = {};
        if (!force_wmma && !force_mma && (auto75 || major < 8) && device >= 0 && device < 16) {
            std::call_once(auto_once[device], [&] {
                cudaEvent_t begin = nullptr;
                cudaEvent_t wmma_done = nullptr;
                cudaEvent_t simt_done = nullptr;
                LFS_CUDA_CHECK(cudaEventCreate(&begin));
                LFS_CUDA_CHECK(cudaEventCreate(&wmma_done));
                LFS_CUDA_CHECK(cudaEventCreate(&simt_done));
                LFS_CUDA_CHECK(cudaEventRecord(begin, stream));
                launch_wmma();
                LFS_CUDA_CHECK(cudaEventRecord(wmma_done, stream));
                launch_simt();
                LFS_CUDA_CHECK(cudaEventRecord(simt_done, stream));
                LFS_CUDA_CHECK(cudaEventSynchronize(simt_done));
                float wmma_ms = 0.0f;
                float simt_ms = 0.0f;
                LFS_CUDA_CHECK(cudaEventElapsedTime(&wmma_ms, begin, wmma_done));
                LFS_CUDA_CHECK(cudaEventElapsedTime(&simt_ms, wmma_done, simt_done));
                auto_choice[device].store(wmma_ms <= simt_ms ? 1 : 2, std::memory_order_relaxed);
                LOG_DEBUG("NN FP16 conv dispatch device={}: {} (WMMA {:.3f} ms, SIMT {:.3f} ms)",
                          device, auto_choice[device].load(std::memory_order_relaxed) == 1 ? "wmma" : "simt", wmma_ms, simt_ms);
                LFS_CUDA_CHECK(cudaEventDestroy(begin));
                LFS_CUDA_CHECK(cudaEventDestroy(wmma_done));
                LFS_CUDA_CHECK(cudaEventDestroy(simt_done));
            });
            if (auto_choice[device].load(std::memory_order_relaxed) == 2) {
                launch_simt();
            } else {
                launch_wmma();
            }
            return;
        }
        if (!force_wmma && (force_mma || weight_taps != nullptr || weight_scratch != nullptr) &&
            conv2d_weight_scratch_bytes(cout, cin, dtype) > 0) {
            if (weight_taps == nullptr) {
                conv3x3_weight_taps(wt, weight_scratch, cout, cin, stream);
                weight_taps = weight_scratch;
            }
            conv2d_implicit_3x3_mma(x, weight_taps, b, y, n, cin, h, w, cout, out_h, out_w,
                                    pad_mode, activation, stream);
            return;
        }
        launch_wmma();
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

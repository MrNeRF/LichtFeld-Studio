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
#include <mma.h>

namespace lfs::core::nn::kernels {
    namespace {

        __device__ __forceinline__ float load_a_f32(const float* a, int row, int col, int m,
                                                    int k, bool trans_a) {
            if (row >= m || col >= k) {
                return 0.0f;
            }
            return trans_a ? __ldg(&a[col * m + row]) : __ldg(&a[row * k + col]);
        }

        __device__ __forceinline__ float load_b_f32(const float* b, int row, int col, int n,
                                                    int k, bool trans_b) {
            if (row >= k || col >= n) {
                return 0.0f;
            }
            return trans_b ? __ldg(&b[col * k + row]) : __ldg(&b[row * n + col]);
        }

        __device__ __forceinline__ void store_c_f32(float* C, int gr, int gc, int m, int n,
                                                    float v, const float* bias, int activation,
                                                    bool trans_c, const float* residual,
                                                    const float* scale) {
            if (gr >= m || gc >= n) {
                return;
            }
            if (bias) {
                v += __ldg(&bias[gc]);
            }
            v = device::apply_activation(v, activation);
            if (scale) {
                v *= __ldg(&scale[gc]);
            }
            if (residual) {
                v += residual[static_cast<long long>(gr) * n + gc];
            }
            if (trans_c) {
                C[static_cast<long long>(gc) * m + gr] = v;
            } else {
                C[static_cast<long long>(gr) * n + gc] = v;
            }
        }

        __device__ __forceinline__ void store_c_f16(__half* C, int gr, int gc, int m, int n,
                                                    float v, const __half* bias, int activation,
                                                    bool trans_c, const __half* residual,
                                                    const __half* scale, int scatter_h,
                                                    int scatter_w) {
            if (gr >= m || gc >= n) {
                return;
            }
            if (bias) {
                v += __half2float(bias[scatter_h > 0 ? gc / 4 : gc]);
            }
            v = device::apply_activation(v, activation);
            if (scale) {
                v *= __half2float(scale[gc]);
            }
            if (residual) {
                v += __half2float(residual[static_cast<long long>(gr) * n + gc]);
            }
            const __half h = __float2half_rn(v);
            if (scatter_h > 0) {
                const int hw = scatter_h * scatter_w;
                const int ni = hw > 0 ? gr / hw : 0;
                const int spatial = hw > 0 ? gr % hw : gr;
                const int ih = scatter_w > 0 ? spatial / scatter_w : 0;
                const int iw = scatter_w > 0 ? spatial % scatter_w : 0;
                const int oc = gc / 4;
                const int kh = (gc & 3) >> 1;
                const int kw = gc & 1;
                const int oh = ih * 2 + kh;
                const int ow = iw * 2 + kw;
                const int out_h = scatter_h * 2;
                const int out_w = scatter_w * 2;
                const int cout = n / 4;
                C[((((static_cast<long long>(ni) * cout + oc) * out_h) + oh) * out_w) + ow] = h;
                return;
            }
            if (trans_c) {
                C[static_cast<long long>(gc) * m + gr] = h;
            } else {
                C[static_cast<long long>(gr) * n + gc] = h;
            }
        }

        template <int BM, int BN, int BK, int TM, int TN>
        __global__ void __launch_bounds__(256)
            sgemm_epilogue_kernel(const float* __restrict__ A, const float* __restrict__ B,
                                  float* __restrict__ C, const float* __restrict__ bias,
                                  int m, int n, int k, long long stride_a, long long stride_b,
                                  long long stride_c, bool trans_a, bool trans_b, int activation,
                                  bool trans_c, const float* __restrict__ residual,
                                  const float* __restrict__ scale) {
            const int batch = static_cast<int>(blockIdx.z);
            A += batch * stride_a;
            B += batch * stride_b;
            C += batch * stride_c;
            if (residual) {
                residual += batch * stride_c;
            }

            __shared__ float As[BM][BK];
            __shared__ float Bs[BK][BN];

            const int tx = static_cast<int>(threadIdx.x);
            const int ty = static_cast<int>(threadIdx.y);
            const int tid = ty * static_cast<int>(blockDim.x) + tx;
            const int num_threads = static_cast<int>(blockDim.x * blockDim.y);

            const int block_row = static_cast<int>(blockIdx.y) * BM;
            const int block_col = static_cast<int>(blockIdx.x) * BN;
            const int thread_row = ty * TM;
            const int thread_col = tx * TN;

            float acc[TM][TN] = {};

            for (int tile = 0; tile < k; tile += BK) {
                for (int i = tid; i < BM * BK; i += num_threads) {
                    const int r = i / BK;
                    const int c = i % BK;
                    As[r][c] = load_a_f32(A, block_row + r, tile + c, m, k, trans_a);
                }
                for (int i = tid; i < BK * BN; i += num_threads) {
                    const int r = i / BN;
                    const int c = i % BN;
                    Bs[r][c] = load_b_f32(B, tile + r, block_col + c, n, k, trans_b);
                }
                __syncthreads();

#pragma unroll
                for (int kk = 0; kk < BK; ++kk) {
                    float a_frag[TM];
                    float b_frag[TN];
#pragma unroll
                    for (int i = 0; i < TM; ++i) {
                        a_frag[i] = As[thread_row + i][kk];
                    }
#pragma unroll
                    for (int j = 0; j < TN; ++j) {
                        b_frag[j] = Bs[kk][thread_col + j];
                    }
#pragma unroll
                    for (int i = 0; i < TM; ++i) {
#pragma unroll
                        for (int j = 0; j < TN; ++j) {
                            acc[i][j] += a_frag[i] * b_frag[j];
                        }
                    }
                }
                __syncthreads();
            }

#pragma unroll
            for (int i = 0; i < TM; ++i) {
#pragma unroll
                for (int j = 0; j < TN; ++j) {
                    const int gr = block_row + thread_row + i;
                    const int gc = block_col + thread_col + j;
                    store_c_f32(C, gr, gc, m, n, acc[i][j], bias, activation, trans_c, residual,
                                scale);
                }
            }
        }

        // Ampere/Ada tensor-core GEMM. 16x16x16 WMMA tiles lower to
        // mma.sync.aligned.m16n8k16 with fp16 inputs and fp32 accumulate.
        // Block tile 64x64, K-tile 32. trans_b selects C = A @ Bᵀ (B is [N,K]).
        template <bool kTransB>
        __global__ void __launch_bounds__(128, 4)
            hgemm_wmma_kernel(const __half* __restrict__ A, const __half* __restrict__ B,
                              __half* __restrict__ C, const __half* __restrict__ bias,
                              int m, int n, int k, long long stride_a, long long stride_b,
                              long long stride_c, bool trans_a, int activation, bool trans_c) {
            const int batch = static_cast<int>(blockIdx.z);
            A += batch * stride_a;
            B += batch * stride_b;
            C += batch * stride_c;

            const int tid = static_cast<int>(threadIdx.x);
            const int block_row = static_cast<int>(blockIdx.y) * 64;
            const int block_col = static_cast<int>(blockIdx.x) * 64;

#if __CUDA_ARCH__ >= 700
            using namespace nvcuda::wmma;
            constexpr int BM = 64;
            constexpr int BN = 64;
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

            auto load_a = [&](int row, int col) -> __half {
                if (row >= m || col >= k) {
                    return __float2half(0.0f);
                }
                return trans_a ? A[col * m + row] : A[row * k + col];
            };
            auto load_b = [&](int brow, int bcol) -> __half {
                if constexpr (kTransB) {
                    if (bcol >= n || brow >= k) {
                        return __float2half(0.0f);
                    }
                    return B[bcol * k + brow];
                } else {
                    if (brow >= k || bcol >= n) {
                        return __float2half(0.0f);
                    }
                    return B[brow * n + bcol];
                }
            };

            for (int k0 = 0; k0 < k; k0 += BK) {
                for (int i = tid; i < BM * BK; i += 128) {
                    const int r = i / BK;
                    const int c = i % BK;
                    As[r][c] = load_a(block_row + r, k0 + c);
                }
                for (int i = tid; i < BN * BK; i += 128) {
                    const int r = i / BK;
                    const int c = i % BK;
                    Bs[r][c] = load_b(k0 + c, block_col + r);
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

            if (trans_c) {
                for (int i = tid; i < BM * BN; i += 128) {
                    const int c = i / BM;
                    const int r = i % BM;
                    store_c_f16(C, block_row + r, block_col + c, m, n, Cs[r][c], bias, activation,
                                true, nullptr, nullptr, 0, 0);
                }
            } else {
                for (int i = tid; i < BM * BN; i += 128) {
                    const int r = i / BN;
                    const int c = i % BN;
                    store_c_f16(C, block_row + r, block_col + c, m, n, Cs[r][c], bias, activation,
                                false, nullptr, nullptr, 0, 0);
                }
            }
#else
            for (int i = tid; i < 64 * 64; i += 128) {
                const int r = i / 64;
                const int c = i % 64;
                const int gr = block_row + r;
                const int gc = block_col + c;
                if (gr >= m || gc >= n) {
                    continue;
                }
                float acc = 0.0f;
                for (int kk = 0; kk < k; ++kk) {
                    const float av = trans_a ? __half2float(A[kk * m + gr])
                                             : __half2float(A[gr * k + kk]);
                    const float bv = kTransB ? __half2float(B[gc * k + kk])
                                             : __half2float(B[kk * n + gc]);
                    acc += av * bv;
                }
                store_c_f16(C, gr, gc, m, n, acc, bias, activation, trans_c, nullptr, nullptr, 0,
                            0);
            }
#endif
        }

        void launch_sgemm_epilogue(const float* a, const float* b, float* c, const float* bias,
                                   int m, int n, int k, long long stride_a, long long stride_b,
                                   long long stride_c, int batch, bool trans_a, bool trans_b,
                                   int activation, bool trans_c, cudaStream_t stream,
                                   const float* residual, const float* scale) {
            constexpr int BM = 64, BN = 64, BK = 8, TM = 4, TN = 4;
            dim3 block(BN / TN, BM / TM);
            // trans_a / trans_c use the full M as the leading dimension, so the
            // whole M is launched in one grid (65535 * 64 covers > 4M rows).
            dim3 grid((n + BN - 1) / BN, (m + BM - 1) / BM, batch);
            sgemm_epilogue_kernel<BM, BN, BK, TM, TN><<<grid, block, 0, stream>>>(
                a, b, c, bias, m, n, k, stride_a, stride_b, stride_c, trans_a, trans_b,
                activation, trans_c, residual, scale);
            LFS_CUDA_LAUNCH_CHECK(stream, "nn.gemm.sgemm_epilogue");
        }

        // 128×64 WMMA tile. 8 warps cover a 128×64 output tile (4 along M × 2
        // along N). Double-buffered cp.async K-tiles; C is stored from fragments
        // (no Cs smem) so both buffers fit in 24 KB.
        template <bool kTransB>
        __global__ void __launch_bounds__(256, 3)
            hgemm_wmma_128x64_kernel(const __half* __restrict__ A, const __half* __restrict__ B,
                                     __half* __restrict__ C, const __half* __restrict__ bias,
                                     int m, int n, int k, long long stride_a, long long stride_b,
                                     long long stride_c, bool trans_a, int activation,
                                     bool trans_c, const __half* __restrict__ residual,
                                     const __half* __restrict__ scale, int scatter_h,
                                     int scatter_w) {
            const int batch = static_cast<int>(blockIdx.z);
            A += batch * stride_a;
            B += batch * stride_b;
            C += batch * stride_c;

            const int tid = static_cast<int>(threadIdx.x);
            const int block_row = static_cast<int>(blockIdx.y) * 128;
            const int block_col = static_cast<int>(blockIdx.x) * 64;

#if __CUDA_ARCH__ >= 700
            using namespace nvcuda::wmma;
            constexpr int BM = 128;
            constexpr int BN = 64;
            constexpr int BK = 32;
            constexpr int WM = 16;
            constexpr int WN = 16;
            constexpr int WK = 16;

            const int warp_id = tid / 32;
            const int warp_m = warp_id / 2;
            const int warp_n = warp_id % 2;

            __shared__ __align__(16) __half As[2][BM][BK];
            __shared__ __align__(16) __half Bs[2][BN][BK];

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

            auto load_ab = [&](int stage, int k0) {
                if (!trans_a) {
#pragma unroll
                    for (int v = 0; v < 2; ++v) {
                        const int vec = tid + v * 256;
                        const int elem = vec * 8;
                        const int r = elem / BK;
                        const int c = elem % BK;
                        const int gr = block_row + r;
                        const int gc = k0 + c;
                        const __half* src = A + static_cast<long long>(gr) * k + gc;
                        __half* dst = &As[stage][r][c];
                        if (r < BM && gr < m && gc + 7 < k &&
                            (reinterpret_cast<uintptr_t>(src) & 15u) == 0 &&
                            (reinterpret_cast<uintptr_t>(dst) & 15u) == 0) {
                            device::cp_async16(dst, src);
                        } else {
#pragma unroll
                            for (int t = 0; t < 8; ++t) {
                                const int cc = c + t;
                                __half val = __float2half(0.0f);
                                if (r < BM && gr < m && k0 + cc < k) {
                                    val = A[static_cast<long long>(gr) * k + (k0 + cc)];
                                }
                                if (r < BM && cc < BK) {
                                    As[stage][r][cc] = val;
                                }
                            }
                        }
                    }
                } else {
                    for (int i = tid; i < BM * BK; i += 256) {
                        const int r = i / BK;
                        const int c = i % BK;
                        const int gr = block_row + r;
                        const int gc = k0 + c;
                        __half val = __float2half(0.0f);
                        if (gr < m && gc < k) {
                            val = A[static_cast<long long>(gc) * m + gr];
                        }
                        As[stage][r][c] = val;
                    }
                }
                if constexpr (kTransB) {
#pragma unroll
                    for (int v = 0; v < 1; ++v) {
                        const int vec = tid + v * 256;
                        const int elem = vec * 8;
                        const int r = elem / BK;
                        const int c = elem % BK;
                        const int gc = block_col + r;
                        const int kk = k0 + c;
                        const __half* src = B + static_cast<long long>(gc) * k + kk;
                        __half* dst = &Bs[stage][r][c];
                        if (r < BN && gc < n && kk + 7 < k &&
                            (reinterpret_cast<uintptr_t>(src) & 15u) == 0 &&
                            (reinterpret_cast<uintptr_t>(dst) & 15u) == 0) {
                            device::cp_async16(dst, src);
                        } else {
#pragma unroll
                            for (int t = 0; t < 8; ++t) {
                                const int cc = c + t;
                                __half val = __float2half(0.0f);
                                if (r < BN && gc < n && k0 + cc < k) {
                                    val = B[static_cast<long long>(gc) * k + (k0 + cc)];
                                }
                                if (r < BN && cc < BK) {
                                    Bs[stage][r][cc] = val;
                                }
                            }
                        }
                    }
                } else {
                    for (int i = tid; i < BN * BK; i += 256) {
                        const int r = i / BK;
                        const int c = i % BK;
                        const int gc = block_col + r;
                        const int kk = k0 + c;
                        __half val = __float2half(0.0f);
                        if (gc < n && kk < k) {
                            val = B[static_cast<long long>(kk) * n + gc];
                        }
                        Bs[stage][r][c] = val;
                    }
                }
                device::cp_async_commit();
            };

            auto mma_stage = [&](int stage) {
#pragma unroll
                for (int kk = 0; kk < BK; kk += WK) {
                    const int am = warp_m * 32;
                    const int bn = warp_n * 32;
                    load_matrix_sync(a_frag[0], &As[stage][am][kk], BK);
                    load_matrix_sync(a_frag[1], &As[stage][am + 16][kk], BK);
                    load_matrix_sync(b_frag[0], &Bs[stage][bn][kk], BK);
                    load_matrix_sync(b_frag[1], &Bs[stage][bn + 16][kk], BK);
                    mma_sync(c_frag[0][0], a_frag[0], b_frag[0], c_frag[0][0]);
                    mma_sync(c_frag[0][1], a_frag[0], b_frag[1], c_frag[0][1]);
                    mma_sync(c_frag[1][0], a_frag[1], b_frag[0], c_frag[1][0]);
                    mma_sync(c_frag[1][1], a_frag[1], b_frag[1], c_frag[1][1]);
                }
            };

            load_ab(0, 0);
            device::cp_async_wait0();
            __syncthreads();

            int stage = 0;
            for (int k0 = 0; k0 < k; k0 += BK) {
                const int next = k0 + BK;
                const int other = stage ^ 1;
                if (next < k) {
                    load_ab(other, next);
                }
                mma_stage(stage);
                if (next < k) {
                    device::cp_async_wait0();
                }
                __syncthreads();
                stage = other;
            }

            const int am = warp_m * 32;
            const int bn = warp_n * 32;
            const int lane = tid % 32;
            const int base_row = lane / 4;
            const int base_col = (lane % 4) * 2;
            const __half* res_ptr = residual ? residual + batch * stride_c : nullptr;
#pragma unroll
            for (int fi = 0; fi < 2; ++fi) {
#pragma unroll
                for (int fj = 0; fj < 2; ++fj) {
#pragma unroll
                    for (int i = 0; i < 8; ++i) {
                        const int r = am + fi * 16 + base_row + ((i >> 1) & 1) * 8;
                        const int c = bn + fj * 16 + base_col + (i & 1) + ((i & 4) ? 8 : 0);
                        store_c_f16(C, block_row + r, block_col + c, m, n, c_frag[fi][fj].x[i],
                                    bias, activation, trans_c, res_ptr, scale, scatter_h,
                                    scatter_w);
                    }
                }
            }
#else
            for (int i = tid; i < 128 * 64; i += 256) {
                const int r = i / 64;
                const int c = i % 64;
                const int gr = block_row + r;
                const int gc = block_col + c;
                if (gr >= m || gc >= n) {
                    continue;
                }
                float acc = 0.0f;
                for (int kk = 0; kk < k; ++kk) {
                    const float av = trans_a ? __half2float(A[kk * m + gr])
                                             : __half2float(A[gr * k + kk]);
                    const float bv = kTransB ? __half2float(B[gc * k + kk])
                                             : __half2float(B[kk * n + gc]);
                    acc += av * bv;
                }
                store_c_f16(C, gr, gc, m, n, acc, bias, activation, trans_c,
                            residual ? residual + batch * stride_c : nullptr, scale, scatter_h,
                            scatter_w);
            }
#endif
        }

        __device__ __forceinline__ void store_nt_pair(__half* C, int gr, int gc, int m, int n,
                                                      float v0, float v1, const __half* bias,
                                                      int activation, const __half* residual,
                                                      const __half* scale) {
            const bool r_ok = gr < m;
            const bool c0 = r_ok && gc < n;
            const bool c1 = r_ok && (gc + 1) < n;
            if (!c0 && !c1) {
                return;
            }
            if (bias) {
                if (c0) {
                    v0 += __half2float(__ldg(bias + gc));
                }
                if (c1) {
                    v1 += __half2float(__ldg(bias + gc + 1));
                }
            }
            v0 = device::apply_activation(v0, activation);
            v1 = device::apply_activation(v1, activation);
            if (scale) {
                if (c0) {
                    v0 *= __half2float(__ldg(scale + gc));
                }
                if (c1) {
                    v1 *= __half2float(__ldg(scale + gc + 1));
                }
            }
            const long long base = static_cast<long long>(gr) * n + gc;
            if (residual) {
                if (c0) {
                    v0 += __half2float(residual[base]);
                }
                if (c1) {
                    v1 += __half2float(residual[base + 1]);
                }
            }
            if (c0 && c1 && (reinterpret_cast<uintptr_t>(C + base) & 3u) == 0) {
                *reinterpret_cast<__half2*>(C + base) = __floats2half2_rn(v0, v1);
            } else if (c0) {
                C[base] = __float2half_rn(v0);
                if (c1) {
                    C[base + 1] = __float2half_rn(v1);
                }
            } else {
                C[base + 1] = __float2half_rn(v1);
            }
        }

        // NT (C = A @ Bᵀ) double-buffered WMMA. A is [M,K] row-major, B is [N,K].
        // Requires K%8==0 so every 16-byte vector is in-bounds or fully OOB
        // (predicated cp.async, no scalar K-tail). 8 warps: 4×2 for BN=64
        // (32×32/warp) or 2×4 for BN=128 (64×32/warp).
        template <int BM, int BN, int BK>
        __global__ void __launch_bounds__(256, (BN <= 64 && BK <= 32) ? 3 : 2)
            hgemm_nt_aligned_kernel(const __half* __restrict__ A, const __half* __restrict__ B,
                                    __half* __restrict__ C, const __half* __restrict__ bias,
                                    int m, int n, int k, long long stride_a, long long stride_b,
                                    long long stride_c, int activation,
                                    const __half* __restrict__ residual,
                                    const __half* __restrict__ scale) {
            const int batch = static_cast<int>(blockIdx.z);
            A += batch * stride_a;
            B += batch * stride_b;
            C += batch * stride_c;
            if (residual) {
                residual += batch * stride_c;
            }

            const int tid = static_cast<int>(threadIdx.x);
            const int block_row = static_cast<int>(blockIdx.y) * BM;
            const int block_col = static_cast<int>(blockIdx.x) * BN;

#if __CUDA_ARCH__ >= 700
            using namespace nvcuda::wmma;
            constexpr int WM = 16;
            constexpr int WN = 16;
            constexpr int WK = 16;
            constexpr int kWarpM = (BN == 64) ? 4 : 2;
            constexpr int kWarpN = (BN == 64) ? 2 : 4;
            constexpr int kFragM = (BM / kWarpM) / WM;
            constexpr int kFragN = (BN / kWarpN) / WN;
            constexpr int kAVecs = (BM * BK) / (256 * 8);
            constexpr int kBVecs = (BN * BK) / (256 * 8);

            const int warp_id = tid / 32;
            const int warp_m = warp_id / kWarpN;
            const int warp_n = warp_id % kWarpN;
            const int lane = tid % 32;

            __shared__ __align__(16) __half As[2][BM][BK];
            __shared__ __align__(16) __half Bs[2][BN][BK];

            fragment<matrix_a, WM, WN, WK, __half, row_major> a_frag[kFragM];
            fragment<matrix_b, WM, WN, WK, __half, col_major> b_frag[kFragN];
            fragment<accumulator, WM, WN, WK, float> c_frag[kFragM][kFragN];
#pragma unroll
            for (int i = 0; i < kFragM; ++i) {
#pragma unroll
                for (int j = 0; j < kFragN; ++j) {
                    fill_fragment(c_frag[i][j], 0.0f);
                }
            }

            auto load_ab = [&](int stage, int k0) {
#pragma unroll
                for (int v = 0; v < kAVecs; ++v) {
                    const int vec = tid + v * 256;
                    const int elem = vec * 8;
                    const int r = elem / BK;
                    const int c = elem % BK;
                    const int gr = block_row + r;
                    const int gc = k0 + c;
                    const bool pred = gr < m && gc + 7 < k;
                    const __half* src = A + static_cast<long long>(pred ? gr : 0) * k +
                                        (pred ? gc : 0);
                    device::cp_async16_pred<true>(&As[stage][r][c], src, pred);
                }
                if constexpr (kBVecs == 0) {
                    if (tid < (BN * BK) / 8) {
                        const int elem = tid * 8;
                        const int r = elem / BK;
                        const int c = elem % BK;
                        const int gc = block_col + r;
                        const int kk = k0 + c;
                        const bool pred = gc < n && kk + 7 < k;
                        const __half* src = B + static_cast<long long>(pred ? gc : 0) * k +
                                            (pred ? kk : 0);
                        device::cp_async16_pred<false>(&Bs[stage][r][c], src, pred);
                    }
                } else {
#pragma unroll
                    for (int v = 0; v < kBVecs; ++v) {
                        const int vec = tid + v * 256;
                        const int elem = vec * 8;
                        const int r = elem / BK;
                        const int c = elem % BK;
                        const int gc = block_col + r;
                        const int kk = k0 + c;
                        const bool pred = gc < n && kk + 7 < k;
                        const __half* src = B + static_cast<long long>(pred ? gc : 0) * k +
                                            (pred ? kk : 0);
                        device::cp_async16_pred<false>(&Bs[stage][r][c], src, pred);
                    }
                }
                device::cp_async_commit();
            };

            auto mma_stage = [&](int stage) {
                const int am = warp_m * (kFragM * WM);
                const int bn = warp_n * (kFragN * WN);
#pragma unroll
                for (int kk = 0; kk < BK; kk += WK) {
#pragma unroll
                    for (int fi = 0; fi < kFragM; ++fi) {
                        load_matrix_sync(a_frag[fi], &As[stage][am + fi * WM][kk], BK);
                    }
#pragma unroll
                    for (int fj = 0; fj < kFragN; ++fj) {
                        load_matrix_sync(b_frag[fj], &Bs[stage][bn + fj * WN][kk], BK);
                    }
#pragma unroll
                    for (int fi = 0; fi < kFragM; ++fi) {
#pragma unroll
                        for (int fj = 0; fj < kFragN; ++fj) {
                            mma_sync(c_frag[fi][fj], a_frag[fi], b_frag[fj], c_frag[fi][fj]);
                        }
                    }
                }
            };

            load_ab(0, 0);
            device::cp_async_wait0();
            __syncthreads();

            int stage = 0;
            for (int k0 = 0; k0 < k; k0 += BK) {
                const int next = k0 + BK;
                const int other = stage ^ 1;
                if (next < k) {
                    load_ab(other, next);
                }
                mma_stage(stage);
                if (next < k) {
                    device::cp_async_wait0();
                }
                __syncthreads();
                stage = other;
            }

            const int am = warp_m * (kFragM * WM);
            const int bn = warp_n * (kFragN * WN);
            const int base_row = lane / 4;
            const int base_col = (lane % 4) * 2;
#pragma unroll
            for (int fi = 0; fi < kFragM; ++fi) {
#pragma unroll
                for (int fj = 0; fj < kFragN; ++fj) {
#pragma unroll
                    for (int i = 0; i < 8; i += 2) {
                        const int r = am + fi * WM + base_row + ((i >> 1) & 1) * 8;
                        const int c = bn + fj * WN + base_col + ((i & 4) ? 8 : 0);
                        store_nt_pair(C, block_row + r, block_col + c, m, n, c_frag[fi][fj].x[i],
                                      c_frag[fi][fj].x[i + 1], bias, activation, residual, scale);
                    }
                }
            }
#else
            for (int i = tid; i < BM * BN; i += 256) {
                const int r = i / BN;
                const int c = i % BN;
                const int gr = block_row + r;
                const int gc = block_col + c;
                if (gr >= m || gc >= n) {
                    continue;
                }
                float acc = 0.0f;
                for (int kk = 0; kk < k; ++kk) {
                    acc += __half2float(A[static_cast<long long>(gr) * k + kk]) *
                           __half2float(B[static_cast<long long>(gc) * k + kk]);
                }
                store_c_f16(C, gr, gc, m, n, acc, bias, activation, false, residual, scale, 0, 0);
            }
#endif
        }

        void launch_hgemm_wmma(const __half* a, const __half* b, __half* c, const __half* bias,
                               int m, int n, int k, long long stride_a, long long stride_b,
                               long long stride_c, int batch, bool trans_a, bool trans_b,
                               int activation, bool trans_c, cudaStream_t stream,
                               const __half* residual, const __half* scale, int scatter_h,
                               int scatter_w) {
            // Linear / NT encoder GEMMs (SAM2 Hiera, MoGe ViT): K is a multiple of
            // 8 so every 16-byte vector is fully in-bounds or fully OOB. Keep the
            // generic 128×64 kernel for trans_a (1×1 NCHW), scatter (conv-transpose),
            // and odd-K shapes used by the parity tests.
            const bool aligned_nt = !trans_a && trans_b && !trans_c && scatter_h == 0 && k >= 8 &&
                                    (k % 8 == 0);
            if (aligned_nt && m >= 96 && n >= 64) {
                dim3 block(256);
                // 128-wide N reuses A across more columns; worth it when N fits
                // one tile, N is a multiple of 128, or K is short (Hiera stage-0
                // K=112 is memory-bound and A traffic dominates).
                const bool tile_128 = n >= 96 && (n <= 128 || n % 128 == 0 || k <= 128);
                const bool k32 = (k % 32 == 0) && k >= 32;
                if (tile_128) {
                    constexpr int BM = 128, BN = 128;
                    dim3 grid((n + BN - 1) / BN, (m + BM - 1) / BM, batch);
                    if (k32) {
                        hgemm_nt_aligned_kernel<BM, BN, 32><<<grid, block, 0, stream>>>(
                            a, b, c, bias, m, n, k, stride_a, stride_b, stride_c, activation,
                            residual, scale);
                        LFS_CUDA_LAUNCH_CHECK(stream, "nn.gemm.hgemm_nt_128x128x32");
                    } else {
                        hgemm_nt_aligned_kernel<BM, BN, 16><<<grid, block, 0, stream>>>(
                            a, b, c, bias, m, n, k, stride_a, stride_b, stride_c, activation,
                            residual, scale);
                        LFS_CUDA_LAUNCH_CHECK(stream, "nn.gemm.hgemm_nt_128x128x16");
                    }
                } else {
                    constexpr int BM = 128, BN = 64;
                    dim3 grid((n + BN - 1) / BN, (m + BM - 1) / BM, batch);
                    if (k32) {
                        hgemm_nt_aligned_kernel<BM, BN, 32><<<grid, block, 0, stream>>>(
                            a, b, c, bias, m, n, k, stride_a, stride_b, stride_c, activation,
                            residual, scale);
                        LFS_CUDA_LAUNCH_CHECK(stream, "nn.gemm.hgemm_nt_128x64x32");
                    } else {
                        hgemm_nt_aligned_kernel<BM, BN, 16><<<grid, block, 0, stream>>>(
                            a, b, c, bias, m, n, k, stride_a, stride_b, stride_c, activation,
                            residual, scale);
                        LFS_CUDA_LAUNCH_CHECK(stream, "nn.gemm.hgemm_nt_128x64x16");
                    }
                }
                return;
            }
            const bool large = (m >= 96 && n >= 64 && k >= 32) || scatter_h > 0 || residual ||
                               scale;
            if (large) {
                constexpr int BM = 128, BN = 64;
                dim3 block(256);
                dim3 grid((n + BN - 1) / BN, (m + BM - 1) / BM, batch);
                if (trans_b) {
                    hgemm_wmma_128x64_kernel<true><<<grid, block, 0, stream>>>(
                        a, b, c, bias, m, n, k, stride_a, stride_b, stride_c, trans_a, activation,
                        trans_c, residual, scale, scatter_h, scatter_w);
                    LFS_CUDA_LAUNCH_CHECK(stream, "nn.gemm.hgemm_wmma_128x64_nt");
                } else {
                    hgemm_wmma_128x64_kernel<false><<<grid, block, 0, stream>>>(
                        a, b, c, bias, m, n, k, stride_a, stride_b, stride_c, trans_a, activation,
                        trans_c, residual, scale, scatter_h, scatter_w);
                    LFS_CUDA_LAUNCH_CHECK(stream, "nn.gemm.hgemm_wmma_128x64_nn");
                }
                return;
            }
            constexpr int BM = 64, BN = 64;
            dim3 block(128);
            dim3 grid((n + BN - 1) / BN, (m + BM - 1) / BM, batch);
            if (trans_b) {
                hgemm_wmma_kernel<true><<<grid, block, 0, stream>>>(
                    a, b, c, bias, m, n, k, stride_a, stride_b, stride_c, trans_a, activation,
                    trans_c);
                LFS_CUDA_LAUNCH_CHECK(stream, "nn.gemm.hgemm_wmma_nt");
            } else {
                hgemm_wmma_kernel<false><<<grid, block, 0, stream>>>(
                    a, b, c, bias, m, n, k, stride_a, stride_b, stride_c, trans_a, activation,
                    trans_c);
                LFS_CUDA_LAUNCH_CHECK(stream, "nn.gemm.hgemm_wmma_nn");
            }
        }

    } // namespace

    void gemm(const void* a, const void* b, void* c, int m, int n, int k,
              long long stride_a, long long stride_b, long long stride_c, int batch,
              bool trans_a, bool trans_b, const void* bias, int activation,
              DataType dtype, cudaStream_t stream, bool trans_c, const void* residual,
              const void* scale, int scatter_h, int scatter_w) {
        NvtxRange nvtx("nn.op/gemm");
        if (m <= 0 || n <= 0 || batch <= 0) {
            return;
        }
        if (k < 0) {
            k = 0;
        }
        if (dtype == DataType::Float16) {
            launch_hgemm_wmma(static_cast<const __half*>(a), static_cast<const __half*>(b),
                              static_cast<__half*>(c), static_cast<const __half*>(bias),
                              m, n, k, stride_a, stride_b, stride_c, batch, trans_a, trans_b,
                              activation, trans_c, stream, static_cast<const __half*>(residual),
                              static_cast<const __half*>(scale), scatter_h, scatter_w);
            return;
        }
        launch_sgemm_epilogue(static_cast<const float*>(a), static_cast<const float*>(b),
                              static_cast<float*>(c), static_cast<const float*>(bias),
                              m, n, k, stride_a, stride_b, stride_c, batch, trans_a, trans_b,
                              activation, trans_c, stream, static_cast<const float*>(residual),
                              static_cast<const float*>(scale));
    }

} // namespace lfs::core::nn::kernels

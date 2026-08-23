/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "core/cuda_error.hpp"
#include "nn_device.cuh"
#include "nn_kernels.hpp"

#include <algorithm>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <mma.h>

namespace lfs::core::nn::kernels {
    namespace {

        constexpr int kMaxGridY = 65535;

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

        template <int BM, int BN, int BK, int TM, int TN>
        __global__ void __launch_bounds__(256)
            sgemm_epilogue_kernel(const float* __restrict__ A, const float* __restrict__ B,
                                  float* __restrict__ C, const float* __restrict__ bias,
                                  int m, int n, int k, long long stride_a, long long stride_b,
                                  long long stride_c, bool trans_a, bool trans_b, int activation) {
            const int batch = static_cast<int>(blockIdx.z);
            A += batch * stride_a;
            B += batch * stride_b;
            C += batch * stride_c;

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
                    if (gr < m && gc < n) {
                        float v = acc[i][j];
                        if (bias) {
                            v += __ldg(&bias[gc]);
                        }
                        C[gr * n + gc] = device::apply_activation(v, activation);
                    }
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
                              long long stride_c, bool trans_a, int activation) {
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

            for (int i = tid; i < BM * BN; i += 128) {
                const int r = i / BN;
                const int c = i % BN;
                const int gr = block_row + r;
                const int gc = block_col + c;
                if (gr < m && gc < n) {
                    float v = Cs[r][c];
                    if (bias) {
                        v += __half2float(bias[gc]);
                    }
                    C[gr * n + gc] = __float2half_rn(device::apply_activation(v, activation));
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
                if (bias) {
                    acc += __half2float(bias[gc]);
                }
                C[gr * n + gc] = __float2half_rn(device::apply_activation(acc, activation));
            }
#endif
        }

        void launch_sgemm_epilogue(const float* a, const float* b, float* c, const float* bias,
                                   int m, int n, int k, long long stride_a, long long stride_b,
                                   long long stride_c, int batch, bool trans_a, bool trans_b,
                                   int activation, cudaStream_t stream) {
            constexpr int BM = 64, BN = 64, BK = 8, TM = 4, TN = 4;
            dim3 block(BN / TN, BM / TM);
            const int max_rows = kMaxGridY * BM;
            for (int row0 = 0; row0 < m; row0 += max_rows) {
                const int rows = std::min(max_rows, m - row0);
                dim3 grid((n + BN - 1) / BN, (rows + BM - 1) / BM, batch);
                const float* a_off = trans_a ? a + row0 : a + static_cast<long long>(row0) * k;
                sgemm_epilogue_kernel<BM, BN, BK, TM, TN><<<grid, block, 0, stream>>>(
                    a_off, b, c + static_cast<long long>(row0) * n, bias, rows, n, k,
                    stride_a, stride_b, stride_c, trans_a, trans_b, activation);
                LFS_CUDA_LAUNCH_CHECK(stream, "nn.gemm.sgemm_epilogue");
            }
        }

        // 128×64 WMMA tile. 8 warps cover a 128×64 output tile (4 along M × 2
        // along N). Used for the large-N ViT GEMMs.
        template <bool kTransB>
        __global__ void __launch_bounds__(256, 3)
            hgemm_wmma_128x64_kernel(const __half* __restrict__ A, const __half* __restrict__ B,
                                     __half* __restrict__ C, const __half* __restrict__ bias,
                                     int m, int n, int k, long long stride_a, long long stride_b,
                                     long long stride_c, bool trans_a, int activation) {
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
                for (int i = tid; i < BM * BK; i += 256) {
                    const int r = i / BK;
                    const int c = i % BK;
                    As[r][c] = load_a(block_row + r, k0 + c);
                }
                for (int i = tid; i < BN * BK; i += 256) {
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

            for (int i = tid; i < BM * BN; i += 256) {
                const int r = i / BN;
                const int c = i % BN;
                const int gr = block_row + r;
                const int gc = block_col + c;
                if (gr < m && gc < n) {
                    float val = Cs[r][c];
                    if (bias) {
                        val += __half2float(bias[gc]);
                    }
                    C[gr * n + gc] = __float2half_rn(device::apply_activation(val, activation));
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
                if (bias) {
                    acc += __half2float(bias[gc]);
                }
                C[gr * n + gc] = __float2half_rn(device::apply_activation(acc, activation));
            }
#endif
        }

        void launch_hgemm_wmma(const __half* a, const __half* b, __half* c, const __half* bias,
                               int m, int n, int k, long long stride_a, long long stride_b,
                               long long stride_c, int batch, bool trans_a, bool trans_b,
                               int activation, cudaStream_t stream) {
            const bool large = m >= 96 && n >= 64 && k >= 32;
            if (large) {
                constexpr int BM = 128, BN = 64;
                dim3 block(256);
                const int max_rows = kMaxGridY * BM;
                for (int row0 = 0; row0 < m; row0 += max_rows) {
                    const int rows = std::min(max_rows, m - row0);
                    dim3 grid((n + BN - 1) / BN, (rows + BM - 1) / BM, batch);
                    const __half* a_off = trans_a ? a + row0 : a + static_cast<long long>(row0) * k;
                    __half* c_off = c + static_cast<long long>(row0) * n;
                    if (trans_b) {
                        hgemm_wmma_128x64_kernel<true><<<grid, block, 0, stream>>>(
                            a_off, b, c_off, bias, rows, n, k, stride_a, stride_b, stride_c,
                            trans_a, activation);
                        LFS_CUDA_LAUNCH_CHECK(stream, "nn.gemm.hgemm_wmma_128x64_nt");
                    } else {
                        hgemm_wmma_128x64_kernel<false><<<grid, block, 0, stream>>>(
                            a_off, b, c_off, bias, rows, n, k, stride_a, stride_b, stride_c,
                            trans_a, activation);
                        LFS_CUDA_LAUNCH_CHECK(stream, "nn.gemm.hgemm_wmma_128x64_nn");
                    }
                }
                return;
            }
            constexpr int BM = 64, BN = 64;
            dim3 block(128);
            const int max_rows = kMaxGridY * BM;
            for (int row0 = 0; row0 < m; row0 += max_rows) {
                const int rows = std::min(max_rows, m - row0);
                dim3 grid((n + BN - 1) / BN, (rows + BM - 1) / BM, batch);
                const __half* a_off = trans_a ? a + row0 : a + static_cast<long long>(row0) * k;
                __half* c_off = c + static_cast<long long>(row0) * n;
                if (trans_b) {
                    hgemm_wmma_kernel<true><<<grid, block, 0, stream>>>(
                        a_off, b, c_off, bias, rows, n, k, stride_a, stride_b, stride_c,
                        trans_a, activation);
                    LFS_CUDA_LAUNCH_CHECK(stream, "nn.gemm.hgemm_wmma_nt");
                } else {
                    hgemm_wmma_kernel<false><<<grid, block, 0, stream>>>(
                        a_off, b, c_off, bias, rows, n, k, stride_a, stride_b, stride_c,
                        trans_a, activation);
                    LFS_CUDA_LAUNCH_CHECK(stream, "nn.gemm.hgemm_wmma_nn");
                }
            }
        }

    } // namespace

    void gemm(const void* a, const void* b, void* c, int m, int n, int k,
              long long stride_a, long long stride_b, long long stride_c, int batch,
              bool trans_a, bool trans_b, const void* bias, int activation,
              DataType dtype, cudaStream_t stream) {
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
                              activation, stream);
            return;
        }
        launch_sgemm_epilogue(static_cast<const float*>(a), static_cast<const float*>(b),
                              static_cast<float*>(c), static_cast<const float*>(bias),
                              m, n, k, stride_a, stride_b, stride_c, batch, trans_a, trans_b,
                              activation, stream);
    }

} // namespace lfs::core::nn::kernels

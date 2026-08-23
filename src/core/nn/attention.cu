/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "core/cuda_error.hpp"
#include "nn_device.cuh"
#include "nn_kernels.hpp"

#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <float.h>

namespace lfs::core::nn::kernels {
    namespace {

        constexpr int kBr = 32;
        constexpr int kBc = 32;
        constexpr int kMaxD = 128;

        __device__ __forceinline__ float warp_sum(float v) {
#pragma unroll
            for (int offset = 16; offset > 0; offset >>= 1) {
                v += __shfl_xor_sync(0xffffffffu, v, offset);
            }
            return v;
        }

        __device__ __forceinline__ float warp_max(float v) {
#pragma unroll
            for (int offset = 16; offset > 0; offset >>= 1) {
                const float other = __shfl_xor_sync(0xffffffffu, v, offset);
                v = fmaxf(v, other);
            }
            return v;
        }

        // Online-softmax attention: Q,K,V,O are [B, H, N, D] (Nq vs Nk allowed).
        // One thread per query row in a Br-tile. Does not materialise N×N scores.
        __global__ void __launch_bounds__(kBr)
            flash_attn_kernel(const void* __restrict__ q_ptr, const void* __restrict__ k_ptr,
                              const void* __restrict__ v_ptr, const void* __restrict__ mask_ptr,
                              void* __restrict__ o_ptr, int batch, int heads, int n_q, int n_k,
                              int d, float scale, long long mask_sb, long long mask_sh,
                              long long mask_sq, long long mask_sk, bool has_mask, bool is_half) {
            const int q_row = static_cast<int>(blockIdx.x) * kBr + static_cast<int>(threadIdx.x);
            const int bh = static_cast<int>(blockIdx.y);
            const int b = bh / heads;
            const int h = bh % heads;
            if (b >= batch) {
                return;
            }

            const long long q_base =
                (static_cast<long long>(b) * heads + h) * n_q * d +
                static_cast<long long>(q_row) * d;
            const long long kv_head =
                (static_cast<long long>(b) * heads + h) * n_k * d;
            const long long o_base = q_base;

            float acc[kMaxD];
#pragma unroll
            for (int i = 0; i < kMaxD; ++i) {
                acc[i] = 0.0f;
            }

            float m_i = -FLT_MAX;
            float l_i = 0.0f;
            const bool valid_q = q_row < n_q;

            for (int k0 = 0; k0 < n_k; k0 += kBc) {
                float s[kBc];
                float row_max = -FLT_MAX;
#pragma unroll
                for (int j = 0; j < kBc; ++j) {
                    const int k_idx = k0 + j;
                    float dot = -FLT_MAX;
                    if (valid_q && k_idx < n_k) {
                        dot = 0.0f;
                        for (int dd = 0; dd < d; ++dd) {
                            const float qv = device::ld_strided(q_ptr, q_base + dd, is_half);
                            const float kv =
                                device::ld_strided(k_ptr, kv_head + static_cast<long long>(k_idx) * d + dd,
                                                   is_half);
                            dot += qv * kv;
                        }
                        dot *= scale;
                        if (has_mask) {
                            const long long mi = b * mask_sb + h * mask_sh +
                                                 static_cast<long long>(q_row) * mask_sq +
                                                 static_cast<long long>(k_idx) * mask_sk;
                            dot += device::ld_strided(mask_ptr, mi, is_half);
                        }
                    }
                    s[j] = dot;
                    row_max = fmaxf(row_max, dot);
                }

                const float m_new = fmaxf(m_i, row_max);
                const float alpha = (m_i == -FLT_MAX) ? 0.0f : expf(m_i - m_new);
                float l_add = 0.0f;
                float p[kBc];
#pragma unroll
                for (int j = 0; j < kBc; ++j) {
                    p[j] = (s[j] == -FLT_MAX) ? 0.0f : expf(s[j] - m_new);
                    l_add += p[j];
                }

                for (int dd = 0; dd < d; ++dd) {
                    float vdot = acc[dd] * alpha;
#pragma unroll
                    for (int j = 0; j < kBc; ++j) {
                        const int k_idx = k0 + j;
                        if (k_idx < n_k && p[j] != 0.0f) {
                            const float vv = device::ld_strided(
                                v_ptr, kv_head + static_cast<long long>(k_idx) * d + dd, is_half);
                            vdot += p[j] * vv;
                        }
                    }
                    acc[dd] = vdot;
                }
                l_i = l_i * alpha + l_add;
                m_i = m_new;
            }

            if (!valid_q) {
                return;
            }
            const float inv = (l_i == 0.0f) ? 0.0f : 1.0f / l_i;
            for (int dd = 0; dd < d; ++dd) {
                device::st_strided(o_ptr, o_base + dd, acc[dd] * inv, is_half);
            }
        }

        __global__ void softmax_kernel(const void* __restrict__ x, const void* __restrict__ mask,
                                       void* __restrict__ y, int rows, int cols,
                                       long long mask_stride_row, long long mask_stride_col,
                                       bool has_mask, bool is_half) {
            const int row = static_cast<int>(blockIdx.x);
            if (row >= rows) {
                return;
            }
            const int tid = static_cast<int>(threadIdx.x);
            const int nthreads = static_cast<int>(blockDim.x);
            const long long base = static_cast<long long>(row) * cols;

            float local_max = -FLT_MAX;
            for (int c = tid; c < cols; c += nthreads) {
                float v = device::ld_strided(x, base + c, is_half);
                if (has_mask) {
                    v += device::ld_strided(mask, row * mask_stride_row + c * mask_stride_col,
                                            is_half);
                }
                local_max = fmaxf(local_max, v);
            }
            __shared__ float red[32];
            float wmax = warp_max(local_max);
            if ((tid & 31) == 0) {
                red[tid / 32] = wmax;
            }
            __syncthreads();
            if (tid < 32) {
                const float v = (tid < (nthreads + 31) / 32) ? red[tid] : -FLT_MAX;
                wmax = warp_max(v);
                if (tid == 0) {
                    red[0] = wmax;
                }
            }
            __syncthreads();
            const float row_max = red[0];

            float local_sum = 0.0f;
            for (int c = tid; c < cols; c += nthreads) {
                float v = device::ld_strided(x, base + c, is_half);
                if (has_mask) {
                    v += device::ld_strided(mask, row * mask_stride_row + c * mask_stride_col,
                                            is_half);
                }
                local_sum += expf(v - row_max);
            }
            float wsum = warp_sum(local_sum);
            if ((tid & 31) == 0) {
                red[tid / 32] = wsum;
            }
            __syncthreads();
            if (tid < 32) {
                const float v = (tid < (nthreads + 31) / 32) ? red[tid] : 0.0f;
                wsum = warp_sum(v);
                if (tid == 0) {
                    red[0] = wsum;
                }
            }
            __syncthreads();
            const float inv = red[0] > 0.0f ? 1.0f / red[0] : 0.0f;

            for (int c = tid; c < cols; c += nthreads) {
                float v = device::ld_strided(x, base + c, is_half);
                if (has_mask) {
                    v += device::ld_strided(mask, row * mask_stride_row + c * mask_stride_col,
                                            is_half);
                }
                device::st_strided(y, base + c, expf(v - row_max) * inv, is_half);
            }
        }

    } // namespace

    void attention(const void* q, const void* k, const void* v, const void* mask, void* o,
                   int batch, int heads, int n_q, int n_k, int d, float scale,
                   long long mask_sb, long long mask_sh, long long mask_sq, long long mask_sk,
                   bool has_mask, DataType dtype, cudaStream_t stream) {
        if (batch <= 0 || heads <= 0 || n_q <= 0 || d <= 0) {
            return;
        }
        dim3 grid((n_q + kBr - 1) / kBr, batch * heads);
        flash_attn_kernel<<<grid, kBr, 0, stream>>>(
            q, k, v, mask, o, batch, heads, n_q, n_k, d, scale, mask_sb, mask_sh, mask_sq,
            mask_sk, has_mask, dtype == DataType::Float16);
        LFS_CUDA_LAUNCH_CHECK(stream, "nn.attention.flash");
    }

    void softmax(const void* x, const void* mask, void* y, int rows, int cols,
                 long long mask_stride_row, long long mask_stride_col, bool has_mask,
                 DataType dtype, cudaStream_t stream) {
        if (rows <= 0 || cols <= 0) {
            return;
        }
        const int threads = cols >= 256 ? 256 : (cols >= 128 ? 128 : 64);
        softmax_kernel<<<rows, threads, 0, stream>>>(
            x, mask, y, rows, cols, mask_stride_row, mask_stride_col, has_mask,
            dtype == DataType::Float16);
        LFS_CUDA_LAUNCH_CHECK(stream, "nn.softmax");
    }

} // namespace lfs::core::nn::kernels

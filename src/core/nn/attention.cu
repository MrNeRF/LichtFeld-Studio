/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "core/cuda_error.hpp"
#include "nn_device.cuh"
#include "nn_kernels.hpp"
#include "nn_nvtx.hpp"

#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <float.h>
#include <mma.h>

namespace lfs::core::nn::kernels {
    namespace {

        constexpr int kMaxD = 128;
        constexpr int kBr = 64;

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

        __device__ __forceinline__ float mask_at(const void* mask, int b, int h, int q_row,
                                                 int k_idx, long long sb, long long sh,
                                                 long long sq, long long sk, bool is_half) {
            const long long mi = static_cast<long long>(b) * sb + static_cast<long long>(h) * sh +
                                 static_cast<long long>(q_row) * sq +
                                 static_cast<long long>(k_idx) * sk;
            return device::ld_strided(mask, mi, is_half);
        }

        // Tiled online-softmax attention. A block owns Br query rows of one
        // (batch, head). K/V stream through shared memory in Bc-wide tiles.
        template <int Br, int Bc, int Dmax>
        __global__ void __launch_bounds__(Br)
            flash_attn_tiled_kernel(const void* __restrict__ q_ptr, const void* __restrict__ k_ptr,
                                    const void* __restrict__ v_ptr, const void* __restrict__ mask_ptr,
                                    void* __restrict__ o_ptr, int batch, int heads, int n_q, int n_k,
                                    int d, float scale, long long mask_sb, long long mask_sh,
                                    long long mask_sq, long long mask_sk, bool has_mask,
                                    bool is_half) {
            extern __shared__ __align__(16) char tiled_raw[];
            constexpr int Dpad = Dmax;
            float* Qs = reinterpret_cast<float*>(tiled_raw);
            float* Ks = Qs + Br * Dpad;
            float* Vs = Ks + Bc * Dpad;

            const int tid = static_cast<int>(threadIdx.x);
            const int nthreads = static_cast<int>(blockDim.x);
            const int q0 = static_cast<int>(blockIdx.x) * Br;
            const int bh = static_cast<int>(blockIdx.y);
            const int b = bh / heads;
            const int h = bh % heads;
            if (b >= batch) {
                return;
            }

            const long long q_head =
                (static_cast<long long>(b) * heads + h) * static_cast<long long>(n_q) * d;
            const long long kv_head =
                (static_cast<long long>(b) * heads + h) * static_cast<long long>(n_k) * d;

            for (int i = tid; i < Br * Dpad; i += nthreads) {
                const int r = i / Dpad;
                const int c = i % Dpad;
                const int q_row = q0 + r;
                float val = 0.0f;
                if (c < d && q_row < n_q) {
                    val = device::ld_strided(q_ptr, q_head + static_cast<long long>(q_row) * d + c,
                                             is_half);
                }
                Qs[i] = val;
            }

            const int row = tid;
            const bool valid_q = row < Br && (q0 + row) < n_q;
            float acc[Dmax];
#pragma unroll
            for (int i = 0; i < Dmax; ++i) {
                acc[i] = 0.0f;
            }
            float m_i = -FLT_MAX;
            float l_i = 0.0f;

            for (int k0 = 0; k0 < n_k; k0 += Bc) {
                for (int i = tid; i < Bc * Dpad; i += nthreads) {
                    const int r = i / Dpad;
                    const int c = i % Dpad;
                    const int k_idx = k0 + r;
                    float kv = 0.0f;
                    float vv = 0.0f;
                    if (c < d && k_idx < n_k) {
                        const long long base =
                            kv_head + static_cast<long long>(k_idx) * d + c;
                        kv = device::ld_strided(k_ptr, base, is_half);
                        vv = device::ld_strided(v_ptr, base, is_half);
                    }
                    Ks[i] = kv;
                    Vs[i] = vv;
                }
                __syncthreads();

                if (valid_q) {
                    float s[Bc];
                    float row_max = -FLT_MAX;
#pragma unroll
                    for (int j = 0; j < Bc; ++j) {
                        const int k_idx = k0 + j;
                        float dot = -FLT_MAX;
                        if (k_idx < n_k) {
                            dot = 0.0f;
#pragma unroll
                            for (int dd = 0; dd < Dmax; ++dd) {
                                dot += Qs[row * Dpad + dd] * Ks[j * Dpad + dd];
                            }
                            dot *= scale;
                            if (has_mask) {
                                dot += mask_at(mask_ptr, b, h, q0 + row, k_idx, mask_sb, mask_sh,
                                               mask_sq, mask_sk, is_half);
                            }
                        }
                        s[j] = dot;
                        row_max = fmaxf(row_max, dot);
                    }

                    const float m_new = fmaxf(m_i, row_max);
                    const float alpha = (m_i == -FLT_MAX) ? 0.0f : expf(m_i - m_new);
                    float l_add = 0.0f;
                    float p[Bc];
#pragma unroll
                    for (int j = 0; j < Bc; ++j) {
                        p[j] = (s[j] == -FLT_MAX) ? 0.0f : expf(s[j] - m_new);
                        l_add += p[j];
                    }
#pragma unroll
                    for (int dd = 0; dd < Dmax; ++dd) {
                        float vdot = acc[dd] * alpha;
#pragma unroll
                        for (int j = 0; j < Bc; ++j) {
                            vdot += p[j] * Vs[j * Dpad + dd];
                        }
                        acc[dd] = vdot;
                    }
                    l_i = l_i * alpha + l_add;
                    m_i = m_new;
                }
                __syncthreads();
            }

            if (!valid_q) {
                return;
            }
            const float inv = (l_i == 0.0f) ? 0.0f : 1.0f / l_i;
            const int q_row = q0 + row;
            const long long o_base = q_head + static_cast<long long>(q_row) * d;
            for (int dd = 0; dd < d; ++dd) {
                device::st_strided(o_ptr, o_base + dd, acc[dd] * inv, is_half);
            }
        }

        // fp16 tensor-core attention. Br query rows × one (batch, head).
        // Q stays in smem (scaled by 1/sqrt(d)); K/V stream in Bc tiles.
        // P is packed into its own buffer so Q is not overwritten. O lives in
        // registers so two blocks fit per SM on Ada (48 KB dynamic smem).
        template <int Br, int Bc, int D>
        __global__ void __launch_bounds__(128, 2)
            flash_attn_wmma_kernel(const __half* __restrict__ q_ptr, const __half* __restrict__ k_ptr,
                                   const __half* __restrict__ v_ptr, const void* __restrict__ mask_ptr,
                                   __half* __restrict__ o_ptr, int batch, int heads, int n_q, int n_k,
                                   int d, float scale, long long mask_sb, long long mask_sh,
                                   long long mask_sq, long long mask_sk, bool has_mask,
                                   bool mask_is_half) {
            const int tid = static_cast<int>(threadIdx.x);
            const int q0 = static_cast<int>(blockIdx.x) * Br;
            const int bh = static_cast<int>(blockIdx.y);
            const int b = bh / heads;
            const int h = bh % heads;
            if (b >= batch) {
                return;
            }

            extern __shared__ __align__(16) char raw[];
            auto* Qs = reinterpret_cast<__half*>(raw);
            auto* Ks = Qs + Br * D;
            auto* Vs = Ks + Bc * D;
            auto* Ps = Vs + Bc * D;
            auto* Ss = reinterpret_cast<float*>(Ps + Br * Bc);

            const long long q_head =
                (static_cast<long long>(b) * heads + h) * static_cast<long long>(n_q) * d;
            const long long kv_head =
                (static_cast<long long>(b) * heads + h) * static_cast<long long>(n_k) * d;

            for (int i = tid; i < Br * D; i += 128) {
                const int r = i / D;
                const int c = i % D;
                const int q_row = q0 + r;
                __half val = __float2half(0.0f);
                if (c < d && q_row < n_q) {
                    val = q_ptr[q_head + static_cast<long long>(q_row) * d + c];
                }
                Qs[i] = val;
            }

            const int row = tid;
            const bool softmax_lane = tid < Br;
            const bool valid_q = softmax_lane && (q0 + row) < n_q;
            float m_i = -FLT_MAX;
            float l_i = 0.0f;
            float acc[D];
#pragma unroll
            for (int i = 0; i < D; ++i) {
                acc[i] = 0.0f;
            }
            const int warp = tid / 32;
            const int warp_row = warp * 16;
            __syncthreads();

            for (int k0 = 0; k0 < n_k; k0 += Bc) {
                for (int i = tid; i < Bc * D; i += 128) {
                    const int r = i / D;
                    const int c = i % D;
                    const int k_idx = k0 + r;
                    __half kv = __float2half(0.0f);
                    __half vv = __float2half(0.0f);
                    if (c < d && k_idx < n_k) {
                        const long long base =
                            kv_head + static_cast<long long>(k_idx) * d + c;
                        kv = k_ptr[base];
                        vv = v_ptr[base];
                    }
                    Ks[i] = kv;
                    Vs[i] = vv;
                }
                __syncthreads();

#if __CUDA_ARCH__ >= 700
                {
                    using namespace nvcuda::wmma;
                    fragment<matrix_a, 16, 16, 16, __half, row_major> a_frag;
                    fragment<matrix_b, 16, 16, 16, __half, col_major> b_frag;
                    fragment<accumulator, 16, 16, 16, float> s_frag;
#pragma unroll
                    for (int ns = 0; ns < Bc; ns += 16) {
                        fill_fragment(s_frag, 0.0f);
#pragma unroll
                        for (int ds = 0; ds < D; ds += 16) {
                            load_matrix_sync(a_frag, Qs + warp_row * D + ds, D);
                            load_matrix_sync(b_frag, Ks + ns * D + ds, D);
                            mma_sync(s_frag, a_frag, b_frag, s_frag);
                        }
                        store_matrix_sync(Ss + warp_row * Bc + ns, s_frag, Bc, mem_row_major);
                    }
                }
#else
                for (int i = tid; i < Br * Bc; i += 128) {
                    const int r = i / Bc;
                    const int c = i % Bc;
                    float dot = 0.0f;
                    for (int dd = 0; dd < d; ++dd) {
                        dot += __half2float(Qs[r * D + dd]) * __half2float(Ks[c * D + dd]);
                    }
                    Ss[i] = dot;
                }
#endif
                __syncthreads();

                if (softmax_lane) {
                    float row_max = -FLT_MAX;
#pragma unroll
                    for (int j = 0; j < Bc; ++j) {
                        const int k_idx = k0 + j;
                        float s = -FLT_MAX;
                        if (valid_q && k_idx < n_k) {
                            s = Ss[row * Bc + j] * scale;
                            if (has_mask) {
                                s += mask_at(mask_ptr, b, h, q0 + row, k_idx, mask_sb, mask_sh,
                                             mask_sq, mask_sk, mask_is_half);
                            }
                        }
                        Ss[row * Bc + j] = s;
                        row_max = fmaxf(row_max, s);
                    }
                    const float m_new = fmaxf(m_i, row_max);
                    const float alpha = (m_i == -FLT_MAX) ? 0.0f : expf(m_i - m_new);
                    float l_add = 0.0f;
#pragma unroll
                    for (int j = 0; j < Bc; ++j) {
                        const float s = Ss[row * Bc + j];
                        const float p = (s == -FLT_MAX) ? 0.0f : expf(s - m_new);
                        Ss[row * Bc + j] = p;
                        l_add += p;
                    }
                    if (valid_q) {
#pragma unroll
                        for (int dd = 0; dd < D; ++dd) {
                            acc[dd] *= alpha;
                        }
                    }
                    l_i = l_i * alpha + l_add;
                    m_i = m_new;
                }
                __syncthreads();

                for (int i = tid; i < Br * Bc; i += 128) {
                    Ps[i] = __float2half_rn(Ss[i]);
                }
                __syncthreads();

#if __CUDA_ARCH__ >= 700
                {
                    using namespace nvcuda::wmma;
                    fragment<matrix_a, 16, 16, 16, __half, row_major> p_frag;
                    fragment<matrix_b, 16, 16, 16, __half, row_major> v_frag;
                    fragment<accumulator, 16, 16, 16, float> o_frag;
#pragma unroll
                    for (int ds = 0; ds < D; ds += 16) {
                        fill_fragment(o_frag, 0.0f);
#pragma unroll
                        for (int ns = 0; ns < Bc; ns += 16) {
                            load_matrix_sync(p_frag, Ps + warp_row * Bc + ns, Bc);
                            load_matrix_sync(v_frag, Vs + ns * D + ds, D);
                            mma_sync(o_frag, p_frag, v_frag, o_frag);
                        }
                        store_matrix_sync(Ss + warp_row * Bc + ds, o_frag, Bc, mem_row_major);
                    }
                }
#else
                for (int i = tid; i < Br * D; i += 128) {
                    const int r = i / D;
                    const int c = i % D;
                    float sum = 0.0f;
                    if (c < d) {
                        for (int j = 0; j < Bc; ++j) {
                            sum += __half2float(Ps[r * Bc + j]) * __half2float(Vs[j * D + c]);
                        }
                    }
                    Ss[r * Bc + c] = sum;
                }
#endif
                __syncthreads();

                if (valid_q) {
#pragma unroll
                    for (int dd = 0; dd < D; ++dd) {
                        if (dd < d) {
                            acc[dd] += Ss[row * Bc + dd];
                        }
                    }
                }
                __syncthreads();
            }

            if (valid_q) {
                const float inv = (l_i == 0.0f) ? 0.0f : 1.0f / l_i;
                const int q_row = q0 + row;
                const long long o_base = q_head + static_cast<long long>(q_row) * d;
#pragma unroll
                for (int dd = 0; dd < D; ++dd) {
                    if (dd < d) {
                        o_ptr[o_base + dd] = __float2half_rn(acc[dd] * inv);
                    }
                }
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

        int tiled_smem_bytes(int br, int bc, int dpad) {
            return (br + bc + bc) * dpad * static_cast<int>(sizeof(float));
        }

        int wmma_smem_bytes(int br, int bc, int dpad) {
            const int qkv = (br + bc + bc) * dpad * static_cast<int>(sizeof(__half));
            const int p = br * bc * static_cast<int>(sizeof(__half));
            const int s = br * bc * static_cast<int>(sizeof(float));
            return qkv + p + s;
        }

        constexpr int kFlashBr = 128;
        constexpr int kFlashTile = 64;
        constexpr int kFlashBc = 64;
        constexpr int kFlashD = 64;
        constexpr int kFlashLd = 72;

        // K/V tiles live at leading-dimension 72 (64 useful + 8 pad). The pad
        // is the load_matrix_sync-compatible form of an 8-half xor swizzle:
        // 8 consecutive rows at a fixed column hit distinct smem banks.
        __device__ __forceinline__ void load_kv_tile(__half* Ks, __half* Vs, const __half* K,
                                                     const __half* V, int k0, int n_k,
                                                     long long kv_head, int tid) {
#pragma unroll
            for (int i = 0; i < 4; ++i) {
                const int off = (tid + i * 128) * 8;
                const int row = off / kFlashD;
                const int col = off % kFlashD;
                const int k_idx = k0 + row;
                const int sm = row * kFlashLd + col;
                if (k_idx < n_k) {
                    const long long base = kv_head + static_cast<long long>(k_idx) * kFlashD + col;
                    device::cp_async16(Ks + sm, K + base);
                    device::cp_async16(Vs + sm, V + base);
                } else {
                    *reinterpret_cast<uint4*>(Ks + sm) = uint4{0, 0, 0, 0};
                    *reinterpret_cast<uint4*>(Vs + sm) = uint4{0, 0, 0, 0};
                }
            }
        }

        // Flash attention for d=64: two 64-row Q tiles per block (Br=128),
        // Q/O fragment-resident, K/V cp.async double-buffered with xor-swizzle,
        // P kept in A fragments for the PV mma (no BrxBc overlay on the inner ds loop).
        __global__ void __launch_bounds__(128, 2)
            flash_attn_d64_kernel(const __half* __restrict__ q_ptr, const __half* __restrict__ k_ptr,
                                  const __half* __restrict__ v_ptr, __half* __restrict__ o_ptr,
                                  int batch, int heads, int n_q, int n_k, float scale) {
            const int tid = static_cast<int>(threadIdx.x);
            const int q0 = static_cast<int>(blockIdx.x) * kFlashBr;
            const int bh = static_cast<int>(blockIdx.y);
            const int b = bh / heads;
            const int h = bh % heads;
            if (b >= batch) {
                return;
            }

            extern __shared__ __align__(16) char raw[];
            auto* Qs = reinterpret_cast<__half*>(raw);
            auto* Ks0 = Qs + kFlashTile * kFlashD;
            auto* Vs0 = Ks0 + kFlashBc * kFlashLd;
            auto* Ks1 = Vs0 + kFlashBc * kFlashLd;
            auto* Vs1 = Ks1 + kFlashBc * kFlashLd;
            auto* Ps = Qs;

            const long long q_head =
                (static_cast<long long>(b) * heads + h) * static_cast<long long>(n_q) * kFlashD;
            const long long kv_head =
                (static_cast<long long>(b) * heads + h) * static_cast<long long>(n_k) * kFlashD;

#if __CUDA_ARCH__ >= 700
            using namespace nvcuda::wmma;
            const int warp = tid / 32;
            const int lane = tid % 32;
            const int warp_row = warp * 16;
            const int base_row = lane / 4;
            const int base_col = (lane % 4) * 2;
            fragment<matrix_a, 16, 16, 16, __half, row_major> q_frag[2][4];
            fragment<accumulator, 16, 16, 16, float> o_frag[2][4];
            float m_i[2][2] = {{-FLT_MAX, -FLT_MAX}, {-FLT_MAX, -FLT_MAX}};
            float l_i[2][2] = {{0.0f, 0.0f}, {0.0f, 0.0f}};

#pragma unroll
            for (int tile = 0; tile < 2; ++tile) {
                const int q_base = q0 + tile * kFlashTile;
                for (int i = tid; i < kFlashTile * kFlashD; i += 128) {
                    const int r = i / kFlashD;
                    const int c = i % kFlashD;
                    const int q_row = q_base + r;
                    float val = 0.0f;
                    if (q_row < n_q) {
                        val = __half2float(
                            q_ptr[q_head + static_cast<long long>(q_row) * kFlashD + c]);
                    }
                    Qs[i] = __float2half_rn(val * scale);
                }
                __syncthreads();
#pragma unroll
                for (int ds = 0; ds < 4; ++ds) {
                    load_matrix_sync(q_frag[tile][ds], Qs + warp_row * kFlashD + ds * 16, kFlashD);
                    fill_fragment(o_frag[tile][ds], 0.0f);
                }
                __syncthreads();
            }

            auto* Kcur = Ks0;
            auto* Vcur = Vs0;
            load_kv_tile(Kcur, Vcur, k_ptr, v_ptr, 0, n_k, kv_head, tid);
            device::cp_async_commit();
            device::cp_async_wait0();
            __syncthreads();

            for (int k0 = 0; k0 < n_k; k0 += kFlashBc) {
                const int next = k0 + kFlashBc;
                auto* Koth = (Kcur == Ks0) ? Ks1 : Ks0;
                auto* Voth = (Vcur == Vs0) ? Vs1 : Vs0;
                if (next < n_k) {
                    load_kv_tile(Koth, Voth, k_ptr, v_ptr, next, n_k, kv_head, tid);
                    device::cp_async_commit();
                }

#pragma unroll
                for (int tile = 0; tile < 2; ++tile) {
                    fragment<matrix_b, 16, 16, 16, __half, col_major> k_frag;
                    fragment<accumulator, 16, 16, 16, float> s_frag[4];
#pragma unroll
                    for (int ns = 0; ns < 4; ++ns) {
                        fill_fragment(s_frag[ns], 0.0f);
#pragma unroll
                        for (int ds = 0; ds < 4; ++ds) {
                            load_matrix_sync(k_frag, Kcur + ns * 16 * kFlashLd + ds * 16,
                                             kFlashLd);
                            mma_sync(s_frag[ns], q_frag[tile][ds], k_frag, s_frag[ns]);
                        }
                    }

                    float row_max[2] = {-FLT_MAX, -FLT_MAX};
#pragma unroll
                    for (int ns = 0; ns < 4; ++ns) {
#pragma unroll
                        for (int i = 0; i < 8; ++i) {
                            const int col = base_col + (i & 1) + ((i & 4) ? 8 : 0) + ns * 16;
                            const int row_off = (i >> 1) & 1;
                            const int k_idx = k0 + col;
                            float s = s_frag[ns].x[i];
                            if (k_idx >= n_k) {
                                s = -FLT_MAX;
                            }
                            s_frag[ns].x[i] = s;
                            row_max[row_off] = fmaxf(row_max[row_off], s);
                        }
                    }
#pragma unroll
                    for (int r = 0; r < 2; ++r) {
                        row_max[r] = fmaxf(row_max[r], __shfl_xor_sync(0xffffffffu, row_max[r], 1));
                        row_max[r] = fmaxf(row_max[r], __shfl_xor_sync(0xffffffffu, row_max[r], 2));
                    }

                    float alpha[2];
                    float m_new[2];
#pragma unroll
                    for (int r = 0; r < 2; ++r) {
                        m_new[r] = fmaxf(m_i[tile][r], row_max[r]);
                        alpha[r] = (m_i[tile][r] == -FLT_MAX) ? 0.0f : expf(m_i[tile][r] - m_new[r]);
                    }
#pragma unroll
                    for (int ds = 0; ds < 4; ++ds) {
#pragma unroll
                        for (int i = 0; i < 8; ++i) {
                            o_frag[tile][ds].x[i] *= alpha[(i >> 1) & 1];
                        }
                    }

                    float l_add[2] = {0.0f, 0.0f};
#pragma unroll
                    for (int ns = 0; ns < 4; ++ns) {
#pragma unroll
                        for (int i = 0; i < 8; ++i) {
                            const int col = base_col + (i & 1) + ((i & 4) ? 8 : 0) + ns * 16;
                            const int row_off = (i >> 1) & 1;
                            const int row = warp_row + base_row + row_off * 8;
                            const float s = s_frag[ns].x[i];
                            const float p = (s == -FLT_MAX) ? 0.0f : expf(s - m_new[row_off]);
                            s_frag[ns].x[i] = p;
                            l_add[row_off] += p;
                            if (row < kFlashTile) {
                                Ps[row * kFlashBc + col] = __float2half_rn(p);
                            }
                        }
                    }
#pragma unroll
                    for (int r = 0; r < 2; ++r) {
                        l_add[r] += __shfl_xor_sync(0xffffffffu, l_add[r], 1);
                        l_add[r] += __shfl_xor_sync(0xffffffffu, l_add[r], 2);
                        l_i[tile][r] = l_i[tile][r] * alpha[r] + l_add[r];
                        m_i[tile][r] = m_new[r];
                    }
                    __syncthreads();

                    fragment<matrix_a, 16, 16, 16, __half, row_major> p_frag[4];
#pragma unroll
                    for (int ns = 0; ns < 4; ++ns) {
                        load_matrix_sync(p_frag[ns], Ps + warp_row * kFlashBc + ns * 16, kFlashBc);
                    }

                    fragment<matrix_b, 16, 16, 16, __half, row_major> v_frag;
                    fragment<accumulator, 16, 16, 16, float> pv_frag;
#pragma unroll
                    for (int ds = 0; ds < 4; ++ds) {
                        fill_fragment(pv_frag, 0.0f);
#pragma unroll
                        for (int ns = 0; ns < 4; ++ns) {
                            load_matrix_sync(v_frag, Vcur + ns * 16 * kFlashLd + ds * 16,
                                             kFlashLd);
                            mma_sync(pv_frag, p_frag[ns], v_frag, pv_frag);
                        }
#pragma unroll
                        for (int i = 0; i < 8; ++i) {
                            o_frag[tile][ds].x[i] += pv_frag.x[i];
                        }
                    }
                    __syncthreads();
                }

                if (next < n_k) {
                    device::cp_async_wait0();
                }
                __syncthreads();
                Kcur = Koth;
                Vcur = Voth;
            }

#pragma unroll
            for (int tile = 0; tile < 2; ++tile) {
                const int q_base = q0 + tile * kFlashTile;
#pragma unroll
                for (int ds = 0; ds < 4; ++ds) {
#pragma unroll
                    for (int i = 0; i < 8; ++i) {
                        const int row_off = (i >> 1) & 1;
                        const int row = warp_row + base_row + row_off * 8;
                        const int col = base_col + (i & 1) + ((i & 4) ? 8 : 0) + ds * 16;
                        const int q_row = q_base + row;
                        if (q_row < n_q && col < kFlashD) {
                            const float inv =
                                (l_i[tile][row_off] == 0.0f) ? 0.0f : 1.0f / l_i[tile][row_off];
                            o_ptr[q_head + static_cast<long long>(q_row) * kFlashD + col] =
                                __float2half_rn(o_frag[tile][ds].x[i] * inv);
                        }
                    }
                }
            }
#else
            (void)Qs;
            (void)Ks0;
            (void)Vs0;
            (void)Ks1;
            (void)Vs1;
            (void)Ps;
            (void)scale;
            (void)q_ptr;
            (void)k_ptr;
            (void)v_ptr;
            (void)o_ptr;
            (void)n_q;
            (void)n_k;
            (void)q_head;
            (void)kv_head;
            (void)tid;
#endif
        }

        int flash_d64_smem_bytes() {
            return (kFlashTile * kFlashD + 2 * kFlashBc * kFlashLd + 2 * kFlashBc * kFlashLd) *
                   static_cast<int>(sizeof(__half));
        }

    } // namespace

    void attention(const void* q, const void* k, const void* v, const void* mask, void* o,
                   int batch, int heads, int n_q, int n_k, int d, float scale,
                   long long mask_sb, long long mask_sh, long long mask_sq, long long mask_sk,
                   bool has_mask, DataType dtype, cudaStream_t stream) {
        NvtxRange nvtx("nn.op/attention");
        if (batch <= 0 || heads <= 0 || n_q <= 0 || d <= 0) {
            return;
        }
        const bool is_half = dtype == DataType::Float16;

        if (is_half && d == 64 && !has_mask && n_k > 0) {
            dim3 grid((n_q + kFlashBr - 1) / kFlashBr, batch * heads);
            const int smem = flash_d64_smem_bytes();
            auto* fn = flash_attn_d64_kernel;
            LFS_CUDA_CHECK(cudaFuncSetAttribute(reinterpret_cast<const void*>(fn),
                                                cudaFuncAttributeMaxDynamicSharedMemorySize,
                                                smem));
            fn<<<grid, 128, smem, stream>>>(
                static_cast<const __half*>(q), static_cast<const __half*>(k),
                static_cast<const __half*>(v), static_cast<__half*>(o), batch, heads, n_q, n_k,
                scale);
            LFS_CUDA_LAUNCH_CHECK(stream, "nn.attention.flash_d64");
            return;
        }

        dim3 grid((n_q + kBr - 1) / kBr, batch * heads);

        if (is_half && (d % 16) == 0 && d <= 64 && n_k > 0) {
            constexpr int Br = 64;
            constexpr int Bc = 64;
            constexpr int D = 64;
            const int smem = wmma_smem_bytes(Br, Bc, D);
            auto* fn = flash_attn_wmma_kernel<Br, Bc, D>;
            LFS_CUDA_CHECK(cudaFuncSetAttribute(reinterpret_cast<const void*>(fn),
                                                cudaFuncAttributeMaxDynamicSharedMemorySize,
                                                smem));
            fn<<<grid, 128, smem, stream>>>(
                static_cast<const __half*>(q), static_cast<const __half*>(k),
                static_cast<const __half*>(v), mask, static_cast<__half*>(o), batch, heads, n_q,
                n_k, d, scale, mask_sb, mask_sh, mask_sq, mask_sk, has_mask, is_half);
            LFS_CUDA_LAUNCH_CHECK(stream, "nn.attention.wmma");
            return;
        }

        if (d <= 64) {
            constexpr int Br = 64;
            constexpr int Bc = 32;
            const int smem = tiled_smem_bytes(Br, Bc, 64);
            flash_attn_tiled_kernel<Br, Bc, 64><<<grid, Br, smem, stream>>>(
                q, k, v, mask, o, batch, heads, n_q, n_k, d, scale, mask_sb, mask_sh, mask_sq,
                mask_sk, has_mask, is_half);
            LFS_CUDA_LAUNCH_CHECK(stream, "nn.attention.tiled64");
            return;
        }

        dim3 grid128((n_q + 32 - 1) / 32, batch * heads);
        const int smem = tiled_smem_bytes(32, 32, kMaxD);
        flash_attn_tiled_kernel<32, 32, kMaxD><<<grid128, 32, smem, stream>>>(
            q, k, v, mask, o, batch, heads, n_q, n_k, d, scale, mask_sb, mask_sh, mask_sq,
            mask_sk, has_mask, is_half);
        LFS_CUDA_LAUNCH_CHECK(stream, "nn.attention.tiled128");
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

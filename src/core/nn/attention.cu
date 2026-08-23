/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "core/cuda_error.hpp"
#include "nn_device.cuh"
#include "nn_kernels.hpp"

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
        // K/V stream in Bc-wide tiles. QKᵀ / PV on WMMA 16×16×16 with fp32
        // accumulate; online softmax in fp32. Dpad must be 64 (ViT-B and any
        // d<=64 padded to 64).
        template <int Br, int Bc, int D>
        __global__ void __launch_bounds__(128, 3)
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
            auto* Ss = reinterpret_cast<float*>(Vs + Bc * D);
            auto* Os = Ss + Br * Bc;

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
            for (int i = tid; i < Br * D; i += 128) {
                Os[i] = 0.0f;
            }
            __syncthreads();

            const int row = tid;
            const bool softmax_lane = tid < Br;
            const bool valid_q = softmax_lane && (q0 + row) < n_q;
            float m_i = -FLT_MAX;
            float l_i = 0.0f;
            const int warp = tid / 32;
            const int warp_row = warp * 16;

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
                            if (dd < d) {
                                Os[row * D + dd] *= alpha;
                            }
                        }
                    }
                    l_i = l_i * alpha + l_add;
                    m_i = m_new;
                }
                __syncthreads();

                // Pack P into Qs as fp16, [Br, Bc] with ld = D (D == Bc).
                for (int i = tid; i < Br * Bc; i += 128) {
                    const int r = i / Bc;
                    const int c = i % Bc;
                    Qs[r * D + c] = __float2half_rn(Ss[r * Bc + c]);
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
                            load_matrix_sync(p_frag, Qs + warp_row * D + ns, D);
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
                            sum += __half2float(Qs[r * D + j]) * __half2float(Vs[j * D + c]);
                        }
                    }
                    Ss[r * Bc + c] = sum;
                }
#endif
                __syncthreads();

                for (int i = tid; i < Br * D; i += 128) {
                    const int r = i / D;
                    const int c = i % D;
                    if (c < d) {
                        Os[r * D + c] += Ss[r * Bc + c];
                    }
                }
                __syncthreads();

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
                __syncthreads();
            }

            if (valid_q) {
                const float inv = (l_i == 0.0f) ? 0.0f : 1.0f / l_i;
                const int q_row = q0 + row;
                const long long o_base = q_head + static_cast<long long>(q_row) * d;
                for (int dd = 0; dd < d; ++dd) {
                    o_ptr[o_base + dd] = __float2half_rn(Os[row * D + dd] * inv);
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
            const int so = (br * bc + br * dpad) * static_cast<int>(sizeof(float));
            return qkv + so;
        }

    } // namespace

    void attention(const void* q, const void* k, const void* v, const void* mask, void* o,
                   int batch, int heads, int n_q, int n_k, int d, float scale,
                   long long mask_sb, long long mask_sh, long long mask_sq, long long mask_sk,
                   bool has_mask, DataType dtype, cudaStream_t stream) {
        if (batch <= 0 || heads <= 0 || n_q <= 0 || d <= 0) {
            return;
        }
        const bool is_half = dtype == DataType::Float16;
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

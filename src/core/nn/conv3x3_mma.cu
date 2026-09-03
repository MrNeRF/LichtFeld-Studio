/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "core/assert.hpp"
#include "core/cuda_error.hpp"
#include "nn_device.cuh"
#include "nn_kernels.hpp"

#include <algorithm>
#include <cstdint>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

// Implicit-GEMM 3x3 convolution on tensor cores (sm_80+), NCHW fp16 in/out,
// fp32 accumulate. C^T = W x X^T: A = weights [Cout][Cin] per tap, B = a
// (TH+2) x (TW+2) input halo transposed into [pixel][ic] smem rows.
// Weights arrive tap-major ([9][Cout][Cin]) so every B tile is a dense
// cp.async copy; the halo is gathered once per 32-channel chunk and reused by
// all nine taps.

namespace lfs::core::nn::kernels {
    namespace {

        constexpr int kBK = 32;
        constexpr int kBStages = 3;
        constexpr int kThreads = 256;
        constexpr int kTaps = 9;
        constexpr int kHaloRound = 3;

        template <int BN, int TH, int TW>
        struct ConvTile {
            static constexpr int kM = TH * TW;
            static constexpr int kHaloW = TW + 2;
            static constexpr int kHaloRows = (TH + 2) * kHaloW;
            static constexpr int kHaloItems = (4 * kHaloRows + kThreads - 1) / kThreads;
            static constexpr int kHaloRounds = (kHaloItems + kHaloRound - 1) / kHaloRound;
            static constexpr int kWarpsN = BN / 32;
            static constexpr int kWarpsM = 8 / kWarpsN;
            static constexpr int kWarpM = kM / kWarpsM;
            static constexpr int kMGroups = kWarpM / 8;
            static constexpr int kMPairs = kMGroups / 2;
            static constexpr int kBItems = BN * 4 / kThreads;
            static_assert(kM % (8 * kWarpsM) == 0 && kMGroups % 2 == 0);
        };

        // 64-byte rows of 32 halves. The 16-byte chunk index is xor-swizzled by
        // the row so the eight row reads of one ldmatrix hit distinct banks.
        __device__ __forceinline__ int row_chunk_offset(const int row, const int chunk) {
            return row * kBK + ((chunk ^ ((row >> 1) & 3)) << 3);
        }

        __device__ __forceinline__ void ldmatrix_x4(unsigned (&r)[4], const __half* smem) {
#if __CUDA_ARCH__ >= 800
            const unsigned addr = static_cast<unsigned>(__cvta_generic_to_shared(smem));
            asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                         : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3])
                         : "r"(addr));
#else
            (void)r;
            (void)smem;
#endif
        }

        __device__ __forceinline__ void mma_16816(float (&d)[4], const unsigned (&a)[4],
                                                  const unsigned* b) {
#if __CUDA_ARCH__ >= 800
            asm volatile(
                "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, "
                "{%8,%9}, {%0,%1,%2,%3};\n"
                : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
                : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
#else
            (void)d;
            (void)a;
            (void)b;
#endif
        }

        __device__ __forceinline__ void cp_async_wait1() {
#if __CUDA_ARCH__ >= 800
            asm volatile("cp.async.wait_group 1;\n");
#endif
        }

        __device__ __forceinline__ unsigned pack_half2(const __half lo, const __half hi) {
            return static_cast<unsigned>(__half_as_ushort(lo)) |
                   (static_cast<unsigned>(__half_as_ushort(hi)) << 16);
        }

        // Eight consecutive input channels of one halo pixel, packed for a 16-byte
        // smem row chunk. Lanes run along the halo so global reads coalesce. Items
        // are processed in rounds of kHaloRound so the in-flight registers stay
        // bounded next to the accumulators.
        template <class Tile>
        __device__ __forceinline__ void gather_halo(uint4 (&regs)[kHaloRound],
                                                    const __half* __restrict__ X, const int cin,
                                                    const int h, const int w, const int ic0,
                                                    const int oh0, const int ow0,
                                                    const int pad_mode, const int tid,
                                                    const int round) {
            const long long plane = static_cast<long long>(h) * w;
#pragma unroll
            for (int i = 0; i < kHaloRound; ++i) {
                const int item = tid + (round * kHaloRound + i) * kThreads;
                uint4 packed = {0u, 0u, 0u, 0u};
                if (item < 4 * Tile::kHaloRows) {
                    const int icg = item / Tile::kHaloRows;
                    const int p = item - icg * Tile::kHaloRows;
                    const int hr = p / Tile::kHaloW;
                    const int hc = p - hr * Tile::kHaloW;
                    int ih = oh0 - 1 + hr;
                    int iw = ow0 - 1 + hc;
                    if (pad_mode == 1) {
                        ih = ih < 0 ? 0 : (ih >= h ? h - 1 : ih);
                        iw = iw < 0 ? 0 : (iw >= w ? w - 1 : iw);
                    }
                    const int ic = ic0 + icg * 8;
                    if (static_cast<unsigned>(ih) < static_cast<unsigned>(h) &&
                        static_cast<unsigned>(iw) < static_cast<unsigned>(w) && ic < cin) {
                        const __half* src = X + (static_cast<long long>(ic) * h + ih) * w + iw;
                        __half v[8];
#pragma unroll
                        for (int j = 0; j < 8; ++j) {
                            v[j] = __ldg(src + j * plane);
                        }
                        packed.x = pack_half2(v[0], v[1]);
                        packed.y = pack_half2(v[2], v[3]);
                        packed.z = pack_half2(v[4], v[5]);
                        packed.w = pack_half2(v[6], v[7]);
                    }
                }
                regs[i] = packed;
            }
        }

        template <class Tile>
        __device__ __forceinline__ void store_halo(const uint4 (&regs)[kHaloRound], __half* hs,
                                                   const int tid, const int round) {
#pragma unroll
            for (int i = 0; i < kHaloRound; ++i) {
                const int item = tid + (round * kHaloRound + i) * kThreads;
                if (item < 4 * Tile::kHaloRows) {
                    const int icg = item / Tile::kHaloRows;
                    const int p = item - icg * Tile::kHaloRows;
                    *reinterpret_cast<uint4*>(hs + row_chunk_offset(p, icg)) = regs[i];
                }
            }
        }

        template <class Tile>
        __device__ __forceinline__ void load_b_tile(__half* bs, const __half* __restrict__ Wp,
                                                    const int tap, const int n0, const int ic0,
                                                    const int cout, const int cin, const int tid) {
#pragma unroll
            for (int i = 0; i < Tile::kBItems; ++i) {
                const int item = tid + i * kThreads;
                const int row = item >> 2;
                const int chunk = item & 3;
                const int n = n0 + row;
                const int ic = ic0 + chunk * 8;
                const bool pred = n < cout && ic < cin;
                const __half* src =
                    Wp + (static_cast<long long>(tap) * cout + (pred ? n : 0)) * cin + (pred ? ic : 0);
                device::cp_async16_pred<true>(bs + row_chunk_offset(row, chunk), src, pred);
            }
        }

        template <int BN, int TH, int TW>
        __global__ void __launch_bounds__(kThreads, 2)
            conv3x3_mma_kernel(const __half* __restrict__ X, const __half* __restrict__ Wp,
                               const __half* __restrict__ bias, __half* __restrict__ Y, int cin,
                               int h, int w, int cout, int out_h, int out_w, int pad_mode,
                               int activation) {
#if __CUDA_ARCH__ >= 800
            using Tile = ConvTile<BN, TH, TW>;
            constexpr int kMPairs = Tile::kMPairs;
            constexpr int kSteps = 2;

            __shared__ __align__(128) __half hs[Tile::kHaloRows * kBK];
            __shared__ __align__(128) __half bs[kBStages][BN * kBK];

            const int tid = static_cast<int>(threadIdx.x);
            const int warp = tid >> 5;
            const int lane = tid & 31;
            const int warp_n = warp % Tile::kWarpsN;
            const int warp_m = warp / Tile::kWarpsN;
            const int ni = static_cast<int>(blockIdx.z);
            const int n0 = static_cast<int>(blockIdx.x) * BN;
            const int tiles_w = (out_w + TW - 1) / TW;
            const int tile = static_cast<int>(blockIdx.y);
            const int oh0 = (tile / tiles_w) * TH;
            const int ow0 = (tile % tiles_w) * TW;
            const int nchunks = (cin + kBK - 1) / kBK;
            const int nsteps = nchunks * kTaps;
            const long long plane = static_cast<long long>(h) * w;
            const __half* Xn = X + static_cast<long long>(ni) * cin * plane;

            // ldmatrix.x4 lane roles. A (weights): lanes 0-7 rows 0-7 k-lo, 8-15 rows
            // 8-15 k-lo, 16-23 rows 0-7 k-hi, 24-31 rows 8-15 k-hi. B (halo): two m8
            // pixel groups, lanes >>4 select the group, (lane>>3)&1 the k half.
            int a_row[2];
#pragma unroll
            for (int j = 0; j < 2; ++j) {
                a_row[j] = warp_n * 32 + j * 16 + (lane & 7) + ((lane >> 3) & 1) * 8;
            }
            const int a_chunk = lane >> 4;
            int b_row[kMPairs];
#pragma unroll
            for (int q = 0; q < kMPairs; ++q) {
                const int m = warp_m * Tile::kWarpM + (2 * q + (lane >> 4)) * 8 + (lane & 7);
                b_row[q] = (m / TW) * Tile::kHaloW + (m % TW);
            }
            const int b_chunk = (lane >> 3) & 1;

            {
                // Accumulators are not live yet, so every round's loads go out at once.
                uint4 pre[Tile::kHaloRounds][kHaloRound];
#pragma unroll
                for (int round = 0; round < Tile::kHaloRounds; ++round) {
                    gather_halo<Tile>(pre[round], Xn, cin, h, w, 0, oh0, ow0, pad_mode, tid, round);
                }
#pragma unroll
                for (int round = 0; round < Tile::kHaloRounds; ++round) {
                    store_halo<Tile>(pre[round], hs, tid, round);
                }
            }
            load_b_tile<Tile>(bs[0], Wp, 0, n0, 0, cout, cin, tid);
            device::cp_async_commit();
            if (nsteps > 1) {
                load_b_tile<Tile>(bs[1], Wp, 1, n0, 0, cout, cin, tid);
            }
            device::cp_async_commit();
            __syncthreads();

            float acc[2][Tile::kMGroups][4];
#pragma unroll
            for (int j = 0; j < 2; ++j) {
#pragma unroll
                for (int g = 0; g < Tile::kMGroups; ++g) {
#pragma unroll
                    for (int e = 0; e < 4; ++e) {
                        acc[j][g][e] = 0.0f;
                    }
                }
            }

            int chunk = 0;
            int tap = 0;
            int stage = 0;
            for (int t = 0; t < nsteps; ++t) {
                cp_async_wait1();
                __syncthreads();
                if (t + 2 < nsteps) {
                    const int t2 = t + 2;
                    const int chunk2 = t2 / kTaps;
                    const int tap2 = t2 - chunk2 * kTaps;
                    const int stage2 = stage + 2 >= kBStages ? stage + 2 - kBStages : stage + 2;
                    load_b_tile<Tile>(bs[stage2], Wp, tap2, n0, chunk2 * kBK, cout, cin, tid);
                }
                device::cp_async_commit();

                const __half* bcur = bs[stage];
                const int kh = tap / 3;
                const int tap_off = kh * Tile::kHaloW + (tap - kh * 3);

                // Fragments for the next group are fetched before the current group's
                // mma so the ldmatrix latency overlaps tensor-pipe execution.
                unsigned afrag[kSteps][2][4];
                unsigned bfrag[2][4];
#pragma unroll
                for (int j = 0; j < 2; ++j) {
                    ldmatrix_x4(afrag[0][j], bcur + row_chunk_offset(a_row[j], a_chunk));
                }
                ldmatrix_x4(bfrag[0], hs + row_chunk_offset(b_row[0] + tap_off, b_chunk));
#pragma unroll
                for (int kk = 0; kk < kSteps; ++kk) {
#pragma unroll
                    for (int q = 0; q < kMPairs; ++q) {
                        const int cur = (kk * kMPairs + q) & 1;
                        if (q + 1 < kMPairs) {
                            ldmatrix_x4(bfrag[cur ^ 1],
                                        hs + row_chunk_offset(b_row[q + 1] + tap_off,
                                                              kk * 2 + b_chunk));
                        } else if (kk + 1 < kSteps) {
#pragma unroll
                            for (int j = 0; j < 2; ++j) {
                                ldmatrix_x4(afrag[kk + 1][j],
                                            bcur + row_chunk_offset(a_row[j], (kk + 1) * 2 + a_chunk));
                            }
                            ldmatrix_x4(bfrag[cur ^ 1],
                                        hs + row_chunk_offset(b_row[0] + tap_off,
                                                              (kk + 1) * 2 + b_chunk));
                        }
                        mma_16816(acc[0][2 * q], afrag[kk][0], bfrag[cur]);
                        mma_16816(acc[0][2 * q + 1], afrag[kk][0], bfrag[cur] + 2);
                        mma_16816(acc[1][2 * q], afrag[kk][1], bfrag[cur]);
                        mma_16816(acc[1][2 * q + 1], afrag[kk][1], bfrag[cur] + 2);
                    }
                }

                // The next chunk's halo is gathered while this step's mma drain, then
                // stored once every warp has finished reading the current halo.
                if (tap == kTaps - 1 && chunk + 1 < nchunks) {
#pragma unroll 1
                    for (int round = 0; round < Tile::kHaloRounds; ++round) {
                        uint4 pre[kHaloRound];
                        gather_halo<Tile>(pre, Xn, cin, h, w, (chunk + 1) * kBK, oh0, ow0,
                                          pad_mode, tid, round);
                        if (round == 0) {
                            __syncthreads();
                        }
                        store_halo<Tile>(pre, hs, tid, round);
                    }
                }
                if (++tap == kTaps) {
                    tap = 0;
                    ++chunk;
                }
                stage = stage + 1 == kBStages ? 0 : stage + 1;
            }

            const long long spatial = static_cast<long long>(out_h) * out_w;
            __half* Yn = Y + static_cast<long long>(ni) * cout * spatial;
#pragma unroll
            for (int j = 0; j < 2; ++j) {
#pragma unroll
                for (int hh = 0; hh < 2; ++hh) {
                    const int n = n0 + warp_n * 32 + j * 16 + (lane >> 2) + hh * 8;
                    if (n >= cout) {
                        continue;
                    }
                    const float b = bias ? __half2float(bias[n]) : 0.0f;
                    __half* Yc = Yn + static_cast<long long>(n) * spatial;
#pragma unroll
                    for (int g = 0; g < Tile::kMGroups; ++g) {
                        const int m = warp_m * Tile::kWarpM + g * 8 + (lane & 3) * 2;
                        const int oh = oh0 + m / TW;
                        const int ow = ow0 + m % TW;
                        if (oh >= out_h || ow >= out_w) {
                            continue;
                        }
                        const float v0 = device::apply_activation(acc[j][g][hh * 2] + b, activation);
                        const float v1 =
                            device::apply_activation(acc[j][g][hh * 2 + 1] + b, activation);
                        __half* dst = Yc + static_cast<long long>(oh) * out_w + ow;
                        if (ow + 1 < out_w && (reinterpret_cast<uintptr_t>(dst) & 3u) == 0) {
                            *reinterpret_cast<__half2*>(dst) = __floats2half2_rn(v0, v1);
                        } else {
                            dst[0] = __float2half_rn(v0);
                            if (ow + 1 < out_w) {
                                dst[1] = __float2half_rn(v1);
                            }
                        }
                    }
                }
            }
#else
            (void)X;
            (void)Wp;
            (void)bias;
            (void)Y;
            (void)cin;
            (void)h;
            (void)w;
            (void)cout;
            (void)out_h;
            (void)out_w;
            (void)pad_mode;
            (void)activation;
#endif
        }

        __global__ void permute_weight_taps_kernel(const __half* __restrict__ w,
                                                   __half* __restrict__ wp, const int cout,
                                                   const int cin) {
            const int total = kTaps * cout * cin;
            for (int idx = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x); idx < total;
                 idx += static_cast<int>(blockDim.x * gridDim.x)) {
                const int ic = idx % cin;
                const int rest = idx / cin;
                const int n = rest % cout;
                const int tap = rest / cout;
                wp[idx] = w[(static_cast<long long>(n) * cin + ic) * kTaps + tap];
            }
        }

        __device__ int g_compiled_arch = 0;

        __global__ void compiled_arch_probe_kernel() {
#if defined(__CUDA_ARCH__)
            g_compiled_arch = __CUDA_ARCH__;
#endif
        }

        template <int BN, int TH, int TW>
        void launch_conv3x3_mma(const __half* x, const __half* wp, const __half* b, __half* y,
                                const int n, const int cin, const int h, const int w,
                                const int cout, const int out_h, const int out_w,
                                const int pad_mode, const int activation,
                                const cudaStream_t stream) {
            const int tiles = ((out_w + TW - 1) / TW) * ((out_h + TH - 1) / TH);
            dim3 grid((cout + BN - 1) / BN, tiles, n);
            conv3x3_mma_kernel<BN, TH, TW><<<grid, kThreads, 0, stream>>>(
                x, wp, b, y, cin, h, w, cout, out_h, out_w, pad_mode, activation);
            LFS_CUDA_LAUNCH_CHECK(stream, "nn.conv.implicit_3x3_mma");
        }

    } // namespace

    bool conv3x3_mma_available() {
        static int cached[16] = {};
        int device = 0;
        LFS_CUDA_CHECK(cudaGetDevice(&device));
        if (device < 0 || device >= 16) {
            return false;
        }
        if (cached[device] == 0) {
            int major = 0;
            LFS_CUDA_CHECK(cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, device));
            int compiled = 0;
            if (major >= 8) {
                compiled_arch_probe_kernel<<<1, 1>>>();
                LFS_CUDA_CHECK(cudaGetLastError());
                LFS_CUDA_CHECK(cudaMemcpyFromSymbol(&compiled, g_compiled_arch, sizeof(int)));
            }
            cached[device] = (major >= 8 && compiled >= 800) ? 1 : -1;
        }
        return cached[device] > 0;
    }

    void conv3x3_weight_taps(const void* weight, void* weight_taps, const int cout, const int cin,
                             const cudaStream_t stream) {
        const int total = kTaps * cout * cin;
        if (total <= 0) {
            return;
        }
        const int blocks = std::min(2048, (total + 255) / 256);
        permute_weight_taps_kernel<<<blocks, 256, 0, stream>>>(
            static_cast<const __half*>(weight), static_cast<__half*>(weight_taps), cout, cin);
        LFS_CUDA_LAUNCH_CHECK(stream, "nn.conv.weight_taps");
    }

    void conv2d_implicit_3x3_mma(const void* input, const void* weight_taps, const void* bias,
                                 void* output, const int n, const int cin, const int h,
                                 const int w, const int cout, const int out_h, const int out_w,
                                 const int pad_mode, const int activation,
                                 const cudaStream_t stream) {
        LFS_ASSERT_MSG(cin % 8 == 0, "tensor-core 3x3 conv needs C_in % 8 == 0");
        auto* x = static_cast<const __half*>(input);
        auto* wp = static_cast<const __half*>(weight_taps);
        auto* b = static_cast<const __half*>(bias);
        auto* y = static_cast<__half*>(output);

        // BN=128 amortises the halo gather over twice the channels; it needs at
        // least two blocks per SM in flight to hide the per-step barrier.
        const int tiles = ((out_w + 15) / 16) * ((out_h + 7) / 8);
        constexpr int kMinBlocks = 256;
        if (cout % 128 == 0 && tiles * (cout / 128) >= kMinBlocks) {
            launch_conv3x3_mma<128, 8, 16>(x, wp, b, y, n, cin, h, w, cout, out_h, out_w, pad_mode,
                                           activation, stream);
        } else {
            launch_conv3x3_mma<64, 8, 16>(x, wp, b, y, n, cin, h, w, cout, out_h, out_w, pad_mode,
                                          activation, stream);
        }
    }

} // namespace lfs::core::nn::kernels

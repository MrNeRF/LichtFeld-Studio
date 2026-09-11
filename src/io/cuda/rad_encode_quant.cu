/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

// GPU side of the RAD chunk encoders for the streaming writer. Output bytes
// must match rad.cpp's PropertyEncoder bit for bit, which constrains the
// arithmetic: compiled with --fmad=false and without -use_fast_math so
// subtract/divide/multiply/round sequences round exactly like the host
// (mirrors lod_page_dequant_cuda.cu). Only pure-arithmetic encodings live
// here; libm-dependent planes (ln_f16 scales, oct88r8 orientation) stay on
// the CPU. Reductions reproduce the CPU fold comparators; like the CPU
// encoders they assume NaN-free input, and a +/-0 tie in the rgb min/max can
// in principle pick the other zero's sign bit.

#include "rad_encode_quant.hpp"

#include "core/cuda_error.hpp"
#include "core/logger.hpp"
#include "io/formats/rad_dequant_math.hpp"

#include <cstring>
#include <cuda_runtime.h>
#include <limits>
#include <vector>

namespace lfs::io::cuda {

    namespace {

        namespace radmath = lfs::io::radmath;

        constexpr unsigned kThreads = 256;
        constexpr std::uint32_t kNumBands = 3;

        // SH bands as encoded by encode_rad_chunk: sh1 coeffs [0,3), sh2
        // [3,8), sh3 [8,15); a band is present when sh_coeffs covers it.
        __host__ __device__ constexpr std::uint32_t bandStart(const std::uint32_t b) {
            return b == 0 ? 0u : (b == 1 ? 3u : 8u);
        }
        __host__ __device__ constexpr std::uint32_t bandCoeffs(const std::uint32_t b) {
            return b == 0 ? 3u : (b == 1 ? 5u : 7u);
        }
        __host__ __device__ constexpr std::uint32_t bandRequiredCoeffs(const std::uint32_t b) {
            return bandStart(b) + bandCoeffs(b);
        }

        struct ChunkDesc {
            std::uint32_t count;
            std::uint32_t sh_coeffs; // 0 when the chunk carries no shN
            std::uint32_t means_in;  // float offsets into the input arena
            std::uint32_t alpha_in;
            std::uint32_t rgb_in;
            std::uint32_t sh_in;
            std::uint32_t center_out; // byte offsets into the output arena
            std::uint32_t alpha_out;
            std::uint32_t rgb_out;
            std::uint32_t sh_out[kNumBands];
        };

        struct ChunkStats {
            float rgb_min;
            float rgb_max;
            float alpha_max;
            float sh_max[kNumBands];
        };

        // encode_r8: clamp(round((v - min) / range * 255), 0, 255).
        __device__ inline std::uint8_t quantR8(const float v, const float min_val, const float range) {
            const float r = roundf((v - min_val) / range * 255.0f);
            const float c = r < 0.0f ? 0.0f : (255.0f < r ? 255.0f : r);
            return static_cast<std::uint8_t>(c);
        }

        // CPU fold comparators: std::min(m, v) == (v < m ? v : m),
        // std::max(m, v) == (m < v ? v : m).
        __device__ inline float combineMin(const float a, const float b) { return b < a ? b : a; }
        __device__ inline float combineMax(const float a, const float b) { return a < b ? b : a; }

        template <typename Combine>
        __device__ inline float blockReduce(const float v, float* const shared, const Combine combine) {
            shared[threadIdx.x] = v;
            __syncthreads();
            for (unsigned s = kThreads / 2; s > 0; s >>= 1) {
                if (threadIdx.x < s) {
                    shared[threadIdx.x] = combine(shared[threadIdx.x], shared[threadIdx.x + s]);
                }
                __syncthreads();
            }
            const float r = shared[0];
            __syncthreads();
            return r;
        }

        // gridDim = (chunks, 5): plane 0 alpha max, 1 rgb min/max, 2-4 SH
        // band |max|.
        __global__ void reduceChunkStatsKernel(const ChunkDesc* const descs,
                                               const float* const in,
                                               ChunkStats* const stats) {
            __shared__ float sa[kThreads];
            const ChunkDesc d = descs[blockIdx.x];
            const std::uint32_t plane = blockIdx.y;
            const unsigned tid = threadIdx.x;

            if (plane == 0) {
                float m = 0.0f;
                for (std::uint32_t j = tid; j < d.count; j += kThreads) {
                    m = combineMax(m, in[d.alpha_in + j]);
                }
                const float r = blockReduce(m, sa, combineMax);
                if (tid == 0) {
                    stats[blockIdx.x].alpha_max = r;
                }
                return;
            }
            if (plane == 1) {
                const std::uint32_t n = d.count * 3u;
                float mn = in[d.rgb_in];
                float mx = mn;
                for (std::uint32_t j = tid; j < n; j += kThreads) {
                    const float v = in[d.rgb_in + j];
                    mn = combineMin(mn, v);
                    mx = combineMax(mx, v);
                }
                const float rmn = blockReduce(mn, sa, combineMin);
                const float rmx = blockReduce(mx, sa, combineMax);
                if (tid == 0) {
                    stats[blockIdx.x].rgb_min = rmn;
                    stats[blockIdx.x].rgb_max = rmx;
                }
                return;
            }

            const std::uint32_t b = plane - 2u;
            if (d.sh_coeffs < bandRequiredCoeffs(b)) {
                if (tid == 0) {
                    stats[blockIdx.x].sh_max[b] = 0.0f;
                }
                return;
            }
            const std::uint32_t dims = bandCoeffs(b) * 3u;
            const std::uint32_t stride = d.sh_coeffs * 3u;
            const std::uint32_t base = d.sh_in + bandStart(b) * 3u;
            const std::uint32_t n = d.count * dims;
            float m = 0.0f;
            for (std::uint32_t j = tid; j < n; j += kThreads) {
                const std::uint32_t i = j / dims;
                const std::uint32_t k = j - i * dims;
                m = combineMax(m, fabsf(in[base + i * stride + k]));
            }
            const float r = blockReduce(m, sa, combineMax);
            if (tid == 0) {
                // encode_s8 floor: max(max_abs, 1e-6f).
                stats[blockIdx.x].sh_max[b] = r < 1e-6f ? 1e-6f : r;
            }
        }

        // gridDim = (chunks, 6): plane 0 center, 1 alpha, 2 rgb, 3-5 SH bands.
        __global__ void quantChunkPlanesKernel(const ChunkDesc* const descs,
                                               const float* const in,
                                               const ChunkStats* const stats,
                                               std::uint8_t* const out,
                                               const float alpha_forced_max) {
            const ChunkDesc d = descs[blockIdx.x];
            const std::uint32_t plane = blockIdx.y;
            const unsigned tid = threadIdx.x;

            if (plane == 0) { // center, f32_lebytes byte shuffle
                const std::uint32_t stride = d.count * 3u;
                const std::uint32_t n = d.count * 12u;
                for (std::uint32_t j = tid; j < n; j += kThreads) {
                    const std::uint32_t b = j / stride;
                    const std::uint32_t rem = j - b * stride;
                    const std::uint32_t dd = rem / d.count;
                    const std::uint32_t i = rem - dd * d.count;
                    const std::uint32_t bits = radmath::floatToBits(in[d.means_in + i * 3u + dd]);
                    out[d.center_out + j] = static_cast<std::uint8_t>((bits >> (8u * b)) & 0xFFu);
                }
                return;
            }
            if (plane == 1) { // alpha: f16 above 1, else r8 over [0, forced max]
                if (stats[blockIdx.x].alpha_max > 1.0f) {
                    for (std::uint32_t j = tid; j < d.count; j += kThreads) {
                        const std::uint16_t h = radmath::floatToHalf(in[d.alpha_in + j]);
                        out[d.alpha_out + j * 2u] = static_cast<std::uint8_t>(h & 0xFFu);
                        out[d.alpha_out + j * 2u + 1u] = static_cast<std::uint8_t>((h >> 8u) & 0xFFu);
                    }
                } else {
                    for (std::uint32_t j = tid; j < d.count; j += kThreads) {
                        out[d.alpha_out + j] = quantR8(in[d.alpha_in + j], 0.0f, alpha_forced_max);
                    }
                }
                return;
            }
            if (plane == 2) { // rgb, r8_delta along i within each dimension
                float range = stats[blockIdx.x].rgb_max - stats[blockIdx.x].rgb_min;
                if (range < 1e-7f) {
                    range = 1e-7f;
                }
                const float mn = stats[blockIdx.x].rgb_min;
                const std::uint32_t n = d.count * 3u;
                for (std::uint32_t j = tid; j < n; j += kThreads) {
                    const std::uint32_t dd = j / d.count;
                    const std::uint32_t i = j - dd * d.count;
                    const std::uint8_t q = quantR8(in[d.rgb_in + i * 3u + dd], mn, range);
                    const std::uint8_t prev =
                        i > 0u ? quantR8(in[d.rgb_in + (i - 1u) * 3u + dd], mn, range)
                               : static_cast<std::uint8_t>(0);
                    out[d.rgb_out + j] = static_cast<std::uint8_t>(q - prev);
                }
                return;
            }

            const std::uint32_t b = plane - 3u; // SH band, s8
            if (d.sh_coeffs < bandRequiredCoeffs(b)) {
                return;
            }
            const float max_abs = stats[blockIdx.x].sh_max[b];
            const std::uint32_t dims = bandCoeffs(b) * 3u;
            const std::uint32_t stride = d.sh_coeffs * 3u;
            const std::uint32_t base = d.sh_in + bandStart(b) * 3u;
            const std::uint32_t n = d.count * dims;
            for (std::uint32_t j = tid; j < n; j += kThreads) {
                const std::uint32_t dd = j / d.count;
                const std::uint32_t i = j - dd * d.count;
                const float scaled = in[base + i * stride + dd] / max_abs * 127.0f;
                const float r = roundf(scaled);
                const float c = r < -127.0f ? -127.0f : (127.0f < r ? 127.0f : r);
                out[d.sh_out[b] + j] = static_cast<std::uint8_t>(static_cast<std::int8_t>(c));
            }
        }

    } // namespace

    bool rad_encode_gpu_available() {
        static const bool available = [] {
            int n = 0;
            return cudaGetDeviceCount(&n) == cudaSuccess && n > 0;
        }();
        return available;
    }

    struct RadEncodeGpuQuantizer::Impl {
        cudaStream_t stream = nullptr;
        float* h_in = nullptr;
        std::uint8_t* h_out = nullptr;
        ChunkDesc* h_desc = nullptr;
        ChunkStats* h_stats = nullptr;
        float* d_in = nullptr;
        std::uint8_t* d_out = nullptr;
        ChunkDesc* d_desc = nullptr;
        ChunkStats* d_stats = nullptr;
        std::size_t in_cap = 0;  // floats
        std::size_t out_cap = 0; // bytes
        std::size_t desc_cap = 0;
        bool failed = false;

        bool ensure(const std::size_t in_floats, const std::size_t out_bytes,
                    const std::size_t n_chunks) {
            if (stream == nullptr && cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking) != cudaSuccess) {
                return false;
            }
            if (in_floats > in_cap) {
                cudaFreeHost(h_in);
                cudaFree(d_in);
                h_in = nullptr;
                d_in = nullptr;
                in_cap = 0;
                if (cudaMallocHost(&h_in, in_floats * sizeof(float)) != cudaSuccess ||
                    cudaMalloc(&d_in, in_floats * sizeof(float)) != cudaSuccess) {
                    return false;
                }
                in_cap = in_floats;
            }
            if (out_bytes > out_cap) {
                cudaFreeHost(h_out);
                cudaFree(d_out);
                h_out = nullptr;
                d_out = nullptr;
                out_cap = 0;
                if (cudaMallocHost(&h_out, out_bytes) != cudaSuccess ||
                    cudaMalloc(&d_out, out_bytes) != cudaSuccess) {
                    return false;
                }
                out_cap = out_bytes;
            }
            if (n_chunks > desc_cap) {
                cudaFreeHost(h_desc);
                cudaFreeHost(h_stats);
                cudaFree(d_desc);
                cudaFree(d_stats);
                h_desc = nullptr;
                h_stats = nullptr;
                d_desc = nullptr;
                d_stats = nullptr;
                desc_cap = 0;
                if (cudaMallocHost(&h_desc, n_chunks * sizeof(ChunkDesc)) != cudaSuccess ||
                    cudaMallocHost(&h_stats, n_chunks * sizeof(ChunkStats)) != cudaSuccess ||
                    cudaMalloc(&d_desc, n_chunks * sizeof(ChunkDesc)) != cudaSuccess ||
                    cudaMalloc(&d_stats, n_chunks * sizeof(ChunkStats)) != cudaSuccess) {
                    return false;
                }
                desc_cap = n_chunks;
            }
            return true;
        }

        ~Impl() {
            cudaFreeHost(h_in);
            cudaFreeHost(h_out);
            cudaFreeHost(h_desc);
            cudaFreeHost(h_stats);
            cudaFree(d_in);
            cudaFree(d_out);
            cudaFree(d_desc);
            cudaFree(d_stats);
            if (stream != nullptr) {
                cudaStreamDestroy(stream);
            }
        }
    };

    RadEncodeGpuQuantizer::RadEncodeGpuQuantizer()
        : impl_(std::make_unique<Impl>()) {}

    RadEncodeGpuQuantizer::~RadEncodeGpuQuantizer() = default;

    bool RadEncodeGpuQuantizer::quantize_batch(const std::span<const RadEncodeQuantChunkIn> chunks,
                                               const int sh_coeffs,
                                               const bool lod_tree,
                                               const std::span<RadEncodeQuantChunkOut> out) {
        auto& s = *impl_;
        if (s.failed || chunks.size() != out.size()) {
            return false;
        }
        const float alpha_forced_max = lod_tree ? 2.0f : 1.0f;
        const std::size_t n = chunks.size();
        if (n == 0) {
            return true;
        }

        // Arenas span the whole batch: out views must all stay valid at once.
        std::size_t in_floats = 0;
        std::size_t out_bytes = 0;
        std::vector<ChunkDesc> descs(n);
        for (std::size_t c = 0; c < n; ++c) {
            const auto& chunk = chunks[c];
            if (chunk.count == 0 || chunk.means == nullptr || chunk.alpha == nullptr ||
                chunk.rgb == nullptr) {
                return false;
            }
            const std::uint32_t count = chunk.count;
            const std::uint32_t chunk_sh =
                chunk.shN != nullptr ? static_cast<std::uint32_t>(sh_coeffs) : 0u;
            auto& d = descs[c];
            d.count = count;
            d.sh_coeffs = chunk_sh;
            d.means_in = static_cast<std::uint32_t>(in_floats);
            in_floats += static_cast<std::size_t>(count) * 3u;
            d.alpha_in = static_cast<std::uint32_t>(in_floats);
            in_floats += count;
            d.rgb_in = static_cast<std::uint32_t>(in_floats);
            in_floats += static_cast<std::size_t>(count) * 3u;
            d.sh_in = static_cast<std::uint32_t>(in_floats);
            in_floats += static_cast<std::size_t>(count) * chunk_sh * 3u;

            d.center_out = static_cast<std::uint32_t>(out_bytes);
            out_bytes += static_cast<std::size_t>(count) * 12u;
            d.alpha_out = static_cast<std::uint32_t>(out_bytes);
            out_bytes += static_cast<std::size_t>(count) * 2u;
            d.rgb_out = static_cast<std::uint32_t>(out_bytes);
            out_bytes += static_cast<std::size_t>(count) * 3u;
            for (std::uint32_t b = 0; b < kNumBands; ++b) {
                d.sh_out[b] = static_cast<std::uint32_t>(out_bytes);
                if (chunk_sh >= bandRequiredCoeffs(b)) {
                    out_bytes += static_cast<std::size_t>(count) * bandCoeffs(b) * 3u;
                }
            }
            if (in_floats > std::numeric_limits<std::uint32_t>::max() ||
                out_bytes > std::numeric_limits<std::uint32_t>::max()) {
                return false;
            }
        }

        if (!s.ensure(in_floats, out_bytes, n)) {
            s.failed = true;
            return false;
        }

        for (std::size_t c = 0; c < n; ++c) {
            const auto& chunk = chunks[c];
            const auto& d = descs[c];
            std::memcpy(s.h_in + d.means_in, chunk.means,
                        static_cast<std::size_t>(d.count) * 3u * sizeof(float));
            std::memcpy(s.h_in + d.alpha_in, chunk.alpha, d.count * sizeof(float));
            std::memcpy(s.h_in + d.rgb_in, chunk.rgb,
                        static_cast<std::size_t>(d.count) * 3u * sizeof(float));
            if (d.sh_coeffs > 0u) {
                std::memcpy(s.h_in + d.sh_in, chunk.shN,
                            static_cast<std::size_t>(d.count) * d.sh_coeffs * 3u * sizeof(float));
            }
            s.h_desc[c] = d;
        }

        const dim3 reduce_grid(static_cast<unsigned>(n), 2u + kNumBands);
        const dim3 quant_grid(static_cast<unsigned>(n), 3u + kNumBands);
        const bool ok =
            cudaMemcpyAsync(s.d_in, s.h_in, in_floats * sizeof(float),
                            cudaMemcpyHostToDevice, s.stream) == cudaSuccess &&
            cudaMemcpyAsync(s.d_desc, s.h_desc, n * sizeof(ChunkDesc),
                            cudaMemcpyHostToDevice, s.stream) == cudaSuccess &&
            // r8 alpha fills only half of its reserved f16-sized slot; zero
            // the arena so the D2H copy reads defined memory.
            cudaMemsetAsync(s.d_out, 0, out_bytes, s.stream) == cudaSuccess;
        if (!ok) {
            s.failed = true;
            return false;
        }
        // Hybrid: throw-style launch/await attribution, bool API preserved.
        // Named catch(const lfs::Exception&) is unavailable in .cu TUs
        // (core/error.hpp is host-C++23-only); lfs::Exception derives from
        // std::exception so the base catch covers it.
        try {
            const auto rad_ticket =
                ::lfs::core::cuda_record_range(s.stream, "io.rad.quantize_batch");
            reduceChunkStatsKernel<<<reduce_grid, kThreads, 0, s.stream>>>(s.d_desc, s.d_in,
                                                                           s.d_stats);
            LFS_CUDA_LAUNCH_CHECK(s.stream, "io.rad.reduce_chunk_stats");
            quantChunkPlanesKernel<<<quant_grid, kThreads, 0, s.stream>>>(
                s.d_desc, s.d_in, s.d_stats, s.d_out, alpha_forced_max);
            const bool done =
                cudaGetLastError() == cudaSuccess &&
                cudaMemcpyAsync(s.h_out, s.d_out, out_bytes, cudaMemcpyDeviceToHost,
                                s.stream) == cudaSuccess &&
                cudaMemcpyAsync(s.h_stats, s.d_stats, n * sizeof(ChunkStats),
                                cudaMemcpyDeviceToHost, s.stream) == cudaSuccess;
            if (!done) {
                s.failed = true;
                return false;
            }
            LFS_CUDA_AWAIT(rad_ticket, cudaStreamSynchronize(s.stream),
                           "io.rad.quantize_batch_sync");
        } catch (const std::exception& e) {
            LOG_WARN(
                "RAD GPU quantize_batch CUDA failure (CPU fallback exists): {}", e.what());
            s.failed = true;
            return false;
        } catch (...) {
            LOG_WARN(
                "RAD GPU quantize_batch unknown failure (CPU fallback exists)");
            s.failed = true;
            return false;
        }

        for (std::size_t c = 0; c < n; ++c) {
            const auto& d = descs[c];
            const auto& st = s.h_stats[c];
            auto& o = out[c];
            o.center = s.h_out + d.center_out;
            o.alpha = s.h_out + d.alpha_out;
            o.alpha_f16 = st.alpha_max > 1.0f;
            o.alpha_min = 0.0f;
            o.alpha_max = alpha_forced_max;
            o.rgb = s.h_out + d.rgb_out;
            o.rgb_min = st.rgb_min;
            o.rgb_max = st.rgb_max;
            for (std::uint32_t b = 0; b < kNumBands; ++b) {
                const bool present = d.sh_coeffs >= bandRequiredCoeffs(b);
                o.sh[b] = present ? s.h_out + d.sh_out[b] : nullptr;
                o.sh_max_abs[b] = present ? st.sh_max[b] : 0.0f;
            }
        }
        return true;
    }

} // namespace lfs::io::cuda

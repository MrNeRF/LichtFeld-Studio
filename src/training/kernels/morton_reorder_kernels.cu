/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "morton_reorder_kernels.hpp"

#include "core/cuda/sh_layout.cuh"
#include "core/cuda_error.hpp"
#include "core/logger.hpp"
#include "kernel_stream.hpp"
#include "lfs/training/joint_adam_codec.cuh"
#include "lfs/training/joint_adam_codec.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cub/device/device_radix_sort.cuh>
#include <cuda_runtime.h>
#include <stdexcept>
#include <thrust/execution_policy.h>
#include <thrust/sequence.h>
#include <vector>

namespace lfs::training::kernels {
    namespace {

        constexpr int kThreads = 256;
        constexpr float kInf = 1e30f;

        __device__ __forceinline__ std::uint32_t part1by2(std::uint32_t x) {
            x &= 0x000003ffu;
            x = (x ^ (x << 16)) & 0xff0000ffu;
            x = (x ^ (x << 8)) & 0x0300f00fu;
            x = (x ^ (x << 4)) & 0x030c30c3u;
            x = (x ^ (x << 2)) & 0x09249249u;
            return x;
        }

        __device__ __forceinline__ std::uint32_t encode_morton3(
            std::uint32_t x, std::uint32_t y, std::uint32_t z) {
            return (part1by2(z) << 2) + (part1by2(y) << 1) + part1by2(x);
        }

        __device__ __forceinline__ std::uint32_t quantize_axis(float v, float origin, float mul) {
            const float t = (v - origin) * mul;
            if (!(t > 0.0f)) {
                return 0u;
            }
            return min(1023u, static_cast<std::uint32_t>(t));
        }

        // Codes over a contracted space: p = (x - centre) / radius, and p * (2 - 1/|p|) / |p| beyond the unit
        // ball, so every point lands in the ball of radius 2 and the grid spans [-2, 2]^3. The bulk of the scene
        // keeps the inner half of the grid at a resolution set by its own extent: a few distant points must not
        // stretch a global bounding box and collapse the rest of the scene into a handful of cells.
        constexpr float kContractedHalfExtent = 2.0f;

        __global__ void morton_encode_kernel(
            const float* __restrict__ means,
            std::uint32_t* __restrict__ codes,
            int n,
            float cx, float cy, float cz, float inv_radius) {
            const int i = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
            if (i >= n) {
                return;
            }
            float x = (means[i * 3 + 0] - cx) * inv_radius;
            float y = (means[i * 3 + 1] - cy) * inv_radius;
            float z = (means[i * 3 + 2] - cz) * inv_radius;
            const float r = sqrtf(x * x + y * y + z * z);
            if (r > 1.0f) {
                const float f = (2.0f - 1.0f / r) / r;
                x *= f;
                y *= f;
                z *= f;
            }
            constexpr float origin = -kContractedHalfExtent;
            constexpr float mul = 1024.0f / (2.0f * kContractedHalfExtent);
            codes[i] = encode_morton3(quantize_axis(x, origin, mul), quantize_axis(y, origin, mul),
                                      quantize_axis(z, origin, mul));
        }

        __global__ void widen_order_kernel(
            const std::int32_t* __restrict__ order,
            std::int64_t* __restrict__ perm,
            int n) {
            const int i = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
            if (i < n) {
                perm[i] = order[i];
            }
        }

        constexpr int kMortonBits = 30; // 10 bits per axis

        [[nodiscard]] std::size_t align_scratch(const std::size_t bytes) {
            return (bytes + 255) & ~std::size_t{255};
        }

        [[nodiscard]] std::size_t radix_sort_temp_bytes(const int n) {
            std::size_t bytes = 0;
            cub::DoubleBuffer<std::uint32_t> keys(nullptr, nullptr);
            cub::DoubleBuffer<std::int32_t> values(nullptr, nullptr);
            LFS_CUDA_CHECK_MSG(
                cub::DeviceRadixSort::SortPairs(nullptr, bytes, keys, values, n, 0, kMortonBits),
                "morton argsort workspace query (n={})", n);
            return bytes;
        }

        __device__ __forceinline__ float4 reduce_us_block(
            float u_min, float u_max, float s_min, float s_max) {
            __shared__ float s_lo[kThreads];
            __shared__ float s_hi[kThreads];
            __shared__ float t_lo[kThreads];
            __shared__ float t_hi[kThreads];
            const int lane = static_cast<int>(threadIdx.x);
            s_lo[lane] = u_min;
            s_hi[lane] = u_max;
            t_lo[lane] = s_min;
            t_hi[lane] = s_max;
            __syncthreads();
            for (int stride = kThreads / 2; stride > 0; stride >>= 1) {
                if (lane < stride) {
                    s_lo[lane] = fminf(s_lo[lane], s_lo[lane + stride]);
                    s_hi[lane] = fmaxf(s_hi[lane], s_hi[lane + stride]);
                    t_lo[lane] = fminf(t_lo[lane], t_lo[lane + stride]);
                    t_hi[lane] = fmaxf(t_hi[lane], t_hi[lane + stride]);
                }
                __syncthreads();
            }
            if (s_lo[0] > s_hi[0]) {
                return make_float4(0.0f, 0.0f, 0.0f, 0.0f);
            }
            return make_float4(s_lo[0], s_hi[0], t_lo[0], t_hi[0]);
        }

        template <int BITS>
        __global__ void joint_permute_contiguous_bounds_cu(
            const std::uint8_t* __restrict__ src_packed,
            const float* __restrict__ src_bounds,
            float* __restrict__ dst_bounds,
            const std::int64_t* __restrict__ perm,
            int n_prims,
            int n_attr) {
            using C = lfs::training::joint_adam::DeviceCodec<BITS>;
            const int prim = static_cast<int>(blockIdx.x) * kThreads + static_cast<int>(threadIdx.x);
            const bool in_range = prim < n_prims && n_attr > 0;
            float u_min = kInf, u_max = -kInf, s_min = kInf, s_max = -kInf;
            if (in_range) {
                const std::int64_t src = perm[prim];
                if (src >= 0 && src < n_prims) {
                    const float4 src_mm =
                        *reinterpret_cast<const float4*>(src_bounds + 4 * static_cast<int>(src / kThreads));
                    for (int a = 0; a < n_attr; ++a) {
                        const float2 us = C::decode_us(
                            src_packed, src * static_cast<std::int64_t>(n_attr) + a, src_mm);
                        u_min = fminf(u_min, us.x);
                        u_max = fmaxf(u_max, us.x);
                        s_min = fminf(s_min, us.y);
                        s_max = fmaxf(s_max, us.y);
                    }
                }
            }
            const float4 mm = reduce_us_block(u_min, u_max, s_min, s_max);
            if (threadIdx.x == 0) {
                *reinterpret_cast<float4*>(dst_bounds + 4 * static_cast<int>(blockIdx.x)) = mm;
            }
        }

        template <int BITS>
        __global__ void joint_permute_contiguous_encode_cu(
            const std::uint8_t* __restrict__ src_packed,
            const float* __restrict__ src_bounds,
            std::uint8_t* __restrict__ dst_packed,
            const float* __restrict__ dst_bounds,
            const std::int64_t* __restrict__ perm,
            int n_prims,
            int n_attr) {
            using C = lfs::training::joint_adam::DeviceCodec<BITS>;
            const int prim = static_cast<int>(blockIdx.x) * kThreads + static_cast<int>(threadIdx.x);
            if (prim >= n_prims || n_attr <= 0) {
                return;
            }
            const std::int64_t src = perm[prim];
            if (src < 0 || src >= n_prims) {
                return;
            }
            const float4 src_mm =
                *reinterpret_cast<const float4*>(src_bounds + 4 * static_cast<int>(src / kThreads));
            const float4 dst_mm =
                *reinterpret_cast<const float4*>(dst_bounds + 4 * static_cast<int>(prim / kThreads));
            for (int a = 0; a < n_attr; ++a) {
                const float2 us = C::decode_us(
                    src_packed, src * static_cast<std::int64_t>(n_attr) + a, src_mm);
                C::encode_us(dst_packed, static_cast<std::int64_t>(prim) * n_attr + a,
                             us.x, us.y, dst_mm);
            }
        }

        __device__ __forceinline__ std::int64_t sh_moment_cell(
            std::uint32_t prim, std::uint32_t k, int c, std::uint32_t slots) {
            constexpr std::uint32_t R = lfs::core::kShReorderSize;
            const std::uint32_t slot = (prim / R) * (slots * R) + k * R + (prim % R);
            return static_cast<std::int64_t>(slot) * 4 + c;
        }

        template <int BITS>
        __global__ void joint_permute_shN_bounds_cu(
            const std::uint8_t* __restrict__ src_packed,
            const float* __restrict__ src_bounds,
            float* __restrict__ dst_bounds,
            const std::int64_t* __restrict__ perm,
            int n_prims,
            int slots_per_primitive) {
            using C = lfs::training::joint_adam::DeviceCodec<BITS>;
            const int prim = static_cast<int>(blockIdx.x) * kThreads + static_cast<int>(threadIdx.x);
            const bool in_range = prim < n_prims && slots_per_primitive > 0;
            float u_min = kInf, u_max = -kInf, s_min = kInf, s_max = -kInf;
            if (in_range) {
                const std::int64_t src = perm[prim];
                if (src >= 0 && src < n_prims) {
                    const float4 src_mm =
                        *reinterpret_cast<const float4*>(src_bounds + 4 * static_cast<int>(src / kThreads));
                    const auto slots = static_cast<std::uint32_t>(slots_per_primitive);
                    for (std::uint32_t k = 0; k < slots; ++k) {
                        for (int c = 0; c < 4; ++c) {
                            const float2 us = C::decode_us(
                                src_packed,
                                sh_moment_cell(static_cast<std::uint32_t>(src), k, c, slots),
                                src_mm);
                            u_min = fminf(u_min, us.x);
                            u_max = fmaxf(u_max, us.x);
                            s_min = fminf(s_min, us.y);
                            s_max = fmaxf(s_max, us.y);
                        }
                    }
                }
            }
            const float4 mm = reduce_us_block(u_min, u_max, s_min, s_max);
            if (threadIdx.x == 0) {
                *reinterpret_cast<float4*>(dst_bounds + 4 * static_cast<int>(blockIdx.x)) = mm;
            }
        }

        template <int BITS>
        __global__ void joint_permute_shN_encode_cu(
            const std::uint8_t* __restrict__ src_packed,
            const float* __restrict__ src_bounds,
            std::uint8_t* __restrict__ dst_packed,
            const float* __restrict__ dst_bounds,
            const std::int64_t* __restrict__ perm,
            int n_prims,
            int slots_per_primitive,
            int slot_begin,
            int slot_end) {
            using C = lfs::training::joint_adam::DeviceCodec<BITS>;
            const int prim = static_cast<int>(blockIdx.x) * kThreads + static_cast<int>(threadIdx.x);
            if (prim >= n_prims || slots_per_primitive <= 0) {
                return;
            }
            const std::int64_t src = perm[prim];
            if (src < 0 || src >= n_prims) {
                return;
            }
            const float4 src_mm =
                *reinterpret_cast<const float4*>(src_bounds + 4 * static_cast<int>(src / kThreads));
            const float4 dst_mm =
                *reinterpret_cast<const float4*>(dst_bounds + 4 * static_cast<int>(prim / kThreads));
            const auto slots = static_cast<std::uint32_t>(slots_per_primitive);
            const auto first = static_cast<std::uint32_t>(slot_begin);
            const auto group_slots = static_cast<std::uint32_t>(slot_end - slot_begin);
            for (std::uint32_t k = first; k < static_cast<std::uint32_t>(slot_end); ++k) {
                for (int c = 0; c < 4; ++c) {
                    const float2 us = C::decode_us(
                        src_packed,
                        sh_moment_cell(static_cast<std::uint32_t>(src), k, c, slots),
                        src_mm);
                    C::encode_us(
                        dst_packed,
                        sh_moment_cell(static_cast<std::uint32_t>(prim), k - first, c, group_slots),
                        us.x, us.y, dst_mm);
                }
            }
        }

        int grid_for_prims(int n_prims) {
            return std::max(1, (n_prims + kThreads - 1) / kThreads);
        }

        struct ContractionFrame {
            float centre[3] = {0.0f, 0.0f, 0.0f};
            float radius = 1.0f;
        };

        // Centre (per-axis median) and radius (median distance from it) of the splat means, from a strided
        // sample of at most kSample rows copied to the host. Runs once per reorder.
        ContractionFrame contraction_frame(const float* means, const int n, cudaStream_t stream) {
            constexpr int kSample = 65536;
            const int stride = std::max(1, n / kSample);
            const int m = (n + stride - 1) / stride;
            std::vector<float> host(static_cast<std::size_t>(m) * 3);
            LFS_CUDA_CHECK(cudaMemcpy2DAsync(host.data(), 3 * sizeof(float), means,
                                             static_cast<std::size_t>(stride) * 3 * sizeof(float), 3 * sizeof(float),
                                             static_cast<std::size_t>(m), cudaMemcpyDeviceToHost, stream));
            LFS_CUDA_CHECK(cudaStreamSynchronize(stream));
            // Non-finite rows would break the selection's ordering; they get code 0 either way.
            std::size_t finite = 0;
            for (int j = 0; j < m; ++j) {
                const float* p = host.data() + static_cast<std::size_t>(j) * 3;
                if (std::isfinite(p[0]) && std::isfinite(p[1]) && std::isfinite(p[2]))
                    std::copy(p, p + 3, host.begin() + static_cast<std::ptrdiff_t>(3 * finite++));
            }
            ContractionFrame frame;
            if (finite == 0)
                return frame;
            std::vector<float> values(finite);
            const auto median = [&] {
                const auto mid = values.begin() + static_cast<std::ptrdiff_t>(finite / 2);
                std::nth_element(values.begin(), mid, values.end());
                return *mid;
            };
            for (int k = 0; k < 3; ++k) {
                for (std::size_t j = 0; j < finite; ++j)
                    values[j] = host[j * 3 + k];
                frame.centre[k] = median();
            }
            for (std::size_t j = 0; j < finite; ++j) {
                const float* p = host.data() + j * 3;
                const float dx = p[0] - frame.centre[0], dy = p[1] - frame.centre[1], dz = p[2] - frame.centre[2];
                values[j] = std::sqrt(dx * dx + dy * dy + dz * dz);
            }
            const float radius = median();
            if (std::isfinite(radius) && radius > 0.0f)
                frame.radius = radius;
            return frame;
        }

    } // namespace

    std::size_t morton_permutation_scratch_bytes(const std::size_t n) {
        if (n == 0) {
            return 0;
        }
        return 4 * align_scratch(n * sizeof(std::uint32_t)) +
               align_scratch(radix_sort_temp_bytes(static_cast<int>(n)));
    }

    lfs::core::Tensor launch_morton_permutation(
        const lfs::core::Tensor& means,
        cudaStream_t stream,
        void* scratch,
        const std::size_t scratch_bytes) {
        using lfs::core::DataType;
        using lfs::core::Device;
        using lfs::core::Tensor;

        if (!means.is_valid() || means.ndim() != 2 || means.size(1) != 3 ||
            means.dtype() != DataType::Float32 || means.device() != Device::CUDA) {
            LOG_ERROR("morton permutation: means must be CUDA Float32 [N, 3]");
            return {};
        }
        const int n = static_cast<int>(means.size(0));
        if (n <= 0) {
            return {};
        }

        stream = resolve_stream(stream);
        if (means.stream() != stream) {
            lfs::core::waitForCUDAStream(stream, means.stream());
        }

        const ContractionFrame frame = contraction_frame(means.ptr<float>(), n, stream);

        // Codes, the int32 order and their sort double buffers; the caller
        // may lend idle memory for them, so only the int64 result is new.
        const std::size_t row_bytes = align_scratch(static_cast<std::size_t>(n) * sizeof(std::uint32_t));
        std::size_t temp_bytes = radix_sort_temp_bytes(n);
        const std::size_t needed = 4 * row_bytes + align_scratch(temp_bytes);
        Tensor owned_scratch;
        auto* base = static_cast<char*>(scratch);
        if (base == nullptr || scratch_bytes < needed) {
            owned_scratch = Tensor::empty_exact({needed}, DataType::UInt8);
            owned_scratch.set_stream(stream);
            base = static_cast<char*>(owned_scratch.data_ptr());
        }
        cub::DoubleBuffer<std::uint32_t> codes(reinterpret_cast<std::uint32_t*>(base),
                                               reinterpret_cast<std::uint32_t*>(base + row_bytes));
        cub::DoubleBuffer<std::int32_t> order(reinterpret_cast<std::int32_t*>(base + 2 * row_bytes),
                                              reinterpret_cast<std::int32_t*>(base + 3 * row_bytes));

        const int grid = grid_for_prims(n);
        morton_encode_kernel<<<grid, kThreads, 0, stream>>>(
            means.ptr<float>(),
            codes.Current(),
            n,
            frame.centre[0], frame.centre[1], frame.centre[2], 1.0f / frame.radius);
        LFS_CUDA_LAUNCH_CHECK(stream, "training.morton.encode");

        thrust::sequence(thrust::cuda::par_nosync.on(stream), order.Current(), order.Current() + n, 0);
        LFS_CUDA_CHECK_MSG(
            cub::DeviceRadixSort::SortPairs(base + 4 * row_bytes, temp_bytes, codes, order, n, 0,
                                            kMortonBits, stream),
            "morton argsort (n={})", n);

        auto indices = Tensor::empty_exact({static_cast<std::size_t>(n)}, DataType::Int64);
        indices.set_stream(stream);
        widen_order_kernel<<<grid, kThreads, 0, stream>>>(order.Current(), indices.ptr<std::int64_t>(), n);
        LFS_CUDA_LAUNCH_CHECK(stream, "training.morton.argsort");
        return indices;
    }

    void launch_joint_permute_contiguous(
        const std::uint8_t* src_packed,
        const float* src_bounds,
        std::uint8_t* dst_packed,
        float* dst_bounds,
        const std::int64_t* perm,
        int n_prims,
        int n_attr,
        int bits,
        cudaStream_t stream) {
        if (n_prims <= 0 || n_attr <= 0 || src_packed == nullptr || dst_packed == nullptr ||
            src_bounds == nullptr || dst_bounds == nullptr || perm == nullptr) {
            return;
        }
        stream = resolve_stream(stream);
        const int grid = grid_for_prims(n_prims);
        if (bits == 16) {
            joint_permute_contiguous_bounds_cu<16><<<grid, kThreads, 0, stream>>>(
                src_packed, src_bounds, dst_bounds, perm, n_prims, n_attr);
            LFS_CUDA_LAUNCH_CHECK(stream, "training.morton.joint_contig_bounds16");
            joint_permute_contiguous_encode_cu<16><<<grid, kThreads, 0, stream>>>(
                src_packed, src_bounds, dst_packed, dst_bounds, perm, n_prims, n_attr);
            LFS_CUDA_LAUNCH_CHECK(stream, "training.morton.joint_contig_encode16");
        } else if (bits == 8) {
            joint_permute_contiguous_bounds_cu<8><<<grid, kThreads, 0, stream>>>(
                src_packed, src_bounds, dst_bounds, perm, n_prims, n_attr);
            LFS_CUDA_LAUNCH_CHECK(stream, "training.morton.joint_contig_bounds8");
            joint_permute_contiguous_encode_cu<8><<<grid, kThreads, 0, stream>>>(
                src_packed, src_bounds, dst_packed, dst_bounds, perm, n_prims, n_attr);
            LFS_CUDA_LAUNCH_CHECK(stream, "training.morton.joint_contig_encode8");
        } else {
            throw std::runtime_error("joint permute contiguous: bits must be 8 or 16");
        }
    }

    void launch_joint_permute_shN(
        const std::uint8_t* src_packed,
        const float* src_bounds,
        std::uint8_t* dst_packed,
        float* dst_bounds,
        const std::int64_t* perm,
        int n_prims,
        int slots_per_primitive,
        int bits,
        cudaStream_t stream) {
        if (n_prims <= 0 || slots_per_primitive <= 0 || src_packed == nullptr ||
            dst_packed == nullptr || src_bounds == nullptr || dst_bounds == nullptr ||
            perm == nullptr) {
            return;
        }
        stream = resolve_stream(stream);
        const int grid = grid_for_prims(n_prims);
        if (bits == 8) {
            joint_permute_shN_bounds_cu<8><<<grid, kThreads, 0, stream>>>(
                src_packed, src_bounds, dst_bounds, perm, n_prims, slots_per_primitive);
            LFS_CUDA_LAUNCH_CHECK(stream, "training.morton.joint_shN_bounds8");
            joint_permute_shN_encode_cu<8><<<grid, kThreads, 0, stream>>>(
                src_packed, src_bounds, dst_packed, dst_bounds, perm, n_prims,
                slots_per_primitive, 0, slots_per_primitive);
            LFS_CUDA_LAUNCH_CHECK(stream, "training.morton.joint_shN_encode8");
        } else if (bits == 16) {
            joint_permute_shN_bounds_cu<16><<<grid, kThreads, 0, stream>>>(
                src_packed, src_bounds, dst_bounds, perm, n_prims, slots_per_primitive);
            LFS_CUDA_LAUNCH_CHECK(stream, "training.morton.joint_shN_bounds16");
            joint_permute_shN_encode_cu<16><<<grid, kThreads, 0, stream>>>(
                src_packed, src_bounds, dst_packed, dst_bounds, perm, n_prims,
                slots_per_primitive, 0, slots_per_primitive);
            LFS_CUDA_LAUNCH_CHECK(stream, "training.morton.joint_shN_encode16");
        } else {
            throw std::runtime_error("joint permute shN: bits must be 8 or 16");
        }
    }

    void launch_joint_permute_shN_grouped(
        std::uint8_t* packed,
        const float* src_bounds,
        float* dst_bounds,
        const std::int64_t* perm,
        int n_prims,
        int slots_per_primitive,
        int bits,
        std::uint8_t* group_scratch,
        std::size_t group_scratch_bytes,
        cudaStream_t stream) {
        if (n_prims <= 0 || slots_per_primitive <= 0 || packed == nullptr || src_bounds == nullptr ||
            dst_bounds == nullptr || perm == nullptr || group_scratch == nullptr) {
            return;
        }
        if (bits != 8 && bits != 16) {
            throw std::runtime_error("joint permute shN: bits must be 8 or 16");
        }
        stream = resolve_stream(stream);
        const int grid = grid_for_prims(n_prims);
        constexpr std::size_t R = lfs::core::kShReorderSize;
        const std::size_t tiles = (static_cast<std::size_t>(n_prims) + R - 1) / R;
        const std::size_t slot_tile_bytes =
            R * 4 * static_cast<std::size_t>(lfs::training::joint_adam::bytes_per_cell(bits));
        const std::size_t slots_per_group =
            std::clamp<std::size_t>(group_scratch_bytes / (tiles * slot_tile_bytes), 1,
                                    static_cast<std::size_t>(slots_per_primitive));
        if (group_scratch_bytes < tiles * slot_tile_bytes) {
            throw std::invalid_argument("joint permute shN: group scratch smaller than one slot");
        }

        if (bits == 8) {
            joint_permute_shN_bounds_cu<8><<<grid, kThreads, 0, stream>>>(
                packed, src_bounds, dst_bounds, perm, n_prims, slots_per_primitive);
            LFS_CUDA_LAUNCH_CHECK(stream, "training.morton.joint_shN_bounds_grouped8");
        } else {
            joint_permute_shN_bounds_cu<16><<<grid, kThreads, 0, stream>>>(
                packed, src_bounds, dst_bounds, perm, n_prims, slots_per_primitive);
            LFS_CUDA_LAUNCH_CHECK(stream, "training.morton.joint_shN_bounds_grouped16");
        }

        const std::size_t dst_pitch = static_cast<std::size_t>(slots_per_primitive) * slot_tile_bytes;
        for (std::size_t first = 0; first < static_cast<std::size_t>(slots_per_primitive);
             first += slots_per_group) {
            const std::size_t last =
                std::min(first + slots_per_group, static_cast<std::size_t>(slots_per_primitive));
            const std::size_t width = (last - first) * slot_tile_bytes;
            LFS_CUDA_CHECK(cudaMemsetAsync(group_scratch, 0, width * tiles, stream));
            if (bits == 8) {
                joint_permute_shN_encode_cu<8><<<grid, kThreads, 0, stream>>>(
                    packed, src_bounds, group_scratch, dst_bounds, perm, n_prims,
                    slots_per_primitive, static_cast<int>(first), static_cast<int>(last));
                LFS_CUDA_LAUNCH_CHECK(stream, "training.morton.joint_shN_encode_grouped8");
            } else {
                joint_permute_shN_encode_cu<16><<<grid, kThreads, 0, stream>>>(
                    packed, src_bounds, group_scratch, dst_bounds, perm, n_prims,
                    slots_per_primitive, static_cast<int>(first), static_cast<int>(last));
                LFS_CUDA_LAUNCH_CHECK(stream, "training.morton.joint_shN_encode_grouped16");
            }
            LFS_CUDA_CHECK(cudaMemcpy2DAsync(packed + first * slot_tile_bytes, dst_pitch, group_scratch,
                                             width, width, tiles, cudaMemcpyDeviceToDevice, stream));
        }
    }

} // namespace lfs::training::kernels

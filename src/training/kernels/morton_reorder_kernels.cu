/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "morton_reorder_kernels.hpp"

#include "core/cuda/sh_layout.cuh"
#include "core/cuda_error.hpp"
#include "core/logger.hpp"
#include "kernel_stream.hpp"
#include "lfs/training/joint_adam_codec.cuh"

#include <algorithm>
#include <cstdint>
#include <cuda_runtime.h>
#include <limits>
#include <stdexcept>
#include <thrust/device_ptr.h>
#include <thrust/execution_policy.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/sequence.h>
#include <thrust/sort.h>
#include <thrust/transform_reduce.h>

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

        __global__ void morton_encode_kernel(
            const float* __restrict__ means,
            std::int32_t* __restrict__ codes,
            int n,
            float min_x, float min_y, float min_z,
            float xmul, float ymul, float zmul) {
            const int i = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
            if (i >= n) {
                return;
            }
            const float x = means[i * 3 + 0];
            const float y = means[i * 3 + 1];
            const float z = means[i * 3 + 2];
            const std::uint32_t ix = quantize_axis(x, min_x, xmul);
            const std::uint32_t iy = quantize_axis(y, min_y, ymul);
            const std::uint32_t iz = quantize_axis(z, min_z, zmul);
            codes[i] = static_cast<std::int32_t>(encode_morton3(ix, iy, iz));
        }

        struct float3_minmax {
            float3 min_val;
            float3 max_val;
        };

        struct minmax_op {
            __host__ __device__ float3_minmax operator()(
                const float3_minmax& a, const float3_minmax& b) const {
                float3_minmax r;
                r.min_val.x = fminf(a.min_val.x, b.min_val.x);
                r.min_val.y = fminf(a.min_val.y, b.min_val.y);
                r.min_val.z = fminf(a.min_val.z, b.min_val.z);
                r.max_val.x = fmaxf(a.max_val.x, b.max_val.x);
                r.max_val.y = fmaxf(a.max_val.y, b.max_val.y);
                r.max_val.z = fmaxf(a.max_val.z, b.max_val.z);
                return r;
            }
        };

        struct position_to_minmax {
            const float* positions;
            __host__ __device__ explicit position_to_minmax(const float* pos) : positions(pos) {}
            __host__ __device__ float3_minmax operator()(int idx) const {
                float3 p;
                p.x = positions[idx * 3 + 0];
                p.y = positions[idx * 3 + 1];
                p.z = positions[idx * 3 + 2];
                return float3_minmax{p, p};
            }
        };

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
            int slots_per_primitive) {
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
            for (std::uint32_t k = 0; k < slots; ++k) {
                for (int c = 0; c < 4; ++c) {
                    const float2 us = C::decode_us(
                        src_packed,
                        sh_moment_cell(static_cast<std::uint32_t>(src), k, c, slots),
                        src_mm);
                    C::encode_us(
                        dst_packed,
                        sh_moment_cell(static_cast<std::uint32_t>(prim), k, c, slots),
                        us.x, us.y, dst_mm);
                }
            }
        }

        int grid_for_prims(int n_prims) {
            return std::max(1, (n_prims + kThreads - 1) / kThreads);
        }

    } // namespace

    lfs::core::Tensor launch_morton_permutation(
        const lfs::core::Tensor& means,
        cudaStream_t stream) {
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

        float3_minmax init;
        init.min_val = make_float3(
            std::numeric_limits<float>::infinity(),
            std::numeric_limits<float>::infinity(),
            std::numeric_limits<float>::infinity());
        init.max_val = make_float3(
            -std::numeric_limits<float>::infinity(),
            -std::numeric_limits<float>::infinity(),
            -std::numeric_limits<float>::infinity());
        position_to_minmax transform_op(means.ptr<float>());
        const float3_minmax bbox = thrust::transform_reduce(
            thrust::cuda::par_nosync.on(stream),
            thrust::counting_iterator<int>(0),
            thrust::counting_iterator<int>(n),
            transform_op,
            init,
            minmax_op());

        const float xlen = bbox.max_val.x - bbox.min_val.x;
        const float ylen = bbox.max_val.y - bbox.min_val.y;
        const float zlen = bbox.max_val.z - bbox.min_val.z;
        const float xmul = (xlen == 0.0f) ? 0.0f : 1024.0f / xlen;
        const float ymul = (ylen == 0.0f) ? 0.0f : 1024.0f / ylen;
        const float zmul = (zlen == 0.0f) ? 0.0f : 1024.0f / zlen;

        auto codes = Tensor::empty({static_cast<std::size_t>(n)}, Device::CUDA, DataType::Int32);
        auto indices = Tensor::empty({static_cast<std::size_t>(n)}, Device::CUDA, DataType::Int64);
        codes.set_stream(stream);
        indices.set_stream(stream);

        const int grid = grid_for_prims(n);
        morton_encode_kernel<<<grid, kThreads, 0, stream>>>(
            means.ptr<float>(),
            codes.ptr<std::int32_t>(),
            n,
            bbox.min_val.x, bbox.min_val.y, bbox.min_val.z,
            xmul, ymul, zmul);
        LFS_CUDA_LAUNCH_CHECK(stream, "training.morton.encode");

        auto idx_ptr = thrust::device_pointer_cast(indices.ptr<std::int64_t>());
        thrust::sequence(thrust::cuda::par_nosync.on(stream), idx_ptr, idx_ptr + n, std::int64_t{0});
        auto key_ptr = thrust::device_pointer_cast(codes.ptr<std::int32_t>());
        thrust::sort_by_key(
            thrust::cuda::par_nosync.on(stream),
            key_ptr, key_ptr + n, idx_ptr);
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
                slots_per_primitive);
            LFS_CUDA_LAUNCH_CHECK(stream, "training.morton.joint_shN_encode8");
        } else if (bits == 16) {
            joint_permute_shN_bounds_cu<16><<<grid, kThreads, 0, stream>>>(
                src_packed, src_bounds, dst_bounds, perm, n_prims, slots_per_primitive);
            LFS_CUDA_LAUNCH_CHECK(stream, "training.morton.joint_shN_bounds16");
            joint_permute_shN_encode_cu<16><<<grid, kThreads, 0, stream>>>(
                src_packed, src_bounds, dst_packed, dst_bounds, perm, n_prims,
                slots_per_primitive);
            LFS_CUDA_LAUNCH_CHECK(stream, "training.morton.joint_shN_encode16");
        } else {
            throw std::runtime_error("joint permute shN: bits must be 8 or 16");
        }
    }

} // namespace lfs::training::kernels

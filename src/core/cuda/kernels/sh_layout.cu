/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "core/cuda/sh_layout.cuh"
#include "core/cuda_error.hpp"
#include "core/logger.hpp"
#include <cuda_runtime.h>

namespace lfs::core {

    namespace {

        // Float4 slot index in the swizzled buffer for primitive p, float4 slot k.
        __device__ __forceinline__ std::uint32_t shAt_device(
            std::uint32_t p,
            std::uint32_t k,
            std::uint32_t slots_per_primitive) {
            const std::uint32_t block = p / kShReorderSize;
            const std::uint32_t lane = p % kShReorderSize;
            return block * (slots_per_primitive * kShReorderSize) + k * kShReorderSize + lane;
        }

        // Packing: read 4 consecutive floats from the 45-float canonical block (c0..c14, 15
        // float3) at `float_start_offset`. Out-of-range slots (offsets 45..47) read as zero.
        // For active_coeffs_rest < 15, also zero positions >= active_coeffs_rest * 3.
        __device__ __forceinline__ float4 read_pack4(
            const float* __restrict__ canonical_row,
            std::uint32_t float_start_offset,
            std::uint32_t active_floats) {
            const auto get = [&](std::uint32_t offset) -> float {
                return offset < active_floats ? canonical_row[offset] : 0.0f;
            };
            return make_float4(
                get(float_start_offset + 0),
                get(float_start_offset + 1),
                get(float_start_offset + 2),
                get(float_start_offset + 3));
        }

        // Unpacking: write up to 4 consecutive floats to the canonical row at `float_start_offset`.
        // Stops when target offset >= active_floats (skips tail padding).
        __device__ __forceinline__ void write_unpack4(
            float* __restrict__ canonical_row,
            std::uint32_t float_start_offset,
            std::uint32_t active_floats,
            float4 v) {
            const float src[4] = {v.x, v.y, v.z, v.w};
#pragma unroll
            for (std::uint32_t i = 0; i < 4u; ++i) {
                const std::uint32_t off = float_start_offset + i;
                if (off < active_floats)
                    canonical_row[off] = src[i];
            }
        }

        constexpr int BLOCK = 256;

        // One thread per primitive in the padded block range. Writes all active float4 slots,
        // zero-filling padding lanes / coefficients beyond active_coeffs_rest.
        __global__ void reorder_sh_kernel(
            const float* __restrict__ src,
            float4* __restrict__ dst,
            std::uint32_t n_primitives,
            std::uint32_t active_coeffs_rest,
            std::uint32_t padded_n,
            std::uint32_t slots_per_primitive) {
            const std::uint32_t p = blockIdx.x * blockDim.x + threadIdx.x;
            if (p >= padded_n)
                return;

            const bool valid_primitive = p < n_primitives;
            const std::uint32_t active_floats = valid_primitive ? active_coeffs_rest * 3u : 0u;
            const float* canonical_row = valid_primitive ? src + p * active_coeffs_rest * 3u : nullptr;

            for (std::uint32_t k = 0; k < slots_per_primitive; ++k) {
                float4 v = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
                if (valid_primitive) {
                    v = read_pack4(canonical_row, k * 4u, active_floats);
                }
                dst[shAt_device(p, k, slots_per_primitive)] = v;
            }
        }

        __global__ void undo_reorder_sh_kernel(
            const float4* __restrict__ src,
            float* __restrict__ dst,
            std::uint32_t n_primitives,
            std::uint32_t active_coeffs_rest,
            std::uint32_t slots_per_primitive) {
            const std::uint32_t p = blockIdx.x * blockDim.x + threadIdx.x;
            if (p >= n_primitives)
                return;

            const std::uint32_t active_floats = active_coeffs_rest * 3u;
            float* canonical_row = dst + p * active_floats;

            for (std::uint32_t k = 0; k < slots_per_primitive; ++k) {
                const std::uint32_t start_off = k * 4u;
                if (start_off >= active_floats)
                    break;
                const float4 v = src[shAt_device(p, k, slots_per_primitive)];
                write_unpack4(canonical_row, start_off, active_floats, v);
            }
        }

        template <typename IndexT>
        __global__ void zero_at_indices_kernel(
            float4* __restrict__ buffer,
            const IndexT* __restrict__ indices,
            std::uint32_t n_indices,
            std::uint32_t slots_per_primitive) {
            const std::uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
            if (i >= n_indices)
                return;
            const std::uint32_t p = static_cast<std::uint32_t>(indices[i]);
            const float4 zero = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
            for (std::uint32_t k = 0; k < slots_per_primitive; ++k) {
                buffer[shAt_device(p, k, slots_per_primitive)] = zero;
            }
        }

        template <typename IndexT, typename ElemT = float4>
        __global__ void gather_self_kernel(
            const ElemT* __restrict__ src,
            ElemT* __restrict__ dst,
            const IndexT* __restrict__ src_indices,
            std::uint32_t n_dst,
            std::uint32_t dst_offset,
            std::uint32_t slots_per_primitive) {
            const std::uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
            if (i >= n_dst)
                return;
            const std::uint32_t src_p = static_cast<std::uint32_t>(src_indices[i]);
            const std::uint32_t dst_p = dst_offset + i;
            for (std::uint32_t k = 0; k < slots_per_primitive; ++k) {
                dst[shAt_device(dst_p, k, slots_per_primitive)] = src[shAt_device(src_p, k, slots_per_primitive)];
            }
        }

        __global__ void copy_contiguous_kernel(
            const float4* __restrict__ src,
            float4* __restrict__ dst,
            std::uint32_t n_src,
            std::uint32_t dst_offset,
            std::uint32_t src_slots_per_primitive,
            std::uint32_t dst_slots_per_primitive) {
            const std::uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
            if (i >= n_src)
                return;
            const std::uint32_t dst_p = dst_offset + i;
            const float4 zero = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
            for (std::uint32_t k = 0; k < dst_slots_per_primitive; ++k) {
                dst[shAt_device(dst_p, k, dst_slots_per_primitive)] =
                    k < src_slots_per_primitive ? src[shAt_device(i, k, src_slots_per_primitive)] : zero;
            }
        }

        __global__ void copy_range_kernel(
            const float4* __restrict__ src,
            float4* __restrict__ dst,
            std::uint32_t src_offset,
            std::uint32_t n_src,
            std::uint32_t dst_offset,
            std::uint32_t src_slots_per_primitive,
            std::uint32_t dst_slots_per_primitive) {
            const std::uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
            if (i >= n_src)
                return;
            const std::uint32_t src_p = src_offset + i;
            const std::uint32_t dst_p = dst_offset + i;
            const float4 zero = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
            for (std::uint32_t k = 0; k < dst_slots_per_primitive; ++k) {
                dst[shAt_device(dst_p, k, dst_slots_per_primitive)] =
                    k < src_slots_per_primitive ? src[shAt_device(src_p, k, src_slots_per_primitive)] : zero;
            }
        }

        template <typename IndexT>
        __global__ void gather_to_linear_kernel(
            const float4* __restrict__ src,
            const IndexT* __restrict__ src_indices,
            float* __restrict__ dst_linear,
            std::uint32_t n_src,
            std::uint32_t active_coeffs_rest,
            std::uint32_t slots_per_primitive) {
            const std::uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
            if (i >= n_src)
                return;

            const std::uint32_t src_p = static_cast<std::uint32_t>(src_indices[i]);
            const std::uint32_t active_floats = active_coeffs_rest * 3u;
            float* dst_row = dst_linear + i * active_floats;
            for (std::uint32_t k = 0; k < slots_per_primitive; ++k) {
                const std::uint32_t start_off = k * 4u;
                if (start_off >= active_floats)
                    break;
                const float4 v = src[shAt_device(src_p, k, slots_per_primitive)];
                write_unpack4(dst_row, start_off, active_floats, v);
            }
        }

        __device__ __forceinline__ float float4_component(const float4 v, const std::uint32_t component) {
            switch (component) {
            case 0:
                return v.x;
            case 1:
                return v.y;
            case 2:
                return v.z;
            default:
                return v.w;
            }
        }

        __device__ __forceinline__ float read_swizzled_rest_float(
            const float4* __restrict__ src,
            const std::uint32_t primitive_idx,
            const std::uint32_t rest_float_offset,
            const std::uint32_t src_slots_per_primitive) {
            if (!src || rest_float_offset >= kShMaxCoeffsRest * kShChannels) {
                return 0.0f;
            }
            const std::uint32_t slot = rest_float_offset / 4u;
            const std::uint32_t component = rest_float_offset % 4u;
            if (slot >= src_slots_per_primitive) {
                return 0.0f;
            }
            return float4_component(src[shAt_device(primitive_idx, slot, src_slots_per_primitive)], component);
        }

        __global__ void pack_full_from_split_kernel(
            const float* __restrict__ src_sh0,
            const float4* __restrict__ src_shN,
            float4* __restrict__ dst,
            std::uint32_t n_primitives,
            std::uint32_t padded_n,
            std::uint32_t src_slots_per_primitive) {
            const std::uint32_t p = blockIdx.x * blockDim.x + threadIdx.x;
            if (p >= padded_n)
                return;

            const bool valid_primitive = p < n_primitives;
            const float* sh0_row = valid_primitive ? src_sh0 + p * kShChannels : nullptr;

#pragma unroll
            for (std::uint32_t k = 0; k < kShRestFloat4PerPrimitive; ++k) {
                float packed[4] = {0.0f, 0.0f, 0.0f, 0.0f};
#pragma unroll
                for (std::uint32_t c = 0; c < 4u; ++c) {
                    if (!valid_primitive)
                        continue;
                    const std::uint32_t full_float_offset = k * 4u + c;
                    packed[c] = full_float_offset < kShChannels
                                    ? sh0_row[full_float_offset]
                                    : read_swizzled_rest_float(src_shN, p, full_float_offset - kShChannels, src_slots_per_primitive);
                }
                dst[shAt_device(p, k, kShRestFloat4PerPrimitive)] = make_float4(packed[0], packed[1], packed[2], packed[3]);
            }
        }

        __global__ void gather_from_linear_kernel(
            float4* __restrict__ dst,
            std::uint32_t dst_offset,
            const float* __restrict__ src_linear,
            std::uint32_t n_src,
            std::uint32_t active_coeffs_rest,
            std::uint32_t slots_per_primitive) {
            const std::uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
            if (i >= n_src)
                return;
            const std::uint32_t dst_p = dst_offset + i;
            const std::uint32_t active_floats = active_coeffs_rest * 3u;
            const float* canonical_row = src_linear + i * active_floats;
            for (std::uint32_t k = 0; k < slots_per_primitive; ++k) {
                dst[shAt_device(dst_p, k, slots_per_primitive)] = read_pack4(canonical_row, k * 4u, active_floats);
            }
        }

        __global__ void scatter_linear_kernel(
            float4* __restrict__ dst,
            const int* __restrict__ dst_indices,
            const float* __restrict__ src_linear,
            std::uint32_t n_src,
            std::uint32_t active_coeffs_rest,
            std::uint32_t slots_per_primitive) {
            const std::uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
            if (i >= n_src)
                return;
            const std::uint32_t dst_p = static_cast<std::uint32_t>(dst_indices[i]);
            const std::uint32_t active_floats = active_coeffs_rest * 3u;
            const float* canonical_row = src_linear + i * active_floats;
            for (std::uint32_t k = 0; k < slots_per_primitive; ++k) {
                dst[shAt_device(dst_p, k, slots_per_primitive)] = read_pack4(canonical_row, k * 4u, active_floats);
            }
        }

    } // namespace

    void reorder_sh_to_swizzled(
        const float* src_canonical,
        float* dst_swizzled,
        std::size_t n_primitives,
        std::uint32_t active_coeffs_rest,
        cudaStream_t stream) {
        reorder_sh_to_swizzled(src_canonical,
                               dst_swizzled,
                               n_primitives,
                               active_coeffs_rest,
                               active_coeffs_rest,
                               stream);
    }

    void reorder_sh_to_swizzled(
        const float* src_canonical,
        float* dst_swizzled,
        std::size_t n_primitives,
        std::uint32_t src_coeffs_rest,
        std::uint32_t layout_coeffs_rest,
        cudaStream_t stream) {
        if (src_coeffs_rest == 0) {
            return;
        }
        const std::uint32_t padded_n = static_cast<std::uint32_t>(sh_swizzled_padded_n(n_primitives));
        if (padded_n == 0)
            return;
        const auto slots = sh_float4_slots_for_rest(layout_coeffs_rest);
        if (slots == 0)
            return;
        const int grid = static_cast<int>((padded_n + BLOCK - 1) / BLOCK);
        reorder_sh_kernel<<<grid, BLOCK, 0, stream>>>(
            src_canonical, reinterpret_cast<float4*>(dst_swizzled),
            static_cast<std::uint32_t>(n_primitives), src_coeffs_rest, padded_n, slots);
        LFS_CUDA_LAUNCH_CHECK(stream, "core.sh_layout.reorder");
    }

    void undo_reorder_sh_from_swizzled(
        const float* src_swizzled,
        float* dst_canonical,
        std::size_t n_primitives,
        std::uint32_t active_coeffs_rest,
        cudaStream_t stream) {
        undo_reorder_sh_from_swizzled(src_swizzled,
                                      dst_canonical,
                                      n_primitives,
                                      active_coeffs_rest,
                                      active_coeffs_rest,
                                      stream);
    }

    void undo_reorder_sh_from_swizzled(
        const float* src_swizzled,
        float* dst_canonical,
        std::size_t n_primitives,
        std::uint32_t dst_coeffs_rest,
        std::uint32_t layout_coeffs_rest,
        cudaStream_t stream) {
        if (n_primitives == 0 || dst_coeffs_rest == 0)
            return;
        const auto slots = sh_float4_slots_for_rest(layout_coeffs_rest);
        if (slots == 0)
            return;
        const int grid = static_cast<int>((n_primitives + BLOCK - 1) / BLOCK);
        undo_reorder_sh_kernel<<<grid, BLOCK, 0, stream>>>(
            reinterpret_cast<const float4*>(src_swizzled), dst_canonical,
            static_cast<std::uint32_t>(n_primitives), dst_coeffs_rest, slots);
        LFS_CUDA_LAUNCH_CHECK(stream, "core.sh_layout.undo_reorder");
    }

    void shN_swizzled_zero_at_indices(
        float* buffer_swizzled,
        const int* indices,
        std::size_t n_indices,
        std::uint32_t active_coeffs_rest,
        cudaStream_t stream) {
        const auto slots = sh_float4_slots_for_rest(active_coeffs_rest);
        if (n_indices == 0 || slots == 0)
            return;
        const int grid = static_cast<int>((n_indices + BLOCK - 1) / BLOCK);
        zero_at_indices_kernel<int><<<grid, BLOCK, 0, stream>>>(
            reinterpret_cast<float4*>(buffer_swizzled), indices, static_cast<std::uint32_t>(n_indices), slots);
        LFS_CUDA_LAUNCH_CHECK(stream, "core.sh_layout.zero_at_indices_i32");
    }

    void shN_swizzled_zero_at_indices_i64(
        float* buffer_swizzled,
        const std::int64_t* indices,
        std::size_t n_indices,
        std::uint32_t active_coeffs_rest,
        cudaStream_t stream) {
        const auto slots = sh_float4_slots_for_rest(active_coeffs_rest);
        if (n_indices == 0 || slots == 0)
            return;
        const int grid = static_cast<int>((n_indices + BLOCK - 1) / BLOCK);
        zero_at_indices_kernel<std::int64_t><<<grid, BLOCK, 0, stream>>>(
            reinterpret_cast<float4*>(buffer_swizzled), indices, static_cast<std::uint32_t>(n_indices), slots);
        LFS_CUDA_LAUNCH_CHECK(stream, "core.sh_layout.zero_at_indices_i64");
    }

    void shN_swizzled_gather_self(
        const float* src_swizzled,
        float* dst_swizzled,
        const int* src_indices,
        std::size_t n_dst,
        std::size_t dst_offset,
        std::uint32_t active_coeffs_rest,
        cudaStream_t stream) {
        const auto slots = sh_float4_slots_for_rest(active_coeffs_rest);
        if (n_dst == 0 || slots == 0)
            return;
        const int grid = static_cast<int>((n_dst + BLOCK - 1) / BLOCK);
        gather_self_kernel<int><<<grid, BLOCK, 0, stream>>>(
            reinterpret_cast<const float4*>(src_swizzled),
            reinterpret_cast<float4*>(dst_swizzled), src_indices,
            static_cast<std::uint32_t>(n_dst),
            static_cast<std::uint32_t>(dst_offset), slots);
        LFS_CUDA_LAUNCH_CHECK(stream, "core.sh_layout.gather_self_i32");
    }

    void shN_swizzled_gather_self_u8(
        const std::uint8_t* src_swizzled,
        std::uint8_t* dst_swizzled,
        const int* src_indices,
        std::size_t n_dst,
        std::size_t dst_offset,
        std::uint32_t active_coeffs_rest,
        cudaStream_t stream) {
        const auto slots = sh_float4_slots_for_rest(active_coeffs_rest);
        if (n_dst == 0 || slots == 0)
            return;
        const int grid = static_cast<int>((n_dst + BLOCK - 1) / BLOCK);
        gather_self_kernel<int, uchar4><<<grid, BLOCK, 0, stream>>>(
            reinterpret_cast<const uchar4*>(src_swizzled),
            reinterpret_cast<uchar4*>(dst_swizzled), src_indices,
            static_cast<std::uint32_t>(n_dst),
            static_cast<std::uint32_t>(dst_offset), slots);
        LFS_CUDA_LAUNCH_CHECK(stream, "core.sh_layout.gather_self_i32_u8");
    }

    void shN_swizzled_gather_self_u8_i64(
        const std::uint8_t* src_swizzled,
        std::uint8_t* dst_swizzled,
        const std::int64_t* src_indices,
        std::size_t n_dst,
        std::size_t dst_offset,
        std::uint32_t active_coeffs_rest,
        cudaStream_t stream) {
        const auto slots = sh_float4_slots_for_rest(active_coeffs_rest);
        if (n_dst == 0 || slots == 0)
            return;
        const int grid = static_cast<int>((n_dst + BLOCK - 1) / BLOCK);
        gather_self_kernel<std::int64_t, uchar4><<<grid, BLOCK, 0, stream>>>(
            reinterpret_cast<const uchar4*>(src_swizzled),
            reinterpret_cast<uchar4*>(dst_swizzled), src_indices,
            static_cast<std::uint32_t>(n_dst),
            static_cast<std::uint32_t>(dst_offset), slots);
        LFS_CUDA_LAUNCH_CHECK(stream, "core.sh_layout.gather_self_i64_u8");
    }

    void shN_swizzled_gather_self_i64(
        const float* src_swizzled,
        float* dst_swizzled,
        const std::int64_t* src_indices,
        std::size_t n_dst,
        std::size_t dst_offset,
        std::uint32_t active_coeffs_rest,
        cudaStream_t stream) {
        const auto slots = sh_float4_slots_for_rest(active_coeffs_rest);
        if (n_dst == 0 || slots == 0)
            return;
        const int grid = static_cast<int>((n_dst + BLOCK - 1) / BLOCK);
        gather_self_kernel<std::int64_t><<<grid, BLOCK, 0, stream>>>(
            reinterpret_cast<const float4*>(src_swizzled),
            reinterpret_cast<float4*>(dst_swizzled), src_indices,
            static_cast<std::uint32_t>(n_dst),
            static_cast<std::uint32_t>(dst_offset), slots);
        LFS_CUDA_LAUNCH_CHECK(stream, "core.sh_layout.gather_self_i64");
    }

    void shN_swizzled_copy_contiguous(
        const float* src_swizzled,
        float* dst_swizzled,
        std::size_t n_src,
        std::size_t dst_offset,
        std::uint32_t src_active_coeffs_rest,
        std::uint32_t dst_active_coeffs_rest,
        cudaStream_t stream) {
        const auto src_slots = sh_float4_slots_for_rest(src_active_coeffs_rest);
        const auto dst_slots = sh_float4_slots_for_rest(dst_active_coeffs_rest);
        if (n_src == 0 || dst_slots == 0)
            return;
        const int grid = static_cast<int>((n_src + BLOCK - 1) / BLOCK);
        copy_contiguous_kernel<<<grid, BLOCK, 0, stream>>>(
            reinterpret_cast<const float4*>(src_swizzled),
            reinterpret_cast<float4*>(dst_swizzled),
            static_cast<std::uint32_t>(n_src),
            static_cast<std::uint32_t>(dst_offset),
            src_slots,
            dst_slots);
        LFS_CUDA_LAUNCH_CHECK(stream, "core.sh_layout.copy_contiguous");
    }

    void shN_swizzled_copy_range(
        const float* src_swizzled,
        float* dst_swizzled,
        std::size_t src_offset,
        std::size_t n_src,
        std::size_t dst_offset,
        std::uint32_t src_active_coeffs_rest,
        std::uint32_t dst_active_coeffs_rest,
        cudaStream_t stream) {
        const auto src_slots = sh_float4_slots_for_rest(src_active_coeffs_rest);
        const auto dst_slots = sh_float4_slots_for_rest(dst_active_coeffs_rest);
        if (n_src == 0 || dst_slots == 0)
            return;
        const int grid = static_cast<int>((n_src + BLOCK - 1) / BLOCK);
        copy_range_kernel<<<grid, BLOCK, 0, stream>>>(
            reinterpret_cast<const float4*>(src_swizzled),
            reinterpret_cast<float4*>(dst_swizzled),
            static_cast<std::uint32_t>(src_offset),
            static_cast<std::uint32_t>(n_src),
            static_cast<std::uint32_t>(dst_offset),
            src_slots,
            dst_slots);
        LFS_CUDA_LAUNCH_CHECK(stream, "core.sh_layout.copy_range");
    }

    void shN_swizzled_gather_to_linear(
        const float* src_swizzled,
        const int* src_indices,
        float* dst_linear,
        std::size_t n_src,
        std::uint32_t active_coeffs_rest,
        cudaStream_t stream) {
        shN_swizzled_gather_to_linear(src_swizzled,
                                      src_indices,
                                      dst_linear,
                                      n_src,
                                      active_coeffs_rest,
                                      active_coeffs_rest,
                                      stream);
    }

    void shN_swizzled_gather_to_linear(
        const float* src_swizzled,
        const int* src_indices,
        float* dst_linear,
        std::size_t n_src,
        std::uint32_t dst_coeffs_rest,
        std::uint32_t layout_coeffs_rest,
        cudaStream_t stream) {
        if (n_src == 0 || dst_coeffs_rest == 0)
            return;
        const auto slots = sh_float4_slots_for_rest(layout_coeffs_rest);
        if (slots == 0)
            return;
        const int grid = static_cast<int>((n_src + BLOCK - 1) / BLOCK);
        gather_to_linear_kernel<int><<<grid, BLOCK, 0, stream>>>(
            reinterpret_cast<const float4*>(src_swizzled), src_indices, dst_linear,
            static_cast<std::uint32_t>(n_src), dst_coeffs_rest, slots);
        LFS_CUDA_LAUNCH_CHECK(stream, "core.sh_layout.gather_to_linear_i32");
    }

    void shN_swizzled_gather_to_linear_i64(
        const float* src_swizzled,
        const std::int64_t* src_indices,
        float* dst_linear,
        std::size_t n_src,
        std::uint32_t active_coeffs_rest,
        cudaStream_t stream) {
        shN_swizzled_gather_to_linear_i64(src_swizzled,
                                          src_indices,
                                          dst_linear,
                                          n_src,
                                          active_coeffs_rest,
                                          active_coeffs_rest,
                                          stream);
    }

    void shN_swizzled_gather_to_linear_i64(
        const float* src_swizzled,
        const std::int64_t* src_indices,
        float* dst_linear,
        std::size_t n_src,
        std::uint32_t dst_coeffs_rest,
        std::uint32_t layout_coeffs_rest,
        cudaStream_t stream) {
        if (n_src == 0 || dst_coeffs_rest == 0)
            return;
        const auto slots = sh_float4_slots_for_rest(layout_coeffs_rest);
        if (slots == 0)
            return;
        const int grid = static_cast<int>((n_src + BLOCK - 1) / BLOCK);
        gather_to_linear_kernel<std::int64_t><<<grid, BLOCK, 0, stream>>>(
            reinterpret_cast<const float4*>(src_swizzled), src_indices, dst_linear,
            static_cast<std::uint32_t>(n_src), dst_coeffs_rest, slots);
        LFS_CUDA_LAUNCH_CHECK(stream, "core.sh_layout.gather_to_linear_i64");
    }

    void shN_swizzled_gather_from_linear(
        float* dst_swizzled,
        std::size_t dst_offset,
        const float* src_linear,
        std::size_t n_src,
        std::uint32_t active_coeffs_rest,
        cudaStream_t stream) {
        shN_swizzled_gather_from_linear(dst_swizzled,
                                        dst_offset,
                                        src_linear,
                                        n_src,
                                        active_coeffs_rest,
                                        active_coeffs_rest,
                                        stream);
    }

    void shN_swizzled_gather_from_linear(
        float* dst_swizzled,
        std::size_t dst_offset,
        const float* src_linear,
        std::size_t n_src,
        std::uint32_t src_coeffs_rest,
        std::uint32_t layout_coeffs_rest,
        cudaStream_t stream) {
        const auto slots = sh_float4_slots_for_rest(layout_coeffs_rest);
        if (n_src == 0 || slots == 0)
            return;
        const int grid = static_cast<int>((n_src + BLOCK - 1) / BLOCK);
        gather_from_linear_kernel<<<grid, BLOCK, 0, stream>>>(
            reinterpret_cast<float4*>(dst_swizzled), static_cast<std::uint32_t>(dst_offset),
            src_linear, static_cast<std::uint32_t>(n_src), src_coeffs_rest, slots);
        LFS_CUDA_LAUNCH_CHECK(stream, "core.sh_layout.gather_from_linear");
    }

    void shN_swizzled_scatter_linear(
        float* dst_swizzled,
        const int* dst_indices,
        const float* src_linear,
        std::size_t n_src,
        std::uint32_t active_coeffs_rest,
        cudaStream_t stream) {
        shN_swizzled_scatter_linear(dst_swizzled,
                                    dst_indices,
                                    src_linear,
                                    n_src,
                                    active_coeffs_rest,
                                    active_coeffs_rest,
                                    stream);
    }

    void shN_swizzled_scatter_linear(
        float* dst_swizzled,
        const int* dst_indices,
        const float* src_linear,
        std::size_t n_src,
        std::uint32_t src_coeffs_rest,
        std::uint32_t layout_coeffs_rest,
        cudaStream_t stream) {
        const auto slots = sh_float4_slots_for_rest(layout_coeffs_rest);
        if (n_src == 0 || slots == 0)
            return;
        const int grid = static_cast<int>((n_src + BLOCK - 1) / BLOCK);
        scatter_linear_kernel<<<grid, BLOCK, 0, stream>>>(
            reinterpret_cast<float4*>(dst_swizzled), dst_indices, src_linear,
            static_cast<std::uint32_t>(n_src), src_coeffs_rest, slots);
        LFS_CUDA_LAUNCH_CHECK(stream, "core.sh_layout.scatter_linear");
    }

    void sh_swizzled_pack_full_from_split(
        const float* src_sh0,
        const float* src_shN_swizzled,
        float* dst_full_swizzled,
        std::size_t n_primitives,
        std::uint32_t active_coeffs_rest,
        cudaStream_t stream) {
        if (n_primitives == 0 || !src_sh0 || !dst_full_swizzled)
            return;
        const std::uint32_t padded_n = static_cast<std::uint32_t>(sh_swizzled_padded_n(n_primitives));
        const auto src_slots = sh_float4_slots_for_rest(active_coeffs_rest);
        const int grid = static_cast<int>((padded_n + BLOCK - 1) / BLOCK);
        pack_full_from_split_kernel<<<grid, BLOCK, 0, stream>>>(
            src_sh0,
            reinterpret_cast<const float4*>(src_shN_swizzled),
            reinterpret_cast<float4*>(dst_full_swizzled),
            static_cast<std::uint32_t>(n_primitives),
            padded_n,
            src_slots);
        LFS_CUDA_LAUNCH_CHECK(stream, "core.sh_layout.pack_full_from_split");
    }

} // namespace lfs::core

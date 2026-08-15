/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

/**
 * SH value quant conversion kernels.
 */

#include "core/cuda/sh_layout.cuh"
#include "core/cuda_error.hpp"
#include "lfs/training/sh_value_codec.cuh"
#include "lfs/training/sh_value_codec.hpp"

#include <cstdint>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <limits>
#include <stdexcept>

namespace lfs::training::sh_value {
    namespace {

        constexpr int kThreads = 256;

        __device__ __forceinline__ std::uint32_t shAtF4(
            std::uint32_t p, std::uint32_t k, std::uint32_t slots) {
            constexpr std::uint32_t R = lfs::core::kShReorderSize;
            return (p / R) * (slots * R) + k * R + (p % R);
        }

        // One CUDA block of 256 threads per quant-block of prims.
        __global__ void encode_float4_to_u16_block_kernel(
            const float* __restrict__ src_f4_as_float,
            std::uint16_t* __restrict__ dst_u16,
            float2* __restrict__ bounds,
            std::uint32_t n_primitives,
            std::uint32_t slots_per_prim,
            std::uint32_t n_cells_per_prim) {
            using DC = DeviceCodec16;
            const std::uint32_t quant_block = blockIdx.x;
            const std::uint32_t lane = threadIdx.x;
            const std::uint32_t p = quant_block * 256u + lane;
            const bool in_range = p < n_primitives;

            const float4* src = reinterpret_cast<const float4*>(src_f4_as_float);

            float local_lo = 1e30f, local_hi = -1e30f;
            float cells[48];
            const std::uint32_t n_cells =
                n_cells_per_prim > 48u ? 48u : n_cells_per_prim;

            if (in_range) {
                for (std::uint32_t c = 0; c < n_cells; ++c) {
                    const std::uint32_t slot = c / 4u;
                    const std::uint32_t comp = c % 4u;
                    float v = 0.0f;
                    if (slot < slots_per_prim) {
                        const float4 f4 = src[shAtF4(p, slot, slots_per_prim)];
                        v = (comp == 0) ? f4.x : (comp == 1) ? f4.y
                                             : (comp == 2)   ? f4.z
                                                             : f4.w;
                    }
                    cells[c] = v;
                    local_lo = fminf(local_lo, v);
                    local_hi = fmaxf(local_hi, v);
                }
            }

            __shared__ float s_lo[256];
            __shared__ float s_hi[256];
            s_lo[lane] = in_range ? local_lo : 1e30f;
            s_hi[lane] = in_range ? local_hi : -1e30f;
            __syncthreads();
            for (int stride = 128; stride > 0; stride >>= 1) {
                if (static_cast<int>(lane) < stride) {
                    s_lo[lane] = fminf(s_lo[lane], s_lo[lane + stride]);
                    s_hi[lane] = fmaxf(s_hi[lane], s_hi[lane + stride]);
                }
                __syncthreads();
            }
            const float2 mm = (s_lo[0] > s_hi[0]) ? make_float2(0.0f, 0.0f)
                                                  : make_float2(s_lo[0], s_hi[0]);
            if (lane == 0) {
                bounds[quant_block] = mm;
            }
            __syncthreads();

            if (!in_range)
                return;
            for (std::uint32_t c = 0; c < n_cells; ++c) {
                dst_u16[shAtU16(p, c, n_cells_per_prim)] =
                    DC::encode(cells[c], mm.x, mm.y);
            }
        }

        __global__ void decode_u16_to_float4_kernel(
            const std::uint16_t* __restrict__ src_u16,
            const float2* __restrict__ bounds,
            float* __restrict__ dst_f4_as_float,
            std::uint32_t n_primitives,
            std::uint32_t slots_per_prim,
            std::uint32_t n_cells_per_prim) {
            using DC = DeviceCodec16;
            const std::uint32_t p = blockIdx.x * blockDim.x + threadIdx.x;
            if (p >= n_primitives)
                return;

            float4* dst = reinterpret_cast<float4*>(dst_f4_as_float);
            const float2 mm = bounds[p / 256u];

            for (std::uint32_t k = 0; k < slots_per_prim; ++k) {
                dst[shAtF4(p, k, slots_per_prim)] = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
            }
            for (std::uint32_t c = 0; c < n_cells_per_prim; ++c) {
                const float v = DC::decode(src_u16[shAtU16(p, c, n_cells_per_prim)], mm.x, mm.y);
                const std::uint32_t slot = c / 4u;
                const std::uint32_t comp = c % 4u;
                if (slot >= slots_per_prim)
                    break;
                float4 f4 = dst[shAtF4(p, slot, slots_per_prim)];
                if (comp == 0)
                    f4.x = v;
                else if (comp == 1)
                    f4.y = v;
                else if (comp == 2)
                    f4.z = v;
                else
                    f4.w = v;
                dst[shAtF4(p, slot, slots_per_prim)] = f4;
            }
        }

        __global__ void decode_u16_range_to_canonical_kernel(
            const std::uint16_t* __restrict__ src_u16,
            const float2* __restrict__ bounds,
            float* __restrict__ dst,
            std::uint64_t canonical_float_offset,
            std::uint64_t float_count,
            std::uint32_t floats_per_primitive,
            std::uint32_t n_cells_per_primitive) {
            const std::uint64_t output_index =
                static_cast<std::uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
            if (output_index >= float_count)
                return;

            const std::uint64_t canonical_index = canonical_float_offset + output_index;
            const auto primitive = static_cast<std::uint32_t>(
                canonical_index / floats_per_primitive);
            const auto cell = static_cast<std::uint32_t>(
                canonical_index % floats_per_primitive);
            const float2 mm = bounds[primitive / 256u];
            dst[output_index] = DeviceCodec16::decode(
                src_u16[shAtU16(primitive, cell, n_cells_per_primitive)],
                mm.x,
                mm.y);
        }

        __global__ void decode_f16_range_to_canonical_kernel(
            const __half* __restrict__ src_f16,
            float* __restrict__ dst,
            std::uint64_t canonical_float_offset,
            std::uint64_t float_count,
            std::uint32_t floats_per_primitive,
            std::uint32_t slots_per_primitive) {
            const std::uint64_t output_index =
                static_cast<std::uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
            if (output_index >= float_count)
                return;

            const std::uint64_t canonical_index = canonical_float_offset + output_index;
            const auto primitive = static_cast<std::uint32_t>(
                canonical_index / floats_per_primitive);
            const auto row_offset = static_cast<std::uint32_t>(
                canonical_index % floats_per_primitive);
            const auto slot = row_offset / 4u;
            const auto component = row_offset % 4u;
            const auto packed_index =
                static_cast<std::uint64_t>(shAtF4(
                    primitive, slot, slots_per_primitive)) *
                    4u +
                component;
            dst[output_index] = __half2float(src_f16[packed_index]);
        }

        void validate_canonical_range(
            const std::uint64_t canonical_float_offset,
            const std::uint64_t float_count,
            const std::size_t n_primitives,
            const std::uint32_t dst_coeffs_rest) {
            if (float_count == 0)
                return;
            if (n_primitives == 0 || dst_coeffs_rest == 0) {
                throw std::invalid_argument("Invalid bounded SH decode arguments");
            }
            const std::uint64_t floats_per_primitive =
                static_cast<std::uint64_t>(dst_coeffs_rest) *
                lfs::core::kShChannels;
            if (n_primitives >
                    std::numeric_limits<std::uint64_t>::max() /
                        floats_per_primitive ||
                canonical_float_offset >
                    n_primitives * floats_per_primitive ||
                float_count >
                    n_primitives * floats_per_primitive -
                        canonical_float_offset) {
                throw std::out_of_range(
                    "Bounded SH decode range exceeds canonical tensor");
            }
        }

    } // namespace

    void encode_shN_float4_to_u16(
        const float* src_float4_swizzled,
        std::uint16_t* dst_u16,
        float* bounds_float2,
        std::size_t n_primitives,
        std::uint32_t coeffs_rest,
        cudaStream_t stream) {
        if (n_primitives == 0 || coeffs_rest == 0)
            return;
        const auto slots = lfs::core::sh_float4_slots_for_rest(coeffs_rest);
        const auto n_cells = n_value_cells_per_prim(coeffs_rest);
        const auto n_bounds = n_bounds_for_prims(n_primitives);
        if (n_bounds == 0)
            return;
        encode_float4_to_u16_block_kernel<<<static_cast<unsigned>(n_bounds), 256, 0, stream>>>(
            src_float4_swizzled,
            dst_u16,
            reinterpret_cast<float2*>(bounds_float2),
            static_cast<std::uint32_t>(n_primitives),
            slots,
            n_cells);
        LFS_CUDA_CHECK_MSG(cudaGetLastError(), "encode_shN_float4_to_u16");
    }

    void decode_shN_u16_to_float4(
        const std::uint16_t* src_u16,
        const float* bounds_float2,
        float* dst_float4_swizzled,
        std::size_t n_primitives,
        std::uint32_t coeffs_rest,
        cudaStream_t stream) {
        if (n_primitives == 0 || coeffs_rest == 0)
            return;
        const auto slots = lfs::core::sh_float4_slots_for_rest(coeffs_rest);
        const auto n_cells = n_value_cells_per_prim(coeffs_rest);
        const unsigned blocks =
            static_cast<unsigned>((n_primitives + kThreads - 1) / kThreads);
        decode_u16_to_float4_kernel<<<blocks, kThreads, 0, stream>>>(
            src_u16,
            reinterpret_cast<const float2*>(bounds_float2),
            dst_float4_swizzled,
            static_cast<std::uint32_t>(n_primitives),
            slots,
            n_cells);
        LFS_CUDA_CHECK_MSG(cudaGetLastError(), "decode_shN_u16_to_float4");
    }

    void decode_shN_u16_range_to_canonical(
        const std::uint16_t* src_u16,
        const float* bounds_float2,
        float* dst_canonical,
        const std::uint64_t canonical_float_offset,
        const std::uint64_t float_count,
        const std::size_t n_primitives,
        const std::uint32_t dst_coeffs_rest,
        const std::uint32_t layout_coeffs_rest,
        cudaStream_t stream) {
        if (float_count == 0)
            return;
        if (!src_u16 || !bounds_float2 || !dst_canonical ||
            layout_coeffs_rest < dst_coeffs_rest) {
            throw std::invalid_argument("Invalid bounded q16 SH decode arguments");
        }
        validate_canonical_range(
            canonical_float_offset,
            float_count,
            n_primitives,
            dst_coeffs_rest);
        const auto n_cells = n_value_cells_per_prim(layout_coeffs_rest);
        const auto blocks = static_cast<unsigned>(
            (float_count + kThreads - 1) / kThreads);
        decode_u16_range_to_canonical_kernel<<<blocks, kThreads, 0, stream>>>(
            src_u16,
            reinterpret_cast<const float2*>(bounds_float2),
            dst_canonical,
            canonical_float_offset,
            float_count,
            dst_coeffs_rest * lfs::core::kShChannels,
            n_cells);
        LFS_CUDA_CHECK_MSG(
            cudaGetLastError(), "decode_shN_u16_range_to_canonical");
    }

    void decode_shN_f16_range_to_canonical(
        const std::uint16_t* src_f16,
        float* dst_canonical,
        const std::uint64_t canonical_float_offset,
        const std::uint64_t float_count,
        const std::size_t n_primitives,
        const std::uint32_t dst_coeffs_rest,
        const std::uint32_t layout_coeffs_rest,
        cudaStream_t stream) {
        if (float_count == 0)
            return;
        if (!src_f16 || !dst_canonical ||
            layout_coeffs_rest < dst_coeffs_rest) {
            throw std::invalid_argument("Invalid bounded f16 SH decode arguments");
        }
        validate_canonical_range(
            canonical_float_offset,
            float_count,
            n_primitives,
            dst_coeffs_rest);
        const auto slots = lfs::core::sh_float4_slots_for_rest(
            layout_coeffs_rest);
        if (slots == 0) {
            throw std::invalid_argument("Bounded f16 SH decode has no source slots");
        }
        const auto blocks = static_cast<unsigned>(
            (float_count + kThreads - 1) / kThreads);
        decode_f16_range_to_canonical_kernel<<<blocks, kThreads, 0, stream>>>(
            reinterpret_cast<const __half*>(src_f16),
            dst_canonical,
            canonical_float_offset,
            float_count,
            dst_coeffs_rest * lfs::core::kShChannels,
            slots);
        LFS_CUDA_CHECK_MSG(
            cudaGetLastError(), "decode_shN_f16_range_to_canonical");
    }

} // namespace lfs::training::sh_value

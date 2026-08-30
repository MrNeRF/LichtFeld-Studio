/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

/**
 * SH value quant conversion kernels.
 */

#include "core/cuda/sh_layout.cuh"
#include "core/cuda_allocation.hpp"
#include "core/cuda_error.hpp"
#include "core/sh_value_codec.cuh"
#include "core/sh_value_quant.hpp"
#include "core/sh_value_quant_kernels.hpp"

#include <cstdint>
#include <cub/device/device_scan.cuh>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <limits>
#include <stdexcept>

namespace lfs::core::sh_value_quant {
    namespace {

        constexpr int kThreads = 256;

        __device__ __forceinline__ std::uint32_t shAtF4(
            std::uint32_t p, std::uint32_t k, std::uint32_t slots) {
            constexpr std::uint32_t R = lfs::core::kShReorderSize;
            return (p / R) * (slots * R) + k * R + (p % R);
        }

        // Identical min/max tree as the original encode kernel: 256-wide
        // shared reduction with 1e30/-1e30 sentinels for inactive lanes.
        __device__ __forceinline__ float2 reduce_quant_block_minmax(
            const std::uint32_t lane,
            const bool in_range,
            const float local_lo,
            const float local_hi) {
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
            return (s_lo[0] > s_hi[0]) ? make_float2(0.0f, 0.0f)
                                       : make_float2(s_lo[0], s_hi[0]);
        }

        __device__ __forceinline__ void decode_prim_u16_cells(
            const std::uint16_t* __restrict__ src_u16,
            const float2 mm,
            const std::uint32_t p,
            const std::uint32_t n_cells,
            const std::uint32_t n_cells_per_prim,
            float* cells) {
            using DC = lfs::core::sh_value::DeviceCodec16;
            for (std::uint32_t c = 0; c < n_cells; ++c) {
                cells[c] = DC::decode(
                    src_u16[lfs::core::sh_value::shAtU16(p, c, n_cells_per_prim)],
                    mm.x,
                    mm.y);
            }
        }

        __device__ __forceinline__ void encode_prim_u16_cells(
            std::uint16_t* __restrict__ dst_u16,
            const float2 mm,
            const std::uint32_t p,
            const std::uint32_t n_cells,
            const std::uint32_t n_cells_per_prim,
            const float* cells) {
            using DC = lfs::core::sh_value::DeviceCodec16;
            for (std::uint32_t c = 0; c < n_cells; ++c) {
                dst_u16[lfs::core::sh_value::shAtU16(p, c, n_cells_per_prim)] =
                    DC::encode(cells[c], mm.x, mm.y);
            }
        }

        // One CUDA block of 256 threads per quant-block of prims.
        __global__ void encode_float4_to_u16_block_kernel(
            const float* __restrict__ src_f4_as_float,
            std::uint16_t* __restrict__ dst_u16,
            float2* __restrict__ bounds,
            std::uint32_t n_primitives,
            std::uint32_t slots_per_prim,
            std::uint32_t n_cells_per_prim) {
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

            const float2 mm = reduce_quant_block_minmax(lane, in_range, local_lo, local_hi);
            if (lane == 0) {
                bounds[quant_block] = mm;
            }
            __syncthreads();

            if (!in_range)
                return;
            encode_prim_u16_cells(dst_u16, mm, p, n_cells, n_cells_per_prim, cells);
        }

        __global__ void decode_u16_to_float4_kernel(
            const std::uint16_t* __restrict__ src_u16,
            const float2* __restrict__ bounds,
            float* __restrict__ dst_f4_as_float,
            std::uint32_t n_primitives,
            std::uint32_t slots_per_prim,
            std::uint32_t n_cells_per_prim) {
            using DC = lfs::core::sh_value::DeviceCodec16;
            const std::uint32_t p = blockIdx.x * blockDim.x + threadIdx.x;
            if (p >= n_primitives)
                return;

            float4* dst = reinterpret_cast<float4*>(dst_f4_as_float);
            const float2 mm = bounds[p / 256u];

            for (std::uint32_t k = 0; k < slots_per_prim; ++k) {
                dst[shAtF4(p, k, slots_per_prim)] = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
            }
            for (std::uint32_t c = 0; c < n_cells_per_prim; ++c) {
                const float v = DC::decode(
                    src_u16[lfs::core::sh_value::shAtU16(p, c, n_cells_per_prim)], mm.x, mm.y);
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
            dst[output_index] = lfs::core::sh_value::DeviceCodec16::decode(
                src_u16[lfs::core::sh_value::shAtU16(primitive, cell, n_cells_per_primitive)],
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

        __global__ void decode_u16_gathered_to_float4_kernel(
            const std::uint16_t* __restrict__ src_u16,
            const float2* __restrict__ src_bounds,
            const std::int64_t* __restrict__ perm,
            float* __restrict__ dst_f4_as_float,
            std::uint32_t dest_offset,
            std::uint32_t n_dst,
            std::uint32_t n_src,
            std::uint32_t slots_per_prim,
            std::uint32_t n_cells_per_prim) {
            using DC = lfs::core::sh_value::DeviceCodec16;
            const std::uint32_t p_local = blockIdx.x * blockDim.x + threadIdx.x;
            if (p_local >= n_dst)
                return;

            float4* dst = reinterpret_cast<float4*>(dst_f4_as_float);
            for (std::uint32_t k = 0; k < slots_per_prim; ++k) {
                dst[shAtF4(p_local, k, slots_per_prim)] = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
            }

            const std::int64_t src = perm[dest_offset + p_local];
            if (src < 0 || src >= static_cast<std::int64_t>(n_src))
                return;

            const auto src_p = static_cast<std::uint32_t>(src);
            const float2 mm = src_bounds[src_p / 256u];
            for (std::uint32_t c = 0; c < n_cells_per_prim; ++c) {
                const float v = DC::decode(
                    src_u16[lfs::core::sh_value::shAtU16(src_p, c, n_cells_per_prim)],
                    mm.x, mm.y);
                const std::uint32_t slot = c / 4u;
                const std::uint32_t comp = c % 4u;
                if (slot >= slots_per_prim)
                    break;
                float4 f4 = dst[shAtF4(p_local, slot, slots_per_prim)];
                if (comp == 0)
                    f4.x = v;
                else if (comp == 1)
                    f4.y = v;
                else if (comp == 2)
                    f4.z = v;
                else
                    f4.w = v;
                dst[shAtF4(p_local, slot, slots_per_prim)] = f4;
            }
        }

        __global__ void decode_u16_range_to_float4_kernel(
            const std::uint16_t* __restrict__ src_u16,
            const float2* __restrict__ src_bounds,
            float* __restrict__ dst_f4_as_float,
            std::uint32_t src_offset,
            std::uint32_t n_dst,
            std::uint32_t n_src,
            std::uint32_t slots_per_prim,
            std::uint32_t n_cells_per_prim) {
            using DC = lfs::core::sh_value::DeviceCodec16;
            const std::uint32_t p_local = blockIdx.x * blockDim.x + threadIdx.x;
            if (p_local >= n_dst)
                return;

            float4* dst = reinterpret_cast<float4*>(dst_f4_as_float);
            for (std::uint32_t k = 0; k < slots_per_prim; ++k) {
                dst[shAtF4(p_local, k, slots_per_prim)] = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
            }

            const std::uint32_t p_src = src_offset + p_local;
            if (p_src >= n_src)
                return;

            const float2 mm = src_bounds[p_src / 256u];
            for (std::uint32_t c = 0; c < n_cells_per_prim; ++c) {
                const float v = DC::decode(
                    src_u16[lfs::core::sh_value::shAtU16(p_src, c, n_cells_per_prim)],
                    mm.x, mm.y);
                const std::uint32_t slot = c / 4u;
                const std::uint32_t comp = c % 4u;
                if (slot >= slots_per_prim)
                    break;
                float4 f4 = dst[shAtF4(p_local, slot, slots_per_prim)];
                if (comp == 0)
                    f4.x = v;
                else if (comp == 1)
                    f4.y = v;
                else if (comp == 2)
                    f4.z = v;
                else
                    f4.w = v;
                dst[shAtF4(p_local, slot, slots_per_prim)] = f4;
            }
        }

        __global__ void decode_u16_gathered_to_canonical_kernel(
            const std::uint16_t* __restrict__ src_u16,
            const float2* __restrict__ src_bounds,
            const std::int64_t* __restrict__ perm,
            float* __restrict__ dst_canonical,
            std::uint32_t n_dst,
            std::uint32_t n_src,
            std::uint32_t n_cells_per_prim) {
            using DC = lfs::core::sh_value::DeviceCodec16;
            const std::uint32_t p_local = blockIdx.x * blockDim.x + threadIdx.x;
            if (p_local >= n_dst)
                return;

            float* row = dst_canonical +
                         static_cast<std::size_t>(p_local) *
                             static_cast<std::size_t>(n_cells_per_prim);
            for (std::uint32_t c = 0; c < n_cells_per_prim; ++c) {
                row[c] = 0.0f;
            }

            const std::int64_t src = perm[p_local];
            if (src < 0 || src >= static_cast<std::int64_t>(n_src))
                return;

            const auto src_p = static_cast<std::uint32_t>(src);
            const float2 mm = src_bounds[src_p / 256u];
            for (std::uint32_t c = 0; c < n_cells_per_prim; ++c) {
                row[c] = DC::decode(
                    src_u16[lfs::core::sh_value::shAtU16(src_p, c, n_cells_per_prim)],
                    mm.x, mm.y);
            }
        }

        __global__ void overlay_canonical_into_float4_kernel(
            const float* __restrict__ src_canonical,
            const std::int64_t* __restrict__ dest_indices,
            float* __restrict__ dst_f4_as_float,
            std::uint32_t group_offset,
            std::uint32_t n_group,
            std::uint32_t block_start,
            std::uint32_t n_chunk,
            std::uint32_t slots_per_prim,
            std::uint32_t n_cells_per_prim) {
            const std::uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
            if (i >= n_group)
                return;

            const std::int64_t dest = dest_indices[group_offset + i];
            const std::int64_t local64 = dest - static_cast<std::int64_t>(block_start);
            if (local64 < 0 || local64 >= static_cast<std::int64_t>(n_chunk))
                return;
            const auto local = static_cast<std::uint32_t>(local64);

            const float* row = src_canonical +
                               static_cast<std::size_t>(group_offset + i) *
                                   static_cast<std::size_t>(n_cells_per_prim);
            float4* dst = reinterpret_cast<float4*>(dst_f4_as_float);
            for (std::uint32_t k = 0; k < slots_per_prim; ++k) {
                const std::uint32_t base = k * 4u;
                const float x = base < n_cells_per_prim ? row[base] : 0.0f;
                const float y = base + 1u < n_cells_per_prim ? row[base + 1u] : 0.0f;
                const float z = base + 2u < n_cells_per_prim ? row[base + 2u] : 0.0f;
                const float w = base + 3u < n_cells_per_prim ? row[base + 3u] : 0.0f;
                dst[shAtF4(local, k, slots_per_prim)] = make_float4(x, y, z, w);
            }
        }

        __global__ void fill_quant_block_ids_f32_kernel(
            const std::int64_t* __restrict__ dest_indices,
            float* __restrict__ block_ids,
            std::uint32_t n) {
            const std::uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
            if (i >= n)
                return;
            const std::int64_t dest = dest_indices[i];
            block_ids[i] = dest < 0 ? -1.0f : static_cast<float>(dest / 256);
        }

        __global__ void mark_sorted_block_run_starts_kernel(
            const float* __restrict__ keys,
            int* __restrict__ flags,
            int n) {
            const int i = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
            if (i >= n)
                return;
            flags[i] = (i == 0 || keys[i] != keys[i - 1]) ? 1 : 0;
        }

        __global__ void compact_sorted_block_runs_kernel(
            const float* __restrict__ keys,
            const int* __restrict__ flags,
            const int* __restrict__ compact_idx,
            int* __restrict__ unique_block_ids,
            int* __restrict__ run_offsets,
            int* __restrict__ n_runs_out,
            int n) {
            const int i = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
            if (i >= n)
                return;
            if (flags[i]) {
                unique_block_ids[compact_idx[i]] = static_cast<int>(keys[i]);
                run_offsets[compact_idx[i]] = i;
            }
            if (i == n - 1) {
                *n_runs_out = compact_idx[i] + flags[i];
            }
        }

        // One CUDA block per unique touched 256-splat quant-block.
        __global__ void reencode_touched_q16_block_kernel(
            std::uint16_t* __restrict__ codes,
            float2* __restrict__ bounds,
            const float* __restrict__ sorted_canonical,
            const std::int64_t* __restrict__ sorted_dest,
            const int* __restrict__ unique_block_ids,
            const int* __restrict__ run_offsets,
            const int* __restrict__ n_runs_device,
            std::uint32_t n_sorted,
            std::uint32_t n_prims,
            std::uint32_t n_decode_src,
            std::uint32_t n_cells_per_prim) {
            const int n_runs = *n_runs_device;
            if (static_cast<int>(blockIdx.x) >= n_runs)
                return;

            const int qblock = unique_block_ids[blockIdx.x];
            if (qblock < 0)
                return;

            const std::uint32_t block_start = static_cast<std::uint32_t>(qblock) * 256u;
            if (block_start >= n_prims)
                return;

            const std::uint32_t lane = threadIdx.x;
            const std::uint32_t n_in =
                (n_prims - block_start < 256u) ? (n_prims - block_start) : 256u;
            const std::uint32_t p = block_start + lane;
            const bool in_range = lane < n_in;
            const std::uint32_t n_decode =
                n_decode_src > block_start
                    ? ((n_decode_src - block_start < n_in) ? (n_decode_src - block_start) : n_in)
                    : 0u;
            const std::uint32_t n_cells =
                n_cells_per_prim > 48u ? 48u : n_cells_per_prim;

            // 256 * 45 u16 = 23040 B. Layout matches pad-dropped shAtU16
            // for local lanes 0..255 of this quant-block.
            __shared__ std::uint16_t s_codes[256 * 45];
            if (n_decode > 0u && lane < n_in) {
                for (std::uint32_t c = 0; c < n_cells; ++c) {
                    s_codes[lfs::core::sh_value::shAtU16(lane, c, n_cells_per_prim)] =
                        codes[lfs::core::sh_value::shAtU16(p, c, n_cells_per_prim)];
                }
            }
            __syncthreads();

            float cells[48];
            if (in_range && lane < n_decode) {
                using DC = lfs::core::sh_value::DeviceCodec16;
                const float2 old_mm = bounds[qblock];
                for (std::uint32_t c = 0; c < n_cells; ++c) {
                    cells[c] = DC::decode(
                        s_codes[lfs::core::sh_value::shAtU16(lane, c, n_cells_per_prim)],
                        old_mm.x,
                        old_mm.y);
                }
            } else if (in_range) {
                for (std::uint32_t c = 0; c < n_cells; ++c) {
                    cells[c] = 0.0f;
                }
            }

            const int run_start = run_offsets[blockIdx.x];
            const int run_end =
                (static_cast<int>(blockIdx.x) + 1 == n_runs)
                    ? static_cast<int>(n_sorted)
                    : run_offsets[blockIdx.x + 1];
            if (in_range) {
                for (int i = run_start; i < run_end; ++i) {
                    const std::int64_t dest = sorted_dest[i];
                    if (dest != static_cast<std::int64_t>(p))
                        continue;
                    const float* row =
                        sorted_canonical +
                        static_cast<std::size_t>(i) *
                            static_cast<std::size_t>(n_cells_per_prim);
                    for (std::uint32_t c = 0; c < n_cells; ++c) {
                        cells[c] = row[c];
                    }
                }
            }

            float local_lo = 1e30f, local_hi = -1e30f;
            if (in_range) {
                for (std::uint32_t c = 0; c < n_cells; ++c) {
                    local_lo = fminf(local_lo, cells[c]);
                    local_hi = fmaxf(local_hi, cells[c]);
                }
            }
            const float2 mm = reduce_quant_block_minmax(lane, in_range, local_lo, local_hi);
            if (lane == 0) {
                bounds[qblock] = mm;
            }
            __syncthreads();
            if (!in_range)
                return;
            encode_prim_u16_cells(codes, mm, p, n_cells, n_cells_per_prim, cells);
        }

        __global__ void encode_u16_gathered_block_kernel(
            const std::uint16_t* __restrict__ src_u16,
            const float2* __restrict__ src_bounds,
            const std::int64_t* __restrict__ perm,
            std::uint16_t* __restrict__ dest_u16,
            float2* __restrict__ dest_bounds,
            std::uint32_t n_dst,
            std::uint32_t n_src,
            std::uint32_t n_cells_per_prim) {
            const std::uint32_t quant_block = blockIdx.x;
            const std::uint32_t lane = threadIdx.x;
            const std::uint32_t p = quant_block * 256u + lane;
            const bool in_range = p < n_dst;
            const std::uint32_t n_cells =
                n_cells_per_prim > 48u ? 48u : n_cells_per_prim;

            float cells[48];
            float local_lo = 1e30f, local_hi = -1e30f;
            if (in_range) {
                for (std::uint32_t c = 0; c < n_cells; ++c) {
                    cells[c] = 0.0f;
                }
                const std::int64_t src = perm[p];
                if (src >= 0 && src < static_cast<std::int64_t>(n_src)) {
                    const auto src_p = static_cast<std::uint32_t>(src);
                    decode_prim_u16_cells(
                        src_u16, src_bounds[src_p / 256u], src_p, n_cells, n_cells_per_prim, cells);
                }
                for (std::uint32_t c = 0; c < n_cells; ++c) {
                    local_lo = fminf(local_lo, cells[c]);
                    local_hi = fmaxf(local_hi, cells[c]);
                }
            }

            const float2 mm = reduce_quant_block_minmax(lane, in_range, local_lo, local_hi);
            if (lane == 0) {
                dest_bounds[quant_block] = mm;
            }
            __syncthreads();
            if (!in_range)
                return;
            encode_prim_u16_cells(dest_u16, mm, p, n_cells, n_cells_per_prim, cells);
        }

        void exclusive_sum_i32(
            const int* in,
            int* out,
            int n,
            cudaStream_t stream) {
            std::size_t bytes = 0;
            LFS_CUDA_CHECK_MSG(
                cub::DeviceScan::ExclusiveSum(nullptr, bytes, in, out, n, stream),
                "q16 block-run exclusive-sum workspace query");
            lfs::core::UniqueCudaAllocation<lfs::core::StreamOrderedCudaAllocator> workspace;
            void* ws = nullptr;
            if (bytes > 0) {
                workspace.allocate(bytes, stream, "q16.block_runs.scan");
                ws = workspace.get();
            }
            LFS_CUDA_CHECK_MSG(
                cub::DeviceScan::ExclusiveSum(ws, bytes, in, out, n, stream),
                "q16 block-run exclusive-sum");
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
        const auto n_cells = lfs::core::sh_value_quant::n_value_cells_per_prim(coeffs_rest);
        const auto n_bounds = lfs::core::sh_value_quant::n_bounds_for_prims(n_primitives);
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

    void decode_shN_u16_gathered_to_float4(
        const std::uint16_t* src_u16,
        const float* src_bounds_float2,
        const std::int64_t* perm,
        float* dst_float4_swizzled,
        std::size_t dest_offset,
        std::size_t n_dst,
        std::size_t n_src_primitives,
        std::uint32_t coeffs_rest,
        cudaStream_t stream) {
        if (n_dst == 0 || coeffs_rest == 0)
            return;
        if (!src_u16 || !src_bounds_float2 || !perm || !dst_float4_swizzled) {
            throw std::invalid_argument("Invalid gathered q16 SH decode arguments");
        }
        const auto slots = lfs::core::sh_float4_slots_for_rest(coeffs_rest);
        const auto n_cells = lfs::core::sh_value_quant::n_value_cells_per_prim(coeffs_rest);
        const unsigned blocks =
            static_cast<unsigned>((n_dst + kThreads - 1) / kThreads);
        decode_u16_gathered_to_float4_kernel<<<blocks, kThreads, 0, stream>>>(
            src_u16,
            reinterpret_cast<const float2*>(src_bounds_float2),
            perm,
            dst_float4_swizzled,
            static_cast<std::uint32_t>(dest_offset),
            static_cast<std::uint32_t>(n_dst),
            static_cast<std::uint32_t>(n_src_primitives),
            slots,
            n_cells);
        LFS_CUDA_CHECK_MSG(cudaGetLastError(), "decode_shN_u16_gathered_to_float4");
    }

    void decode_shN_u16_range_to_float4(
        const std::uint16_t* src_u16,
        const float* src_bounds_float2,
        float* dst_float4_swizzled,
        std::size_t src_offset,
        std::size_t n_dst,
        std::size_t n_src_primitives,
        std::uint32_t coeffs_rest,
        cudaStream_t stream) {
        if (n_dst == 0 || coeffs_rest == 0)
            return;
        if (!src_u16 || !src_bounds_float2 || !dst_float4_swizzled) {
            throw std::invalid_argument("Invalid ranged q16 SH decode arguments");
        }
        const auto slots = lfs::core::sh_float4_slots_for_rest(coeffs_rest);
        const auto n_cells = lfs::core::sh_value_quant::n_value_cells_per_prim(coeffs_rest);
        const unsigned blocks =
            static_cast<unsigned>((n_dst + kThreads - 1) / kThreads);
        decode_u16_range_to_float4_kernel<<<blocks, kThreads, 0, stream>>>(
            src_u16,
            reinterpret_cast<const float2*>(src_bounds_float2),
            dst_float4_swizzled,
            static_cast<std::uint32_t>(src_offset),
            static_cast<std::uint32_t>(n_dst),
            static_cast<std::uint32_t>(n_src_primitives),
            slots,
            n_cells);
        LFS_CUDA_CHECK_MSG(cudaGetLastError(), "decode_shN_u16_range_to_float4");
    }

    void decode_shN_u16_gathered_to_canonical(
        const std::uint16_t* src_u16,
        const float* src_bounds_float2,
        const std::int64_t* perm,
        float* dst_canonical,
        std::size_t n_dst,
        std::size_t n_src_primitives,
        std::uint32_t coeffs_rest,
        cudaStream_t stream) {
        if (n_dst == 0 || coeffs_rest == 0)
            return;
        if (!src_u16 || !src_bounds_float2 || !perm || !dst_canonical) {
            throw std::invalid_argument("Invalid gathered q16 SH canonical decode arguments");
        }
        const auto n_cells = lfs::core::sh_value_quant::n_value_cells_per_prim(coeffs_rest);
        const unsigned blocks =
            static_cast<unsigned>((n_dst + kThreads - 1) / kThreads);
        decode_u16_gathered_to_canonical_kernel<<<blocks, kThreads, 0, stream>>>(
            src_u16,
            reinterpret_cast<const float2*>(src_bounds_float2),
            perm,
            dst_canonical,
            static_cast<std::uint32_t>(n_dst),
            static_cast<std::uint32_t>(n_src_primitives),
            n_cells);
        LFS_CUDA_CHECK_MSG(cudaGetLastError(), "decode_shN_u16_gathered_to_canonical");
    }

    void overlay_canonical_into_float4_chunk(
        const float* src_canonical,
        const std::int64_t* dest_indices,
        float* dst_float4_swizzled,
        std::size_t group_offset,
        std::size_t n_group,
        std::size_t block_start,
        std::size_t n_chunk,
        std::uint32_t coeffs_rest,
        cudaStream_t stream) {
        if (n_group == 0 || n_chunk == 0 || coeffs_rest == 0)
            return;
        if (!src_canonical || !dest_indices || !dst_float4_swizzled) {
            throw std::invalid_argument("Invalid q16 SH overlay arguments");
        }
        const auto slots = lfs::core::sh_float4_slots_for_rest(coeffs_rest);
        const auto n_cells = lfs::core::sh_value_quant::n_value_cells_per_prim(coeffs_rest);
        const unsigned blocks =
            static_cast<unsigned>((n_group + kThreads - 1) / kThreads);
        overlay_canonical_into_float4_kernel<<<blocks, kThreads, 0, stream>>>(
            src_canonical,
            dest_indices,
            dst_float4_swizzled,
            static_cast<std::uint32_t>(group_offset),
            static_cast<std::uint32_t>(n_group),
            static_cast<std::uint32_t>(block_start),
            static_cast<std::uint32_t>(n_chunk),
            slots,
            n_cells);
        LFS_CUDA_CHECK_MSG(cudaGetLastError(), "overlay_canonical_into_float4_chunk");
    }

    void fill_quant_block_ids_f32(
        const std::int64_t* dest_indices,
        float* block_ids_f32,
        std::size_t n,
        cudaStream_t stream) {
        if (n == 0)
            return;
        if (!dest_indices || !block_ids_f32) {
            throw std::invalid_argument("Invalid q16 block-id fill arguments");
        }
        const unsigned blocks =
            static_cast<unsigned>((n + kThreads - 1) / kThreads);
        fill_quant_block_ids_f32_kernel<<<blocks, kThreads, 0, stream>>>(
            dest_indices,
            block_ids_f32,
            static_cast<std::uint32_t>(n));
        LFS_CUDA_CHECK_MSG(cudaGetLastError(), "fill_quant_block_ids_f32");
    }

    void build_sorted_block_runs(
        const float* sorted_block_ids,
        std::int32_t* unique_block_ids,
        std::int32_t* run_offsets,
        std::int32_t* n_runs_device,
        std::size_t n,
        cudaStream_t stream) {
        if (n == 0)
            return;
        if (!sorted_block_ids || !unique_block_ids || !run_offsets || !n_runs_device) {
            throw std::invalid_argument("Invalid q16 block-run arguments");
        }
        if (n > static_cast<std::size_t>(std::numeric_limits<int>::max())) {
            throw std::out_of_range("q16 block-run item count exceeds int range");
        }
        const int n_i = static_cast<int>(n);
        const std::size_t bytes = static_cast<std::size_t>(n_i) * sizeof(int);
        lfs::core::UniqueCudaAllocation<lfs::core::StreamOrderedCudaAllocator> flags_alloc(
            bytes, stream, "q16.block_runs.flags");
        lfs::core::UniqueCudaAllocation<lfs::core::StreamOrderedCudaAllocator> compact_alloc(
            bytes, stream, "q16.block_runs.compact");
        auto* flags = flags_alloc.as<int>();
        auto* compact_idx = compact_alloc.as<int>();

        const unsigned blocks =
            static_cast<unsigned>((n + kThreads - 1) / kThreads);
        mark_sorted_block_run_starts_kernel<<<blocks, kThreads, 0, stream>>>(
            sorted_block_ids, flags, n_i);
        LFS_CUDA_CHECK_MSG(cudaGetLastError(), "mark_sorted_block_run_starts");
        exclusive_sum_i32(flags, compact_idx, n_i, stream);
        compact_sorted_block_runs_kernel<<<blocks, kThreads, 0, stream>>>(
            sorted_block_ids,
            flags,
            compact_idx,
            unique_block_ids,
            run_offsets,
            n_runs_device,
            n_i);
        LFS_CUDA_CHECK_MSG(cudaGetLastError(), "compact_sorted_block_runs");
    }

    void reencode_touched_q16_blocks(
        std::uint16_t* codes,
        float* bounds_float2,
        const float* sorted_canonical,
        const std::int64_t* sorted_dest,
        const std::int32_t* unique_block_ids,
        const std::int32_t* run_offsets,
        const std::int32_t* n_runs_device,
        std::size_t n_sorted,
        std::size_t n_prims,
        std::size_t n_decode_src,
        std::uint32_t coeffs_rest,
        cudaStream_t stream) {
        if (n_sorted == 0 || coeffs_rest == 0 || n_prims == 0)
            return;
        if (!codes || !bounds_float2 || !sorted_canonical || !sorted_dest ||
            !unique_block_ids || !run_offsets || !n_runs_device) {
            throw std::invalid_argument("Invalid q16 touched-block reencode arguments");
        }
        const auto n_cells = lfs::core::sh_value_quant::n_value_cells_per_prim(coeffs_rest);
        const unsigned grid = static_cast<unsigned>(n_sorted);
        const auto n_decode_u32 =
            n_decode_src > std::numeric_limits<std::uint32_t>::max()
                ? std::numeric_limits<std::uint32_t>::max()
                : static_cast<std::uint32_t>(n_decode_src);
        reencode_touched_q16_block_kernel<<<grid, 256, 0, stream>>>(
            codes,
            reinterpret_cast<float2*>(bounds_float2),
            sorted_canonical,
            sorted_dest,
            unique_block_ids,
            run_offsets,
            n_runs_device,
            static_cast<std::uint32_t>(n_sorted),
            static_cast<std::uint32_t>(n_prims),
            n_decode_u32,
            n_cells);
        LFS_CUDA_CHECK_MSG(cudaGetLastError(), "reencode_touched_q16_blocks");
    }

    void encode_shN_u16_gathered(
        const std::uint16_t* src_u16,
        const float* src_bounds_float2,
        const std::int64_t* perm,
        std::uint16_t* dest_u16,
        float* dest_bounds_float2,
        std::size_t n_dst,
        std::size_t n_src_primitives,
        std::uint32_t coeffs_rest,
        cudaStream_t stream) {
        if (n_dst == 0 || coeffs_rest == 0)
            return;
        if (!src_u16 || !src_bounds_float2 || !perm || !dest_u16 || !dest_bounds_float2) {
            throw std::invalid_argument("Invalid gathered q16 SH encode arguments");
        }
        const auto n_cells = lfs::core::sh_value_quant::n_value_cells_per_prim(coeffs_rest);
        const auto n_bounds = lfs::core::sh_value_quant::n_bounds_for_prims(n_dst);
        if (n_bounds == 0)
            return;
        encode_u16_gathered_block_kernel<<<static_cast<unsigned>(n_bounds), 256, 0, stream>>>(
            src_u16,
            reinterpret_cast<const float2*>(src_bounds_float2),
            perm,
            dest_u16,
            reinterpret_cast<float2*>(dest_bounds_float2),
            static_cast<std::uint32_t>(n_dst),
            static_cast<std::uint32_t>(n_src_primitives),
            n_cells);
        LFS_CUDA_CHECK_MSG(cudaGetLastError(), "encode_shN_u16_gathered");
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
        const auto n_cells = lfs::core::sh_value_quant::n_value_cells_per_prim(coeffs_rest);
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
        const auto n_cells = lfs::core::sh_value_quant::n_value_cells_per_prim(layout_coeffs_rest);
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

} // namespace lfs::core::sh_value_quant

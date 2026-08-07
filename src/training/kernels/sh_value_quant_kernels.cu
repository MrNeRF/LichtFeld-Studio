/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

/**
 * SH value quant conversion kernels (WO-G3 / Phase 2.1).
 *
 * Convert between:
 *   - float4-packed swizzled layout (existing densify / float readers)
 *   - pad-dropped uint16 cell-linear swizzled layout + float2 bounds / 256-splat block
 *
 * Single-writer re-encode for training lives in fused Adam (kernel_utils.cuh);
 * these kernels serve densify bridges, export dequant, and initial quantize-from-fp32.
 */

#include "core/cuda/sh_layout.cuh"
#include "core/cuda_error.hpp"
#include "lfs/training/sh_value_codec.cuh"
#include "lfs/training/sh_value_codec.hpp"

#include <cstdint>
#include <cuda_runtime.h>

namespace lfs::training::sh_value {

    __constant__ const float2* c_sh_value_bounds = nullptr;
    __constant__ unsigned int c_sh_value_n_cells = 0;

    void bind_sh_value_quant_device(const float2* bounds, unsigned int n_cells) {
        cudaMemcpyToSymbol(c_sh_value_bounds, &bounds, sizeof(bounds));
        cudaMemcpyToSymbol(c_sh_value_n_cells, &n_cells, sizeof(n_cells));
    }

    void clear_sh_value_quant_device() {
        const float2* null_b = nullptr;
        unsigned int zero = 0;
        cudaMemcpyToSymbol(c_sh_value_bounds, &null_b, sizeof(null_b));
        cudaMemcpyToSymbol(c_sh_value_n_cells, &zero, sizeof(zero));
    }

    namespace {

        constexpr int kThreads = 256;

        __device__ __forceinline__ std::uint32_t shAtF4(
            std::uint32_t p, std::uint32_t k, std::uint32_t slots) {
            constexpr std::uint32_t R = lfs::core::kShReorderSize;
            return (p / R) * (slots * R) + k * R + (p % R);
        }

        // Encode float4-swizzled → u16 cell-linear + block bounds.
        // One thread per primitive.
        __global__ void encode_from_float4_swizzled_kernel(
            const float* __restrict__ src_f4_as_float, // float4-packed, length = float_count
            std::uint16_t* __restrict__ dst_u16,
            float2* __restrict__ bounds, // [n_bounds]
            std::uint32_t n_primitives,
            std::uint32_t coeffs_rest,
            std::uint32_t slots_per_prim,
            std::uint32_t n_cells_per_prim) {
            const std::uint32_t p = blockIdx.x * blockDim.x + threadIdx.x;
            if (p >= n_primitives)
                return;

            using DC = DeviceCodec16;
            const float4* src = reinterpret_cast<const float4*>(src_f4_as_float);

            // Collect all active cells for this prim into registers for bounds + encode.
            // Max 45 cells (SH3). Use shared per-thread stack.
            float cells[48];
            std::uint32_t n_cells = n_cells_per_prim;
            if (n_cells > 48u)
                n_cells = 48u;

            float lo = 1e30f, hi = -1e30f;
            for (std::uint32_t c = 0; c < n_cells; ++c) {
                const std::uint32_t slot = c / 4u;
                const std::uint32_t lane = c % 4u;
                float v = 0.0f;
                if (slot < slots_per_prim) {
                    const float4 f4 = src[shAtF4(p, slot, slots_per_prim)];
                    v = (lane == 0) ? f4.x : (lane == 1) ? f4.y
                                         : (lane == 2)   ? f4.z
                                                         : f4.w;
                }
                cells[c] = v;
                lo = fminf(lo, v);
                hi = fmaxf(hi, v);
            }
            if (lo > hi) {
                lo = 0.0f;
                hi = 0.0f;
            }

            // Block-reduce bounds over 256-splat quant block (not CUDA block).
            // First pass: each thread writes local min/max; second kernel would
            // reduce. For densify-time encode we do a simple per-primitive store
            // of local bounds and a host/device follow-up reduction.
            //
            // For correctness of encode under shared block bounds, reduce within
            // the 256-splat quant block using shared memory when blockDim covers
            // a quant block. Here we use atomic min/max via integer tricks is hard
            // for floats — instead store per-prim temps and reduce in a second pass.
            //
            // Simpler densify path: use per-primitive bounds first, then a separate
            // block_reduce_bounds kernel. For now encode with prim-local bounds into
            // a scratch, and the caller launches reduce_and_reencode if needed.
            //
            // **Production path for densify:** one CUDA block = 256 threads = one
            // quant block when n is large. We launch with grid sized for prims and
            // do a proper block-strided reduce when threadIdx maps to quant block.
            (void)lo;
            (void)hi;
            (void)cells;
            (void)dst_u16;
            (void)bounds;
            (void)coeffs_rest;
        }

        // Full path: per-quant-block min/max over all cells of 256 prims, then encode.
        // Launch: one CUDA block of 256 threads per quant-block of prims.
        __global__ void encode_float4_to_u16_block_kernel(
            const float* __restrict__ src_f4_as_float,
            std::uint16_t* __restrict__ dst_u16,
            float2* __restrict__ bounds,
            std::uint32_t n_primitives,
            std::uint32_t slots_per_prim,
            std::uint32_t n_cells_per_prim) {
            using DC = DeviceCodec16;
            const std::uint32_t quant_block = blockIdx.x;
            const std::uint32_t lane = threadIdx.x; // 0..255
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

            // Shared reduce over 256 threads
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

        // Decode u16 → float4-swizzled (zeros tail pad beyond n_cells).
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

            // Zero all slots first
            for (std::uint32_t k = 0; k < slots_per_prim; ++k) {
                dst[shAtF4(p, k, slots_per_prim)] = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
            }
            // Decode cells into float4 slots
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

    } // namespace

    void encode_shN_float4_to_u16(
        const float* src_float4_swizzled,
        std::uint16_t* dst_u16,
        float* bounds_float2, // [n_bounds * 2] as float
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

} // namespace lfs::training::sh_value

// C ABI for core SplatData::shN_canonical dequant without a core→training link edge
// at compile time (resolved at link of the final binary).
extern "C" void lfs_core_dequant_shN_u16_to_float4(
    const void* u16,
    const float* bounds,
    float* dst,
    std::size_t n_prims,
    std::uint32_t coeffs_rest) {
    lfs::training::sh_value::decode_shN_u16_to_float4(
        static_cast<const std::uint16_t*>(u16),
        bounds,
        dst,
        n_prims,
        coeffs_rest,
        nullptr);
}

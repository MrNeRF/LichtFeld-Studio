/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "core/cuda/sh_layout.cuh"
#include "core/logger.hpp"
#include <cuda_runtime.h>

namespace lfs::core {

    namespace {

        __device__ __forceinline__ std::uint32_t shAt_device(std::uint32_t p, std::uint32_t k) {
            const std::uint32_t block = p / SH_REORDER_SIZE;
            const std::uint32_t lane = p % SH_REORDER_SIZE;
            return block * (SH_MAX_COEFFS_REST * SH_REORDER_SIZE) + k * SH_REORDER_SIZE + lane;
        }

        constexpr int BLOCK = 256;

        __global__ void reorder_sh_kernel(
            const float* __restrict__ src,
            float* __restrict__ dst,
            std::uint32_t n_primitives,
            std::uint32_t active_coeffs_rest,
            std::uint32_t padded_n) {
            const std::uint32_t p = blockIdx.x * blockDim.x + threadIdx.x;
            if (p >= padded_n)
                return;

            // For each of the 16 (SH_MAX_COEFFS_REST) coefficient slots, write either source data
            // or zero (for padding lanes or coefficient slots beyond the active SH degree).
            for (std::uint32_t k = 0; k < SH_MAX_COEFFS_REST; ++k) {
                const std::uint32_t dst_idx = shAt_device(p, k);
                const bool valid = (p < n_primitives) && (k < active_coeffs_rest);
                if (valid) {
                    const std::uint32_t src_idx = p * active_coeffs_rest + k;
                    dst[dst_idx * 3 + 0] = src[src_idx * 3 + 0];
                    dst[dst_idx * 3 + 1] = src[src_idx * 3 + 1];
                    dst[dst_idx * 3 + 2] = src[src_idx * 3 + 2];
                } else {
                    dst[dst_idx * 3 + 0] = 0.0f;
                    dst[dst_idx * 3 + 1] = 0.0f;
                    dst[dst_idx * 3 + 2] = 0.0f;
                }
            }
        }

        __global__ void undo_reorder_sh_kernel(
            const float* __restrict__ src,
            float* __restrict__ dst,
            std::uint32_t n_primitives,
            std::uint32_t active_coeffs_rest) {
            const std::uint32_t p = blockIdx.x * blockDim.x + threadIdx.x;
            if (p >= n_primitives)
                return;

            for (std::uint32_t k = 0; k < active_coeffs_rest; ++k) {
                const std::uint32_t src_idx = shAt_device(p, k);
                const std::uint32_t dst_idx = p * active_coeffs_rest + k;
                dst[dst_idx * 3 + 0] = src[src_idx * 3 + 0];
                dst[dst_idx * 3 + 1] = src[src_idx * 3 + 1];
                dst[dst_idx * 3 + 2] = src[src_idx * 3 + 2];
            }
        }

        template <typename IndexT>
        __global__ void zero_at_indices_kernel(
            float* __restrict__ buffer,
            const IndexT* __restrict__ indices,
            std::uint32_t n_indices) {
            const std::uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
            if (i >= n_indices)
                return;
            const std::uint32_t p = static_cast<std::uint32_t>(indices[i]);
            for (std::uint32_t k = 0; k < SH_MAX_COEFFS_REST; ++k) {
                const std::uint32_t idx = shAt_device(p, k);
                buffer[idx * 3 + 0] = 0.0f;
                buffer[idx * 3 + 1] = 0.0f;
                buffer[idx * 3 + 2] = 0.0f;
            }
        }

        template <typename IndexT>
        __global__ void gather_self_kernel(
            const float* __restrict__ src,
            float* __restrict__ dst,
            const IndexT* __restrict__ src_indices,
            std::uint32_t n_dst,
            std::uint32_t dst_offset) {
            const std::uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
            if (i >= n_dst)
                return;
            const std::uint32_t src_p = static_cast<std::uint32_t>(src_indices[i]);
            const std::uint32_t dst_p = dst_offset + i;
            for (std::uint32_t k = 0; k < SH_MAX_COEFFS_REST; ++k) {
                const std::uint32_t s = shAt_device(src_p, k);
                const std::uint32_t d = shAt_device(dst_p, k);
                dst[d * 3 + 0] = src[s * 3 + 0];
                dst[d * 3 + 1] = src[s * 3 + 1];
                dst[d * 3 + 2] = src[s * 3 + 2];
            }
        }

        __global__ void gather_from_linear_kernel(
            float* __restrict__ dst,
            std::uint32_t dst_offset,
            const float* __restrict__ src_linear,
            std::uint32_t n_src,
            std::uint32_t active_coeffs_rest) {
            const std::uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
            if (i >= n_src)
                return;
            const std::uint32_t dst_p = dst_offset + i;
            for (std::uint32_t k = 0; k < SH_MAX_COEFFS_REST; ++k) {
                const std::uint32_t d = shAt_device(dst_p, k);
                if (k < active_coeffs_rest) {
                    const std::uint32_t s = i * active_coeffs_rest + k;
                    dst[d * 3 + 0] = src_linear[s * 3 + 0];
                    dst[d * 3 + 1] = src_linear[s * 3 + 1];
                    dst[d * 3 + 2] = src_linear[s * 3 + 2];
                } else {
                    dst[d * 3 + 0] = 0.0f;
                    dst[d * 3 + 1] = 0.0f;
                    dst[d * 3 + 2] = 0.0f;
                }
            }
        }

        __global__ void scatter_linear_kernel(
            float* __restrict__ dst,
            const int* __restrict__ dst_indices,
            const float* __restrict__ src_linear,
            std::uint32_t n_src,
            std::uint32_t active_coeffs_rest) {
            const std::uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
            if (i >= n_src)
                return;
            const std::uint32_t dst_p = static_cast<std::uint32_t>(dst_indices[i]);
            for (std::uint32_t k = 0; k < active_coeffs_rest; ++k) {
                const std::uint32_t d = shAt_device(dst_p, k);
                const std::uint32_t s = i * active_coeffs_rest + k;
                dst[d * 3 + 0] = src_linear[s * 3 + 0];
                dst[d * 3 + 1] = src_linear[s * 3 + 1];
                dst[d * 3 + 2] = src_linear[s * 3 + 2];
            }
        }

    } // namespace

    void reorder_sh_to_swizzled(
        const float* src_canonical,
        float* dst_swizzled,
        std::size_t n_primitives,
        std::uint32_t active_coeffs_rest,
        cudaStream_t stream) {
        if (active_coeffs_rest == 0) {
            // No source data; zero the entire destination so dead lanes are safe for Adam.
            const std::size_t bytes = sh_swizzled_byte_count(n_primitives);
            if (bytes > 0) {
                cudaMemsetAsync(dst_swizzled, 0, bytes, stream);
            }
            return;
        }
        const std::uint32_t padded_n = static_cast<std::uint32_t>(sh_swizzled_padded_n(n_primitives));
        if (padded_n == 0)
            return;
        const int grid = static_cast<int>((padded_n + BLOCK - 1) / BLOCK);
        reorder_sh_kernel<<<grid, BLOCK, 0, stream>>>(
            src_canonical, dst_swizzled,
            static_cast<std::uint32_t>(n_primitives), active_coeffs_rest, padded_n);
    }

    void undo_reorder_sh_from_swizzled(
        const float* src_swizzled,
        float* dst_canonical,
        std::size_t n_primitives,
        std::uint32_t active_coeffs_rest,
        cudaStream_t stream) {
        if (n_primitives == 0 || active_coeffs_rest == 0)
            return;
        const int grid = static_cast<int>((n_primitives + BLOCK - 1) / BLOCK);
        undo_reorder_sh_kernel<<<grid, BLOCK, 0, stream>>>(
            src_swizzled, dst_canonical,
            static_cast<std::uint32_t>(n_primitives), active_coeffs_rest);
    }

    void shN_swizzled_zero_at_indices(
        float* buffer_swizzled,
        const int* indices,
        std::size_t n_indices,
        cudaStream_t stream) {
        if (n_indices == 0)
            return;
        const int grid = static_cast<int>((n_indices + BLOCK - 1) / BLOCK);
        zero_at_indices_kernel<int><<<grid, BLOCK, 0, stream>>>(
            buffer_swizzled, indices, static_cast<std::uint32_t>(n_indices));
    }

    void shN_swizzled_zero_at_indices_i64(
        float* buffer_swizzled,
        const std::int64_t* indices,
        std::size_t n_indices,
        cudaStream_t stream) {
        if (n_indices == 0)
            return;
        const int grid = static_cast<int>((n_indices + BLOCK - 1) / BLOCK);
        zero_at_indices_kernel<std::int64_t><<<grid, BLOCK, 0, stream>>>(
            buffer_swizzled, indices, static_cast<std::uint32_t>(n_indices));
    }

    void shN_swizzled_gather_self(
        const float* src_swizzled,
        float* dst_swizzled,
        const int* src_indices,
        std::size_t n_dst,
        std::size_t dst_offset,
        cudaStream_t stream) {
        if (n_dst == 0)
            return;
        const int grid = static_cast<int>((n_dst + BLOCK - 1) / BLOCK);
        gather_self_kernel<int><<<grid, BLOCK, 0, stream>>>(
            src_swizzled, dst_swizzled, src_indices,
            static_cast<std::uint32_t>(n_dst),
            static_cast<std::uint32_t>(dst_offset));
    }

    void shN_swizzled_gather_self_i64(
        const float* src_swizzled,
        float* dst_swizzled,
        const std::int64_t* src_indices,
        std::size_t n_dst,
        std::size_t dst_offset,
        cudaStream_t stream) {
        if (n_dst == 0)
            return;
        const int grid = static_cast<int>((n_dst + BLOCK - 1) / BLOCK);
        gather_self_kernel<std::int64_t><<<grid, BLOCK, 0, stream>>>(
            src_swizzled, dst_swizzled, src_indices,
            static_cast<std::uint32_t>(n_dst),
            static_cast<std::uint32_t>(dst_offset));
    }

    void shN_swizzled_gather_from_linear(
        float* dst_swizzled,
        std::size_t dst_offset,
        const float* src_linear,
        std::size_t n_src,
        std::uint32_t active_coeffs_rest,
        cudaStream_t stream) {
        if (n_src == 0)
            return;
        const int grid = static_cast<int>((n_src + BLOCK - 1) / BLOCK);
        gather_from_linear_kernel<<<grid, BLOCK, 0, stream>>>(
            dst_swizzled, static_cast<std::uint32_t>(dst_offset),
            src_linear, static_cast<std::uint32_t>(n_src), active_coeffs_rest);
    }

    void shN_swizzled_scatter_linear(
        float* dst_swizzled,
        const int* dst_indices,
        const float* src_linear,
        std::size_t n_src,
        std::uint32_t active_coeffs_rest,
        cudaStream_t stream) {
        if (n_src == 0)
            return;
        const int grid = static_cast<int>((n_src + BLOCK - 1) / BLOCK);
        scatter_linear_kernel<<<grid, BLOCK, 0, stream>>>(
            dst_swizzled, dst_indices, src_linear,
            static_cast<std::uint32_t>(n_src), active_coeffs_rest);
    }

} // namespace lfs::core

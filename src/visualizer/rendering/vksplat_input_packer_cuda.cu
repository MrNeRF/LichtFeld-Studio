/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "vksplat_input_packer_cuda.hpp"

#include <cuda_runtime.h>

namespace lfs::vis::vksplat::detail {
    namespace {
        constexpr int kBlockSize = 256;

        // Pre-sigmoid raw value that drives sigmoid to ~0 (-20 → ≈2e-9). Picked
        // far enough below the activation threshold that fp32 rounding can't
        // promote a soft-deleted splat back into the contributing range.
        constexpr float kDeletedOpacityRaw = -20.0f;

        __global__ void packOpacityMaskingDeletedKernel(
            const float* __restrict__ opacity_raw,
            const bool* __restrict__ deleted_mask,
            float* __restrict__ opacity_dst,
            const std::size_t num_splats) {
            const std::size_t i = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
            if (i >= num_splats) {
                return;
            }
            opacity_dst[i] = deleted_mask[i] ? kDeletedOpacityRaw : opacity_raw[i];
        }
    } // namespace

    cudaError_t launchPackOpacityMaskingDeleted(
        const float* const opacity_raw,
        const bool* const deleted_mask,
        float* const opacity_dst,
        const std::size_t num_splats,
        const cudaStream_t stream) {
        if (opacity_raw == nullptr || deleted_mask == nullptr || opacity_dst == nullptr || num_splats == 0) {
            return cudaErrorInvalidValue;
        }
        const int grid = static_cast<int>((num_splats + kBlockSize - 1) / kBlockSize);
        packOpacityMaskingDeletedKernel<<<grid, kBlockSize, 0, stream>>>(
            opacity_raw,
            deleted_mask,
            opacity_dst,
            num_splats);
        return cudaGetLastError();
    }

} // namespace lfs::vis::vksplat::detail

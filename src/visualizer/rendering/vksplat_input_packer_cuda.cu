/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "vksplat_input_packer_cuda.hpp"

#include <cuda_runtime.h>

namespace lfs::vis::vksplat::detail {
    namespace {
        constexpr int kBlockSize = 256;

        __global__ void packActivatedRotationsKernel(
            const float* __restrict__ rotation_raw,
            float4* __restrict__ rotations_dst,
            const std::size_t num_splats) {
            const std::size_t i = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
            if (i >= num_splats) {
                return;
            }
            const float x = rotation_raw[i * 4 + 0];
            const float y = rotation_raw[i * 4 + 1];
            const float z = rotation_raw[i * 4 + 2];
            const float w = rotation_raw[i * 4 + 3];
            const float norm_sq = fmaxf(x * x + y * y + z * z + w * w, 1.0e-20f);
            const float inv_norm = rsqrtf(norm_sq);
            rotations_dst[i] = make_float4(x * inv_norm, y * inv_norm, z * inv_norm, w * inv_norm);
        }

        __global__ void packScalesOpacsKernel(
            const float* __restrict__ scaling_raw,
            const float* __restrict__ opacity_raw,
            float4* __restrict__ scales_opacs_dst,
            const std::size_t num_splats) {
            const std::size_t i = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
            if (i >= num_splats) {
                return;
            }
            const float opacity = opacity_raw[i];
            const float alpha = 1.0f / (1.0f + expf(-opacity));
            scales_opacs_dst[i] = make_float4(
                expf(scaling_raw[i * 3 + 0]),
                expf(scaling_raw[i * 3 + 1]),
                expf(scaling_raw[i * 3 + 2]),
                alpha);
        }
    } // namespace

    cudaError_t launchPackActivatedRotations(
        const float* const rotation_raw,
        float* const rotations_dst,
        const std::size_t num_splats,
        const cudaStream_t stream) {
        if (rotation_raw == nullptr || rotations_dst == nullptr || num_splats == 0) {
            return cudaErrorInvalidValue;
        }
        const int grid = static_cast<int>((num_splats + kBlockSize - 1) / kBlockSize);
        packActivatedRotationsKernel<<<grid, kBlockSize, 0, stream>>>(
            rotation_raw,
            reinterpret_cast<float4*>(rotations_dst),
            num_splats);
        return cudaGetLastError();
    }

    cudaError_t launchPackScalesOpacs(
        const float* const scaling_raw,
        const float* const opacity_raw,
        float* const scales_opacs_dst,
        const std::size_t num_splats,
        const cudaStream_t stream) {
        if (scaling_raw == nullptr || opacity_raw == nullptr || scales_opacs_dst == nullptr || num_splats == 0) {
            return cudaErrorInvalidValue;
        }
        const int grid = static_cast<int>((num_splats + kBlockSize - 1) / kBlockSize);
        packScalesOpacsKernel<<<grid, kBlockSize, 0, stream>>>(
            scaling_raw,
            opacity_raw,
            reinterpret_cast<float4*>(scales_opacs_dst),
            num_splats);
        return cudaGetLastError();
    }

} // namespace lfs::vis::vksplat::detail

/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "core/cuda_error.hpp"
#include "pruning_kernels.hpp"
#include <cuda_runtime.h>

#include "kernel_stream.hpp"

namespace lfs::training::pruning {

    __global__ void compute_dead_mask_kernel(
        const float* opacities, // [N]
        const float* rotations, // [N, 4]
        uint8_t* dead_mask,     // [N]
        size_t N,
        float min_opacity) {

        const size_t idx = threadIdx.x + blockIdx.x * blockDim.x;
        if (idx >= N)
            return;

        bool is_dead = opacities[idx] <= min_opacity;
        if (!is_dead) {
            const float* q = &rotations[idx * 4];
            const float mag_sq = q[0] * q[0] + q[1] * q[1] + q[2] * q[2] + q[3] * q[3];
            is_dead = mag_sq < 1e-8f;
        }

        dead_mask[idx] = is_dead ? 1 : 0;
    }

    void launch_compute_dead_mask(
        const float* opacities,
        const float* rotations,
        uint8_t* dead_mask,
        size_t N,
        float min_opacity,
        void* stream) {

        if (N == 0) {
            return;
        }

        dim3 threads(256);
        dim3 grid((N + threads.x - 1) / threads.x);

        cudaStream_t cuda_stream = resolve_stream(stream);
        compute_dead_mask_kernel<<<grid, threads, 0, cuda_stream>>>(
            opacities,
            rotations,
            dead_mask,
            N,
            min_opacity);
        LFS_CUDA_LAUNCH_CHECK(cuda_stream, "training.pruning.dead_mask");
    }

    __global__ void compute_near_zero_rotation_mask_kernel(
        const float* rotations, // [N, 4]
        uint8_t* mask,          // [N]
        size_t N) {

        const size_t idx = threadIdx.x + blockIdx.x * blockDim.x;
        if (idx >= N)
            return;

        const float* q = &rotations[idx * 4];
        const float mag_sq = q[0] * q[0] + q[1] * q[1] + q[2] * q[2] + q[3] * q[3];
        mask[idx] = mag_sq < 1e-8f ? 1 : 0;
    }

    void launch_compute_near_zero_rotation_mask(
        const float* rotations,
        uint8_t* mask,
        size_t N,
        void* stream) {

        if (N == 0) {
            return;
        }

        dim3 threads(256);
        dim3 grid((N + threads.x - 1) / threads.x);

        cudaStream_t cuda_stream = resolve_stream(stream);
        compute_near_zero_rotation_mask_kernel<<<grid, threads, 0, cuda_stream>>>(
            rotations,
            mask,
            N);
        LFS_CUDA_LAUNCH_CHECK(cuda_stream, "training.pruning.near_zero_rotation_mask");
    }

} // namespace lfs::training::pruning

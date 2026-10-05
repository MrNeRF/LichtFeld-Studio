/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "core/cuda_error.hpp"
#include "core/tensor/backend/cuda/kernels/tensor_ops.hpp"
#include <cuda_runtime.h>
#include <curand_kernel.h>

namespace lfs::core::tensor_ops {

    namespace {

        __device__ float uniform_unit_interval(curandState* state) {
            constexpr float SCALE = 1.0f / 16777216.0f;
            return static_cast<float>(curand(state) >> 8) * SCALE;
        }

    } // namespace

    // ============= Random Operations Kernels =============

    // Uniform random generation
    __global__ void uniform_kernel(float* data, size_t n, float low, float high,
                                   unsigned long long seed) {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;

        if (idx < n) {
            curandState state;
            curand_init(seed, idx, 0, &state);
            if (low == high) {
                data[idx] = low;
                return;
            }
            const float val = uniform_unit_interval(&state);
            const float result = fmaf(val, high - low, low);
            data[idx] = result < high ? result : nextafterf(high, low);
        }
    }

    // Random integer generation
    __global__ void randint_kernel(int* data, size_t n, int low, int high,
                                   unsigned long long seed) {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;

        if (idx < n) {
            curandState state;
            curand_init(seed, idx, 0, &state);

            const uint64_t random = curand(&state);
            const uint64_t range = static_cast<uint64_t>(
                static_cast<int64_t>(high) - static_cast<int64_t>(low));
            const uint64_t offset = (random * range) >> 32;
            data[idx] = static_cast<int>(static_cast<int64_t>(low) +
                                         static_cast<int64_t>(offset));
        }
    }

    // ============= Launch Functions =============

    void launch_uniform(float* data, size_t n, float low, float high,
                        unsigned long long seed, cudaStream_t stream) {
        if (n == 0)
            return;

        int block_size = 256;
        int grid_size = (n + block_size - 1) / block_size;
        uniform_kernel<<<grid_size, block_size, 0, stream>>>(data, n, low, high, seed);
        LFS_CUDA_LAUNCH_CHECK(stream, "tensor.random.uniform");
    }

    void launch_randint(int* data, size_t n, int low, int high,
                        unsigned long long seed, cudaStream_t stream) {
        if (n == 0)
            return;

        int block_size = 256;
        int grid_size = (n + block_size - 1) / block_size;
        randint_kernel<<<grid_size, block_size, 0, stream>>>(data, n, low, high, seed);
        LFS_CUDA_LAUNCH_CHECK(stream, "tensor.random.randint");
    }

} // namespace lfs::core::tensor_ops

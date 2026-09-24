/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "core/cuda_error.hpp"
#include "tensor_histogram.hpp"

#include <algorithm>

namespace lfs::core::tensor_ops {
    namespace {
        constexpr unsigned kThreads = 256;
        constexpr unsigned kMaxBlocks = 4096;

        __global__ void histogram_u8_kernel(const uint8_t* __restrict__ values,
                                            int32_t* __restrict__ counts, const unsigned size) {
            __shared__ int32_t local_counts[256];
            local_counts[threadIdx.x] = 0;
            __syncthreads();
            const unsigned stride = blockDim.x * gridDim.x;
            for (unsigned index = blockIdx.x * blockDim.x + threadIdx.x; index < size; index += stride) {
                const uint8_t value = values[index];
                if (value != 0)
                    atomicAdd(&local_counts[value], 1);
            }
            __syncthreads();
            const int32_t count = local_counts[threadIdx.x];
            if (count != 0)
                atomicAdd(&counts[threadIdx.x], count);
        }
    } // namespace

    void launch_histogram_u8(const uint8_t* values, int32_t* counts, const size_t size,
                             const cudaStream_t stream) {
        const unsigned blocks = static_cast<unsigned>(std::min<size_t>((size + kThreads - 1) / kThreads, kMaxBlocks));
        histogram_u8_kernel<<<blocks, kThreads, 0, stream>>>(values, counts, static_cast<unsigned>(size));
        LFS_CUDA_LAUNCH_CHECK(stream, "tensor.histogram_u8");
    }
} // namespace lfs::core::tensor_ops

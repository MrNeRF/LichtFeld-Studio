/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#include "core/cuda_error.hpp"
#include "where_scalar.cuh"
#include <algorithm>
#include <cuda_fp16.h>

namespace lfs::core::internal {
    namespace {
        template <typename T>
        __global__ void where_scalar_kernel(T* output, const unsigned char* condition,
                                            float value, const T* source, size_t count) {
            const T scalar = static_cast<T>(value);
            for (size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
                 i < count; i += size_t(blockDim.x) * gridDim.x)
                output[i] = condition[i] ? scalar : source[i];
        }
    } // namespace

    void launch_where_scalar(void* output, const void* condition, float value, const void* source,
                             size_t count, bool half, cudaStream_t stream) {
        const auto blocks = static_cast<unsigned>(std::min<size_t>((count + 255) / 256, 65535));
        const auto* mask = static_cast<const unsigned char*>(condition);
        if (half) {
            where_scalar_kernel<<<blocks, 256, 0, stream>>>(static_cast<__half*>(output), mask,
                                                            value, static_cast<const __half*>(source), count);
            LFS_CUDA_LAUNCH_CHECK(stream, "where_scalar.half");
        } else {
            where_scalar_kernel<<<blocks, 256, 0, stream>>>(static_cast<float*>(output), mask,
                                                            value, static_cast<const float*>(source), count);
            LFS_CUDA_LAUNCH_CHECK(stream, "where_scalar.float");
        }
    }
} // namespace lfs::core::internal

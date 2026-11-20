/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#include "core/cuda_error.hpp"
#include "sharpness_edge.hpp"
#include "sharpness_score.cuh"
#include <algorithm>
#include <cub/block/block_reduce.cuh>
#include <cuda_runtime.h>

namespace lfs::io {
    namespace {
        constexpr int BLOCK_SIZE = 256;
        __global__ void countEdges(const uint8_t* rgb, int w, int h, SharpnessAlgorithm algorithm,
                                   unsigned long long* total) {
            const auto pixels = static_cast<unsigned long long>(w - 2) * (h - 2);
            const auto stride = static_cast<std::ptrdiff_t>(w) * 3;
            unsigned long long count = 0;
            for (auto index = static_cast<unsigned long long>(blockIdx.x) * blockDim.x + threadIdx.x;
                 index < pixels; index += static_cast<unsigned long long>(gridDim.x) * blockDim.x) {
                const auto y = index / (w - 2) + 1;
                const auto x = index % (w - 2) + 1;
                count += detail::sharpEdge(rgb + y * stride + x * 3 + 1, stride, algorithm);
            }
            using Reduce = cub::BlockReduce<unsigned long long, BLOCK_SIZE>;
            __shared__ typename Reduce::TempStorage storage;
            const auto sum = Reduce(storage).Sum(count);
            if (threadIdx.x == 0)
                atomicAdd(total, sum);
        }
    } // namespace

    void detail::launchSharpnessCount(const uint8_t* rgb, int w, int h, SharpnessAlgorithm algorithm,
                                      unsigned long long* count, cudaStream_t stream) {
        const auto pixels = static_cast<unsigned long long>(w - 2) * (h - 2);
        const int blocks = static_cast<int>(std::min<unsigned long long>((pixels + BLOCK_SIZE - 1) / BLOCK_SIZE, 4096));
        countEdges<<<blocks, BLOCK_SIZE, 0, stream>>>(rgb, w, h, algorithm, count);
        LFS_CUDA_LAUNCH_CHECK(stream, "io.video.sharpness_count");
    }
} // namespace lfs::io

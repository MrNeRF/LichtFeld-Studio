/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#include "FreeSpace.h"
#include "core/cuda_error.hpp"

namespace gsplat_lfs {
    namespace {
        constexpr uint32_t kThreads = 256;

        __device__ __forceinline__ float2 block_sum(float2 v) {
            __shared__ float2 warp_sums[kThreads / 32];
            for (int offset = 16; offset > 0; offset /= 2) {
                v.x += __shfl_down_sync(0xffffffffu, v.x, offset);
                v.y += __shfl_down_sync(0xffffffffu, v.y, offset);
            }
            const uint32_t lane = threadIdx.x % 32;
            const uint32_t warp = threadIdx.x / 32;
            if (lane == 0)
                warp_sums[warp] = v;
            __syncthreads();
            v = threadIdx.x < kThreads / 32 ? warp_sums[threadIdx.x] : make_float2(0.f, 0.f);
            if (warp == 0) {
                for (int offset = 16; offset > 0; offset /= 2) {
                    v.x += __shfl_down_sync(0xffffffffu, v.x, offset);
                    v.y += __shfl_down_sync(0xffffffffu, v.y, offset);
                }
            }
            return v;
        }

        // Per block: (sum of 1 - T_front, pixels with scan depth).
        __global__ void partial_kernel(const FreeSpaceRender f, const uint32_t W, const uint32_t H, float2* partials) {
            const uint32_t i = blockIdx.x * kThreads + threadIdx.x;
            float2 v = make_float2(0.f, 0.f);
            if (i < W * H && f.scan_depth(0, i % W, i / W, W, H) > 0.f)
                v = make_float2(1.f - f.T_front[i], 1.f);
            v = block_sum(v);
            if (threadIdx.x == 0)
                partials[1 + blockIdx.x] = v;
        }

        __global__ void reduce_kernel(float2* partials, const uint32_t blocks, const float weight, float* loss) {
            float2 v = make_float2(0.f, 0.f);
            for (uint32_t b = threadIdx.x; b < blocks; b += kThreads) {
                v.x += partials[1 + b].x;
                v.y += partials[1 + b].y;
            }
            v = block_sum(v);
            if (threadIdx.x == 0) {
                partials[0] = v;
                loss[0] = v.y > 0.f ? weight * v.x / v.y : 0.f;
            }
        }

        __global__ void grad_kernel(const FreeSpaceRender f, const uint32_t W, const uint32_t H,
                                    const float2* partials, const float weight, float* v_T_front) {
            const uint32_t i = blockIdx.x * kThreads + threadIdx.x;
            if (i >= W * H)
                return;
            const float count = partials[0].y;
            v_T_front[i] = count > 0.f && f.scan_depth(0, i % W, i / W, W, H) > 0.f ? -weight / count : 0.f;
        }
    } // namespace

    uint32_t free_space_partial_count(const uint32_t pixels) {
        return 2 * (1 + (pixels + kThreads - 1) / kThreads);
    }

    void free_space_loss(const FreeSpaceRender& free_space, const uint32_t width, const uint32_t height,
                         const float weight, float* v_T_front, float* loss, float* partials,
                         cudaStream_t stream) {
        const uint32_t pixels = width * height;
        if (!pixels || !free_space.enabled())
            return;
        const uint32_t blocks = (pixels + kThreads - 1) / kThreads;
        auto* sums = reinterpret_cast<float2*>(partials);
        partial_kernel<<<blocks, kThreads, 0, stream>>>(free_space, width, height, sums);
        LFS_CUDA_LAUNCH_CHECK(stream, "gut.free_space.partials");
        reduce_kernel<<<1, kThreads, 0, stream>>>(sums, blocks, weight, loss);
        LFS_CUDA_LAUNCH_CHECK(stream, "gut.free_space.reduce");
        grad_kernel<<<blocks, kThreads, 0, stream>>>(free_space, width, height, sums, weight, v_T_front);
        LFS_CUDA_LAUNCH_CHECK(stream, "gut.free_space.grad");
    }
} // namespace gsplat_lfs

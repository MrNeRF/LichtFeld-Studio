/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "core/cuda_error.hpp"
#include "lfs/core/memory_ops.cuh"
#include "lfs/core/warp_reduce.cuh"
#include "lfs/kernels/bilateral_grid.cuh"
#include <cuda_runtime.h>

#include "kernel_stream.hpp"

namespace lfs::training::kernels {

    using namespace lfs::core;

    /**
     * @brief Stage 1: Each block reduces its portion to a partial sum
     *
     * Optimizations:
     * - Grid-stride loop for load balancing
     * - Warp-level reduction (no atomics!)
     * - Streaming cache hints for single-use data
     * - Each block writes one partial sum
     */
    __global__ void bilateral_grid_tv_forward_stage1_kernel(
        const float* __restrict__ grids, // [N,12,L,H,W]
        float* __restrict__ partial_sums,
        int N, int L, int H, int W) {

        int tid = blockIdx.x * blockDim.x + threadIdx.x;
        int stride = gridDim.x * blockDim.x;
        int total = N * L * H * W;

        float local_sum = 0.0f;

        // Grid-stride loop: each thread may process multiple elements
        for (int idx = tid; idx < total; idx += stride) {
            // Decode position
            int tmp = idx;
            int wi = tmp % W;
            tmp /= W;
            int hi = tmp % H;
            tmp /= H;
            int li = tmp % L;
            tmp /= L;
            int ni = tmp;

            // Process all 12 channels with unrolling
#pragma unroll 12
            for (int ci = 0; ci < 12; ci++) {
                int base = (ni * 12 + ci) * L * H * W;
                int cell_idx = base + (li * H + hi) * W + wi;

                // Use streaming cache hint - grid values accessed only once
                float val = load_cs(&grids[cell_idx]);

                // X-direction
                if (wi > 0) {
                    float val0 = load_cs(&grids[cell_idx - 1]);
                    float diff = val - val0;
                    local_sum += diff * diff / (L * H * (W - 1));
                }

                // Y-direction
                if (hi > 0) {
                    float val0 = load_cs(&grids[cell_idx - W]);
                    float diff = val - val0;
                    local_sum += diff * diff / (L * (H - 1) * W);
                }

                // Z-direction
                if (li > 0) {
                    float val0 = load_cs(&grids[cell_idx - W * H]);
                    float diff = val - val0;
                    local_sum += diff * diff / ((L - 1) * H * W);
                }
            }
        }

        local_sum /= (12 * N);

        // Block-level reduction using warp shuffles (NO ATOMIC!)
        local_sum = warp_ops::block_reduce_sum(local_sum);

        // Each block writes its partial sum (no contention!)
        if (threadIdx.x == 0) {
            partial_sums[blockIdx.x] = local_sum;
        }
    }

    /**
     * @brief Stage 2: Single block aggregates all partial sums
     *
     * Deterministic and fast - no atomic contention!
     */
    __global__ void bilateral_grid_tv_forward_stage2_kernel(
        const float* __restrict__ partial_sums,
        float* __restrict__ tv_loss,
        int num_partials) {

        // Single block processes all partial sums
        float thread_sum = 0.0f;
        for (int i = threadIdx.x; i < num_partials; i += blockDim.x) {
            thread_sum += partial_sums[i];
        }

        // Final block reduction
        float result = warp_ops::block_reduce_sum(thread_sum);

        if (threadIdx.x == 0) {
            *tv_loss = result; // Direct write, no atomic!
        }
    }

    /**
     * @brief Backward pass - compute gradients of TV loss
     *
     * Optimizations:
     * - 3D grid for better thread organization
     * - Read-only cache hints for grid values
     * - Direct writes (no atomics needed)
     */
    __global__ void bilateral_grid_tv_backward_kernel(
        const float* __restrict__ grids, // [N,12,L,H,W]
        const float grad_output,         // scalar gradient dL/d(tv_loss)
        float* __restrict__ grad_grids,  // [N,12,L,H,W]
        int N, int L, int H, int W) {

        int wi = blockIdx.x * blockDim.x + threadIdx.x;
        int hi = blockIdx.y * blockDim.y + threadIdx.y;
        int idx = blockIdx.z * blockDim.z + threadIdx.z;
        if (wi >= W || hi >= H || idx >= (N * L))
            return;

        int li = idx % L;
        idx /= L;
        int ni = idx;

        float s = grad_output / (6 * N);
        float sx = s / (float)(L * H * (W - 1));
        float sy = s / (float)(L * (H - 1) * W);
        float sz = s / (float)((L - 1) * H * W);

        for (int ci = 0; ci < 12; ci++) {
            int cell_idx = (((ni * 12 + ci) * L + li) * H + hi) * W + wi;

            float half_grad = 0.0f;
            // Use read-only cache for grid (shared across threads)
            float val = load_ro(&grids[cell_idx]);

            if (wi > 0) {
                float val0 = load_ro(&grids[cell_idx - 1]);
                half_grad += (val - val0) * sx;
            }
            if (wi < W - 1) {
                float val0 = load_ro(&grids[cell_idx + 1]);
                half_grad += (val - val0) * sx;
            }
            if (hi > 0) {
                float val0 = load_ro(&grids[cell_idx - W]);
                half_grad += (val - val0) * sy;
            }
            if (hi < H - 1) {
                float val0 = load_ro(&grids[cell_idx + W]);
                half_grad += (val - val0) * sy;
            }
            if (li > 0) {
                float val0 = load_ro(&grids[cell_idx - W * H]);
                half_grad += (val - val0) * sz;
            }
            if (li < L - 1) {
                float val0 = load_ro(&grids[cell_idx + W * H]);
                half_grad += (val - val0) * sz;
            }

            grad_grids[cell_idx] = half_grad;
        }
    }

    void launch_bilateral_grid_tv_forward(
        const float* grids,
        float* tv_loss,
        float* temp_buffer,
        int N, int L, int H, int W,
        cudaStream_t stream) {
        stream = resolve_stream(stream);

        const int threads = 256;
        const int total = N * L * H * W;
        const int num_blocks = min((total + threads - 1) / threads, 2048);

        // Stage 1: Each block reduces to a partial sum (NO ATOMICS!)
        bilateral_grid_tv_forward_stage1_kernel<<<num_blocks, threads, 0, stream>>>(
            grids, temp_buffer, N, L, H, W);
        LFS_CUDA_LAUNCH_CHECK(stream, "training.bilateral.tv_forward_stage1");

        // Stage 2: Single block aggregates partial sums (DETERMINISTIC!)
        bilateral_grid_tv_forward_stage2_kernel<<<1, threads, 0, stream>>>(
            temp_buffer, tv_loss, num_blocks);
        LFS_CUDA_LAUNCH_CHECK(stream, "training.bilateral.tv_forward_stage2");
    }

    void launch_bilateral_grid_tv_backward(
        const float* grids,
        float grad_output,
        float* grad_grids,
        int N, int L, int H, int W,
        cudaStream_t stream) {
        stream = resolve_stream(stream);

        // 3D grid for better thread organization
        dim3 block(4, 4, 4);
        dim3 grid_dim(
            (W + block.x - 1) / block.x,
            (H + block.y - 1) / block.y,
            (N * L + block.z - 1) / block.z);

        bilateral_grid_tv_backward_kernel<<<grid_dim, block, 0, stream>>>(
            grids, grad_output, grad_grids, N, L, H, W);
        LFS_CUDA_LAUNCH_CHECK(stream, "training.bilateral.tv_backward");
    }

} // namespace lfs::training::kernels

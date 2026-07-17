/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "core/cuda_error.hpp"
#include "lfs/core/memory_ops.cuh"
#include "lfs/kernels/bilateral_grid.cuh"
#include <cuda_runtime.h>

#include "kernel_stream.hpp"

namespace lfs::training::kernels {

    using namespace lfs::core;

    constexpr int BLOCK_SIZE = 256;

    // HWC layout forward kernel
    __global__ void bilateral_grid_slice_forward_kernel(
        const float* __restrict__ grid,
        const float* __restrict__ rgb,
        float* __restrict__ output,
        const int L, const int H, const int W,
        const int h, const int w) {

        const int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx >= h * w)
            return;

        const int hi = idx / w;
        const int wi = idx % w;
        const int rgb_idx = idx * 3;

        const RGB color = load_rgb_cs(&rgb[rgb_idx]);
        const float sr = isfinite(color.r) ? color.r : 0.5f;
        const float sg = isfinite(color.g) ? color.g : 0.5f;
        const float sb = isfinite(color.b) ? color.b : 0.5f;
        float dr = 0.0f, dg = 0.0f, db = 0.0f;

        const float x = w > 1 ? static_cast<float>(wi) / (w - 1) * (W - 1) : 0.0f;
        const float y = h > 1 ? static_cast<float>(hi) / (h - 1) * (H - 1) : 0.0f;
        const float guidance = fminf(1.0f, fmaxf(0.0f, kC2G_r * sr + kC2G_g * sg + kC2G_b * sb));
        const float z = guidance * (L - 1);

        const int x0 = floorf(x), y0 = floorf(y);
        int z0 = floorf(z);
        const int x1 = min(x0 + 1, W - 1);
        const int y1 = min(y0 + 1, H - 1);
        int z1 = z0 + 1;
        z0 = min(max(z0, 0), L - 1);
        z1 = min(max(z1, 0), L - 1);

        const float fx = x - x0, fy = y - y0, fz = z - z0;

#pragma unroll
        for (int ci = 0; ci < 12; ++ci) {
            const int base = ci * L * H * W;
            const float v000 = load_ro(&grid[base + (z0 * H + y0) * W + x0]);
            const float v001 = load_ro(&grid[base + (z0 * H + y0) * W + x1]);
            const float v010 = load_ro(&grid[base + (z0 * H + y1) * W + x0]);
            const float v011 = load_ro(&grid[base + (z0 * H + y1) * W + x1]);
            const float v100 = load_ro(&grid[base + (z1 * H + y0) * W + x0]);
            const float v101 = load_ro(&grid[base + (z1 * H + y0) * W + x1]);
            const float v110 = load_ro(&grid[base + (z1 * H + y1) * W + x0]);
            const float v111 = load_ro(&grid[base + (z1 * H + y1) * W + x1]);

            const float c00 = v000 * (1 - fx) + v001 * fx;
            const float c01 = v010 * (1 - fx) + v011 * fx;
            const float c10 = v100 * (1 - fx) + v101 * fx;
            const float c11 = v110 * (1 - fx) + v111 * fx;
            const float c0 = c00 * (1 - fy) + c01 * fy;
            const float c1 = c10 * (1 - fy) + c11 * fy;
            const float val = c0 * (1 - fz) + c1 * fz;

            const int si = ci % 4, di = ci / 4;
            (di == 0 ? dr : di == 1 ? dg
                                    : db) += val *
                                             (si == 0 ? sr : si == 1 ? sg
                                                         : si == 2   ? sb
                                                                     : 1.0f);
        }

        output[rgb_idx + 0] = isfinite(dr) ? dr : 0.5f;
        output[rgb_idx + 1] = isfinite(dg) ? dg : 0.5f;
        output[rgb_idx + 2] = isfinite(db) ? db : 0.5f;
    }

    void launch_bilateral_grid_slice_forward(
        const float* grid, const float* rgb, float* output,
        int L, int H, int W, int h, int w,
        cudaStream_t stream) {
        stream = resolve_stream(stream);

        const int blocks = (h * w + BLOCK_SIZE - 1) / BLOCK_SIZE;
        bilateral_grid_slice_forward_kernel<<<blocks, BLOCK_SIZE, 0, stream>>>(
            grid, rgb, output, L, H, W, h, w);
        LFS_CUDA_LAUNCH_CHECK(stream, "training.bilateral.slice_forward");
    }

    // CHW layout forward kernel
    __global__ void bilateral_grid_slice_forward_chw_kernel(
        const float* __restrict__ grid,
        const float* __restrict__ rgb,
        float* __restrict__ output,
        const int L, const int H, const int W,
        const int h, const int w) {

        const int idx = blockIdx.x * blockDim.x + threadIdx.x;
        const int hw = h * w;
        if (idx >= hw)
            return;

        const int hi = idx / w;
        const int wi = idx % w;

        const float sr = isfinite(rgb[0 * hw + idx]) ? rgb[0 * hw + idx] : 0.5f;
        const float sg = isfinite(rgb[1 * hw + idx]) ? rgb[1 * hw + idx] : 0.5f;
        const float sb = isfinite(rgb[2 * hw + idx]) ? rgb[2 * hw + idx] : 0.5f;
        float dr = 0.0f, dg = 0.0f, db = 0.0f;

        const float x = w > 1 ? static_cast<float>(wi) / (w - 1) * (W - 1) : 0.0f;
        const float y = h > 1 ? static_cast<float>(hi) / (h - 1) * (H - 1) : 0.0f;
        const float guidance = fminf(1.0f, fmaxf(0.0f, kC2G_r * sr + kC2G_g * sg + kC2G_b * sb));
        const float z = guidance * (L - 1);

        const int x0 = floorf(x), y0 = floorf(y);
        int z0 = floorf(z);
        const int x1 = min(x0 + 1, W - 1);
        const int y1 = min(y0 + 1, H - 1);
        int z1 = z0 + 1;
        z0 = min(max(z0, 0), L - 1);
        z1 = min(max(z1, 0), L - 1);

        const float fx = x - x0, fy = y - y0, fz = z - z0;

#pragma unroll
        for (int ci = 0; ci < 12; ++ci) {
            const int base = ci * L * H * W;
            const float v000 = load_ro(&grid[base + (z0 * H + y0) * W + x0]);
            const float v001 = load_ro(&grid[base + (z0 * H + y0) * W + x1]);
            const float v010 = load_ro(&grid[base + (z0 * H + y1) * W + x0]);
            const float v011 = load_ro(&grid[base + (z0 * H + y1) * W + x1]);
            const float v100 = load_ro(&grid[base + (z1 * H + y0) * W + x0]);
            const float v101 = load_ro(&grid[base + (z1 * H + y0) * W + x1]);
            const float v110 = load_ro(&grid[base + (z1 * H + y1) * W + x0]);
            const float v111 = load_ro(&grid[base + (z1 * H + y1) * W + x1]);

            const float c00 = v000 * (1 - fx) + v001 * fx;
            const float c01 = v010 * (1 - fx) + v011 * fx;
            const float c10 = v100 * (1 - fx) + v101 * fx;
            const float c11 = v110 * (1 - fx) + v111 * fx;
            const float c0 = c00 * (1 - fy) + c01 * fy;
            const float c1 = c10 * (1 - fy) + c11 * fy;
            const float val = c0 * (1 - fz) + c1 * fz;

            const int si = ci % 4, di = ci / 4;
            (di == 0 ? dr : di == 1 ? dg
                                    : db) += val *
                                             (si == 0 ? sr : si == 1 ? sg
                                                         : si == 2   ? sb
                                                                     : 1.0f);
        }

        output[0 * hw + idx] = isfinite(dr) ? dr : 0.5f;
        output[1 * hw + idx] = isfinite(dg) ? dg : 0.5f;
        output[2 * hw + idx] = isfinite(db) ? db : 0.5f;
    }

    void launch_bilateral_grid_slice_forward_chw(
        const float* grid, const float* rgb, float* output,
        int L, int H, int W, int h, int w,
        cudaStream_t stream) {
        stream = resolve_stream(stream);

        const int blocks = (h * w + BLOCK_SIZE - 1) / BLOCK_SIZE;
        bilateral_grid_slice_forward_chw_kernel<<<blocks, BLOCK_SIZE, 0, stream>>>(
            grid, rgb, output, L, H, W, h, w);
        LFS_CUDA_LAUNCH_CHECK(stream, "training.bilateral.slice_forward_chw");
    }

} // namespace lfs::training::kernels

/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "lfs/core/memory_ops.cuh"
#include "lfs/kernels/bilateral_grid.cuh"
#include <cuda_runtime.h>

#include "kernel_stream.hpp"

namespace lfs::training::kernels {

    using namespace lfs::core;

    constexpr int BLOCK_SIZE = 256;

    // HWC layout backward kernel
    __global__ void bilateral_grid_slice_backward_kernel(
        const float* __restrict__ grid,
        const float* __restrict__ rgb,
        const float* __restrict__ grad_output,
        float* __restrict__ grad_grid,
        float* __restrict__ grad_rgb,
        const int L, const int H, const int W,
        const int h, const int w) {

        const int wi = threadIdx.x * ((w + blockDim.x - 1) / blockDim.x) + blockIdx.x;
        const int hi = threadIdx.y * ((h + blockDim.y - 1) / blockDim.y) + blockIdx.y;
        if (wi >= w || hi >= h)
            return;

        const int pixel_idx = hi * w + wi;
        const int rgb_offset = pixel_idx * 3;

        const RGB rgb_val = load_rgb_cs(&rgb[rgb_offset]);
        const float sr = isfinite(rgb_val.r) ? rgb_val.r : 0.5f;
        const float sg = isfinite(rgb_val.g) ? rgb_val.g : 0.5f;
        const float sb = isfinite(rgb_val.b) ? rgb_val.b : 0.5f;

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

        const RGB grad = load_rgb_cs(&grad_output[rgb_offset]);
        const float dr = isfinite(grad.r) ? grad.r : 0.0f;
        const float dg = isfinite(grad.g) ? grad.g : 0.0f;
        const float db = isfinite(grad.b) ? grad.b : 0.0f;

        float vr = 0.0f, vg = 0.0f, vb = 0.0f;
        const float weights[8] = {
            (1 - fx) * (1 - fy) * (1 - fz), fx * (1 - fy) * (1 - fz),
            (1 - fx) * fy * (1 - fz), fx * fy * (1 - fz),
            (1 - fx) * (1 - fy) * fz, fx * (1 - fy) * fz,
            (1 - fx) * fy * fz, fx * fy * fz};

        float gz_grad = 0.0f;

#pragma unroll
        for (int corner = 0; corner < 8; ++corner) {
            const int xi = (corner & 1) ? x1 : x0;
            const int yi = (corner & 2) ? y1 : y0;
            const int zi = (corner & 4) ? z1 : z0;
            const float wt = weights[corner];
            const float dfdz = ((corner & 1) ? fx : (1 - fx)) *
                               ((corner & 2) ? fy : (1 - fy)) *
                               ((corner & 4) ? 1.0f : -1.0f);

            float trilerp = 0.0f;
#pragma unroll
            for (int ci = 0; ci < 12; ++ci) {
                const int grid_idx = (ci * L + zi) * H * W + yi * W + xi;
                const int si = ci % 4, di = ci / 4;
                const float r_coeff = (si == 0 ? sr : si == 1 ? sg
                                                  : si == 2   ? sb
                                                              : 1.0f);
                const float gout = (di == 0 ? dr : di == 1 ? dg
                                                           : db);
                const float v = load_ro(&grid[grid_idx]);

                if (si < 3)
                    (si == 0 ? vr : si == 1 ? vg
                                            : vb) += v * wt * gout;

                const float grad_weight = r_coeff * gout;
                trilerp += v * grad_weight;
                atomicAdd(grad_grid + grid_idx, wt * grad_weight);
            }
            gz_grad += dfdz * (L - 1) * trilerp;
        }

        gz_grad *= static_cast<float>(z0 != z && z1 != z);
        grad_rgb[rgb_offset + 0] = vr + kC2G_r * gz_grad;
        grad_rgb[rgb_offset + 1] = vg + kC2G_g * gz_grad;
        grad_rgb[rgb_offset + 2] = vb + kC2G_b * gz_grad;
    }

    void launch_bilateral_grid_slice_backward(
        const float* grid, const float* rgb, const float* grad_output,
        float* grad_grid, float* grad_rgb,
        int L, int H, int W, int h, int w,
        cudaStream_t stream) {
        stream = resolve_stream(stream);

        const dim3 block(16, 16);
        const dim3 grid_dim((w + block.x - 1) / block.x, (h + block.y - 1) / block.y);
        bilateral_grid_slice_backward_kernel<<<grid_dim, block, 0, stream>>>(
            grid, rgb, grad_output, grad_grid, grad_rgb, L, H, W, h, w);
    }

    // CHW layout backward kernel
    __global__ void bilateral_grid_slice_backward_chw_kernel(
        const float* __restrict__ grid,
        const float* __restrict__ rgb,
        const float* __restrict__ grad_output,
        float* __restrict__ grad_grid,
        float* __restrict__ grad_rgb,
        const int L, const int H, const int W,
        const int h, const int w) {

        const int idx = blockIdx.x * blockDim.x + threadIdx.x;
        const int hw = h * w;
        if (idx >= hw)
            return;

        const int hi = idx / w;
        const int wi = idx % w;

        float sr = rgb[0 * hw + idx];
        float sg = rgb[1 * hw + idx];
        float sb = rgb[2 * hw + idx];
        sr = isfinite(sr) ? sr : 0.5f;
        sg = isfinite(sg) ? sg : 0.5f;
        sb = isfinite(sb) ? sb : 0.5f;

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

        float dr = grad_output[0 * hw + idx];
        float dg = grad_output[1 * hw + idx];
        float db = grad_output[2 * hw + idx];
        dr = isfinite(dr) ? dr : 0.0f;
        dg = isfinite(dg) ? dg : 0.0f;
        db = isfinite(db) ? db : 0.0f;

        float vr = 0.0f, vg = 0.0f, vb = 0.0f;
        const float weights[8] = {
            (1 - fx) * (1 - fy) * (1 - fz), fx * (1 - fy) * (1 - fz),
            (1 - fx) * fy * (1 - fz), fx * fy * (1 - fz),
            (1 - fx) * (1 - fy) * fz, fx * (1 - fy) * fz,
            (1 - fx) * fy * fz, fx * fy * fz};
        const int cx[8] = {x0, x1, x0, x1, x0, x1, x0, x1};
        const int cy[8] = {y0, y0, y1, y1, y0, y0, y1, y1};
        const int cz[8] = {z0, z0, z0, z0, z1, z1, z1, z1};
        const float dwdz[8] = {
            -(1 - fx) * (1 - fy), -fx * (1 - fy),
            -(1 - fx) * fy, -fx * fy,
            (1 - fx) * (1 - fy), fx * (1 - fy),
            (1 - fx) * fy, fx * fy};

        float gz_grad = 0.0f;

#pragma unroll
        for (int ci = 0; ci < 12; ++ci) {
            const int si = ci % 4, di = ci / 4;
            const float r_coeff = (si == 0 ? sr : si == 1 ? sg
                                              : si == 2   ? sb
                                                          : 1.0f);
            const float gout = (di == 0 ? dr : di == 1 ? dg
                                                       : db);
            const float grad_base = r_coeff * gout;

#pragma unroll
            for (int corner = 0; corner < 8; ++corner) {
                const int grid_idx = (ci * L + cz[corner]) * H * W + cy[corner] * W + cx[corner];
                const float wt = weights[corner];
                const float v = load_ro(&grid[grid_idx]);

                if (si < 3) {
                    const float contrib = v * wt * gout;
                    if (si == 0)
                        vr += contrib;
                    else if (si == 1)
                        vg += contrib;
                    else
                        vb += contrib;
                }

                atomicAdd(grad_grid + grid_idx, wt * grad_base);
                gz_grad += dwdz[corner] * (L - 1) * v * grad_base;
            }
        }

        gz_grad *= static_cast<float>(z0 != z && z1 != z);
        grad_rgb[0 * hw + idx] = vr + kC2G_r * gz_grad;
        grad_rgb[1 * hw + idx] = vg + kC2G_g * gz_grad;
        grad_rgb[2 * hw + idx] = vb + kC2G_b * gz_grad;
    }

    void launch_bilateral_grid_slice_backward_chw(
        const float* grid, const float* rgb, const float* grad_output,
        float* grad_grid, float* grad_rgb,
        int L, int H, int W, int h, int w,
        cudaStream_t stream) {
        stream = resolve_stream(stream);

        const int blocks = (h * w + BLOCK_SIZE - 1) / BLOCK_SIZE;
        bilateral_grid_slice_backward_chw_kernel<<<blocks, BLOCK_SIZE, 0, stream>>>(
            grid, rgb, grad_output, grad_grid, grad_rgb, L, H, W, h, w);
    }

    // Adam update kernel
    __global__ void bilateral_grid_adam_update_kernel(
        float* __restrict__ grid,
        float* __restrict__ exp_avg,
        float* __restrict__ exp_avg_sq,
        const float* __restrict__ grad_grid,
        const int num_elements,
        const float lr, const float beta1, const float beta2,
        const float bias_corr1_rcp, const float bias_corr2_sqrt_rcp, const float eps) {

        const int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx >= num_elements)
            return;

        const float g = grad_grid[idx];
        float m = exp_avg[idx];
        float v = exp_avg_sq[idx];

        m = beta1 * m + (1.0f - beta1) * g;
        v = beta2 * v + (1.0f - beta2) * g * g;

        exp_avg[idx] = m;
        exp_avg_sq[idx] = v;

        const float m_hat = m * bias_corr1_rcp;
        const float v_hat = v * bias_corr2_sqrt_rcp * bias_corr2_sqrt_rcp;

        grid[idx] -= lr * m_hat / (sqrtf(v_hat) + eps);
    }

    void launch_bilateral_grid_adam_update(
        float* grid, float* exp_avg, float* exp_avg_sq, const float* grad_grid,
        int num_elements, float lr, float beta1, float beta2,
        float bias_corr1_rcp, float bias_corr2_sqrt_rcp, float eps,
        cudaStream_t stream) {
        stream = resolve_stream(stream);

        const int blocks = (num_elements + BLOCK_SIZE - 1) / BLOCK_SIZE;
        bilateral_grid_adam_update_kernel<<<blocks, BLOCK_SIZE, 0, stream>>>(
            grid, exp_avg, exp_avg_sq, grad_grid, num_elements,
            lr, beta1, beta2, bias_corr1_rcp, bias_corr2_sqrt_rcp, eps);
    }

    // Gradient accumulation kernel
    __global__ void bilateral_grid_accumulate_grad_kernel(
        float* __restrict__ dst,
        const float* __restrict__ src,
        const int num_elements) {

        const int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx >= num_elements)
            return;
        dst[idx] += src[idx];
    }

    void launch_bilateral_grid_accumulate_grad(
        float* dst, const float* src, int num_elements,
        cudaStream_t stream) {
        stream = resolve_stream(stream);

        const int blocks = (num_elements + BLOCK_SIZE - 1) / BLOCK_SIZE;
        bilateral_grid_accumulate_grad_kernel<<<blocks, BLOCK_SIZE, 0, stream>>>(
            dst, src, num_elements);
    }

    // Initialize 3x4 affine identity: channels 0,5,10 = 1.0 (diagonal)
    __global__ void bilateral_grid_init_identity_kernel(
        float* __restrict__ grids,
        const int num_cells,
        const int num_elements) {

        const int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx >= num_elements)
            return;

        const int ci = (idx / num_cells) % 12;
        const float val = (ci == 0 || ci == 5 || ci == 10) ? 1.0f : 0.0f;
        grids[idx] = val;
    }

    void launch_bilateral_grid_init_identity(
        float* grids, int N, int L, int H, int W,
        cudaStream_t stream) {
        stream = resolve_stream(stream);

        const int num_cells = L * H * W;
        const int num_elements = N * 12 * num_cells;
        const int threads = BLOCK_SIZE;
        const int blocks = (num_elements + threads - 1) / threads;

        bilateral_grid_init_identity_kernel<<<blocks, threads, 0, stream>>>(
            grids, num_cells, num_elements);
    }

} // namespace lfs::training::kernels

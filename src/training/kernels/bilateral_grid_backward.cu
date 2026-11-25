/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "core/cuda_error.hpp"
#include "lfs/core/memory_ops.cuh"
#include "lfs/kernels/bilateral_grid.cuh"
#include "ppisp_math_bwd.cuh"
#include <algorithm>
#include <cassert>
#include <cuda_runtime.h>

#include "kernel_stream.hpp"

namespace lfs::training::kernels {

    using namespace lfs::core;

    constexpr int BLOCK_SIZE = 256;

    // HWC layout backward kernel
    __device__ __forceinline__ float load_grid_offset(const float* __restrict__ shared_offset, int ci) {
        return shared_offset ? shared_offset[ci] : 0.0f;
    }

    __global__ void bilateral_grid_slice_backward_reference_kernel(
        const float* __restrict__ grid,
        const float* __restrict__ rgb,
        const float* __restrict__ grad_output,
        float* __restrict__ grad_grid,
        float* __restrict__ grad_rgb,
        const int L, const int H, const int W,
        const int h, const int w,
        const float* __restrict__ shared_offset) {

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
                const float v = load_ro(&grid[grid_idx]) + load_grid_offset(shared_offset, ci);

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

    void launch_bilateral_grid_slice_backward_reference(
        const float* grid, const float* rgb, const float* grad_output,
        float* grad_grid, float* grad_rgb,
        int L, int H, int W, int h, int w,
        const float* shared_offset,
        cudaStream_t stream) {
        assert(L > 0 && H > 0 && W > 0 && h > 0 && w > 0);
        stream = resolve_stream(stream);

        const dim3 block(16, 16);
        const dim3 grid_dim((w + block.x - 1) / block.x, (h + block.y - 1) / block.y);
        bilateral_grid_slice_backward_reference_kernel<<<grid_dim, block, 0, stream>>>(
            grid, rgb, grad_output, grad_grid, grad_rgb, L, H, W, h, w, shared_offset);
        LFS_CUDA_LAUNCH_CHECK(stream, "training.bilateral.slice_backward_reference");
    }

    // CHW layout backward kernel
    __global__ void bilateral_grid_slice_backward_chw_reference_kernel(
        const float* __restrict__ grid,
        const float* __restrict__ rgb,
        const float* __restrict__ grad_output,
        float* __restrict__ grad_grid,
        float* __restrict__ grad_rgb,
        const int L, const int H, const int W,
        const int h, const int w,
        const float* __restrict__ shared_offset) {

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
                const float v = load_ro(&grid[grid_idx]) + load_grid_offset(shared_offset, ci);

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

    void launch_bilateral_grid_slice_backward_chw_reference(
        const float* grid, const float* rgb, const float* grad_output,
        float* grad_grid, float* grad_rgb,
        int L, int H, int W, int h, int w,
        const float* shared_offset,
        cudaStream_t stream) {
        assert(L > 0 && H > 0 && W > 0 && h > 0 && w > 0);
        stream = resolve_stream(stream);

        const int blocks = (h * w + BLOCK_SIZE - 1) / BLOCK_SIZE;
        bilateral_grid_slice_backward_chw_reference_kernel<<<blocks, BLOCK_SIZE, 0, stream>>>(
            grid, rgb, grad_output, grad_grid, grad_rgb, L, H, W, h, w, shared_offset);
        LFS_CUDA_LAUNCH_CHECK(stream, "training.bilateral.slice_backward_chw_reference");
    }

    __device__ __forceinline__ void sample_exposure_chroma_grid_bwd(
        const float* __restrict__ grid,
        const int L, const int H, const int W,
        const int x0, const int y0, const int z0,
        const int x1, const int y1, const int z1,
        const float fx, const float fy, const float fz,
        const float* __restrict__ shared_offset,
        float* __restrict__ sampled) {
#pragma unroll
        for (int ci = 0; ci < 9; ++ci) {
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
            sampled[ci] = c0 * (1 - fz) + c1 * fz + load_grid_offset(shared_offset, ci);
        }
    }

    __device__ __forceinline__ void exposure_chroma_param_vjp(
        const float sr, const float sg, const float sb,
        const float* __restrict__ sampled,
        const float dr, const float dg, const float db,
        float3& grad_rgb_in,
        float* __restrict__ dL_dsampled) {
        ColorPPISPParams params;
        params.b = make_float2(sampled[1], sampled[2]);
        params.r = make_float2(sampled[3], sampled[4]);
        params.g = make_float2(sampled[5], sampled[6]);
        params.n = make_float2(sampled[7], sampled[8]);

        const float3 rgb_in = make_float3(sr, sg, sb);
        float3 rgb_exp;
        ppisp_apply_exposure(rgb_in, sampled[0], rgb_exp);

        const float3 grad_out = make_float3(dr, dg, db);
        ColorPPISPParams grad_color;
        grad_color.b = make_float2(0.0f, 0.0f);
        grad_color.r = make_float2(0.0f, 0.0f);
        grad_color.g = make_float2(0.0f, 0.0f);
        grad_color.n = make_float2(0.0f, 0.0f);

        float3 grad_rgb_exp;
        ppisp_apply_color_correction_bwd(rgb_exp, &params, grad_out, grad_rgb_exp, &grad_color);

        float grad_ev = 0.0f;
        ppisp_apply_exposure_bwd(rgb_in, sampled[0], grad_rgb_exp, grad_rgb_in, grad_ev);

        dL_dsampled[0] = grad_ev;
        dL_dsampled[1] = grad_color.b.x;
        dL_dsampled[2] = grad_color.b.y;
        dL_dsampled[3] = grad_color.r.x;
        dL_dsampled[4] = grad_color.r.y;
        dL_dsampled[5] = grad_color.g.x;
        dL_dsampled[6] = grad_color.g.y;
        dL_dsampled[7] = grad_color.n.x;
        dL_dsampled[8] = grad_color.n.y;
    }

    __global__ void bilateral_grid_slice_backward_exposure_chroma_reference_kernel(
        const float* __restrict__ grid,
        const float* __restrict__ rgb,
        const float* __restrict__ grad_output,
        float* __restrict__ grad_grid,
        float* __restrict__ grad_rgb,
        const int L, const int H, const int W,
        const int h, const int w,
        const float* __restrict__ shared_offset) {

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

        float sampled[9];
        sample_exposure_chroma_grid_bwd(grid, L, H, W, x0, y0, z0, x1, y1, z1, fx, fy, fz, shared_offset, sampled);

        float3 grad_rgb_in;
        float dL_dsampled[9];
        exposure_chroma_param_vjp(sr, sg, sb, sampled, dr, dg, db, grad_rgb_in, dL_dsampled);

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
            for (int ci = 0; ci < 9; ++ci) {
                const int grid_idx = (ci * L + zi) * H * W + yi * W + xi;
                const float v = load_ro(&grid[grid_idx]);
                trilerp += v * dL_dsampled[ci];
                atomicAdd(grad_grid + grid_idx, wt * dL_dsampled[ci]);
            }
            gz_grad += dfdz * (L - 1) * trilerp;
        }

        gz_grad *= static_cast<float>(z0 != z && z1 != z);
        grad_rgb[rgb_offset + 0] = grad_rgb_in.x + kC2G_r * gz_grad;
        grad_rgb[rgb_offset + 1] = grad_rgb_in.y + kC2G_g * gz_grad;
        grad_rgb[rgb_offset + 2] = grad_rgb_in.z + kC2G_b * gz_grad;
    }

    void launch_bilateral_grid_slice_backward_exposure_chroma_reference(
        const float* grid, const float* rgb, const float* grad_output,
        float* grad_grid, float* grad_rgb,
        int L, int H, int W, int h, int w,
        const float* shared_offset,
        cudaStream_t stream) {
        assert(L > 0 && H > 0 && W > 0 && h > 0 && w > 0);
        stream = resolve_stream(stream);

        const dim3 block(16, 16);
        const dim3 grid_dim((w + block.x - 1) / block.x, (h + block.y - 1) / block.y);
        bilateral_grid_slice_backward_exposure_chroma_reference_kernel<<<grid_dim, block, 0, stream>>>(
            grid, rgb, grad_output, grad_grid, grad_rgb, L, H, W, h, w, shared_offset);
        LFS_CUDA_LAUNCH_CHECK(stream, "training.bilateral.slice_backward_exposure_chroma_reference");
    }

    __global__ void bilateral_grid_slice_backward_exposure_chroma_chw_reference_kernel(
        const float* __restrict__ grid,
        const float* __restrict__ rgb,
        const float* __restrict__ grad_output,
        float* __restrict__ grad_grid,
        float* __restrict__ grad_rgb,
        const int L, const int H, const int W,
        const int h, const int w,
        const float* __restrict__ shared_offset) {

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

        float sampled[9];
        sample_exposure_chroma_grid_bwd(grid, L, H, W, x0, y0, z0, x1, y1, z1, fx, fy, fz, shared_offset, sampled);

        float3 grad_rgb_in;
        float dL_dsampled[9];
        exposure_chroma_param_vjp(sr, sg, sb, sampled, dr, dg, db, grad_rgb_in, dL_dsampled);

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
        for (int ci = 0; ci < 9; ++ci) {
#pragma unroll
            for (int corner = 0; corner < 8; ++corner) {
                const int grid_idx = (ci * L + cz[corner]) * H * W + cy[corner] * W + cx[corner];
                const float wt = weights[corner];
                const float v = load_ro(&grid[grid_idx]);
                atomicAdd(grad_grid + grid_idx, wt * dL_dsampled[ci]);
                gz_grad += dwdz[corner] * (L - 1) * v * dL_dsampled[ci];
            }
        }

        gz_grad *= static_cast<float>(z0 != z && z1 != z);
        grad_rgb[0 * hw + idx] = grad_rgb_in.x + kC2G_r * gz_grad;
        grad_rgb[1 * hw + idx] = grad_rgb_in.y + kC2G_g * gz_grad;
        grad_rgb[2 * hw + idx] = grad_rgb_in.z + kC2G_b * gz_grad;
    }

    void launch_bilateral_grid_slice_backward_exposure_chroma_chw_reference(
        const float* grid, const float* rgb, const float* grad_output,
        float* grad_grid, float* grad_rgb,
        int L, int H, int W, int h, int w,
        const float* shared_offset,
        cudaStream_t stream) {
        assert(L > 0 && H > 0 && W > 0 && h > 0 && w > 0);
        stream = resolve_stream(stream);

        const int blocks = (h * w + BLOCK_SIZE - 1) / BLOCK_SIZE;
        bilateral_grid_slice_backward_exposure_chroma_chw_reference_kernel<<<blocks, BLOCK_SIZE, 0, stream>>>(
            grid, rgb, grad_output, grad_grid, grad_rgb, L, H, W, h, w, shared_offset);
        LFS_CUDA_LAUNCH_CHECK(stream, "training.bilateral.slice_backward_exposure_chroma_chw_reference");
    }

    // grad_rgb kernels: 16x16 threads, one pixel each.
    constexpr int kBwdThreads = 16;

    __device__ __forceinline__ void fill_trilinear_corners(
        int cx[8], int cy[8], int cz[8],
        const int x0, const int x1, const int y0, const int y1, const int z0, const int z1) {
        cx[0] = x0;
        cx[1] = x1;
        cx[2] = x0;
        cx[3] = x1;
        cx[4] = x0;
        cx[5] = x1;
        cx[6] = x0;
        cx[7] = x1;
        cy[0] = y0;
        cy[1] = y0;
        cy[2] = y1;
        cy[3] = y1;
        cy[4] = y0;
        cy[5] = y0;
        cy[6] = y1;
        cy[7] = y1;
        cz[0] = z0;
        cz[1] = z0;
        cz[2] = z0;
        cz[3] = z0;
        cz[4] = z1;
        cz[5] = z1;
        cz[6] = z1;
        cz[7] = z1;
    }

    // Grid gradient, shared by the affine (12 coefficients) and the exposure-chroma (9) kernels.
    //
    // A block covers kGradTileX x kGradTileY pixels, one warp per pixel row, 32 pixels at a time. Each lane computes
    // dL/dcoefficient for its pixel and stages it in shared memory with fx, fz and the cell key (x0, z0); y0 and fy
    // are the same for the whole row. Then the lanes swap roles: lane 2 * ci + zc owns coefficient ci at z corner zc
    // and walks the 32 staged pixels in order, summing the weighted gradient for both x corners in registers while
    // the key stays the same. When it changes, and at the end of the row, every owner adds its two sums, split by
    // (1 - fy) and fy, to the block histogram: four atomics per lane, on addresses no other lane of the warp touches.
    // The block histogram goes to global memory once at the end.
    //
    // Scattering every pixel's 8 corners x C coefficients from each lane instead serialised: on a large image every
    // lane of a warp sits in the same grid cell, and a shared float atomicAdd is a compare-and-swap loop, so each of
    // those 8 * C atomics retried up to 32 times on a single address.
    constexpr int kGradWarps = 8;
    constexpr int kGradSegments = 4;
    constexpr int kGradTileX = 32 * kGradSegments;
    constexpr int kGradTileY = kGradWarps;
    // Odd stride: owners of different coefficients read the staged pixel from different banks.
    constexpr int kGradStageStride = 33;
    // Larger footprints (small images under a fine grid) add straight to global memory.
    constexpr size_t kGradMaxHistBytes = 16 * 1024;

    __host__ __device__ __forceinline__ float grid_coord(const int i, const int n, const int N) {
        return n > 1 ? static_cast<float>(i) / (n - 1) * (N - 1) : 0.0f;
    }

    struct GridGradPixel {
        float sr, sg, sb;
        float dr, dg, db;
        int x0, x1, y0, y1, z0, z1;
        float fx, fy, fz;
    };

    // hist == nullptr: add to grad_grid directly. Histogram layout is (y, x, z, coefficient), coefficient fastest,
    // so the 2 * C owners of one flush hit 2 * C consecutive words.
    struct GridGradTarget {
        float* hist;
        int xmin, ymin, width;
    };

    template <int C>
    __device__ __forceinline__ void add_grid_grad(
        float* __restrict__ grad_grid, const GridGradTarget& target,
        const int L, const int H, const int W,
        const int ci, const int zi, const int yi, const int xi, const float value) {
        if (target.hist)
            atomicAdd(target.hist + ((((yi - target.ymin) * target.width + (xi - target.xmin)) * L + zi) * C + ci), value);
        else
            atomicAdd(grad_grid + ((ci * L + zi) * H + yi) * W + xi, value);
    }

    template <int C>
    __device__ __forceinline__ void flush_grid_grad_run(
        float* __restrict__ grad_grid, const GridGradTarget& target,
        const int L, const int H, const int W,
        const int key, const int ci, const int zc,
        const int y0, const int y1, const float fy,
        const float acc0, const float acc1) {
        const int x0 = key >> 16;
        const int x1 = min(x0 + 1, W - 1);
        const int zi = min((key & 0xFFFF) + zc, L - 1);
        add_grid_grad<C>(grad_grid, target, L, H, W, ci, zi, y0, x0, acc0 * (1 - fy));
        add_grid_grad<C>(grad_grid, target, L, H, W, ci, zi, y0, x1, acc1 * (1 - fy));
        add_grid_grad<C>(grad_grid, target, L, H, W, ci, zi, y1, x0, acc0 * fy);
        add_grid_grad<C>(grad_grid, target, L, H, W, ci, zi, y1, x1, acc1 * fy);
    }

    // coeff_grad(const GridGradPixel&, float (&dL)[C]) fills dL/dcoefficient for one pixel.
    template <int C, bool kCHW, typename CoeffGrad>
    __device__ __forceinline__ void accumulate_slice_grid_grad(
        const float* __restrict__ rgb,
        const float* __restrict__ grad_output,
        float* __restrict__ grad_grid,
        const int L, const int H, const int W,
        const int h, const int w,
        const int fx_cap, const int fy_cap,
        CoeffGrad coeff_grad) {
        static_assert(2 * C <= 32, "one owner lane per coefficient and z corner");
        extern __shared__ float hist[];
        __shared__ float stage[kGradWarps][(C + 2) * kGradStageStride];
        __shared__ int stage_key[kGradWarps][32];

        const int lane = threadIdx.x & 31;
        const int warp = threadIdx.x >> 5;
        const int px0 = blockIdx.x * kGradTileX;
        const int py0 = blockIdx.y * kGradTileY;
        const int px1 = min(px0 + kGradTileX, w);
        const int py1 = min(py0 + kGradTileY, h);

        // Cells the tile touches; coordinates grow with the pixel index, so the end pixels bound them.
        const int xmin = static_cast<int>(floorf(grid_coord(px0, w, W)));
        const int ymin = static_cast<int>(floorf(grid_coord(py0, h, H)));
        const int xmax = min(static_cast<int>(floorf(grid_coord(px1 - 1, w, W))) + 1, W - 1);
        const int ymax = min(static_cast<int>(floorf(grid_coord(py1 - 1, h, H))) + 1, H - 1);
        const int tile_fx = xmax - xmin + 1;
        const int tile_fy = ymax - ymin + 1;
        const bool use_hist = tile_fx <= fx_cap && tile_fy <= fy_cap;
        const int nslots = use_hist ? C * L * tile_fx * tile_fy : 0;
        for (int i = threadIdx.x; i < nslots; i += blockDim.x)
            hist[i] = 0.0f;
        __syncthreads();

        const GridGradTarget target{use_hist ? hist : nullptr, xmin, ymin, tile_fx};
        const bool owner = lane < 2 * C;
        const int ci = lane >> 1;
        const int zc = lane & 1;
        float* const st = stage[warp];
        int* const sk = stage_key[warp];
        const int hw = h * w;

        const int hi = py0 + warp;
        if (hi < h) {
            const float y = grid_coord(hi, h, H);
            const int y0 = floorf(y);
            const int y1 = min(y0 + 1, H - 1);
            const float fy = y - y0;

            int run_key = -1;
            float acc0 = 0.0f, acc1 = 0.0f;
            const auto switch_run = [&](const int key) {
                if (owner && run_key >= 0)
                    flush_grid_grad_run<C>(grad_grid, target, L, H, W, run_key, ci, zc, y0, y1, fy, acc0, acc1);
                acc0 = 0.0f;
                acc1 = 0.0f;
                run_key = key;
            };
            const auto add_pixel = [&](const int p) {
                const float fx = st[C * kGradStageStride + p];
                const float fz = st[(C + 1) * kGradStageStride + p];
                const float t = (zc ? fz : 1 - fz) * st[ci * kGradStageStride + p];
                acc0 += (1 - fx) * t;
                acc1 += fx * t;
            };
            for (int seg = px0; seg < px1; seg += 32) {
                const int wi = seg + lane;
                int key = -1;
                if (wi < px1) {
                    const int pixel_idx = hi * w + wi;
                    GridGradPixel s;
                    if constexpr (kCHW) {
                        s.sr = rgb[0 * hw + pixel_idx];
                        s.sg = rgb[1 * hw + pixel_idx];
                        s.sb = rgb[2 * hw + pixel_idx];
                        s.dr = grad_output[0 * hw + pixel_idx];
                        s.dg = grad_output[1 * hw + pixel_idx];
                        s.db = grad_output[2 * hw + pixel_idx];
                    } else {
                        const RGB rgb_val = load_rgb_cs(&rgb[pixel_idx * 3]);
                        s.sr = rgb_val.r;
                        s.sg = rgb_val.g;
                        s.sb = rgb_val.b;
                        const RGB grad = load_rgb_cs(&grad_output[pixel_idx * 3]);
                        s.dr = grad.r;
                        s.dg = grad.g;
                        s.db = grad.b;
                    }
                    s.sr = isfinite(s.sr) ? s.sr : 0.5f;
                    s.sg = isfinite(s.sg) ? s.sg : 0.5f;
                    s.sb = isfinite(s.sb) ? s.sb : 0.5f;
                    s.dr = isfinite(s.dr) ? s.dr : 0.0f;
                    s.dg = isfinite(s.dg) ? s.dg : 0.0f;
                    s.db = isfinite(s.db) ? s.db : 0.0f;

                    const float x = grid_coord(wi, w, W);
                    const float guidance = fminf(1.0f, fmaxf(0.0f, kC2G_r * s.sr + kC2G_g * s.sg + kC2G_b * s.sb));
                    const float z = guidance * (L - 1);
                    s.x0 = floorf(x);
                    s.x1 = min(s.x0 + 1, W - 1);
                    s.y0 = y0;
                    s.y1 = y1;
                    const int zf = floorf(z);
                    s.z0 = min(max(zf, 0), L - 1);
                    s.z1 = min(max(zf + 1, 0), L - 1);
                    s.fx = x - s.x0;
                    s.fy = fy;
                    s.fz = z - s.z0;

                    float dL[C];
                    coeff_grad(s, dL);
#pragma unroll
                    for (int c = 0; c < C; ++c)
                        st[c * kGradStageStride + lane] = dL[c];
                    st[C * kGradStageStride + lane] = s.fx;
                    st[(C + 1) * kGradStageStride + lane] = s.fz;
                    key = (s.x0 << 16) | s.z0;
                    sk[lane] = key;
                }
                __syncwarp();

                // Whole segment in one cell (the usual case on large images): no key checks per pixel.
                const int first = __shfl_sync(0xffffffffu, key, 0);
                if (__all_sync(0xffffffffu, key == first)) {
                    if (first != run_key)
                        switch_run(first);
                    if (owner) {
#pragma unroll
                        for (int p = 0; p < 32; ++p)
                            add_pixel(p);
                    }
                } else {
                    const int count = min(32, px1 - seg);
                    for (int p = 0; p < count; ++p) {
                        if (sk[p] != run_key)
                            switch_run(sk[p]);
                        if (owner)
                            add_pixel(p);
                    }
                }
                __syncwarp();
            }
            switch_run(-1);
        }
        __syncthreads();

        for (int i = threadIdx.x; i < nslots; i += blockDim.x) {
            const float v = hist[i];
            if (v == 0.0f)
                continue;
            int rest = i;
            const int c = rest % C;
            rest /= C;
            const int zi = rest % L;
            rest /= L;
            const int lx = rest % tile_fx;
            const int ly = rest / tile_fx;
            atomicAdd(grad_grid + ((c * L + zi) * H + ymin + ly) * W + xmin + lx, v);
        }
    }

    // Backward is split into two kernels. The fused version kept grad_rgb state
    // (96 cached grid loads, gz chain) and the grid-gradient histogram alive at
    // once and spilled past 250 registers; splitting keeps both below the spill
    // threshold and restores occupancy.

    template <bool kCHW>
    __global__ void bilateral_grid_slice_backward_rgbgrad_kernel(
        const float* __restrict__ grid,
        const float* __restrict__ rgb,
        const float* grad_output,
        float* grad_rgb,
        const int L, const int H, const int W,
        const int h, const int w,
        const float* __restrict__ shared_offset) {
        const int wi = blockIdx.x * blockDim.x + threadIdx.x;
        const int hi = blockIdx.y * blockDim.y + threadIdx.y;
        if (wi >= w || hi >= h)
            return;
        const int hw = h * w;
        const int pixel_idx = hi * w + wi;
        float sr, sg, sb, dr, dg, db;
        if constexpr (kCHW) {
            sr = rgb[0 * hw + pixel_idx];
            sg = rgb[1 * hw + pixel_idx];
            sb = rgb[2 * hw + pixel_idx];
            dr = grad_output[0 * hw + pixel_idx];
            dg = grad_output[1 * hw + pixel_idx];
            db = grad_output[2 * hw + pixel_idx];
        } else {
            const int rgb_offset = pixel_idx * 3;
            const RGB rgb_val = load_rgb_cs(&rgb[rgb_offset]);
            sr = rgb_val.r;
            sg = rgb_val.g;
            sb = rgb_val.b;
            const RGB grad = load_rgb_cs(&grad_output[rgb_offset]);
            dr = grad.r;
            dg = grad.g;
            db = grad.b;
        }
        sr = isfinite(sr) ? sr : 0.5f;
        sg = isfinite(sg) ? sg : 0.5f;
        sb = isfinite(sb) ? sb : 0.5f;
        dr = isfinite(dr) ? dr : 0.0f;
        dg = isfinite(dg) ? dg : 0.0f;
        db = isfinite(db) ? db : 0.0f;

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
        const float w00 = (1 - fx) * (1 - fy);
        const float w01 = fx * (1 - fy);
        const float w10 = (1 - fx) * fy;
        const float w11 = fx * fy;

        float vr = 0.0f, vg = 0.0f, vb = 0.0f;
        float gz_grad = 0.0f;
#pragma unroll
        for (int ci = 0; ci < 12; ++ci) {
            const int si = ci % 4, di = ci / 4;
            const float r_coeff = (si == 0 ? sr : si == 1 ? sg
                                              : si == 2   ? sb
                                                          : 1.0f);
            const float gout = (di == 0 ? dr : di == 1 ? dg
                                                       : db);
            const int base0 = (ci * L + z0) * H * W;
            const int base1 = (ci * L + z1) * H * W;
            const int r0 = y0 * W, r1 = y1 * W;
            const float off = load_grid_offset(shared_offset, ci);
            const float p0 = w00 * load_ro(&grid[base0 + r0 + x0]) + w01 * load_ro(&grid[base0 + r0 + x1]) +
                             w10 * load_ro(&grid[base0 + r1 + x0]) + w11 * load_ro(&grid[base0 + r1 + x1]) + off;
            const float p1 = w00 * load_ro(&grid[base1 + r0 + x0]) + w01 * load_ro(&grid[base1 + r0 + x1]) +
                             w10 * load_ro(&grid[base1 + r1 + x0]) + w11 * load_ro(&grid[base1 + r1 + x1]) + off;
            const float v = p0 + (p1 - p0) * fz;
            if (si < 3) {
                const float add = v * gout;
                if (si == 0)
                    vr += add;
                else if (si == 1)
                    vg += add;
                else
                    vb += add;
            }
            gz_grad += (p1 - p0) * (L - 1) * r_coeff * gout;
        }

        gz_grad *= static_cast<float>(z0 != z && z1 != z);
        if constexpr (kCHW) {
            grad_rgb[0 * hw + pixel_idx] = vr + kC2G_r * gz_grad;
            grad_rgb[1 * hw + pixel_idx] = vg + kC2G_g * gz_grad;
            grad_rgb[2 * hw + pixel_idx] = vb + kC2G_b * gz_grad;
        } else {
            const int rgb_offset = pixel_idx * 3;
            grad_rgb[rgb_offset + 0] = vr + kC2G_r * gz_grad;
            grad_rgb[rgb_offset + 1] = vg + kC2G_g * gz_grad;
            grad_rgb[rgb_offset + 2] = vb + kC2G_b * gz_grad;
        }
    }

    template <bool kCHW>
    __global__ void __launch_bounds__(kGradWarps * 32) bilateral_grid_slice_gridgrad_kernel(
        const float* __restrict__ rgb,
        const float* __restrict__ grad_output,
        float* __restrict__ grad_grid,
        const int L, const int H, const int W,
        const int h, const int w,
        const int fx_cap, const int fy_cap) {
        accumulate_slice_grid_grad<12, kCHW>(
            rgb, grad_output, grad_grid, L, H, W, h, w, fx_cap, fy_cap,
            [](const GridGradPixel& s, float(&dL)[12]) {
#pragma unroll
                for (int ci = 0; ci < 12; ++ci) {
                    const int si = ci % 4, di = ci / 4;
                    const float r_coeff = (si == 0 ? s.sr : si == 1 ? s.sg
                                                        : si == 2   ? s.sb
                                                                    : 1.0f);
                    const float gout = (di == 0 ? s.dr : di == 1 ? s.dg
                                                                 : s.db);
                    dL[ci] = r_coeff * gout;
                }
            });
    }

    template <bool kCHW>
    __global__ void bilateral_grid_slice_backward_exposure_chroma_rgbgrad_kernel(
        const float* __restrict__ grid,
        const float* __restrict__ rgb,
        const float* grad_output,
        float* grad_rgb,
        const int L, const int H, const int W,
        const int h, const int w,
        const float* __restrict__ shared_offset) {
        const int wi = blockIdx.x * blockDim.x + threadIdx.x;
        const int hi = blockIdx.y * blockDim.y + threadIdx.y;
        if (wi >= w || hi >= h)
            return;
        const int hw = h * w;
        const int pixel_idx = hi * w + wi;
        float sr, sg, sb, dr, dg, db;
        if constexpr (kCHW) {
            sr = rgb[0 * hw + pixel_idx];
            sg = rgb[1 * hw + pixel_idx];
            sb = rgb[2 * hw + pixel_idx];
            dr = grad_output[0 * hw + pixel_idx];
            dg = grad_output[1 * hw + pixel_idx];
            db = grad_output[2 * hw + pixel_idx];
        } else {
            const int rgb_offset = pixel_idx * 3;
            const RGB rgb_val = load_rgb_cs(&rgb[rgb_offset]);
            sr = rgb_val.r;
            sg = rgb_val.g;
            sb = rgb_val.b;
            const RGB grad = load_rgb_cs(&grad_output[rgb_offset]);
            dr = grad.r;
            dg = grad.g;
            db = grad.b;
        }
        sr = isfinite(sr) ? sr : 0.5f;
        sg = isfinite(sg) ? sg : 0.5f;
        sb = isfinite(sb) ? sb : 0.5f;
        dr = isfinite(dr) ? dr : 0.0f;
        dg = isfinite(dg) ? dg : 0.0f;
        db = isfinite(db) ? db : 0.0f;

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
        float sampled[9];
        sample_exposure_chroma_grid_bwd(
            grid, L, H, W, x0, y0, z0, x1, y1, z1, fx, fy, fz, shared_offset, sampled);

        float dL[9];
        float3 grad_rgb_in;
        exposure_chroma_param_vjp(sr, sg, sb, sampled, dr, dg, db, grad_rgb_in, dL);

        int pcx[8], pcy[8], pcz[8];
        fill_trilinear_corners(pcx, pcy, pcz, x0, x1, y0, y1, z0, z1);
        const float dwdz[8] = {
            -(1 - fx) * (1 - fy), -fx * (1 - fy),
            -(1 - fx) * fy, -fx * fy,
            (1 - fx) * (1 - fy), fx * (1 - fy),
            (1 - fx) * fy, fx * fy};

        float gz_grad = 0.0f;
#pragma unroll
        for (int ci = 0; ci < 9; ++ci) {
#pragma unroll
            for (int corner = 0; corner < 8; ++corner) {
                const int grid_idx = (ci * L + pcz[corner]) * H * W + pcy[corner] * W + pcx[corner];
                const float v = load_ro(&grid[grid_idx]);
                gz_grad += dwdz[corner] * (L - 1) * v * dL[ci];
            }
        }

        gz_grad *= static_cast<float>(z0 != z && z1 != z);
        if constexpr (kCHW) {
            grad_rgb[0 * hw + pixel_idx] = grad_rgb_in.x + kC2G_r * gz_grad;
            grad_rgb[1 * hw + pixel_idx] = grad_rgb_in.y + kC2G_g * gz_grad;
            grad_rgb[2 * hw + pixel_idx] = grad_rgb_in.z + kC2G_b * gz_grad;
        } else {
            const int rgb_offset = pixel_idx * 3;
            grad_rgb[rgb_offset + 0] = grad_rgb_in.x + kC2G_r * gz_grad;
            grad_rgb[rgb_offset + 1] = grad_rgb_in.y + kC2G_g * gz_grad;
            grad_rgb[rgb_offset + 2] = grad_rgb_in.z + kC2G_b * gz_grad;
        }
    }

    template <bool kCHW>
    __global__ void __launch_bounds__(kGradWarps * 32) bilateral_grid_slice_exposure_chroma_gridgrad_kernel(
        const float* __restrict__ grid,
        const float* __restrict__ rgb,
        const float* __restrict__ grad_output,
        float* __restrict__ grad_grid,
        const int L, const int H, const int W,
        const int h, const int w,
        const float* __restrict__ shared_offset,
        const int fx_cap, const int fy_cap) {
        accumulate_slice_grid_grad<9, kCHW>(
            rgb, grad_output, grad_grid, L, H, W, h, w, fx_cap, fy_cap,
            [=](const GridGradPixel& s, float(&dL)[9]) {
                float sampled[9];
                sample_exposure_chroma_grid_bwd(
                    grid, L, H, W, s.x0, s.y0, s.z0, s.x1, s.y1, s.z1, s.fx, s.fy, s.fz, shared_offset, sampled);
                float3 grad_rgb_in;
                exposure_chroma_param_vjp(s.sr, s.sg, s.sb, sampled, s.dr, s.dg, s.db, grad_rgb_in, dL);
            });
    }

    struct GridGradLaunch {
        dim3 blocks;
        int fx_cap;
        int fy_cap;
        size_t smem;
    };

    // Upper bound of the grid cells along one axis under a tile of `tile` pixels, the +1 corner included.
    static int grid_grad_footprint_bound(const int tile, const int n, const int N) {
        if (n <= 1)
            return std::min(N, 2);
        const double span = static_cast<double>(tile - 1) * (N - 1) / (n - 1);
        return std::min(N, static_cast<int>(span) + 3);
    }

    static GridGradLaunch grid_grad_launch(const int C, const int L, const int H, const int W, const int h, const int w) {
        GridGradLaunch launch{
            dim3((w + kGradTileX - 1) / kGradTileX, (h + kGradTileY - 1) / kGradTileY),
            grid_grad_footprint_bound(kGradTileX, w, W),
            grid_grad_footprint_bound(kGradTileY, h, H),
            0};
        const size_t bytes = sizeof(float) * C * L * launch.fx_cap * launch.fy_cap;
        if (bytes > kGradMaxHistBytes) {
            launch.fx_cap = 0;
            launch.fy_cap = 0;
        } else {
            launch.smem = bytes;
        }
        return launch;
    }

    void launch_bilateral_grid_slice_backward(
        const float* grid, const float* rgb, const float* grad_output,
        float* grad_grid, float* grad_rgb,
        int L, int H, int W, int h, int w,
        const float* shared_offset,
        cudaStream_t stream) {
        assert(L > 0 && H > 0 && W > 0 && h > 0 && w > 0);
        stream = resolve_stream(stream);
        const dim3 block(kBwdThreads, kBwdThreads);
        const dim3 pix_grid((w + kBwdThreads - 1) / kBwdThreads, (h + kBwdThreads - 1) / kBwdThreads);
        const GridGradLaunch gg = grid_grad_launch(12, L, H, W, h, w);
        bilateral_grid_slice_gridgrad_kernel<false>
            <<<gg.blocks, kGradWarps * 32, gg.smem, stream>>>(
                rgb, grad_output, grad_grid, L, H, W, h, w, gg.fx_cap, gg.fy_cap);
        LFS_CUDA_LAUNCH_CHECK(stream, "training.bilateral.slice_backward");
        bilateral_grid_slice_backward_rgbgrad_kernel<false>
            <<<pix_grid, block, 0, stream>>>(
                grid, rgb, grad_output, grad_rgb, L, H, W, h, w, shared_offset);
        LFS_CUDA_LAUNCH_CHECK(stream, "training.bilateral.slice_backward_rgbgrad");
    }

    void launch_bilateral_grid_slice_backward_chw(
        const float* grid, const float* rgb, const float* grad_output,
        float* grad_grid, float* grad_rgb,
        int L, int H, int W, int h, int w,
        const float* shared_offset,
        cudaStream_t stream) {
        assert(L > 0 && H > 0 && W > 0 && h > 0 && w > 0);
        stream = resolve_stream(stream);
        const dim3 block(kBwdThreads, kBwdThreads);
        const dim3 pix_grid((w + kBwdThreads - 1) / kBwdThreads, (h + kBwdThreads - 1) / kBwdThreads);
        const GridGradLaunch gg = grid_grad_launch(12, L, H, W, h, w);
        bilateral_grid_slice_gridgrad_kernel<true>
            <<<gg.blocks, kGradWarps * 32, gg.smem, stream>>>(
                rgb, grad_output, grad_grid, L, H, W, h, w, gg.fx_cap, gg.fy_cap);
        LFS_CUDA_LAUNCH_CHECK(stream, "training.bilateral.slice_backward_chw");
        bilateral_grid_slice_backward_rgbgrad_kernel<true>
            <<<pix_grid, block, 0, stream>>>(
                grid, rgb, grad_output, grad_rgb, L, H, W, h, w, shared_offset);
        LFS_CUDA_LAUNCH_CHECK(stream, "training.bilateral.slice_backward_rgbgrad_chw");
    }

    void launch_bilateral_grid_slice_backward_exposure_chroma(
        const float* grid, const float* rgb, const float* grad_output,
        float* grad_grid, float* grad_rgb,
        int L, int H, int W, int h, int w,
        const float* shared_offset,
        cudaStream_t stream) {
        assert(L > 0 && H > 0 && W > 0 && h > 0 && w > 0);
        stream = resolve_stream(stream);
        const dim3 block(kBwdThreads, kBwdThreads);
        const dim3 pix_grid((w + kBwdThreads - 1) / kBwdThreads, (h + kBwdThreads - 1) / kBwdThreads);
        const GridGradLaunch gg = grid_grad_launch(9, L, H, W, h, w);
        bilateral_grid_slice_exposure_chroma_gridgrad_kernel<false>
            <<<gg.blocks, kGradWarps * 32, gg.smem, stream>>>(
                grid, rgb, grad_output, grad_grid, L, H, W, h, w, shared_offset, gg.fx_cap, gg.fy_cap);
        LFS_CUDA_LAUNCH_CHECK(stream, "training.bilateral.slice_backward_exposure_chroma");
        bilateral_grid_slice_backward_exposure_chroma_rgbgrad_kernel<false>
            <<<pix_grid, block, 0, stream>>>(
                grid, rgb, grad_output, grad_rgb, L, H, W, h, w, shared_offset);
        LFS_CUDA_LAUNCH_CHECK(stream, "training.bilateral.slice_backward_exposure_chroma_rgbgrad");
    }

    void launch_bilateral_grid_slice_backward_exposure_chroma_chw(
        const float* grid, const float* rgb, const float* grad_output,
        float* grad_grid, float* grad_rgb,
        int L, int H, int W, int h, int w,
        const float* shared_offset,
        cudaStream_t stream) {
        assert(L > 0 && H > 0 && W > 0 && h > 0 && w > 0);
        stream = resolve_stream(stream);
        const dim3 block(kBwdThreads, kBwdThreads);
        const dim3 pix_grid((w + kBwdThreads - 1) / kBwdThreads, (h + kBwdThreads - 1) / kBwdThreads);
        const GridGradLaunch gg = grid_grad_launch(9, L, H, W, h, w);
        bilateral_grid_slice_exposure_chroma_gridgrad_kernel<true>
            <<<gg.blocks, kGradWarps * 32, gg.smem, stream>>>(
                grid, rgb, grad_output, grad_grid, L, H, W, h, w, shared_offset, gg.fx_cap, gg.fy_cap);
        LFS_CUDA_LAUNCH_CHECK(stream, "training.bilateral.slice_backward_exposure_chroma_chw");
        bilateral_grid_slice_backward_exposure_chroma_rgbgrad_kernel<true>
            <<<pix_grid, block, 0, stream>>>(
                grid, rgb, grad_output, grad_rgb, L, H, W, h, w, shared_offset);
        LFS_CUDA_LAUNCH_CHECK(stream, "training.bilateral.slice_backward_exposure_chroma_rgbgrad_chw");
    }

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
        LFS_CUDA_LAUNCH_CHECK(stream, "training.bilateral.adam_update");
    }

    __global__ void bilateral_grid_scale_moments_kernel(
        float* __restrict__ exp_avg,
        float* __restrict__ exp_avg_sq,
        const int num_elements,
        const float scale_avg,
        const float scale_avg_sq) {

        const int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx >= num_elements)
            return;
        exp_avg[idx] *= scale_avg;
        exp_avg_sq[idx] *= scale_avg_sq;
    }

    void launch_bilateral_grid_scale_moments(
        float* exp_avg, float* exp_avg_sq, int num_elements,
        float scale_avg, float scale_avg_sq,
        cudaStream_t stream) {
        if (num_elements <= 0)
            return;
        stream = resolve_stream(stream);

        const int blocks = (num_elements + BLOCK_SIZE - 1) / BLOCK_SIZE;
        bilateral_grid_scale_moments_kernel<<<blocks, BLOCK_SIZE, 0, stream>>>(
            exp_avg, exp_avg_sq, num_elements, scale_avg, scale_avg_sq);
        LFS_CUDA_LAUNCH_CHECK(stream, "training.bilateral.scale_moments");
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
        LFS_CUDA_LAUNCH_CHECK(stream, "training.bilateral.accumulate_grad");
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

    __global__ void bilateral_grid_project_mean_kernel(
        float* __restrict__ grids,
        const float* __restrict__ mean,
        const float* __restrict__ identity,
        const int N, const int C, const int spatial, const int per_image) {

        const int idx = blockIdx.x * blockDim.x + threadIdx.x;
        const int total = N * C * spatial;
        if (idx >= total)
            return;

        int tmp = idx;
        tmp /= spatial;
        const int ci = tmp % C;
        tmp /= C;
        const int ni = tmp;
        const float m = per_image ? mean[ni * C + ci] : mean[ci];
        grids[idx] += identity[ci] - m;
    }

    void launch_bilateral_grid_project_mean(
        float* grids, const float* mean, const float* identity,
        int N, int C, int L, int H, int W, int per_image,
        cudaStream_t stream) {
        assert(N > 0 && C > 0 && L > 0 && H > 0 && W > 0);
        stream = resolve_stream(stream);
        const int spatial = L * H * W;
        const int total = N * C * spatial;
        const int blocks = (total + BLOCK_SIZE - 1) / BLOCK_SIZE;
        bilateral_grid_project_mean_kernel<<<blocks, BLOCK_SIZE, 0, stream>>>(
            grids, mean, identity, N, C, spatial, per_image);
        LFS_CUDA_LAUNCH_CHECK(stream, "training.bilateral.project_mean");
    }

    __global__ void bilateral_grid_update_shared_offset_kernel(
        float* __restrict__ channel_sum,
        float* __restrict__ shared_offset,
        const float* __restrict__ identity,
        const float* __restrict__ mean_old,
        const float* __restrict__ mean_new,
        const int C, const float spatial, const float inv_n_spatial) {
        const int c = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
        if (c >= C)
            return;
        channel_sum[c] += (mean_new[c] - mean_old[c]) * spatial;
        shared_offset[c] = identity[c] - channel_sum[c] * inv_n_spatial;
    }

    void launch_bilateral_grid_update_shared_offset(
        float* channel_sum, float* shared_offset,
        const float* identity, const float* mean_old, const float* mean_new,
        int C, float spatial, float inv_n_spatial,
        cudaStream_t stream) {
        assert(C > 0);
        stream = resolve_stream(stream);
        const int blocks = (C + BLOCK_SIZE - 1) / BLOCK_SIZE;
        bilateral_grid_update_shared_offset_kernel<<<blocks, BLOCK_SIZE, 0, stream>>>(
            channel_sum, shared_offset, identity, mean_old, mean_new, C, spatial, inv_n_spatial);
        LFS_CUDA_LAUNCH_CHECK(stream, "training.bilateral.update_shared_offset");
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
        LFS_CUDA_LAUNCH_CHECK(stream, "training.bilateral.init_identity");
    }

} // namespace lfs::training::kernels

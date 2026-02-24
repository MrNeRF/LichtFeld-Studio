/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "config.h"

#ifdef CUDA_GL_INTEROP_ENABLED

#include <cassert>
#include <cuda_runtime.h>

#include "core/cuda_debug.hpp"

namespace lfs {

    // CUDA kernel to interleave position and color data
    __global__ void writeInterleavedPosColorKernel(
        const float* __restrict__ positions, // [N, 3]
        const float* __restrict__ colors,    // [N, 3]
        float* __restrict__ output,          // [N, 6] interleaved
        int num_points) {

        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx >= num_points)
            return;

        // Read position (x, y, z)
        float px = positions[idx * 3 + 0];
        float py = positions[idx * 3 + 1];
        float pz = positions[idx * 3 + 2];

        // Read color (r, g, b)
        float cr = colors[idx * 3 + 0];
        float cg = colors[idx * 3 + 1];
        float cb = colors[idx * 3 + 2];

        // Write interleaved (x, y, z, r, g, b)
        int out_idx = idx * 6;
        output[out_idx + 0] = px;
        output[out_idx + 1] = py;
        output[out_idx + 2] = pz;
        output[out_idx + 3] = cr;
        output[out_idx + 4] = cg;
        output[out_idx + 5] = cb;
    }

    // Host function to launch the kernel
    void launchWriteInterleavedPosColor(
        const float* positions,
        const float* colors,
        float* output,
        int num_points,
        cudaStream_t stream) {

        if (num_points <= 0)
            return;

        const int threads = 256;
        const int blocks = (num_points + threads - 1) / threads;

        writeInterleavedPosColorKernel<<<blocks, threads, 0, stream>>>(
            positions, colors, output, num_points);
        CUDA_KERNEL_CHECK("writeInterleavedPosColorKernel");
    }

    __global__ void floatImageToRGBA8SurfaceKernel(
        const float* __restrict__ input,
        cudaSurfaceObject_t output,
        int width, int height, int channels, bool is_chw) {

        const int x = blockIdx.x * blockDim.x + threadIdx.x;
        const int y = blockIdx.y * blockDim.y + threadIdx.y;
        if (x >= width || y >= height)
            return;

        float r, g, b, a;
        if (is_chw) {
            const int hw = height * width;
            const int idx = y * width + x;
            r = input[idx];
            g = input[hw + idx];
            b = input[2 * hw + idx];
            a = (channels == 4) ? input[3 * hw + idx] : 1.0f;
        } else {
            const int base = (y * width + x) * channels;
            r = input[base];
            g = input[base + 1];
            b = input[base + 2];
            a = (channels == 4) ? input[base + 3] : 1.0f;
        }

        uchar4 pixel;
        pixel.x = static_cast<unsigned char>(fminf(fmaxf(r, 0.0f), 1.0f) * 255.0f + 0.5f);
        pixel.y = static_cast<unsigned char>(fminf(fmaxf(g, 0.0f), 1.0f) * 255.0f + 0.5f);
        pixel.z = static_cast<unsigned char>(fminf(fmaxf(b, 0.0f), 1.0f) * 255.0f + 0.5f);
        pixel.w = static_cast<unsigned char>(fminf(fmaxf(a, 0.0f), 1.0f) * 255.0f + 0.5f);

        surf2Dwrite(pixel, output, x * static_cast<int>(sizeof(uchar4)), y);
    }

    void launchFloatImageToRGBA8Surface(
        const float* input,
        cudaSurfaceObject_t output,
        int width, int height, int channels, bool is_chw,
        cudaStream_t stream) {

        assert(input != nullptr);
        assert(width > 0 && height > 0);
        assert(channels == 3 || channels == 4);

        dim3 block(16, 16);
        dim3 grid((width + block.x - 1) / block.x,
                  (height + block.y - 1) / block.y);

        floatImageToRGBA8SurfaceKernel<<<grid, block, 0, stream>>>(
            input, output, width, height, channels, is_chw);
        CUDA_KERNEL_CHECK("floatImageToRGBA8SurfaceKernel");
    }

} // namespace lfs

#endif // CUDA_GL_INTEROP_ENABLED

// Saturation adjustment kernel
namespace lfs {

    constexpr float SH_C0 = 0.28209479177387814f;

    __global__ void adjustSaturationKernel(
        float* __restrict__ sh0,
        const float* __restrict__ screen_positions,
        const float brush_x,
        const float brush_y,
        const float brush_radius_sq,
        const float saturation_delta,
        const int num_gaussians) {

        const int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx >= num_gaussians)
            return;

        const float dx = screen_positions[idx * 2 + 0] - brush_x;
        const float dy = screen_positions[idx * 2 + 1] - brush_y;
        if (dx * dx + dy * dy > brush_radius_sq)
            return;

        // SH to RGB: color = SH_C0 * sh + 0.5
        const float r = SH_C0 * sh0[idx * 3 + 0] + 0.5f;
        const float g = SH_C0 * sh0[idx * 3 + 1] + 0.5f;
        const float b = SH_C0 * sh0[idx * 3 + 2] + 0.5f;

        // Luminance (Rec. 709)
        const float lum = 0.2126f * r + 0.7152f * g + 0.0722f * b;

        // Saturation adjustment
        const float factor = 1.0f + saturation_delta;
        const float new_r = fmaxf(0.0f, fminf(1.0f, lum + factor * (r - lum)));
        const float new_g = fmaxf(0.0f, fminf(1.0f, lum + factor * (g - lum)));
        const float new_b = fmaxf(0.0f, fminf(1.0f, lum + factor * (b - lum)));

        // RGB to SH: sh = (color - 0.5) / SH_C0
        sh0[idx * 3 + 0] = (new_r - 0.5f) / SH_C0;
        sh0[idx * 3 + 1] = (new_g - 0.5f) / SH_C0;
        sh0[idx * 3 + 2] = (new_b - 0.5f) / SH_C0;
    }

    void launchAdjustSaturation(
        float* sh0,
        const float* screen_positions,
        const float brush_x,
        const float brush_y,
        const float brush_radius,
        const float saturation_delta,
        const int num_gaussians,
        cudaStream_t stream) {

        if (num_gaussians <= 0)
            return;

        constexpr int threads = 256;
        const int blocks = (num_gaussians + threads - 1) / threads;
        adjustSaturationKernel<<<blocks, threads, 0, stream>>>(
            sh0, screen_positions, brush_x, brush_y,
            brush_radius * brush_radius, saturation_delta, num_gaussians);
    }

} // namespace lfs

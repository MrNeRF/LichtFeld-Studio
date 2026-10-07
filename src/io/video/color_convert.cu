/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "color_convert.cuh"

#include "core/cuda_error.hpp"

namespace lfs::io::video {

    namespace {

        constexpr int BLOCK_SIZE = 16;

        __device__ __forceinline__ uint8_t clampU8(const int val) {
            return static_cast<uint8_t>(min(max(val, 0), 255));
        }

    } // namespace

    __global__ void nv12ToRgbKernel(
        const uint8_t* __restrict__ y_plane,
        const uint8_t* __restrict__ uv_plane,
        uint8_t* __restrict__ rgb,
        const int width,
        const int height,
        const int y_pitch,
        const int uv_pitch) {

        const int x = blockIdx.x * blockDim.x + threadIdx.x;
        const int y = blockIdx.y * blockDim.y + threadIdx.y;

        if (x >= width || y >= height)
            return;

        const int y_val = y_plane[y * y_pitch + x];

        const int uv_x = (x >> 1) << 1;
        const int uv_y = y >> 1;
        const int uv_idx = uv_y * uv_pitch + uv_x;
        const int u = uv_plane[uv_idx];
        const int v = uv_plane[uv_idx + 1];

        // BT.601 YUV→RGB conversion (scaled integer math)
        // R = 1.164 * (Y - 16) + 1.596 * (V - 128)
        // G = 1.164 * (Y - 16) - 0.813 * (V - 128) - 0.391 * (U - 128)
        // B = 1.164 * (Y - 16) + 2.018 * (U - 128)
        const int c = y_val - 16;
        const int d = u - 128;
        const int e = v - 128;

        const int r = (298 * c + 409 * e + 128) >> 8;
        const int g = (298 * c - 100 * d - 208 * e + 128) >> 8;
        const int b = (298 * c + 516 * d + 128) >> 8;

        const int rgb_idx = (y * width + x) * 3;
        rgb[rgb_idx] = clampU8(r);
        rgb[rgb_idx + 1] = clampU8(g);
        rgb[rgb_idx + 2] = clampU8(b);
    }

    void nv12ToRgbCuda(
        const uint8_t* const y_src,
        const uint8_t* const uv_src,
        uint8_t* const rgb_dst,
        const int width,
        const int height,
        const int y_pitch,
        const int uv_pitch,
        cudaStream_t stream) {

        const int effective_y_pitch = (y_pitch > 0) ? y_pitch : width;
        const int effective_uv_pitch = (uv_pitch > 0) ? uv_pitch : width;

        const dim3 block(BLOCK_SIZE, BLOCK_SIZE);
        const dim3 grid((width + BLOCK_SIZE - 1) / BLOCK_SIZE,
                        (height + BLOCK_SIZE - 1) / BLOCK_SIZE);

        nv12ToRgbKernel<<<grid, block, 0, stream>>>(
            y_src, uv_src, rgb_dst, width, height, effective_y_pitch, effective_uv_pitch);
        LFS_CUDA_LAUNCH_CHECK(stream, "io.video.nv12_to_rgb");
    }

    __global__ void rotateRgbKernel(
        const uint8_t* __restrict__ src,
        uint8_t* __restrict__ dst,
        const int width,
        const int height,
        const int angle) {

        const int x = blockIdx.x * blockDim.x + threadIdx.x;
        const int y = blockIdx.y * blockDim.y + threadIdx.y;

        if (angle == 90) {
            const int h = width;
            const int w = height;
            if (x >= w || y >= h)
                return;
            // 90° CW: dst[y][x] = src[height-1-x][y]
            const int src_x = y;
            const int src_y = height - 1 - x;
            const int src_idx = (src_y * width + src_x) * 3;
            const int dst_idx = (y * w + x) * 3;
            dst[dst_idx] = src[src_idx];
            dst[dst_idx + 1] = src[src_idx + 1];
            dst[dst_idx + 2] = src[src_idx + 2];
        } else if (angle == 180) {
            if (x >= width || y >= height)
                return;
            // 180°: dst[y][x] = src[height-1-y][width-1-x]
            const int src_x = width - 1 - x;
            const int src_y = height - 1 - y;
            const int src_idx = (src_y * width + src_x) * 3;
            const int dst_idx = (y * width + x) * 3;
            dst[dst_idx] = src[src_idx];
            dst[dst_idx + 1] = src[src_idx + 1];
            dst[dst_idx + 2] = src[src_idx + 2];
        } else { // 270
            const int h = width;
            const int w = height;
            if (x >= w || y >= h)
                return;
            // 270° CW (= 90° CCW): dst[y][x] = src[x][width-1-y]
            const int src_x = width - 1 - y;
            const int src_y = x;
            const int src_idx = (src_y * width + src_x) * 3;
            const int dst_idx = (y * w + x) * 3;
            dst[dst_idx] = src[src_idx];
            dst[dst_idx + 1] = src[src_idx + 1];
            dst[dst_idx + 2] = src[src_idx + 2];
        }
    }

    void rotateRgbCuda(
        const uint8_t* const src,
        uint8_t* const dst,
        const int width,
        const int height,
        const int angle,
        cudaStream_t stream) {

        if (angle == 180) {
            const dim3 block(BLOCK_SIZE, BLOCK_SIZE);
            const dim3 grid((width + BLOCK_SIZE - 1) / BLOCK_SIZE,
                            (height + BLOCK_SIZE - 1) / BLOCK_SIZE);
            rotateRgbKernel<<<grid, block, 0, stream>>>(src, dst, width, height, angle);
            LFS_CUDA_LAUNCH_CHECK(stream, "io.video.rotate_rgb");
        } else {
            // 90/270: swapped dimensions for output
            const int h = width;
            const int w = height;
            const dim3 block(BLOCK_SIZE, BLOCK_SIZE);
            const dim3 grid((w + BLOCK_SIZE - 1) / BLOCK_SIZE,
                            (h + BLOCK_SIZE - 1) / BLOCK_SIZE);
            rotateRgbKernel<<<grid, block, 0, stream>>>(src, dst, width, height, angle);
            LFS_CUDA_LAUNCH_CHECK(stream, "io.video.rotate_rgb");
        }
    }

} // namespace lfs::io::video

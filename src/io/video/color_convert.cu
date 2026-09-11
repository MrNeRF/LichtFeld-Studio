/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "color_convert.cuh"

#include "core/cuda_error.hpp"

namespace lfs::io::video {

    namespace {

        // BT.601 coefficients (scaled for integer math)
        constexpr int KR_Y = 66;
        constexpr int KG_Y = 129;
        constexpr int KB_Y = 25;
        constexpr int KR_U = -38;
        constexpr int KG_U = -74;
        constexpr int KB_U = 112;
        constexpr int KR_V = 112;
        constexpr int KG_V = -94;
        constexpr int KB_V = -18;

        constexpr int BLOCK_SIZE = 16;

        __device__ __forceinline__ int floatToU8(const float val) {
            return static_cast<int>(__saturatef(val) * 255.0f + 0.5f);
        }

        __device__ __forceinline__ uint8_t clampU8(const int val) {
            return static_cast<uint8_t>(min(max(val, 0), 255));
        }

    } // namespace

    __global__ void rgbToNv12Kernel(
        const float* __restrict__ rgb,
        uint8_t* __restrict__ y_plane,
        uint8_t* __restrict__ uv_plane,
        const int width,
        const int height,
        const int y_pitch,
        const int uv_pitch) {

        const int x = blockIdx.x * blockDim.x + threadIdx.x;
        const int y = blockIdx.y * blockDim.y + threadIdx.y;

        if (x >= width || y >= height)
            return;

        const int rgb_idx = (y * width + x) * 3;
        const int r = floatToU8(rgb[rgb_idx]);
        const int g = floatToU8(rgb[rgb_idx + 1]);
        const int b = floatToU8(rgb[rgb_idx + 2]);

        // Y plane
        const int y_val = (KR_Y * r + KG_Y * g + KB_Y * b + 128) >> 8;
        y_plane[y * y_pitch + x] = clampU8(y_val + 16);

        // UV plane - half resolution, process even pixels only
        if ((x & 1) == 0 && (y & 1) == 0) {
            int r_sum = r, g_sum = g, b_sum = b;
            int count = 1;

            // 2x2 block averaging for chroma subsampling
            if (x + 1 < width) {
                const int idx = rgb_idx + 3;
                r_sum += floatToU8(rgb[idx]);
                g_sum += floatToU8(rgb[idx + 1]);
                b_sum += floatToU8(rgb[idx + 2]);
                ++count;
            }
            if (y + 1 < height) {
                const int idx = rgb_idx + width * 3;
                r_sum += floatToU8(rgb[idx]);
                g_sum += floatToU8(rgb[idx + 1]);
                b_sum += floatToU8(rgb[idx + 2]);
                ++count;
            }
            if (x + 1 < width && y + 1 < height) {
                const int idx = rgb_idx + width * 3 + 3;
                r_sum += floatToU8(rgb[idx]);
                g_sum += floatToU8(rgb[idx + 1]);
                b_sum += floatToU8(rgb[idx + 2]);
                ++count;
            }

            const int r_avg = r_sum / count;
            const int g_avg = g_sum / count;
            const int b_avg = b_sum / count;

            const int u = ((KR_U * r_avg + KG_U * g_avg + KB_U * b_avg + 128) >> 8) + 128;
            const int v = ((KR_V * r_avg + KG_V * g_avg + KB_V * b_avg + 128) >> 8) + 128;

            const int uv_idx = (y / 2) * uv_pitch + x;
            uv_plane[uv_idx] = clampU8(u);
            uv_plane[uv_idx + 1] = clampU8(v);
        }
    }

    __global__ void rgbToYuv420pKernel(
        const float* __restrict__ rgb,
        uint8_t* __restrict__ y_plane,
        uint8_t* __restrict__ u_plane,
        uint8_t* __restrict__ v_plane,
        const int width,
        const int height) {

        const int x = blockIdx.x * blockDim.x + threadIdx.x;
        const int y = blockIdx.y * blockDim.y + threadIdx.y;

        if (x >= width || y >= height)
            return;

        const int rgb_idx = (y * width + x) * 3;
        const int r = floatToU8(rgb[rgb_idx]);
        const int g = floatToU8(rgb[rgb_idx + 1]);
        const int b = floatToU8(rgb[rgb_idx + 2]);

        // Y plane
        const int y_val = (KR_Y * r + KG_Y * g + KB_Y * b + 128) >> 8;
        y_plane[y * width + x] = clampU8(y_val + 16);

        // UV planes - half resolution
        if ((x & 1) == 0 && (y & 1) == 0) {
            int r_sum = r, g_sum = g, b_sum = b;
            int count = 1;

            if (x + 1 < width) {
                const int idx = rgb_idx + 3;
                r_sum += floatToU8(rgb[idx]);
                g_sum += floatToU8(rgb[idx + 1]);
                b_sum += floatToU8(rgb[idx + 2]);
                ++count;
            }
            if (y + 1 < height) {
                const int idx = rgb_idx + width * 3;
                r_sum += floatToU8(rgb[idx]);
                g_sum += floatToU8(rgb[idx + 1]);
                b_sum += floatToU8(rgb[idx + 2]);
                ++count;
            }
            if (x + 1 < width && y + 1 < height) {
                const int idx = rgb_idx + width * 3 + 3;
                r_sum += floatToU8(rgb[idx]);
                g_sum += floatToU8(rgb[idx + 1]);
                b_sum += floatToU8(rgb[idx + 2]);
                ++count;
            }

            const int r_avg = r_sum / count;
            const int g_avg = g_sum / count;
            const int b_avg = b_sum / count;

            const int u = ((KR_U * r_avg + KG_U * g_avg + KB_U * b_avg + 128) >> 8) + 128;
            const int v = ((KR_V * r_avg + KG_V * g_avg + KB_V * b_avg + 128) >> 8) + 128;

            const int uv_x = x >> 1;
            const int uv_y = y >> 1;
            const int uv_width = width >> 1;

            u_plane[uv_y * uv_width + uv_x] = clampU8(u);
            v_plane[uv_y * uv_width + uv_x] = clampU8(v);
        }
    }

    void rgbToNv12Cuda(
        const float* const rgb_src,
        uint8_t* const y_dst,
        uint8_t* const uv_dst,
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

        rgbToNv12Kernel<<<grid, block, 0, stream>>>(
            rgb_src, y_dst, uv_dst, width, height, effective_y_pitch, effective_uv_pitch);
        LFS_CUDA_LAUNCH_CHECK(stream, "io.video.rgb_to_nv12");
    }

    void rgbToYuv420pCuda(
        const float* const rgb_src,
        uint8_t* const y_dst,
        uint8_t* const u_dst,
        uint8_t* const v_dst,
        const int width,
        const int height,
        cudaStream_t stream) {

        const dim3 block(BLOCK_SIZE, BLOCK_SIZE);
        const dim3 grid((width + BLOCK_SIZE - 1) / BLOCK_SIZE,
                        (height + BLOCK_SIZE - 1) / BLOCK_SIZE);

        rgbToYuv420pKernel<<<grid, block, 0, stream>>>(rgb_src, y_dst, u_dst, v_dst, width, height);
        LFS_CUDA_LAUNCH_CHECK(stream, "io.video.rgb_to_yuv420p");
    }

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

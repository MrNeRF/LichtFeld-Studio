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

    template <bool FullRange>
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
        const int c = FullRange ? y_val : y_val - 16;
        const int d = u - 128;
        const int e = v - 128;

        const int r = FullRange ? (256 * c + 359 * e + 128) >> 8 : (298 * c + 409 * e + 128) >> 8;
        const int g = FullRange ? (256 * c - 88 * d - 183 * e + 128) >> 8 : (298 * c - 100 * d - 208 * e + 128) >> 8;
        const int b = FullRange ? (256 * c + 454 * d + 128) >> 8 : (298 * c + 516 * d + 128) >> 8;

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
        cudaStream_t stream,
        const bool full_range) {

        const int effective_y_pitch = (y_pitch > 0) ? y_pitch : width;
        const int effective_uv_pitch = (uv_pitch > 0) ? uv_pitch : width;

        const dim3 block(BLOCK_SIZE, BLOCK_SIZE);
        const dim3 grid((width + BLOCK_SIZE - 1) / BLOCK_SIZE,
                        (height + BLOCK_SIZE - 1) / BLOCK_SIZE);

        if (full_range) {
            nv12ToRgbKernel<true><<<grid, block, 0, stream>>>(
                y_src, uv_src, rgb_dst, width, height, effective_y_pitch, effective_uv_pitch);
            LFS_CUDA_LAUNCH_CHECK(stream, "io.video.nv12_to_rgb");
        } else {
            nv12ToRgbKernel<false><<<grid, block, 0, stream>>>(
                y_src, uv_src, rgb_dst, width, height, effective_y_pitch, effective_uv_pitch);
            LFS_CUDA_LAUNCH_CHECK(stream, "io.video.nv12_to_rgb");
        }
    }

    template <bool P010>
    __global__ void semiplanarToRgbKernel(const uint8_t* y_plane, const uint8_t* uv_plane,
                                          uint8_t* rgb, int width, int height,
                                          int y_pitch, int uv_pitch, core::color::VideoColorParameters color) {
        const int x = blockIdx.x * blockDim.x + threadIdx.x;
        const int y = blockIdx.y * blockDim.y + threadIdx.y;
        if (x >= width || y >= height)
            return;
        const auto* y_row = y_plane + static_cast<size_t>(y) * y_pitch;
        const int uv_x = (x / 2) * 2;
        float code[3];
        if constexpr (P010)
            code[0] = reinterpret_cast<const uint16_t*>(y_row)[x] >> 6;
        else
            code[0] = y_row[x];
        // Match the generic unscaled RGB8 swscale path: paired horizontal
        // samples and pixel-center vertical chroma interpolation.
        const float sy = y * color.chroma_y_scale + color.chroma_y_offset;
        const int first = static_cast<int>(floorf(sy));
        const float weight = sy - first;
        const int rows = (height + 1) / 2;
        const auto* a = uv_plane + static_cast<size_t>(max(0, min(rows - 1, first))) * uv_pitch;
        const auto* b = uv_plane + static_cast<size_t>(max(0, min(rows - 1, first + 1))) * uv_pitch;
        for (int channel = 1; channel < 3; ++channel) {
            float first_code, second_code;
            if constexpr (P010) {
                first_code = reinterpret_cast<const uint16_t*>(a)[uv_x + channel - 1] >> 6;
                second_code = reinterpret_cast<const uint16_t*>(b)[uv_x + channel - 1] >> 6;
            } else {
                first_code = a[uv_x + channel - 1];
                second_code = b[uv_x + channel - 1];
            }
            code[channel] = first_code + (second_code - first_code) * weight;
        }
        for (int channel = 0; channel < 3; ++channel)
            code[channel] = code[channel] * color.scale[channel] - color.bias[channel];
        const size_t pixel = (static_cast<size_t>(y) * width + x) * 3;
        for (int channel = 0; channel < 3; ++channel) {
            const float value = color.decode[channel * 3] * code[0] +
                                color.decode[channel * 3 + 1] * code[1] +
                                color.decode[channel * 3 + 2] * code[2];
            rgb[pixel + channel] = clampU8(static_cast<int>(floorf(value * 255.f + .5f)));
        }
    }
    void semiplanarToRgbCuda(const uint8_t* y, const uint8_t* uv, uint8_t* rgb,
                             int width, int height, int y_pitch, int uv_pitch,
                             bool p010, const core::color::VideoColorParameters& color, cudaStream_t stream) {
        const dim3 block(BLOCK_SIZE, BLOCK_SIZE);
        const dim3 grid((width + BLOCK_SIZE - 1) / BLOCK_SIZE, (height + BLOCK_SIZE - 1) / BLOCK_SIZE);
        if (p010)
            semiplanarToRgbKernel<true><<<grid, block, 0, stream>>>(y, uv, rgb, width, height, y_pitch, uv_pitch, color);
        else
            semiplanarToRgbKernel<false><<<grid, block, 0, stream>>>(y, uv, rgb, width, height, y_pitch, uv_pitch, color);
        LFS_CUDA_LAUNCH_CHECK(stream, "io.video.semiplanar_to_rgb");
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

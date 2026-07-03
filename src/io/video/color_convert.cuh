/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include <cstdint>
#include <cuda_runtime.h>

namespace lfs::io::video {

    // Convert RGB float32 (0.0-1.0) to NV12 (Y plane + interleaved UV plane) on GPU
    // NV12 is NVENC's preferred input format
    // y_pitch and uv_pitch allow handling hardware frames with stride != width
    void rgbToNv12Cuda(
        const float* rgb_src, // GPU pointer, RGB float32 packed [height * width * 3]
        uint8_t* y_dst,       // GPU pointer, Y plane
        uint8_t* uv_dst,      // GPU pointer, UV interleaved
        int width,
        int height,
        int y_pitch = 0,  // Y plane pitch (0 = use width)
        int uv_pitch = 0, // UV plane pitch (0 = use width)
        cudaStream_t stream = nullptr);

    // Convert RGB float32 (0.0-1.0) to YUV420P (separate Y, U, V planes) on GPU
    // Fallback for x264 if NVENC unavailable
    void rgbToYuv420pCuda(
        const float* rgb_src, // GPU pointer, RGB float32
        uint8_t* y_dst,       // GPU pointer, Y plane [height * width]
        uint8_t* u_dst,       // GPU pointer, U plane [(height/2) * (width/2)]
        uint8_t* v_dst,       // GPU pointer, V plane [(height/2) * (width/2)]
        int width,
        int height,
        cudaStream_t stream = nullptr);

    // Convert NV12 (Y plane + interleaved UV plane) to RGB uint8 on GPU
    // Used for NVDEC hardware decoding output
    void nv12ToRgbCuda(
        const uint8_t* y_src,  // GPU pointer, Y plane
        const uint8_t* uv_src, // GPU pointer, UV interleaved
        uint8_t* rgb_dst,      // GPU pointer, RGB uint8 packed [height * width * 3]
        int width,
        int height,
        int y_pitch = 0,  // Y plane pitch (0 = use width)
        int uv_pitch = 0, // UV plane pitch (0 = use width)
        cudaStream_t stream = nullptr);

    // Rotate RGB uint8 image on GPU by 90/180/270 degrees
    // src and dst must be separate GPU buffers
    // For 90/270: dst must be [width * height * 3] (swapped dimensions)
    // For 180: dst must be [height * width * 3] (same as src)
    void rotateRgbCuda(
        const uint8_t* src,   // GPU pointer, RGB uint8 packed
        uint8_t* dst,         // GPU pointer, rotated output
        int width,            // source width
        int height,           // source height
        int angle,            // rotation angle: 90, 180, or 270
        cudaStream_t stream = nullptr);

} // namespace lfs::io::video

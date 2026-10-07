/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include <cstdint>
#include <cuda_runtime.h>

namespace lfs::io::video {

    // Contiguous float32 HWC RGB to planar BT.601 limited-range uint8.
    // Matches Studio's clamp, byte rounding and 2x2 chroma averaging.
    // All buffers belong to the producer's CUDA context; width/height are even.
    void rgbToYuv420pCuda(const float* rgb, uint8_t* y, uint8_t* u, uint8_t* v,
                          int width, int height, cudaStream_t stream);

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
        const uint8_t* src, // GPU pointer, RGB uint8 packed
        uint8_t* dst,       // GPU pointer, rotated output
        int width,          // source width
        int height,         // source height
        int angle,          // rotation angle: 90, 180, or 270
        cudaStream_t stream = nullptr);

} // namespace lfs::io::video

/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include <cstdint>
#include <cuda_runtime.h>

namespace lfs::io::cuda {

    // Fused kernel: uint8 HWC -> float32 CHW normalized [0,1]
    void launch_uint8_hwc_to_float32_chw(
        const uint8_t* input,
        float* output,
        size_t height,
        size_t width,
        size_t channels,
        cudaStream_t stream = nullptr);

    // Grayscale uint8 [H,W] -> float32 [H,W] normalized [0,1]
    void launch_uint8_hw_to_float32_hw(
        const uint8_t* input,
        float* output,
        size_t height,
        size_t width,
        cudaStream_t stream = nullptr);

    // Split uint8 RGBA [H,W,4] into float32 RGB [3,H,W] + float32 alpha [H,W], normalized [0,1]
    void launch_uint8_rgba_split_to_float32_rgb_and_alpha(
        const uint8_t* input,
        float* rgb_output,
        float* alpha_output,
        size_t height,
        size_t width,
        cudaStream_t stream = nullptr);

    // In-place mask inversion: mask = 1.0 - mask
    void launch_mask_invert(
        float* data,
        size_t height,
        size_t width,
        cudaStream_t stream = nullptr);

    // In-place mask thresholding: if mask >= threshold, set to 1.0
    void launch_mask_threshold(
        float* data,
        size_t height,
        size_t width,
        float threshold,
        cudaStream_t stream = nullptr);

} // namespace lfs::io::cuda

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

    // Fused kernel: uint16 HWC -> float32 CHW normalized [0,1]
    void launch_uint16_hwc_to_float32_chw(
        const uint16_t* input,
        float* output,
        size_t height,
        size_t width,
        size_t channels,
        cudaStream_t stream = nullptr);

    // Fused kernel: float32 HWC [0,1] -> uint16 HWC [0, 65535]
    void launch_float32_hwc_to_uint16_hwc(
        const float* input,
        uint16_t* output,
        size_t height,
        size_t width,
        size_t channels,
        cudaStream_t stream = nullptr);

    // Fused kernel: uint16 HWC [0, 65535] -> float32 HWC [0,1]
    void launch_uint16_hwc_to_float32_hwc(
        const uint16_t* input,
        float* output,
        size_t height,
        size_t width,
        size_t channels,
        cudaStream_t stream = nullptr);

    // Fused kernel: float32 CHW normals [-1,1] -> float32 HWC [0,1]
    void launch_normal_chw_to_jpeg2k_hwc(
        const float* input,
        float* output,
        size_t height,
        size_t width,
        cudaStream_t stream = nullptr);

    // Fused kernel: float32 HWC [0,1] -> float32 CHW normals [-1,1]
    void launch_jpeg2k_hwc_to_normal_chw(
        const float* input,
        float* output,
        size_t height,
        size_t width,
        cudaStream_t stream = nullptr);

    // Decode-side transform parameters for normal priors (v = n*0.5 + 0.5 file encoding).
    struct NormalPriorTransform {
        bool srgb = false;
        bool flip_yz = false;
        bool world_to_camera = false;
        float w2c[9] = {};
    };

    // Fused kernel: RGB HWC (uint8) -> float32 CHW normals in [-1, 1], with optional
    // sRGB->linear, Y/Z flip, and world->camera rotation applied per pixel.
    void launch_normal_prior_u8_hwc_to_float32_chw(
        const uint8_t* input,
        float* output,
        size_t height,
        size_t width,
        const NormalPriorTransform& transform,
        cudaStream_t stream = nullptr);

    // Fused kernel: RGB HWC (uint16) -> float32 CHW normals in [-1, 1].
    void launch_normal_prior_u16_hwc_to_float32_chw(
        const uint16_t* input,
        float* output,
        size_t height,
        size_t width,
        const NormalPriorTransform& transform,
        cudaStream_t stream = nullptr);

    // Fused kernel: uint8 HWC -> uint8 CHW
    void launch_uint8_hwc_to_uint8_chw(
        const uint8_t* input,
        uint8_t* output,
        size_t height,
        size_t width,
        size_t channels,
        cudaStream_t stream = nullptr);

    // Fused kernel: uint16 HWC -> uint8 CHW
    void launch_uint16_hwc_to_uint8_chw(
        const uint16_t* input,
        uint8_t* output,
        size_t height,
        size_t width,
        size_t channels,
        cudaStream_t stream = nullptr);

    // Fused kernel: float32 CHW normalized [0,1] -> uint8 CHW
    void launch_float32_chw_to_uint8_chw(
        const float* input,
        uint8_t* output,
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

    // Split uint8 RGBA [H,W,4] into uint8 RGB [3,H,W] + float32 alpha [H,W], normalized [0,1]
    void launch_uint8_rgba_split_to_uint8_rgb_and_float32_alpha(
        const uint8_t* input,
        uint8_t* rgb_output,
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

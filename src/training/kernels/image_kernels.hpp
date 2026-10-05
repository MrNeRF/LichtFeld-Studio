/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include "core/tensor.hpp"

#include <cstddef>
#include <cstdint>
#include <cuda_runtime.h>

namespace lfs::training::kernels {

    void launch_fused_canny_edge_filter_chw(
        const float* d_input_chw,
        float* d_output_hw,
        const int height,
        const int width,
        cudaStream_t stream = nullptr);
    void launch_fused_canny_edge_filter_chw(
        const uint8_t* d_input_chw,
        float* d_output_hw,
        const int height,
        const int width,
        cudaStream_t stream = nullptr);
    // Divides d_data by *d_scalar in place; the whole kernel no-ops when
    // *d_scalar <= skip_below (device-side replacement for host mean/median
    // readback + branch).
    void launch_normalize_by_device_scalar(
        float* d_data,
        const std::size_t n,
        const float* d_scalar,
        float skip_below = 0.0f,
        cudaStream_t stream = nullptr);

    // Rounds float values in [0, 1] to the nearest 8-bit level, as saving and reloading the image would.
    lfs::core::Tensor quantize_to_8bit_grid(const lfs::core::Tensor& image);
    // Clamps to [0, 1] and rounds to the nearest of `levels` + 1 evenly spaced values.
    lfs::core::Tensor quantize_to_grid(const lfs::core::Tensor& image, float levels);

    // out = rgb * alpha + background * (1 - alpha) for a CHW image (3 channels, uint8 or float in [0, 1]) and
    // an HW alpha. background_image ([3, H, W]) wins over background_color ([3]) when both are given.
    void launch_composite_over_background(
        const float* d_rgb_chw,
        const float* d_alpha_hw,
        const float* d_background_color,
        const float* d_background_image_chw,
        float* d_output_chw,
        int height,
        int width,
        cudaStream_t stream = nullptr);
    void launch_composite_over_background(
        const uint8_t* d_rgb_chw,
        const float* d_alpha_hw,
        const float* d_background_color,
        const float* d_background_image_chw,
        float* d_output_chw,
        int height,
        int width,
        cudaStream_t stream = nullptr);

    // Composites an RGB target ([3, H, W], uint8 or float) with its alpha ([H, W]) over background, a [3] colour
    // or a [3, H, W] image, so that it shows what a render with that background shows where it is transparent.
    lfs::core::Tensor composite_over_background(const lfs::core::Tensor& rgb,
                                                const lfs::core::Tensor& alpha,
                                                const lfs::core::Tensor& background);

} // namespace lfs::training::kernels

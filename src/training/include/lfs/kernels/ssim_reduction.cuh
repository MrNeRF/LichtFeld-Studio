/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once
#include <cuda_runtime.h>

namespace lfs::training::kernels {

    /**
     * @brief Fused mean reduction for SSIM map with optional valid padding
     *
     * Computes mean(ssim_map) with optional cropping (5 pixels from each side)
     * in a single optimized pass without creating an intermediate cropped tensor.
     *
     * @param ssim_map Input SSIM map [N, C, H, W]
     * @param temp_buffer Temporary buffer for partial sums (min(1024, total_pixels/256) elements)
     * @param result_buffer Device buffer for result (1 element)
     * @param N Batch size
     * @param C Number of channels
     * @param H Height
     * @param W Width
     * @param apply_valid_padding If true, crop 5 pixels from each side
     * @param stream CUDA stream
     */
    void launch_fused_ssim_mean_device(
        const float* ssim_map,
        float* temp_buffer,
        float* result_buffer,
        int N, int C, int H, int W,
        bool apply_valid_padding,
        cudaStream_t stream = nullptr);

    /**
     * @brief Reduce fused L1+SSIM loss directly to a scalar mean
     *
     * Computes mean((1-w)*abs(img1-img2) + w*(1-ssim_map)) with optional valid padding
     * without materializing an intermediate full-resolution loss map. `ssim_map`
     * is a channel-mean map with shape [N, 1, H, W].
     */
    void launch_fused_l1_ssim_mean_device(
        const float* img1,
        const float* img2,
        const float* ssim_map,
        float ssim_weight,
        float* temp_buffer,
        float* result_buffer,
        int N, int C, int H, int W,
        bool apply_valid_padding,
        cudaStream_t stream = nullptr);

    /**
     * @brief Reduce masked fused L1+SSIM directly to a normalized scalar loss
     *
     * Computes the masked numerator and denominator in one pass without a loss map.
     * `ssim_map` is a channel-mean map with shape [N, 1, H, W].
     * `temp_buffer` must provide room for 2 * min(1024, total_pixels/256) floats.
     */
    void launch_masked_fused_l1_ssim_mean_device(
        const float* img1,
        const float* img2,
        const float* ssim_map,
        const float* mask,
        float ssim_weight,
        float* temp_buffer,
        float* loss_buffer,
        float* mask_sum_buffer,
        int N, int C, int H, int W,
        cudaStream_t stream = nullptr);

} // namespace lfs::training::kernels

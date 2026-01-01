/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 * SPDX-FileCopyrightText: 2025 Youyu Chen (original Lanczos implementation)
 * SPDX-License-Identifier: GPL-3.0-or-later
 * SPDX-License-Identifier: MIT (original Lanczos implementation)
 */

#pragma once

#include "core/tensor.hpp"

namespace lfs::core {

    /**
     * High-quality Lanczos resampling on GPU
     *
     * @param input Input tensor in [H, W, C] format (uint8)
     * @param output_h Target height
     * @param output_w Target width
     * @param kernel_size Lanczos kernel size (typically 2 or 3)
     * @param cuda_stream CUDA stream for async execution
     * @return Resized tensor in [C, H, W] format (float32)
     */
    Tensor lanczos_resize(
        const Tensor& input,
        int output_h,
        int output_w,
        int kernel_size = 2,
        cudaStream_t cuda_stream = nullptr);

    /**
     * High-quality Lanczos resampling for grayscale images on GPU
     *
     * @param input Input tensor in [H, W] format (uint8)
     * @param output_h Target height
     * @param output_w Target width
     * @param kernel_size Lanczos kernel size (typically 2 or 3)
     * @param cuda_stream CUDA stream for async execution
     * @return Resized tensor in [H, W] format (float32)
     */
    Tensor lanczos_resize_grayscale(
        const Tensor& input,
        int output_h,
        int output_w,
        int kernel_size = 2,
        cudaStream_t cuda_stream = nullptr);

} // namespace lfs::core

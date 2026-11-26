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
     * @param input Input tensor in [H, W, C] format (uint8 or float32, 1 to 4 channels)
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
        cudaStream_t cuda_stream = nullptr, bool normalize_mask_alpha = false);

    /**
     * High-quality Lanczos resampling for grayscale images on GPU
     *
     * @param input Input tensor in [H, W] format (uint8, normalized to [0,1], or float32 passed through)
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
        cudaStream_t cuda_stream = nullptr,
        bool normalize_mask = false);

    /**
     * High-quality Lanczos resampling for planar 3-channel float images on GPU
     *
     * @param input Input tensor in [C, H, W] format (float32, 3 channels)
     * @param output_h Target height
     * @param output_w Target width
     * @param kernel_size Lanczos kernel size (typically 2 or 3)
     * @param cuda_stream CUDA stream for async execution
     * @return Resized tensor in [C, H, W] format (float32)
     */
    Tensor lanczos_resize_float_chw(
        const Tensor& input,
        int output_h,
        int output_w,
        int kernel_size = 2,
        cudaStream_t cuda_stream = nullptr);

    // Bilinear prior resampling excludes invalid neighbors and carries validity
    // with nearest sampling. Depth <= 0/nonfinite and normal norms < 0.5 are
    // invalid; outputs use zero sentinels and valid normals have unit length.
    Tensor resize_depth_prior(const Tensor& input, int output_h, int output_w,
                              cudaStream_t cuda_stream = nullptr);
    Tensor resize_normal_prior(const Tensor& input, int output_h, int output_w,
                               cudaStream_t cuda_stream = nullptr);

    // Fits a prior that travelled at its own size (depth [H,W], normals [3,H,W])
    // to a render of render_h x render_w: the prior resize to the size the loader
    // would have resized it to (request_h x request_w; 0 means the render's size),
    // then Lanczos if the render differs from that size. The prior resize runs on
    // cuda_stream without waiting for it; read the result on that stream.
    Tensor fit_depth_prior(const Tensor& input, int request_h, int request_w,
                           int render_h, int render_w,
                           cudaStream_t cuda_stream = nullptr);
    Tensor fit_normal_prior(const Tensor& input, int request_h, int request_w,
                            int render_h, int render_w,
                            cudaStream_t cuda_stream = nullptr);

} // namespace lfs::core

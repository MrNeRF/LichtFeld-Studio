/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include <cstddef>
#include <cstdint>
#include <cuda_runtime.h>

namespace lfs::training::kernels {

    // Computes grad_alpha[h,w] = -sum_c(grad_image[..., c, ...] * bg_color[c]).
    void launch_fused_grad_alpha(
        const float* grad_image,
        const float* bg_color,
        float* grad_alpha,
        int H, int W,
        bool is_chw_layout,
        cudaStream_t stream = nullptr);

    // Computes grad_alpha[h,w] = -sum_c(grad_image[c,h,w] * bg_image[c,h,w]).
    void launch_fused_grad_alpha_with_image(
        const float* grad_image,
        const float* bg_image,
        float* grad_alpha,
        int H, int W,
        cudaStream_t stream = nullptr);

    // Computes output[c,h,w] = image[c,h,w] + (1 - alpha[h,w]) * bg_color[c].
    void launch_fused_background_blend(
        const float* image,
        const float* alpha,
        const float* bg_color,
        float* output,
        int H, int W,
        cudaStream_t stream = nullptr);

    // Computes output[c,h,w] = image[c,h,w] + (1 - alpha[h,w]) * bg_image[c,h,w].
    void launch_fused_background_blend_with_image(
        const float* image,
        const float* alpha,
        const float* bg_image,
        float* output,
        int H, int W,
        cudaStream_t stream = nullptr);

    // In-place inverse of launch_fused_background_blend.
    void launch_fused_background_unblend(
        float* image,
        const float* alpha,
        const float* bg_color,
        int H, int W,
        cudaStream_t stream = nullptr);

    // In-place inverse of launch_fused_background_blend_with_image.
    void launch_fused_background_unblend_with_image(
        float* image,
        const float* alpha,
        const float* bg_image,
        int H, int W,
        cudaStream_t stream = nullptr);

    /**
     * @brief Add gradients from src to dst buffer (dst += src)
     *
     * Simple element-wise addition for gradient accumulation.
     * Both buffers must have the same number of elements.
     *
     * @param dst Destination gradient buffer, modified in-place
     * @param src Source gradient buffer
     * @param n_elements Total number of float elements
     * @param stream CUDA stream
     */
    void launch_grad_accumulate(
        float* dst,
        const float* src,
        int64_t n_elements,
        cudaStream_t stream = nullptr);

    /**
     * @brief Add gradients with unsqueeze (src [N] -> dst [N, 1])
     *
     * Accumulates gradients from a 1D buffer into a 2D buffer.
     * Used for opacity gradients [N] -> [N, 1].
     *
     * @param dst Destination gradient buffer [N, 1], modified in-place
     * @param src Source gradient buffer [N]
     * @param N Number of elements
     * @param stream CUDA stream
     */
    void launch_grad_accumulate_unsqueeze(
        float* dst,
        const float* src,
        int64_t N,
        cudaStream_t stream = nullptr);

    /**
     * @brief Add SH gradients into split sh0 + swizzled shN storage.
     *
     * @param dst_sh0 Destination sh0 gradient buffer [N, 1, 3], modified in-place
     * @param dst_shN Destination swizzled shN gradient buffer, modified in-place (nullable when K_src=1)
     * @param src Source canonical gradient buffer [N, K_src, 3]
     * @param N Number of Gaussians
     * @param K_src Active SH coefficients in source, including sh0
     * @param shN_layout_rest Resident swizzled shN rest coefficient count
     * @param stream CUDA stream
     */
    void launch_grad_accumulate_sh_swizzled(
        float* dst_sh0,
        float* dst_shN,
        const float* src,
        int64_t N,
        int64_t K_src,
        uint32_t shN_layout_rest,
        cudaStream_t stream = nullptr);

    /**
     * @brief Compute L2 norm of each row and add to destination buffer
     *
     * For each Gaussian, computes ||grad_means[i]||_2 and adds it to densification_info[i].
     * This avoids allocating a temporary tensor for the norm result.
     *
     * @param densification_info Destination buffer [N], modified in-place (adds norm)
     * @param grad_means Source gradient buffer [N, 3]
     * @param N Number of Gaussians
     * @param stream CUDA stream
     */
    void launch_grad_norm_accumulate(
        float* densification_info,
        const float* grad_means,
        int64_t N,
        cudaStream_t stream = nullptr);

    /**
     * @brief Bilinear resize for [C, H, W] float32 tensors
     *
     * High-quality bilinear interpolation for resizing images.
     * Used for resizing background images to match camera dimensions.
     *
     * @param src Source tensor [C, src_H, src_W]
     * @param dst Destination tensor [C, dst_H, dst_W]
     * @param C Number of channels (typically 3)
     * @param src_H Source height
     * @param src_W Source width
     * @param dst_H Destination height
     * @param dst_W Destination width
     * @param stream CUDA stream
     */
    void launch_bilinear_resize_chw(
        const float* src,
        float* dst,
        int C,
        int src_H, int src_W,
        int dst_H, int dst_W,
        cudaStream_t stream = nullptr);

    // Generate random per-pixel RGB background [3, H, W] with values in [0, 1]
    void launch_random_background(
        float* output,
        int H, int W,
        uint64_t seed,
        cudaStream_t stream = nullptr);

} // namespace lfs::training::kernels

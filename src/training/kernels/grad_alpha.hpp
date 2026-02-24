/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include <cstddef>
#include <cstdint>
#include <cuda_runtime.h>

namespace lfs::training::kernels {

    /**
     * @brief Fused kernel for computing grad_alpha in rasterization backward pass
     *
     * Computes: grad_alpha[h,w] = -sum_c(grad_image[...,c,...] * bg_color[c])
     *
     * This fuses multiply + sum + negate into a single kernel, avoiding:
     * - Multiple kernel launches (3-4 separate ops)
     * - Intermediate memory allocations
     * - Poor memory coalescing from segmented reduce
     *
     * Expected performance: 10-50x faster than separate tensor ops
     *
     * @param grad_image Input gradient [3,H,W] or [H,W,3]
     * @param bg_color Background color [3]
     * @param grad_alpha Output gradient [H,W]
     * @param H Height
     * @param W Width
     * @param is_chw_layout true if [3,H,W], false if [H,W,3]
     * @param stream CUDA stream
     */
    void launch_fused_grad_alpha(
        const float* grad_image,
        const float* bg_color,
        float* grad_alpha,
        int H, int W,
        bool is_chw_layout,
        cudaStream_t stream = nullptr);

    /**
     * @brief Fused kernel for computing grad_alpha with background image
     *
     * Computes: grad_alpha[h,w] = -sum_c(grad_image[c,h,w] * bg_image[c,h,w])
     *
     * This variant uses per-pixel background values instead of a solid color.
     *
     * @param grad_image Input gradient [3,H,W]
     * @param bg_image Background image [3,H,W]
     * @param grad_alpha Output gradient [H,W]
     * @param H Height
     * @param W Width
     * @param stream CUDA stream
     */
    void launch_fused_grad_alpha_with_image(
        const float* grad_image,
        const float* bg_image,
        float* grad_alpha,
        int H, int W,
        cudaStream_t stream = nullptr);

    /**
     * @brief Fused kernel for background blending in rasterization forward pass
     *
     * Computes: output[c,h,w] = image[c,h,w] + (1 - alpha[h,w]) * bg_color[c]
     *
     * This fuses 5 separate operations into a single kernel:
     * - alpha negation: -alpha
     * - scalar add: + 1.0
     * - bg_color broadcast multiply
     * - image + bg_contribution
     *
     * Expected performance: 5-10x faster than separate tensor ops
     *
     * @param image Raw rendered image [3,H,W]
     * @param alpha Alpha channel [1,H,W]
     * @param bg_color Background color [3]
     * @param output Output image [3,H,W]
     * @param H Height
     * @param W Width
     * @param stream CUDA stream
     */
    void launch_fused_background_blend(
        const float* image,
        const float* alpha,
        const float* bg_color,
        float* output,
        int H, int W,
        cudaStream_t stream = nullptr);

    /**
     * @brief Fused kernel for background blending with full background image
     *
     * Computes: output[c,h,w] = image[c,h,w] + (1 - alpha[h,w]) * bg_image[c,h,w]
     *
     * This variant accepts a full background image [3, H, W] instead of a solid color [3].
     * The background image must be pre-resized to match the render dimensions.
     *
     * @param image Raw rendered image [3,H,W]
     * @param alpha Alpha channel [1,H,W]
     * @param bg_image Background image [3,H,W]
     * @param output Output image [3,H,W]
     * @param H Height
     * @param W Width
     * @param stream CUDA stream
     */
    void launch_fused_background_blend_with_image(
        const float* image,
        const float* alpha,
        const float* bg_image,
        float* output,
        int H, int W,
        cudaStream_t stream = nullptr);

    /**
     * @brief Fused kernel for sigmoid backward chain rule
     *
     * Computes in-place: v_raw = v_activated * sigmoid * (1 - sigmoid)
     * where sigmoid values are already computed (activated opacities)
     *
     * @param v_opacities Gradient w.r.t. activated opacities [N], modified in-place
     * @param sigmoid Activated opacities (sigmoid output) [N]
     * @param N Number of elements
     * @param stream CUDA stream
     */
    void launch_sigmoid_backward(
        float* v_opacities,
        const float* sigmoid,
        int64_t N,
        cudaStream_t stream = nullptr);

    /**
     * @brief Fused kernel for exp backward chain rule
     *
     * Computes in-place: v_raw = v_activated * activated
     * (derivative of exp(x) is exp(x))
     *
     * @param v_scales Gradient w.r.t. activated scales [N, 3], modified in-place
     * @param scales Activated scales (exp output) [N, 3]
     * @param N Number of Gaussians
     * @param stream CUDA stream
     */
    void launch_exp_backward(
        float* v_scales,
        const float* scales,
        int64_t N,
        cudaStream_t stream = nullptr);

    /**
     * @brief Fused kernel for quaternion normalization backward
     *
     * Computes: v_raw = (v_activated - q_norm * dot(q_norm, v_activated)) / ||q_raw||
     *
     * This is the chain rule for f(q) = q / ||q||
     * The Jacobian is: (I - q_norm * q_norm^T) / ||q||
     *
     * @param v_quats Gradient w.r.t. normalized quats [N, 4], modified in-place
     * @param quats_normalized Normalized quaternions [N, 4]
     * @param quats_raw Raw (unnormalized) quaternions [N, 4]
     * @param N Number of quaternions
     * @param stream CUDA stream
     */
    void launch_quat_normalize_backward(
        float* v_quats,
        const float* quats_normalized,
        const float* quats_raw,
        int64_t N,
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
     * @brief Add SH gradients with slice (src [N, K_src, 3] -> dst0 [N, 1, 3], dst_rest [N, K_dst, 3])
     *
     * Accumulates SH coefficient gradients, splitting sh0 and shN.
     * K_src is the active SH coefficients from gsplat backward.
     * K_dst is the full buffer size (max_sh_degree^2 - 1).
     *
     * @param dst_sh0 Destination sh0 gradient buffer [N, 1, 3], modified in-place
     * @param dst_shN Destination shN gradient buffer [N, K_dst, 3], modified in-place (can be nullptr if K_src=1)
     * @param src Source gradient buffer [N, K_src, 3]
     * @param N Number of Gaussians
     * @param K_src Active SH coefficients in source
     * @param K_dst Destination buffer width (may be larger than K_src-1)
     * @param stream CUDA stream
     */
    void launch_grad_accumulate_sh(
        float* dst_sh0,
        float* dst_shN,
        const float* src,
        int64_t N,
        int64_t K_src,
        int64_t K_dst,
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
     * @brief Permute CHW to HWC layout
     *
     * Converts [C, H, W] tensor to [H, W, C] layout.
     * Used for converting grad_image from CHW to HWC format for gsplat backward.
     *
     * @param src Source tensor [C, H, W]
     * @param dst Destination tensor [H, W, C]
     * @param C Number of channels
     * @param H Height
     * @param W Width
     * @param stream CUDA stream
     */
    void launch_permute_chw_to_hwc(
        const float* src,
        float* dst,
        int C, int H, int W,
        cudaStream_t stream = nullptr);

    /**
     * @brief Permute HWC to CHW layout
     *
     * Converts [H, W, C] tensor to [C, H, W] layout.
     * Used to materialize gsplat arena outputs into reusable CHW tensors.
     *
     * @param src Source tensor [H, W, C]
     * @param dst Destination tensor [C, H, W]
     * @param C Number of channels
     * @param H Height
     * @param W Width
     * @param stream CUDA stream
     */
    void launch_permute_hwc_to_chw(
        const float* src,
        float* dst,
        int C, int H, int W,
        cudaStream_t stream = nullptr);

    /**
     * @brief Squeeze 1HW to HW
     *
     * Removes leading dimension of 1: [1, H, W] -> [H, W].
     * Just a memory copy since layout is identical.
     *
     * @param src Source tensor [1, H, W]
     * @param dst Destination tensor [H, W]
     * @param H Height
     * @param W Width
     * @param stream CUDA stream
     */
    void launch_squeeze_1hw_to_hw(
        const float* src,
        float* dst,
        int H, int W,
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

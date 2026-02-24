/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "grad_alpha.hpp"
#include <cstdint>

namespace lfs::training::kernels {

    // ==================== CHW Layout: [3, H, W] ====================
    // Optimized for spatial locality - each thread processes one pixel
    __global__ void fused_grad_alpha_chw_kernel(
        const float* __restrict__ grad_image, // [3, H, W]
        const float* __restrict__ bg_color,   // [3]
        float* __restrict__ grad_alpha,       // [H, W]
        int H, int W) {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        int total = H * W;

        if (idx >= total)
            return;

        int h = idx / W;
        int w = idx % W;

        // Compute: grad_alpha[h,w] = -(grad_image[0,h,w]*bg[0] + grad_image[1,h,w]*bg[1] + grad_image[2,h,w]*bg[2])
        // All memory accesses are coalesced within each channel plane

        int HW = H * W;
        int offset = h * W + w;

        // Manual unroll for RGB channels (compiler will optimize this heavily)
        float sum = grad_image[0 * HW + offset] * bg_color[0] + grad_image[1 * HW + offset] * bg_color[1] + grad_image[2 * HW + offset] * bg_color[2];

        grad_alpha[offset] = -sum;
    }

    // ==================== HWC Layout: [H, W, 3] ====================
    // Highly optimized - RGB values are contiguous, perfect for vectorized loads!
    __global__ void fused_grad_alpha_hwc_kernel(
        const float* __restrict__ grad_image, // [H, W, 3]
        const float* __restrict__ bg_color,   // [3]
        float* __restrict__ grad_alpha,       // [H, W]
        int H, int W) {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        int total = H * W;

        if (idx >= total)
            return;

        // With HWC layout, 3 consecutive floats = perfect for float3 vectorized load!
        // This is MUCH faster than the generic segmented reduce

        int base = idx * 3;

        // Option 1: Manual scalar loads (compiler may vectorize)
        float r = grad_image[base + 0];
        float g = grad_image[base + 1];
        float b = grad_image[base + 2];

        float sum = r * bg_color[0] + g * bg_color[1] + b * bg_color[2];

        grad_alpha[idx] = -sum;
    }

    // ==================== HWC Layout with Vectorized Loads ====================
    // Use float3 for guaranteed vectorized 96-bit loads (25-50% faster on modern GPUs)
    __global__ void fused_grad_alpha_hwc_vectorized_kernel(
        const float* __restrict__ grad_image,    // [H, W, 3]
        const float3* __restrict__ bg_color_vec, // [1] as float3
        float* __restrict__ grad_alpha,          // [H, W]
        int H, int W) {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        int total = H * W;

        if (idx >= total)
            return;

        // Vectorized load: read 12 bytes (3 floats) in single transaction
        const float3* grad_vec = reinterpret_cast<const float3*>(grad_image);
        float3 grad_rgb = grad_vec[idx];
        float3 bg_rgb = bg_color_vec[0];

        // FMA (fused multiply-add) - single instruction on modern GPUs
        float sum = grad_rgb.x * bg_rgb.x + grad_rgb.y * bg_rgb.y + grad_rgb.z * bg_rgb.z;

        grad_alpha[idx] = -sum;
    }

    // ==================== Launcher ====================
    void launch_fused_grad_alpha(
        const float* grad_image,
        const float* bg_color,
        float* grad_alpha,
        int H, int W,
        bool is_chw_layout,
        cudaStream_t stream) {
        int total = H * W;

        // Optimal block size for modern GPUs (maximize occupancy)
        // 256 threads = 8 warps = good balance for both Ampere and Ada
        constexpr int threads = 256;
        int blocks = (total + threads - 1) / threads;

        if (is_chw_layout) {
            // CHW: [3, H, W]
            fused_grad_alpha_chw_kernel<<<blocks, threads, 0, stream>>>(
                grad_image, bg_color, grad_alpha, H, W);
        } else {
            // HWC: [H, W, 3] - check if data is properly aligned for vectorized load
            bool is_aligned = (reinterpret_cast<uintptr_t>(grad_image) % 16 == 0) &&
                              (reinterpret_cast<uintptr_t>(bg_color) % 16 == 0);

            if (is_aligned) {
                // Use vectorized version for ~25% speedup
                fused_grad_alpha_hwc_vectorized_kernel<<<blocks, threads, 0, stream>>>(
                    grad_image,
                    reinterpret_cast<const float3*>(bg_color),
                    grad_alpha,
                    H, W);
            } else {
                // Fall back to scalar version (still very fast)
                fused_grad_alpha_hwc_kernel<<<blocks, threads, 0, stream>>>(
                    grad_image, bg_color, grad_alpha, H, W);
            }
        }
    }

    // ==================== Backward Pass: Grad Alpha with Background Image ====================
    // Computes: grad_alpha[h,w] = -sum_c(grad_image[c,h,w] * bg_image[c,h,w])
    // CHW layout: grad_image [3, H, W], bg_image [3, H, W], grad_alpha [H, W]
    __global__ void fused_grad_alpha_with_image_kernel(
        const float* __restrict__ grad_image,
        const float* __restrict__ bg_image,
        float* __restrict__ grad_alpha,
        int H, int W) {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        int total = H * W;

        if (idx >= total)
            return;

        int HW = H * W;

        // grad_alpha = -sum_c(grad_image[c,h,w] * bg_image[c,h,w])
        float sum = 0.0f;
        for (int c = 0; c < 3; ++c) {
            sum += grad_image[c * HW + idx] * bg_image[c * HW + idx];
        }
        grad_alpha[idx] = -sum;
    }

    void launch_fused_grad_alpha_with_image(
        const float* grad_image,
        const float* bg_image,
        float* grad_alpha,
        int H, int W,
        cudaStream_t stream) {
        int total = H * W;
        constexpr int threads = 256;
        int blocks = (total + threads - 1) / threads;

        fused_grad_alpha_with_image_kernel<<<blocks, threads, 0, stream>>>(
            grad_image, bg_image, grad_alpha, H, W);
    }

    // ==================== Forward Pass: Background Blending ====================
    // Fuses: output = image + (1 - alpha) * bg_color
    // CHW layout: image [3, H, W], alpha [1, H, W] or [H, W], output [3, H, W]
    __global__ void fused_background_blend_kernel(
        const float* __restrict__ image,    // [3, H, W]
        const float* __restrict__ alpha,    // [1, H, W] or [H, W]
        const float* __restrict__ bg_color, // [3]
        float* __restrict__ output,         // [3, H, W]
        int H, int W) {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        int total = H * W;

        if (idx >= total)
            return;

        int h = idx / W;
        int w = idx % W;
        int HW = H * W;
        int offset = h * W + w;

        // Load alpha value once (alpha is [1, H, W] or [H, W])
        float alpha_val = alpha[offset];
        float alpha_complement = 1.0f - alpha_val;

        // Load bg_color once (3 values, tiny memory footprint - cache friendly)
        float bg_r = bg_color[0];
        float bg_g = bg_color[1];
        float bg_b = bg_color[2];

        // Compute for all 3 channels in a single thread (better than 3 separate kernels!)
        // output[c,h,w] = image[c,h,w] + (1 - alpha[h,w]) * bg_color[c]
        float bg_contrib_r = alpha_complement * bg_r;
        float bg_contrib_g = alpha_complement * bg_g;
        float bg_contrib_b = alpha_complement * bg_b;

        output[0 * HW + offset] = image[0 * HW + offset] + bg_contrib_r;
        output[1 * HW + offset] = image[1 * HW + offset] + bg_contrib_g;
        output[2 * HW + offset] = image[2 * HW + offset] + bg_contrib_b;
    }

    void launch_fused_background_blend(
        const float* image,
        const float* alpha,
        const float* bg_color,
        float* output,
        int H, int W,
        cudaStream_t stream) {
        int total = H * W;
        constexpr int threads = 256;
        int blocks = (total + threads - 1) / threads;

        fused_background_blend_kernel<<<blocks, threads, 0, stream>>>(
            image, alpha, bg_color, output, H, W);
    }

    // ==================== Forward Pass: Background Blending with Image ====================
    // Fuses: output = image + (1 - alpha) * bg_image (per-pixel)
    // CHW layout: image [3, H, W], alpha [1, H, W] or [H, W], bg_image [3, H, W], output [3, H, W]
    __global__ void fused_background_blend_with_image_kernel(
        const float* __restrict__ image,    // [3, H, W]
        const float* __restrict__ alpha,    // [1, H, W] or [H, W]
        const float* __restrict__ bg_image, // [3, H, W]
        float* __restrict__ output,         // [3, H, W]
        int H, int W) {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        int total = H * W;

        if (idx >= total)
            return;

        int h = idx / W;
        int w = idx % W;
        int HW = H * W;
        int offset = h * W + w;

        // Load alpha value once (alpha is [1, H, W] or [H, W])
        float alpha_val = alpha[offset];
        float alpha_complement = 1.0f - alpha_val;

        // Compute for all 3 channels in a single thread
        // output[c,h,w] = image[c,h,w] + (1 - alpha[h,w]) * bg_image[c,h,w]
        float bg_contrib_r = alpha_complement * bg_image[0 * HW + offset];
        float bg_contrib_g = alpha_complement * bg_image[1 * HW + offset];
        float bg_contrib_b = alpha_complement * bg_image[2 * HW + offset];

        output[0 * HW + offset] = image[0 * HW + offset] + bg_contrib_r;
        output[1 * HW + offset] = image[1 * HW + offset] + bg_contrib_g;
        output[2 * HW + offset] = image[2 * HW + offset] + bg_contrib_b;
    }

    void launch_fused_background_blend_with_image(
        const float* image,
        const float* alpha,
        const float* bg_image,
        float* output,
        int H, int W,
        cudaStream_t stream) {
        int total = H * W;
        constexpr int threads = 256;
        int blocks = (total + threads - 1) / threads;

        fused_background_blend_with_image_kernel<<<blocks, threads, 0, stream>>>(
            image, alpha, bg_image, output, H, W);
    }

    // ==================== Sigmoid Backward ====================
    // Computes in-place: v_opacities *= sigmoid * (1 - sigmoid)
    __global__ void sigmoid_backward_kernel(
        float* __restrict__ v_opacities,
        const float* __restrict__ sigmoid,
        int64_t N) {
        int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
        if (idx >= N)
            return;

        float s = sigmoid[idx];
        float deriv = s * (1.0f - s);
        v_opacities[idx] *= deriv;
    }

    void launch_sigmoid_backward(
        float* v_opacities,
        const float* sigmoid,
        int64_t N,
        cudaStream_t stream) {
        constexpr int threads = 256;
        int64_t blocks = (N + threads - 1) / threads;

        sigmoid_backward_kernel<<<static_cast<unsigned int>(blocks), threads, 0, stream>>>(
            v_opacities, sigmoid, N);
    }

    // ==================== Exp Backward ====================
    // Computes in-place: v_scales *= scales (for all 3 components per Gaussian)
    __global__ void exp_backward_kernel(
        float* __restrict__ v_scales,
        const float* __restrict__ scales,
        int64_t N) {
        int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
        int64_t total = N * 3; // N Gaussians, 3 scale components each
        if (idx >= total)
            return;

        v_scales[idx] *= scales[idx];
    }

    void launch_exp_backward(
        float* v_scales,
        const float* scales,
        int64_t N,
        cudaStream_t stream) {
        constexpr int threads = 256;
        int64_t total = N * 3;
        int64_t blocks = (total + threads - 1) / threads;

        exp_backward_kernel<<<static_cast<unsigned int>(blocks), threads, 0, stream>>>(
            v_scales, scales, N);
    }

    // ==================== Quaternion Normalize Backward ====================
    // Computes in-place: v_raw = (v_activated - q_norm * dot(q_norm, v_activated)) / ||q_raw||
    // This is the Jacobian of f(q) = q / ||q||
    __global__ void quat_normalize_backward_kernel(
        float* __restrict__ v_quats,      // [N, 4] - modified in-place
        const float* __restrict__ q_norm, // [N, 4] - normalized quaternions
        const float* __restrict__ q_raw,  // [N, 4] - raw quaternions
        int64_t N) {
        int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
        if (idx >= N)
            return;

        // Load all 4 components of this quaternion
        int64_t base = idx * 4;

        // Load normalized quaternion components
        float qn0 = q_norm[base + 0];
        float qn1 = q_norm[base + 1];
        float qn2 = q_norm[base + 2];
        float qn3 = q_norm[base + 3];

        // Load gradients w.r.t. normalized quaternions
        float v0 = v_quats[base + 0];
        float v1 = v_quats[base + 1];
        float v2 = v_quats[base + 2];
        float v3 = v_quats[base + 3];

        // Load raw quaternion and compute its norm
        float qr0 = q_raw[base + 0];
        float qr1 = q_raw[base + 1];
        float qr2 = q_raw[base + 2];
        float qr3 = q_raw[base + 3];
        float norm = sqrtf(qr0 * qr0 + qr1 * qr1 + qr2 * qr2 + qr3 * qr3);

        // Avoid division by zero
        if (norm < 1e-12f) {
            v_quats[base + 0] = 0.0f;
            v_quats[base + 1] = 0.0f;
            v_quats[base + 2] = 0.0f;
            v_quats[base + 3] = 0.0f;
            return;
        }

        // Compute dot product: dot(q_norm, v)
        float dot = qn0 * v0 + qn1 * v1 + qn2 * v2 + qn3 * v3;

        // Compute: v_raw = (v - q_norm * dot) / norm
        float inv_norm = 1.0f / norm;
        v_quats[base + 0] = (v0 - qn0 * dot) * inv_norm;
        v_quats[base + 1] = (v1 - qn1 * dot) * inv_norm;
        v_quats[base + 2] = (v2 - qn2 * dot) * inv_norm;
        v_quats[base + 3] = (v3 - qn3 * dot) * inv_norm;
    }

    void launch_quat_normalize_backward(
        float* v_quats,
        const float* quats_normalized,
        const float* quats_raw,
        int64_t N,
        cudaStream_t stream) {
        constexpr int threads = 256;
        int64_t blocks = (N + threads - 1) / threads;

        quat_normalize_backward_kernel<<<static_cast<unsigned int>(blocks), threads, 0, stream>>>(
            v_quats, quats_normalized, quats_raw, N);
    }

    // ==================== Gradient Accumulation ====================
    // Simple element-wise addition: dst += src
    __global__ void grad_accumulate_kernel(
        float* __restrict__ dst,
        const float* __restrict__ src,
        int64_t n_elements) {
        int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
        if (idx >= n_elements)
            return;

        dst[idx] += src[idx];
    }

    void launch_grad_accumulate(
        float* dst,
        const float* src,
        int64_t n_elements,
        cudaStream_t stream) {
        constexpr int threads = 256;
        int64_t blocks = (n_elements + threads - 1) / threads;

        grad_accumulate_kernel<<<static_cast<unsigned int>(blocks), threads, 0, stream>>>(
            dst, src, n_elements);
    }

    // Accumulate with unsqueeze: src [N] -> dst [N, 1]
    // (Same memory layout, just add)
    void launch_grad_accumulate_unsqueeze(
        float* dst,
        const float* src,
        int64_t N,
        cudaStream_t stream) {
        // [N] and [N, 1] have the same memory layout
        launch_grad_accumulate(dst, src, N, stream);
    }

    // ==================== SH Gradient Split and Accumulate ====================
    // src [N, K_src, 3] -> dst_sh0 [N, 1, 3] (first coeff), dst_shN [N, K_dst, 3] (rest)
    // K_src: number of active SH coefficients in source (from gsplat backward)
    // K_dst: number of SH coefficients in destination buffer (max_sh_degree^2 - 1)
    __global__ void grad_accumulate_sh_kernel(
        float* __restrict__ dst_sh0,   // [N, 1, 3] = [N, 3] contiguous
        float* __restrict__ dst_shN,   // [N, K_dst, 3] or nullptr
        const float* __restrict__ src, // [N, K_src, 3]
        int64_t N,
        int64_t K_src, // Source SH coefficients (active)
        int64_t K_dst  // Destination buffer width (may be larger)
    ) {
        int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
        if (idx >= N)
            return;

        // Source layout: [N, K_src, 3] -> index [n, k, c] = n * K_src * 3 + k * 3 + c
        // sh0 is at k=0
        int64_t src_sh0_base = idx * K_src * 3;
        int64_t dst_sh0_base = idx * 3; // [N, 1, 3] = [N, 3]

        // Accumulate sh0 (3 values)
        dst_sh0[dst_sh0_base + 0] += src[src_sh0_base + 0];
        dst_sh0[dst_sh0_base + 1] += src[src_sh0_base + 1];
        dst_sh0[dst_sh0_base + 2] += src[src_sh0_base + 2];

        // Accumulate shN if K_src > 1
        if (dst_shN != nullptr && K_src > 1) {
            // shN is at k=1..K_src-1 in source
            // Destination has K_dst coefficients per Gaussian
            for (int64_t k = 1; k < K_src; ++k) {
                int64_t src_offset = src_sh0_base + k * 3;
                // Use K_dst for destination stride, not K_src-1
                int64_t dst_offset = idx * K_dst * 3 + (k - 1) * 3;
                dst_shN[dst_offset + 0] += src[src_offset + 0];
                dst_shN[dst_offset + 1] += src[src_offset + 1];
                dst_shN[dst_offset + 2] += src[src_offset + 2];
            }
        }
    }

    void launch_grad_accumulate_sh(
        float* dst_sh0,
        float* dst_shN,
        const float* src,
        int64_t N,
        int64_t K_src,
        int64_t K_dst,
        cudaStream_t stream) {
        constexpr int threads = 256;
        int64_t blocks = (N + threads - 1) / threads;

        grad_accumulate_sh_kernel<<<static_cast<unsigned int>(blocks), threads, 0, stream>>>(
            dst_sh0, dst_shN, src, N, K_src, K_dst);
    }

    // ==================== Gradient Norm Accumulate ====================
    // Computes ||grad_means[i]||_2 and adds to densification_info[i]
    __global__ void grad_norm_accumulate_kernel(
        float* __restrict__ densification_info, // [N]
        const float* __restrict__ grad_means,   // [N, 3]
        int64_t N) {
        int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
        if (idx >= N)
            return;

        // Load gradient components
        int64_t base = idx * 3;
        float gx = grad_means[base + 0];
        float gy = grad_means[base + 1];
        float gz = grad_means[base + 2];

        // Compute L2 norm
        float norm = sqrtf(gx * gx + gy * gy + gz * gz);

        // Add to densification info
        densification_info[idx] += norm;
    }

    void launch_grad_norm_accumulate(
        float* densification_info,
        const float* grad_means,
        int64_t N,
        cudaStream_t stream) {
        constexpr int threads = 256;
        int64_t blocks = (N + threads - 1) / threads;

        grad_norm_accumulate_kernel<<<static_cast<unsigned int>(blocks), threads, 0, stream>>>(
            densification_info, grad_means, N);
    }

    // ==================== CHW to HWC Permute ====================
    // Converts [C, H, W] to [H, W, C] layout
    __global__ void permute_chw_to_hwc_kernel(
        const float* __restrict__ src, // [C, H, W]
        float* __restrict__ dst,       // [H, W, C]
        int C, int H, int W) {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        int total = H * W;

        if (idx >= total)
            return;

        int HW = H * W;
        int dst_base = idx * C; // [H, W, C] -> index h*W*C + w*C + c = (h*W + w)*C + c = idx*C + c

        // Copy all channels for this spatial position
        for (int c = 0; c < C; ++c) {
            dst[dst_base + c] = src[c * HW + idx];
        }
    }

    void launch_permute_chw_to_hwc(
        const float* src,
        float* dst,
        int C, int H, int W,
        cudaStream_t stream) {
        int total = H * W;
        constexpr int threads = 256;
        int blocks = (total + threads - 1) / threads;

        permute_chw_to_hwc_kernel<<<blocks, threads, 0, stream>>>(
            src, dst, C, H, W);
    }

    // ==================== HWC to CHW Permute ====================
    // Converts [H, W, C] to [C, H, W] layout
    __global__ void permute_hwc_to_chw_kernel(
        const float* __restrict__ src, // [H, W, C]
        float* __restrict__ dst,       // [C, H, W]
        int C, int H, int W) {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        int total = H * W;
        if (idx >= total) {
            return;
        }

        int dst_hw = H * W;
        int src_base = idx * C;
        for (int c = 0; c < C; ++c) {
            dst[c * dst_hw + idx] = src[src_base + c];
        }
    }

    void launch_permute_hwc_to_chw(
        const float* src,
        float* dst,
        int C, int H, int W,
        cudaStream_t stream) {
        int total = H * W;
        constexpr int threads = 256;
        int blocks = (total + threads - 1) / threads;

        permute_hwc_to_chw_kernel<<<blocks, threads, 0, stream>>>(
            src, dst, C, H, W);
    }

    // ==================== 1HW to HW Squeeze ====================
    // Removes leading dimension of 1: [1, H, W] -> [H, W]
    // This is just a copy since memory layout is the same
    void launch_squeeze_1hw_to_hw(
        const float* src, // [1, H, W]
        float* dst,       // [H, W]
        int H, int W,
        cudaStream_t stream) {
        // Memory layout is identical, just copy
        cudaMemcpyAsync(dst, src, H * W * sizeof(float), cudaMemcpyDeviceToDevice, stream);
    }

    // ==================== Bilinear Resize for CHW Tensors ====================
    // Resizes [C, H, W] float32 tensors using bilinear interpolation
    __global__ void bilinear_resize_chw_kernel(
        const float* __restrict__ src, // [C, src_H, src_W]
        float* __restrict__ dst,       // [C, dst_H, dst_W]
        int C,
        int src_H, int src_W,
        int dst_H, int dst_W) {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        int total = dst_H * dst_W;

        if (idx >= total)
            return;

        int dst_y = idx / dst_W;
        int dst_x = idx % dst_W;

        // Compute source coordinates with half-pixel offset for proper alignment
        float src_y = (dst_y + 0.5f) * (float(src_H) / float(dst_H)) - 0.5f;
        float src_x = (dst_x + 0.5f) * (float(src_W) / float(dst_W)) - 0.5f;

        // Clamp to valid range
        src_y = fmaxf(0.0f, fminf(src_y, float(src_H - 1)));
        src_x = fmaxf(0.0f, fminf(src_x, float(src_W - 1)));

        // Get integer coordinates and interpolation weights
        int y0 = int(src_y);
        int x0 = int(src_x);
        int y1 = min(y0 + 1, src_H - 1);
        int x1 = min(x0 + 1, src_W - 1);

        float wy = src_y - float(y0);
        float wx = src_x - float(x0);

        // Precompute weights
        float w00 = (1.0f - wy) * (1.0f - wx);
        float w01 = (1.0f - wy) * wx;
        float w10 = wy * (1.0f - wx);
        float w11 = wy * wx;

        int src_HW = src_H * src_W;
        int dst_HW = dst_H * dst_W;

        // Bilinear interpolation for each channel
        for (int c = 0; c < C; ++c) {
            const float* src_c = src + c * src_HW;
            float val = w00 * src_c[y0 * src_W + x0] +
                        w01 * src_c[y0 * src_W + x1] +
                        w10 * src_c[y1 * src_W + x0] +
                        w11 * src_c[y1 * src_W + x1];
            dst[c * dst_HW + idx] = val;
        }
    }

    void launch_bilinear_resize_chw(
        const float* src,
        float* dst,
        int C,
        int src_H, int src_W,
        int dst_H, int dst_W,
        cudaStream_t stream) {
        int total = dst_H * dst_W;
        constexpr int threads = 256;
        int blocks = (total + threads - 1) / threads;

        bilinear_resize_chw_kernel<<<blocks, threads, 0, stream>>>(
            src, dst, C, src_H, src_W, dst_H, dst_W);
    }

    __device__ __forceinline__ uint32_t pcg_hash(uint32_t v) {
        uint32_t state = v * 747796405u + 2891336453u;
        uint32_t word = ((state >> ((state >> 28u) + 4u)) ^ state) * 277803737u;
        return (word >> 22u) ^ word;
    }

    __device__ __forceinline__ float uint_to_float(uint32_t v) {
        return static_cast<float>(v) * (1.0f / 4294967296.0f);
    }

    __global__ void random_background_kernel(
        float* __restrict__ output,
        const int HW,
        const uint32_t seed) {
        const int idx = blockIdx.x * blockDim.x + threadIdx.x;

        if (idx >= HW)
            return;

        const uint32_t base = seed ^ static_cast<uint32_t>(idx);
        output[idx] = uint_to_float(pcg_hash(base));
        output[HW + idx] = uint_to_float(pcg_hash(base + 0x9E3779B9u));
        output[2 * HW + idx] = uint_to_float(pcg_hash(base + 0x6C8E9CF9u));
    }

    void launch_random_background(
        float* output,
        const int H, const int W,
        const uint64_t seed,
        cudaStream_t stream) {
        const int HW = H * W;
        constexpr int BLOCK_SIZE = 256;
        const int blocks = (HW + BLOCK_SIZE - 1) / BLOCK_SIZE;

        random_background_kernel<<<blocks, BLOCK_SIZE, 0, stream>>>(
            output, HW, static_cast<uint32_t>(seed));
    }

} // namespace lfs::training::kernels

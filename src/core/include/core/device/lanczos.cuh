/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include "resize_pixels.cuh"

__device__ float sinc(const float x) {
    if (fabsf(x) < 1e-12f)
        return 1.0f;
    return sinf(M_PI * x) / (M_PI * x);
}

__device__ float lanczos_kernel(const float x, const float a) {
    if (x <= -a || x >= a)
        return 0.0f;
    return sinc(x) * sinc(x / a);
}

// The coefficient table and every resample kernel must agree on each sample's support, or a kernel reads
// a weight the table never wrote. Explicit rounding stops the compiler from contracting the copies differently.
__device__ __forceinline__ int2 lanczos_support(const int output_idx, const int input_size,
                                                const int output_size, const int kernel_size) {
    const float scale = lfs::core::device::div_rn(static_cast<float>(input_size), static_cast<float>(output_size));
    const float center = lfs::core::device::mul_rn(lfs::core::device::add_rn(static_cast<float>(output_idx), 0.5f), scale);
    const float reach = lfs::core::device::mul_rn(static_cast<float>(kernel_size), scale);
    return {max(static_cast<int>(lfs::core::device::add_rn(lfs::core::device::sub_rn(center, reach), 0.5f)), 0),
            min(static_cast<int>(lfs::core::device::add_rn(lfs::core::device::add_rn(center, reach), 0.5f)), input_size)};
}

__global__ void __launch_bounds__(BLOCK_X* BLOCK_Y)
    PreComputeCoef(
        const int input_size,
        const int output_size,
        const int kernel_size,
        const uint32_t coefficient_stride,
        float* __restrict__ kernel_values) {
    const auto block = cg::this_thread_block();
    const uint32_t thread_idx = block.thread_index().x;
    const uint32_t output_idx = block.group_index().x * BLOCK_X * BLOCK_Y + thread_idx;
    const float output_ax = (float)output_idx;
    const float scale = 1.0f * input_size / output_size;
    const bool inside = output_idx < output_size;

    if (!inside)
        return;

    const float center = (output_ax + 0.5f) * scale;
    const int2 box = lanczos_support(output_idx, input_size, output_size, kernel_size);
    LFS_DEBUG_ASSERT_MSG(box.y > box.x && static_cast<uint32_t>(box.y - box.x) <= coefficient_stride,
                         "Lanczos support must be non-empty and fit the coefficient stride");

    const uint32_t offset = output_idx * coefficient_stride;
    float norm = 0.0f;
    for (int i = box.x; i < box.y; i++) {
        float value = lanczos_kernel((i + 0.5f - center) / scale, kernel_size);
        kernel_values[offset + i - box.x] = value;
        norm += value;
    }
    for (int i = box.x; i < box.y; i++) {
        kernel_values[offset + i - box.x] /= norm;
    }
    for (uint32_t i = static_cast<uint32_t>(box.y - box.x); i < coefficient_stride; i++) {
        kernel_values[offset + i] = __int_as_float(0x7fffffff);
    }
}

template <uint32_t CHANNELS, typename T, bool ExactAlpha = false>
__global__ void __launch_bounds__(BLOCK_X* BLOCK_Y)
    LanczosResampleCUDA(
        const int input_h, const int input_w,
        const int output_h, const int output_w,
        const int kernel_size,
        const uint32_t coefficient_stride_x,
        const uint32_t coefficient_stride_y,
        const float* __restrict__ pre_coef_x,
        const float* __restrict__ pre_coef_y,
        const T* __restrict__ input, // [H, W, C] T
        float* __restrict__ output   // [C, H, W] float32
    ) {
    const auto block = cg::this_thread_block();
    const uint32_t thread_idx_x = block.thread_index().x;
    const uint32_t thread_idx_y = block.thread_index().y;
    const uint2 pix = {
        block.group_index().x * BLOCK_X + thread_idx_x,
        block.group_index().y * BLOCK_Y + thread_idx_y};
    const bool inside = (pix.x < output_w && pix.y < output_h);

    if (!inside)
        return;

    const int2 support_x = lanczos_support(pix.x, input_w, output_w, kernel_size);
    const int2 support_y = lanczos_support(pix.y, input_h, output_h, kernel_size);
    const int2 LU = {support_x.x, support_y.x};
    const int2 RD = {support_x.y, support_y.y};
    LFS_DEBUG_ASSERT_MSG(RD.x > LU.x && RD.y > LU.y &&
                             static_cast<uint32_t>(RD.x - LU.x) <= coefficient_stride_x &&
                             static_cast<uint32_t>(RD.y - LU.y) <= coefficient_stride_y,
                         "Lanczos support must be non-empty and fit the coefficient stride");

    float accumulator[CHANNELS] = {0.0f};

    for (int y = LU.y; y < RD.y; y++) {
        const float kernel_value_y = pre_coef_y[pix.y * coefficient_stride_y + y - LU.y];
        for (int x = LU.x; x < RD.x; x++) {
            const uint32_t input_pix_id = input_w * y + x;
            const float kernel_value_x = pre_coef_x[pix.x * coefficient_stride_x + x - LU.x];
            LFS_DEBUG_ASSERT_MSG(isfinite(kernel_value_x) && isfinite(kernel_value_y),
                                 "Lanczos read a coefficient the table did not write");
            const float kernel_value = kernel_value_y * kernel_value_x;

            for (int ch = 0; ch < CHANNELS; ch++) {
                if constexpr (ExactAlpha && CHANNELS == 4 && std::is_same_v<T, uint8_t>) {
                    const float value = ch == 3 ? cuda::normalize_mask_byte(input[input_pix_id * CHANNELS + ch])
                                                : PixelTraits<T>::load_and_normalize(input, input_pix_id * CHANNELS + ch);
                    accumulator[ch] += value * kernel_value;
                } else {
                    accumulator[ch] += PixelTraits<T>::load_and_normalize(input, input_pix_id * CHANNELS + ch) * kernel_value;
                }
            }
        }
    }

    const int H_out = output_h;
    const int W_out = output_w;
    for (int ch = 0; ch < CHANNELS; ch++) {
        output[ch * (H_out * W_out) + pix.y * W_out + pix.x] = accumulator[ch];
    }
}

template <typename TIn, bool ExactMask = false>
__global__ void __launch_bounds__(BLOCK_X* BLOCK_Y)
    LanczosResampleGrayscaleCUDA(
        const int input_h, const int input_w,
        const int output_h, const int output_w,
        const int kernel_size,
        const uint32_t coefficient_stride_x,
        const uint32_t coefficient_stride_y,
        const float* __restrict__ pre_coef_x,
        const float* __restrict__ pre_coef_y,
        const TIn* __restrict__ input, // [H, W]
        const float input_scale,       // 1/255 for uint8, 1 for float32
        float* __restrict__ output     // [H, W] float32
    ) {
    const auto block = cg::this_thread_block();
    const uint32_t thread_idx_x = block.thread_index().x;
    const uint32_t thread_idx_y = block.thread_index().y;
    const uint2 pix = {
        block.group_index().x * BLOCK_X + thread_idx_x,
        block.group_index().y * BLOCK_Y + thread_idx_y};
    const bool inside = (pix.x < output_w && pix.y < output_h);

    if (!inside)
        return;

    const int2 support_x = lanczos_support(pix.x, input_w, output_w, kernel_size);
    const int2 support_y = lanczos_support(pix.y, input_h, output_h, kernel_size);
    const int2 LU = {support_x.x, support_y.x};
    const int2 RD = {support_x.y, support_y.y};
    LFS_DEBUG_ASSERT_MSG(RD.x > LU.x && RD.y > LU.y &&
                             static_cast<uint32_t>(RD.x - LU.x) <= coefficient_stride_x &&
                             static_cast<uint32_t>(RD.y - LU.y) <= coefficient_stride_y,
                         "Lanczos support must be non-empty and fit the coefficient stride");

    float accumulator = 0.0f;

    for (int y = LU.y; y < RD.y; y++) {
        const float kernel_value_y = pre_coef_y[pix.y * coefficient_stride_y + y - LU.y];
        for (int x = LU.x; x < RD.x; x++) {
            const uint32_t input_pix_id = input_w * y + x;
            const float kernel_value_x = pre_coef_x[pix.x * coefficient_stride_x + x - LU.x];
            LFS_DEBUG_ASSERT_MSG(isfinite(kernel_value_x) && isfinite(kernel_value_y),
                                 "Lanczos read a coefficient the table did not write");
            const float kernel_value = kernel_value_y * kernel_value_x;
            float pixel_value;
            if constexpr (ExactMask && std::is_same_v<TIn, uint8_t>)
                pixel_value = cuda::normalize_mask_byte(input[input_pix_id]);
            else
                pixel_value = (float)input[input_pix_id] * input_scale;
            accumulator += pixel_value * kernel_value;
        }
    }

    output[pix.y * output_w + pix.x] = accumulator;
}

template <uint32_t CHANNELS>
__global__ void __launch_bounds__(BLOCK_X* BLOCK_Y)
    LanczosResamplePlanarCUDA(
        const int input_h, const int input_w,
        const int output_h, const int output_w,
        const int kernel_size,
        const uint32_t coefficient_stride_x,
        const uint32_t coefficient_stride_y,
        const float* __restrict__ pre_coef_x,
        const float* __restrict__ pre_coef_y,
        const float* __restrict__ input, // [C, H, W] float32
        float* __restrict__ output       // [C, H, W] float32
    ) {
    const auto block = cg::this_thread_block();
    const uint32_t thread_idx_x = block.thread_index().x;
    const uint32_t thread_idx_y = block.thread_index().y;
    const uint2 pix = {
        block.group_index().x * BLOCK_X + thread_idx_x,
        block.group_index().y * BLOCK_Y + thread_idx_y};
    const bool inside = (pix.x < output_w && pix.y < output_h);

    if (!inside)
        return;

    const int2 support_x = lanczos_support(pix.x, input_w, output_w, kernel_size);
    const int2 support_y = lanczos_support(pix.y, input_h, output_h, kernel_size);
    const int2 LU = {support_x.x, support_y.x};
    const int2 RD = {support_x.y, support_y.y};
    LFS_DEBUG_ASSERT_MSG(RD.x > LU.x && RD.y > LU.y &&
                             static_cast<uint32_t>(RD.x - LU.x) <= coefficient_stride_x &&
                             static_cast<uint32_t>(RD.y - LU.y) <= coefficient_stride_y,
                         "Lanczos support must be non-empty and fit the coefficient stride");

    const int input_plane = input_h * input_w;
    const int output_plane = output_h * output_w;

    float accumulator[CHANNELS] = {0.0f};

    for (int y = LU.y; y < RD.y; y++) {
        const float kernel_value_y = pre_coef_y[pix.y * coefficient_stride_y + y - LU.y];
        for (int x = LU.x; x < RD.x; x++) {
            const uint32_t input_pix_id = input_w * y + x;
            const float kernel_value_x = pre_coef_x[pix.x * coefficient_stride_x + x - LU.x];
            LFS_DEBUG_ASSERT_MSG(isfinite(kernel_value_x) && isfinite(kernel_value_y),
                                 "Lanczos read a coefficient the table did not write");
            const float kernel_value = kernel_value_y * kernel_value_x;

            for (int ch = 0; ch < CHANNELS; ch++) {
                accumulator[ch] += input[ch * input_plane + input_pix_id] * kernel_value;
            }
        }
    }

    for (int ch = 0; ch < CHANNELS; ch++) {
        output[ch * output_plane + pix.y * output_w + pix.x] = accumulator[ch];
    }
}

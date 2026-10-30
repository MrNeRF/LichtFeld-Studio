/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include "image_sentinel.cuh"

__global__ void fill_u8_sentinel_kernel(
    uint8_t* __restrict__ output,
    const size_t byte_count,
    const uint32_t seed) {
    const size_t stride = static_cast<size_t>(blockDim.x) * gridDim.x;
    for (size_t idx = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         idx < byte_count;
         idx += stride) {
        output[idx] = sentinel_byte(idx, seed);
    }
}

__global__ void flag_u8_sentinel_unchanged_kernel(
    const uint8_t* __restrict__ input,
    const size_t byte_count,
    const uint32_t seed,
    uint32_t* unchanged) {
    const size_t stride = static_cast<size_t>(blockDim.x) * gridDim.x;
    for (size_t idx = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         idx < byte_count;
         idx += stride) {
        if (input[idx] != sentinel_byte(idx, seed)) {
            atomicExch(unchanged, 0u);
            return;
        }
    }
}

template <bool ExactAlpha = false>
__global__ void uint8_hwc_to_float32_chw_kernel(
    const uint8_t* __restrict__ input,
    float* __restrict__ output,
    const size_t H,
    const size_t W,
    const size_t C) {

    const size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    const size_t total = H * W * C;
    if (idx >= total)
        return;

    const size_t c = idx % C;
    const size_t tmp = idx / C;
    const size_t w = tmp % W;
    const size_t h = tmp / W;

    const size_t out_idx = c * (H * W) + h * W + w;
    if constexpr (ExactAlpha) {
        output[out_idx] = c == 3 ? lfs::core::cuda::normalize_mask_byte(input[idx])
                                 : static_cast<float>(input[idx]) * NORMALIZE_SCALE_U8;
    } else {
        output[out_idx] = static_cast<float>(input[idx]) * NORMALIZE_SCALE_U8;
    }
}

template <bool ExactAlpha = false>
__global__ void uint16_hwc_to_float32_chw_kernel(
    const uint16_t* __restrict__ input,
    float* __restrict__ output,
    const size_t H,
    const size_t W,
    const size_t C) {

    const size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    const size_t total = H * W * C;
    if (idx >= total)
        return;

    const size_t c = idx % C;
    const size_t tmp = idx / C;
    const size_t w = tmp % W;
    const size_t h = tmp / W;

    const size_t out_idx = c * (H * W) + h * W + w;
    if constexpr (ExactAlpha) {
        output[out_idx] = c == 3 ? lfs::core::cuda::normalize_mask_u16_value(input[idx])
                                 : static_cast<float>(input[idx]) * NORMALIZE_SCALE_U16;
    } else {
        output[out_idx] = static_cast<float>(input[idx]) * NORMALIZE_SCALE_U16;
    }
}

__global__ void float32_hwc_to_uint16_hwc_kernel(
    const float* __restrict__ input,
    uint16_t* __restrict__ output,
    const size_t total) {

    const size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total)
        return;

    const float scaled = fminf(fmaxf(input[idx] * 65535.0f, 0.0f), 65535.0f);
    output[idx] = static_cast<uint16_t>(scaled + 0.5f);
}

template <bool ExactAlpha = false>
__global__ void uint16_hwc_to_float_hwc_kernel(
    const uint16_t* __restrict__ input,
    float* __restrict__ output,
    const size_t total) {

    const size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total)
        return;
    if constexpr (ExactAlpha) {
        output[idx] = idx % 4 == 3 ? lfs::core::cuda::normalize_mask_u16_value(input[idx])
                                   : static_cast<float>(input[idx]) * NORMALIZE_SCALE_U16;
    } else {
        output[idx] = static_cast<float>(input[idx]) * NORMALIZE_SCALE_U16;
    }
}

__global__ void normal_chw_to_jpeg2k_hwc_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    const size_t pixel_count) {

    const size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= pixel_count)
        return;

#pragma unroll
    for (int c = 0; c < 3; ++c) {
        const float encoded = fminf(fmaxf(input[c * pixel_count + idx] * 0.5f + 0.5f, 0.0f), 1.0f);
        output[idx * 3 + c] = encoded;
    }
}

__global__ void jpeg2k_hwc_to_normal_chw_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    const size_t pixel_count) {

    const size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= pixel_count)
        return;

#pragma unroll
    for (int c = 0; c < 3; ++c) {
        output[c * pixel_count + idx] = input[idx * 3 + c] * 2.0f - 1.0f;
    }
}

__device__ __forceinline__ float srgb_encoding_to_linear_device(const float v) {
    if (v <= 0.04045f) {
        return v / 12.92f;
    }
    return powf((v + 0.055f) / 1.055f, 2.4f);
}

template <typename T>
__global__ void normal_prior_hwc_to_float32_chw_kernel(
    const T* __restrict__ input,
    float* __restrict__ output,
    const size_t pixel_count,
    const NormalPriorTransform transform) {

    const size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= pixel_count)
        return;

    constexpr float scale = sizeof(T) == sizeof(float) ? 1.0f : sizeof(T) == 2 ? NORMALIZE_SCALE_U16
                                                                               : NORMALIZE_SCALE_U8;
    float n[3];
#pragma unroll
    for (int c = 0; c < 3; ++c) {
        float encoded = static_cast<float>(input[idx * 3 + c]) * scale;
        if (transform.srgb) {
            encoded = srgb_encoding_to_linear_device(encoded);
        }
        n[c] = encoded * 2.0f - 1.0f;
    }
    if (transform.flip_yz) {
        n[1] = -n[1];
        n[2] = -n[2];
    }
    if (transform.world_to_camera) {
        const float x = n[0];
        const float y = n[1];
        const float z = n[2];
        n[0] = transform.w2c[0] * x + transform.w2c[1] * y + transform.w2c[2] * z;
        n[1] = transform.w2c[3] * x + transform.w2c[4] * y + transform.w2c[5] * z;
        n[2] = transform.w2c[6] * x + transform.w2c[7] * y + transform.w2c[8] * z;
    }
#pragma unroll
    for (int c = 0; c < 3; ++c) {
        output[c * pixel_count + idx] = n[c];
    }
}

__global__ void uint8_hwc_to_uint8_chw_kernel(
    const uint8_t* __restrict__ input,
    uint8_t* __restrict__ output,
    const size_t H,
    const size_t W,
    const size_t C) {

    const size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    const size_t total = H * W * C;
    if (idx >= total)
        return;

    const size_t c = idx % C;
    const size_t tmp = idx / C;
    const size_t w = tmp % W;
    const size_t h = tmp / W;

    const size_t out_idx = c * (H * W) + h * W + w;
    output[out_idx] = input[idx];
}

__global__ void uint16_hwc_to_uint8_chw_kernel(
    const uint16_t* __restrict__ input,
    uint8_t* __restrict__ output,
    const size_t H,
    const size_t W,
    const size_t C) {

    const size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    const size_t total = H * W * C;
    if (idx >= total)
        return;

    const size_t c = idx % C;
    const size_t tmp = idx / C;
    const size_t w = tmp % W;
    const size_t h = tmp / W;

    const size_t out_idx = c * (H * W) + h * W + w;
    output[out_idx] = static_cast<uint8_t>(
        (static_cast<uint32_t>(input[idx]) * 255u + 32767u) / 65535u);
}

__device__ __forceinline__ uint8_t float_to_u8(const float v) {
    const float scaled = fminf(fmaxf(v, 0.0f), 1.0f) * 255.0f;
    return static_cast<uint8_t>(scaled + 0.5f);
}

__global__ void float32_chw_to_uint8_chw_kernel(
    const float* __restrict__ input,
    uint8_t* __restrict__ output,
    const size_t total) {

    const size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total)
        return;

    output[idx] = float_to_u8(input[idx]);
}

__global__ void uint8_hw_to_float32_hw_kernel(
    const uint8_t* __restrict__ input,
    float* __restrict__ output,
    const size_t total) {

    const size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total)
        return;

    output[idx] = static_cast<float>(input[idx]) * NORMALIZE_SCALE_U8;
}

__global__ void uint8_rgba_split_kernel(
    const uint8_t* __restrict__ input,
    float* __restrict__ rgb_output,
    float* __restrict__ alpha_output,
    const size_t total) {

    const size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total)
        return;

    const size_t src = idx * 4;
    const float r = static_cast<float>(input[src + 0]) * NORMALIZE_SCALE_U8;
    const float g = static_cast<float>(input[src + 1]) * NORMALIZE_SCALE_U8;
    const float b = static_cast<float>(input[src + 2]) * NORMALIZE_SCALE_U8;
    const float a = lfs::core::cuda::normalize_mask_byte(input[src + 3]);

    rgb_output[idx] = r;
    rgb_output[total + idx] = g;
    rgb_output[2 * total + idx] = b;
    alpha_output[idx] = a;
}

__global__ void uint8_rgba_split_uint8_rgb_kernel(
    const uint8_t* __restrict__ input,
    uint8_t* __restrict__ rgb_output,
    float* __restrict__ alpha_output,
    const size_t total) {

    const size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total)
        return;

    const size_t src = idx * 4;
    rgb_output[idx] = input[src + 0];
    rgb_output[total + idx] = input[src + 1];
    rgb_output[2 * total + idx] = input[src + 2];
    alpha_output[idx] = lfs::core::cuda::normalize_mask_byte(input[src + 3]);
}

__global__ void mask_invert_kernel(
    float* __restrict__ data,
    const size_t total) {

    const size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total)
        return;

    data[idx] = 1.0f - data[idx];
}

__global__ void mask_threshold_kernel(
    float* __restrict__ data,
    const size_t total,
    const float threshold) {

    const size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total)
        return;

    data[idx] = lfs::core::cuda::mask_value_at_least(data[idx], threshold) ? 1.0f : 0.0f;
}

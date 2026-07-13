/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "image_format_kernels.cuh"
#include <cassert>

namespace lfs::io::cuda {

    namespace {
        constexpr int BLOCK_SIZE = 256;
        constexpr float NORMALIZE_SCALE_U8 = 1.0f / 255.0f;
        constexpr float NORMALIZE_SCALE_U16 = 1.0f / 65535.f;
    } // namespace

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
        output[out_idx] = static_cast<float>(input[idx]) * NORMALIZE_SCALE_U8;
    }

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
        output[out_idx] = static_cast<float>(input[idx]) * NORMALIZE_SCALE_U16;
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

    __global__ void uint16_hwc_to_float_hwc_kernel(
        const uint16_t* __restrict__ input,
        float* __restrict__ output,
        const size_t total) {

        const size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx >= total)
            return;
        output[idx] = static_cast<float>(input[idx]) * NORMALIZE_SCALE_U16;
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

    namespace {
        __device__ __forceinline__ float srgb_encoding_to_linear_device(const float v) {
            if (v <= 0.04045f) {
                return v / 12.92f;
            }
            return powf((v + 0.055f) / 1.055f, 2.4f);
        }
    } // namespace

    template <typename T>
    __global__ void normal_prior_hwc_to_float32_chw_kernel(
        const T* __restrict__ input,
        float* __restrict__ output,
        const size_t pixel_count,
        const NormalPriorTransform transform) {

        const size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx >= pixel_count)
            return;

        constexpr float scale = sizeof(T) == 2 ? NORMALIZE_SCALE_U16 : NORMALIZE_SCALE_U8;
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

    void launch_uint8_hwc_to_float32_chw(
        const uint8_t* input,
        float* output,
        const size_t height,
        const size_t width,
        const size_t channels,
        cudaStream_t stream) {

        const size_t total = height * width * channels;
        const int num_blocks = static_cast<int>((total + BLOCK_SIZE - 1) / BLOCK_SIZE);

        uint8_hwc_to_float32_chw_kernel<<<num_blocks, BLOCK_SIZE, 0, stream>>>(
            input, output, height, width, channels);
    }

    void launch_uint16_hwc_to_float32_chw(
        const uint16_t* input,
        float* output,
        size_t height,
        size_t width,
        size_t channels,
        cudaStream_t stream) {

        const size_t total = height * width * channels;
        const int num_blocks = static_cast<int>((total + BLOCK_SIZE - 1) / BLOCK_SIZE);

        uint16_hwc_to_float32_chw_kernel<<<num_blocks, BLOCK_SIZE, 0, stream>>>(
            input, output, height, width, channels);
    }

    void launch_normal_prior_u8_hwc_to_float32_chw(
        const uint8_t* input,
        float* output,
        size_t height,
        size_t width,
        const NormalPriorTransform& transform,
        cudaStream_t stream) {

        const size_t pixel_count = height * width;
        const int num_blocks = static_cast<int>((pixel_count + BLOCK_SIZE - 1) / BLOCK_SIZE);

        normal_prior_hwc_to_float32_chw_kernel<uint8_t><<<num_blocks, BLOCK_SIZE, 0, stream>>>(
            input, output, pixel_count, transform);
    }

    void launch_normal_prior_u16_hwc_to_float32_chw(
        const uint16_t* input,
        float* output,
        size_t height,
        size_t width,
        const NormalPriorTransform& transform,
        cudaStream_t stream) {

        const size_t pixel_count = height * width;
        const int num_blocks = static_cast<int>((pixel_count + BLOCK_SIZE - 1) / BLOCK_SIZE);

        normal_prior_hwc_to_float32_chw_kernel<uint16_t><<<num_blocks, BLOCK_SIZE, 0, stream>>>(
            input, output, pixel_count, transform);
    }

    void launch_float32_hwc_to_uint16_hwc(
        const float* input,
        uint16_t* output,
        size_t height,
        size_t width,
        size_t channels,
        cudaStream_t stream) {

        const size_t total = height * width * channels;
        const int num_blocks = static_cast<int>((total + BLOCK_SIZE - 1) / BLOCK_SIZE);

        float32_hwc_to_uint16_hwc_kernel<<<num_blocks, BLOCK_SIZE, 0, stream>>>(
            input, output, total);
    }

    void launch_uint16_hwc_to_float32_hwc(
        const uint16_t* input,
        float* output,
        size_t height,
        size_t width,
        size_t channels,
        cudaStream_t stream) {

        const size_t total = height * width * channels;
        const int num_blocks = static_cast<int>((total + BLOCK_SIZE - 1) / BLOCK_SIZE);

        uint16_hwc_to_float_hwc_kernel<<<num_blocks, BLOCK_SIZE, 0, stream>>>(
            input, output, total);
    }

    void launch_normal_chw_to_jpeg2k_hwc(
        const float* input,
        float* output,
        size_t height,
        size_t width,
        cudaStream_t stream) {

        const size_t pixel_count = height * width;
        const int num_blocks = static_cast<int>((pixel_count + BLOCK_SIZE - 1) / BLOCK_SIZE);

        normal_chw_to_jpeg2k_hwc_kernel<<<num_blocks, BLOCK_SIZE, 0, stream>>>(
            input, output, pixel_count);
    }

    void launch_jpeg2k_hwc_to_normal_chw(
        const float* input,
        float* output,
        size_t height,
        size_t width,
        cudaStream_t stream) {

        const size_t pixel_count = height * width;
        const int num_blocks = static_cast<int>((pixel_count + BLOCK_SIZE - 1) / BLOCK_SIZE);

        jpeg2k_hwc_to_normal_chw_kernel<<<num_blocks, BLOCK_SIZE, 0, stream>>>(
            input, output, pixel_count);
    }

    void launch_uint8_hwc_to_uint8_chw(
        const uint8_t* input,
        uint8_t* output,
        const size_t height,
        const size_t width,
        const size_t channels,
        cudaStream_t stream) {

        const size_t total = height * width * channels;
        const int num_blocks = static_cast<int>((total + BLOCK_SIZE - 1) / BLOCK_SIZE);

        uint8_hwc_to_uint8_chw_kernel<<<num_blocks, BLOCK_SIZE, 0, stream>>>(
            input, output, height, width, channels);
    }

    void launch_uint16_hwc_to_uint8_chw(
        const uint16_t* input,
        uint8_t* output,
        const size_t height,
        const size_t width,
        const size_t channels,
        cudaStream_t stream) {

        const size_t total = height * width * channels;
        const int num_blocks = static_cast<int>((total + BLOCK_SIZE - 1) / BLOCK_SIZE);

        uint16_hwc_to_uint8_chw_kernel<<<num_blocks, BLOCK_SIZE, 0, stream>>>(
            input, output, height, width, channels);
    }

    void launch_float32_chw_to_uint8_chw(
        const float* input,
        uint8_t* output,
        const size_t height,
        const size_t width,
        const size_t channels,
        cudaStream_t stream) {

        const size_t total = height * width * channels;
        const int num_blocks = static_cast<int>((total + BLOCK_SIZE - 1) / BLOCK_SIZE);

        float32_chw_to_uint8_chw_kernel<<<num_blocks, BLOCK_SIZE, 0, stream>>>(
            input, output, total);
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

    void launch_uint8_hw_to_float32_hw(
        const uint8_t* input,
        float* output,
        const size_t height,
        const size_t width,
        cudaStream_t stream) {

        const size_t total = height * width;
        const int num_blocks = static_cast<int>((total + BLOCK_SIZE - 1) / BLOCK_SIZE);

        uint8_hw_to_float32_hw_kernel<<<num_blocks, BLOCK_SIZE, 0, stream>>>(
            input, output, total);
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
        const float a = static_cast<float>(input[src + 3]) * NORMALIZE_SCALE_U8;

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
        alpha_output[idx] = static_cast<float>(input[src + 3]) * NORMALIZE_SCALE_U8;
    }

    void launch_uint8_rgba_split_to_float32_rgb_and_alpha(
        const uint8_t* input,
        float* rgb_output,
        float* alpha_output,
        const size_t height,
        const size_t width,
        cudaStream_t stream) {

        assert(input && "input must not be null");
        assert(rgb_output && "rgb_output must not be null");
        assert(alpha_output && "alpha_output must not be null");
        assert(height > 0 && width > 0);

        const size_t total = height * width;
        const int num_blocks = static_cast<int>((total + BLOCK_SIZE - 1) / BLOCK_SIZE);

        uint8_rgba_split_kernel<<<num_blocks, BLOCK_SIZE, 0, stream>>>(
            input, rgb_output, alpha_output, total);
    }

    void launch_uint8_rgba_split_to_uint8_rgb_and_float32_alpha(
        const uint8_t* input,
        uint8_t* rgb_output,
        float* alpha_output,
        const size_t height,
        const size_t width,
        cudaStream_t stream) {

        assert(input && "input must not be null");
        assert(rgb_output && "rgb_output must not be null");
        assert(alpha_output && "alpha_output must not be null");
        assert(height > 0 && width > 0);

        const size_t total = height * width;
        const int num_blocks = static_cast<int>((total + BLOCK_SIZE - 1) / BLOCK_SIZE);

        uint8_rgba_split_uint8_rgb_kernel<<<num_blocks, BLOCK_SIZE, 0, stream>>>(
            input, rgb_output, alpha_output, total);
    }

    __global__ void mask_invert_kernel(
        float* __restrict__ data,
        const size_t total) {

        const size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx >= total)
            return;

        data[idx] = 1.0f - data[idx];
    }

    void launch_mask_invert(
        float* data,
        const size_t height,
        const size_t width,
        cudaStream_t stream) {

        const size_t total = height * width;
        const int num_blocks = static_cast<int>((total + BLOCK_SIZE - 1) / BLOCK_SIZE);

        mask_invert_kernel<<<num_blocks, BLOCK_SIZE, 0, stream>>>(data, total);
    }

    __global__ void mask_threshold_kernel(
        float* __restrict__ data,
        const size_t total,
        const float threshold) {

        const size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx >= total)
            return;

        data[idx] = (data[idx] >= threshold) ? 1.0f : 0.0f;
    }

    void launch_mask_threshold(
        float* data,
        const size_t height,
        const size_t width,
        const float threshold,
        cudaStream_t stream) {

        const size_t total = height * width;
        const int num_blocks = static_cast<int>((total + BLOCK_SIZE - 1) / BLOCK_SIZE);

        mask_threshold_kernel<<<num_blocks, BLOCK_SIZE, 0, stream>>>(data, total, threshold);
    }

} // namespace lfs::io::cuda

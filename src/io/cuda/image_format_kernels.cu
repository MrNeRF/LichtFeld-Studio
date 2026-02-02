/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "image_format_kernels.cuh"
#include <cassert>

namespace lfs::io::cuda {

    namespace {
        constexpr int BLOCK_SIZE = 256;
        constexpr float NORMALIZE_SCALE = 1.0f / 255.0f;
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
        output[out_idx] = static_cast<float>(input[idx]) * NORMALIZE_SCALE;
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

    __global__ void uint8_hw_to_float32_hw_kernel(
        const uint8_t* __restrict__ input,
        float* __restrict__ output,
        const size_t total) {

        const size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx >= total)
            return;

        output[idx] = static_cast<float>(input[idx]) * NORMALIZE_SCALE;
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
        const float r = static_cast<float>(input[src + 0]) * NORMALIZE_SCALE;
        const float g = static_cast<float>(input[src + 1]) * NORMALIZE_SCALE;
        const float b = static_cast<float>(input[src + 2]) * NORMALIZE_SCALE;
        const float a = static_cast<float>(input[src + 3]) * NORMALIZE_SCALE;

        rgb_output[idx] = r;
        rgb_output[total + idx] = g;
        rgb_output[2 * total + idx] = b;
        alpha_output[idx] = a;
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

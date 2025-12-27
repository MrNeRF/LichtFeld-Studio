/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "image_format_kernels.cuh"

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

} // namespace lfs::io::cuda

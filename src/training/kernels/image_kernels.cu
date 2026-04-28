/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "image_kernels.hpp"

#include "cuda.h"
#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <type_traits>

namespace lfs::filters {

    // Adapted from spirulae-splat Densify.cu canny_edge_filter_kernel
    // (Apache-2.0, commit 8f2ecddc76e6de2e04f88ba8ee5f03b2439766d4):
    // https://github.com/harry7557558/spirulae-splat/blob/8f2ecddc76e6de2e04f88ba8ee5f03b2439766d4/spirulae_splat/splat/cuda/csrc/Densify.cu#L663
    // Changes here: raw CHW input/output pointers instead of Spirulae's batched HWC TensorView.
    __constant__ float SPIRULAE_BLUR_5x5[25] = {
        2.f / 159.f, 4.f / 159.f, 5.f / 159.f, 4.f / 159.f, 2.f / 159.f,
        4.f / 159.f, 9.f / 159.f, 12.f / 159.f, 9.f / 159.f, 4.f / 159.f,
        5.f / 159.f, 12.f / 159.f, 15.f / 159.f, 12.f / 159.f, 5.f / 159.f,
        4.f / 159.f, 9.f / 159.f, 12.f / 159.f, 9.f / 159.f, 4.f / 159.f,
        2.f / 159.f, 4.f / 159.f, 5.f / 159.f, 4.f / 159.f, 2.f / 159.f};

    __constant__ float SPIRULAE_CANNY_3x3[9] = {
        -1.0f, 0.0f, 1.0f,
        -2.0f, 0.0f, 2.0f,
        -1.0f, 0.0f, 1.0f};
} // namespace lfs::filters

namespace lfs::training::kernels {

    template <typename T>
    __device__ __forceinline__ float canny_input_value(const T* input, const int idx) {
        if constexpr (std::is_same_v<std::remove_cv_t<T>, uint8_t>) {
            return static_cast<float>(input[idx]) * (1.0f / 255.0f);
        } else {
            return static_cast<float>(input[idx]);
        }
    }

    template <typename InputT>
    __global__ void fused_canny_edge_filter_chw_kernel(
        const InputT* __restrict__ input,
        float* __restrict__ output,
        const int height,
        const int width) {
        constexpr int BLOCK = 32;
        constexpr int HALO = 4;
        constexpr int HALO1 = 2;
        constexpr int HALO2 = 1;
        constexpr int PIXELS_SHARED = BLOCK + 2 * HALO;
        constexpr int BLURRED_SHARED = BLOCK + 2 * HALO1;
        constexpr int FILTERED_SHARED = BLOCK + 2 * HALO2;

        const int xid = blockIdx.x * BLOCK + threadIdx.x;
        const int yid = blockIdx.y * BLOCK + threadIdx.y;
        const int plane_size = height * width;

        __shared__ float shared_pixels[PIXELS_SHARED][PIXELS_SHARED];
#pragma unroll
        for (int batch = 0; batch < PIXELS_SHARED * PIXELS_SHARED; batch += BLOCK * BLOCK) {
            const int tid = batch + threadIdx.y * BLOCK + threadIdx.x;
            const int y = tid / PIXELS_SHARED;
            const int x = tid % PIXELS_SHARED;
            if (tid < PIXELS_SHARED * PIXELS_SHARED) {
                const int yi = min(max(static_cast<int>(blockIdx.y * BLOCK) + y - HALO, 0), height - 1);
                const int xi = min(max(static_cast<int>(blockIdx.x * BLOCK) + x - HALO, 0), width - 1);
                const int idx = yi * width + xi;
                shared_pixels[y][x] =
                    0.299f * canny_input_value(input, idx) +
                    0.587f * canny_input_value(input, idx + plane_size) +
                    0.114f * canny_input_value(input, idx + 2 * plane_size);
            }
        }
        __syncthreads();

        __shared__ float shared_blurred[BLURRED_SHARED][BLURRED_SHARED];
#pragma unroll
        for (int batch = 0; batch < BLURRED_SHARED * BLURRED_SHARED; batch += BLOCK * BLOCK) {
            const int tid = batch + threadIdx.y * BLOCK + threadIdx.x;
            const int y = tid / BLURRED_SHARED;
            const int x = tid % BLURRED_SHARED;
            if (tid >= BLURRED_SHARED * BLURRED_SHARED) {
                continue;
            }

            float total = 0.0f;
#pragma unroll
            for (int cy = -2; cy <= 2; ++cy) {
#pragma unroll
                for (int cx = -2; cx <= 2; ++cx) {
                    const float conv_weight = lfs::filters::SPIRULAE_BLUR_5x5[(cy + 2) * 5 + (cx + 2)];
                    const int yi = y - HALO1 + cy;
                    const int xi = x - HALO1 + cx;
                    total += conv_weight * shared_pixels[yi + HALO][xi + HALO];
                }
            }
            shared_blurred[y][x] = total;
        }
        __syncthreads();

        __shared__ float2 shared_filtered[FILTERED_SHARED][FILTERED_SHARED];
#pragma unroll
        for (int batch = 0; batch < FILTERED_SHARED * FILTERED_SHARED; batch += BLOCK * BLOCK) {
            const int tid = batch + threadIdx.y * BLOCK + threadIdx.x;
            const int y = tid / FILTERED_SHARED;
            const int x = tid % FILTERED_SHARED;
            if (tid >= FILTERED_SHARED * FILTERED_SHARED) {
                continue;
            }

            float total1 = 0.0f;
            float total2 = 0.0f;
#pragma unroll
            for (int cy = -1; cy <= 1; ++cy) {
#pragma unroll
                for (int cx = -1; cx <= 1; ++cx) {
                    const float conv_weight_1 = lfs::filters::SPIRULAE_CANNY_3x3[(cy + 1) * 3 + (cx + 1)];
                    const float conv_weight_2 = lfs::filters::SPIRULAE_CANNY_3x3[(cx + 1) * 3 + (cy + 1)];
                    const int yi = y - HALO2 + cy;
                    const int xi = x - HALO2 + cx;
                    const float value = shared_blurred[yi + HALO1][xi + HALO1];
                    total1 += conv_weight_1 * value;
                    total2 += conv_weight_2 * value;
                }
            }
            shared_filtered[y][x] = make_float2(total1, total2);
        }
        __syncthreads();

        float2 gradient = shared_filtered[threadIdx.y + HALO2][threadIdx.x + HALO2];
        float mag = hypotf(gradient.x, gradient.y);
        if (mag > 0.0f) {
            const int dx = min(max(static_cast<int>(roundf(gradient.x / mag)), -HALO2), HALO2);
            const int dy = min(max(static_cast<int>(roundf(gradient.y / mag)), -HALO2), HALO2);
            const float2 forward = shared_filtered[static_cast<int>(threadIdx.y) + dy + HALO2][static_cast<int>(threadIdx.x) + dx + HALO2];
            const float2 backward = shared_filtered[static_cast<int>(threadIdx.y) - dy + HALO2][static_cast<int>(threadIdx.x) - dx + HALO2];
            if (mag < hypotf(forward.x, forward.y) || mag < hypotf(backward.x, backward.y)) {
                mag = 0.0f;
            }
        }

        if (yid < height && xid < width) {
            output[yid * width + xid] = mag;
        }
    }

    __global__ void normalize_by_device_scalar_kernel(
        float* __restrict__ data,
        const std::size_t n,
        const float* __restrict__ scalar) {
        const float divisor = fmaxf(*scalar, 1e-9f);
        const std::size_t idx = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
        const std::size_t stride = static_cast<std::size_t>(blockDim.x) * gridDim.x;
        for (std::size_t i = idx; i < n; i += stride) {
            data[i] /= divisor;
        }
    }

    // ============================================================================
    // Launch functions
    // ============================================================================

    template <typename InputT>
    void launch_fused_canny_edge_filter_chw_impl(
        const InputT* d_input_chw,
        float* d_output_hw,
        const int height,
        const int width,
        cudaStream_t stream) {
        dim3 blockDim(32, 32, 1);
        dim3 gridDim((width + blockDim.x - 1) / blockDim.x,
                     (height + blockDim.y - 1) / blockDim.y);

        fused_canny_edge_filter_chw_kernel<InputT><<<gridDim, blockDim, 0, stream>>>(
            d_input_chw, d_output_hw, height, width);
    }

    void launch_fused_canny_edge_filter_chw(
        const float* d_input_chw,
        float* d_output_hw,
        const int height,
        const int width,
        cudaStream_t stream) {
        launch_fused_canny_edge_filter_chw_impl(d_input_chw, d_output_hw, height, width, stream);
    }

    void launch_fused_canny_edge_filter_chw(
        const uint8_t* d_input_chw,
        float* d_output_hw,
        const int height,
        const int width,
        cudaStream_t stream) {
        launch_fused_canny_edge_filter_chw_impl(d_input_chw, d_output_hw, height, width, stream);
    }

    void launch_normalize_by_device_scalar(
        float* d_data,
        const std::size_t n,
        const float* d_scalar,
        cudaStream_t stream) {
        if (n == 0) {
            return;
        }

        constexpr int block_size = 256;
        const int grid_size = static_cast<int>(std::min<std::size_t>((n + block_size - 1) / block_size, 4096));
        normalize_by_device_scalar_kernel<<<grid_size, block_size, 0, stream>>>(d_data, n, d_scalar);
        cudaStreamSynchronize(stream);
    }
} // namespace lfs::training::kernels

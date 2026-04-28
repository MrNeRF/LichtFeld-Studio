/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "image_kernels.hpp"
#include <stdio.h>

#include "cuda.h"
#include <algorithm>
#include <cstddef>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <math_functions.h>

#define FILTER_RADIUS_1 1
#define FILTER_RADIUS_2 2
#define KERNEL_SIZE_K3  (1 * FILTER_RADIUS_1 + 1)
#define KERNEL_SIZE_K5  (1 * FILTER_RADIUS_2 + 1)

namespace lfs::filters {

    __constant__ float LAPLACIAN_3x3[9] = {
        0.0f, 1.0f, 0.0f,
        1.0f, -4.0f, 1.0f,
        0.0f, 1.0f, 0.0f};

    __constant__ float GAUSS_3x3[9] = {
        0.0625f, 0.125f, 0.0625f,
        0.125f, 0.25f, 0.125f,
        0.0625f, 0.125f, 0.0625f};

    __constant__ float GAUSS_5x5[25] = {
        0.0030, 0.0133, 0.0219, 0.0133, 0.0030,
        0.0133, 0.0596, 0.0983, 0.0596, 0.0133,
        0.0219, 0.0983, 0.1621, 0.0983, 0.0219,
        0.0133, 0.0596, 0.0983, 0.0596, 0.0133,
        0.0030, 0.0133, 0.0219, 0.0133, 0.0030};

    __constant__ float SOBEL_X[9] = {-1, 0, 1, -2, 0, 2, -1, 0, 1};
    __constant__ float SOBEL_Y[9] = {1, 2, 1, 0, 0, 0, -1, -2, -1};

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

    // ============================================================================
    // Image Filtering Kernels (Convolution Kernels)
    // ============================================================================

    __global__ void rgb_to_grayscale(const float* input, float* output, const int height, const int width) {
        int x = blockIdx.x * blockDim.x + threadIdx.x;
        int y = blockIdx.y * blockDim.y + threadIdx.y;

        if (x < width && y < height) {
            int plane_size = width * height;
            int idx = y * width + x;

            // Accessing CHW: Channel 0 is R, 1 is G, 2 is B
            float r = input[idx];
            float g = input[idx + plane_size];
            float b = input[idx + 2 * plane_size];

            output[idx] = 0.299f * r + 0.587f * g + 0.114f * b;
        }
    }

    __global__ void gaussian_blur_5x5(const float* d_input, float* d_output, const int height, const int width) {
        int col = blockIdx.x * blockDim.x + threadIdx.x;
        int row = blockIdx.y * blockDim.y + threadIdx.y;

        if (row < height && col < width) {
            float p_value = 0.0f;

            for (int fRow = 0; fRow < KERNEL_SIZE_K5; fRow++) {
                for (int fCol = 0; fCol < KERNEL_SIZE_K5; fCol++) {

                    int inRow = max(0, min(height - 1, row - 2 + fRow));
                    int inCol = max(0, min(width - 1, col - 2 + fCol));

                    if (inCol >= 0 && inCol < width && inRow >= 0 && inRow < height) {
                        // Pixel * pixel weight
                        p_value += lfs::filters::GAUSS_5x5[fRow * KERNEL_SIZE_K5 + fCol] * d_input[inRow * width + inCol];
                    }
                }
            }
            d_output[row * width + col] = p_value;
        }
    }

    __global__ void laplacian_filter_3x3(const float* d_input, float* d_output, const int height, const int width) {
        int col = blockIdx.x * blockDim.x + threadIdx.x;
        int row = blockIdx.y * blockDim.y + threadIdx.y;

        if (row < height && col < width) {
            float p_value = 0.0f;

            for (int fRow = 0; fRow < KERNEL_SIZE_K3; fRow++) {
                for (int fCol = 0; fCol < KERNEL_SIZE_K3; fCol++) {

                    const int inCol = col - FILTER_RADIUS_1 + fCol;
                    const int inRow = row - FILTER_RADIUS_1 + fRow;

                    if (inCol >= 0 && inCol < width && inRow >= 0 && inRow < height) {
                        // Pixel * pixel weight
                        p_value += lfs::filters::LAPLACIAN_3x3[fRow * KERNEL_SIZE_K3 + fCol] * d_input[inRow * width + inCol];
                    }
                }
            }

            float output_value = fabsf(p_value);
            if (output_value < 0.15f) {
                output_value = 0.0f;
            }

            d_output[row * width + col] = output_value * 2.0f;
        }
    }

    __global__ void sobel_gradient_kernel(const float* input, float* magnitude, float* angle, const int height, const int width) {
        int col = blockIdx.x * blockDim.x + threadIdx.x;
        int row = blockIdx.y * blockDim.y + threadIdx.y;

        if (row < height && col < width) {
            float gx = 0.0f;
            float gy = 0.0f;

            for (int fRow = 0; fRow < 3; fRow++) {
                for (int fCol = 0; fCol < 3; fCol++) {
                    int inRow = max(0, min(height - 1, row - 1 + fRow));
                    int inCol = max(0, min(width - 1, col - 1 + fCol));
                    float pixel = input[inRow * width + inCol];

                    gx += pixel * lfs::filters::SOBEL_X[fRow * 3 + fCol];
                    gy += pixel * lfs::filters::SOBEL_Y[fRow * 3 + fCol];
                }
            }

            // Calculate Magnitude
            magnitude[row * width + col] = hypotf(gx, gy);

            // Calculate Direction in Radians
            angle[row * width + col] = atan2f(gy, gx);
        }
    }

    __global__ void nms_kernel(const float* magnitude, const float* angle, float* output, int height, int width) {
        int col = blockIdx.x * blockDim.x + threadIdx.x;
        int row = blockIdx.y * blockDim.y + threadIdx.y;

        if (row > 0 && row < height - 1 && col > 0 && col < width - 1) {
            int idx = row * width + col;
            float mag = magnitude[idx];
            float ang = angle[idx];

            // Convert radians to degrees and normalize to [0, 180]
            float deg = ang * (180.0f / 3.14159f);
            if (deg < 0)
                deg += 180.0f;

            float n1, n2;

            // 1. Determine neighbors based on rounded direction
            if ((deg >= 0 && deg < 22.5) || (deg >= 157.5 && deg <= 180)) {
                // Horizontal (0 degrees)
                n1 = magnitude[idx - 1]; // Left
                n2 = magnitude[idx + 1]; // Right
            } else if (deg >= 22.5 && deg < 67.5) {
                // Positive Diagonal (45 degrees)
                n1 = magnitude[(row - 1) * width + (col + 1)]; // Top-Right
                n2 = magnitude[(row + 1) * width + (col - 1)]; // Bottom-Left
            } else if (deg >= 67.5 && deg < 112.5) {
                // Vertical (90 degrees)
                n1 = magnitude[(row - 1) * width + col]; // Top
                n2 = magnitude[(row + 1) * width + col]; // Bottom
            } else {
                // Negative Diagonal (135 degrees)
                n1 = magnitude[(row - 1) * width + (col - 1)]; // Top-Left
                n2 = magnitude[(row + 1) * width + (col + 1)]; // Bottom-Right
            }

            // 2. Suppress non-maxima
            if (mag >= n1 && mag >= n2) {
                output[idx] = mag;
            } else {
                output[idx] = 0.0f;
            }
        }
    }

    __global__ void fused_canny_edge_filter_chw_kernel(
        const float* __restrict__ input,
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
                    0.299f * input[idx] +
                    0.587f * input[idx + plane_size] +
                    0.114f * input[idx + 2 * plane_size];
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

    void launch_nms_kernel(const float* d_magnitude, const float* d_angle, float* d_output, const int height, const int width) {

        dim3 blockDim = {32, 32, 1};
        dim3 gridDim((width + blockDim.x - 1) / blockDim.x,
                     (height + blockDim.y - 1) / blockDim.y);

        nms_kernel<<<gridDim, blockDim>>>(d_magnitude, d_angle, d_output, height, width);
    }

    void launch_grayscale_filter(const float* d_input, float* d_output, const int height, const int width) {

        dim3 blockDim = {32, 32, 1};
        dim3 gridDim((width + blockDim.x - 1) / blockDim.x,
                     (height + blockDim.y - 1) / blockDim.y);

        rgb_to_grayscale<<<gridDim, blockDim>>>(d_input, d_output, height, width);
    }

    void launch_gausssian_blur(const float* d_input, float* d_output, const int k_size, const int height, const int width) {
        dim3 blockDim = {32, 32, 1};
        dim3 gridDim((width + blockDim.x - 1) / blockDim.x,
                     (height + blockDim.y - 1) / blockDim.y);

        gaussian_blur_5x5<<<gridDim, blockDim>>>(d_input, d_output, height, width);
    }

    void launch_laplacian_filter(const float* d_input, float* d_output, const int k_size, const int width, const int height) {

        dim3 blockDim = {32, 32, 1};
        dim3 gridDim((width + blockDim.x - 1) / blockDim.x,
                     (height + blockDim.y - 1) / blockDim.y);

        switch (k_size) {
        case 3:
            laplacian_filter_3x3<<<gridDim, blockDim>>>(d_input, d_output, height, width);
            break;
        }
    }

    void launch_sobel_gradient_filter(const float* d_input, float* d_magnitude, float* d_angle, const int height, const int width) {

        dim3 blockDim(32, 32, 1);
        dim3 gridDim((width + blockDim.x - 1) / blockDim.x,
                     (height + blockDim.y - 1) / blockDim.y);

        sobel_gradient_kernel<<<gridDim, blockDim>>>(d_input, d_magnitude, d_angle, height, width);
    }

    void launch_fused_canny_edge_filter_chw(
        const float* d_input_chw,
        float* d_output_hw,
        const int height,
        const int width,
        cudaStream_t stream) {
        dim3 blockDim(32, 32, 1);
        dim3 gridDim((width + blockDim.x - 1) / blockDim.x,
                     (height + blockDim.y - 1) / blockDim.y);

        fused_canny_edge_filter_chw_kernel<<<gridDim, blockDim, 0, stream>>>(d_input_chw, d_output_hw, height, width);
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

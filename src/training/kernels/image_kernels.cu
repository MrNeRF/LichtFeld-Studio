/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "image_kernels.hpp"
#include <stdio.h>

#include "cuda.h"
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
} // namespace lfs::training::kernels
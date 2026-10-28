/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#include "core/cuda_error.hpp"
#include "core/tensor/internal/cuda_stream_context.hpp"
#include "image_sharpen.hpp"
#include <algorithm>
#include <array>
#include <cmath>
#include <stdexcept>
#include <type_traits>

namespace lfs::io {
    namespace {
        constexpr int max_radius = 30;
        constexpr int row_width = 128, row_lines = 2;
        // The column pass covers a 32x32 tile; each of its 32x8 threads sharpens four pixels.
        constexpr int tile = 32, tile_lines = 8;
        constexpr int lap_x = 32, lap_y = 16;

        // Half of a normalized symmetric Gaussian; passed by value so concurrent loads never share weights.
        struct GaussWeights {
            float w[max_radius + 1];
            int radius;
        };

        template <typename T>
        __device__ __forceinline__ float load_value(const T value) {
            if constexpr (std::is_same_v<T, uint8_t>)
                return static_cast<float>(value) * (1.0f / 255.0f);
            else
                return value;
        }

        // Rounds like the loader's float->u8 and float->u16 conversions, so cached copies decode to the same values.
        template <typename T>
        __device__ __forceinline__ T store_value(const float value, const bool quantize_u16) {
            if constexpr (std::is_same_v<T, uint8_t>) {
                return static_cast<uint8_t>(fminf(fmaxf(value, 0.0f), 1.0f) * 255.0f + 0.5f);
            } else {
                // Full-precision float images may hold HDR values above 1.
                if (!quantize_u16)
                    return fmaxf(value, 0.0f);
                const float scaled = fminf(fmaxf(value * 65535.0f, 0.0f), 65535.0f);
                return static_cast<float>(static_cast<uint16_t>(scaled + 0.5f)) * (1.0f / 65535.0f);
            }
        }

        __device__ __forceinline__ float luma(const float r, const float g, const float b) {
            return 0.2126f * r + 0.7152f * g + 0.0722f * b;
        }

        template <typename T>
        __device__ __forceinline__ float luma_at(const T* image, const size_t index, const size_t plane) {
            return luma(load_value(image[index]), load_value(image[index + plane]), load_value(image[index + 2 * plane]));
        }

        // Adds the luma detail above `blur` to all three channels of one pixel.
        template <typename T>
        __device__ __forceinline__ void sharpen_pixel(const T* image, T* output, const size_t index, const size_t plane,
                                                      const float blur, const float gain, const float threshold,
                                                      const bool quantize_u16) {
            float rgb[3];
#pragma unroll
            for (int c = 0; c < 3; ++c)
                rgb[c] = load_value(image[index + c * plane]);
            float detail = luma(rgb[0], rgb[1], rgb[2]) - blur;
            if (fabsf(detail) < threshold)
                detail = 0.0f;
#pragma unroll
            for (int c = 0; c < 3; ++c)
                output[index + c * plane] = store_value<T>(rgb[c] + gain * detail, quantize_u16);
        }

        template <typename T>
        __global__ void usm_blur_rows(const T* __restrict__ image, float* __restrict__ blurred, const int width,
                                      const int height, const size_t plane, const GaussWeights weights) {
            __shared__ float lines[row_lines][row_width + 2 * max_radius];
            const int r = weights.radius;
            const int x0 = blockIdx.x * row_width;
            const int y = blockIdx.y * row_lines + threadIdx.y;
            float* const line = lines[threadIdx.y];
            const size_t row = static_cast<size_t>(min(y, height - 1)) * width;
            for (int i = threadIdx.x; i < row_width + 2 * r; i += row_width)
                line[i] = luma_at(image, row + min(max(x0 + i - r, 0), width - 1), plane);
            __syncthreads();
            const int x = x0 + threadIdx.x;
            if (x >= width || y >= height)
                return;
            const float* const center = line + threadIdx.x + r;
            float sum = weights.w[0] * center[0];
#pragma unroll
            for (int k = 1; k <= max_radius; ++k) {
                if (k > r)
                    break;
                sum += weights.w[k] * (center[-k] + center[k]);
            }
            blurred[row + x] = sum;
        }

        template <typename T>
        __global__ void usm_blur_columns_and_sharpen(const T* __restrict__ image, const float* __restrict__ blurred,
                                                     T* __restrict__ output, const int width, const int height,
                                                     const size_t plane, const GaussWeights weights, const float amount,
                                                     const float threshold, const bool quantize_u16) {
            __shared__ float columns[(tile + 2 * max_radius) * tile];
            const int r = weights.radius;
            const int x0 = blockIdx.x * tile, y0 = blockIdx.y * tile;
            const int x = x0 + threadIdx.x;
            const int source_x = min(x, width - 1);
            for (int i = threadIdx.y; i < tile + 2 * r; i += tile_lines)
                columns[i * tile + threadIdx.x] =
                    blurred[static_cast<size_t>(min(max(y0 + i - r, 0), height - 1)) * width + source_x];
            __syncthreads();
            if (x >= width)
                return;
            for (int i = threadIdx.y; i < tile && y0 + i < height; i += tile_lines) {
                const float* const center = columns + (i + r) * tile + threadIdx.x;
                float blur = weights.w[0] * center[0];
#pragma unroll
                for (int k = 1; k <= max_radius; ++k) {
                    if (k > r)
                        break;
                    blur += weights.w[k] * (center[-k * tile] + center[k * tile]);
                }
                sharpen_pixel(image, output, static_cast<size_t>(y0 + i) * width + x, plane, blur, amount, threshold,
                              quantize_u16);
            }
        }

        template <typename T>
        __global__ void laplacian_sharpen(const T* __restrict__ image, T* __restrict__ output, const int width,
                                          const int height, const size_t plane, const bool eight_neighbours,
                                          const float amount, const float threshold, const bool quantize_u16) {
            __shared__ float tile_luma[lap_y + 2][lap_x + 2];
            const int x0 = blockIdx.x * lap_x, y0 = blockIdx.y * lap_y;
            for (int i = threadIdx.y * lap_x + threadIdx.x; i < (lap_x + 2) * (lap_y + 2); i += lap_x * lap_y) {
                const int tx = i % (lap_x + 2), ty = i / (lap_x + 2);
                const int x = min(max(x0 + tx - 1, 0), width - 1);
                const int y = min(max(y0 + ty - 1, 0), height - 1);
                tile_luma[ty][tx] = luma_at(image, static_cast<size_t>(y) * width + x, plane);
            }
            __syncthreads();
            const int x = x0 + threadIdx.x, y = y0 + threadIdx.y;
            if (x >= width || y >= height)
                return;
            const int tx = threadIdx.x + 1, ty = threadIdx.y + 1;
            // Pairwise sums keep a flat neighbourhood exact.
            const float sides = (tile_luma[ty - 1][tx] + tile_luma[ty + 1][tx]) + (tile_luma[ty][tx - 1] + tile_luma[ty][tx + 1]);
            float blur = sides * 0.25f, gain = 4.0f * amount;
            if (eight_neighbours) {
                const float corners = (tile_luma[ty - 1][tx - 1] + tile_luma[ty - 1][tx + 1]) +
                                      (tile_luma[ty + 1][tx - 1] + tile_luma[ty + 1][tx + 1]);
                blur = (sides + corners) * 0.125f;
                gain = 8.0f * amount;
            }
            sharpen_pixel(image, output, static_cast<size_t>(y) * width + x, plane, blur, gain, threshold,
                          quantize_u16);
        }

        GaussWeights gaussian_weights(const float sigma) {
            GaussWeights weights{};
            weights.radius = std::min(max_radius, static_cast<int>(std::ceil(3.0f * sigma)));
            double total = 0.0;
            std::array<double, max_radius + 1> raw{};
            for (int k = 0; k <= weights.radius; ++k) {
                raw[k] = std::exp(-0.5 * k * k / (static_cast<double>(sigma) * sigma));
                total += k == 0 ? raw[k] : 2.0 * raw[k];
            }
            for (int k = 0; k <= weights.radius; ++k)
                weights.w[k] = static_cast<float>(raw[k] / total);
            return weights;
        }

        template <typename T>
        void launch(const lfs::core::Tensor& image, lfs::core::Tensor& output, const lfs::core::SharpenParams& params,
                    const bool quantize_u16, const cudaStream_t stream) {
            const int height = static_cast<int>(image.shape()[1]);
            const int width = static_cast<int>(image.shape()[2]);
            const size_t plane = static_cast<size_t>(width) * height;
            const auto* source = static_cast<const T*>(image.data_ptr());
            auto* target = static_cast<T*>(output.data_ptr());
            if (params.kernel == lfs::core::SharpenKernel::Usm) {
                const auto weights = gaussian_weights(params.radius);
                auto blurred = lfs::core::Tensor::empty({static_cast<size_t>(height), static_cast<size_t>(width)},
                                                        lfs::core::Device::CUDA, lfs::core::DataType::Float32);
                usm_blur_rows<T><<<dim3((width + row_width - 1) / row_width, (height + row_lines - 1) / row_lines),
                                   dim3(row_width, row_lines), 0, stream>>>(source, blurred.ptr<float>(), width,
                                                                            height, plane, weights);
                LFS_CUDA_CHECK_MSG(cudaGetLastError(), "blur image rows for sharpening");
                usm_blur_columns_and_sharpen<T><<<dim3((width + tile - 1) / tile, (height + tile - 1) / tile),
                                                  dim3(tile, tile_lines), 0, stream>>>(
                    source, blurred.ptr<float>(), target, width, height, plane, weights, params.amount,
                    params.threshold, quantize_u16);
                LFS_CUDA_CHECK_MSG(cudaGetLastError(), "sharpen image");
            } else {
                laplacian_sharpen<T><<<dim3((width + lap_x - 1) / lap_x, (height + lap_y - 1) / lap_y),
                                       dim3(lap_x, lap_y), 0, stream>>>(
                    source, target, width, height, plane, params.kernel == lfs::core::SharpenKernel::Laplacian8,
                    params.amount, params.threshold, quantize_u16);
                LFS_CUDA_CHECK_MSG(cudaGetLastError(), "sharpen image");
            }
        }
    } // namespace

    lfs::core::Tensor sharpen_image(const lfs::core::Tensor& image, const lfs::core::SharpenParams& params,
                                    const bool quantize_u16, const cudaStream_t stream) {
        using namespace lfs::core;
        if (image.ndim() != 3 || image.shape()[0] != 3 || image.device() != Device::CUDA ||
            (image.dtype() != DataType::UInt8 && image.dtype() != DataType::Float32))
            throw std::invalid_argument("Image sharpening needs a [3,H,W] UInt8 or Float32 CUDA image");
        if (!params.enabled() || image.numel() == 0)
            return image;
        const CUDAStreamGuard guard(stream);
        image.sync_to_stream(stream);
        const auto source = image.contiguous();
        auto output = Tensor::empty(source.shape(), Device::CUDA, source.dtype());
        if (source.dtype() == DataType::UInt8)
            launch<uint8_t>(source, output, params, quantize_u16, stream);
        else
            launch<float>(source, output, params, quantize_u16, stream);
        return output;
    }
} // namespace lfs::io

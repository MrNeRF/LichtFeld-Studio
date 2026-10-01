/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "core/assert.hpp"
#include "core/cuda_error.hpp"
#include "lfs/core/warp_reduce.cuh"
#include "thin_structure.hpp"
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <limits>
#include <nvtx3/nvToolsExt.h>
#include <type_traits>

namespace lfs::training::kernels {
    namespace {
        constexpr int BLOCK_SIZE = 256;
        constexpr int MAX_FILTER_RADIUS = 10;
        constexpr int HORIZONTAL_TILE_WIDTH = 128;
        constexpr int VERTICAL_TILE_WIDTH = 32;
        constexpr int VERTICAL_TILE_HEIGHT = 8;
        constexpr std::array<double, 4> RIDGE_SIGMAS{0.8, 1.2, 1.8, 2.5};
        constexpr float STRUCTURE_RESPONSE_CAP = 4.0f;
        struct DerivativeFilter {
            float g[2 * MAX_FILTER_RADIUS + 1];
            float d1[2 * MAX_FILTER_RADIUS + 1];
            float d2[2 * MAX_FILTER_RADIUS + 1];
            int radius;
            float sigma_sq;
        };

        __device__ int reflect_index(int index, const int length) {
            const int period = 2 * length;
            index %= period;
            if (index < 0)
                index += period;
            return index < length ? index : period - 1 - index;
        }

        template <typename T, bool ComputeLuminance>
        __global__ void hessian_horizontal(const T* input, float* luminance, float* horizontal,
                                           const int height, const int width, const DerivativeFilter filter) {
            __shared__ float tile[HORIZONTAL_TILE_WIDTH + 2 * MAX_FILTER_RADIUS];
            const int tiles_per_row = (width + HORIZONTAL_TILE_WIDTH - 1) / HORIZONTAL_TILE_WIDTH;
            const int row = blockIdx.x / tiles_per_row;
            const int tile_x = (blockIdx.x % tiles_per_row) * HORIZONTAL_TILE_WIDTH;
            for (int j = threadIdx.x; j < HORIZONTAL_TILE_WIDTH + 2 * filter.radius; j += blockDim.x) {
                const int source_x = tile_x + j - filter.radius;
                const int reflected_x = source_x >= 0 && source_x < width ? source_x : reflect_index(source_x, width);
                const int source = row * width + reflected_x;
                if constexpr (ComputeLuminance) {
                    const int n = height * width;
                    const float scale = std::is_same_v<T, uint8_t> ? 1.0f / 255.0f : 1.0f;
                    tile[j] = scale * (0.2126f * input[source] + 0.7152f * input[n + source] + 0.0722f * input[2 * n + source]);
                } else {
                    tile[j] = input[source];
                }
            }
            __syncthreads();
            const int x = tile_x + threadIdx.x;
            if (x >= width)
                return;
            float g = 0.0f, d1 = 0.0f, d2 = 0.0f;
            for (int k = -filter.radius; k <= filter.radius; ++k) {
                const float value = tile[threadIdx.x + k + filter.radius];
                const int j = k + filter.radius;
                g += value * filter.g[j];
                d1 += value * filter.d1[j];
                d2 += value * filter.d2[j];
            }
            const int i = row * width + x, n = height * width;
            if constexpr (ComputeLuminance)
                luminance[i] = tile[threadIdx.x + filter.radius];
            horizontal[i] = g;
            horizontal[n + i] = d1;
            horizontal[2 * n + i] = d2;
        }

        __global__ void hessian_vertical(const float* horizontal, float* output,
                                         const int height, const int width, const DerivativeFilter filter,
                                         const bool first_scale) {
            __shared__ float tile[3][VERTICAL_TILE_WIDTH * (VERTICAL_TILE_HEIGHT + 2 * MAX_FILTER_RADIUS)];
            const int tiles_per_row = (width + VERTICAL_TILE_WIDTH - 1) / VERTICAL_TILE_WIDTH;
            const int tile_x = (blockIdx.x % tiles_per_row) * VERTICAL_TILE_WIDTH;
            const int tile_y = (blockIdx.x / tiles_per_row) * VERTICAL_TILE_HEIGHT;
            const int thread = threadIdx.y * blockDim.x + threadIdx.x;
            const int n = height * width;
            const int tile_elements = VERTICAL_TILE_WIDTH * (VERTICAL_TILE_HEIGHT + 2 * filter.radius);
            for (int j = thread; j < tile_elements; j += blockDim.x * blockDim.y) {
                const int x = tile_x + j % VERTICAL_TILE_WIDTH;
                const int source_y = tile_y + j / VERTICAL_TILE_WIDTH - filter.radius;
                const int reflected_y = source_y >= 0 && source_y < height ? source_y : reflect_index(source_y, height);
                const int source = reflected_y * width + x;
                tile[0][j] = x < width ? horizontal[source] : 0.0f;
                tile[1][j] = x < width ? horizontal[n + source] : 0.0f;
                tile[2][j] = x < width ? horizontal[2 * n + source] : 0.0f;
            }
            __syncthreads();
            const int x = tile_x + threadIdx.x, y = tile_y + threadIdx.y;
            if (x >= width || y >= height)
                return;
            float hxx = 0.0f, hxy = 0.0f, hyy = 0.0f;
            for (int k = -filter.radius; k <= filter.radius; ++k) {
                const int source = (threadIdx.y + k + filter.radius) * VERTICAL_TILE_WIDTH + threadIdx.x;
                const int j = k + filter.radius;
                hxx += tile[2][source] * filter.g[j];
                hxy += tile[1][source] * filter.d1[j];
                hyy += tile[0][source] * filter.d2[j];
            }
            hxx *= filter.sigma_sq;
            hxy *= filter.sigma_sq;
            hyy *= filter.sigma_sq;
            const float disc = sqrtf(fmaxf((hxx - hyy) * (hxx - hyy) + 4.0f * hxy * hxy, 0.0f));
            const float a = fabsf(0.5f * (hxx + hyy + disc));
            const float b = fabsf(0.5f * (hxx + hyy - disc));
            const float hi = fmaxf(a, b), lo = fminf(a, b);
            const float response = hi * (hi - lo) / (hi + lo + 1e-6f);
            const int i = y * width + x;
            output[i] = first_scale ? response : fmaxf(output[i], response);
        }

        __global__ void structure_sum(const float* data, const float* luminance, float* partial, const int n) {
            float sum = 0.0f, minimum = std::numeric_limits<float>::infinity(), maximum = -std::numeric_limits<float>::infinity();
            for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x) {
                sum += data[i];
                minimum = fminf(minimum, luminance[i]);
                maximum = fmaxf(maximum, luminance[i]);
            }
            sum = lfs::core::warp_ops::block_reduce_sum(sum);
            minimum = lfs::core::warp_ops::block_reduce_min(minimum);
            maximum = lfs::core::warp_ops::block_reduce_max(maximum);
            if (threadIdx.x == 0) {
                partial[blockIdx.x] = sum;
                partial[1024 + blockIdx.x] = minimum;
                partial[2048 + blockIdx.x] = maximum;
            }
        }

        __global__ void normalize_structure(float* data, const float* partial, const int blocks, const int n) {
            __shared__ float inv_mean;
            float sum = 0.0f, minimum = std::numeric_limits<float>::infinity(), maximum = -std::numeric_limits<float>::infinity();
            for (int i = threadIdx.x; i < blocks; i += blockDim.x) {
                sum += partial[i];
                minimum = fminf(minimum, partial[1024 + i]);
                maximum = fmaxf(maximum, partial[2048 + i]);
            }
            sum = lfs::core::warp_ops::block_reduce_sum(sum);
            minimum = lfs::core::warp_ops::block_reduce_min(minimum);
            maximum = lfs::core::warp_ops::block_reduce_max(maximum);
            if (threadIdx.x == 0)
                inv_mean = sum > 0.0f && maximum > minimum ? static_cast<float>(n) / sum : 0.0f;
            __syncthreads();
            const int i = blockIdx.x * blockDim.x + threadIdx.x;
            if (i < n)
                data[i] *= inv_mean;
        }

        template <typename T>
        __global__ void weight_kernel(const float* structure, const T* base, float* output,
                                      const int height, const int width, const float gain,
                                      const bool valid_padding) {
            const int i = blockIdx.x * blockDim.x + threadIdx.x;
            if (i >= height * width)
                return;
            const int x = i % width, y = i / width;
            const bool inside = !valid_padding || height <= 10 || width <= 10 ||
                                (x >= 5 && x < width - 5 && y >= 5 && y < height - 5);
            float base_value = 1.0f;
            if (base != nullptr) {
                if constexpr (std::is_same_v<T, uint8_t>)
                    base_value = base[i] != 0 ? 1.0f : 0.0f;
                else
                    base_value = base[i];
            }
            output[i] = inside ? base_value * (1.0f + gain * fminf(structure[i], STRUCTURE_RESPONSE_CAP)) : 0.0f;
        }
    } // namespace

    float structure_base_denominator(const lfs::core::Tensor& base_weight, const int height, const int width,
                                     const bool valid_padding) {
        using namespace lfs::core;
        LFS_ASSERT(height > 0 && width > 0);
        float pixels;
        if (base_weight.is_valid()) {
            LFS_ASSERT(base_weight.dtype() == DataType::Float32 && base_weight.ndim() == 2);
            LFS_ASSERT(base_weight.shape()[0] == static_cast<size_t>(height) &&
                       base_weight.shape()[1] == static_cast<size_t>(width));
            pixels = base_weight.sum().item<float>();
        } else {
            pixels = static_cast<float>(valid_padding && height > 10 && width > 10
                                            ? (height - 10) * (width - 10)
                                            : height * width);
        }
        return 3.0f * pixels + 1e-8f;
    }

    void RidgeWorkspace::ensure_size(const size_t height, const size_t width) {
        if (luminance.is_valid() && luminance.shape() == lfs::core::TensorShape{height, width})
            return;
        luminance = lfs::core::Tensor::empty({height, width}, lfs::core::Device::CUDA);
        horizontal = lfs::core::Tensor::empty({3, height, width}, lfs::core::Device::CUDA);
        reduction = lfs::core::Tensor::empty({3072}, lfs::core::Device::CUDA);
    }

    void ridge_structure_map(const lfs::core::Tensor& image, lfs::core::Tensor& output,
                             RidgeWorkspace& workspace) {
        using namespace lfs::core;
        LFS_ASSERT(image.device() == Device::CUDA && image.ndim() == 3 && image.shape()[0] == 3);
        LFS_ASSERT(image.is_contiguous() && image.shape()[1] > 0 && image.shape()[2] > 0);
        LFS_ASSERT(image.dtype() == DataType::Float32 || image.dtype() == DataType::UInt8);
        const int height = static_cast<int>(image.shape()[1]);
        const int width = static_cast<int>(image.shape()[2]);
        const int n = height * width;
        workspace.ensure_size(static_cast<size_t>(height), static_cast<size_t>(width));
        LFS_ASSERT((output.shape() == TensorShape{static_cast<size_t>(height), static_cast<size_t>(width)}));
        LFS_ASSERT(output.dtype() == DataType::Float32 && output.device() == Device::CUDA);
        const auto stream = image.stream();
        workspace.luminance.set_stream(stream);
        workspace.horizontal.set_stream(stream);
        workspace.reduction.set_stream(stream);
        output.set_stream(stream);
        nvtxRangePushA("structure_map");
        const int blocks = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;
        const int horizontal_blocks = ((width + HORIZONTAL_TILE_WIDTH - 1) / HORIZONTAL_TILE_WIDTH) * height;
        const int vertical_blocks = ((width + VERTICAL_TILE_WIDTH - 1) / VERTICAL_TILE_WIDTH) *
                                    ((height + VERTICAL_TILE_HEIGHT - 1) / VERTICAL_TILE_HEIGHT);
        const dim3 vertical_threads(VERTICAL_TILE_WIDTH, VERTICAL_TILE_HEIGHT);
        bool first_scale = true;
        for (const double sigma : RIDGE_SIGMAS) {
            DerivativeFilter filter{};
            filter.radius = static_cast<int>(4.0 * sigma + 0.5);
            LFS_ASSERT(filter.radius <= MAX_FILTER_RADIUS);
            filter.sigma_sq = static_cast<float>(sigma * sigma);
            double sum = 0.0;
            for (int k = -filter.radius; k <= filter.radius; ++k)
                sum += std::exp(-0.5 * k * k / (sigma * sigma));
            for (int k = -filter.radius; k <= filter.radius; ++k) {
                const double g = std::exp(-0.5 * k * k / (sigma * sigma)) / sum;
                const int j = k + filter.radius;
                filter.g[j] = static_cast<float>(g);
                filter.d1[j] = static_cast<float>(k * g / (sigma * sigma));
                filter.d2[j] = static_cast<float>((k * k / (sigma * sigma) - 1.0) * g / (sigma * sigma));
            }
            if (first_scale) {
                if (image.dtype() == DataType::UInt8)
                    hessian_horizontal<uint8_t, true><<<horizontal_blocks, HORIZONTAL_TILE_WIDTH, 0, stream>>>(
                        image.ptr<uint8_t>(), workspace.luminance.ptr<float>(), workspace.horizontal.ptr<float>(), height, width, filter);
                else
                    hessian_horizontal<float, true><<<horizontal_blocks, HORIZONTAL_TILE_WIDTH, 0, stream>>>(
                        image.ptr<float>(), workspace.luminance.ptr<float>(), workspace.horizontal.ptr<float>(), height, width, filter);
            } else {
                hessian_horizontal<float, false><<<horizontal_blocks, HORIZONTAL_TILE_WIDTH, 0, stream>>>(
                    workspace.luminance.ptr<float>(), workspace.luminance.ptr<float>(), workspace.horizontal.ptr<float>(), height, width, filter);
            }
            hessian_vertical<<<vertical_blocks, vertical_threads, 0, stream>>>(workspace.horizontal.ptr<float>(), output.ptr<float>(),
                                                                               height, width, filter, first_scale);
            first_scale = false;
        }
        const int reduction_blocks = std::min(blocks, 1024);
        structure_sum<<<reduction_blocks, BLOCK_SIZE, 0, stream>>>(output.ptr<float>(), workspace.luminance.ptr<float>(), workspace.reduction.ptr<float>(), n);
        normalize_structure<<<blocks, BLOCK_SIZE, 0, stream>>>(output.ptr<float>(), workspace.reduction.ptr<float>(), reduction_blocks, n);
        nvtxRangePop();
        LFS_CUDA_LAUNCH_CHECK(stream, "training.structure.ridge");
    }

    void structure_photometric_weight(const lfs::core::Tensor& structure, const lfs::core::Tensor& base_weight,
                                      lfs::core::Tensor& output, const float gain,
                                      const bool valid_padding) {
        using namespace lfs::core;
        LFS_ASSERT(structure.device() == Device::CUDA && structure.ndim() == 2 && structure.dtype() == DataType::Float32);
        LFS_ASSERT(std::isfinite(gain) && gain >= 0.0f && gain <= 4.0f);
        LFS_ASSERT(!base_weight.is_valid() || (base_weight.device() == Device::CUDA && base_weight.shape() == structure.shape()));
        if (!output.is_valid() || output.shape() != structure.shape())
            output = Tensor::empty(structure.shape(), Device::CUDA);
        const int height = static_cast<int>(structure.shape()[0]);
        const int width = static_cast<int>(structure.shape()[1]);
        const int blocks = (height * width + BLOCK_SIZE - 1) / BLOCK_SIZE;
        const auto stream = structure.stream();
        output.set_stream(stream);
        if (base_weight.is_valid() && (base_weight.dtype() == DataType::UInt8 || base_weight.dtype() == DataType::Bool))
            weight_kernel<<<blocks, BLOCK_SIZE, 0, stream>>>(structure.ptr<float>(), base_weight.ptr<uint8_t>(), output.ptr<float>(),
                                                             height, width, gain, valid_padding);
        else {
            LFS_ASSERT(!base_weight.is_valid() || base_weight.dtype() == DataType::Float32);
            weight_kernel<<<blocks, BLOCK_SIZE, 0, stream>>>(structure.ptr<float>(), base_weight.is_valid() ? base_weight.ptr<float>() : nullptr,
                                                             output.ptr<float>(), height, width, gain, valid_padding);
        }
        LFS_CUDA_LAUNCH_CHECK(stream, "training.structure.weight");
    }
} // namespace lfs::training::kernels

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
#include <type_traits>

namespace lfs::training::kernels {
    namespace {
        constexpr int BLOCK_SIZE = 256;
        constexpr std::array<double, 4> RIDGE_SIGMAS{0.8, 1.2, 1.8, 2.5};
        constexpr float STRUCTURE_RESPONSE_CAP = 4.0f;
        struct DerivativeFilter {
            float g[21];
            float d1[21];
            float d2[21];
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

        template <typename T>
        __global__ void luminance_kernel(const T* image, float* output, const int n) {
            const int i = blockIdx.x * blockDim.x + threadIdx.x;
            if (i >= n)
                return;
            const float scale = std::is_same_v<T, uint8_t> ? 1.0f / 255.0f : 1.0f;
            output[i] = scale * (0.2126f * image[i] + 0.7152f * image[n + i] + 0.0722f * image[2 * n + i]);
        }

        __global__ void hessian_horizontal(const float* lum, float* horizontal,
                                           const int height, const int width, const DerivativeFilter filter) {
            const int i = blockIdx.x * blockDim.x + threadIdx.x;
            const int n = height * width;
            if (i >= n)
                return;
            const int x = i % width;
            const int row = i - x;
            float g = 0.0f, d1 = 0.0f, d2 = 0.0f;
            for (int k = -filter.radius; k <= filter.radius; ++k) {
                const float value = lum[row + reflect_index(x + k, width)];
                const int j = k + filter.radius;
                g += value * filter.g[j];
                d1 += value * filter.d1[j];
                d2 += value * filter.d2[j];
            }
            horizontal[i] = g;
            horizontal[n + i] = d1;
            horizontal[2 * n + i] = d2;
        }

        __global__ void hessian_vertical(const float* horizontal, float* output,
                                         const int height, const int width, const DerivativeFilter filter,
                                         const bool first_scale) {
            const int i = blockIdx.x * blockDim.x + threadIdx.x;
            const int n = height * width;
            if (i >= n)
                return;
            const int x = i % width, y = i / width;
            float hxx = 0.0f, hxy = 0.0f, hyy = 0.0f;
            for (int k = -filter.radius; k <= filter.radius; ++k) {
                const int source = reflect_index(y + k, height) * width + x;
                const int j = k + filter.radius;
                hxx += horizontal[2 * n + source] * filter.g[j];
                hxy += horizontal[n + source] * filter.d1[j];
                hyy += horizontal[source] * filter.d2[j];
            }
            hxx *= filter.sigma_sq;
            hxy *= filter.sigma_sq;
            hyy *= filter.sigma_sq;
            const float disc = sqrtf(fmaxf((hxx - hyy) * (hxx - hyy) + 4.0f * hxy * hxy, 0.0f));
            const float a = fabsf(0.5f * (hxx + hyy + disc));
            const float b = fabsf(0.5f * (hxx + hyy - disc));
            const float hi = fmaxf(a, b), lo = fminf(a, b);
            const float response = hi * (hi - lo) / (hi + lo + 1e-6f);
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
        const int blocks = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;
        if (image.dtype() == DataType::UInt8)
            luminance_kernel<<<blocks, BLOCK_SIZE, 0, stream>>>(image.ptr<uint8_t>(), workspace.luminance.ptr<float>(), n);
        else
            luminance_kernel<<<blocks, BLOCK_SIZE, 0, stream>>>(image.ptr<float>(), workspace.luminance.ptr<float>(), n);
        bool first_scale = true;
        for (const double sigma : RIDGE_SIGMAS) {
            DerivativeFilter filter{};
            filter.radius = static_cast<int>(4.0 * sigma + 0.5);
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
            hessian_horizontal<<<blocks, BLOCK_SIZE, 0, stream>>>(workspace.luminance.ptr<float>(),
                                                                  workspace.horizontal.ptr<float>(), height, width, filter);
            hessian_vertical<<<blocks, BLOCK_SIZE, 0, stream>>>(workspace.horizontal.ptr<float>(), output.ptr<float>(),
                                                                height, width, filter, first_scale);
            first_scale = false;
        }
        const int reduction_blocks = std::min(blocks, 1024);
        structure_sum<<<reduction_blocks, BLOCK_SIZE, 0, stream>>>(output.ptr<float>(), workspace.luminance.ptr<float>(), workspace.reduction.ptr<float>(), n);
        normalize_structure<<<blocks, BLOCK_SIZE, 0, stream>>>(output.ptr<float>(), workspace.reduction.ptr<float>(), reduction_blocks, n);
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

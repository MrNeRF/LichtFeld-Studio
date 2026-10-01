/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "core/assert.hpp"
#include "core/cuda_error.hpp"
#include "gradient_residual.hpp"
#include "lfs/core/warp_reduce.cuh"
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <type_traits>

namespace lfs::training::kernels {
    namespace {
        constexpr int BLOCK_SIZE = 256;
        constexpr int MAX_BLOCKS = 1024;

        __device__ float sobel_x(const int dx, const int dy) {
            return dx * (dy == 0 ? 2.0f : 1.0f) * 0.125f;
        }

        __device__ float sobel_y(const int dx, const int dy) {
            return dy * (dx == 0 ? 2.0f : 1.0f) * 0.125f;
        }

        template <typename T>
        __device__ float mask_weight(const T* mask, const int i) {
            if (!mask)
                return 1.0f;
            if constexpr (std::is_same_v<T, uint8_t>)
                return mask[i] != 0 ? 1.0f : 0.0f;
            else
                return mask[i];
        }

        template <typename T, typename M>
        __global__ void gradient_residual_forward(
            const float* image, const T* target, const M* mask,
            float* coefficients, float* partial,
            const int height, const int width, const float epsilon) {
            const int n = height * width;
            float loss_sum = 0.0f, weight_sum = 0.0f;
            for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x) {
                const int x = i % width, y = i / width;
                bool valid = x > 0 && x < width - 1 && y > 0 && y < height - 1;
                float gx = 0.0f, gy = 0.0f;
                if (valid) {
                    for (int dy = -1; dy <= 1; ++dy) {
                        for (int dx = -1; dx <= 1; ++dx) {
                            const int j = i + dy * width + dx;
                            valid = valid && mask_weight(mask, j) > 0.0f;
                            const float scale = std::is_same_v<T, uint8_t> ? 1.0f / 255.0f : 1.0f;
                            const float r = 0.2126f * (image[j] - scale * target[j]) +
                                            0.7152f * (image[n + j] - scale * target[n + j]) +
                                            0.0722f * (image[2 * n + j] - scale * target[2 * n + j]);
                            gx += sobel_x(dx, dy) * r;
                            gy += sobel_y(dx, dy) * r;
                        }
                    }
                }
                const float m = valid ? mask_weight(mask, i) : 0.0f;
                const float hx = hypotf(gx, epsilon);
                const float hy = hypotf(gy, epsilon);
                coefficients[i] = m * gx / hx;
                coefficients[n + i] = m * gy / hy;
                loss_sum += m * (hx + hy - 2.0f * epsilon);
                weight_sum += m;
            }
            loss_sum = lfs::core::warp_ops::block_reduce_sum(loss_sum);
            __syncthreads();
            weight_sum = lfs::core::warp_ops::block_reduce_sum(weight_sum);
            if (threadIdx.x == 0) {
                partial[blockIdx.x] = loss_sum;
                partial[MAX_BLOCKS + blockIdx.x] = weight_sum;
            }
        }

        __global__ void gradient_residual_reduce(const float* partial, float* totals, float* loss,
                                                 const int blocks, const float weight) {
            float loss_sum = 0.0f, weight_sum = 0.0f;
            for (int i = threadIdx.x; i < blocks; i += blockDim.x) {
                loss_sum += partial[i];
                weight_sum += partial[MAX_BLOCKS + i];
            }
            loss_sum = lfs::core::warp_ops::block_reduce_sum(loss_sum);
            __syncthreads();
            weight_sum = lfs::core::warp_ops::block_reduce_sum(weight_sum);
            if (threadIdx.x == 0) {
                totals[0] = weight_sum > 0.0f ? weight / (2.0f * weight_sum) : 0.0f;
                totals[1] = weight_sum;
                loss[0] = loss_sum * totals[0];
            }
        }

        __global__ void gradient_residual_backward(const float* coefficients, const float* totals,
                                                   float* gradient, const int height, const int width) {
            const int i = blockIdx.x * blockDim.x + threadIdx.x;
            const int n = height * width;
            if (i >= n)
                return;
            const int x = i % width, y = i / width;
            float value = 0.0f;
            for (int dy = -1; dy <= 1; ++dy) {
                for (int dx = -1; dx <= 1; ++dx) {
                    const int cx = x - dx, cy = y - dy;
                    if (cx > 0 && cx < width - 1 && cy > 0 && cy < height - 1) {
                        const int center = cy * width + cx;
                        value += sobel_x(dx, dy) * coefficients[center] +
                                 sobel_y(dx, dy) * coefficients[n + center];
                    }
                }
            }
            value *= totals[0];
            gradient[i] += 0.2126f * value;
            gradient[n + i] += 0.7152f * value;
            gradient[2 * n + i] += 0.0722f * value;
        }
    } // namespace

    void GradientResidualWorkspace::ensure_size(const size_t pixels) {
        using namespace lfs::core;
        if (coefficients.is_valid() && coefficients.numel() >= 2 * pixels)
            return;
        coefficients = Tensor::empty({2 * pixels}, Device::CUDA);
        if (!partial.is_valid()) {
            partial = Tensor::empty({2 * MAX_BLOCKS}, Device::CUDA);
            totals = Tensor::empty({2}, Device::CUDA);
            loss = Tensor::empty({1}, Device::CUDA);
        }
    }

    lfs::core::Tensor gradient_residual_loss_gradient(
        const lfs::core::Tensor& image, const lfs::core::Tensor& target,
        const lfs::core::Tensor& pixel_weight, lfs::core::Tensor& gradient,
        const float weight, GradientResidualWorkspace& workspace) {
        using namespace lfs::core;
        if (weight == 0.0f)
            return {};
        LFS_ASSERT(image.device() == Device::CUDA && image.dtype() == DataType::Float32);
        LFS_ASSERT(image.ndim() == 3 && image.shape()[0] == 3 && image.is_contiguous());
        LFS_ASSERT(target.shape() == image.shape() && target.device() == Device::CUDA && target.is_contiguous());
        LFS_ASSERT(target.dtype() == DataType::Float32 || target.dtype() == DataType::UInt8);
        LFS_ASSERT(gradient.shape() == image.shape() && gradient.device() == Device::CUDA);
        LFS_ASSERT(gradient.dtype() == DataType::Float32 && gradient.is_contiguous());
        LFS_ASSERT(std::isfinite(weight) && weight > 0.0f && weight <= 8.0f);
        const int height = static_cast<int>(image.shape()[1]), width = static_cast<int>(image.shape()[2]);
        LFS_ASSERT(height > 0 && width > 0);
        const int n = height * width;
        if (pixel_weight.is_valid()) {
            LFS_ASSERT((pixel_weight.shape() == TensorShape{static_cast<size_t>(height), static_cast<size_t>(width)}));
            LFS_ASSERT(pixel_weight.device() == Device::CUDA && pixel_weight.is_contiguous());
            LFS_ASSERT(pixel_weight.dtype() == DataType::Float32 || pixel_weight.dtype() == DataType::UInt8 ||
                       pixel_weight.dtype() == DataType::Bool);
        }
        workspace.ensure_size(static_cast<size_t>(n));
        const auto stream = image.stream();
        workspace.coefficients.set_stream(stream);
        workspace.partial.set_stream(stream);
        workspace.totals.set_stream(stream);
        workspace.loss.set_stream(stream);
        gradient.set_stream(stream);
        const int blocks = std::min((n + BLOCK_SIZE - 1) / BLOCK_SIZE, MAX_BLOCKS);
        auto launch = [&]<typename T>(const T* target_ptr) {
            if (pixel_weight.is_valid() && pixel_weight.dtype() != DataType::Float32) {
                gradient_residual_forward<<<blocks, BLOCK_SIZE, 0, stream>>>(
                    image.ptr<float>(), target_ptr, pixel_weight.ptr<uint8_t>(),
                    workspace.coefficients.ptr<float>(), workspace.partial.ptr<float>(), height, width, GRADIENT_LOSS_EPSILON);
            } else {
                const float* mask = pixel_weight.is_valid() ? pixel_weight.ptr<float>() : nullptr;
                gradient_residual_forward<<<blocks, BLOCK_SIZE, 0, stream>>>(
                    image.ptr<float>(), target_ptr, mask,
                    workspace.coefficients.ptr<float>(), workspace.partial.ptr<float>(), height, width, GRADIENT_LOSS_EPSILON);
            }
        };
        if (target.dtype() == DataType::UInt8)
            launch(target.ptr<uint8_t>());
        else
            launch(target.ptr<float>());
        gradient_residual_reduce<<<1, BLOCK_SIZE, 0, stream>>>(
            workspace.partial.ptr<float>(), workspace.totals.ptr<float>(), workspace.loss.ptr<float>(), blocks, weight);
        gradient_residual_backward<<<(n + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE, 0, stream>>>(
            workspace.coefficients.ptr<float>(), workspace.totals.ptr<float>(), gradient.ptr<float>(), height, width);
        LFS_CUDA_LAUNCH_CHECK(stream, "training.gradient_residual");
        return workspace.loss;
    }
} // namespace lfs::training::kernels

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
        constexpr int TILE_X = 32;
        constexpr int TILE_Y = 8;

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

        template <typename T>
        __device__ float residual_luminance(const float* image, const T* target, const int n, const int i) {
            const float scale = std::is_same_v<T, uint8_t> ? 1.0f / 255.0f : 1.0f;
            return 0.2126f * (image[i] - scale * target[i]) +
                   0.7152f * (image[n + i] - scale * target[n + i]) +
                   0.0722f * (image[2 * n + i] - scale * target[2 * n + i]);
        }

        __device__ bool interior(const int x, const int y, const int height, const int width) {
            return x > 0 && x < width - 1 && y > 0 && y < height - 1;
        }

        template <typename T, typename M>
        __global__ void gradient_residual_forward(
            const float* image, const T* target, const M* mask, float* partial,
            const int height, const int width, const float epsilon) {
            const int n = height * width;
            float loss_sum = 0.0f, weight_sum = 0.0f;
            for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x) {
                const int x = i % width, y = i / width;
                bool valid = interior(x, y, height, width);
                float gx = 0.0f, gy = 0.0f;
                if (valid) {
                    for (int dy = -1; dy <= 1; ++dy) {
                        for (int dx = -1; dx <= 1; ++dx) {
                            const int j = i + dy * width + dx;
                            valid = valid && mask_weight(mask, j) > 0.0f;
                            const float r = residual_luminance(image, target, n, j);
                            gx += sobel_x(dx, dy) * r;
                            gy += sobel_y(dx, dy) * r;
                        }
                    }
                }
                const float m = valid ? mask_weight(mask, i) : 0.0f;
                const float hx = hypotf(gx, epsilon);
                const float hy = hypotf(gy, epsilon);
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

        // Recomputes the forward coefficients of the tile and its one-pixel ring from a two-pixel residual halo,
        // so no full-resolution coefficient map is stored between the passes.
        template <typename T, typename M>
        __global__ void gradient_residual_backward(const float* image, const T* target, const M* mask,
                                                   const float* totals, float* gradient,
                                                   const int height, const int width, const float epsilon) {
            __shared__ float residual[TILE_Y + 4][TILE_X + 4];
            __shared__ float weight[TILE_Y + 4][TILE_X + 4];
            __shared__ float coefficient_x[TILE_Y + 2][TILE_X + 2];
            __shared__ float coefficient_y[TILE_Y + 2][TILE_X + 2];
            const int n = height * width;
            const int x0 = blockIdx.x * TILE_X, y0 = blockIdx.y * TILE_Y;
            const int thread = threadIdx.y * TILE_X + threadIdx.x;
            for (int k = thread; k < (TILE_Y + 4) * (TILE_X + 4); k += TILE_X * TILE_Y) {
                const int ly = k / (TILE_X + 4), lx = k % (TILE_X + 4);
                const int x = x0 + lx - 2, y = y0 + ly - 2;
                const bool inside = x >= 0 && x < width && y >= 0 && y < height;
                residual[ly][lx] = inside ? residual_luminance(image, target, n, y * width + x) : 0.0f;
                weight[ly][lx] = inside ? mask_weight(mask, y * width + x) : 0.0f;
            }
            __syncthreads();
            for (int k = thread; k < (TILE_Y + 2) * (TILE_X + 2); k += TILE_X * TILE_Y) {
                const int ly = k / (TILE_X + 2), lx = k % (TILE_X + 2);
                bool valid = interior(x0 + lx - 1, y0 + ly - 1, height, width);
                float gx = 0.0f, gy = 0.0f;
                if (valid) {
                    for (int dy = -1; dy <= 1; ++dy) {
                        for (int dx = -1; dx <= 1; ++dx) {
                            valid = valid && weight[ly + 1 + dy][lx + 1 + dx] > 0.0f;
                            const float r = residual[ly + 1 + dy][lx + 1 + dx];
                            gx += sobel_x(dx, dy) * r;
                            gy += sobel_y(dx, dy) * r;
                        }
                    }
                }
                const float m = valid ? weight[ly + 1][lx + 1] : 0.0f;
                coefficient_x[ly][lx] = m * gx / hypotf(gx, epsilon);
                coefficient_y[ly][lx] = m * gy / hypotf(gy, epsilon);
            }
            __syncthreads();
            const int x = x0 + threadIdx.x, y = y0 + threadIdx.y;
            if (x >= width || y >= height)
                return;
            const int i = y * width + x;
            float value = 0.0f;
            for (int dy = -1; dy <= 1; ++dy) {
                for (int dx = -1; dx <= 1; ++dx) {
                    if (interior(x - dx, y - dy, height, width)) {
                        const int ly = threadIdx.y - dy + 1, lx = threadIdx.x - dx + 1;
                        value += sobel_x(dx, dy) * coefficient_x[ly][lx] +
                                 sobel_y(dx, dy) * coefficient_y[ly][lx];
                    }
                }
            }
            value *= totals[0];
            gradient[i] += 0.2126f * value;
            gradient[n + i] += 0.7152f * value;
            gradient[2 * n + i] += 0.0722f * value;
        }
    } // namespace

    void GradientResidualWorkspace::ensure_allocated() {
        using namespace lfs::core;
        if (partial.is_valid())
            return;
        partial = Tensor::empty({2 * MAX_BLOCKS}, Device::CUDA);
        totals = Tensor::empty({2}, Device::CUDA);
        loss = Tensor::empty({1}, Device::CUDA);
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
        workspace.ensure_allocated();
        const auto stream = image.stream();
        workspace.partial.set_stream(stream);
        workspace.totals.set_stream(stream);
        workspace.loss.set_stream(stream);
        gradient.set_stream(stream);
        const int blocks = std::min((n + BLOCK_SIZE - 1) / BLOCK_SIZE, MAX_BLOCKS);
        const dim3 tiles((width + TILE_X - 1) / TILE_X, (height + TILE_Y - 1) / TILE_Y);
        const dim3 tile_threads(TILE_X, TILE_Y);
        auto launch = [&]<typename T, typename M>(const T* target_ptr, const M* mask) {
            gradient_residual_forward<<<blocks, BLOCK_SIZE, 0, stream>>>(
                image.ptr<float>(), target_ptr, mask, workspace.partial.ptr<float>(), height, width, GRADIENT_LOSS_EPSILON);
            LFS_CUDA_LAUNCH_CHECK(stream, "training.gradient_residual.forward");
            gradient_residual_reduce<<<1, BLOCK_SIZE, 0, stream>>>(
                workspace.partial.ptr<float>(), workspace.totals.ptr<float>(), workspace.loss.ptr<float>(), blocks, weight);
            LFS_CUDA_LAUNCH_CHECK(stream, "training.gradient_residual.reduce");
            gradient_residual_backward<<<tiles, tile_threads, 0, stream>>>(
                image.ptr<float>(), target_ptr, mask, workspace.totals.ptr<float>(), gradient.ptr<float>(),
                height, width, GRADIENT_LOSS_EPSILON);
            LFS_CUDA_LAUNCH_CHECK(stream, "training.gradient_residual.backward");
        };
        auto launch_masked = [&]<typename T>(const T* target_ptr) {
            if (pixel_weight.is_valid() && pixel_weight.dtype() != DataType::Float32)
                launch(target_ptr, pixel_weight.ptr<uint8_t>());
            else
                launch(target_ptr, pixel_weight.is_valid() ? pixel_weight.ptr<float>() : static_cast<const float*>(nullptr));
        };
        if (target.dtype() == DataType::UInt8)
            launch_masked(target.ptr<uint8_t>());
        else
            launch_masked(target.ptr<float>());
        return workspace.loss;
    }
} // namespace lfs::training::kernels

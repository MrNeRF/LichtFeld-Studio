/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#include "core/cuda/mask_normalization.cuh"
#include "core/cuda/mask_normalization.hpp"
#include "core/cuda_error.hpp"
#include "core/image_io.hpp"
#include "core/nn/ops.hpp"
#include "core/tensor/internal/cuda_stream_context.hpp"
#include "masked_image_fill.hpp"
#include <limits>
#include <stdexcept>
#include <type_traits>

namespace lfs::io {
    namespace {
        constexpr unsigned missing = 0xffffffffu;
        constexpr int tile = 16, radius = 16, extent = tile + 2 * radius;
        template <typename T>
        __device__ float alpha_value(T value) {
            if constexpr (std::is_same_v<T, uint8_t>)
                return lfs::core::cuda::normalize_mask_byte(value);
            else if constexpr (std::is_same_v<T, uint16_t>)
                return lfs::core::cuda::normalize_mask_u16_value(value);
            else
                return value;
        }
        template <typename T>
        __global__ void make_keep(const T* image, const float* mask, const uint8_t* mask_bytes, uint8_t* keep, unsigned* first,
                                  unsigned count, int channels, bool alpha, bool invert, float threshold, bool high_band) {
            const unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
            unsigned candidate = missing;
            if (i < count) {
                float value = alpha        ? alpha_value(image[static_cast<size_t>(i) * channels + 3])
                              : mask_bytes ? lfs::core::cuda::normalize_mask_byte(mask_bytes[i])
                                           : mask[i];
                if (invert)
                    value = __fsub_rn(1.f, value);
                const bool retained = high_band         ? value > (250.5f / 255.0f)
                                      : threshold > 0.f ? lfs::core::cuda::mask_value_at_least(value, threshold)
                                                        : value > 0.f;
                keep[i] = retained;
                if (retained)
                    candidate = i;
            }
            __shared__ unsigned minima[256];
            minima[threadIdx.x] = candidate;
            __syncthreads();
            for (unsigned step = blockDim.x / 2; step; step /= 2) {
                if (threadIdx.x < step)
                    minima[threadIdx.x] = min(minima[threadIdx.x], minima[threadIdx.x + step]);
                __syncthreads();
            }
            if (!threadIdx.x && minima[0] != missing)
                atomicMin(first, minima[0]);
        }
        template <typename T>
        __global__ void fill_rgb(T* image, const uint8_t* keep, const unsigned* first, int width, int height, int channels) {
            __shared__ uint8_t halo[extent * extent];
            __shared__ unsigned any_kept;
            const int tx = threadIdx.x % tile, ty = threadIdx.x / tile;
            const int bx = blockIdx.x * tile, by = blockIdx.y * tile;
            if (!threadIdx.x)
                any_kept = 0;
            __syncthreads();
            for (int j = threadIdx.x; j < extent * extent; j += tile * tile) {
                const int x = bx + j % extent - radius, y = by + j / extent - radius;
                const uint8_t k = x >= 0 && x < width && y >= 0 && y < height ? keep[y * width + x] : 0;
                halo[j] = k;
                if (k)
                    atomicOr(&any_kept, 1u);
            }
            __syncthreads();
            const int x = bx + tx, y = by + ty;
            if (x >= width || y >= height || keep[y * width + x])
                return;
            unsigned nearest = missing;
            int distance = 2 * radius * radius + 1;
            if (any_kept) {
                for (int r = 1; r <= radius; ++r) {
                    for (int dy = -r; dy <= r; ++dy) {
                        for (int dx = -r; dx <= r; dx += abs(dy) == r ? 1 : 2 * r) {
                            if (!halo[(ty + radius + dy) * extent + tx + radius + dx])
                                continue;
                            const int d = dx * dx + dy * dy;
                            const unsigned index = (y + dy) * width + x + dx;
                            if (d < distance || (d == distance && index < nearest)) {
                                distance = d;
                                nearest = index;
                            }
                        }
                    }
                    if (distance <= r * r)
                        break;
                }
            }
            if (nearest == missing)
                nearest = *first;
            // Reads target kept pixels only: they are never written by this kernel.
            for (int c = 0; c < 3; ++c)
                image[(static_cast<size_t>(y) * width + x) * channels + c] =
                    nearest == missing ? T{} : image[static_cast<size_t>(nearest) * channels + c];
        }
        template <typename T>
        void launch(lfs::core::Tensor& image, const lfs::core::Tensor& mask, const MaskedImageFill& policy,
                    uint8_t* keep, unsigned* first, cudaStream_t stream) {
            const int h = image.shape()[0], w = image.shape()[1], c = image.shape()[2];
            const float* values = policy.alpha || mask.dtype() == lfs::core::DataType::UInt8 ? nullptr : mask.ptr<float>();
            const uint8_t* bytes = !policy.alpha && mask.dtype() == lfs::core::DataType::UInt8 ? mask.ptr<uint8_t>() : nullptr;
            make_keep<<<(static_cast<size_t>(h) * w + 255) / 256, 256, 0, stream>>>(
                static_cast<const T*>(image.data_ptr()), values, bytes, keep, first,
                h * w, c, policy.alpha, policy.invert, policy.threshold, policy.keep_high_band);
            LFS_CUDA_CHECK_MSG(cudaGetLastError(), "classify source mask for image fill");
            fill_rgb<<<dim3((w + tile - 1) / tile, (h + tile - 1) / tile), tile * tile, 0, stream>>>(
                static_cast<T*>(image.data_ptr()), keep, first, w, h, c);
            LFS_CUDA_CHECK_MSG(cudaGetLastError(), "fill ignored image pixels");
        }
    } // namespace
    void fill_masked_image(lfs::core::Tensor& image, const MaskedImageFill& policy, cudaStream_t stream) {
        using namespace lfs::core;
        if (image.ndim() != 3 || (image.shape()[2] != 3 && image.shape()[2] != 4) ||
            image.device() != Device::CUDA || (policy.alpha && image.shape()[2] != 4))
            throw std::invalid_argument(masked_image_fill_error());
        const CUDAStreamGuard guard(stream);
        image.sync_to_stream(stream);
        Tensor mask;
        const size_t count = image.shape()[0] * image.shape()[1];
        // Preserve decoding when a source exceeds this preprocessing index domain.
        if (count > std::numeric_limits<int>::max())
            return;
        if (!policy.alpha) {
            mask = policy.mask.to(Device::CUDA).contiguous();
            if (mask.ndim() == 3 && mask.shape()[0] == 1)
                mask = mask.squeeze(0);
            else if (mask.ndim() == 3 && mask.shape()[2] == 1)
                mask = mask.squeeze(2);
            if (mask.ndim() != 2)
                return;
            const bool source_grid = mask.shape()[0] == image.shape()[0] && mask.shape()[1] == image.shape()[1];
            if (!source_grid) {
                const auto [target_width, target_height] = resized_image_dimensions(
                    static_cast<int>(image.shape()[1]), static_cast<int>(image.shape()[0]),
                    policy.resize_factor, policy.max_width);
                // Unknown grids retain their accepted source decode.
                if (mask.shape()[0] != static_cast<size_t>(target_height) ||
                    mask.shape()[1] != static_cast<size_t>(target_width))
                    return;
            }
            if (mask.dtype() != DataType::Float32 && mask.dtype() != DataType::UInt8)
                mask = mask.to(DataType::Float32);
            if (!source_grid) {
                if (mask.dtype() == DataType::UInt8)
                    mask = lfs::core::cuda::normalize_mask_bytes(mask);
                // An output-grid mask is expanded before source RGB fill.
                mask = nn::resize2d(mask.reshape({1, 1, static_cast<int>(mask.shape()[0]), static_cast<int>(mask.shape()[1])}),
                                    static_cast<int>(image.shape()[0]), static_cast<int>(image.shape()[1]),
                                    nn::ResizeMode::Nearest, nn::CoordTransform::Asymmetric)
                           .reshape({static_cast<int>(image.shape()[0]), static_cast<int>(image.shape()[1])});
            }
            mask.sync_to_stream(stream);
        }
        if (!count)
            return;
        auto keep = Tensor::empty({count}, Device::CUDA, DataType::UInt8);
        auto first = Tensor::empty({1}, Device::CUDA, DataType::Int32);
        LFS_CUDA_CHECK(cudaMemsetAsync(first.data_ptr(), 0xff, sizeof(unsigned), stream));
        auto* index = static_cast<unsigned*>(first.data_ptr());
        if (image.dtype() == DataType::UInt8)
            launch<uint8_t>(image, mask, policy, keep.ptr<uint8_t>(), index, stream);
        else if (image.dtype() == DataType::Float16)
            launch<uint16_t>(image, mask, policy, keep.ptr<uint8_t>(), index, stream);
        else if (image.dtype() == DataType::Float32)
            launch<float>(image, mask, policy, keep.ptr<uint8_t>(), index, stream);
        else
            throw std::invalid_argument(masked_image_fill_error());
    }
} // namespace lfs::io

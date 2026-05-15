/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "cuda_vulkan_interop.hpp"

namespace lfs::rendering::detail {
    namespace {
        __device__ __forceinline__ unsigned char toByte(const float value) {
            const float clamped = fminf(fmaxf(value, 0.0f), 1.0f);
            return static_cast<unsigned char>(clamped * 255.0f + 0.5f);
        }

        __device__ __forceinline__ unsigned char loadUInt8Channel(
            const unsigned char* source,
            const std::uint32_t width,
            const std::uint32_t height,
            const int channels,
            const CudaVulkanTensorLayout layout,
            const std::uint32_t x,
            const std::uint32_t y,
            const int channel) {
            if (channel >= channels) {
                return channel == 3 ? 255 : 0;
            }
            if (layout == CudaVulkanTensorLayout::Hwc) {
                return source[(static_cast<std::size_t>(y) * width + x) * channels + channel];
            }
            return source[(static_cast<std::size_t>(channel) * height + y) * width + x];
        }

        __device__ __forceinline__ unsigned char loadFloatChannel(
            const float* source,
            const std::uint32_t width,
            const std::uint32_t height,
            const int channels,
            const CudaVulkanTensorLayout layout,
            const std::uint32_t x,
            const std::uint32_t y,
            const int channel) {
            if (channel >= channels) {
                return channel == 3 ? 255 : 0;
            }
            const float value = layout == CudaVulkanTensorLayout::Hwc
                                    ? source[(static_cast<std::size_t>(y) * width + x) * channels + channel]
                                    : source[(static_cast<std::size_t>(channel) * height + y) * width + x];
            return toByte(value);
        }

        __global__ void copyTensorToSurfaceKernel(
            cudaSurfaceObject_t surface,
            const void* source,
            std::uint32_t width,
            std::uint32_t height,
            int channels,
            CudaVulkanTensorLayout layout,
            CudaVulkanTensorElementType element_type,
            bool flip_y) {
            const std::uint32_t x = blockIdx.x * blockDim.x + threadIdx.x;
            const std::uint32_t y = blockIdx.y * blockDim.y + threadIdx.y;
            if (x >= width || y >= height) {
                return;
            }

            uchar4 rgba{};
            if (element_type == CudaVulkanTensorElementType::UInt8) {
                const auto* bytes = static_cast<const unsigned char*>(source);
                rgba.x = loadUInt8Channel(bytes, width, height, channels, layout, x, y, 0);
                rgba.y = loadUInt8Channel(bytes, width, height, channels, layout, x, y, 1);
                rgba.z = loadUInt8Channel(bytes, width, height, channels, layout, x, y, 2);
                rgba.w = loadUInt8Channel(bytes, width, height, channels, layout, x, y, 3);
            } else {
                const auto* floats = static_cast<const float*>(source);
                rgba.x = loadFloatChannel(floats, width, height, channels, layout, x, y, 0);
                rgba.y = loadFloatChannel(floats, width, height, channels, layout, x, y, 1);
                rgba.z = loadFloatChannel(floats, width, height, channels, layout, x, y, 2);
                rgba.w = loadFloatChannel(floats, width, height, channels, layout, x, y, 3);
            }

            const std::uint32_t out_y = flip_y ? (height - 1u - y) : y;
            surf2Dwrite(rgba, surface, static_cast<int>(x * sizeof(uchar4)), static_cast<int>(out_y));
        }

        __global__ void copyTensorToSurfaceR32fKernel(
            cudaSurfaceObject_t surface,
            const float* source,
            std::uint32_t width,
            std::uint32_t height,
            int channels,
            CudaVulkanTensorLayout layout,
            bool flip_y) {
            const std::uint32_t x = blockIdx.x * blockDim.x + threadIdx.x;
            const std::uint32_t y = blockIdx.y * blockDim.y + threadIdx.y;
            if (x >= width || y >= height) {
                return;
            }
            // Read channel 0 from the source tensor regardless of CHW/HWC layout.
            const float value = layout == CudaVulkanTensorLayout::Hwc
                                    ? source[(static_cast<std::size_t>(y) * width + x) * channels]
                                    : source[(static_cast<std::size_t>(y) * width + x)];
            const std::uint32_t out_y = flip_y ? (height - 1u - y) : y;
            surf2Dwrite(value, surface, static_cast<int>(x * sizeof(float)), static_cast<int>(out_y));
        }

    } // namespace

    cudaError_t launchCudaVulkanCopyTensorToSurface(
        const cudaSurfaceObject_t surface,
        const void* source,
        const std::uint32_t width,
        const std::uint32_t height,
        const int channels,
        const CudaVulkanTensorLayout layout,
        const CudaVulkanTensorElementType element_type,
        const bool flip_y,
        const cudaStream_t stream) {
        if (surface == 0 || source == nullptr || width == 0 || height == 0) {
            return cudaErrorInvalidValue;
        }

        const dim3 block{16, 16, 1};
        const dim3 grid{
            (width + block.x - 1) / block.x,
            (height + block.y - 1) / block.y,
            1,
        };
        copyTensorToSurfaceKernel<<<grid, block, 0, stream>>>(
            surface,
            source,
            width,
            height,
            channels,
            layout,
            element_type,
            flip_y);
        return cudaGetLastError();
    }

    cudaError_t launchCudaVulkanCopyTensorToSurfaceR32f(
        const cudaSurfaceObject_t surface,
        const float* source,
        const std::uint32_t width,
        const std::uint32_t height,
        const int channels,
        const CudaVulkanTensorLayout layout,
        const bool flip_y,
        const cudaStream_t stream) {
        if (surface == 0 || source == nullptr || width == 0 || height == 0) {
            return cudaErrorInvalidValue;
        }
        const dim3 block{16, 16, 1};
        const dim3 grid{
            (width + block.x - 1) / block.x,
            (height + block.y - 1) / block.y,
            1,
        };
        copyTensorToSurfaceR32fKernel<<<grid, block, 0, stream>>>(
            surface, source, width, height, channels, layout, flip_y);
        return cudaGetLastError();
    }

} // namespace lfs::rendering::detail

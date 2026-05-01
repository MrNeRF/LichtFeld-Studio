/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include "core/tensor.hpp"
#include <cuda_runtime.h>
#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace lfs::rendering {

#ifdef _WIN32
    using CudaVulkanExternalHandle = void*;
    static constexpr CudaVulkanExternalHandle kInvalidCudaVulkanExternalHandle = nullptr;
#else
    using CudaVulkanExternalHandle = int;
    static constexpr CudaVulkanExternalHandle kInvalidCudaVulkanExternalHandle = -1;
#endif

    struct CudaVulkanExtent2D {
        std::uint32_t width = 0;
        std::uint32_t height = 0;
    };

    enum class CudaVulkanImageFormat : std::uint8_t {
        Rgba8Unorm,
    };

    struct CudaVulkanExternalImageImport {
        CudaVulkanExternalHandle memory_handle = kInvalidCudaVulkanExternalHandle;
        std::size_t allocation_size = 0;
        CudaVulkanExtent2D extent{};
        CudaVulkanImageFormat format = CudaVulkanImageFormat::Rgba8Unorm;
        bool dedicated_allocation = false;
    };

    struct CudaVulkanExternalSemaphoreImport {
        CudaVulkanExternalHandle semaphore_handle = kInvalidCudaVulkanExternalHandle;
        std::uint64_t initial_value = 0;
    };

    struct CudaVulkanRgba8HostBuffer {
        std::vector<std::uint8_t> pixels;
        std::string error;

        [[nodiscard]] explicit operator bool() const {
            return error.empty() && !pixels.empty();
        }
    };

    [[nodiscard]] CudaVulkanRgba8HostBuffer packTensorToRgba8Host(
        const lfs::core::Tensor& tensor,
        CudaVulkanExtent2D extent,
        cudaStream_t stream = nullptr);

    namespace detail {
        enum class CudaVulkanTensorLayout : std::uint8_t {
            Hwc,
            Chw,
        };

        enum class CudaVulkanTensorElementType : std::uint8_t {
            UInt8,
            Float32,
        };
    } // namespace detail

    class CudaVulkanInterop {
    public:
        CudaVulkanInterop() = default;
        CudaVulkanInterop(CudaVulkanExternalImageImport image,
                          CudaVulkanExternalSemaphoreImport semaphore);
        ~CudaVulkanInterop();

        CudaVulkanInterop(const CudaVulkanInterop&) = delete;
        CudaVulkanInterop& operator=(const CudaVulkanInterop&) = delete;
        CudaVulkanInterop(CudaVulkanInterop&& other) noexcept;
        CudaVulkanInterop& operator=(CudaVulkanInterop&& other) noexcept;

        [[nodiscard]] bool init(CudaVulkanExternalImageImport image,
                                CudaVulkanExternalSemaphoreImport semaphore);
        void reset();

        [[nodiscard]] bool valid() const;
        [[nodiscard]] const std::string& lastError() const { return last_error_; }
        [[nodiscard]] CudaVulkanExtent2D extent() const { return extent_; }
        [[nodiscard]] CudaVulkanImageFormat format() const { return format_; }
        [[nodiscard]] cudaSurfaceObject_t surfaceObject() const { return surface_; }
        [[nodiscard]] cudaArray_t cudaArray() const { return cuda_array_; }

        [[nodiscard]] lfs::core::Tensor view_as_tensor() const;
        [[nodiscard]] bool copyViewToSurface(cudaStream_t stream = nullptr) const;
        [[nodiscard]] bool copyTensorToSurface(const lfs::core::Tensor& tensor,
                                               cudaStream_t stream = nullptr) const;
        [[nodiscard]] bool wait(std::uint64_t value, cudaStream_t stream = nullptr) const;
        [[nodiscard]] bool signal(std::uint64_t value, cudaStream_t stream = nullptr) const;

    private:
        [[nodiscard]] bool fail(std::string message) const;
        [[nodiscard]] bool failCuda(const char* operation, cudaError_t status) const;

        cudaExternalMemory_t cuda_mem_ = nullptr;
        cudaMipmappedArray_t cuda_mip_ = nullptr;
        cudaArray_t cuda_array_ = nullptr;
        cudaSurfaceObject_t surface_ = 0;
        cudaExternalSemaphore_t cuda_timeline_ = nullptr;
        CudaVulkanExtent2D extent_{};
        CudaVulkanImageFormat format_ = CudaVulkanImageFormat::Rgba8Unorm;
        mutable lfs::core::Tensor staging_tensor_;
        mutable lfs::core::Tensor upload_source_;
        mutable std::string last_error_;
    };

} // namespace lfs::rendering

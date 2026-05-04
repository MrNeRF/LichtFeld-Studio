/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include "core/tensor.hpp"
#include <array>
#include <cstddef>
#include <cstdint>
#include <cuda_runtime.h>
#include <optional>
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
        R32Sfloat,
    };

    struct CudaVulkanExternalImageImport {
        CudaVulkanExternalHandle memory_handle = kInvalidCudaVulkanExternalHandle;
        std::size_t allocation_size = 0;
        CudaVulkanExtent2D extent{};
        CudaVulkanImageFormat format = CudaVulkanImageFormat::Rgba8Unorm;
        bool dedicated_allocation = false;
    };

    struct CudaVulkanExternalBufferImport {
        CudaVulkanExternalHandle memory_handle = kInvalidCudaVulkanExternalHandle;
        std::size_t allocation_size = 0;
        std::size_t size = 0;
        bool dedicated_allocation = false;
    };

    struct CudaVulkanExternalSemaphoreImport {
        CudaVulkanExternalHandle semaphore_handle = kInvalidCudaVulkanExternalHandle;
        std::uint64_t initial_value = 0;
    };

    // Records the Vulkan physical-device UUID expected to back any subsequent
    // CUDA/Vulkan interop. Call once at startup, after Vulkan device selection.
    // The first interop init() then verifies the current CUDA device matches;
    // mismatch is a hard failure (returns false from init() with a clear error).
    void setExpectedVulkanDeviceUuid(const std::array<std::uint8_t, 16>& uuid);

    // Lazily verifies the CUDA current device's UUID against the value passed
    // to setExpectedVulkanDeviceUuid(). Returns std::nullopt on success, or an
    // error message on mismatch / missing setup. Result is cached.
    [[nodiscard]] std::optional<std::string> verifyCudaMatchesVulkanDevice();

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

    // Standalone CUDA-importable timeline semaphore. Used to gate Vulkan work
    // on CUDA work without a CPU-side cudaStreamSynchronize: the producing
    // CUDA stream signals a monotonic value, and a Vulkan submit waits on the
    // same value via the Vulkan-side VkSemaphore. The Vulkan handle is owned
    // by VulkanContext (createExternalTimelineSemaphore); this class only
    // owns the imported cudaExternalSemaphore_t.
    class CudaTimelineSemaphore {
    public:
        CudaTimelineSemaphore() = default;
        ~CudaTimelineSemaphore();

        CudaTimelineSemaphore(const CudaTimelineSemaphore&) = delete;
        CudaTimelineSemaphore& operator=(const CudaTimelineSemaphore&) = delete;
        CudaTimelineSemaphore(CudaTimelineSemaphore&& other) noexcept;
        CudaTimelineSemaphore& operator=(CudaTimelineSemaphore&& other) noexcept;

        [[nodiscard]] bool init(CudaVulkanExternalSemaphoreImport semaphore);
        void reset();

        [[nodiscard]] bool valid() const { return cuda_timeline_ != nullptr; }
        [[nodiscard]] const std::string& lastError() const { return last_error_; }

        [[nodiscard]] bool cudaSignal(std::uint64_t value, cudaStream_t stream = nullptr) const;
        [[nodiscard]] bool cudaWait(std::uint64_t value, cudaStream_t stream = nullptr) const;

    private:
        [[nodiscard]] bool fail(std::string message) const;
        [[nodiscard]] bool failCuda(const char* operation, cudaError_t status) const;

        cudaExternalSemaphore_t cuda_timeline_ = nullptr;
        mutable std::string last_error_;
    };

    class CudaVulkanBufferInterop {
    public:
        CudaVulkanBufferInterop() = default;
        explicit CudaVulkanBufferInterop(CudaVulkanExternalBufferImport buffer);
        ~CudaVulkanBufferInterop();

        CudaVulkanBufferInterop(const CudaVulkanBufferInterop&) = delete;
        CudaVulkanBufferInterop& operator=(const CudaVulkanBufferInterop&) = delete;
        CudaVulkanBufferInterop(CudaVulkanBufferInterop&& other) noexcept;
        CudaVulkanBufferInterop& operator=(CudaVulkanBufferInterop&& other) noexcept;

        [[nodiscard]] bool init(CudaVulkanExternalBufferImport buffer);
        void reset();

        [[nodiscard]] bool valid() const;
        [[nodiscard]] const std::string& lastError() const { return last_error_; }
        [[nodiscard]] void* devicePointer() const { return device_ptr_; }
        [[nodiscard]] std::size_t size() const { return size_; }
        [[nodiscard]] bool copyFromTensor(const lfs::core::Tensor& tensor,
                                          std::size_t byte_count,
                                          cudaStream_t stream = nullptr) const;
        // Offset-aware variant for coalesced layouts where one CUDA-imported
        // VkBuffer holds multiple sub-regions (xyz | rotations | scales+opacs |
        // sh) instead of four separate allocations.
        [[nodiscard]] bool copyFromTensor(const lfs::core::Tensor& tensor,
                                          std::size_t byte_count,
                                          std::size_t dst_offset,
                                          cudaStream_t stream) const;

    private:
        [[nodiscard]] bool fail(std::string message) const;
        [[nodiscard]] bool failCuda(const char* operation, cudaError_t status) const;

        cudaExternalMemory_t cuda_mem_ = nullptr;
        void* device_ptr_ = nullptr;
        std::size_t allocation_size_ = 0;
        std::size_t size_ = 0;
        mutable lfs::core::Tensor upload_source_;
        mutable std::string last_error_;
    };

} // namespace lfs::rendering

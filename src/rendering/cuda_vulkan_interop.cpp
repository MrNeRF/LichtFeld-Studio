/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "cuda_vulkan_interop.hpp"

#include "core/tensor/internal/cuda_stream_context.hpp"
#include "image_layout.hpp"

#include <algorithm>
#include <format>
#include <stdexcept>
#include <utility>

#ifdef _WIN32
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>
#else
#include <unistd.h>
#endif

namespace lfs::rendering {
    namespace {
#ifdef _WIN32
        constexpr cudaExternalMemoryHandleType kCudaExternalMemoryHandleType =
            cudaExternalMemoryHandleTypeOpaqueWin32;
        constexpr cudaExternalSemaphoreHandleType kCudaExternalSemaphoreHandleType =
            cudaExternalSemaphoreHandleTypeTimelineSemaphoreWin32;
#else
        constexpr cudaExternalMemoryHandleType kCudaExternalMemoryHandleType =
            cudaExternalMemoryHandleTypeOpaqueFd;
        constexpr cudaExternalSemaphoreHandleType kCudaExternalSemaphoreHandleType =
            cudaExternalSemaphoreHandleTypeTimelineSemaphoreFd;
#endif

        [[nodiscard]] bool nativeHandleValid(const CudaVulkanExternalHandle handle) {
#ifdef _WIN32
            return handle != nullptr;
#else
            return handle >= 0;
#endif
        }

        void closeNativeHandle(CudaVulkanExternalHandle& handle) {
            if (!nativeHandleValid(handle)) {
                return;
            }
#ifdef _WIN32
            CloseHandle(static_cast<HANDLE>(handle));
            handle = nullptr;
#else
            ::close(handle);
            handle = -1;
#endif
        }

        class NativeHandleOwner {
        public:
            explicit NativeHandleOwner(const CudaVulkanExternalHandle handle)
                : handle_(handle) {}

            ~NativeHandleOwner() {
                closeNativeHandle(handle_);
            }

            NativeHandleOwner(const NativeHandleOwner&) = delete;
            NativeHandleOwner& operator=(const NativeHandleOwner&) = delete;

            [[nodiscard]] CudaVulkanExternalHandle get() const { return handle_; }

            void release() {
                handle_ = kInvalidCudaVulkanExternalHandle;
            }

        private:
            CudaVulkanExternalHandle handle_ = kInvalidCudaVulkanExternalHandle;
        };

        [[nodiscard]] cudaChannelFormatDesc channelDescForFormat(const CudaVulkanImageFormat format) {
            switch (format) {
            case CudaVulkanImageFormat::Rgba8Unorm:
                return cudaCreateChannelDesc(8, 8, 8, 8, cudaChannelFormatKindUnsigned);
            }
            return {};
        }

        [[nodiscard]] const char* formatName(const CudaVulkanImageFormat format) {
            switch (format) {
            case CudaVulkanImageFormat::Rgba8Unorm:
                return "RGBA8_UNORM";
            }
            return "unknown";
        }

        [[nodiscard]] bool formatSupported(const CudaVulkanImageFormat format) {
            return format == CudaVulkanImageFormat::Rgba8Unorm;
        }
    } // namespace

    namespace detail {
        [[nodiscard]] cudaError_t launchCudaVulkanCopyTensorToSurface(
            cudaSurfaceObject_t surface,
            const void* source,
            std::uint32_t width,
            std::uint32_t height,
            int channels,
            CudaVulkanTensorLayout layout,
            CudaVulkanTensorElementType element_type,
            cudaStream_t stream);
    } // namespace detail

    CudaVulkanInterop::CudaVulkanInterop(CudaVulkanExternalImageImport image,
                                         CudaVulkanExternalSemaphoreImport semaphore) {
        if (!init(std::move(image), std::move(semaphore))) {
            throw std::runtime_error(last_error_);
        }
    }

    CudaVulkanInterop::~CudaVulkanInterop() {
        reset();
    }

    CudaVulkanInterop::CudaVulkanInterop(CudaVulkanInterop&& other) noexcept {
        *this = std::move(other);
    }

    CudaVulkanInterop& CudaVulkanInterop::operator=(CudaVulkanInterop&& other) noexcept {
        if (this == &other) {
            return *this;
        }

        reset();
        cuda_mem_ = std::exchange(other.cuda_mem_, nullptr);
        cuda_mip_ = std::exchange(other.cuda_mip_, nullptr);
        cuda_array_ = std::exchange(other.cuda_array_, nullptr);
        surface_ = std::exchange(other.surface_, 0);
        cuda_timeline_ = std::exchange(other.cuda_timeline_, nullptr);
        extent_ = std::exchange(other.extent_, {});
        format_ = std::exchange(other.format_, CudaVulkanImageFormat::Rgba8Unorm);
        staging_tensor_ = std::move(other.staging_tensor_);
        upload_source_ = std::move(other.upload_source_);
        last_error_ = std::move(other.last_error_);
        return *this;
    }

    bool CudaVulkanInterop::init(CudaVulkanExternalImageImport image,
                                 CudaVulkanExternalSemaphoreImport semaphore) {
        reset();
        last_error_.clear();

        if (!nativeHandleValid(image.memory_handle)) {
            return fail("CUDA/Vulkan external image import requires a valid memory handle");
        }
        if (!nativeHandleValid(semaphore.semaphore_handle)) {
            return fail("CUDA/Vulkan external semaphore import requires a valid semaphore handle");
        }
        if (image.allocation_size == 0 || image.extent.width == 0 || image.extent.height == 0) {
            return fail("CUDA/Vulkan external image import requires non-zero allocation and extent");
        }
        if (!formatSupported(image.format)) {
            return fail(std::format("CUDA/Vulkan external image format {} is unsupported",
                                    formatName(image.format)));
        }
        int cuda_device = 0;
        cudaError_t status = cudaGetDevice(&cuda_device);
        if (status != cudaSuccess) {
            return failCuda("cudaGetDevice", status);
        }
        int timeline_interop_supported = 0;
        status = cudaDeviceGetAttribute(&timeline_interop_supported,
                                        cudaDevAttrTimelineSemaphoreInteropSupported,
                                        cuda_device);
        if (status != cudaSuccess) {
            return failCuda("cudaDeviceGetAttribute(cudaDevAttrTimelineSemaphoreInteropSupported)", status);
        }
        if (timeline_interop_supported == 0) {
            return fail("CUDA device does not support external timeline semaphore interop");
        }

        NativeHandleOwner memory_handle(image.memory_handle);
        NativeHandleOwner semaphore_handle(semaphore.semaphore_handle);

        cudaExternalMemoryHandleDesc memory_desc{};
        memory_desc.type = kCudaExternalMemoryHandleType;
        memory_desc.size = image.allocation_size;
        if (image.dedicated_allocation) {
            memory_desc.flags = cudaExternalMemoryDedicated;
        }
#ifdef _WIN32
        memory_desc.handle.win32.handle = memory_handle.get();
#else
        memory_desc.handle.fd = memory_handle.get();
#endif

        status = cudaImportExternalMemory(&cuda_mem_, &memory_desc);
        if (status != cudaSuccess) {
            reset();
            return failCuda("cudaImportExternalMemory", status);
        }
#ifndef _WIN32
        memory_handle.release();
#endif

        cudaExternalMemoryMipmappedArrayDesc array_desc{};
        array_desc.offset = 0;
        array_desc.formatDesc = channelDescForFormat(image.format);
        array_desc.extent = make_cudaExtent(image.extent.width, image.extent.height, 0);
        array_desc.flags = cudaArraySurfaceLoadStore;
        array_desc.numLevels = 1;

        status = cudaExternalMemoryGetMappedMipmappedArray(&cuda_mip_, cuda_mem_, &array_desc);
        if (status != cudaSuccess) {
            reset();
            return failCuda("cudaExternalMemoryGetMappedMipmappedArray", status);
        }

        status = cudaGetMipmappedArrayLevel(&cuda_array_, cuda_mip_, 0);
        if (status != cudaSuccess) {
            reset();
            return failCuda("cudaGetMipmappedArrayLevel", status);
        }

        cudaResourceDesc resource_desc{};
        resource_desc.resType = cudaResourceTypeArray;
        resource_desc.res.array.array = cuda_array_;
        status = cudaCreateSurfaceObject(&surface_, &resource_desc);
        if (status != cudaSuccess) {
            reset();
            return failCuda("cudaCreateSurfaceObject", status);
        }

        cudaExternalSemaphoreHandleDesc semaphore_desc{};
        semaphore_desc.type = kCudaExternalSemaphoreHandleType;
#ifdef _WIN32
        semaphore_desc.handle.win32.handle = semaphore_handle.get();
#else
        semaphore_desc.handle.fd = semaphore_handle.get();
#endif

        status = cudaImportExternalSemaphore(&cuda_timeline_, &semaphore_desc);
        if (status != cudaSuccess) {
            reset();
            return failCuda("cudaImportExternalSemaphore", status);
        }
#ifndef _WIN32
        semaphore_handle.release();
#endif

        extent_ = image.extent;
        format_ = image.format;
        return true;
    }

    void CudaVulkanInterop::reset() {
        staging_tensor_ = {};
        upload_source_ = {};
        if (surface_ != 0) {
            cudaDestroySurfaceObject(surface_);
            surface_ = 0;
        }
        cuda_array_ = nullptr;
        if (cuda_mip_ != nullptr) {
            cudaFreeMipmappedArray(cuda_mip_);
            cuda_mip_ = nullptr;
        }
        if (cuda_mem_ != nullptr) {
            cudaDestroyExternalMemory(cuda_mem_);
            cuda_mem_ = nullptr;
        }
        if (cuda_timeline_ != nullptr) {
            cudaDestroyExternalSemaphore(cuda_timeline_);
            cuda_timeline_ = nullptr;
        }
        extent_ = {};
        format_ = CudaVulkanImageFormat::Rgba8Unorm;
    }

    bool CudaVulkanInterop::valid() const {
        return cuda_mem_ != nullptr &&
               cuda_mip_ != nullptr &&
               cuda_array_ != nullptr &&
               surface_ != 0 &&
               cuda_timeline_ != nullptr &&
               extent_.width > 0 &&
               extent_.height > 0;
    }

    lfs::core::Tensor CudaVulkanInterop::view_as_tensor() const {
        if (!valid()) {
            last_error_ = "CUDA/Vulkan interop target is not initialized";
            return {};
        }
        const lfs::core::TensorShape shape{
            static_cast<std::size_t>(extent_.height),
            static_cast<std::size_t>(extent_.width),
            std::size_t{4},
        };
        if (!staging_tensor_.is_valid() ||
            staging_tensor_.shape() != shape ||
            staging_tensor_.dtype() != lfs::core::DataType::UInt8 ||
            staging_tensor_.device() != lfs::core::Device::CUDA) {
            staging_tensor_ = lfs::core::Tensor::empty(shape, lfs::core::Device::CUDA, lfs::core::DataType::UInt8);
            staging_tensor_.set_name("cuda_vulkan_interop_staging");
        }
        return staging_tensor_;
    }

    bool CudaVulkanInterop::copyViewToSurface(const cudaStream_t stream) const {
        if (!staging_tensor_.is_valid()) {
            return fail("CUDA/Vulkan interop staging tensor has not been requested");
        }
        return copyTensorToSurface(staging_tensor_, stream);
    }

    bool CudaVulkanInterop::copyTensorToSurface(const lfs::core::Tensor& tensor,
                                                const cudaStream_t stream) const {
        last_error_.clear();
        if (!valid()) {
            return fail("CUDA/Vulkan interop target is not initialized");
        }
        if (!tensor.is_valid() || tensor.ndim() != 3) {
            return fail("CUDA/Vulkan interop copy requires a valid 3D tensor");
        }

        const ImageLayout layout = detectImageLayout(tensor);
        if (layout == ImageLayout::Unknown) {
            return fail("CUDA/Vulkan interop copy received an unsupported tensor layout");
        }

        const int width = imageWidth(tensor, layout);
        const int height = imageHeight(tensor, layout);
        const int channels = imageChannels(tensor, layout);
        if (width != static_cast<int>(extent_.width) || height != static_cast<int>(extent_.height)) {
            return fail(std::format("CUDA/Vulkan interop copy size mismatch: tensor {}x{}, target {}x{}",
                                    width,
                                    height,
                                    extent_.width,
                                    extent_.height));
        }
        if (channels != 1 && channels != 3 && channels != 4) {
            return fail(std::format("CUDA/Vulkan interop copy requires 1, 3, or 4 channels, got {}",
                                    channels));
        }

        bool owns_prepared_source = false;
        lfs::core::Tensor prepared = tensor;
        if (prepared.dtype() != lfs::core::DataType::UInt8 &&
            prepared.dtype() != lfs::core::DataType::Float32) {
            prepared = prepared.to(lfs::core::DataType::Float32);
            owns_prepared_source = true;
        }
        if (prepared.device() != lfs::core::Device::CUDA) {
            prepared = prepared.to(lfs::core::Device::CUDA, stream);
            owns_prepared_source = true;
        }
        if (!prepared.is_contiguous()) {
            prepared = prepared.contiguous();
            owns_prepared_source = true;
        }
        if (owns_prepared_source) {
            upload_source_ = std::move(prepared);
        } else {
            upload_source_ = prepared;
        }
        prepared = upload_source_;

        const void* data = prepared.data_ptr();
        if (data == nullptr) {
            return fail("CUDA/Vulkan interop copy received a tensor with null data");
        }

        const auto tensor_layout = layout == ImageLayout::HWC
                                       ? detail::CudaVulkanTensorLayout::Hwc
                                       : detail::CudaVulkanTensorLayout::Chw;
        const auto element_type = prepared.dtype() == lfs::core::DataType::UInt8
                                      ? detail::CudaVulkanTensorElementType::UInt8
                                      : detail::CudaVulkanTensorElementType::Float32;

        lfs::core::waitForCUDAStream(stream, prepared.stream());
        const cudaError_t status = detail::launchCudaVulkanCopyTensorToSurface(
            surface_,
            data,
            extent_.width,
            extent_.height,
            channels,
            tensor_layout,
            element_type,
            stream);
        return failCuda("copy tensor to CUDA surface", status);
    }

    bool CudaVulkanInterop::wait(const std::uint64_t value, const cudaStream_t stream) const {
        last_error_.clear();
        if (cuda_timeline_ == nullptr) {
            return fail("CUDA/Vulkan timeline semaphore is not initialized");
        }

        cudaExternalSemaphoreWaitParams params{};
        params.params.fence.value = value;
        const cudaError_t status = cudaWaitExternalSemaphoresAsync(&cuda_timeline_, &params, 1, stream);
        return failCuda("cudaWaitExternalSemaphoresAsync", status);
    }

    bool CudaVulkanInterop::signal(const std::uint64_t value, const cudaStream_t stream) const {
        last_error_.clear();
        if (cuda_timeline_ == nullptr) {
            return fail("CUDA/Vulkan timeline semaphore is not initialized");
        }

        cudaExternalSemaphoreSignalParams params{};
        params.params.fence.value = value;
        const cudaError_t status = cudaSignalExternalSemaphoresAsync(&cuda_timeline_, &params, 1, stream);
        return failCuda("cudaSignalExternalSemaphoresAsync", status);
    }

    bool CudaVulkanInterop::fail(std::string message) const {
        last_error_ = std::move(message);
        return false;
    }

    bool CudaVulkanInterop::failCuda(const char* const operation, const cudaError_t status) const {
        if (status == cudaSuccess) {
            return true;
        }
        last_error_ = std::format("{} failed: {} ({})",
                                  operation,
                                  cudaGetErrorName(status),
                                  cudaGetErrorString(status));
        return false;
    }

} // namespace lfs::rendering

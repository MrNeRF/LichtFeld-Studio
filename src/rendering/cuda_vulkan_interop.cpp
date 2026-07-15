/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "cuda_vulkan_interop.hpp"

#include "core/tensor/internal/cuda_stream_context.hpp"
#include "core/tensor/internal/memory_pool.hpp"
#include "image_layout.hpp"

#include <algorithm>
#include <cstring>
#include <format>
#include <limits>
#include <mutex>
#include <optional>
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
        std::mutex g_device_uuid_mutex;
        std::optional<std::array<std::uint8_t, 16>> g_expected_vk_uuid;
        bool g_device_match_resolved = false;
        bool g_device_match_ok = false;
        std::string g_device_match_error;

        std::string formatUuid(const std::array<std::uint8_t, 16>& uuid) {
            std::string s;
            s.reserve(36);
            for (std::size_t i = 0; i < uuid.size(); ++i) {
                if (i == 4 || i == 6 || i == 8 || i == 10) {
                    s.push_back('-');
                }
                std::array<char, 3> buf{};
                std::snprintf(buf.data(), buf.size(), "%02x", uuid[i]);
                s.append(buf.data(), 2);
            }
            return s;
        }

        [[nodiscard]] std::string withSourceLocation(
            std::string message,
            const core::SourceSite location) {
            return std::format("{} ({}:{})",
                               std::move(message),
                               location.file_name(),
                               location.line());
        }

        [[nodiscard]] bool setFailure(
            std::string& last_error,
            std::string message,
            const core::SourceSite location = LFS_SOURCE_SITE_CURRENT()) {
            last_error = withSourceLocation(std::move(message), location);
            return false;
        }

        [[nodiscard]] bool setCudaFailure(
            std::string& last_error,
            const char* const operation,
            const cudaError_t status,
            const core::SourceSite location = LFS_SOURCE_SITE_CURRENT()) {
            if (status == cudaSuccess) {
                return true;
            }
            return setFailure(
                last_error,
                std::format("{} failed: {} ({})",
                            operation,
                            cudaGetErrorName(status),
                            cudaGetErrorString(status)),
                location);
        }

        [[nodiscard]] std::string nativeHandleString(const CudaVulkanExternalHandle handle) {
#ifdef _WIN32
            return std::format("{:#x}", reinterpret_cast<std::uintptr_t>(handle));
#else
            return std::format("fd={}", handle);
#endif
        }

        [[nodiscard]] std::size_t imageBytesPerPixel(const CudaVulkanImageFormat format) {
            switch (format) {
            case CudaVulkanImageFormat::Rgba8Unorm: return 4;
            case CudaVulkanImageFormat::R32Sfloat: return 4;
            }
            return 0;
        }
    } // namespace

    void setExpectedVulkanDeviceUuid(const std::array<std::uint8_t, 16>& uuid) {
        std::lock_guard lk(g_device_uuid_mutex);
        g_expected_vk_uuid = uuid;
        g_device_match_resolved = false;
        g_device_match_ok = false;
        g_device_match_error.clear();
    }

    std::optional<std::string> verifyCudaMatchesVulkanDevice() {
        std::lock_guard lk(g_device_uuid_mutex);
        if (g_device_match_resolved) {
            if (g_device_match_ok) {
                return std::nullopt;
            }
            return g_device_match_error;
        }
        g_device_match_resolved = true;
        if (!g_expected_vk_uuid) {
            g_device_match_error =
                "Vulkan device UUID was not registered before CUDA/Vulkan interop init; "
                "call setExpectedVulkanDeviceUuid() once at startup";
            return g_device_match_error;
        }
        int cuda_device = 0;
        cudaError_t status = cudaGetDevice(&cuda_device);
        if (status != cudaSuccess) {
            g_device_match_error = std::format("cudaGetDevice failed: {} ({})",
                                               cudaGetErrorName(status), cudaGetErrorString(status));
            return g_device_match_error;
        }
        cudaDeviceProp props{};
        status = cudaGetDeviceProperties(&props, cuda_device);
        if (status != cudaSuccess) {
            g_device_match_error = std::format("cudaGetDeviceProperties failed: {} ({})",
                                               cudaGetErrorName(status), cudaGetErrorString(status));
            return g_device_match_error;
        }
        std::array<std::uint8_t, 16> cuda_uuid_bytes{};
        std::memcpy(cuda_uuid_bytes.data(), props.uuid.bytes, 16);
        if (cuda_uuid_bytes != *g_expected_vk_uuid) {
            g_device_match_error = std::format(
                "CUDA device {} (UUID {}) does not match the selected Vulkan physical device (UUID {}). "
                "Set CUDA_VISIBLE_DEVICES to expose the same GPU to both APIs.",
                cuda_device,
                formatUuid(cuda_uuid_bytes),
                formatUuid(*g_expected_vk_uuid));
            return g_device_match_error;
        }
        g_device_match_ok = true;
        return std::nullopt;
    }

    CudaVulkanUploadStream::~CudaVulkanUploadStream() {
        reset();
    }

    CudaVulkanUploadStream::CudaVulkanUploadStream(CudaVulkanUploadStream&& other) noexcept {
        *this = std::move(other);
    }

    CudaVulkanUploadStream& CudaVulkanUploadStream::operator=(CudaVulkanUploadStream&& other) noexcept {
        if (this == &other) {
            return *this;
        }
        reset();
        stream_ = std::exchange(other.stream_, nullptr);
        last_error_ = std::move(other.last_error_);
        return *this;
    }

    bool CudaVulkanUploadStream::init() {
        reset();
        last_error_.clear();
        if (auto error = verifyCudaMatchesVulkanDevice(); error) {
            last_error_ = std::move(*error);
            return false;
        }
        cudaStream_t created_stream = nullptr;
        const cudaError_t status =
            cudaStreamCreateWithFlags(&created_stream, cudaStreamNonBlocking);
        if (!setCudaFailure(last_error_, "cudaStreamCreateWithFlags(cudaStreamNonBlocking)", status)) {
            return false;
        }
        stream_ = created_stream;
        return true;
    }

    bool CudaVulkanUploadStream::synchronize() {
        last_error_.clear();
        if (stream_ == nullptr) {
            return true;
        }
        return setCudaFailure(last_error_,
                              "cudaStreamSynchronize(CUDA/Vulkan upload)",
                              cudaStreamSynchronize(stream_));
    }

    void CudaVulkanUploadStream::reset() noexcept {
        if (stream_ != nullptr) {
            // Image uploads record this lane against every tensor they read.
            // Remove those allocator references while the stream is still
            // valid; otherwise a later tensor free can bridge through a
            // destroyed cudaStream_t.
            lfs::core::CudaMemoryPool::instance().release_stream(stream_);
            (void)cudaStreamDestroy(stream_);
            stream_ = nullptr;
        }
        last_error_.clear();
    }

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
            case CudaVulkanImageFormat::R32Sfloat:
                return cudaCreateChannelDesc(32, 0, 0, 0, cudaChannelFormatKindFloat);
            }
            return {};
        }

        [[nodiscard]] const char* formatName(const CudaVulkanImageFormat format) {
            switch (format) {
            case CudaVulkanImageFormat::Rgba8Unorm:
                return "RGBA8_UNORM";
            case CudaVulkanImageFormat::R32Sfloat:
                return "R32_SFLOAT";
            }
            return "unknown";
        }

        [[nodiscard]] bool formatSupported(const CudaVulkanImageFormat format) {
            return format == CudaVulkanImageFormat::Rgba8Unorm ||
                   format == CudaVulkanImageFormat::R32Sfloat;
        }

        struct PreparedCudaImageTensor {
            lfs::core::Tensor tensor;
            int width = 0;
            int height = 0;
            int channels = 0;
            detail::CudaVulkanTensorLayout layout = detail::CudaVulkanTensorLayout::Hwc;
            detail::CudaVulkanTensorElementType element_type = detail::CudaVulkanTensorElementType::UInt8;
        };

        [[nodiscard]] bool prepareCudaImageTensor(const lfs::core::Tensor& tensor,
                                                  const CudaVulkanExtent2D extent,
                                                  const cudaStream_t stream,
                                                  PreparedCudaImageTensor& out,
                                                  std::string& error) {
            if (!tensor.is_valid() || tensor.ndim() != 3) {
                error = "CUDA/Vulkan image copy requires a valid 3D tensor";
                return false;
            }

            const ImageLayout layout = detectImageLayout(tensor);
            if (layout == ImageLayout::Unknown) {
                error = "CUDA/Vulkan image copy received an unsupported tensor layout";
                return false;
            }

            const int width = imageWidth(tensor, layout);
            const int height = imageHeight(tensor, layout);
            const int channels = imageChannels(tensor, layout);
            if (width != static_cast<int>(extent.width) || height != static_cast<int>(extent.height)) {
                error = std::format("CUDA/Vulkan image copy size mismatch: tensor {}x{}, target {}x{}",
                                    width,
                                    height,
                                    extent.width,
                                    extent.height);
                return false;
            }
            if (channels != 1 && channels != 3 && channels != 4) {
                error = std::format("CUDA/Vulkan image copy requires 1, 3, or 4 channels, got {}",
                                    channels);
                return false;
            }

            lfs::core::Tensor prepared = tensor;
            if (prepared.dtype() != lfs::core::DataType::UInt8 &&
                prepared.dtype() != lfs::core::DataType::Float32) {
                prepared = prepared.to(lfs::core::DataType::Float32);
            }
            if (prepared.device() != lfs::core::Device::CUDA) {
                prepared = prepared.to(lfs::core::Device::CUDA, stream);
            }
            if (!prepared.is_contiguous()) {
                prepared = prepared.contiguous();
            }

            out.tensor = std::move(prepared);
            out.width = width;
            out.height = height;
            out.channels = channels;
            out.layout = layout == ImageLayout::HWC
                             ? detail::CudaVulkanTensorLayout::Hwc
                             : detail::CudaVulkanTensorLayout::Chw;
            out.element_type = out.tensor.dtype() == lfs::core::DataType::UInt8
                                   ? detail::CudaVulkanTensorElementType::UInt8
                                   : detail::CudaVulkanTensorElementType::Float32;
            error.clear();
            return true;
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
            bool flip_y,
            const cudaStream_t stream);
        [[nodiscard]] cudaError_t launchCudaVulkanCopyTensorToSurfaceR32f(
            cudaSurfaceObject_t surface,
            const float* source,
            std::uint32_t width,
            std::uint32_t height,
            int channels,
            CudaVulkanTensorLayout layout,
            bool flip_y,
            const cudaStream_t stream);
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
        last_signaled_ = std::exchange(other.last_signaled_, 0);
        last_waited_ = std::exchange(other.last_waited_, 0);
        allocation_size_ = std::exchange(other.allocation_size_, 0);
        cuda_visible_size_ = std::exchange(other.cuda_visible_size_, 0);
        extent_ = std::exchange(other.extent_, {});
        format_ = std::exchange(other.format_, CudaVulkanImageFormat::Rgba8Unorm);
        upload_source_ = std::move(other.upload_source_);
        last_error_ = std::move(other.last_error_);
        return *this;
    }

    bool CudaVulkanInterop::init(CudaVulkanExternalImageImport image,
                                 CudaVulkanExternalSemaphoreImport semaphore) {
        reset();
        last_error_.clear();

        if (auto err = verifyCudaMatchesVulkanDevice(); err) {
            return setFailure(last_error_, *err);
        }
        if (!nativeHandleValid(image.memory_handle)) {
            return setFailure(last_error_, std::format(
                                               "CUDA/Vulkan external image import requires a valid memory handle (memory_handle={}, allocation_size={}, extent={}x{}, format={})",
                                               nativeHandleString(image.memory_handle),
                                               image.allocation_size,
                                               image.extent.width,
                                               image.extent.height,
                                               formatName(image.format)));
        }
        if (!nativeHandleValid(semaphore.semaphore_handle)) {
            return setFailure(last_error_, std::format(
                                               "CUDA/Vulkan external image import requires a valid timeline handle (semaphore_handle={}, initial_value={})",
                                               nativeHandleString(semaphore.semaphore_handle),
                                               semaphore.initial_value));
        }
        if (image.allocation_size == 0 || image.extent.width == 0 || image.extent.height == 0) {
            return setFailure(last_error_, std::format(
                                               "CUDA/Vulkan external image import requires a non-zero allocation and extent (allocation_size={}, extent={}x{}, format={})",
                                               image.allocation_size,
                                               image.extent.width,
                                               image.extent.height,
                                               formatName(image.format)));
        }
        if (!formatSupported(image.format)) {
            return setFailure(last_error_,
                              std::format("CUDA/Vulkan external image format {} is unsupported",
                                          formatName(image.format)));
        }
        const std::size_t bytes_per_pixel = imageBytesPerPixel(image.format);
        const std::size_t width = image.extent.width;
        const std::size_t height = image.extent.height;
        if (bytes_per_pixel == 0 || width > std::numeric_limits<std::size_t>::max() / height ||
            width * height > std::numeric_limits<std::size_t>::max() / bytes_per_pixel) {
            return setFailure(last_error_, std::format(
                                               "CUDA-visible image byte size overflows host sizing (extent={}x{}, bytes_per_pixel={}, allocation_size={})",
                                               width,
                                               height,
                                               bytes_per_pixel,
                                               image.allocation_size));
        }
        const std::size_t cuda_visible_size = width * height * bytes_per_pixel;
        if (image.allocation_size < cuda_visible_size) {
            return setFailure(last_error_, std::format(
                                               "Vulkan image allocation is smaller than the CUDA-visible payload (vulkan_allocation_size={}, cuda_visible_size={}, extent={}x{}, bytes_per_pixel={}, format={})",
                                               image.allocation_size,
                                               cuda_visible_size,
                                               width,
                                               height,
                                               bytes_per_pixel,
                                               formatName(image.format)));
        }
        int cuda_device = 0;
        cudaError_t status = cudaGetDevice(&cuda_device);
        if (status != cudaSuccess) {
            return setCudaFailure(last_error_, "cudaGetDevice", status);
        }
        int timeline_interop_supported = 0;
        status = cudaDeviceGetAttribute(&timeline_interop_supported,
                                        cudaDevAttrTimelineSemaphoreInteropSupported,
                                        cuda_device);
        if (status != cudaSuccess) {
            return setCudaFailure(
                last_error_,
                "cudaDeviceGetAttribute(cudaDevAttrTimelineSemaphoreInteropSupported)",
                status);
        }
        if (timeline_interop_supported == 0) {
            return setFailure(last_error_, std::format(
                                               "CUDA device does not support external timeline semaphore interop (cuda_device={}, timeline_interop_supported={})",
                                               cuda_device,
                                               timeline_interop_supported));
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
            return setCudaFailure(last_error_, "cudaImportExternalMemory", status);
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
            return setCudaFailure(last_error_, "cudaExternalMemoryGetMappedMipmappedArray", status);
        }

        status = cudaGetMipmappedArrayLevel(&cuda_array_, cuda_mip_, 0);
        if (status != cudaSuccess) {
            reset();
            return setCudaFailure(last_error_, "cudaGetMipmappedArrayLevel", status);
        }

        cudaResourceDesc resource_desc{};
        resource_desc.resType = cudaResourceTypeArray;
        resource_desc.res.array.array = cuda_array_;
        status = cudaCreateSurfaceObject(&surface_, &resource_desc);
        if (status != cudaSuccess) {
            reset();
            return setCudaFailure(last_error_, "cudaCreateSurfaceObject", status);
        }

        // cudaExternalSemaphoreHandleDesc has no initialValue field, so CUDA cannot validate or
        // communicate a non-zero Vulkan initial value during import.
        if (semaphore.initial_value != 0) {
            reset();
            return setFailure(last_error_, std::format(
                                               "CUDA/Vulkan timeline import requires Vulkan initialValue == 0 because CUDA cannot import it (initial_value={}, semaphore_handle={})",
                                               semaphore.initial_value,
                                               nativeHandleString(semaphore.semaphore_handle)));
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
            return setCudaFailure(last_error_, "cudaImportExternalSemaphore", status);
        }
#ifndef _WIN32
        semaphore_handle.release();
#endif
        last_signaled_ = semaphore.initial_value;
        last_waited_ = 0;
        allocation_size_ = image.allocation_size;
        cuda_visible_size_ = cuda_visible_size;

        extent_ = image.extent;
        format_ = image.format;
        return true;
    }

    void CudaVulkanInterop::reset() {
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
        last_signaled_ = 0;
        last_waited_ = 0;
        allocation_size_ = 0;
        cuda_visible_size_ = 0;
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

    bool CudaVulkanInterop::copyTensorToSurface(const lfs::core::Tensor& tensor,
                                                const cudaStream_t stream,
                                                const bool flip_y) const {
        last_error_.clear();
        if (!valid()) {
            return setFailure(last_error_, std::format(
                                               "CUDA/Vulkan image copy requires a complete imported target (cuda_memory={:#x}, mipmapped_array={:#x}, array={:#x}, surface={:#x}, timeline={:#x}, extent={}x{}, allocation_size={}, cuda_visible_size={})",
                                               reinterpret_cast<std::uintptr_t>(cuda_mem_),
                                               reinterpret_cast<std::uintptr_t>(cuda_mip_),
                                               reinterpret_cast<std::uintptr_t>(cuda_array_),
                                               static_cast<std::uint64_t>(surface_),
                                               reinterpret_cast<std::uintptr_t>(cuda_timeline_),
                                               extent_.width,
                                               extent_.height,
                                               allocation_size_,
                                               cuda_visible_size_));
        }
        if (stream == nullptr) {
            return setFailure(last_error_, std::format(
                                               "CUDA/Vulkan image copy requires an explicit non-default CUDA stream (stream={:#x}, extent={}x{}, format={}, allocation_size={}, cuda_visible_size={})",
                                               reinterpret_cast<std::uintptr_t>(stream),
                                               extent_.width,
                                               extent_.height,
                                               formatName(format_),
                                               allocation_size_,
                                               cuda_visible_size_));
        }
        PreparedCudaImageTensor prepared{};
        if (!prepareCudaImageTensor(tensor, extent_, stream, prepared, last_error_)) {
            return false;
        }
        upload_source_ = std::move(prepared.tensor);

        const void* data = upload_source_.data_ptr();
        if (data == nullptr) {
            return setFailure(last_error_, std::format(
                                               "CUDA/Vulkan image copy received a tensor with null storage (data={:#x}, tensor_bytes={}, tensor_rank={}, target_extent={}x{}, target_visible_bytes={})",
                                               reinterpret_cast<std::uintptr_t>(data),
                                               upload_source_.bytes(),
                                               upload_source_.shape().rank(),
                                               extent_.width,
                                               extent_.height,
                                               cuda_visible_size_));
        }

        upload_source_.sync_to_stream(stream);
        cudaError_t status = cudaSuccess;
        if (format_ == CudaVulkanImageFormat::R32Sfloat) {
            if (prepared.element_type != detail::CudaVulkanTensorElementType::Float32) {
                return setFailure(last_error_, std::format(
                                                   "CUDA/Vulkan R32_SFLOAT surface requires Float32 tensor elements (observed_element_type={}, required_element_type={}, channels={}, extent={}x{})",
                                                   static_cast<int>(prepared.element_type),
                                                   static_cast<int>(detail::CudaVulkanTensorElementType::Float32),
                                                   prepared.channels,
                                                   extent_.width,
                                                   extent_.height));
            }
            status = detail::launchCudaVulkanCopyTensorToSurfaceR32f(
                surface_,
                static_cast<const float*>(data),
                extent_.width,
                extent_.height,
                prepared.channels,
                prepared.layout,
                flip_y,
                stream);
        } else {
            status = detail::launchCudaVulkanCopyTensorToSurface(
                surface_,
                data,
                extent_.width,
                extent_.height,
                prepared.channels,
                prepared.layout,
                prepared.element_type,
                flip_y,
                stream);
        }
        return setCudaFailure(last_error_, "copy tensor to CUDA surface", status);
    }

    bool CudaVulkanInterop::wait(const std::uint64_t value, const cudaStream_t stream) const {
        last_error_.clear();
        if (cuda_timeline_ == nullptr) {
            return setFailure(last_error_, std::format(
                                               "CUDA/Vulkan timeline wait requires an imported semaphore (timeline={:#x}, requested_value={}, previous_wait_value={}, stream={:#x})",
                                               reinterpret_cast<std::uintptr_t>(cuda_timeline_),
                                               value,
                                               last_waited_,
                                               reinterpret_cast<std::uintptr_t>(stream)));
        }
        if (stream == nullptr) {
            return setFailure(last_error_, std::format(
                                               "CUDA/Vulkan timeline wait requires an explicit non-default CUDA stream (timeline={:#x}, requested_value={}, previous_wait_value={}, stream={:#x})",
                                               reinterpret_cast<std::uintptr_t>(cuda_timeline_),
                                               value,
                                               last_waited_,
                                               reinterpret_cast<std::uintptr_t>(stream)));
        }
        if (value == 0 || value <= last_waited_) {
            return setFailure(last_error_, std::format(
                                               "CUDA/Vulkan timeline waits must increase strictly and remain non-zero (timeline={:#x}, requested_value={}, previous_wait_value={}, stream={:#x})",
                                               reinterpret_cast<std::uintptr_t>(cuda_timeline_),
                                               value,
                                               last_waited_,
                                               reinterpret_cast<std::uintptr_t>(stream)));
        }

        cudaExternalSemaphoreWaitParams params{};
        params.params.fence.value = value;
        const cudaError_t status =
            cudaWaitExternalSemaphoresAsync(&cuda_timeline_, &params, 1, stream);
        if (!setCudaFailure(last_error_, "cudaWaitExternalSemaphoresAsync", status)) {
            return false;
        }
        last_waited_ = value;
        return true;
    }

    bool CudaVulkanInterop::signal(const std::uint64_t value, const cudaStream_t stream) const {
        last_error_.clear();
        if (cuda_timeline_ == nullptr) {
            return setFailure(last_error_, std::format(
                                               "CUDA/Vulkan timeline signal requires an imported semaphore (timeline={:#x}, requested_value={}, previous_signal_value={}, stream={:#x})",
                                               reinterpret_cast<std::uintptr_t>(cuda_timeline_),
                                               value,
                                               last_signaled_,
                                               reinterpret_cast<std::uintptr_t>(stream)));
        }
        if (stream == nullptr) {
            return setFailure(last_error_, std::format(
                                               "CUDA/Vulkan timeline signal requires an explicit non-default CUDA stream (timeline={:#x}, requested_value={}, previous_signal_value={}, stream={:#x})",
                                               reinterpret_cast<std::uintptr_t>(cuda_timeline_),
                                               value,
                                               last_signaled_,
                                               reinterpret_cast<std::uintptr_t>(stream)));
        }

        if (value <= last_signaled_) {
            return setFailure(last_error_, std::format(
                                               "CUDA/Vulkan timeline signals must increase strictly (timeline={:#x}, requested_value={}, previous_signal_value={}, stream={:#x})",
                                               reinterpret_cast<std::uintptr_t>(cuda_timeline_),
                                               value,
                                               last_signaled_,
                                               reinterpret_cast<std::uintptr_t>(stream)));
        }

        cudaExternalSemaphoreSignalParams params{};
        params.params.fence.value = value;
        const cudaError_t status = cudaSignalExternalSemaphoresAsync(&cuda_timeline_, &params, 1, stream);
        if (!setCudaFailure(last_error_, "cudaSignalExternalSemaphoresAsync", status)) {
            return false;
        }
        last_signaled_ = value;
        return true;
    }

    // ===== CudaTimelineSemaphore =================================================

    CudaTimelineSemaphore::~CudaTimelineSemaphore() {
        reset();
    }

    CudaTimelineSemaphore::CudaTimelineSemaphore(CudaTimelineSemaphore&& other) noexcept {
        *this = std::move(other);
    }

    CudaTimelineSemaphore& CudaTimelineSemaphore::operator=(CudaTimelineSemaphore&& other) noexcept {
        if (this == &other) {
            return *this;
        }
        reset();
        cuda_timeline_ = std::exchange(other.cuda_timeline_, nullptr);
        last_signaled_ = std::exchange(other.last_signaled_, 0);
        last_waited_ = std::exchange(other.last_waited_, 0);
        last_error_ = std::move(other.last_error_);
        return *this;
    }

    bool CudaTimelineSemaphore::init(CudaVulkanExternalSemaphoreImport semaphore) {
        reset();
        last_error_.clear();

        if (auto err = verifyCudaMatchesVulkanDevice(); err) {
            return setFailure(last_error_, *err);
        }
        if (!nativeHandleValid(semaphore.semaphore_handle)) {
            return setFailure(last_error_, std::format(
                                               "CUDA timeline semaphore import requires a valid external handle (semaphore_handle={}, initial_value={})",
                                               nativeHandleString(semaphore.semaphore_handle),
                                               semaphore.initial_value));
        }

        NativeHandleOwner semaphore_handle(semaphore.semaphore_handle);
        // cudaExternalSemaphoreHandleDesc has no initialValue field, so CUDA cannot validate or
        // communicate a non-zero Vulkan initial value during import.
        if (semaphore.initial_value != 0) {
            return setFailure(last_error_, std::format(
                                               "CUDA timeline semaphore import requires Vulkan initialValue == 0 because CUDA cannot import it (initial_value={}, semaphore_handle={})",
                                               semaphore.initial_value,
                                               nativeHandleString(semaphore.semaphore_handle)));
        }

        cudaExternalSemaphoreHandleDesc semaphore_desc{};
        semaphore_desc.type = kCudaExternalSemaphoreHandleType;
#ifdef _WIN32
        semaphore_desc.handle.win32.handle = semaphore_handle.get();
#else
        semaphore_desc.handle.fd = semaphore_handle.get();
#endif

        cudaError_t status = cudaImportExternalSemaphore(&cuda_timeline_, &semaphore_desc);
        if (status != cudaSuccess) {
            return setCudaFailure(last_error_, "cudaImportExternalSemaphore", status);
        }
#ifndef _WIN32
        semaphore_handle.release();
#endif
        last_signaled_ = semaphore.initial_value;
        last_waited_ = 0;
        return true;
    }

    void CudaTimelineSemaphore::reset() {
        if (cuda_timeline_ != nullptr) {
            cudaDestroyExternalSemaphore(cuda_timeline_);
            cuda_timeline_ = nullptr;
        }
        last_signaled_ = 0;
        last_waited_ = 0;
    }

    bool CudaTimelineSemaphore::cudaSignal(const std::uint64_t value, const cudaStream_t stream) const {
        last_error_.clear();
        if (cuda_timeline_ == nullptr) {
            return setFailure(last_error_, std::format(
                                               "CUDA timeline signal requires an imported semaphore (timeline={:#x}, requested_value={}, previous_signal_value={}, stream={:#x})",
                                               reinterpret_cast<std::uintptr_t>(cuda_timeline_),
                                               value,
                                               last_signaled_,
                                               reinterpret_cast<std::uintptr_t>(stream)));
        }
        if (stream == nullptr) {
            return setFailure(last_error_, std::format(
                                               "CUDA timeline signal requires an explicit non-default stream (timeline={:#x}, requested_value={}, previous_signal_value={}, stream={:#x})",
                                               reinterpret_cast<std::uintptr_t>(cuda_timeline_),
                                               value,
                                               last_signaled_,
                                               reinterpret_cast<std::uintptr_t>(stream)));
        }
        if (value <= last_signaled_) {
            return setFailure(last_error_, std::format(
                                               "CUDA timeline signals must increase strictly (timeline={:#x}, requested_value={}, previous_signal_value={}, stream={:#x})",
                                               reinterpret_cast<std::uintptr_t>(cuda_timeline_),
                                               value,
                                               last_signaled_,
                                               reinterpret_cast<std::uintptr_t>(stream)));
        }

        cudaExternalSemaphoreSignalParams params{};
        params.params.fence.value = value;
        const cudaError_t status =
            cudaSignalExternalSemaphoresAsync(&cuda_timeline_, &params, 1, stream);
        if (!setCudaFailure(last_error_, "cudaSignalExternalSemaphoresAsync", status)) {
            return false;
        }
        last_signaled_ = value;
        return true;
    }

    bool CudaTimelineSemaphore::cudaWait(const std::uint64_t value, const cudaStream_t stream) const {
        last_error_.clear();
        if (cuda_timeline_ == nullptr) {
            return setFailure(last_error_, std::format(
                                               "CUDA timeline wait requires an imported semaphore (timeline={:#x}, requested_value={}, previous_wait_value={}, stream={:#x})",
                                               reinterpret_cast<std::uintptr_t>(cuda_timeline_),
                                               value,
                                               last_waited_,
                                               reinterpret_cast<std::uintptr_t>(stream)));
        }
        if (stream == nullptr) {
            return setFailure(last_error_, std::format(
                                               "CUDA timeline wait requires an explicit non-default stream (timeline={:#x}, requested_value={}, previous_wait_value={}, stream={:#x})",
                                               reinterpret_cast<std::uintptr_t>(cuda_timeline_),
                                               value,
                                               last_waited_,
                                               reinterpret_cast<std::uintptr_t>(stream)));
        }
        if (value == 0 || value <= last_waited_) {
            return setFailure(last_error_, std::format(
                                               "CUDA timeline waits must increase strictly and remain non-zero (timeline={:#x}, requested_value={}, previous_wait_value={}, stream={:#x})",
                                               reinterpret_cast<std::uintptr_t>(cuda_timeline_),
                                               value,
                                               last_waited_,
                                               reinterpret_cast<std::uintptr_t>(stream)));
        }
        cudaExternalSemaphoreWaitParams params{};
        params.params.fence.value = value;
        const cudaError_t status =
            cudaWaitExternalSemaphoresAsync(&cuda_timeline_, &params, 1, stream);
        if (!setCudaFailure(last_error_, "cudaWaitExternalSemaphoresAsync", status)) {
            return false;
        }
        last_waited_ = value;
        return true;
    }

    // ===== CudaVulkanBufferInterop ===============================================

    CudaVulkanBufferInterop::CudaVulkanBufferInterop(CudaVulkanExternalBufferImport buffer) {
        if (!init(std::move(buffer))) {
            throw std::runtime_error(last_error_);
        }
    }

    CudaVulkanBufferInterop::~CudaVulkanBufferInterop() {
        reset();
    }

    CudaVulkanBufferInterop::CudaVulkanBufferInterop(CudaVulkanBufferInterop&& other) noexcept {
        *this = std::move(other);
    }

    CudaVulkanBufferInterop& CudaVulkanBufferInterop::operator=(CudaVulkanBufferInterop&& other) noexcept {
        if (this == &other) {
            return *this;
        }
        reset();
        cuda_mem_ = std::exchange(other.cuda_mem_, nullptr);
        device_ptr_ = std::exchange(other.device_ptr_, nullptr);
        allocation_size_ = std::exchange(other.allocation_size_, 0);
        size_ = std::exchange(other.size_, 0);
        upload_source_ = std::move(other.upload_source_);
        last_error_ = std::move(other.last_error_);
        return *this;
    }

    bool CudaVulkanBufferInterop::init(CudaVulkanExternalBufferImport buffer) {
        reset();
        last_error_.clear();

        if (auto err = verifyCudaMatchesVulkanDevice(); err) {
            return setFailure(last_error_, *err);
        }
        if (!nativeHandleValid(buffer.memory_handle)) {
            return setFailure(last_error_, std::format(
                                               "CUDA/Vulkan external buffer import requires a valid memory handle (memory_handle={}, allocation_size={}, cuda_visible_size={}, dedicated={})",
                                               nativeHandleString(buffer.memory_handle),
                                               buffer.allocation_size,
                                               buffer.size,
                                               buffer.dedicated_allocation));
        }
        if (buffer.allocation_size == 0 || buffer.size == 0 || buffer.size > buffer.allocation_size) {
            return setFailure(last_error_, std::format(
                                               "CUDA/Vulkan external buffer import requires a non-zero CUDA-visible size within the Vulkan allocation (vulkan_allocation_size={}, cuda_visible_size={}, memory_handle={}, dedicated={})",
                                               buffer.allocation_size,
                                               buffer.size,
                                               nativeHandleString(buffer.memory_handle),
                                               buffer.dedicated_allocation));
        }

        NativeHandleOwner memory_handle(buffer.memory_handle);

        cudaExternalMemoryHandleDesc memory_desc{};
        memory_desc.type = kCudaExternalMemoryHandleType;
        memory_desc.size = buffer.allocation_size;
        if (buffer.dedicated_allocation) {
            memory_desc.flags = cudaExternalMemoryDedicated;
        }
#ifdef _WIN32
        memory_desc.handle.win32.handle = memory_handle.get();
#else
        memory_desc.handle.fd = memory_handle.get();
#endif

        cudaError_t status = cudaImportExternalMemory(&cuda_mem_, &memory_desc);
        if (status != cudaSuccess) {
            reset();
            return setCudaFailure(last_error_, "cudaImportExternalMemory(buffer)", status);
        }
#ifndef _WIN32
        memory_handle.release();
#endif

        cudaExternalMemoryBufferDesc buffer_desc{};
        buffer_desc.offset = 0;
        buffer_desc.size = buffer.size;
        status = cudaExternalMemoryGetMappedBuffer(&device_ptr_, cuda_mem_, &buffer_desc);
        if (status != cudaSuccess) {
            reset();
            return setCudaFailure(last_error_, "cudaExternalMemoryGetMappedBuffer", status);
        }

        allocation_size_ = buffer.allocation_size;
        size_ = buffer.size;
        return true;
    }

    void CudaVulkanBufferInterop::reset() {
        upload_source_ = {};
        if (device_ptr_ != nullptr) {
            cudaFree(device_ptr_);
            device_ptr_ = nullptr;
        }
        if (cuda_mem_ != nullptr) {
            cudaDestroyExternalMemory(cuda_mem_);
            cuda_mem_ = nullptr;
        }
        allocation_size_ = 0;
        size_ = 0;
    }

    bool CudaVulkanBufferInterop::valid() const {
        return cuda_mem_ != nullptr && device_ptr_ != nullptr && size_ > 0;
    }

    bool CudaVulkanBufferInterop::copyFromTensor(const lfs::core::Tensor& tensor,
                                                 const std::size_t byte_count,
                                                 const cudaStream_t stream) const {
        return copyFromTensor(tensor, byte_count, 0, stream);
    }

    bool CudaVulkanBufferInterop::copyFromTensor(const lfs::core::Tensor& tensor,
                                                 const std::size_t byte_count,
                                                 const std::size_t dst_offset,
                                                 const cudaStream_t stream) const {
        last_error_.clear();
        if (!valid()) {
            return setFailure(last_error_, std::format(
                                               "CUDA/Vulkan buffer copy requires a complete imported buffer (cuda_memory={:#x}, device_pointer={:#x}, cuda_visible_size={}, vulkan_allocation_size={}, requested_bytes={}, dst_offset={})",
                                               reinterpret_cast<std::uintptr_t>(cuda_mem_),
                                               reinterpret_cast<std::uintptr_t>(device_ptr_),
                                               size_,
                                               allocation_size_,
                                               byte_count,
                                               dst_offset));
        }
        if (stream == nullptr) {
            return setFailure(last_error_, std::format(
                                               "CUDA/Vulkan buffer copy requires an explicit non-default stream (stream={:#x}, requested_bytes={}, dst_offset={}, cuda_visible_size={}, vulkan_allocation_size={})",
                                               reinterpret_cast<std::uintptr_t>(stream),
                                               byte_count,
                                               dst_offset,
                                               size_,
                                               allocation_size_));
        }
        if (byte_count == 0 || dst_offset > size_ || byte_count > size_ - dst_offset) {
            return setFailure(last_error_, std::format(
                                               "CUDA/Vulkan buffer copy range exceeds the CUDA-visible import (dst_offset={}, requested_bytes={}, range_end={}, cuda_visible_size={}, vulkan_allocation_size={})",
                                               dst_offset,
                                               byte_count,
                                               dst_offset <= std::numeric_limits<std::size_t>::max() - byte_count
                                                   ? dst_offset + byte_count
                                                   : std::numeric_limits<std::size_t>::max(),
                                               size_,
                                               allocation_size_));
        }
        if (!tensor.is_valid() || tensor.data_ptr() == nullptr) {
            return setFailure(last_error_, std::format(
                                               "CUDA/Vulkan buffer copy requires valid tensor storage (tensor_valid={}, tensor_pointer={:#x}, tensor_bytes={}, requested_bytes={}, dst_offset={})",
                                               tensor.is_valid(),
                                               reinterpret_cast<std::uintptr_t>(tensor.data_ptr()),
                                               tensor.is_valid() ? tensor.bytes() : 0,
                                               byte_count,
                                               dst_offset));
        }

        upload_source_ = tensor;
        if (upload_source_.device() != lfs::core::Device::CUDA) {
            upload_source_ = upload_source_.to(lfs::core::Device::CUDA, stream);
        }
        if (!upload_source_.is_contiguous()) {
            upload_source_ = upload_source_.contiguous();
        }
        if (byte_count > upload_source_.bytes()) {
            return setFailure(
                last_error_,
                std::format("CUDA/Vulkan buffer copy requested {} bytes from {} byte tensor",
                            byte_count,
                            upload_source_.bytes()));
        }

        upload_source_.sync_to_stream(stream);
        auto* const dst = static_cast<std::uint8_t*>(device_ptr_) + dst_offset;
        const cudaError_t status = cudaMemcpyAsync(
            dst, upload_source_.data_ptr(), byte_count, cudaMemcpyDeviceToDevice, stream);
        return setCudaFailure(
            last_error_, "cudaMemcpyAsync(CUDA tensor -> Vulkan buffer)", status);
    }

} // namespace lfs::rendering

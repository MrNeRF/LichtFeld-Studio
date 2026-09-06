/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "vk_cuda_bridge.hpp"

#include "core/logger.hpp"
#include "vk_context.hpp"

#include <cstdint>
#include <format>
#include <string>
#include <type_traits>
#include <utility>

#ifdef _WIN32
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>
#else
#include <unistd.h>
#endif

namespace lfs::core::internal {
    namespace {
        template <class Handle>
        uint64_t memory_key(const Handle memory) {
            if constexpr (std::is_pointer_v<Handle>) {
                return reinterpret_cast<uintptr_t>(memory);
            } else {
                return static_cast<uint64_t>(memory);
            }
        }

        bool handle_valid(const VulkanExportHandle handle) {
#ifdef _WIN32
            return handle != nullptr;
#else
            return handle >= 0;
#endif
        }

        void close_export_handle(VulkanExportHandle& handle) noexcept {
            if (!handle_valid(handle)) {
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

        lfs::Error cuda_error(const char* const operation, const cudaError_t status) {
            return lfs::make_error(lfs::ErrorInit{
                .code = lfs::ErrorCode::Internal,
                .domain = lfs::ErrorDomain::CUDA,
                .user_message = std::format("{} failed: {}", operation, cudaGetErrorString(status)),
                .detail = std::format("{} returned {}", operation, cudaGetErrorName(status)),
                .detection = LFS_SOURCE_SITE_CURRENT(),
                .native = lfs::NativeError{
                    .domain = lfs::ErrorDomain::CUDA,
                    .code = static_cast<int64_t>(status),
                    .name = cudaGetErrorName(status),
                },
            });
        }

        lfs::Error vulkan_error(const char* const message) {
            return lfs::make_error(lfs::ErrorInit{
                .code = lfs::ErrorCode::Internal,
                .domain = lfs::ErrorDomain::Vulkan,
                .user_message = message,
                .detection = LFS_SOURCE_SITE_CURRENT(),
            });
        }
    } // namespace

    void VulkanCudaMemoryImport::release() noexcept {
        std::lock_guard lock(mutex);
        if (mapped != nullptr) {
            (void)cudaFree(mapped);
            mapped = nullptr;
        }
        if (memory != nullptr) {
            (void)cudaDestroyExternalMemory(memory);
            memory = nullptr;
        }
#ifdef _WIN32
        close_export_handle(keep_handle);
#endif
        block_size = 0;
        dedicated = false;
    }

    VulkanCudaImportRegistry::VulkanCudaImportRegistry(VulkanContext& context)
        : context_(context) {
        const VkDevice device = context.device();
#ifdef _WIN32
        get_memory_ = reinterpret_cast<PFN_vkGetMemoryWin32HandleKHR>(
            vkGetDeviceProcAddr(device, "vkGetMemoryWin32HandleKHR"));
        get_semaphore_ = reinterpret_cast<PFN_vkGetSemaphoreWin32HandleKHR>(
            vkGetDeviceProcAddr(device, "vkGetSemaphoreWin32HandleKHR"));
#else
        get_memory_ = reinterpret_cast<PFN_vkGetMemoryFdKHR>(
            vkGetDeviceProcAddr(device, "vkGetMemoryFdKHR"));
        get_semaphore_ = reinterpret_cast<PFN_vkGetSemaphoreFdKHR>(
            vkGetDeviceProcAddr(device, "vkGetSemaphoreFdKHR"));
#endif
    }

    lfs::Result<VulkanExportHandle>
    VulkanCudaImportRegistry::export_memory_handle(const VkDeviceMemory memory) const {
        if (get_memory_ == nullptr) {
            return vulkan_error("vkGetMemoryFdKHR is unavailable");
        }
#ifdef _WIN32
        VkMemoryGetWin32HandleInfoKHR info{VK_STRUCTURE_TYPE_MEMORY_GET_WIN32_HANDLE_INFO_KHR};
        info.memory = memory;
        info.handleType = kVulkanExportMemoryHandleType;
        HANDLE handle = nullptr;
        const VkResult result = get_memory_(context_.device(), &info, &handle);
        if (result != VK_SUCCESS || handle == nullptr) {
            return lfs::make_error(lfs::ErrorInit{
                .code = lfs::ErrorCode::Internal,
                .domain = lfs::ErrorDomain::Vulkan,
                .user_message = "exporting Vulkan device memory for CUDA failed",
                .detail = std::format("vkGetMemoryWin32HandleKHR returned {}",
                                      static_cast<int>(result)),
                .detection = LFS_SOURCE_SITE_CURRENT(),
            });
        }
        return handle;
#else
        VkMemoryGetFdInfoKHR info{VK_STRUCTURE_TYPE_MEMORY_GET_FD_INFO_KHR};
        info.memory = memory;
        info.handleType = kVulkanExportMemoryHandleType;
        int fd = -1;
        const VkResult result = get_memory_(context_.device(), &info, &fd);
        if (result != VK_SUCCESS || fd < 0) {
            return lfs::make_error(lfs::ErrorInit{
                .code = lfs::ErrorCode::Internal,
                .domain = lfs::ErrorDomain::Vulkan,
                .user_message = "exporting Vulkan device memory for CUDA failed",
                .detail = std::format("vkGetMemoryFdKHR returned {}", static_cast<int>(result)),
                .detection = LFS_SOURCE_SITE_CURRENT(),
            });
        }
        return fd;
#endif
    }

    lfs::Result<VulkanExportHandle>
    VulkanCudaImportRegistry::export_semaphore_handle(const VkSemaphore semaphore) const {
        if (get_semaphore_ == nullptr) {
            return vulkan_error("vkGetSemaphoreFdKHR is unavailable");
        }
#ifdef _WIN32
        VkSemaphoreGetWin32HandleInfoKHR info{
            VK_STRUCTURE_TYPE_SEMAPHORE_GET_WIN32_HANDLE_INFO_KHR};
        info.semaphore = semaphore;
        info.handleType = kVulkanExportSemaphoreHandleType;
        HANDLE handle = nullptr;
        const VkResult result = get_semaphore_(context_.device(), &info, &handle);
        if (result != VK_SUCCESS || handle == nullptr) {
            return lfs::make_error(lfs::ErrorInit{
                .code = lfs::ErrorCode::Internal,
                .domain = lfs::ErrorDomain::Vulkan,
                .user_message = "exporting the Vulkan timeline for CUDA failed",
                .detail = std::format("vkGetSemaphoreWin32HandleKHR returned {}",
                                      static_cast<int>(result)),
                .detection = LFS_SOURCE_SITE_CURRENT(),
            });
        }
        return handle;
#else
        VkSemaphoreGetFdInfoKHR info{VK_STRUCTURE_TYPE_SEMAPHORE_GET_FD_INFO_KHR};
        info.semaphore = semaphore;
        info.handleType = kVulkanExportSemaphoreHandleType;
        int fd = -1;
        const VkResult result = get_semaphore_(context_.device(), &info, &fd);
        if (result != VK_SUCCESS || fd < 0) {
            return lfs::make_error(lfs::ErrorInit{
                .code = lfs::ErrorCode::Internal,
                .domain = lfs::ErrorDomain::Vulkan,
                .user_message = "exporting the Vulkan timeline for CUDA failed",
                .detail = std::format("vkGetSemaphoreFdKHR returned {}", static_cast<int>(result)),
                .detection = LFS_SOURCE_SITE_CURRENT(),
            });
        }
        return fd;
#endif
    }

    VulkanCudaImportRegistry::~VulkanCudaImportRegistry() {
        shutdown();
    }

    void VulkanCudaImportRegistry::import_timeline(const VkSemaphore semaphore) {
        if (semaphore == VK_NULL_HANDLE || timeline_ != nullptr) {
            return;
        }
        const auto exported = export_semaphore_handle(semaphore);
        if (!exported) {
            LOG_WARN("Vulkan timeline was not imported into CUDA: {}",
                     lfs::format_for_developer(exported.error()));
            return;
        }
        VulkanExportHandle handle = *exported;
#ifdef _WIN32
        constexpr cudaExternalSemaphoreHandleType kType =
            cudaExternalSemaphoreHandleTypeTimelineSemaphoreWin32;
#else
        constexpr cudaExternalSemaphoreHandleType kType =
            cudaExternalSemaphoreHandleTypeTimelineSemaphoreFd;
#endif
        cudaExternalSemaphoreHandleDesc desc{};
        desc.type = kType;
#ifdef _WIN32
        desc.handle.win32.handle = handle;
#else
        desc.handle.fd = handle;
#endif
        cudaExternalSemaphore_t imported = nullptr;
        const cudaError_t status = cudaImportExternalSemaphore(&imported, &desc);
#ifndef _WIN32
        close_export_handle(handle);
#endif
        if (status != cudaSuccess) {
            (void)cudaGetLastError();
#ifdef _WIN32
            close_export_handle(handle);
#endif
            LOG_WARN("cudaImportExternalSemaphore(timeline) failed: {} ({})",
                     cudaGetErrorName(status), cudaGetErrorString(status));
            return;
        }
        timeline_ = imported;
#ifdef _WIN32
        timeline_handle_ = handle;
#endif
    }

    lfs::Status VulkanCudaImportRegistry::wait_timeline(const uint64_t value,
                                                        const cudaStream_t stream) {
        if (value == 0) {
            return {};
        }
        if (timeline_ == nullptr) {
            return lfs::Status::failure(vulkan_error(
                "CUDA view of a Vulkan tensor needs an imported timeline semaphore"));
        }
        cudaExternalSemaphoreWaitParams params{};
        params.params.fence.value = value;
        const cudaError_t status =
            cudaWaitExternalSemaphoresAsync(&timeline_, &params, 1, stream);
        if (status != cudaSuccess) {
            (void)cudaGetLastError();
            return lfs::Status::failure(cuda_error("cudaWaitExternalSemaphoresAsync", status));
        }
        return {};
    }

    lfs::Result<std::shared_ptr<VulkanCudaMemoryImport>>
    VulkanCudaImportRegistry::import_memory(const VkDeviceMemory memory,
                                            const VkDeviceSize block_size,
                                            const bool dedicated) {
        if (memory == VK_NULL_HANDLE || block_size == 0) {
            return vulkan_error("CUDA import of Vulkan memory needs a device-memory block");
        }
        const uint64_t key = memory_key(memory);
        {
            std::lock_guard lock(mutex_);
            if (shutting_down_) {
                return vulkan_error(
                    "CUDA import of Vulkan memory is unavailable because the backend is shutting down");
            }
            if (const auto iterator = imports_.find(key); iterator != imports_.end()) {
                return iterator->second;
            }
        }

        const auto exported = export_memory_handle(memory);
        if (!exported) {
            return exported.error();
        }
        VulkanExportHandle handle = *exported;
#ifdef _WIN32
        constexpr cudaExternalMemoryHandleType kType = cudaExternalMemoryHandleTypeOpaqueWin32;
#else
        constexpr cudaExternalMemoryHandleType kType = cudaExternalMemoryHandleTypeOpaqueFd;
#endif
        cudaExternalMemoryHandleDesc desc{};
        desc.type = kType;
        desc.size = block_size;
        if (dedicated) {
            desc.flags = cudaExternalMemoryDedicated;
        }
#ifdef _WIN32
        desc.handle.win32.handle = handle;
#else
        desc.handle.fd = handle;
#endif
        auto created = std::make_shared<VulkanCudaMemoryImport>();
        created->block_size = block_size;
        created->dedicated = dedicated;
        const cudaError_t import_status = cudaImportExternalMemory(&created->memory, &desc);
#ifndef _WIN32
        // CUDA docs: OPAQUE_FD is duplicated; the Vulkan-exported fd must be closed.
        close_export_handle(handle);
#else
        created->keep_handle = handle;
        handle = kInvalidVulkanExportHandle;
#endif
        if (import_status != cudaSuccess) {
            (void)cudaGetLastError();
#ifdef _WIN32
            close_export_handle(created->keep_handle);
#endif
            created->memory = nullptr;
            return cuda_error("cudaImportExternalMemory", import_status);
        }

        cudaExternalMemoryBufferDesc buffer_desc{};
        buffer_desc.offset = 0;
        buffer_desc.size = block_size;
        const cudaError_t map_status =
            cudaExternalMemoryGetMappedBuffer(&created->mapped, created->memory, &buffer_desc);
        if (map_status != cudaSuccess) {
            (void)cudaGetLastError();
            created->release();
            return cuda_error("cudaExternalMemoryGetMappedBuffer", map_status);
        }

        std::lock_guard lock(mutex_);
        if (shutting_down_) {
            created->release();
            return vulkan_error(
                "CUDA import of Vulkan memory is unavailable because the backend is shutting down");
        }
        if (const auto iterator = imports_.find(key); iterator != imports_.end()) {
            created->release();
            return iterator->second;
        }
        imports_.emplace(key, created);
        return created;
    }

    void VulkanCudaImportRegistry::on_device_memory_freed(const VkDeviceMemory memory) noexcept {
        std::shared_ptr<VulkanCudaMemoryImport> imported;
        {
            std::lock_guard lock(mutex_);
            const auto iterator = imports_.find(memory_key(memory));
            if (iterator == imports_.end()) {
                return;
            }
            imported = std::move(iterator->second);
            imports_.erase(iterator);
        }
        if (imported) {
            imported->release();
        }
    }

    void VulkanCudaImportRegistry::shutdown() noexcept {
        std::unordered_map<uint64_t, std::shared_ptr<VulkanCudaMemoryImport>> leftover;
        cudaExternalSemaphore_t timeline = nullptr;
#ifdef _WIN32
        VulkanExportHandle timeline_handle = kInvalidVulkanExportHandle;
#endif
        {
            std::lock_guard lock(mutex_);
            shutting_down_ = true;
            leftover.swap(imports_);
            timeline = timeline_;
            timeline_ = nullptr;
#ifdef _WIN32
            timeline_handle = timeline_handle_;
            timeline_handle_ = kInvalidVulkanExportHandle;
#endif
        }
        for (auto& [key, imported] : leftover) {
            (void)key;
            if (imported) {
                imported->release();
            }
        }
        leftover.clear();
        if (timeline != nullptr) {
            (void)cudaDestroyExternalSemaphore(timeline);
        }
#ifdef _WIN32
        close_export_handle(timeline_handle);
#endif
    }

    bool VulkanCudaImportRegistry::timeline_imported() const noexcept {
        std::lock_guard lock(mutex_);
        return timeline_ != nullptr;
    }

    uint64_t VulkanCudaImportRegistry::memory_import_count() const noexcept {
        std::lock_guard lock(mutex_);
        return imports_.size();
    }

    void VKAPI_PTR VulkanCudaImportRegistry::vma_free_callback(VmaAllocator,
                                                               uint32_t,
                                                               const VkDeviceMemory memory,
                                                               VkDeviceSize,
                                                               void* const user) {
        if (user == nullptr || memory == VK_NULL_HANDLE) {
            return;
        }
        static_cast<VulkanCudaImportRegistry*>(user)->on_device_memory_freed(memory);
    }

} // namespace lfs::core::internal

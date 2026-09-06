/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#pragma once

#include "core/error.hpp"

#include <cstdint>
#include <cuda_runtime.h>
#include <memory>
#include <mutex>
#include <unordered_map>
#include <vulkan/vulkan.h>

struct VmaAllocator_T;
using VmaAllocator = VmaAllocator_T*;

namespace lfs::core::internal {

    class VulkanContext;

#ifdef _WIN32
    using VulkanExportHandle = void*;
    inline constexpr VulkanExportHandle kInvalidVulkanExportHandle = nullptr;
#else
    using VulkanExportHandle = int;
    inline constexpr VulkanExportHandle kInvalidVulkanExportHandle = -1;
#endif

#ifdef _WIN32
    inline constexpr VkExternalMemoryHandleTypeFlagBits kVulkanExportMemoryHandleType =
        VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_WIN32_BIT;
    inline constexpr VkExternalSemaphoreHandleTypeFlagBits kVulkanExportSemaphoreHandleType =
        VK_EXTERNAL_SEMAPHORE_HANDLE_TYPE_OPAQUE_WIN32_BIT;
#else
    inline constexpr VkExternalMemoryHandleTypeFlagBits kVulkanExportMemoryHandleType =
        VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD_BIT;
    inline constexpr VkExternalSemaphoreHandleTypeFlagBits kVulkanExportSemaphoreHandleType =
        VK_EXTERNAL_SEMAPHORE_HANDLE_TYPE_OPAQUE_FD_BIT;
#endif

    // One CUDA import of a VkDeviceMemory block. Views hold a shared_ptr so the
    // mapping outlives the source Tensor; release is idempotent for shutdown
    // while a view is still alive.
    struct VulkanCudaMemoryImport {
        std::mutex mutex;
        cudaExternalMemory_t memory = nullptr;
        void* mapped = nullptr;
        VkDeviceSize block_size = 0;
        bool dedicated = false;
#ifdef _WIN32
        VulkanExportHandle keep_handle = kInvalidVulkanExportHandle;
#endif

        void release() noexcept;
        ~VulkanCudaMemoryImport() { release(); }
        VulkanCudaMemoryImport() = default;
        VulkanCudaMemoryImport(const VulkanCudaMemoryImport&) = delete;
        VulkanCudaMemoryImport& operator=(const VulkanCudaMemoryImport&) = delete;
    };

    class VulkanCudaImportRegistry final {
    public:
        explicit VulkanCudaImportRegistry(VulkanContext& context);
        ~VulkanCudaImportRegistry();

        VulkanCudaImportRegistry(const VulkanCudaImportRegistry&) = delete;
        VulkanCudaImportRegistry& operator=(const VulkanCudaImportRegistry&) = delete;

        void import_timeline(VkSemaphore semaphore);
        [[nodiscard]] lfs::Status wait_timeline(uint64_t value, cudaStream_t stream);
        [[nodiscard]] lfs::Result<std::shared_ptr<VulkanCudaMemoryImport>>
        import_memory(VkDeviceMemory memory, VkDeviceSize block_size, bool dedicated);

        [[nodiscard]] lfs::Result<VulkanExportHandle> export_memory_handle(VkDeviceMemory memory) const;
        [[nodiscard]] lfs::Result<VulkanExportHandle> export_semaphore_handle(VkSemaphore semaphore) const;

        void on_device_memory_freed(VkDeviceMemory memory) noexcept;
        void shutdown() noexcept;

        [[nodiscard]] bool timeline_imported() const noexcept;
        [[nodiscard]] uint64_t memory_import_count() const noexcept;

        static void VKAPI_PTR vma_free_callback(VmaAllocator allocator,
                                                uint32_t memory_type,
                                                VkDeviceMemory memory,
                                                VkDeviceSize size,
                                                void* user);

    private:
        VulkanContext& context_;
        mutable std::mutex mutex_;
        std::unordered_map<uint64_t, std::shared_ptr<VulkanCudaMemoryImport>> imports_;
        cudaExternalSemaphore_t timeline_ = nullptr;
#ifdef _WIN32
        PFN_vkGetMemoryWin32HandleKHR get_memory_ = nullptr;
        PFN_vkGetSemaphoreWin32HandleKHR get_semaphore_ = nullptr;
        VulkanExportHandle timeline_handle_ = kInvalidVulkanExportHandle;
#else
        PFN_vkGetMemoryFdKHR get_memory_ = nullptr;
        PFN_vkGetSemaphoreFdKHR get_semaphore_ = nullptr;
#endif
        bool shutting_down_ = false;
    };

} // namespace lfs::core::internal

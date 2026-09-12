/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#pragma once

#include "../descriptors.hpp"

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <mutex>
#include <optional>
#include <span>
#include <unordered_map>
#include <vector>
#include <vk_mem_alloc.h>
#include <vulkan/vulkan.h>

namespace lfs::core {
    class MemoryInfo;
}

namespace lfs::core::internal {

    class VulkanContext;

    class VulkanMemory final {
    public:
        explicit VulkanMemory(VulkanContext& context);
        ~VulkanMemory();

        VulkanMemory(const VulkanMemory&) = delete;
        VulkanMemory& operator=(const VulkanMemory&) = delete;

        [[nodiscard]] StorageRef allocate(size_t bytes, size_t alignment,
                                          ExecContext context);
        // Host-visible, persistently mapped storage for results the host reads
        // right after they are produced (scalar reductions, counters): shaders
        // write it through its device address and read_readback waits for the
        // producer and copies from the mapping, so the staging ring and its
        // mutex stay out of the path. The block comes back zeroed.
        [[nodiscard]] StorageRef allocate_readback(size_t bytes);
        void read_readback(StorageRef storage, void* destination, size_t bytes);
        void deallocate(StorageRef storage) noexcept;
        void copy_host_to_device(const CopyRequest& request);
        void copy_device_to_host(const CopyRequest& request);
        void copy_device_to_device(const CopyRequest& request);
        void memset(const FillRequest& request);
        void mark_used(std::span<const StorageRef> reads,
                       std::span<const StorageRef> writes,
                       uint64_t timeline_value);

        void trim();
        [[nodiscard]] MemoryInfo stats() const;
        [[nodiscard]] size_t cached_bytes() const noexcept;
        [[nodiscard]] uint64_t live_object_count() const noexcept;
        [[nodiscard]] bool owns_address(const void* pointer) const noexcept;
        [[nodiscard]] bool exports_memory() const noexcept { return exports_memory_; }
        struct CudaBlockInfo {
            VkDeviceMemory memory = VK_NULL_HANDLE;
            VkDeviceSize allocation_offset = 0;
            VkDeviceSize block_size = 0;
            bool dedicated = false;
            bool host_visible = false;
            bool exportable = false;
        };
        [[nodiscard]] std::optional<CudaBlockInfo> cuda_block_info(StorageRef storage) const;
        void shutdown();
        [[nodiscard]] static uint64_t last_shutdown_live_allocations() noexcept;
        static void reset_last_shutdown_live_allocations() noexcept;

    private:
        struct AllocationRecord;
        struct StagingSlice;

        void create_pool();
        [[nodiscard]] StorageRef allocate_storage(size_t bytes, size_t alignment,
                                                  bool host_visible,
                                                  const ExecContext& context);
        void ensure_staging(size_t bytes);
        [[nodiscard]] StagingSlice acquire_staging(size_t bytes, size_t alignment);
        void collect_retired_locked(uint64_t completed);
        void destroy_free_locked();
        [[nodiscard]] AllocationRecord& allocation_for(StorageRef storage) const;
        [[nodiscard]] static VkBuffer buffer_for(StorageRef storage);
        [[nodiscard]] static VkDeviceSize offset_for(StorageRef storage);

        VulkanContext& context_;
        VkExportMemoryAllocateInfo export_alloc_info_{};
        VmaPool device_pool_ = VK_NULL_HANDLE;
        bool exports_memory_ = false;
        mutable std::mutex allocations_mutex_;
        std::unordered_map<uint64_t, std::unique_ptr<AllocationRecord>> allocations_;
        std::vector<std::unique_ptr<AllocationRecord>> retired_;
        std::unordered_map<VkDeviceSize,
                           std::vector<std::unique_ptr<AllocationRecord>>>
            free_lists_;
        std::unordered_map<VkDeviceSize,
                           std::vector<std::unique_ptr<AllocationRecord>>>
            readback_free_lists_;
        mutable std::mutex staging_mutex_;
        VkBuffer staging_buffer_ = VK_NULL_HANDLE;
        VmaAllocation staging_allocation_ = VK_NULL_HANDLE;
        std::byte* staging_mapped_ = nullptr;
        VkDeviceSize staging_size_ = 0;
        VkDeviceSize staging_head_ = 0;
        uint64_t staging_retire_value_ = 0;
        bool shutting_down_ = false;
        static std::atomic<uint64_t> g_last_shutdown_live_allocations;
    };

} // namespace lfs::core::internal

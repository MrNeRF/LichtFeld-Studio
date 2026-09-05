/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "vk_memory.hpp"

#include "../../internal/tensor_impl.hpp"
#include "core/assert.hpp"
#include "vk_context.hpp"
#include "vk_recorder.hpp"

#include <algorithm>
#include <array>
#include <bit>
#include <cstring>
#include <limits>
#include <type_traits>

namespace lfs::core::internal {
    namespace {
        constexpr VkDeviceSize kMib = 1024ull * 1024ull;
        constexpr VkDeviceSize kPoolBlockSize = 64ull * kMib;
        constexpr VkDeviceSize kInitialStagingSize = 64ull * kMib;
        constexpr VkDeviceSize kSlabLimit = 1ull * kMib;
        constexpr VkDeviceSize kDirectLimit = 16ull * kMib;
        constexpr VkBufferUsageFlags kStorageUsage =
            VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT |
            VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
            VK_BUFFER_USAGE_TRANSFER_SRC_BIT |
            VK_BUFFER_USAGE_TRANSFER_DST_BIT;

        template <class Handle>
        uint64_t handle_value(const Handle handle) {
            if constexpr (std::is_pointer_v<Handle>) {
                return reinterpret_cast<uintptr_t>(handle);
            } else {
                return static_cast<uint64_t>(handle);
            }
        }

        template <class Handle>
        Handle handle_from_value(const uint64_t value) {
            if constexpr (std::is_pointer_v<Handle>) {
                return reinterpret_cast<Handle>(static_cast<uintptr_t>(value));
            } else {
                return static_cast<Handle>(value);
            }
        }

        VkDeviceSize align_up(const VkDeviceSize value, const VkDeviceSize alignment) {
            return (value + alignment - 1) & ~(alignment - 1);
        }

        VkDeviceSize allocation_size(const VkDeviceSize bytes) {
            if (bytes < kSlabLimit) {
                return align_up(bytes, 256);
            }
            if (bytes < kDirectLimit) {
                return std::bit_ceil(bytes);
            }
            return bytes;
        }
    } // namespace

    struct VulkanMemory::AllocationRecord {
        VkBuffer buffer = VK_NULL_HANDLE;
        VmaAllocation allocation = VK_NULL_HANDLE;
        VkDeviceSize requested_size = 0;
        VkDeviceSize allocated_size = 0;
        VkDeviceAddress address = 0;
        uint64_t last_use = 0;
        bool direct = false;
        bool host_visible = false;
        std::byte* mapped = nullptr;
        StorageMeta descriptor_owner;
    };

    struct VulkanMemory::StagingSlice {
        VkBuffer buffer = VK_NULL_HANDLE;
        VkDeviceSize offset = 0;
        VkDeviceSize size = 0;
        std::byte* mapped = nullptr;
    };

    VulkanMemory::VulkanMemory(VulkanContext& context)
        : context_(context) {
        try {
            create_pool();
            ensure_staging(kInitialStagingSize);
        } catch (...) {
            shutdown();
            throw;
        }
    }

    std::atomic<uint64_t> VulkanMemory::g_last_shutdown_live_allocations{0};

    uint64_t VulkanMemory::last_shutdown_live_allocations() noexcept {
        return g_last_shutdown_live_allocations.load(std::memory_order_acquire);
    }

    void VulkanMemory::reset_last_shutdown_live_allocations() noexcept {
        g_last_shutdown_live_allocations.store(0, std::memory_order_release);
    }

    VulkanMemory::~VulkanMemory() {
        shutdown();
    }

    void VulkanMemory::create_pool() {
        VkBufferCreateInfo buffer_info{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
        buffer_info.size = 256;
        buffer_info.usage = kStorageUsage;
        buffer_info.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
        VmaAllocationCreateInfo allocation_info{};
        allocation_info.usage = VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE;
        allocation_info.requiredFlags = VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT;
        uint32_t memory_type = 0;
        vk_check(&context_, vmaFindMemoryTypeIndexForBufferInfo(context_.allocator(), &buffer_info, &allocation_info, &memory_type),
                 "vmaFindMemoryTypeIndexForBufferInfo");
        VmaPoolCreateInfo pool_info{};
        pool_info.memoryTypeIndex = memory_type;
        pool_info.blockSize = kPoolBlockSize;
        vk_check(&context_, vmaCreatePool(context_.allocator(), &pool_info, &device_pool_),
                 "vmaCreatePool");
    }

    void VulkanMemory::ensure_staging(const size_t bytes) {
        if (staging_size_ >= bytes) {
            return;
        }
        if (staging_buffer_ != VK_NULL_HANDLE) {
            context_.recorders().wait_all();
            vmaDestroyBuffer(context_.allocator(), staging_buffer_, staging_allocation_);
            staging_buffer_ = VK_NULL_HANDLE;
            staging_allocation_ = VK_NULL_HANDLE;
            staging_mapped_ = nullptr;
        }
        staging_size_ = std::max<VkDeviceSize>(kInitialStagingSize,
                                               std::bit_ceil(bytes));
        VkBufferCreateInfo buffer_info{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
        buffer_info.size = staging_size_;
        buffer_info.usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT |
                            VK_BUFFER_USAGE_TRANSFER_DST_BIT;
        buffer_info.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
        VmaAllocationCreateInfo allocation_info{};
        allocation_info.flags = VMA_ALLOCATION_CREATE_HOST_ACCESS_RANDOM_BIT |
                                VMA_ALLOCATION_CREATE_MAPPED_BIT;
        allocation_info.usage = VMA_MEMORY_USAGE_AUTO;
        allocation_info.requiredFlags = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                                        VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
        VmaAllocationInfo info{};
        vk_check(&context_, vmaCreateBuffer(context_.allocator(), &buffer_info, &allocation_info, &staging_buffer_, &staging_allocation_, &info),
                 "vmaCreateBuffer(staging)");
        staging_mapped_ = static_cast<std::byte*>(info.pMappedData);
        LFS_ASSERT_MSG(staging_mapped_ != nullptr,
                       "Vulkan staging allocation was not mapped");
        staging_head_ = 0;
        staging_retire_value_ = 0;
    }

    VulkanMemory::StagingSlice VulkanMemory::acquire_staging(
        const size_t bytes, const size_t alignment) {
        ensure_staging(bytes + alignment);
        VkDeviceSize offset = align_up(staging_head_, std::max<size_t>(alignment, 4));
        if (offset + bytes > staging_size_) {
            context_.recorders().wait_all();
            if (staging_retire_value_ != 0) {
                context_.wait(staging_retire_value_);
            }
            staging_head_ = 0;
            staging_retire_value_ = 0;
            offset = 0;
        }
        staging_head_ = offset + bytes;
        return StagingSlice{
            .buffer = staging_buffer_,
            .offset = offset,
            .size = bytes,
            .mapped = staging_mapped_ + offset,
        };
    }

    StorageRef VulkanMemory::allocate(const size_t bytes, const size_t alignment,
                                      const ExecContext context) {
        return allocate_storage(bytes, alignment, false,
                                context.allocation_class == AllocationClass::Direct);
    }

    StorageRef VulkanMemory::allocate_readback(const size_t bytes) {
        const StorageRef storage = allocate_storage(bytes, 16, true, false);
        std::byte* mapped = nullptr;
        {
            std::lock_guard lock(allocations_mutex_);
            mapped = allocation_for(storage).mapped;
        }
        std::memset(mapped, 0, bytes);
        return storage;
    }

    void VulkanMemory::read_readback(const StorageRef storage, void* const destination,
                                     const size_t bytes) {
        // The producer's writes become host-visible through a HOST_READ barrier
        // recorded behind it; the wait covers both.
        const std::array writes{storage};
        const uint64_t value = context_.recorders().record(
            {}, writes, [&](const VkCommandBuffer command) {
                const VkMemoryBarrier2 host_read{
                    .sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER_2,
                    .srcStageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT,
                    .srcAccessMask = VK_ACCESS_2_MEMORY_WRITE_BIT,
                    .dstStageMask = VK_PIPELINE_STAGE_2_HOST_BIT,
                    .dstAccessMask = VK_ACCESS_2_HOST_READ_BIT,
                };
                const VkDependencyInfo dependency{
                    .sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO,
                    .memoryBarrierCount = 1,
                    .pMemoryBarriers = &host_read,
                };
                vkCmdPipelineBarrier2(command, &dependency);
            });
        context_.recorders().flush_current();
        context_.wait(value);
        context_.check_fault_buffer();
        const std::byte* source = nullptr;
        {
            std::lock_guard lock(allocations_mutex_);
            const AllocationRecord& record = allocation_for(storage);
            LFS_ASSERT_MSG(record.host_visible && record.mapped != nullptr,
                           "read_readback requires host-visible Vulkan storage");
            source = record.mapped + storage.byte_offset;
        }
        std::memcpy(destination, source, bytes);
    }

    StorageRef VulkanMemory::allocate_storage(const size_t bytes, const size_t alignment,
                                              const bool host_visible,
                                              const bool direct_class) {
        LFS_ASSERT_MSG(bytes > 0, "Vulkan allocation requires a non-zero byte count");
        LFS_ASSERT_MSG(alignment == 0 || (alignment & (alignment - 1)) == 0,
                       "Vulkan allocation alignment must be zero or a power of two");
        const VkDeviceSize bucket_size = allocation_size(bytes);
        const bool direct = bucket_size >= kDirectLimit && !host_visible;
        std::unique_ptr<AllocationRecord> record;
        if (!direct) {
            std::lock_guard lock(allocations_mutex_);
            collect_retired_locked(context_.completed_timeline());
            auto& free_lists = host_visible ? readback_free_lists_ : free_lists_;
            auto free_iterator = free_lists.find(bucket_size);
            if (free_iterator != free_lists.end() && !free_iterator->second.empty()) {
                record = std::move(free_iterator->second.back());
                free_iterator->second.pop_back();
            }
        }
        if (!record) {
            record = std::make_unique<AllocationRecord>();
            record->allocated_size = bucket_size;
            record->direct = direct;
            record->host_visible = host_visible;

            VkBufferCreateInfo buffer_info{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
            buffer_info.size = record->allocated_size;
            buffer_info.usage = kStorageUsage;
            buffer_info.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
            VmaAllocationCreateInfo allocation_info{};
            VmaAllocationInfo mapping_info{};
            if (host_visible) {
                allocation_info.flags = VMA_ALLOCATION_CREATE_HOST_ACCESS_RANDOM_BIT |
                                        VMA_ALLOCATION_CREATE_MAPPED_BIT;
                allocation_info.usage = VMA_MEMORY_USAGE_AUTO;
                allocation_info.requiredFlags = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                                                VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
            } else {
                allocation_info.usage = VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE;
                allocation_info.requiredFlags = VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT;
                allocation_info.pool = record->direct ? VK_NULL_HANDLE : device_pool_;
            }
            VkResult result = vmaCreateBuffer(
                context_.allocator(), &buffer_info, &allocation_info,
                &record->buffer, &record->allocation, &mapping_info);
            if (result == VK_ERROR_OUT_OF_DEVICE_MEMORY ||
                result == VK_ERROR_OUT_OF_HOST_MEMORY ||
                result == VK_ERROR_OUT_OF_POOL_MEMORY ||
                result == VK_ERROR_FRAGMENTED_POOL) {
                context_.recorders().wait_all();
                {
                    std::lock_guard lock(allocations_mutex_);
                    collect_retired_locked(context_.completed_timeline());
                    destroy_free_locked();
                }
                result = vmaCreateBuffer(
                    context_.allocator(), &buffer_info, &allocation_info,
                    &record->buffer, &record->allocation, &mapping_info);
            }
            vk_check(&context_, result, "vmaCreateBuffer(storage)");
            if (host_visible) {
                record->mapped = static_cast<std::byte*>(mapping_info.pMappedData);
                LFS_ASSERT_MSG(record->mapped != nullptr,
                               "host-visible Vulkan storage was not mapped");
            }
            VkBufferDeviceAddressInfo address_info{
                VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO};
            address_info.buffer = record->buffer;
            record->address =
                vkGetBufferDeviceAddress(context_.device(), &address_info);
            LFS_ASSERT_MSG(record->address != 0,
                           "vkGetBufferDeviceAddress returned zero for tensor storage");
        }
        record->requested_size = bytes;
        record->last_use = 0;
        record->descriptor_owner.backend = GpuBackend::Vulkan;
        record->descriptor_owner.pending_recorder.store(0, std::memory_order_relaxed);
        record->descriptor_owner.pending_value.store(0, std::memory_order_relaxed);
        record->descriptor_owner.generation.store(0, std::memory_order_relaxed);
        record->descriptor_owner.gpu_descriptor = GpuStorageDescriptor{
            .native_buffer = handle_value(record->buffer),
            .native_allocation = handle_value(record->allocation),
            .native_context = context_.context_id(),
            .base_address = record->address,
            .byte_size = bytes,
            .accounting_kind = StorageAccountingKind::VulkanOwned,
        };
        StorageMeta* const descriptor = &record->descriptor_owner;
        const uint64_t key = handle_value(record->allocation);
        {
            std::lock_guard lock(allocations_mutex_);
            allocations_.emplace(key, std::move(record));
        }
        return StorageRef{
            .backend = GpuBackend::Vulkan,
            .data = reinterpret_cast<void*>(static_cast<uintptr_t>(
                descriptor->gpu_descriptor.base_address)),
            .byte_offset = 0,
            .dtype = DataType::UInt8,
            .meta = descriptor,
            .flags = direct_class ? STORAGE_REF_DIRECT_ALLOCATION : 0,
        };
    }

    VulkanMemory::AllocationRecord& VulkanMemory::allocation_for(
        const StorageRef storage) const {
        LFS_ASSERT_MSG(storage.meta != nullptr,
                       "Vulkan storage is missing its native descriptor");
        const uint64_t key = storage.meta->gpu_descriptor.native_allocation;
        const auto iterator = allocations_.find(key);
        LFS_ASSERT_MSG(iterator != allocations_.end(),
                       "Vulkan storage descriptor does not name a live allocation");
        return *iterator->second;
    }

    VkBuffer VulkanMemory::buffer_for(const StorageRef storage) {
        LFS_ASSERT_MSG(storage.meta != nullptr,
                       "Vulkan storage is missing its native buffer descriptor");
        return handle_from_value<VkBuffer>(storage.meta->gpu_descriptor.native_buffer);
    }

    VkDeviceSize VulkanMemory::offset_for(const StorageRef storage) {
        return storage.byte_offset;
    }

    void VulkanMemory::mark_used(const std::span<const StorageRef> reads,
                                 const std::span<const StorageRef> writes,
                                 const uint64_t timeline_value) {
        std::lock_guard lock(allocations_mutex_);
        const auto mark = [&](const StorageRef storage) {
            if (storage.meta == nullptr || storage.backend != GpuBackend::Vulkan) {
                return;
            }
            AllocationRecord& allocation = allocation_for(storage);
            allocation.last_use = std::max(allocation.last_use, timeline_value);
        };
        std::ranges::for_each(reads, mark);
        std::ranges::for_each(writes, mark);
    }

    void VulkanMemory::copy_host_to_device(const CopyRequest& request) {
        if (request.bytes == 0) {
            return;
        }
        std::lock_guard staging_lock(staging_mutex_);
        const StagingSlice slice = acquire_staging(request.bytes, 16);
        const auto* source = static_cast<const std::byte*>(request.src.data) +
                             request.src.byte_offset;
        std::memcpy(slice.mapped, source, request.bytes);
        const std::array writes{request.dst};
        const uint64_t value = context_.recorders().record(
            {}, writes, [&](const VkCommandBuffer command) {
                const VkBufferCopy region{
                    .srcOffset = slice.offset,
                    .dstOffset = offset_for(request.dst),
                    .size = request.bytes,
                };
                vkCmdCopyBuffer(command, slice.buffer, buffer_for(request.dst), 1,
                                &region);
            });
        staging_retire_value_ = std::max(staging_retire_value_, value);
        if (request.synchronous) {
            context_.recorders().flush_storage(request.dst);
            context_.wait(value);
            context_.check_fault_buffer();
        }
    }

    void VulkanMemory::copy_device_to_host(const CopyRequest& request) {
        if (request.bytes == 0) {
            return;
        }
        std::lock_guard staging_lock(staging_mutex_);
        const StagingSlice slice = acquire_staging(request.bytes, 16);
        const std::array reads{request.src};
        const uint64_t value = context_.recorders().record(
            reads, {}, [&](const VkCommandBuffer command) {
                const VkBufferCopy region{
                    .srcOffset = offset_for(request.src),
                    .dstOffset = slice.offset,
                    .size = request.bytes,
                };
                vkCmdCopyBuffer(command, buffer_for(request.src), slice.buffer, 1,
                                &region);
                // Makes the transfer write visible to the host domain; the timeline
                // signal alone only guarantees device visibility.
                const VkMemoryBarrier2 host_read{
                    .sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER_2,
                    .srcStageMask = VK_PIPELINE_STAGE_2_TRANSFER_BIT,
                    .srcAccessMask = VK_ACCESS_2_TRANSFER_WRITE_BIT,
                    .dstStageMask = VK_PIPELINE_STAGE_2_HOST_BIT,
                    .dstAccessMask = VK_ACCESS_2_HOST_READ_BIT,
                };
                const VkDependencyInfo dependency{
                    .sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO,
                    .memoryBarrierCount = 1,
                    .pMemoryBarriers = &host_read,
                };
                vkCmdPipelineBarrier2(command, &dependency);
            });
        staging_retire_value_ = std::max(staging_retire_value_, value);
        context_.recorders().flush_current();
        context_.wait(value);
        context_.check_fault_buffer();
        auto* destination = static_cast<std::byte*>(request.dst.data) +
                            request.dst.byte_offset;
        std::memcpy(destination, slice.mapped, request.bytes);
    }

    void VulkanMemory::copy_device_to_device(const CopyRequest& request) {
        if (request.bytes == 0) {
            return;
        }
        const std::array reads{request.src};
        const std::array writes{request.dst};
        const uint64_t value = context_.recorders().record(
            reads, writes, [&](const VkCommandBuffer command) {
                const VkBufferCopy region{
                    .srcOffset = offset_for(request.src),
                    .dstOffset = offset_for(request.dst),
                    .size = request.bytes,
                };
                vkCmdCopyBuffer(command, buffer_for(request.src),
                                buffer_for(request.dst), 1, &region);
            });
        if (request.synchronous) {
            context_.recorders().flush_storage(request.dst);
            context_.wait(value);
            context_.check_fault_buffer();
        }
    }

    void VulkanMemory::memset(const FillRequest& request) {
        if (request.bytes == 0) {
            return;
        }
        // vkCmdFillBuffer requires a 4-byte-aligned dstOffset (VUID-vkCmdFillBuffer-dstOffset-00028),
        // so an odd-offset view (a UInt8/Bool/Float16 slice or append_zeros after an odd count) fills
        // only its 4-byte-aligned interior with the fill command and patches the unaligned head and
        // tail bytes through a staging copy.
        const VkDeviceSize dst_offset = offset_for(request.dst);
        const size_t head_bytes =
            std::min<size_t>(request.bytes, (4 - (static_cast<size_t>(dst_offset) & 3)) & 3);
        const size_t aligned_bytes = (request.bytes - head_bytes) & ~size_t{3};
        const size_t tail_bytes = request.bytes - head_bytes - aligned_bytes;
        const size_t edge_bytes = head_bytes + tail_bytes;
        std::unique_lock staging_lock(staging_mutex_, std::defer_lock);
        StagingSlice edge{};
        if (edge_bytes != 0) {
            staging_lock.lock();
            edge = acquire_staging(edge_bytes, 4);
            std::memset(edge.mapped, request.value, edge_bytes);
        }
        const std::array writes{request.dst};
        const uint64_t value = context_.recorders().record(
            {}, writes, [&](const VkCommandBuffer command) {
                if (aligned_bytes != 0) {
                    const uint32_t pattern = static_cast<uint32_t>(request.value) *
                                             0x01010101u;
                    vkCmdFillBuffer(command, buffer_for(request.dst),
                                    dst_offset + head_bytes, aligned_bytes, pattern);
                }
                std::array<VkBufferCopy, 2> regions{};
                uint32_t region_count = 0;
                if (head_bytes != 0) {
                    regions[region_count++] = VkBufferCopy{
                        .srcOffset = edge.offset,
                        .dstOffset = dst_offset,
                        .size = head_bytes,
                    };
                }
                if (tail_bytes != 0) {
                    regions[region_count++] = VkBufferCopy{
                        .srcOffset = edge.offset + head_bytes,
                        .dstOffset = dst_offset + head_bytes + aligned_bytes,
                        .size = tail_bytes,
                    };
                }
                if (region_count != 0) {
                    vkCmdCopyBuffer(command, edge.buffer, buffer_for(request.dst),
                                    region_count, regions.data());
                }
            });
        if (edge_bytes != 0) {
            staging_retire_value_ = std::max(staging_retire_value_, value);
        }
        if (request.synchronous) {
            context_.recorders().flush_storage(request.dst);
            context_.wait(value);
            context_.check_fault_buffer();
        }
    }

    void VulkanMemory::deallocate(const StorageRef storage) noexcept {
        try {
            if (storage.meta == nullptr) {
                return;
            }
            std::lock_guard lock(allocations_mutex_);
            if (shutting_down_) {
                return;
            }
            if (storage.meta->gpu_descriptor.native_context != context_.context_id()) {
                return;
            }
            const uint64_t key = storage.meta->gpu_descriptor.native_allocation;
            const auto iterator = allocations_.find(key);
            if (iterator == allocations_.end()) {
                return;
            }
            retired_.push_back(std::move(iterator->second));
            allocations_.erase(iterator);
            collect_retired_locked(context_.completed_timeline());
        } catch (...) {
        }
    }

    void VulkanMemory::collect_retired_locked(const uint64_t completed) {
        std::erase_if(retired_, [&](auto& record) {
            if (record->last_use > completed) {
                return false;
            }
            if (record->direct) {
                vmaDestroyBuffer(context_.allocator(), record->buffer,
                                 record->allocation);
            } else if (record->host_visible) {
                readback_free_lists_[record->allocated_size].push_back(std::move(record));
            } else {
                free_lists_[record->allocated_size].push_back(std::move(record));
            }
            return true;
        });
    }

    void VulkanMemory::destroy_free_locked() {
        for (auto* lists : {&free_lists_, &readback_free_lists_}) {
            for (auto& [size, records] : *lists) {
                (void)size;
                for (auto& record : records) {
                    vmaDestroyBuffer(context_.allocator(), record->buffer,
                                     record->allocation);
                }
            }
            lists->clear();
        }
    }

    void VulkanMemory::trim() {
        context_.recorders().wait_all();
        std::lock_guard lock(allocations_mutex_);
        collect_retired_locked(context_.completed_timeline());
        destroy_free_locked();
    }

    MemoryInfo VulkanMemory::stats() const {
        MemoryInfo result;
        std::array<VmaBudget, VK_MAX_MEMORY_HEAPS> budgets{};
        vmaGetHeapBudgets(context_.allocator(), budgets.data());
        for (uint32_t index = 0;
             index < context_.memory_properties().memoryHeapCount; ++index) {
            if ((context_.memory_properties().memoryHeaps[index].flags &
                 VK_MEMORY_HEAP_DEVICE_LOCAL_BIT) == 0) {
                continue;
            }
            result.total_bytes += budgets[index].budget;
            result.allocated_bytes += budgets[index].usage;
        }
        result.free_bytes = result.total_bytes > result.allocated_bytes
                                ? result.total_bytes - result.allocated_bytes
                                : 0;
        result.device_id = static_cast<int>(context_.device_index());
        return result;
    }

    size_t VulkanMemory::cached_bytes() const noexcept {
        std::lock_guard lock(allocations_mutex_);
        size_t result = 0;
        for (const auto& [size, records] : free_lists_) {
            result += static_cast<size_t>(size) * records.size();
        }
        return result;
    }

    uint64_t VulkanMemory::live_object_count() const noexcept {
        std::lock_guard lock(allocations_mutex_);
        size_t free_count = 0;
        for (const auto& [size, records] : free_lists_) {
            (void)size;
            free_count += records.size();
        }
        return allocations_.size() + retired_.size() + free_count +
               (staging_buffer_ != VK_NULL_HANDLE ? 1 : 0);
    }

    bool VulkanMemory::owns_address(const void* const pointer) const noexcept {
        const uint64_t address = reinterpret_cast<uintptr_t>(pointer);
        std::lock_guard lock(allocations_mutex_);
        return std::ranges::any_of(allocations_, [address](const auto& entry) {
            const auto& record = entry.second;
            return address >= record->address &&
                   address < record->address + record->requested_size;
        });
    }

    void VulkanMemory::shutdown() {
        std::scoped_lock lock(allocations_mutex_, staging_mutex_);
        if (shutting_down_) {
            return;
        }
        shutting_down_ = true;
        // Client allocations still alive at shutdown are a leak unless a test keeps
        // them on purpose; the count is exposed so the G4 fixture can assert on it.
        g_last_shutdown_live_allocations.store(allocations_.size(), std::memory_order_release);
        for (auto& [key, record] : allocations_) {
            (void)key;
            vmaDestroyBuffer(context_.allocator(), record->buffer,
                             record->allocation);
        }
        allocations_.clear();
        for (auto& record : retired_) {
            vmaDestroyBuffer(context_.allocator(), record->buffer,
                             record->allocation);
        }
        retired_.clear();
        destroy_free_locked();
        if (staging_buffer_ != VK_NULL_HANDLE) {
            vmaDestroyBuffer(context_.allocator(), staging_buffer_, staging_allocation_);
            staging_buffer_ = VK_NULL_HANDLE;
            staging_allocation_ = VK_NULL_HANDLE;
            staging_mapped_ = nullptr;
        }
        if (device_pool_ != VK_NULL_HANDLE) {
            vmaDestroyPool(context_.allocator(), device_pool_);
            device_pool_ = VK_NULL_HANDLE;
        }
    }

} // namespace lfs::core::internal

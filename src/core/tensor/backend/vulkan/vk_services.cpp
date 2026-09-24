/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "../facade_trace.hpp"
#include "../readback_buffer.hpp"
#include "../tensor_completion.hpp"
#include "vk_backend_ops.hpp"

#include "core/logger.hpp"
#include "core/vulkan_helpers.hpp"

#include "../../internal/tensor_impl.hpp"
#include "vk_context.hpp"
#include "vk_memory.hpp"
#include "vk_recorder.hpp"

namespace lfs::core::internal {

    namespace {
        class VulkanReadbackBuffer final : public ReadbackBuffer {
        public:
            VulkanReadbackBuffer() : context_(acquire_vulkan_context()) {}

            ~VulkanReadbackBuffer() override {
                try {
                    if (pending_)
                        wait();
                } catch (const std::exception& error) {
                    LOG_WARN("Vulkan readback teardown failed: {}", error.what());
                }
                if (capacity_)
                    context_->memory().deallocate(storage_);
            }

            void enqueue(StorageRef source, size_t bytes, ExecContext) override {
                LFS_FACADE_TRACE(service_enqueue_readback);
                LFS_ASSERT_MSG(!pending_, "Readback staging is already in use");
                if (bytes > capacity_) {
                    auto replacement = context_->memory().allocate_readback(bytes);
                    if (capacity_)
                        context_->memory().deallocate(storage_);
                    storage_ = replacement;
                    capacity_ = bytes;
                }
                bytes_ = bytes;
                completion_ = {};
                pending_ = true;
                completion_ = TensorCompletionAccess::vulkan(
                    context_->memory().copy_to_readback(source, storage_, bytes));
            }

            void wait() override {
                if (completion_.timeline().value)
                    completion_.wait();
                else {
                    context_->recorders().wait_all();
                    context_->check_fault_buffer();
                }
            }

            bool poll(void* destination) override {
                LFS_FACADE_TRACE(service_readback_poll);
                if (!completion_.ready())
                    return false;
                context_->check_fault_buffer();
                if (destination)
                    context_->memory().copy_mapped(storage_, destination, bytes_);
                pending_ = false;
                return true;
            }

        private:
            std::shared_ptr<VulkanContext> context_;
            StorageRef storage_{};
            size_t capacity_ = 0, bytes_ = 0;
            TensorCompletion completion_;
            bool pending_ = false;
        };
    } // namespace

    std::unique_ptr<ReadbackBuffer> VulkanBackendOps::create_readback_buffer() {
        return std::make_unique<VulkanReadbackBuffer>();
    }

    StorageRef VulkanBackendOps::allocate(const size_t bytes, const size_t alignment,
                                          const ExecContext context) {
        LFS_FACADE_TRACE(service_allocate);
        return acquire_vulkan_context()->memory().allocate(bytes, alignment, context);
    }

    void VulkanBackendOps::deallocate(const StorageRef storage, ExecContext) noexcept {
        try {
            if (const auto context = try_live_vulkan_context()) {
                context->memory().deallocate(storage);
            }
        } catch (const std::exception& error) {
            LOG_WARN("Vulkan storage release failed: {}", error.what());
        }
    }

    void VulkanBackendOps::record_stream(StorageRef, ExecContext) {}

    void VulkanBackendOps::release_stream(ExecContext) {}

    void VulkanBackendOps::rehome_stream(StorageRef, ExecContext) {}

    void VulkanBackendOps::trim() {
        acquire_vulkan_context()->memory().trim();
    }

    void VulkanBackendOps::trim_if_reserved_unused_exceeds(
        const size_t threshold_bytes) {
        const auto context = acquire_vulkan_context();
        if (context->memory().cached_bytes() > threshold_bytes) {
            context->memory().trim();
        }
    }

    MemoryInfo VulkanBackendOps::stats() {
        return acquire_vulkan_context()->memory().stats();
    }

    void VulkanBackendOps::shutdown() {
        shutdown_vulkan_context();
    }

    void VulkanBackendOps::set_allocation_iteration(int) {}

    void VulkanBackendOps::record_tensor_allocation(
        StorageRef, const StridedLayout&, size_t) {}

    void VulkanBackendOps::copy_host_to_device(const CopyRequest& request) {
        LFS_FACADE_TRACE(service_copy_host_to_device);
        acquire_vulkan_context()->memory().copy_host_to_device(request);
    }

    void VulkanBackendOps::copy_device_to_host(const CopyRequest& request) {
        LFS_FACADE_TRACE(service_copy_device_to_host);
        acquire_vulkan_context()->memory().copy_device_to_host(request);
    }

    void VulkanBackendOps::copy_device_to_device(const CopyRequest& request) {
        LFS_FACADE_TRACE(service_copy_device_to_device);
        acquire_vulkan_context()->memory().copy_device_to_device(request);
    }

    void VulkanBackendOps::memset(const FillRequest& request) {
        LFS_FACADE_TRACE(service_memset);
        acquire_vulkan_context()->memory().memset(request);
    }

    void VulkanBackendOps::synchronize_stream(ExecContext) {
        LFS_FACADE_TRACE(service_synchronize_stream);
        const auto context = acquire_vulkan_context();
        context->recorders().wait_all();
        context->check_fault_buffer();
    }

    void VulkanBackendOps::synchronize_device() {
        synchronize_stream({});
    }

    void VulkanBackendOps::wait_for(const SyncToken token) {
        LFS_ASSERT_MSG(token.backend == GpuBackend::Vulkan,
                       "Vulkan sync service received a non-Vulkan token");
        const auto context = acquire_vulkan_context();
        context->recorders().flush_all();
        context->wait(token.value);
        context->check_fault_buffer();
    }

    SyncToken VulkanBackendOps::bridge(ExecContext, ExecContext) {
        const uint64_t value = acquire_vulkan_context()->recorders().flush_all();
        return SyncToken{
            .backend = GpuBackend::Vulkan,
            .value = value,
            .native = 0,
        };
    }

    PointerClass VulkanBackendOps::classify_pointer(const void* const pointer) {
        if (pointer == nullptr) {
            return PointerClass::Unknown;
        }
        if (const auto context = try_live_vulkan_context();
            context && context->memory().owns_address(pointer)) {
            return PointerClass::Device;
        }
        return PointerClass::Unknown;
    }

    bool VulkanBackendOps::stream_is_capturing(ExecContext) {
        return false;
    }

} // namespace lfs::core::internal

#include "../tensor_vulkan_interop.hpp"
#include "core/tensor_backend.hpp"
#include <cstring>

namespace lfs::core::internal {
    std::optional<TensorVulkanBuffer> native_vulkan_buffer(const Tensor& tensor) {
        if (!tensor.is_valid()) {
            return std::nullopt;
        }
        tensor.materialize_if_deferred();
        if (gpu_backend_of(tensor) != GpuBackend::Vulkan) {
            return std::nullopt;
        }
        const auto context = try_live_vulkan_context();
        if (!context) {
            return std::nullopt;
        }
        const StorageRef storage = storage_ref(tensor);
        if (storage.meta == nullptr ||
            storage.backend != GpuBackend::Vulkan ||
            storage.meta->gpu_descriptor.native_buffer == 0) {
            return std::nullopt;
        }
        TensorVulkanBuffer result;
        result.buffer = reinterpret_cast<void*>(static_cast<uintptr_t>(
            storage.meta->gpu_descriptor.native_buffer));
        result.offset = storage.byte_offset;
        result.device_address =
            storage.meta->gpu_descriptor.base_address + storage.byte_offset;
        result.bytes = tensor.bytes();
        result.device = context->device();
        result.pending_timeline_value =
            storage.meta->pending_value.load(std::memory_order_acquire);
        result.keep_alive = tensor.data_owner_;
        return result;
    }

    namespace {
        struct ForeignTensorBuffer {
            VkDevice device = VK_NULL_HANDLE;
            VkBuffer buffer = VK_NULL_HANDLE;
            VkDeviceMemory memory = VK_NULL_HANDLE;
            void* mapped = nullptr;
            uint64_t address = 0;
            size_t bytes = 0;
            uint64_t pending = 0;
            uint64_t generation = 0;

            ~ForeignTensorBuffer() { release(); }
            void release() {
                if (mapped)
                    vkUnmapMemory(device, memory);
                if (buffer)
                    vkDestroyBuffer(device, buffer, nullptr);
                if (memory)
                    vkFreeMemory(device, memory, nullptr);
                mapped = nullptr;
                buffer = VK_NULL_HANDLE;
                memory = VK_NULL_HANDLE;
            }
        };

        class VulkanTensorVulkanInterop final : public TensorVulkanInteropBackend {
        public:
            explicit VulkanTensorVulkanInterop(VulkanInteropDevice target) : target_(target) {}

            void shutdown() override {
                for (auto& weak : foreign_) {
                    if (auto buffer = weak.lock())
                        buffer->release();
                }
                foreign_.clear();
            }

            void release_timeline(void*) override {}

            void drain() override {
                auto context = acquire_vulkan_context();
                context->recorders().wait_all();
            }

            std::shared_ptr<void> execution_scope() override {
                return std::make_shared<GpuBackendScope>(GpuBackend::Vulkan);
            }

            Tensor empty_splat(TensorShape shape, size_t capacity, DataType dtype,
                               std::string_view, bool) override {
                return empty(std::move(shape), dtype, capacity);
            }

            Tensor empty(TensorShape shape, DataType dtype, size_t capacity) override {
                GpuBackendScope scope(GpuBackend::Vulkan);
                auto result = Tensor::empty(std::move(shape), Device::GPU, dtype);
                if (result.ndim() > 0 && capacity > result.size(0))
                    result.reserve(capacity);
                return result;
            }

            std::optional<TensorVulkanBuffer> buffer(const Tensor& tensor) override {
                auto result = native_vulkan_buffer(tensor);
                if (result && result->device != target_.device)
                    return foreign_buffer(tensor);
                return result;
            }

            TensorCompletion ready(std::span<const Tensor* const> tensors) override {
                const auto context = acquire_vulkan_context();
                if (context->device() != target_.device)
                    return {};
                std::vector<StorageRef> storage;
                for (const auto* tensor : tensors) {
                    if (tensor && tensor->is_valid() && gpu_backend_of(*tensor) == GpuBackend::Vulkan)
                        storage.push_back(storage_ref(*tensor));
                }
                return TensorCompletionAccess::vulkan(context->recorders().flush_storages(storage));
            }

            void wait(std::span<const Tensor* const> tensors, VulkanTimelinePoint point) override {
                const auto context = acquire_vulkan_context();
                if (context->device() != target_.device) {
                    VkSemaphore semaphore = static_cast<VkSemaphore>(point.semaphore);
                    VkSemaphoreWaitInfo wait{VK_STRUCTURE_TYPE_SEMAPHORE_WAIT_INFO};
                    wait.semaphoreCount = 1;
                    wait.pSemaphores = &semaphore;
                    wait.pValues = &point.value;
                    VkResult status;
                    do {
                        status = vkWaitSemaphores(static_cast<VkDevice>(target_.device), &wait, 1'000'000'000);
                    } while (status == VK_TIMEOUT);
                    vk_check(nullptr, status, "vkWaitSemaphores(foreign tensor consumer)");
                    for (const auto* tensor : tensors) {
                        const auto storage = storage_ref(*tensor);
                        std::shared_ptr<ForeignTensorBuffer> buffer;
                        {
                            std::lock_guard lock(storage.meta->vulkan_interop_mutex);
                            const auto found = storage.meta->vulkan_interop_owners.find(target_.device);
                            if (found == storage.meta->vulkan_interop_owners.end())
                                continue;
                            buffer = std::static_pointer_cast<ForeignTensorBuffer>(found->second);
                        }
                        Tensor host = Tensor::empty(tensor->shape(), Device::CPU, tensor->dtype());
                        std::memcpy(host.data_ptr(), static_cast<const std::byte*>(buffer->mapped) + storage.byte_offset, tensor->bytes());
                        Tensor destination = *tensor;
                        destination.copy_from(host);
                        buffer->pending = storage.meta->pending_value.load(std::memory_order_acquire);
                        buffer->generation = storage.meta->generation.load(std::memory_order_acquire);
                    }
                    return;
                }
                std::vector<StorageRef> storage;
                for (const auto* tensor : tensors)
                    storage.push_back(storage_ref(*tensor));
                context->recorders().wait_external(storage, static_cast<VkSemaphore>(point.semaphore),
                                                   point.value, std::move(point.keep_alive));
            }

        private:
            TensorVulkanBuffer foreign_buffer(const Tensor& tensor) {
                const auto storage = storage_ref(tensor);
                std::lock_guard lock(storage.meta->vulkan_interop_mutex);
                const auto pending = storage.meta->pending_value.load(std::memory_order_acquire);
                const auto generation = storage.meta->generation.load(std::memory_order_acquire);
                auto& cached = storage.meta->vulkan_interop_owners[target_.device];
                auto buffer = std::static_pointer_cast<ForeignTensorBuffer>(cached);
                if (!buffer || buffer->pending != pending || buffer->generation != generation ||
                    buffer->bytes != storage.meta->gpu_descriptor.byte_size) {
                    buffer = std::make_shared<ForeignTensorBuffer>();
                    buffer->device = static_cast<VkDevice>(target_.device);
                    buffer->bytes = storage.meta->gpu_descriptor.byte_size;
                    VkBufferCreateInfo info{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
                    info.size = buffer->bytes;
                    info.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT |
                                 VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT;
                    info.sharingMode = target_.queue_family_count > 1 ? VK_SHARING_MODE_CONCURRENT : VK_SHARING_MODE_EXCLUSIVE;
                    if (target_.queue_family_count > 1) {
                        info.queueFamilyIndexCount = target_.queue_family_count;
                        info.pQueueFamilyIndices = target_.queue_families.data();
                    }
                    vk_check(nullptr, vkCreateBuffer(buffer->device, &info, nullptr, &buffer->buffer), "vkCreateBuffer(foreign tensor)");
                    VkMemoryRequirements requirements{};
                    vkGetBufferMemoryRequirements(buffer->device, buffer->buffer, &requirements);
                    VkPhysicalDeviceMemoryProperties memory{};
                    vkGetPhysicalDeviceMemoryProperties(static_cast<VkPhysicalDevice>(target_.physical_device), &memory);
                    constexpr auto needed = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
                    const uint32_t memory_type = find_vulkan_memory_type(memory, requirements.memoryTypeBits, needed);
                    if (memory_type == std::numeric_limits<uint32_t>::max())
                        throw TensorError("Consumer device has no coherent host-visible tensor memory");
                    VkMemoryAllocateFlagsInfo flags{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_FLAGS_INFO};
                    flags.flags = VK_MEMORY_ALLOCATE_DEVICE_ADDRESS_BIT;
                    VkMemoryAllocateInfo allocation{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
                    allocation.pNext = &flags;
                    allocation.allocationSize = requirements.size;
                    allocation.memoryTypeIndex = memory_type;
                    vk_check(nullptr, vkAllocateMemory(buffer->device, &allocation, nullptr, &buffer->memory), "vkAllocateMemory(foreign tensor)");
                    vk_check(nullptr, vkBindBufferMemory(buffer->device, buffer->buffer, buffer->memory, 0), "vkBindBufferMemory(foreign tensor)");
                    vk_check(nullptr, vkMapMemory(buffer->device, buffer->memory, 0, buffer->bytes, 0, &buffer->mapped), "vkMapMemory(foreign tensor)");
                    auto base_storage = storage;
                    base_storage.byte_offset = 0;
                    VulkanReadbackBuffer readback;
                    readback.enqueue(base_storage, buffer->bytes, {});
                    readback.wait();
                    if (!readback.poll(buffer->mapped))
                        throw TensorError("Foreign tensor readback did not complete");
                    VkBufferDeviceAddressInfo address{VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO};
                    address.buffer = buffer->buffer;
                    buffer->address = vkGetBufferDeviceAddress(buffer->device, &address);
                    buffer->pending = storage.meta->pending_value.load(std::memory_order_acquire);
                    buffer->generation = storage.meta->generation.load(std::memory_order_acquire);
                    cached = buffer;
                    std::erase_if(foreign_, [](const auto& value) { return value.expired(); });
                    foreign_.push_back(buffer);
                }
                return {.buffer = buffer->buffer, .offset = storage.byte_offset, .device_address = buffer->address + storage.byte_offset, .bytes = tensor.bytes(), .keep_alive = buffer, .device = target_.device};
            }

            std::vector<std::weak_ptr<ForeignTensorBuffer>> foreign_;
            VulkanInteropDevice target_;
        };
    } // namespace

    std::shared_ptr<TensorVulkanInteropBackend> make_vulkan_vulkan_interop(VulkanInteropDevice target) {
        return std::make_shared<VulkanTensorVulkanInterop>(target);
    }
} // namespace lfs::core::internal

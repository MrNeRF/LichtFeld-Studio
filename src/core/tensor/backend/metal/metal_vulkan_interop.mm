/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

// Shares Metal tensors with a Vulkan (MoltenVK) consumer without copies:
// VK_EXT_metal_objects imports every storage's MTLBuffer into the consumer
// device, and an event that the Metal queue signals behind tensor work into
// a timeline semaphore, so the consumer's queue waits on the GPU. Work queues
// also wait the other way, on the event behind the consumer's timeline.

#include "../tensor_completion.hpp"
#include "../tensor_vulkan_interop.hpp"
#include "metal_context.hpp"

#include "core/tensor_backend.hpp"
#include "core/vulkan_helpers.hpp"

#include <vulkan/vulkan.h>
#include <vulkan/vulkan_metal.h>

#include <algorithm>
#include <limits>

namespace lfs::core::internal {
    namespace {
        void check(const VkResult result, const char* const operation) {
            if (result != VK_SUCCESS)
                throw TensorError(std::format("{} failed for a Metal tensor ({})", operation, static_cast<int>(result)));
        }

        // A consumer-device object that the interop session can release before
        // the device goes away, even while leases still hold it.
        struct Imported {
            VkDevice device = VK_NULL_HANDLE;
            VkBuffer buffer = VK_NULL_HANDLE;
            VkDeviceMemory memory = VK_NULL_HANDLE;
            VkSemaphore semaphore = VK_NULL_HANDLE;
            id<MTLSharedEvent> event;
            uint64_t address = 0;
            uint64_t context = 0;

            ~Imported() { release(); }

            void release() {
                if (buffer)
                    vkDestroyBuffer(device, buffer, nullptr);
                if (memory)
                    vkFreeMemory(device, memory, nullptr);
                if (semaphore)
                    vkDestroySemaphore(device, semaphore, nullptr);
                buffer = VK_NULL_HANDLE;
                memory = VK_NULL_HANDLE;
                semaphore = VK_NULL_HANDLE;
            }
        };

        // The consumer holds the tensor with the import, so neither the storage
        // nor its Metal block is reused while consumer work may still read it.
        struct Lease {
            Tensor tensor;
            std::shared_ptr<Imported> buffer;
        };

        // A timeline semaphore of the device that a new Metal event backs.
        // MoltenVK sets an imported event to the initial value, so the event
        // must be new and only the Metal queue may advance it.
        std::shared_ptr<Imported> import_timeline(const VkDevice device, id<MTLDevice> const metal) {
            auto imported = std::make_shared<Imported>();
            imported->device = device;
            imported->event = [metal newSharedEvent];
            if (!imported->event)
                throw TensorError("Metal could not create the Vulkan interop event");
            VkImportMetalSharedEventInfoEXT source{VK_STRUCTURE_TYPE_IMPORT_METAL_SHARED_EVENT_INFO_EXT};
            source.mtlSharedEvent = imported->event;
            VkSemaphoreTypeCreateInfo type{VK_STRUCTURE_TYPE_SEMAPHORE_TYPE_CREATE_INFO};
            type.pNext = &source;
            type.semaphoreType = VK_SEMAPHORE_TYPE_TIMELINE;
            VkSemaphoreCreateInfo info{VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO};
            info.pNext = &type;
            check(vkCreateSemaphore(device, &info, nullptr, &imported->semaphore), "vkCreateSemaphore");
            return imported;
        }

        class API_AVAILABLE(macos(26.0)) MetalTensorVulkanInterop final : public TensorVulkanInteropBackend {
        public:
            explicit MetalTensorVulkanInterop(const VulkanInteropDevice target) : target_(target) {
                if (!target.metal_objects)
                    throw TensorError("Sharing Metal tensors with Vulkan requires VK_EXT_metal_objects");
            }

            void shutdown() override {
                std::lock_guard lock(mutex_);
                for (const auto& weak : imports_) {
                    if (const auto imported = weak.lock())
                        imported->release();
                }
                imports_.clear();
                timeline_.reset();
            }

            void release_timeline(void*) override {}

            void drain() override { backend_ops(GpuBackend::Metal).synchronize_device(); }

            std::shared_ptr<void> execution_scope() override {
                return std::make_shared<GpuBackendScope>(GpuBackend::Metal);
            }

            Tensor empty_splat(TensorShape shape, const size_t capacity, const DataType dtype, std::string_view,
                               bool) override {
                return empty(std::move(shape), dtype, capacity);
            }

            Tensor empty(TensorShape shape, const DataType dtype, const size_t capacity) override {
                const GpuBackendScope scope(GpuBackend::Metal);
                auto result = Tensor::empty(std::move(shape), Device::GPU, dtype);
                if (result.ndim() > 0 && capacity > result.size(0))
                    result.reserve(capacity);
                return result;
            }

            std::optional<TensorVulkanBuffer> buffer(const Tensor& tensor) override {
                if (!tensor.is_valid() || gpu_backend_of(tensor) != GpuBackend::Metal)
                    return std::nullopt;
                const StorageRef storage = storage_ref(tensor);
                std::shared_ptr<Imported> imported;
                {
                    std::lock_guard lock(storage.meta->vulkan_interop_mutex);
                    auto& cached = storage.meta->vulkan_interop_owners[target_.device];
                    imported = std::static_pointer_cast<Imported>(cached);
                    if (!imported || !imported->buffer) {
                        imported = import_buffer(*storage.meta);
                        cached = imported;
                    }
                }
                return TensorVulkanBuffer{
                    .buffer = imported->buffer,
                    .offset = storage.byte_offset,
                    .device_address = imported->address + storage.byte_offset,
                    .bytes = tensor.bytes(),
                    .pending_timeline_value = storage.meta->pending_value.load(std::memory_order_acquire),
                    .keep_alive = std::make_shared<Lease>(Lease{tensor, imported}),
                    .device = target_.device,
                };
            }

            TensorCompletion ready(const std::span<const Tensor* const> tensors) override {
                uint64_t serial = 0;
                for (const auto* tensor : tensors) {
                    if (!tensor || !tensor->is_valid() || gpu_backend_of(*tensor) != GpuBackend::Metal)
                        continue;
                    serial = std::max(serial, storage_ref(*tensor).meta->pending_value.load(std::memory_order_acquire));
                }
                const auto context = metal::acquire_context();
                if (serial <= context->completed())
                    return {};
                const auto timeline = this->timeline(*context);
                return TensorCompletionAccess::external(target_.device,
                                                        {timeline->semaphore, context->signal(timeline->event), timeline});
            }

            // The consumer's queue is foreign to Metal, so its work completes on the host.
            void wait(std::span<const Tensor* const>, const VulkanTimelinePoint point) override {
                auto semaphore = static_cast<VkSemaphore>(point.semaphore);
                VkSemaphoreWaitInfo wait{VK_STRUCTURE_TYPE_SEMAPHORE_WAIT_INFO};
                wait.semaphoreCount = 1;
                wait.pSemaphores = &semaphore;
                wait.pValues = &point.value;
                VkResult status;
                do {
                    status = vkWaitSemaphores(static_cast<VkDevice>(target_.device), &wait, 1'000'000'000);
                } while (status == VK_TIMEOUT);
                check(status, "vkWaitSemaphores");
            }

        private:
            std::shared_ptr<Imported> import_buffer(const StorageMeta& meta) {
                const auto device = static_cast<VkDevice>(target_.device);
                auto imported = std::make_shared<Imported>();
                imported->device = device;
                VkBufferCreateInfo info{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
                info.size = meta.gpu_descriptor.byte_size;
                info.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT |
                             VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT;
                info.sharingMode = target_.queue_family_count > 1 ? VK_SHARING_MODE_CONCURRENT : VK_SHARING_MODE_EXCLUSIVE;
                if (target_.queue_family_count > 1) {
                    info.queueFamilyIndexCount = target_.queue_family_count;
                    info.pQueueFamilyIndices = target_.queue_families.data();
                }
                check(vkCreateBuffer(device, &info, nullptr, &imported->buffer), "vkCreateBuffer");
                VkMemoryRequirements requirements{};
                vkGetBufferMemoryRequirements(device, imported->buffer, &requirements);
                VkPhysicalDeviceMemoryProperties memory{};
                vkGetPhysicalDeviceMemoryProperties(static_cast<VkPhysicalDevice>(target_.physical_device), &memory);
                // Metal tensors live in shared storage, which MoltenVK maps as coherent host memory.
                const uint32_t type = find_vulkan_memory_type(
                    memory, requirements.memoryTypeBits, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
                if (type == std::numeric_limits<uint32_t>::max())
                    throw TensorError("The Vulkan device has no memory type for shared Metal buffers");
                id<MTLBuffer> const buffer = (__bridge id<MTLBuffer>)reinterpret_cast<void*>(meta.gpu_descriptor.native_buffer);
                VkImportMetalBufferInfoEXT source{VK_STRUCTURE_TYPE_IMPORT_METAL_BUFFER_INFO_EXT};
                source.mtlBuffer = buffer;
                VkMemoryAllocateFlagsInfo flags{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_FLAGS_INFO};
                flags.pNext = &source;
                flags.flags = VK_MEMORY_ALLOCATE_DEVICE_ADDRESS_BIT;
                VkMemoryAllocateInfo allocation{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
                allocation.pNext = &flags;
                allocation.allocationSize = meta.gpu_descriptor.byte_size;
                allocation.memoryTypeIndex = type;
                check(vkAllocateMemory(device, &allocation, nullptr, &imported->memory), "vkAllocateMemory");
                check(vkBindBufferMemory(device, imported->buffer, imported->memory, 0), "vkBindBufferMemory");
                VkBufferDeviceAddressInfo address{VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO};
                address.buffer = imported->buffer;
                imported->address = vkGetBufferDeviceAddress(device, &address);
                // MoltenVK backs buffers with MTLHeaps unless the instance turns them off.
                if (imported->address != buffer.gpuAddress)
                    throw TensorError("The Vulkan device does not alias imported Metal buffers");
                remember(imported);
                return imported;
            }

            // Serials restart with a new Metal context, so each context signals
            // its own timeline.
            std::shared_ptr<Imported> timeline(metal::Context& context) {
                std::lock_guard lock(mutex_);
                if (timeline_ && timeline_->semaphore && timeline_->context == context.context_id())
                    return timeline_;
                auto imported = import_timeline(static_cast<VkDevice>(target_.device), context.device());
                imported->context = context.context_id();
                imports_.push_back(imported);
                timeline_ = imported;
                return imported;
            }

            void remember(const std::shared_ptr<Imported>& imported) {
                std::lock_guard lock(mutex_);
                std::erase_if(imports_, [](const auto& weak) { return weak.expired(); });
                imports_.push_back(imported);
            }

            VulkanInteropDevice target_;
            std::mutex mutex_;
            std::vector<std::weak_ptr<Imported>> imports_;
            std::shared_ptr<Imported> timeline_;
        };

        class API_AVAILABLE(macos(26.0)) MetalQueue final : public MetalVulkanQueue {
        public:
            MetalQueue(const VkDevice device, const VkSemaphore consumer) {
                if (!device)
                    return;
                timeline_ = import_timeline(device, metal::acquire_context()->device());
                if (!consumer)
                    return;
                const auto export_objects = reinterpret_cast<PFN_vkExportMetalObjectsEXT>(
                    vkGetDeviceProcAddr(device, "vkExportMetalObjectsEXT"));
                VkExportMetalSharedEventInfoEXT event{VK_STRUCTURE_TYPE_EXPORT_METAL_SHARED_EVENT_INFO_EXT};
                event.semaphore = consumer;
                VkExportMetalObjectsInfoEXT info{VK_STRUCTURE_TYPE_EXPORT_METAL_OBJECTS_INFO_EXT};
                info.pNext = &event;
                if (export_objects)
                    export_objects(device, &info);
                if (!event.mtlSharedEvent)
                    throw TensorError("The consumer timeline has no Metal event to wait on");
                consumer_ = event.mtlSharedEvent;
            }

            void* timeline() const override { return timeline_ ? timeline_->semaphore : VK_NULL_HANDLE; }

            void wait(const uint64_t consumer_value) override {
                if (consumer_)
                    metal::acquire_context()->queue_wait(consumer_, consumer_value);
            }

            TensorCompletion signal() override {
                const auto context = metal::acquire_context();
                if (!timeline_)
                    return TensorCompletionAccess::metal(context->flush());
                const uint64_t serial = context->signal(timeline_->event);
                return TensorCompletionAccess::metal(serial, {timeline_->semaphore, serial, timeline_});
            }

        private:
            std::shared_ptr<Imported> timeline_;
            id<MTLSharedEvent> consumer_;
        };
    } // namespace

    std::shared_ptr<TensorVulkanInteropBackend> make_metal_vulkan_interop(const VulkanInteropDevice target) {
        if (@available(macOS 26.0, *))
            return std::make_shared<MetalTensorVulkanInterop>(target);
        throw TensorError("Metal tensors require macOS 26");
    }

    std::unique_ptr<MetalVulkanQueue> make_metal_vulkan_queue(void* const device, void* const consumer_timeline) {
        if (@available(macOS 26.0, *))
            return std::make_unique<MetalQueue>(static_cast<VkDevice>(device), static_cast<VkSemaphore>(consumer_timeline));
        throw TensorError("Metal tensors require macOS 26");
    }
} // namespace lfs::core::internal

/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#include "core/tensor_backend.hpp"
#include "core/tensor_metal_reader.hpp"
#include "metal_context.hpp"
#ifdef LFS_TENSOR_VULKAN
#include "../vulkan/vk_context.hpp"
#include "../vulkan/vk_memory.hpp"
#include "../vulkan/vk_recorder.hpp"
#include <vulkan/vulkan_metal.h>
#endif
#include <atomic>
#include <limits>
#include <mutex>
#include <stdexcept>
#include <utility>
#include <vector>

namespace lfs::core {
#ifdef LFS_TENSOR_VULKAN
    namespace {
        struct VulkanConsumerEvent {
            VkDevice device;
            VkSemaphore semaphore = VK_NULL_HANDLE;
            ~VulkanConsumerEvent() { release(); }
            void release() {
                if (semaphore)
                    vkDestroySemaphore(device, std::exchange(semaphore, VK_NULL_HANDLE), nullptr);
            }
        };
        struct VulkanNativeAccess {
            std::shared_ptr<internal::VulkanContext> context;
            std::shared_ptr<VulkanConsumerEvent> consumer;
            id<MTLSharedEvent> producer, done;
        };
        std::shared_ptr<VulkanNativeAccess> vulkanAccess(id<MTLDevice> device) {
            const auto context = internal::acquire_vulkan_context();
            if (!context->caps().metal_objects || context->dead())
                throw std::invalid_argument("Native Metal access requires a live Vulkan Metal-object device");
            const auto export_objects = reinterpret_cast<PFN_vkExportMetalObjectsEXT>(vkGetDeviceProcAddr(context->device(), "vkExportMetalObjectsEXT"));
            if (!export_objects)
                throw std::invalid_argument("Vulkan device cannot export native Metal objects");
            VkExportMetalSharedEventInfoEXT producer{VK_STRUCTURE_TYPE_EXPORT_METAL_SHARED_EVENT_INFO_EXT};
            producer.semaphore = context->timeline();
            VkExportMetalObjectsInfoEXT export_info{VK_STRUCTURE_TYPE_EXPORT_METAL_OBJECTS_INFO_EXT};
            export_info.pNext = &producer;
            export_objects(context->device(), &export_info);
            const auto done = [device newSharedEvent];
            if (!producer.mtlSharedEvent || !done)
                throw std::invalid_argument("Vulkan tensor timeline cannot synchronize with native Metal");
            auto consumer = std::make_shared<VulkanConsumerEvent>();
            consumer->device = context->device();
            VkImportMetalSharedEventInfoEXT imported{VK_STRUCTURE_TYPE_IMPORT_METAL_SHARED_EVENT_INFO_EXT};
            imported.mtlSharedEvent = done;
            VkSemaphoreTypeCreateInfo type{VK_STRUCTURE_TYPE_SEMAPHORE_TYPE_CREATE_INFO};
            type.semaphoreType = VK_SEMAPHORE_TYPE_TIMELINE;
            type.pNext = &imported;
            VkSemaphoreCreateInfo info{VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO};
            info.pNext = &type;
            internal::vk_check(context.get(), vkCreateSemaphore(context->device(), &info, nullptr, &consumer->semaphore), "Import native tensor consumer event");
            const std::weak_ptr<VulkanConsumerEvent> weak = consumer;
            context->on_shutdown([weak] { if (auto event = weak.lock()) event->release(); });
            return std::make_shared<VulkanNativeAccess>(VulkanNativeAccess{context, consumer, producer.mtlSharedEvent, done});
        }
        MetalTensorView vulkanView(const Tensor& tensor, const VulkanNativeAccess& access, id<MTLDevice> device) {
            const auto storage = internal::storage_ref(tensor);
            if (!storage.meta || storage.meta->gpu_descriptor.native_context != access.context->context_id())
                throw std::invalid_argument("Vulkan tensor belongs to another native context");
            const auto allocation = access.context->memory().cuda_block_info(storage);
            if (!allocation)
                throw std::invalid_argument("Vulkan tensor has no native buffer allocation");
            VkExportMetalBufferInfoEXT exported{VK_STRUCTURE_TYPE_EXPORT_METAL_BUFFER_INFO_EXT};
            exported.memory = allocation->memory;
            VkExportMetalObjectsInfoEXT info{VK_STRUCTURE_TYPE_EXPORT_METAL_OBJECTS_INFO_EXT};
            info.pNext = &exported;
            const auto export_objects = reinterpret_cast<PFN_vkExportMetalObjectsEXT>(vkGetDeviceProcAddr(access.context->device(), "vkExportMetalObjectsEXT"));
            export_objects(access.context->device(), &info);
            const id<MTLBuffer> buffer = exported.mtlBuffer;
            const auto offset = allocation->allocation_offset + storage.byte_offset;
            if (!buffer || buffer.device != device || offset > buffer.length || tensor.bytes() > buffer.length - offset)
                throw std::invalid_argument("Invalid native Vulkan-to-Metal tensor view");
            return {buffer, static_cast<NSUInteger>(offset), tensor.bytes()};
        }
    } // namespace
#endif
    struct MetalTensorReader::Impl {
        std::shared_ptr<internal::metal::Context> context;
        id<MTLCommandQueue> queue;
        id<MTLSharedEvent> producer, consumer;
        uint64_t serial = 0;
        std::shared_ptr<std::atomic_bool> failed = std::make_shared<std::atomic_bool>(false);
        std::mutex mutex;
#ifdef LFS_TENSOR_VULKAN
        std::shared_ptr<VulkanNativeAccess> vulkan;
#endif
    };

    MetalTensorReader::MetalTensorReader() : impl_(std::make_unique<Impl>()) {
        if (@available(macOS 26.0, *)) {
            if (!gpu_backend_available(GpuBackend::Metal))
                throw std::runtime_error("Resident Metal tensor access is unavailable");
            impl_->context = internal::metal::acquire_context();
            impl_->queue = [impl_->context->device() newCommandQueue];
            impl_->producer = [impl_->context->device() newSharedEvent];
            impl_->consumer = [impl_->context->device() newSharedEvent];
            if (!impl_->queue || !impl_->producer || !impl_->consumer)
                throw std::runtime_error("Could not create Metal tensor reader queue/events");
        } else {
            throw std::runtime_error("Resident Metal tensor access requires macOS 26");
        }
    }
    MetalTensorReader::~MetalTensorReader() = default;
    id<MTLDevice> MetalTensorReader::device() const { return impl_->queue.device; }

    id<MTLCommandBuffer> MetalTensorReader::submit(std::span<const Tensor* const> tensors, const Encode& encode) {
        return submitAccess(tensors, encode, false);
    }
    id<MTLCommandBuffer> MetalTensorReader::submitWrites(std::span<const Tensor* const> inputs,
                                                         std::span<Tensor* const> outputs, const EncodeWrite& encode) {
        if (!encode || outputs.empty())
            throw std::invalid_argument("Native Metal writes require an encoder and explicit outputs");
        std::vector<const Tensor*> tensors(inputs.begin(), inputs.end());
        for (auto* output : outputs) {
            if (!output || !output->is_valid())
                throw std::invalid_argument("Native Metal write output must be a valid tensor");
            tensors.push_back(output);
        }
        return submitAccess(tensors, [&](id<MTLCommandBuffer> command, std::span<const MetalTensorView> views) { encode(command, views.first(inputs.size()), views.subspan(inputs.size())); }, true);
    }
    id<MTLCommandBuffer> MetalTensorReader::submitAccess(std::span<const Tensor* const> tensors, const Encode& encode, bool writes) {
        if (@available(macOS 26.0, *)) {
            if (!encode)
                throw std::invalid_argument("Metal tensor reader needs an encoder");
            std::lock_guard lock(impl_->mutex);
            if (impl_->failed->load(std::memory_order_acquire))
                throw std::runtime_error("Metal tensor consumer is quarantined after a GPU command failure");
            auto owners = std::make_shared<std::vector<Tensor>>();
            owners->reserve(tensors.size());
            std::vector<MetalTensorView> views;
            views.reserve(tensors.size());
            std::vector<internal::StorageRef> uses;
            uses.reserve(tensors.size());
#ifdef LFS_TENSOR_VULKAN
            std::vector<internal::StorageRef> vulkan_uses;
#endif
            for (const auto* tensor : tensors) {
                if (!tensor || !tensor->is_valid() || tensor->numel() == 0) {
                    views.push_back({});
                    continue;
                }
                if (tensor->device() != Device::GPU || !tensor->is_contiguous())
                    throw std::invalid_argument("Native Metal tensor access requires contiguous storage");
                const auto storage = internal::storage_ref(*tensor);
                owners->push_back(*tensor);
#ifdef LFS_TENSOR_VULKAN
                if (gpu_backend_of(*tensor) == GpuBackend::Vulkan) {
                    if (!impl_->vulkan)
                        impl_->vulkan = vulkanAccess(device());
                    views.push_back(vulkanView(*tensor, *impl_->vulkan, device()));
                    vulkan_uses.push_back(internal::storage_ref(*tensor));
                    continue;
                }
#endif
                if (gpu_backend_of(*tensor) != GpuBackend::Metal)
                    throw std::invalid_argument("Native Metal tensor access requires Metal or Vulkan storage");
                const auto at = impl_->context->locate(storage);
                if (!at.buffer || at.buffer.device != device() || at.offset > at.buffer.length ||
                    tensor->bytes() > at.buffer.length - at.offset)
                    throw std::invalid_argument("Invalid resident Metal tensor view");
                views.push_back({at.buffer, at.offset, tensor->bytes()});
                uses.push_back(storage);
            }
            if (impl_->serial == std::numeric_limits<uint64_t>::max())
                throw std::overflow_error("Metal tensor reader timeline exhausted");
            auto command = [impl_->queue commandBuffer];
            if (!command)
                throw std::runtime_error("Could not allocate Metal tensor reader command");
            command.label = @"LichtFeld native tensor consumer";
            const uint64_t ready = impl_->context->signal(impl_->producer);
            if (ready)
                [command encodeWaitForEvent:impl_->producer value:ready];
#ifdef LFS_TENSOR_VULKAN
            const auto vulkan = vulkan_uses.empty() ? nullptr : impl_->vulkan;
            if (vulkan) {
                const auto ready = vulkan->context->recorders().flush_storages(vulkan_uses);
                if (ready)
                    [command encodeWaitForEvent:vulkan->producer value:ready];
            }
#endif
            encode(command, views);
            const uint64_t done = ++impl_->serial;
            [command encodeSignalEvent:impl_->consumer value:done];
#ifdef LFS_TENSOR_VULKAN
            if (vulkan)
                [command encodeSignalEvent:vulkan->done value:done];
#endif
            const auto consumer = impl_->consumer;
            const auto failed = impl_->failed;
            const auto context = impl_->context;
            [command addCompletedHandler:^(id<MTLCommandBuffer> completed) {
                (void)owners;
                if (completed.status == MTLCommandBufferStatusError) {
                    failed->store(true, std::memory_order_release);
                    // A failed read has stopped accessing its inputs. A failed
                    // write may have partial outputs: quarantine the tensor context
                    // before releasing the GPU wait. Neither is reported as success.
                    if (writes)
                        context->record_external_write_failure();
#ifdef LFS_TENSOR_VULKAN
                    if (writes && vulkan)
                        vulkan->context->mark_device_lost_once();
#endif
                    consumer.signaledValue = done;
#ifdef LFS_TENSOR_VULKAN
                    if (vulkan)
                        vulkan->done.signaledValue = done;
#endif
                }
            }];
            [command commit];
#ifdef LFS_TENSOR_VULKAN
            if (vulkan)
                (void)vulkan->context->recorders().wait_external(vulkan_uses, vulkan->consumer->semaphore, done, vulkan->consumer);
#endif
            // The tensor context waits on the GPU, and its ordinary host access and
            // memory-pool reuse see the same dependency through each storage's stamp.
            const uint64_t guard = impl_->context->queue_wait(impl_->consumer, done);
            for (const auto& storage : uses)
                if (storage.meta)
                    const_cast<StorageMeta*>(storage.meta)->pending_value.store(guard, std::memory_order_release);
            return command;
        }
        throw std::runtime_error("Resident Metal tensor access requires macOS 26");
    }
} // namespace lfs::core

/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#include "core/tensor_backend.hpp"
#include "core/tensor_metal_reader.hpp"
#include "metal_context.hpp"
#include <atomic>
#include <limits>
#include <mutex>
#include <stdexcept>
#include <vector>

namespace lfs::core {
    struct MetalTensorReader::Impl {
        std::shared_ptr<internal::metal::Context> context;
        id<MTLCommandQueue> queue;
        id<MTLSharedEvent> producer, consumer;
        uint64_t serial = 0;
        std::shared_ptr<std::atomic_bool> failed = std::make_shared<std::atomic_bool>(false);
        std::mutex mutex;
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
            for (const auto* tensor : tensors) {
                if (!tensor || !tensor->is_valid() || tensor->numel() == 0) {
                    views.push_back({});
                    continue;
                }
                if (gpu_backend_of(*tensor) != GpuBackend::Metal || !tensor->is_contiguous())
                    throw std::invalid_argument("Metal tensor reader requires contiguous Metal storage");
                owners->push_back(*tensor);
                const auto storage = internal::storage_ref(*tensor);
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
            encode(command, views);
            const uint64_t done = ++impl_->serial;
            [command encodeSignalEvent:impl_->consumer value:done];
            const auto consumer = impl_->consumer;
            const auto failed = impl_->failed;
            [command addCompletedHandler:^(id<MTLCommandBuffer> completed) {
                (void)owners;
                if (completed.status == MTLCommandBufferStatusError) {
                    failed->store(true, std::memory_order_release);
                    // This consumer only reads tensors. A failed command has stopped
                    // accessing them, so later producer mutations can safely resume.
                    // Keep failure sticky; releasing the wait is not render success.
                    consumer.signaledValue = done;
                }
            }];
            [command commit];
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

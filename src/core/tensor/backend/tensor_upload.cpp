/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#include "core/tensor_upload.hpp"
#include "../internal/tensor_impl.hpp"
#include "backend_driver.hpp"
#include "backend_work_queue.hpp"
#include "core/logger.hpp"
#include "core/tensor_backend.hpp"
#include "core/tensor_readback.hpp"
#include "gpu_backend_ops.hpp"
#include <algorithm>
#include <cstring>
#include <stdexcept>
#include <utility>
#include <vector>

namespace lfs::core {
    struct TensorUpload::Impl {
        Tensor source, destination;
        internal::GpuBackendOps* ops = nullptr;
        std::unique_ptr<internal::BackendUploadBuffer> upload;
        bool pending = false;

        void prepare(Tensor&& target) {
            auto& backend = internal::backend_ops_for(target);
            if (ops != &backend) {
                auto replacement = backend.create_upload_buffer();
                if (!replacement)
                    throw std::runtime_error("Tensor backend does not support uploads");
                upload = std::move(replacement);
                ops = &backend;
            }
            destination = std::move(target);
            source = {};
            pending = false;
        }
        void finish(internal::BackendUploadSubmission status) {
            if (status.source_consumed)
                source = {};
            if (status.complete) {
                pending = false;
                if (!status.source_consumed)
                    source = {};
                destination = {};
            }
        }
        void failed_enqueue() {
            upload->settle_enqueue_failure();
            pending = false;
            source = {};
            destination = {};
            upload.reset();
            ops = nullptr;
        }
        void enqueue(Tensor&& target, const Tensor& input, internal::BackendQueueTarget queue) {
            prepare(std::move(target));
            source = input;
            pin_operands({&source, &destination});
            pending = true;
            try {
                finish(upload->enqueue(destination, source, queue, false));
            } catch (...) {
                failed_enqueue();
                throw;
            }
        }
        void enqueue_bytes(Tensor&& target, std::span<const std::byte> input, internal::BackendQueueTarget queue, bool in_batch) {
            prepare(std::move(target));
            pin_operands({&destination});
            pending = true;
            try {
                finish(upload->enqueue_bytes(destination, input, queue, in_batch));
            } catch (...) {
                failed_enqueue();
                throw;
            }
        }
    };
    TensorUpload::TensorUpload() = default;
    TensorUpload::~TensorUpload() {
        try {
            wait();
        } catch (const std::exception& error) {
            if (impl_ && impl_->upload && impl_->upload->abandon_after_failure()) {
                LOG_ERROR("Tensor upload failed; cleanup delegated to native ownership: {}", error.what());
            } else {
                LOG_ERROR("Tensor upload storage retained after incomplete transfer: {}", error.what());
                (void)impl_.release();
            }
        }
    }
    TensorUpload::TensorUpload(TensorUpload&&) noexcept = default;
    TensorUpload& TensorUpload::operator=(TensorUpload&& other) noexcept {
        if (this != &other) {
            TensorUpload previous;
            previous.impl_ = std::move(impl_);
            impl_ = std::move(other.impl_);
        }
        return *this;
    }
    bool TensorUpload::pending() const noexcept { return impl_ && impl_->pending; }
    namespace {
        void validate_upload(const Tensor& destination, const Tensor& source) {
            if (!gpu_backend_of(destination) || source.device() != Device::CPU || !source.is_contiguous() ||
                !destination.is_contiguous() || destination.dtype() != source.dtype() || destination.bytes() != source.bytes())
                throw std::invalid_argument("TensorUpload requires matching contiguous CPU and GPU tensors");
        }
        void validate_upload(const Tensor& destination, std::span<const std::byte> source) {
            if (!gpu_backend_of(destination) || !destination.is_contiguous() || destination.bytes() != source.size())
                throw std::invalid_argument("TensorUpload byte span must match a contiguous destination");
        }
    } // namespace
    void TensorUpload::enqueue(Tensor destination, const Tensor& source, TensorExecutionTarget target) {
        if (gpu_backend_of(destination) != target.backend())
            throw std::invalid_argument("TensorUpload queue backend mismatch");
        enqueue(std::move(destination), source, target.native_handle());
    }
    void TensorUpload::enqueue(Tensor destination, std::span<const std::byte> source, TensorExecutionTarget target) {
        if (gpu_backend_of(destination) != target.backend())
            throw std::invalid_argument("TensorUpload queue backend mismatch");
        enqueue(std::move(destination), source, target.native_handle());
    }
    void TensorUpload::enqueue_in_batch(Tensor destination, std::span<const std::byte> source) {
        if (pending())
            throw std::logic_error("TensorUpload already pending");
        validate_upload(destination, source);
        if (!impl_)
            impl_ = std::make_unique<Impl>();
        impl_->enqueue_bytes(std::move(destination), source, {nullptr, false}, true);
    }
    void TensorUpload::enqueue(Tensor destination, const Tensor& source) {
        if (pending())
            throw std::logic_error("TensorUpload already pending");
        validate_upload(destination, source);
        if (!impl_)
            impl_ = std::make_unique<Impl>();
        impl_->enqueue(std::move(destination), source, {});
    }
    void TensorUpload::enqueue(Tensor destination, const Tensor& source, void* target) {
        if (pending())
            throw std::logic_error("TensorUpload already pending");
        validate_upload(destination, source);
        if (!impl_)
            impl_ = std::make_unique<Impl>();
        impl_->enqueue(std::move(destination), source, {target, false});
    }
    void TensorUpload::enqueue(Tensor destination, std::span<const std::byte> source, void* target) {
        if (pending())
            throw std::logic_error("TensorUpload already pending");
        validate_upload(destination, source);
        if (!impl_)
            impl_ = std::make_unique<Impl>();
        impl_->enqueue_bytes(std::move(destination), source, {target, false}, false);
    }
    bool TensorUpload::poll() {
        if (!pending())
            return true;
        if (!impl_->upload->poll())
            return false;
        impl_->pending = false;
        impl_->source = {};
        impl_->destination = {};
        return true;
    }
    void TensorUpload::wait() {
        if (!pending())
            return;
        impl_->upload->wait();
        (void)poll();
    }

    struct TensorFence::Impl {
        GpuBackend backend;
        std::unique_ptr<internal::BackendFence> fence;
    };
    TensorFence::TensorFence(GpuBackend backend) : impl_(std::make_unique<Impl>(Impl{backend, {}})) {
        impl_->fence = internal::backend_ops(backend).create_fence();
        if (!impl_->fence)
            throw std::runtime_error("Tensor backend does not support reusable fences");
    }
    TensorFence::TensorFence(GpuBackend backend, void* imported) : impl_(std::make_unique<Impl>(Impl{backend, {}})) {
        impl_->fence = internal::backend_ops(backend).create_fence(imported);
        if (!impl_->fence)
            throw std::invalid_argument("Tensor backend cannot import this fence handle");
    }
    TensorFence::~TensorFence() = default;
    TensorFence::TensorFence(TensorFence&&) noexcept = default;
    TensorFence& TensorFence::operator=(TensorFence&&) noexcept = default;
    TensorFence TensorFence::adopt(GpuBackend backend, void* event) { return TensorFence(backend, event); }
    void TensorFence::record(TensorExecutionTarget target) {
        if (target.backend() != impl_->backend)
            throw std::invalid_argument("TensorFence queue backend mismatch");
        record(target.native_handle());
    }
    void TensorFence::wait_on(TensorExecutionTarget target) const {
        if (target.backend() != impl_->backend)
            throw std::invalid_argument("TensorFence queue backend mismatch");
        wait_on(target.native_handle());
    }
    void TensorFence::record(void* target) { impl_->fence->record(internal::ExecContext{static_cast<cudaStream_t>(target)}); }
    void TensorFence::wait_on(void* target) const { impl_->fence->wait_on(internal::ExecContext{static_cast<cudaStream_t>(target)}); }
    void TensorFence::wait() const { impl_->fence->wait(); }
    bool TensorFence::ready() const { return impl_->fence->ready(); }
    struct TensorWorkQueue::Impl {
        GpuBackend backend;
        std::unique_ptr<internal::BackendWorkQueue> native;
        void* target = nullptr;
        Impl(GpuBackend selected, const internal::BackendQueueCreateInfo& info)
            : backend(selected), native(internal::backend_driver(selected).make_work_queue(info)) {
            if (!native)
                throw std::runtime_error("Tensor backend does not support work queues");
            // Queue identity is immutable; cache it once rather than adding a
            // second virtual call to every fence, target or readback operation.
            target = native->native_handle();
        }
    };
    TensorWorkQueue::TensorWorkQueue(GpuBackend backend, Mode mode)
        : impl_(std::make_unique<Impl>(backend, internal::BackendQueueCreateInfo{.legacy_ordered = mode == Mode::LegacyOrdered})) {}
    TensorWorkQueue::TensorWorkQueue(GpuBackend backend, void* target)
        : impl_(std::make_unique<Impl>(backend, internal::BackendQueueCreateInfo{.kind = internal::BackendQueueCreateInfo::Kind::Borrowed, .target = target})) {}
    TensorWorkQueue::TensorWorkQueue(GpuBackend backend, void* device, void* consumer)
        : TensorWorkQueue(backend, device, consumer, true) {}
    TensorWorkQueue::TensorWorkQueue(GpuBackend backend, void* device, void* consumer, bool direct)
        : impl_(std::make_unique<Impl>(backend, internal::BackendQueueCreateInfo{
                                                    .kind = internal::BackendQueueCreateInfo::Kind::Consumer,
                                                    .consumer_device = device,
                                                    .consumer = consumer,
                                                    .external_interop = direct})) {}
    TensorWorkQueue::Scope::Scope(const TensorWorkQueue& queue)
        : backend_scope_(queue.backend()), native_(queue.impl_->native.get()), binding_(static_cast<internal::BackendWorkQueue*>(native_)->bind(false)) {}
    TensorWorkQueue::Scope::~Scope() { static_cast<internal::BackendWorkQueue*>(native_)->unbind(binding_); }
    void* TensorWorkQueue::native_handle() const { return impl_->target; }
    GpuBackend TensorWorkQueue::backend() const { return impl_->backend; }
    TensorExecutionTarget::TensorExecutionTarget(const TensorWorkQueue& queue)
        : backend_(queue.backend()), target_(queue.native_handle()) {}
    bool TensorWorkQueue::ready() const { return impl_->native->ready(); }
    void TensorWorkQueue::wait() const { impl_->native->wait(); }
    void TensorWorkQueue::record(TensorFence& fence) const { fence.record(TensorExecutionTarget(*this)); }
    void TensorWorkQueue::wait_for(const TensorFence& fence) const { fence.wait_on(TensorExecutionTarget(*this)); }
    void TensorWorkQueue::enqueue_host_callback(void (*callback)(void*), void* user) { impl_->native->enqueue_host_callback(callback, user); }
    void TensorWorkQueue::set_consumer_timeline(void* device, VulkanTimelinePoint point) { impl_->native->set_consumer_timeline(device, std::move(point)); }
    void TensorWorkQueue::wait_timeline(uint64_t value) { impl_->native->wait_timeline(value); }
    void* TensorWorkQueue::timeline() const { return impl_->native->timeline(); }
    TensorCompletion TensorWorkQueue::execute(const std::function<void()>& commands, uint64_t value) { return impl_->native->execute(commands, value); }
    TensorWorkQueue::~TensorWorkQueue() {
        try {
            if (!impl_->native->close())
                (void)impl_.release();
        } catch (const std::exception& error) {
            LOG_ERROR("Tensor queue retained after incomplete work: {}", error.what());
            (void)impl_.release();
        }
    }
    namespace {
        void require_readback_staging(const Tensor& staging, const GpuBackend backend, const size_t bytes) {
            if (gpu_backend_of(staging) != backend || !staging.is_contiguous() ||
                staging.dtype() != DataType::UInt8 || staging.bytes() < bytes)
                throw std::invalid_argument("Readback staging must cover a slot on the queue backend");
        }
    } // namespace
    struct TensorReadbackRing::Impl {
        struct Slot {
            std::unique_ptr<internal::BackendReadbackSlot> native;
            std::unique_ptr<TensorFence> fence;
            std::vector<Tensor> sources;
            bool sealed = false;
        };
        GpuBackend backend;
        TensorWorkQueue* queue = nullptr;
        internal::GpuBackendOps* ops = nullptr;
        size_t bytes = 0;
        std::vector<Slot> slots;
        const Tensor* staging = nullptr;
        Tensor scratch;
        ~Impl() {
            if (!queue)
                return;
            try {
                queue->wait();
            } catch (const std::exception& error) { LOG_WARN("Readback ring drain failed: {}", error.what()); }
        }
    };
    TensorReadbackRing::TensorReadbackRing(GpuBackend backend, size_t slots, size_t bytes, TensorWorkQueue& queue, const Tensor* staging)
        : impl_(std::make_unique<Impl>()) {
        if (queue.backend() != backend)
            throw std::invalid_argument("Readback ring queue backend mismatch");
        queue.impl_->native->validate_readback_ring();
        if (!slots || !bytes)
            throw std::invalid_argument("Readback ring dimensions must be nonzero");
        auto& s = *impl_;
        s.backend = backend;
        s.queue = &queue;
        s.ops = &internal::backend_ops(backend);
        s.bytes = bytes;
        s.staging = staging;
        if (staging && staging->is_valid())
            require_readback_staging(*staging, backend, bytes);
        s.slots.resize(slots);
        for (auto& slot : s.slots) {
            slot.native = s.ops->create_readback_slot(bytes);
            if (!slot.native)
                throw std::runtime_error("Tensor backend does not support packed readback");
            slot.fence = std::make_unique<TensorFence>(backend);
        }
    }
    TensorReadbackRing::~TensorReadbackRing() = default;
    bool TensorReadbackRing::prefers_recycled_host_staging() const noexcept {
        return impl_->slots.front().native->prefers_recycled_host_staging();
    }
    void TensorReadbackRing::enqueue(const Tensor& source, size_t offset, size_t bytes,
                                     size_t index, size_t destination, bool stage) {
        auto& s = *impl_;
        auto& slot = s.slots.at(index);
        if (slot.sealed)
            throw std::logic_error("Readback slot must be released before reuse");
        if (!source.is_valid() || gpu_backend_of(source) != s.backend)
            throw std::invalid_argument("Readback ring requires a tensor on its queue backend");
        if (offset > source.bytes() || bytes > source.bytes() - offset ||
            destination > s.bytes || bytes > s.bytes - destination)
            throw std::out_of_range("Readback ring byte range exceeds source or slot");
        if (!bytes)
            return;
        TensorWorkQueue::Scope scope(*s.queue);
        slot.sources.push_back(source.contiguous());
        const auto& retained = slot.sources.back();
        pin_operands({&retained});
        const internal::ExecContext execution{static_cast<cudaStream_t>(s.queue->native_handle())};
        if (slot.native->begin_append(retained, offset, bytes, destination, execution))
            return;
        auto input = internal::storage_ref(retained);
        input.byte_offset += offset;
        if (stage) {
            const bool borrowed = s.staging && s.staging->is_valid();
            if (borrowed) {
                require_readback_staging(*s.staging, s.backend, s.bytes);
                slot.sources.push_back(*s.staging);
            } else if (!s.scratch.is_valid()) {
                s.scratch = Tensor::empty({s.bytes}, Device::GPU, DataType::UInt8);
            }
            Tensor& scratch = borrowed ? slot.sources.back() : s.scratch;
            const auto output = internal::storage_ref(scratch);
            s.ops->copy_device_to_device({.src = input, .dst = output, .bytes = bytes, .context = execution});
            input = output;
        }
        slot.native->append(input, bytes, destination, execution);
    }
    const TensorFence& TensorReadbackRing::seal(size_t index) {
        auto& slot = impl_->slots.at(index);
        if (slot.sealed)
            throw std::logic_error("Readback slot is already sealed");
        impl_->queue->record(*slot.fence);
        slot.sealed = true;
        return *slot.fence;
    }
    bool TensorReadbackRing::poll(size_t index) const {
        const auto& slot = impl_->slots.at(index);
        if (!slot.sealed)
            throw std::logic_error("Readback slot must be sealed before polling");
        if (!slot.fence->ready())
            return false;
        return slot.native->ready();
    }
    void TensorReadbackRing::wait(size_t index) const {
        const auto& slot = impl_->slots.at(index);
        if (!slot.sealed)
            throw std::logic_error("Readback slot must be sealed before waiting");
        slot.fence->wait();
        slot.native->wait();
    }
    void TensorReadbackRing::release(size_t index) {
        auto& slot = impl_->slots.at(index);
        if ((slot.sealed && !slot.fence->ready()) ||
            (!slot.sealed && !slot.sources.empty() && !impl_->queue->ready()))
            throw std::logic_error("Cannot release a pending readback slot");
        slot.native->release();
        slot.sources.clear();
        slot.sealed = false;
    }
    std::span<std::byte> TensorReadbackRing::slot_bytes(size_t index) {
        auto& slot = impl_->slots.at(index);
        if (!slot.native->ready())
            throw std::logic_error("Packed readback bytes are not complete");
        return {static_cast<std::byte*>(slot.native->data()), impl_->bytes};
    }
} // namespace lfs::core

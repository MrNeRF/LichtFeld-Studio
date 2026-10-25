/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#include "runtime_context.hpp"
#include <cstdio>
#include <utility>

namespace lfs::core::internal {
    namespace {
        struct ContextOwner {
            std::shared_ptr<HipRuntimeModule> module;
            LfsHipContext* context = nullptr;
            std::unique_ptr<ContextOwner> next;
            std::unique_ptr<HipRetiredResource> resources;
        };
        int drain_resources(ContextOwner& owner, LfsHipError& error) noexcept {
            while (owner.resources) {
                auto& node = *owner.resources;
                const auto& api = owner.module->api();
                const int status = node.kind == HipRetiredResource::Kind::Readback
                                       ? api.readback_destroy(owner.context, static_cast<LfsHipReadback*>(node.handle), &error)
                                       : api.timing_destroy(owner.context, static_cast<LfsHipTiming*>(node.handle), &error);
                if (status)
                    return status;
                auto released = std::move(owner.resources);
                owner.resources = std::move(released->next);
            }
            return 0;
        }
        struct RetirementQueue {
            std::mutex mutex;
            std::unique_ptr<ContextOwner> head;
        };
        RetirementQueue& retirement_queue() {
            // Intentionally process-lifetime: a driver failure must never unload
            // the module beneath an undrained context during static teardown.
            static auto* queue = new RetirementQueue();
            return *queue;
        }
    } // namespace

    HipRuntimeError::HipRuntimeError(int code, const char* operation, const LfsHipError& error)
        : std::runtime_error(std::string("HIP Tensor ") + operation + ": " + error.message), native_code_(code) {}

    void hip_runtime_check(int status, const LfsHipError& error, const char* operation) {
        if (status)
            throw HipRuntimeError(error.native_code, operation, error);
    }

    struct HipContextState {
        std::mutex mutex;
        std::unique_ptr<ContextOwner> owner = std::make_unique<ContextOwner>();
        size_t live_buffers = 0;

        ~HipContextState() {
            if (!owner->context)
                return;
            LfsHipError error{};
            if (!drain_resources(*owner, error) && !owner->module->api().context_destroy(owner->context, &error))
                return;
            std::fprintf(stderr, "[error] HIP Tensor context cleanup deferred: %s (%d)\n", error.message, error.native_code);
            auto& queue = retirement_queue();
            std::lock_guard lock(queue.mutex);
            owner->next = std::move(queue.head);
            queue.head = std::move(owner);
        }
    };

    struct HipBufferState {
        std::shared_ptr<HipContextState> context;
        LfsHipBuffer* handle = nullptr;
        size_t size = 0;
        ~HipBufferState() {
            if (!handle)
                return;
            std::lock_guard lock(context->mutex);
            LfsHipError error{};
            if (context->owner->module->api().buffer_release(context->owner->context, handle, &error))
                std::fprintf(stderr, "[error] HIP Tensor buffer cleanup retained by context: %s (%d)\n", error.message, error.native_code);
            --context->live_buffers;
        }
    };

    HipRuntimeContext::Lock::Lock(std::shared_ptr<HipContextState> state)
        : state_(std::move(state)), lock_(state_->mutex) {
        if (!state_->owner->context)
            throw std::logic_error("HIP Tensor context is closed");
    }
    const HipRuntimeApi& HipRuntimeContext::Lock::api() const noexcept { return state_->owner->module->api(); }
    LfsHipContext* HipRuntimeContext::Lock::handle() const noexcept { return state_->owner->context; }
    LfsHipView HipRuntimeContext::Lock::view(const HipRuntimeBuffer& buffer, size_t offset, size_t bytes) const {
        if (!buffer.state_ || buffer.state_->context != state_ || !buffer.state_->handle)
            throw std::invalid_argument("HIP Tensor buffer is foreign or closed");
        if (offset > buffer.state_->size || bytes > buffer.state_->size - offset)
            throw std::out_of_range("HIP Tensor buffer view exceeds allocation");
        return {buffer.state_->handle, offset};
    }

    HipRuntimeContext::HipRuntimeContext(std::shared_ptr<HipRuntimeModule> module, int ordinal)
        : state_(std::make_shared<HipContextState>()) {
        if (!module || ordinal < 0)
            throw std::invalid_argument("HIP Tensor context requires a module and nonnegative device");
        // Prepare the retirement queue before acquiring native resources.
        (void)retirement_queue();
        state_->owner->module = std::move(module);
        LfsHipError error{};
        hip_runtime_check(state_->owner->module->api().context_create(ordinal, &state_->owner->context, &error), error, "context_create");
    }

    HipRuntimeBuffer HipRuntimeContext::allocate(size_t bytes) {
        if (!bytes)
            throw std::invalid_argument("HIP Tensor allocation requires nonzero bytes");
        auto allocation = std::make_shared<HipBufferState>();
        allocation->context = state_;
        allocation->size = bytes;
        auto queue = lock();
        LfsHipError error{};
        hip_runtime_check(queue.api().buffer_allocate(queue.handle(), bytes, &allocation->handle, &error), error, "buffer_allocate");
        ++state_->live_buffers;
        return HipRuntimeBuffer(std::move(allocation));
    }
    HipRuntimeContext::Lock HipRuntimeContext::lock() const { return Lock(state_); }
    void HipRuntimeContext::synchronize() const {
        auto queue = lock();
        LfsHipError error{};
        hip_runtime_check(queue.api().context_synchronize(queue.handle(), &error), error, "context_synchronize");
    }
    void HipRuntimeContext::close() {
        std::lock_guard lock(state_->mutex);
        if (!state_->owner->context)
            return;
        if (state_->live_buffers)
            throw std::logic_error("HIP Tensor context still has live buffer leases");
        LfsHipError error{};
        hip_runtime_check(drain_resources(*state_->owner, error), error, "retired_resource_release");
        hip_runtime_check(state_->owner->module->api().context_destroy(state_->owner->context, &error), error, "context_destroy");
        state_->owner->context = nullptr;
    }

    void HipRuntimeContext::release_readback(LfsHipReadback* handle, std::unique_ptr<HipRetiredResource> node) noexcept {
        std::lock_guard lock(state_->mutex);
        node->kind = HipRetiredResource::Kind::Readback;
        node->handle = handle;
        node->next = std::move(state_->owner->resources);
        state_->owner->resources = std::move(node);
        LfsHipError error{};
        if (drain_resources(*state_->owner, error))
            std::fprintf(stderr, "[error] HIP Tensor readback cleanup deferred: %s (%d)\n", error.message, error.native_code);
    }
    void HipRuntimeContext::release_timing(LfsHipTiming* handle, std::unique_ptr<HipRetiredResource> node) noexcept {
        std::lock_guard lock(state_->mutex);
        node->kind = HipRetiredResource::Kind::Timing;
        node->handle = handle;
        node->next = std::move(state_->owner->resources);
        state_->owner->resources = std::move(node);
        LfsHipError error{};
        if (drain_resources(*state_->owner, error))
            std::fprintf(stderr, "[error] HIP Tensor timing cleanup deferred: %s (%d)\n", error.message, error.native_code);
    }

    size_t HipRuntimeBuffer::bytes() const noexcept { return state_ ? state_->size : 0; }
    void HipRuntimeBuffer::upload(size_t offset, const void* source, size_t bytes) const {
        if (!state_)
            throw std::logic_error("HIP Tensor buffer is empty");
        HipRuntimeContext::Lock queue(state_->context);
        const auto view = queue.view(*this, offset, bytes);
        LfsHipError error{};
        hip_runtime_check(queue.api().upload(queue.handle(), view.buffer, offset, source, bytes, &error), error, "upload");
    }
    void HipRuntimeBuffer::download(size_t offset, void* destination, size_t bytes) const {
        if (!state_)
            throw std::logic_error("HIP Tensor buffer is empty");
        HipRuntimeContext::Lock queue(state_->context);
        const auto view = queue.view(*this, offset, bytes);
        LfsHipError error{};
        hip_runtime_check(queue.api().download(queue.handle(), view.buffer, offset, destination, bytes, &error), error, "download");
    }
    void HipRuntimeBuffer::close() {
        if (!state_)
            return;
        std::lock_guard lock(state_->context->mutex);
        if (!state_->handle)
            return;
        LfsHipError error{};
        hip_runtime_check(state_->context->owner->module->api().buffer_release(state_->context->owner->context, state_->handle, &error), error, "buffer_release");
        state_->handle = nullptr;
        --state_->context->live_buffers;
    }
    size_t HipRuntimeContext::retry_retired_contexts() {
        auto& queue = retirement_queue();
        std::lock_guard lock(queue.mutex);
        size_t retained = 0;
        auto* node = &queue.head;
        while (*node) {
            LfsHipError error{};
            auto& owner = **node;
            if (!drain_resources(owner, error) && !owner.module->api().context_destroy(owner.context, &error)) {
                auto released = std::move(*node);
                *node = std::move(released->next);
            } else {
                ++retained;
                node = &owner.next;
            }
        }
        return retained;
    }
} // namespace lfs::core::internal

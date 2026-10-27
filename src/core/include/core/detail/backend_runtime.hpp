/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#pragma once

#include "core/tensor_completion.hpp"
#include "descriptors.hpp"
#include <cstddef>
#include <memory>
#include <optional>
#include <span>
#include <stdexcept>
#include <type_traits>

namespace lfs::core {
    class Tensor;
}

namespace lfs::core::internal {
    struct BackendQueueTarget {
        void* handle = nullptr;
        bool current = true;
    };
    struct BackendUploadSubmission {
        bool complete = false;
        bool source_consumed = false;
    };
    // Compile out optional native ownership, including its destructors, in
    // builds that cannot construct the backend. get() is used only in compiled
    // backend branches; an accidental access fails instead of inventing state.
    template <GpuBackend Backend, class State>
    class CompiledBackendState {
        struct Empty {};
        [[no_unique_address]] std::conditional_t<gpu_backend_compiled(Backend), State, Empty> state_;

    public:
        State& get() {
            if constexpr (gpu_backend_compiled(Backend))
                return state_;
            else
                throw std::logic_error("State requested for an uncompiled GPU backend");
        }
        const State& get() const {
            if constexpr (gpu_backend_compiled(Backend))
                return state_;
            else
                throw std::logic_error("State requested for an uncompiled GPU backend");
        }
    };

    // Optional native ownership primitives. Implementations and SDK types stay
    // in backend directories; shared Tensor code sees only these contracts.
    struct BackendQueueLease {
        std::shared_ptr<void> keep_alive;
        uint64_t native_queue = 0;
        int32_t device = -1;
    };
    struct BackendCompletionTarget {
        void* external_device = nullptr;
        VulkanTimelinePoint external_point;
    };
    class BackendCompletion {
    public:
        virtual ~BackendCompletion() = default;
        virtual void capture(std::span<const Tensor* const>) = 0;
        virtual void record(ExecContext, uint64_t native_value) = 0;
        virtual VulkanTimelinePoint timeline() const = 0;
        virtual bool ready() = 0;
        virtual void wait() = 0;
    };
    class BackendFence {
    public:
        virtual ~BackendFence() = default;
        virtual void record(ExecContext) = 0;
        virtual void wait_on(ExecContext) const = 0;
        virtual void wait() const = 0;
        virtual bool ready() const = 0;
    };
    class BackendReadbackSlot {
    public:
        virtual ~BackendReadbackSlot() = default;
        virtual void* data() const = 0;
        // Orders the source and may finish a small direct host copy. False
        // means append() must encode the (possibly staged) GPU copy.
        virtual bool begin_append(const Tensor&, size_t source_offset, size_t bytes, size_t destination_offset, ExecContext) = 0;
        virtual void append(StorageRef, size_t bytes, size_t destination_offset, ExecContext) = 0;
        virtual bool prefers_recycled_host_staging() const noexcept = 0;
        virtual bool ready() = 0;
        virtual void wait() = 0;
        virtual void release() = 0;
    };
    class BackendUploadBuffer {
    public:
        virtual ~BackendUploadBuffer() = default;
        virtual BackendUploadSubmission enqueue(Tensor& destination, const Tensor& source, BackendQueueTarget, bool in_batch) = 0;
        virtual BackendUploadSubmission enqueue_bytes(Tensor& destination, std::span<const std::byte> source, BackendQueueTarget, bool in_batch) = 0;
        virtual void wait() = 0;
        virtual bool poll() = 0;
        virtual void settle_enqueue_failure() = 0;
        // True only when native ownership safely retires failed resources.
        virtual bool abandon_after_failure() noexcept = 0;
    };
    class BackendTimestamps {
    public:
        virtual ~BackendTimestamps() = default;
        virtual bool mark(size_t, void* queue) = 0;
        virtual bool wait_event(size_t) = 0;
        virtual bool wait_queue(void* queue) = 0;
        virtual std::optional<float> milliseconds(size_t, size_t) const = 0;
    };
} // namespace lfs::core::internal

/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#pragma once

#include "descriptors.hpp"
#include <memory>
#include <optional>
#include <stdexcept>
#include <type_traits>

namespace lfs::core::internal {
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
        virtual void append(StorageRef, size_t bytes, size_t destination_offset) = 0;
        virtual bool ready() = 0;
        virtual void wait() = 0;
        virtual void release() = 0;
    };
    class BackendUploadBuffer {
    public:
        virtual ~BackendUploadBuffer() = default;
        virtual void enqueue(StorageRef destination, const void* source, size_t bytes, ExecContext) = 0;
        virtual void wait() = 0;
        virtual bool poll() = 0;
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

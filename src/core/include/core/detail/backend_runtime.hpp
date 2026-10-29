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
    // Backend-owned execution primitives. Implementations and SDK types stay
    // in backend directories; shared Tensor code sees only these contracts.
    struct BackendCompletionTarget {
        void* external_device = nullptr;
        VulkanTimelinePoint external_point;
    };
    class BackendCompletion {
    public:
        virtual ~BackendCompletion() = default;
        virtual void capture(std::span<const Tensor* const>) = 0;
        virtual void record(ExecContext, uint64_t native_value) = 0;
        virtual VulkanTimelinePoint timeline() const { return {}; }
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
    };
    class BackendUploadBuffer {
    public:
        virtual ~BackendUploadBuffer() = default;
        virtual BackendUploadSubmission enqueue(Tensor& destination, const Tensor& source, BackendQueueTarget, bool in_batch) = 0;
        virtual BackendUploadSubmission enqueue_bytes(Tensor& destination, std::span<const std::byte> source, BackendQueueTarget, bool in_batch) = 0;
        virtual void wait() = 0;
        virtual bool poll() = 0;
        virtual void settle_enqueue_failure() = 0;
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

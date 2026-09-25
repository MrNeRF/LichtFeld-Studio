/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#pragma once

#include "core/export.hpp"
#include "core/gpu_backend_fwd.hpp"

#include <cstddef>
#include <memory>
#include <span>

namespace lfs::core {
    class Tensor;
    class TensorWorkQueue;

    // One reusable asynchronous download slot. The source and staging memory
    // remain owned until completion, including when the slot is destroyed.
    // Calls on a slot must be externally serialized. CPU inputs complete inline.
    class LFS_CORE_API TensorReadback {
    public:
        TensorReadback();
        ~TensorReadback();
        TensorReadback(TensorReadback&&) noexcept;
        TensorReadback& operator=(TensorReadback&&) noexcept;
        TensorReadback(const TensorReadback&) = delete;
        TensorReadback& operator=(const TensorReadback&) = delete;

        // Accepts strided tensors and follows the source's storage backend.
        // Complete the previous download before reusing this slot.
        void enqueue(const Tensor& source);
        void enqueue(const Tensor& source, TensorWorkQueue& queue);
        // Byte range is measured in the source's contiguous logical layout.
        void enqueue_range(const Tensor& source, std::size_t byte_offset,
                           std::size_t byte_count);
        // Bind contiguous CUDA input and pinned CPU output once for recurring
        // downloads. Both tensors are retained; their storage must not be replaced
        // while bound; do not mutate or create lazy consumers of the destination
        // until unbound. Read its bytes only after poll/wait completes. prepare allocates the reusable ordering/completion markers.
        // Vulkan direct destinations are explicitly unsupported.
        void prepare(const Tensor& source, const Tensor& destination);
        void enqueue(TensorWorkQueue& queue);
        [[nodiscard]] bool poll();
        void wait();
        [[nodiscard]] bool pending() const noexcept;

        // Copies all bytes and consumes the pending download when ready.
        // The destination must have exactly source.bytes() bytes. poll leaves it
        // untouched when not ready. Both functions require a pending download.
        [[nodiscard]] bool poll(std::span<std::byte> destination);
        void wait(std::span<std::byte> destination);

    private:
        void enqueue_range_on(const Tensor& source, std::size_t offset, std::size_t bytes, void* target);
        struct Impl;
        std::unique_ptr<Impl> impl_;
    };
} // namespace lfs::core

namespace lfs::core {
    class TensorWorkQueue;
    class TensorFence;

    // Packed asynchronous downloads into reusable pinned slots. Calls on a slot
    // must be serialized; distinct slots may be drained concurrently. Sources
    // remain retained until release(). Slot bytes are readable after wait/poll.
    // CUDA implemented; Vulkan reports unsupported before allocating resources.
    class LFS_CORE_API TensorReadbackRing {
    public:
        // Optional staging is retained and reused for device staging; it must be
        // contiguous UInt8 storage on this backend, at least slot_bytes long.
        // Serialize any other staging reads/writes on the same queue.
        TensorReadbackRing(GpuBackend backend, std::size_t slots,
                           std::size_t slot_bytes, TensorWorkQueue& queue,
                           const Tensor* device_staging = nullptr);
        ~TensorReadbackRing();
        TensorReadbackRing(const TensorReadbackRing&) = delete;
        TensorReadbackRing& operator=(const TensorReadbackRing&) = delete;
        void enqueue(const Tensor& source, std::size_t source_byte,
                     std::size_t bytes, std::size_t slot, std::size_t slot_byte,
                     bool stage_on_device = false);
        const TensorFence& seal(std::size_t slot);
        [[nodiscard]] bool poll(std::size_t slot) const;
        void wait(std::size_t slot) const;
        void release(std::size_t slot);
        [[nodiscard]] std::span<std::byte> slot_bytes(std::size_t slot);

    private:
        struct Impl;
        std::unique_ptr<Impl> impl_;
    };
} // namespace lfs::core

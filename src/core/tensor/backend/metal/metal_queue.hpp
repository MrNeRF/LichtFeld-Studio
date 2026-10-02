/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#pragma once
#include "core/tensor/internal/private_access.hpp"

#include "core/detail/descriptors.hpp"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>

// Metal runs all tensor work on one in-order queue: every batch starts after
// the batches submitted before it. A Metal tensor work queue is therefore an
// identity on that timeline, a fence is a batch serial, and a wait between two
// queues already holds. Builds without Metal get stubs that reject creation.
namespace lfs::core::internal::metal_queue {

    // Id 0 is the default queue and is never created.
    [[nodiscard]] uint64_t create();
    void destroy(uint64_t id) noexcept;
    [[nodiscard]] bool valid(uint64_t id);

    // Commits the open batch and returns the newest submitted serial.
    uint64_t submit();
    // Commits the batch of serial if it is still open; never blocks.
    [[nodiscard]] bool ready(uint64_t serial);
    // Raises batch failures and device faults.
    void wait(uint64_t serial);
    // Leaves failures and faults to the thread that owns the work.
    void wait_completed(uint64_t serial);

    // CPU view of shared Metal storage; callers order access themselves.
    [[nodiscard]] std::byte* host(const StorageRef& storage);
    // Copies on the host instead of the GPU when that is faster: the copy is
    // small and no batch is open or running, so it cannot pass a queued wait.
    // Returns false when the caller must encode a GPU copy.
    [[nodiscard]] bool host_copy_if_idle(const StorageRef& destination, const StorageRef& source, size_t bytes);

    // Holds later batches until a timeline semaphore of a MoltenVK device
    // reaches value. The host does not wait.
    void wait_vulkan_timeline(void* device, void* semaphore, uint64_t value);

    class Timestamps {
    public:
        virtual ~Timestamps() = default;
        // Returns the serial of the batch that writes the timestamp.
        virtual uint64_t write(size_t index) = 0;
        // nullopt until both marks are written and their batches completed.
        [[nodiscard]] virtual std::optional<float> milliseconds(size_t begin, size_t end) = 0;
    };
    // nullptr when the device cannot record timestamps.
    [[nodiscard]] std::unique_ptr<Timestamps> timestamps(size_t count);

} // namespace lfs::core::internal::metal_queue

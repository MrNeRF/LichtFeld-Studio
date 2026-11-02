/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */
#pragma once

#include "core/cuda/vmm_device_buffer.hpp"

#include <cstddef>
#include <cuda_runtime.h>
#include <memory>
#include <mutex>

namespace lfs::core {

    // Device storage for a tensor that reserves more rows than it holds. The
    // whole capacity is a virtual reservation; physical memory is committed
    // (and zero-filled) only for the prefix the tensor has grown into, so a
    // reserve for max_cap costs nothing until the rows exist. Growth maps new
    // pages behind the same address: no copy and no pointer change.
    struct LazyDeviceStorage {
        // Reservations smaller than this commit their whole capacity: the
        // 2 MiB granule would cost more than deferring saves.
        static constexpr std::size_t kMinReservedBytes = 8u << 20;

        // Returns nullptr when the device has no virtual memory management;
        // the caller then allocates the full capacity as before. Throws
        // MemoryAllocationError when the initial commit cannot be backed.
        [[nodiscard]] static std::shared_ptr<LazyDeviceStorage> create(
            std::size_t reserved_bytes, std::size_t committed_bytes, cudaStream_t stream);

        // Backs [0, bytes) with zeroed memory, ordered on `stream`. Commits a
        // little ahead so row-by-row growth does not map a granule per call.
        void commit(std::size_t bytes, cudaStream_t stream);

        [[nodiscard]] void* data() const noexcept { return buffer_.data(); }
        [[nodiscard]] std::size_t committed_bytes() const;

        // Committed bytes past the prefix any tensor has grown into: memory
        // already held for the next rows.
        [[nodiscard]] static std::size_t committed_ahead_bytes_total() noexcept;
        // Unmaps committed-ahead pages of every lazy storage not busy
        // committing, keeping the grown rows plus the padding granule.
        // Returns the bytes released.
        static std::size_t release_ahead_total();

        LazyDeviceStorage(const LazyDeviceStorage&) = delete;
        LazyDeviceStorage& operator=(const LazyDeviceStorage&) = delete;
        ~LazyDeviceStorage();

    private:
        LazyDeviceStorage() = default;
        void commit_locked(std::size_t bytes, cudaStream_t stream);
        void zero_new_pages(cudaStream_t stream);
        std::size_t release_ahead_locked();

        VmmDeviceBuffer buffer_;
        std::size_t reserved_bytes_ = 0;
        std::size_t zeroed_bytes_ = 0;
        std::size_t requested_bytes_ = 0;
        mutable std::mutex mutex_;
    };

} // namespace lfs::core

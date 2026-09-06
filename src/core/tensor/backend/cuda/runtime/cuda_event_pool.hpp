/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include "core/export.hpp"
#include <atomic>
#include <cstdint>
#include <cuda_runtime.h>
#include <mutex>
#include <vector>

namespace lfs::core {

    // Process-wide pool of cudaEventDisableTiming events. Cross-stream waits and
    // deferred frees record events in hot paths; pooling avoids per-call
    // cudaEventCreate/Destroy. Exported from lfs_core so all DSOs share one pool.
    class LFS_CORE_API CudaEventPool {
    public:
        static constexpr size_t MAX_POOL_SIZE = 512;

        static CudaEventPool& instance();

        // Returns nullptr if event creation fails (no CUDA context, OOM).
        cudaEvent_t acquire();

        // Returning an event with a pending cudaStreamWaitEvent is safe: the wait
        // snapshots the record it saw; later re-record or destroy does not affect it.
        void release(cudaEvent_t event);

        void shutdown();

        struct Stats {
            std::atomic<uint64_t> created{0};
            std::atomic<uint64_t> reused{0};
        };

        const Stats& stats() const { return stats_; }

        size_t pooled_count() const {
            std::lock_guard<std::mutex> lock(mutex_);
            return pool_.size();
        }

        CudaEventPool(const CudaEventPool&) = delete;
        CudaEventPool& operator=(const CudaEventPool&) = delete;

    private:
        CudaEventPool() = default;
        ~CudaEventPool();

        std::vector<cudaEvent_t> pool_;
        mutable std::mutex mutex_;
        std::atomic<bool> shutdown_{false};
        Stats stats_;
    };

    // Toggleable failure seam for exercising the host-sync fallback.
    // Production never enables it.
    LFS_CORE_API void set_cuda_event_acquire_failure_for_testing(bool enabled) noexcept;

    // Orders all work currently enqueued on `from` before future work on `to`
    // (pooled event edge, host-sync fallback). Unlike waitForCUDAStream, a
    // nullptr `from` (legacy default stream) is still bridged — allocator
    // reuse must order against legacy-stream work too. `from` is a stored
    // home stream that may have been released and destroyed since it was
    // recorded, so a handle in the retired registry is skipped without
    // touching the driver. The driver reuses handle values: a caller holding
    // a live stream goes through waitForCUDAStream (or a guard, set_stream,
    // the pools), which drops the handle from the registry first.
    LFS_CORE_API void bridgeStreams(cudaStream_t from, cudaStream_t to);

} // namespace lfs::core

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

    // Orders all work currently enqueued on `from` before future work on `to`
    // (pooled event edge, host-sync fallback). Unlike waitForCUDAStream, a
    // nullptr `from` (legacy default stream) is still bridged — allocator
    // reuse must order against legacy-stream work too.
    LFS_CORE_API void bridgeStreams(cudaStream_t from, cudaStream_t to);

} // namespace lfs::core

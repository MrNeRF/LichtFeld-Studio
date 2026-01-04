/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include "core/logger.hpp"
#include <cuda_runtime.h>
#include <atomic>
#include <vector>
#include <mutex>
#include <cstdint>

namespace lfs::core {

/**
 * @brief Stream-ordered deferred free queue for safe GPU memory recycling
 *
 * PROBLEM:
 * GPU operations are asynchronous. When we free memory, the GPU may still be
 * using it. Calling cudaDeviceSynchronize() on every free is too slow.
 *
 * SOLUTION:
 * Queue freed memory with a CUDA event that marks when it's safe to reuse.
 * Periodically process the queue and recycle memory whose events have completed.
 *
 * BENEFITS:
 * - No cudaDeviceSynchronize() during normal operation
 * - Memory is recycled as soon as GPU operations complete
 * - Batch processing amortizes event checking overhead
 *
 * USAGE:
 *   queue.defer_free(ptr, size, stream);  // Queue for later freeing
 *   queue.process();                       // Call periodically to recycle
 */
class DeferredFreeQueue {
public:
    // Configuration
    static constexpr size_t INITIAL_CAPACITY = 1024;
    static constexpr size_t PROCESS_BATCH_SIZE = 64;
    static constexpr size_t EVENT_POOL_SIZE = 256;

    // Callback type for actual memory freeing
    using FreeCallback = void (*)(void* ptr, size_t size);

    static DeferredFreeQueue& instance() {
        static DeferredFreeQueue queue;
        return queue;
    }

    /**
     * @brief Queue memory for deferred freeing
     * @param ptr Pointer to free
     * @param size Size of allocation (for slab allocator routing)
     * @param stream Stream that last used this memory
     * @param callback Function to call when memory is safe to free
     *
     * Thread-safe: Uses lock
     * Time: O(1) amortized
     */
    void defer_free(void* ptr, size_t size, cudaStream_t stream, FreeCallback callback) {
        if (!ptr) return;

        // Get a CUDA event to track completion
        cudaEvent_t event = acquire_event();
        if (!event) {
            // Event pool exhausted - fall back to sync + immediate free
            cudaStreamSynchronize(stream);
            callback(ptr, size);
            return;
        }

        // Record the event on the stream
        cudaError_t err = cudaEventRecord(event, stream);
        if (err != cudaSuccess) {
            release_event(event);
            cudaStreamSynchronize(stream);
            callback(ptr, size);
            return;
        }

        // Add to pending queue
        {
            std::lock_guard<std::mutex> lock(queue_mutex_);
            pending_.push_back({ptr, size, event, callback});
            stats_.queued_count.fetch_add(1, std::memory_order_relaxed);
            stats_.queued_bytes.fetch_add(size, std::memory_order_relaxed);
        }
    }

    /**
     * @brief Process pending frees, recycling memory whose events have completed
     * @param max_items Maximum items to process (0 = all)
     * @return Number of items freed
     *
     * Thread-safe: Uses lock
     * Call periodically (e.g., every N allocations or at frame boundaries)
     */
    size_t process(size_t max_items = PROCESS_BATCH_SIZE) {
        std::vector<PendingFree> to_free;
        to_free.reserve(max_items > 0 ? max_items : PROCESS_BATCH_SIZE);

        {
            std::lock_guard<std::mutex> lock(queue_mutex_);
            if (pending_.empty()) return 0;

            size_t count = 0;
            size_t i = 0;
            while (i < pending_.size() && (max_items == 0 || count < max_items)) {
                auto& item = pending_[i];

                // Check if event has completed (non-blocking)
                cudaError_t err = cudaEventQuery(item.event);
                if (err == cudaSuccess) {
                    // Event completed - can free this memory
                    to_free.push_back(item);
                    release_event(item.event);

                    // Remove from pending (swap with last for O(1))
                    pending_[i] = pending_.back();
                    pending_.pop_back();
                    count++;
                    // Don't increment i - we need to check the swapped item
                } else if (err == cudaErrorNotReady) {
                    // Not ready yet - skip
                    i++;
                } else {
                    // Error - log and skip
                    LOG_WARN("cudaEventQuery failed: %s", cudaGetErrorString(err));
                    i++;
                }
            }
        }

        // Actually free the memory (outside the lock)
        for (const auto& item : to_free) {
            item.callback(item.ptr, item.size);
            stats_.freed_count.fetch_add(1, std::memory_order_relaxed);
            stats_.freed_bytes.fetch_add(item.size, std::memory_order_relaxed);
            stats_.queued_bytes.fetch_sub(item.size, std::memory_order_relaxed);
        }

        return to_free.size();
    }

    /**
     * @brief Force-flush all pending frees (synchronous)
     * Use at shutdown or when immediate memory release is needed
     */
    void flush() {
        std::vector<PendingFree> to_free;

        {
            std::lock_guard<std::mutex> lock(queue_mutex_);
            to_free = std::move(pending_);
            pending_.clear();
        }

        // Synchronize and free all
        cudaDeviceSynchronize();

        for (const auto& item : to_free) {
            release_event(item.event);
            item.callback(item.ptr, item.size);
            stats_.freed_count.fetch_add(1, std::memory_order_relaxed);
            stats_.freed_bytes.fetch_add(item.size, std::memory_order_relaxed);
        }

        stats_.queued_bytes.store(0, std::memory_order_relaxed);
    }

    /**
     * @brief Get queue statistics
     */
    struct Stats {
        std::atomic<uint64_t> queued_count{0};
        std::atomic<uint64_t> freed_count{0};
        std::atomic<uint64_t> queued_bytes{0};
        std::atomic<uint64_t> freed_bytes{0};
    };

    const Stats& stats() const { return stats_; }

    size_t pending_count() const {
        std::lock_guard<std::mutex> lock(queue_mutex_);
        return pending_.size();
    }

    // Disable copy/move
    DeferredFreeQueue(const DeferredFreeQueue&) = delete;
    DeferredFreeQueue& operator=(const DeferredFreeQueue&) = delete;

private:
    struct PendingFree {
        void* ptr;
        size_t size;
        cudaEvent_t event;
        FreeCallback callback;
    };

    DeferredFreeQueue() {
        pending_.reserve(INITIAL_CAPACITY);
        initialize_event_pool();
    }

    ~DeferredFreeQueue() {
        flush();
        cleanup_event_pool();
    }

    void initialize_event_pool() {
        std::lock_guard<std::mutex> lock(event_pool_mutex_);
        for (size_t i = 0; i < EVENT_POOL_SIZE; i++) {
            cudaEvent_t event;
            cudaError_t err = cudaEventCreateWithFlags(&event, cudaEventDisableTiming);
            if (err == cudaSuccess) {
                event_pool_.push_back(event);
            }
        }
        LOG_DEBUG("DeferredFreeQueue: Created %zu CUDA events", event_pool_.size());
    }

    void cleanup_event_pool() {
        std::lock_guard<std::mutex> lock(event_pool_mutex_);
        for (cudaEvent_t event : event_pool_) {
            cudaEventDestroy(event);
        }
        event_pool_.clear();
    }

    cudaEvent_t acquire_event() {
        std::lock_guard<std::mutex> lock(event_pool_mutex_);
        if (event_pool_.empty()) {
            // Try to create a new event
            cudaEvent_t event;
            cudaError_t err = cudaEventCreateWithFlags(&event, cudaEventDisableTiming);
            if (err == cudaSuccess) {
                return event;
            }
            return nullptr;
        }

        cudaEvent_t event = event_pool_.back();
        event_pool_.pop_back();
        return event;
    }

    void release_event(cudaEvent_t event) {
        std::lock_guard<std::mutex> lock(event_pool_mutex_);
        if (event_pool_.size() < EVENT_POOL_SIZE * 2) {
            event_pool_.push_back(event);
        } else {
            cudaEventDestroy(event);
        }
    }

    std::vector<PendingFree> pending_;
    mutable std::mutex queue_mutex_;

    std::vector<cudaEvent_t> event_pool_;
    std::mutex event_pool_mutex_;

    Stats stats_;
};

} // namespace lfs::core

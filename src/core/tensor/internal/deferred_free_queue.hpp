/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include "core/cuda_error.hpp"
#include "cuda_event_pool.hpp"
#include <atomic>
#include <cstdint>
#include <cuda_runtime.h>
#include <format>
#include <mutex>
#include <vector>

namespace lfs::core {

    // Stream-ordered deferred free queue. Uses CUDA events to recycle GPU memory
    // without blocking. Call process() periodically to free completed allocations.
    class DeferredFreeQueue {
    public:
        static constexpr size_t INITIAL_CAPACITY = 1024;
        static constexpr size_t PROCESS_BATCH_SIZE = 64;

        using FreeCallback = void (*)(void* ptr, size_t size);

        static DeferredFreeQueue& instance() {
            static DeferredFreeQueue queue;
            return queue;
        }

        void shutdown() {
            bool expected = false;
            if (!shutdown_.compare_exchange_strong(expected, true))
                return;
            flush();
        }

        void defer_free(void* ptr, size_t size, cudaStream_t stream, FreeCallback callback) {
            if (!ptr)
                return;
            if (shutdown_.load(std::memory_order_acquire)) {
                callback(ptr, size);
                return;
            }

            cudaEvent_t event = CudaEventPool::instance().acquire();
            if (!event) {
                const cudaError_t sync_status = cudaStreamSynchronize(stream);
                if (sync_status != cudaSuccess) {
                    ensure_cuda_success(
                        sync_status, "cudaStreamSynchronize(deferred-free fallback)",
                        std::format("ptr={}, bytes={}, stream={}", ptr, size,
                                    static_cast<void*>(stream)),
                        std::source_location::current(), CudaFailureDisposition::LogOnly);
                }
                callback(ptr, size);
                return;
            }

            const cudaError_t err = cudaEventRecord(event, stream);
            if (err != cudaSuccess) {
                CudaEventPool::instance().release(event);
                ensure_cuda_success(
                    err, "cudaEventRecord(deferred free)",
                    std::format("ptr={}, bytes={}, stream={}, fallback=stream sync",
                                ptr, size, static_cast<void*>(stream)),
                    std::source_location::current(), CudaFailureDisposition::LogOnly);
                const cudaError_t sync_status = cudaStreamSynchronize(stream);
                if (sync_status != cudaSuccess) {
                    ensure_cuda_success(
                        sync_status, "cudaStreamSynchronize(deferred-free event fallback)",
                        std::format("ptr={}, bytes={}, stream={}", ptr, size,
                                    static_cast<void*>(stream)),
                        std::source_location::current(), CudaFailureDisposition::LogOnly);
                }
                callback(ptr, size);
                return;
            }

            {
                std::lock_guard<std::mutex> lock(queue_mutex_);
                pending_.push_back({ptr, size, event, callback});
                stats_.queued_count.fetch_add(1, std::memory_order_relaxed);
                stats_.queued_bytes.fetch_add(size, std::memory_order_relaxed);
            }
        }

        size_t process(size_t max_items = PROCESS_BATCH_SIZE) {
            std::vector<PendingFree> to_free;
            to_free.reserve(max_items > 0 ? max_items : PROCESS_BATCH_SIZE);

            {
                std::lock_guard<std::mutex> lock(queue_mutex_);
                if (pending_.empty())
                    return 0;

                size_t count = 0;
                size_t i = 0;
                while (i < pending_.size() && (max_items == 0 || count < max_items)) {
                    const auto& item = pending_[i];
                    const cudaError_t err = cudaEventQuery(item.event);
                    if (err == cudaSuccess) {
                        to_free.push_back(item);
                        CudaEventPool::instance().release(item.event);
                        pending_[i] = pending_.back();
                        pending_.pop_back();
                        ++count;
                    } else if (err == cudaErrorNotReady) {
                        ++i;
                    } else {
                        ensure_cuda_success(
                            err, "cudaEventQuery(deferred free)",
                            std::format("ptr={}, bytes={}", item.ptr, item.size),
                            std::source_location::current(), CudaFailureDisposition::LogOnly);
                        ++i;
                    }
                }
            }

            for (const auto& item : to_free) {
                item.callback(item.ptr, item.size);
                stats_.freed_count.fetch_add(1, std::memory_order_relaxed);
                stats_.freed_bytes.fetch_add(item.size, std::memory_order_relaxed);
                stats_.queued_bytes.fetch_sub(item.size, std::memory_order_relaxed);
            }

            return to_free.size();
        }

        void flush() {
            std::vector<PendingFree> to_free;

            {
                std::lock_guard<std::mutex> lock(queue_mutex_);
                to_free = std::move(pending_);
                pending_.clear();
            }

            const cudaError_t sync_status = cudaDeviceSynchronize();
            if (sync_status != cudaSuccess) {
                ensure_cuda_success(
                    sync_status, "cudaDeviceSynchronize(deferred-free flush)", {},
                    std::source_location::current(), CudaFailureDisposition::LogOnly);
            }

            for (const auto& item : to_free) {
                CudaEventPool::instance().release(item.event);
                item.callback(item.ptr, item.size);
                stats_.freed_count.fetch_add(1, std::memory_order_relaxed);
                stats_.freed_bytes.fetch_add(item.size, std::memory_order_relaxed);
            }

            stats_.queued_bytes.store(0, std::memory_order_relaxed);
        }

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
        }

        ~DeferredFreeQueue() {
            shutdown();
        }

        std::vector<PendingFree> pending_;
        mutable std::mutex queue_mutex_;

        std::atomic<bool> shutdown_{false};

        Stats stats_;
    };

} // namespace lfs::core

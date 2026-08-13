/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "internal/cuda_event_pool.hpp"
#include "core/cuda_error.hpp"
#include "core/logger.hpp"
#include "internal/stream_lifetime.hpp"

#include <atomic>
#include <format>
#include <string_view>

namespace lfs::core {

    namespace {
        std::atomic<bool> g_force_event_acquire_failure_for_testing{false};

        void warn_bridge_skipped_once(cudaStream_t from, cudaStream_t to, const char* reason) {
            static std::atomic<bool> warned{false};
            if (!warned.exchange(true, std::memory_order_relaxed)) {
                LOG_WARN("skipping stream bridge: {} (from_stream={}, to_stream={})",
                         reason, static_cast<void*>(from), static_cast<void*>(to));
            }
        }

        void synchronize_stream_bridge_source(cudaStream_t from,
                                              cudaStream_t to,
                                              const std::string_view reason) {
            const cudaError_t sync_status = cudaStreamSynchronize(from);
            if (sync_status != cudaSuccess) {
                ensure_cuda_success(
                    sync_status, "cudaStreamSynchronize(tensor stream bridge fallback)",
                    std::format("from_stream={}, to_stream={}; reason={}",
                                static_cast<void*>(from), static_cast<void*>(to), reason),
                    LFS_SOURCE_SITE_CURRENT(), CudaFailureDisposition::LogOnly);
                // Drop the latched runtime error so later LFS_CUDA_LAUNCH_CHECK
                // (cudaPeekAtLastError) does not misattribute this failure.
                (void)cudaGetLastError();
            }
        }
    } // namespace

    CudaEventPool& CudaEventPool::instance() {
        static CudaEventPool pool;
        return pool;
    }

    cudaEvent_t CudaEventPool::acquire() {
        if (g_force_event_acquire_failure_for_testing.load(std::memory_order_acquire)) {
            return nullptr;
        }
        if (!shutdown_.load(std::memory_order_acquire)) {
            std::lock_guard<std::mutex> lock(mutex_);
            if (!pool_.empty()) {
                cudaEvent_t event = pool_.back();
                pool_.pop_back();
                stats_.reused.fetch_add(1, std::memory_order_relaxed);
                return event;
            }
        }

        cudaEvent_t event = nullptr;
        const cudaError_t create_status =
            cudaEventCreateWithFlags(&event, cudaEventDisableTiming);
        if (create_status != cudaSuccess) {
            ensure_cuda_success(
                create_status, "cudaEventCreateWithFlags(tensor event pool)",
                "fallback=stream synchronization", LFS_SOURCE_SITE_CURRENT(),
                CudaFailureDisposition::LogOnly);
            return nullptr;
        }
        stats_.created.fetch_add(1, std::memory_order_relaxed);
        return event;
    }

    void CudaEventPool::release(cudaEvent_t event) {
        if (!event)
            return;
        if (!shutdown_.load(std::memory_order_acquire)) {
            std::lock_guard<std::mutex> lock(mutex_);
            if (pool_.size() < MAX_POOL_SIZE) {
                pool_.push_back(event);
                return;
            }
        }
        const cudaError_t destroy_status = cudaEventDestroy(event);
        if (destroy_status != cudaSuccess) {
            ensure_cuda_success(
                destroy_status, "cudaEventDestroy(tensor event pool release)", {},
                LFS_SOURCE_SITE_CURRENT(), CudaFailureDisposition::LogOnlyNoLatch);
        }
    }

    void CudaEventPool::shutdown() {
        bool expected = false;
        if (!shutdown_.compare_exchange_strong(expected, true))
            return;
        std::lock_guard<std::mutex> lock(mutex_);
        for (cudaEvent_t event : pool_) {
            const cudaError_t destroy_status = cudaEventDestroy(event);
            if (destroy_status != cudaSuccess) {
                ensure_cuda_success(
                    destroy_status, "cudaEventDestroy(tensor event pool shutdown)", {},
                    LFS_SOURCE_SITE_CURRENT(), CudaFailureDisposition::LogOnlyNoLatch);
            }
        }
        pool_.clear();
    }

    CudaEventPool::~CudaEventPool() {
        shutdown();
    }

    void set_cuda_event_acquire_failure_for_testing(const bool enabled) noexcept {
        g_force_event_acquire_failure_for_testing.store(enabled, std::memory_order_release);
    }

    void bridgeStreams(cudaStream_t from, cudaStream_t to) {
        if (from == to) {
            return;
        }
        // Null stream is the default stream — wait/record still valid, but a
        // destroyed user stream handle can crash in the driver. Detect
        // capture status first; any query failure means the stream is unusable.
        if (from != nullptr) {
            if (is_stream_retired(from)) {
                // release_stream synchronized the stream before retiring it, so
                // there is no pending work to order against.
                warn_bridge_skipped_once(from, to, "source stream retired before destruction");
                return;
            }
            cudaStreamCaptureStatus capture = cudaStreamCaptureStatusNone;
            const cudaError_t capture_status = cudaStreamIsCapturing(from, &capture);
            if (capture_status != cudaSuccess) {
                (void)cudaGetLastError();
                if (capture_status == cudaErrorInvalidResourceHandle ||
                    capture_status == cudaErrorContextIsDestroyed) {
                    warn_bridge_skipped_once(from, to, "source stream handle invalid (already destroyed?)");
                    return;
                }
                synchronize_stream_bridge_source(from, to, "capture status query failed");
                return;
            }
            if (capture != cudaStreamCaptureStatusNone) {
                // Cannot record events into an active capture without joining it.
                synchronize_stream_bridge_source(from, to, "source stream is capturing");
                return;
            }
        }

        if (cudaEvent_t edge = CudaEventPool::instance().acquire()) {
            const cudaError_t record_status = cudaEventRecord(edge, from);
            cudaError_t wait_status = cudaErrorUnknown;
            if (record_status == cudaSuccess) {
                wait_status = cudaStreamWaitEvent(to, edge, 0);
            } else {
                ensure_cuda_success(
                    record_status, "cudaEventRecord(tensor stream bridge)",
                    std::format("from_stream={}, to_stream={}; fallback=stream sync",
                                static_cast<void*>(from), static_cast<void*>(to)),
                    LFS_SOURCE_SITE_CURRENT(), CudaFailureDisposition::LogOnly);
                (void)cudaGetLastError();
            }
            if (record_status == cudaSuccess && wait_status != cudaSuccess) {
                ensure_cuda_success(
                    wait_status, "cudaStreamWaitEvent(tensor stream bridge)",
                    std::format("from_stream={}, to_stream={}; fallback=stream sync",
                                static_cast<void*>(from), static_cast<void*>(to)),
                    LFS_SOURCE_SITE_CURRENT(), CudaFailureDisposition::LogOnly);
                (void)cudaGetLastError();
            }
            CudaEventPool::instance().release(edge);
            if (record_status == cudaSuccess && wait_status == cudaSuccess) {
                return;
            }
        }

        synchronize_stream_bridge_source(from, to, "event edge unavailable or failed");
    }

} // namespace lfs::core

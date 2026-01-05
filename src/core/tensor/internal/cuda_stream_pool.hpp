/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include "core/logger.hpp"
#include <atomic>
#include <cuda_runtime.h>
#include <mutex>
#include <vector>

namespace lfs::core {

    /** RAII wrapper for CUDA events. Move-only, timing disabled by default. */
    class CUDAEvent {
    public:
        explicit CUDAEvent(bool enable_timing = false) {
            unsigned int flags = enable_timing ? cudaEventDefault : cudaEventDisableTiming;
            cudaError_t err = cudaEventCreateWithFlags(&event_, flags);
            if (err != cudaSuccess) {
                LOG_ERROR("Failed to create CUDA event: {}", cudaGetErrorString(err));
                event_ = nullptr;
            }
        }

        ~CUDAEvent() {
            if (event_) {
                cudaEventDestroy(event_);
            }
        }

        CUDAEvent(CUDAEvent&& other) noexcept : event_(other.event_) {
            other.event_ = nullptr;
        }

        CUDAEvent& operator=(CUDAEvent&& other) noexcept {
            if (this != &other) {
                if (event_) {
                    cudaEventDestroy(event_);
                }
                event_ = other.event_;
                other.event_ = nullptr;
            }
            return *this;
        }

        CUDAEvent(const CUDAEvent&) = delete;
        CUDAEvent& operator=(const CUDAEvent&) = delete;

        bool record(cudaStream_t stream = nullptr) {
            if (!event_)
                return false;
            cudaError_t err = cudaEventRecord(event_, stream);
            if (err != cudaSuccess) {
                LOG_ERROR("Failed to record CUDA event: {}", cudaGetErrorString(err));
                return false;
            }
            return true;
        }

        bool synchronize() const {
            if (!event_)
                return false;
            cudaError_t err = cudaEventSynchronize(event_);
            return err == cudaSuccess;
        }

        bool wait(cudaStream_t stream) const {
            if (!event_)
                return false;
            cudaError_t err = cudaStreamWaitEvent(stream, event_, 0);
            if (err != cudaSuccess) {
                LOG_ERROR("cudaStreamWaitEvent failed: {}", cudaGetErrorString(err));
                return false;
            }
            return true;
        }

        bool is_complete() const {
            if (!event_)
                return true;
            cudaError_t err = cudaEventQuery(event_);
            return err == cudaSuccess;
        }

        float elapsed_ms(const CUDAEvent& start) const {
            if (!event_ || !start.event_)
                return 0.0f;
            float ms = 0.0f;
            cudaError_t err = cudaEventElapsedTime(&ms, start.event_, event_);
            if (err != cudaSuccess) {
                LOG_ERROR("cudaEventElapsedTime failed: {}", cudaGetErrorString(err));
                return 0.0f;
            }
            return ms;
        }

        bool valid() const { return event_ != nullptr; }
        cudaEvent_t get() const { return event_; }

    private:
        cudaEvent_t event_ = nullptr;
    };

    /** Fixed pool of non-blocking CUDA streams with round-robin acquisition. Thread-safe. */
    class CUDAStreamPool {
    public:
        static constexpr size_t DEFAULT_POOL_SIZE = 8;
        static constexpr size_t HIGH_PRIORITY_POOL_SIZE = 2;

        static CUDAStreamPool& instance() {
            static CUDAStreamPool pool;
            return pool;
        }

        cudaStream_t acquire() {
            if (streams_.empty())
                return nullptr;
            size_t idx = next_stream_.fetch_add(1, std::memory_order_relaxed) % streams_.size();
            return streams_[idx];
        }

        cudaStream_t acquire_high_priority() {
            if (high_priority_streams_.empty())
                return acquire();
            size_t idx = next_high_priority_.fetch_add(1, std::memory_order_relaxed) %
                         high_priority_streams_.size();
            return high_priority_streams_[idx];
        }

        size_t size() const { return streams_.size(); }
        size_t high_priority_size() const { return high_priority_streams_.size(); }

        cudaStream_t get(size_t index) const {
            if (index >= streams_.size())
                return nullptr;
            return streams_[index];
        }

        void synchronize_all() {
            for (cudaStream_t stream : streams_) {
                cudaStreamSynchronize(stream);
            }
            for (cudaStream_t stream : high_priority_streams_) {
                cudaStreamSynchronize(stream);
            }
        }

        bool is_initialized() const { return !streams_.empty(); }

        CUDAStreamPool(const CUDAStreamPool&) = delete;
        CUDAStreamPool& operator=(const CUDAStreamPool&) = delete;

    private:
        CUDAStreamPool() {
            initialize();
        }

        ~CUDAStreamPool() {
            cleanup();
        }

        void initialize() {
            int device_count = 0;
            if (cudaGetDeviceCount(&device_count) != cudaSuccess || device_count == 0) {
                LOG_WARN("No CUDA devices available, stream pool disabled");
                return;
            }

            streams_.reserve(DEFAULT_POOL_SIZE);
            for (size_t i = 0; i < DEFAULT_POOL_SIZE; ++i) {
                cudaStream_t stream;
                cudaError_t err = cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking);
                if (err == cudaSuccess) {
                    streams_.push_back(stream);
                } else {
                    LOG_WARN("Failed to create stream {}: {}", i, cudaGetErrorString(err));
                }
            }

            int least_priority, greatest_priority;
            cudaDeviceGetStreamPriorityRange(&least_priority, &greatest_priority);

            high_priority_streams_.reserve(HIGH_PRIORITY_POOL_SIZE);
            for (size_t i = 0; i < HIGH_PRIORITY_POOL_SIZE; ++i) {
                cudaStream_t stream;
                cudaError_t err = cudaStreamCreateWithPriority(
                    &stream, cudaStreamNonBlocking, greatest_priority);
                if (err == cudaSuccess) {
                    high_priority_streams_.push_back(stream);
                } else {
                    LOG_WARN("Failed to create high-priority stream {}: {}", i, cudaGetErrorString(err));
                }
            }

            LOG_DEBUG("CUDAStreamPool: created {} regular + {} high-priority streams",
                      streams_.size(), high_priority_streams_.size());
        }

        void cleanup() {
            for (cudaStream_t stream : streams_) {
                cudaStreamDestroy(stream);
            }
            streams_.clear();

            for (cudaStream_t stream : high_priority_streams_) {
                cudaStreamDestroy(stream);
            }
            high_priority_streams_.clear();
        }

        std::vector<cudaStream_t> streams_;
        std::vector<cudaStream_t> high_priority_streams_;
        std::atomic<size_t> next_stream_{0};
        std::atomic<size_t> next_high_priority_{0};
    };

    /** RAII guard: acquires pooled stream, sets as thread's current, restores on destruction. */
    class PooledStreamGuard {
    public:
        explicit PooledStreamGuard(bool high_priority = false);
        ~PooledStreamGuard();

        cudaStream_t stream() const { return stream_; }

        PooledStreamGuard(const PooledStreamGuard&) = delete;
        PooledStreamGuard& operator=(const PooledStreamGuard&) = delete;
        PooledStreamGuard(PooledStreamGuard&&) = delete;
        PooledStreamGuard& operator=(PooledStreamGuard&&) = delete;

    private:
        cudaStream_t stream_;
        cudaStream_t prev_stream_;
    };

} // namespace lfs::core

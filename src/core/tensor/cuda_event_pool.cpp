/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "internal/cuda_event_pool.hpp"
#include "core/cuda_error.hpp"

#include <format>

namespace lfs::core {

    CudaEventPool& CudaEventPool::instance() {
        static CudaEventPool pool;
        return pool;
    }

    cudaEvent_t CudaEventPool::acquire() {
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

    void bridgeStreams(cudaStream_t from, cudaStream_t to) {
        if (from == to) {
            return;
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
            }
            if (record_status == cudaSuccess && wait_status != cudaSuccess) {
                ensure_cuda_success(
                    wait_status, "cudaStreamWaitEvent(tensor stream bridge)",
                    std::format("from_stream={}, to_stream={}; fallback=stream sync",
                                static_cast<void*>(from), static_cast<void*>(to)),
                    LFS_SOURCE_SITE_CURRENT(), CudaFailureDisposition::LogOnly);
            }
            CudaEventPool::instance().release(edge);
            if (record_status == cudaSuccess && wait_status == cudaSuccess) {
                return;
            }
        }

        const cudaError_t sync_status = cudaStreamSynchronize(from);
        if (sync_status != cudaSuccess) {
            ensure_cuda_success(
                sync_status, "cudaStreamSynchronize(tensor stream bridge fallback)",
                std::format("from_stream={}, to_stream={}; event edge also failed",
                            static_cast<void*>(from), static_cast<void*>(to)),
                LFS_SOURCE_SITE_CURRENT(), CudaFailureDisposition::LogOnly);
        }
    }

} // namespace lfs::core

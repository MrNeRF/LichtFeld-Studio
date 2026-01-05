/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include <cuda_runtime.h>
#include <mutex>
#include <thread>
#include <unordered_map>

namespace lfs::core {

    /** Thread-local CUDA stream management. Each thread has its own "current stream". */
    class CUDAStreamContext {
    public:
        static CUDAStreamContext& instance() {
            static CUDAStreamContext inst;
            return inst;
        }

        cudaStream_t getCurrentStream() {
            std::thread::id tid = std::this_thread::get_id();
            std::lock_guard<std::mutex> lock(mutex_);
            auto it = thread_streams_.find(tid);
            if (it != thread_streams_.end()) {
                return it->second;
            }
            return nullptr;
        }

        void setCurrentStream(cudaStream_t stream) {
            std::thread::id tid = std::this_thread::get_id();
            std::lock_guard<std::mutex> lock(mutex_);
            thread_streams_[tid] = stream;
        }

    private:
        CUDAStreamContext() = default;
        ~CUDAStreamContext() = default;
        CUDAStreamContext(const CUDAStreamContext&) = delete;
        CUDAStreamContext& operator=(const CUDAStreamContext&) = delete;

        std::mutex mutex_;
        std::unordered_map<std::thread::id, cudaStream_t> thread_streams_;
    };

    /** RAII guard: sets current stream on construction, restores previous on destruction. */
    class CUDAStreamGuard {
    public:
        explicit CUDAStreamGuard(cudaStream_t stream)
            : prev_stream_(CUDAStreamContext::instance().getCurrentStream()) {
            CUDAStreamContext::instance().setCurrentStream(stream);
        }

        ~CUDAStreamGuard() {
            CUDAStreamContext::instance().setCurrentStream(prev_stream_);
        }

        CUDAStreamGuard(const CUDAStreamGuard&) = delete;
        CUDAStreamGuard& operator=(const CUDAStreamGuard&) = delete;
        CUDAStreamGuard(CUDAStreamGuard&&) = delete;
        CUDAStreamGuard& operator=(CUDAStreamGuard&&) = delete;

    private:
        cudaStream_t prev_stream_;
    };

    inline cudaStream_t getCurrentCUDAStream() {
        return CUDAStreamContext::instance().getCurrentStream();
    }

} // namespace lfs::core

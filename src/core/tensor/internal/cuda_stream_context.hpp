/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include <cuda_runtime.h>
#include <mutex>
#include <thread>
#include <unordered_map>

namespace lfs::core {

    /**
     * Thread-local current CUDA stream management (PyTorch-style)
     *
     * This follows PyTorch's approach where each thread has its own "current stream"
     * that is used by default for all CUDA operations. This allows DataLoader workers
     * to each have their own stream without passing streams explicitly through every
     * operation.
     */
    class CUDAStreamContext {
    public:
        static CUDAStreamContext& instance() {
            static CUDAStreamContext inst;
            return inst;
        }

        // Get the current stream for this thread (default: nullptr = stream 0)
        cudaStream_t getCurrentStream() {
            std::thread::id tid = std::this_thread::get_id();
            std::lock_guard<std::mutex> lock(mutex_);
            auto it = thread_streams_.find(tid);
            if (it != thread_streams_.end()) {
                return it->second;
            }
            return nullptr; // Default stream
        }

        // Set the current stream for this thread
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

    /**
     * RAII guard for temporarily setting the current CUDA stream
     * (PyTorch's CUDAStreamGuard pattern)
     *
     * Usage in DataLoader worker:
     *   cudaStream_t worker_stream;
     *   cudaStreamCreate(&worker_stream);
     *   {
     *       CUDAStreamGuard guard(worker_stream);
     *       // All tensor operations in this scope use worker_stream
     *       auto image = load_image();
     *       image = image.to(Device::CUDA);  // Uses worker_stream!
     *       image = preprocess(image);        // Uses worker_stream!
     *   }
     *   // Stream restored to previous value
     */
    class CUDAStreamGuard {
    public:
        explicit CUDAStreamGuard(cudaStream_t stream)
            : prev_stream_(CUDAStreamContext::instance().getCurrentStream()) {
            CUDAStreamContext::instance().setCurrentStream(stream);
        }

        ~CUDAStreamGuard() {
            CUDAStreamContext::instance().setCurrentStream(prev_stream_);
        }

        // Delete copy/move
        CUDAStreamGuard(const CUDAStreamGuard&) = delete;
        CUDAStreamGuard& operator=(const CUDAStreamGuard&) = delete;
        CUDAStreamGuard(CUDAStreamGuard&&) = delete;
        CUDAStreamGuard& operator=(CUDAStreamGuard&&) = delete;

    private:
        cudaStream_t prev_stream_;
    };

    // Helper function to get current stream (like PyTorch's getCurrentCUDAStream)
    inline cudaStream_t getCurrentCUDAStream() {
        return CUDAStreamContext::instance().getCurrentStream();
    }

} // namespace lfs::core

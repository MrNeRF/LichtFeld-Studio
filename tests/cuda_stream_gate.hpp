/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#pragma once

#include <chrono>
#include <condition_variable>
#include <cuda_runtime.h>
#include <future>
#include <mutex>
#include <thread>

namespace lfs::test {
    class CudaStreamGate {
    public:
        CudaStreamGate()
            : entered_(entered_promise_.get_future()), watchdog_([this] {
                  std::unique_lock lock(mutex_);
                  changed_.wait_for(lock, std::chrono::seconds(5), [this] { return released_; });
                  released_ = true;
                  changed_.notify_all();
              }) {}

        ~CudaStreamGate() {
            release();
            if (stream_)
                (void)cudaStreamSynchronize(stream_);
        }

        cudaError_t block(cudaStream_t stream) {
            stream_ = stream;
            return cudaLaunchHostFunc(stream, [](void* data) {
                auto& gate = *static_cast<CudaStreamGate*>(data);
                gate.entered_promise_.set_value();
                std::unique_lock lock(gate.mutex_);
                gate.changed_.wait(lock, [&] { return gate.released_; }); }, this);
        }

        bool entered() {
            return entered_.wait_for(std::chrono::seconds(1)) == std::future_status::ready;
        }

        bool released() {
            std::lock_guard lock(mutex_);
            return released_;
        }

        void release() {
            std::lock_guard lock(mutex_);
            released_ = true;
            changed_.notify_all();
        }

    private:
        std::mutex mutex_;
        std::condition_variable changed_;
        bool released_ = false;
        std::promise<void> entered_promise_;
        std::future<void> entered_;
        cudaStream_t stream_ = nullptr;
        std::jthread watchdog_;
    };
} // namespace lfs::test

/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#pragma once
#include "core/logger.hpp"
#include <atomic>
#include <condition_variable>
#include <deque>
#include <functional>
#include <mutex>
#include <thread>
namespace lfs::core::internal {
    class QueueCallbackWorker {
        struct Callback {
            uint64_t timeline;
            void (*function)(void*);
            void* user;
        };
        std::function<void(uint64_t)> wait_;
        std::mutex mutex_;
        std::condition_variable cv_;
        std::deque<Callback> callbacks_;
        std::thread thread_;
        std::atomic<size_t> outstanding_{0};
        bool stop_ = false;
        void run() {
            for (;;) {
                Callback callback;
                {
                    std::unique_lock lock(mutex_);
                    cv_.wait(lock, [&] { return stop_ || !callbacks_.empty(); });
                    if (callbacks_.empty())
                        return;
                    callback = callbacks_.front();
                    callbacks_.pop_front();
                }
                try {
                    if (callback.timeline)
                        wait_(callback.timeline);
                    if (callback.function)
                        callback.function(callback.user);
                } catch (const std::exception& error) { LOG_ERROR("Tensor queue host callback failed: {}", error.what()); } catch (...) {
                    LOG_ERROR("Tensor queue host callback failed");
                }
                {
                    std::lock_guard lock(mutex_);
                    outstanding_.fetch_sub(1, std::memory_order_acq_rel);
                    cv_.notify_all();
                }
            }
        }

    public:
        explicit QueueCallbackWorker(std::function<void(uint64_t)> wait) : wait_(std::move(wait)) {}
        ~QueueCallbackWorker() {
            {
                std::lock_guard lock(mutex_);
                stop_ = true;
            }
            cv_.notify_all();
            if (thread_.joinable())
                thread_.join();
        }
        void enqueue(uint64_t timeline, void (*function)(void*), void* user) {
            {
                std::lock_guard lock(mutex_);
                // Start before publishing work: a thread creation failure must
                // not leave an outstanding callback that nobody can complete.
                if (!thread_.joinable())
                    thread_ = std::thread([this] { run(); });
                callbacks_.push_back({timeline, function, user});
                outstanding_.fetch_add(1, std::memory_order_acq_rel);
            }
            cv_.notify_all();
        }
        bool ready() const { return outstanding_.load(std::memory_order_acquire) == 0; }
        void wait() {
            std::unique_lock lock(mutex_);
            cv_.wait(lock, [&] { return ready(); });
        }
    };
} // namespace lfs::core::internal

/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "internal/stream_lifetime.hpp"

#include <algorithm>
#include <atomic>
#include <mutex>
#include <vector>

namespace lfs::core {
    namespace {

        struct RetiredStreamRegistry {
            std::mutex mutex;
            std::vector<cudaStream_t> streams;
            std::atomic<size_t> size{0};
        };

        RetiredStreamRegistry& registry() {
            // Leak-on-exit: never destroyed, so CUDA teardown cannot race a
            // static destructor (same rationale as CudaEventPool::instance).
            static auto* instance = new RetiredStreamRegistry();
            return *instance;
        }

    } // namespace

    void retire_stream(cudaStream_t stream) noexcept {
        if (!stream) {
            return;
        }
        auto& reg = registry();
        std::lock_guard lock(reg.mutex);
        if (std::find(reg.streams.begin(), reg.streams.end(), stream) != reg.streams.end()) {
            return;
        }
        reg.streams.push_back(stream);
        reg.size.store(reg.streams.size(), std::memory_order_release);
    }

    void unretire_stream(cudaStream_t stream) noexcept {
        if (!stream) {
            return;
        }
        auto& reg = registry();
        if (reg.size.load(std::memory_order_acquire) == 0) {
            return;
        }
        std::lock_guard lock(reg.mutex);
        auto it = std::find(reg.streams.begin(), reg.streams.end(), stream);
        if (it == reg.streams.end()) {
            return;
        }
        reg.streams.erase(it);
        reg.size.store(reg.streams.size(), std::memory_order_release);
    }

    bool is_stream_retired(cudaStream_t stream) noexcept {
        if (!stream) {
            return false;
        }
        auto& reg = registry();
        if (reg.size.load(std::memory_order_acquire) == 0) {
            return false;
        }
        std::lock_guard lock(reg.mutex);
        return std::find(reg.streams.begin(), reg.streams.end(), stream) != reg.streams.end();
    }

} // namespace lfs::core

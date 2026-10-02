/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include "io/pipelined_image_loader.hpp"

#include <algorithm>
#include <atomic>
#include <condition_variable>
#include <memory>
#include <mutex>
#include <vector>

namespace lfs::io {

    struct PipelinedImageLoader::DecodedFrameRing
        : std::enable_shared_from_this<PipelinedImageLoader::DecodedFrameRing> {
        struct Slot {
            lfs::core::Tensor storage;
            bool in_use = false;
        };

        struct Lease {
            lfs::core::Tensor storage;
            std::weak_ptr<DecodedFrameRing> owner;
            size_t index = 0;

            ~Lease() {
                if (auto ring = owner.lock()) {
                    ring->release(index, storage);
                }
            }
        };

        explicit DecodedFrameRing(const size_t capacity, const std::atomic<bool>* running)
            : slots_(std::max<size_t>(1, capacity)),
              capacity_limit_(slots_.size()),
              running_(running) {}

        void release(const size_t index, const lfs::core::Tensor& storage) {
            std::lock_guard<std::mutex> lock(mutex_);
            if (index < slots_.size() && slots_[index].in_use) {
                if (index < capacity_limit_)
                    slots_[index].storage = storage;
                else
                    slots_[index].storage = lfs::core::Tensor();
                slots_[index].in_use = false;
                if (live_count_ > 0)
                    --live_count_;
                cv_.notify_one();
            }
        }

        std::vector<std::shared_ptr<Lease>> acquire_batch(const size_t requested) {
            std::unique_lock<std::mutex> lock(mutex_);
            while (running_ && running_->load(std::memory_order_acquire)) {
                const size_t count = std::min(requested, capacity_limit_);
                const size_t free_slots = static_cast<size_t>(std::count_if(
                    slots_.begin(), slots_.begin() + capacity_limit_,
                    [](const Slot& slot) { return !slot.in_use; }));
                if (count > 0 && free_slots >= count && live_count_ + count <= capacity_limit_) {
                    std::vector<std::shared_ptr<Lease>> leases;
                    leases.reserve(count);
                    for (size_t i = 0; i < count; ++i) {
                        const auto it = std::find_if(
                            slots_.begin(), slots_.begin() + capacity_limit_,
                            [](const Slot& slot) { return !slot.in_use; });
                        it->in_use = true;
                        ++live_count_;
                        auto lease = std::make_shared<Lease>();
                        lease->storage = it->storage;
                        lease->owner = shared_from_this();
                        lease->index = static_cast<size_t>(std::distance(slots_.begin(), it));
                        leases.push_back(std::move(lease));
                    }
                    return leases;
                }
                cv_.wait(lock);
            }
            return {};
        }

        void set_capacity(const size_t capacity) {
            std::lock_guard<std::mutex> lock(mutex_);
            capacity_limit_ = std::clamp<size_t>(capacity, 1, slots_.size());
            for (size_t i = capacity_limit_; i < slots_.size(); ++i) {
                if (!slots_[i].in_use)
                    slots_[i].storage = lfs::core::Tensor();
            }
            cv_.notify_all();
        }

        void cancel() { cv_.notify_all(); }

        [[nodiscard]] bool cancelled() const noexcept {
            return !running_ || !running_->load(std::memory_order_acquire);
        }

        void reclaim_idle() {
            std::lock_guard<std::mutex> lock(mutex_);
            for (auto& slot : slots_) {
                if (!slot.in_use)
                    slot.storage = lfs::core::Tensor();
            }
        }

        size_t storage_bytes() const {
            std::lock_guard<std::mutex> lock(mutex_);
            size_t total = 0;
            for (const auto& slot : slots_)
                total += slot.storage.bytes();
            return total;
        }

    private:
        std::vector<Slot> slots_;
        mutable std::mutex mutex_;
        std::condition_variable cv_;
        size_t live_count_ = 0;
        size_t capacity_limit_ = 1;
        const std::atomic<bool>* running_ = nullptr;
    };

} // namespace lfs::io

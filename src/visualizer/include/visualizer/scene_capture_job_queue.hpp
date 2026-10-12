/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include "visualizer/visualizer.hpp"

#include <atomic>
#include <cstdint>
#include <deque>
#include <mutex>
#include <optional>
#include <utility>
#include <vector>

namespace lfs::vis {

    // Each accepted job belongs to one future viewer frame. The frame thread
    // selects one job before rendering, then completes it after the capture gate.
    class SceneCaptureJobQueue {
    public:
        using Job = Visualizer::WorkItem;

        ~SceneCaptureJobQueue() { cancelAll(true); }

        bool post(Job job) {
            const auto observed_epoch = epoch_.load(std::memory_order_acquire);
            {
                std::lock_guard lock(mutex_);
                if (!closed_ && observed_epoch == epoch_.load(std::memory_order_acquire)) {
                    pending_.push_back(std::move(job));
                    has_pending_.store(true, std::memory_order_release);
                    has_work_.store(true, std::memory_order_release);
                    return true;
                }
            }
            return false;
        }

        [[nodiscard]] bool hasPending() const noexcept {
            return has_pending_.load(std::memory_order_acquire);
        }

        [[nodiscard]] bool hasWork() const noexcept {
            return has_work_.load(std::memory_order_acquire);
        }

        // Called on the viewer thread after scene mutations for this frame and
        // before its scene render. Work posted after this point waits for the
        // following frame, including work posted during render callbacks.
        bool beginFrame(const bool has_mesh) {
            if (!hasPending())
                return false;
            std::lock_guard lock(mutex_);
            if (active_ || pending_.empty())
                return false;
            active_ = std::move(pending_.front());
            pending_.pop_front();
            has_pending_.store(!pending_.empty(), std::memory_order_release);
            has_active_.store(true, std::memory_order_release);
            active_has_mesh_.store(has_mesh, std::memory_order_release);
            return true;
        }

        [[nodiscard]] bool hasActive() const noexcept {
            return has_active_.load(std::memory_order_acquire);
        }

        [[nodiscard]] bool needsSceneReadback() const noexcept {
            return active_has_mesh_.load(std::memory_order_acquire);
        }

        void completeFrame() {
            std::optional<Job> job;
            {
                std::lock_guard lock(mutex_);
                job = std::move(active_);
                active_.reset();
                has_active_.store(false, std::memory_order_release);
                active_has_mesh_.store(false, std::memory_order_release);
                has_work_.store(!pending_.empty(), std::memory_order_release);
            }
            if (job && job->run)
                job->run();
        }

        void cancelActive() {
            std::optional<Job> job;
            {
                std::lock_guard lock(mutex_);
                job = std::move(active_);
                active_.reset();
                has_active_.store(false, std::memory_order_release);
                active_has_mesh_.store(false, std::memory_order_release);
                has_work_.store(!pending_.empty(), std::memory_order_release);
            }
            if (job && job->cancel)
                job->cancel();
        }

        // Scene replacement, project close and shutdown settle all waiting
        // callbacks. The GPU request flag is cleared by the owner at the same
        // boundary; no job can leave a later frame with an orphaned request.
        void cancelAll(const bool close = false) {
            std::deque<Job> pending;
            std::optional<Job> active;
            {
                std::lock_guard lock(mutex_);
                epoch_.fetch_add(1, std::memory_order_acq_rel);
                closed_ |= close;
                pending.swap(pending_);
                active = std::move(active_);
                active_.reset();
                has_pending_.store(false, std::memory_order_release);
                has_active_.store(false, std::memory_order_release);
                active_has_mesh_.store(false, std::memory_order_release);
                has_work_.store(false, std::memory_order_release);
            }
            if (active && active->cancel)
                active->cancel();
            for (auto& job : pending) {
                if (job.cancel)
                    job.cancel();
            }
        }

    private:
        mutable std::mutex mutex_;
        std::deque<Job> pending_;
        std::optional<Job> active_;
        std::atomic_bool has_pending_{false};
        std::atomic_bool has_active_{false};
        std::atomic_bool has_work_{false};
        std::atomic_bool active_has_mesh_{false};
        std::atomic<std::uint64_t> epoch_{0};
        bool closed_ = false;
    };

} // namespace lfs::vis

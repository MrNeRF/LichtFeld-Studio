/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include "visualizer/app_store.hpp"

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <mutex>
#include <optional>

namespace lfs::vis {

    // Training progress arrives every ten steps, far more often than a readout
    // needs, and every store change redraws the UI. Progress reaches the store
    // at most once per interval; held-back values are published by flushDue()
    // one interval after the last publish, so the newest values always arrive.
    class TrainingProgressPublisher {
    public:
        using Clock = std::chrono::steady_clock;
        static constexpr std::chrono::milliseconds kInterval{125};

        struct Progress {
            int iteration = 0;
            float loss = 0.0f;
            std::int64_t num_gaussians = 0;
        };

        void offer(const Progress& progress, const Clock::time_point now) {
            std::lock_guard lock(mutex_);
            pending_ = progress;
            publishIfDueLocked(now);
        }

        void flushDue(const Clock::time_point now) {
            std::lock_guard lock(mutex_);
            publishIfDueLocked(now);
        }

        [[nodiscard]] std::optional<double> secondsUntilDue(const Clock::time_point now) const {
            std::lock_guard lock(mutex_);
            if (!pending_) {
                return std::nullopt;
            }
            const auto remaining = last_published_ + kInterval - now;
            return std::max(0.0, std::chrono::duration<double>(remaining).count());
        }

    private:
        void publishIfDueLocked(const Clock::time_point now) {
            if (!pending_ || now - last_published_ < kInterval) {
                return;
            }
            auto& store = app_store();
            lfs::core::reactive::BatchUpdate batch(store.store());
            store.iteration.set(pending_->iteration);
            store.loss.set(pending_->loss);
            store.num_gaussians.set(pending_->num_gaussians);
            pending_.reset();
            last_published_ = now;
        }

        mutable std::mutex mutex_;
        std::optional<Progress> pending_;
        Clock::time_point last_published_{};
    };

} // namespace lfs::vis

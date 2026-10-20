/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include <algorithm>
#include <chrono>

namespace lfs::vis::input {
    // Use wall time: the animation timestep is clamped and OS repeat rates vary.
    class CameraBrowseRepeat {
    public:
        using Clock = std::chrono::steady_clock;
        void start(int key, int direction, Clock::time_point now) {
            key_ = key;
            direction_ = direction;
            last_ = now;
            first_ = true;
        }
        void stop() { direction_ = 0; }
        void release(int key) {
            if (key == key_)
                stop();
        }
        [[nodiscard]] bool active() const { return direction_ != 0; }
        [[nodiscard]] int key() const { return key_; }
        [[nodiscard]] int poll(Clock::time_point now, float rate) {
            if (!active())
                return 0;
            const double interval = 1.0 / std::clamp(rate, 1.0f, 20.0f);
            const double delay = first_ ? std::max(0.35, interval) : interval;
            if (std::chrono::duration<double>(now - last_).count() < delay)
                return 0;
            last_ += std::chrono::duration_cast<Clock::duration>(std::chrono::duration<double>(delay));
            if (std::chrono::duration<double>(now - last_).count() >= interval)
                last_ = now;
            first_ = false;
            // Keep fractional frame time, but never replay a stalled backlog.
            return direction_;
        }

    private:
        int key_ = -1;
        int direction_ = 0;
        bool first_ = true;
        Clock::time_point last_{};
    };
} // namespace lfs::vis::input

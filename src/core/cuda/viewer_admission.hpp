/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include <algorithm>
#include <chrono>
#include <cstdint>

namespace lfs::core {

    // Host-only forecast of a navigating viewer's next scratch request. The
    // acquire-to-request period excludes the admission wait, so late training
    // steps cannot teach the predictor to accept an ever slower viewport.
    class ViewerAdmission {
    public:
        using Clock = std::chrono::steady_clock;
        using Time = Clock::time_point;
        static constexpr auto grace = std::chrono::milliseconds(25);

        void prepare(const Time now) {
            if (acquired_ != Time{}) {
                const double sample = std::chrono::duration<double, std::milli>(now - acquired_).count();
                if (sample > 0.0 && sample <= 500.0) {
                    // Follow a faster viewport immediately, and smooth slower
                    // frames so one delayed present does not admit extra work.
                    period_ms_ = period_ms_ == 0.0 ? sample : std::min(sample, 0.9 * period_ms_ + 0.1 * sample);
                } else {
                    period_ms_ = 0.0;
                }
            }
            preparing_ = true;
            expires_ = now + grace;
        }

        void acquired(const Time now) {
            if (!preparing_)
                return;
            preparing_ = false;
            acquired_ = now;
            due_ = now + std::chrono::duration_cast<Clock::duration>(
                             std::chrono::duration<double, std::milli>(period_ms_));
            expires_ = due_ + grace;
        }

        [[nodiscard]] bool predicted(const Time now) const {
            return !preparing_ && period_ms_ > 0.0 && now < expires_;
        }

        [[nodiscard]] bool admits(const Time now, const double step_ms, const uint32_t owed) const {
            if (owed != 0 || now >= expires_)
                return true;
            if (preparing_ || !(step_ms > 0.0))
                return false;
            const double remaining = std::chrono::duration<double, std::milli>(due_ - now).count();
            // Leave room for normal step jitter and the next frame's CPU work.
            return remaining >= step_ms * 1.05 + 2.0;
        }

        [[nodiscard]] Time expires() const { return expires_; }

    private:
        Time acquired_{};
        Time due_{};
        Time expires_{};
        double period_ms_ = 0.0;
        bool preparing_ = false;
    };

} // namespace lfs::core

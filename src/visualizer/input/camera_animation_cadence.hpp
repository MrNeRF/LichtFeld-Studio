/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include <algorithm>
#include <chrono>
#include <optional>

namespace lfs::vis {

    // Autonomous camera motion owns this deadline; it never creates demand.
    // The frame loop bypasses it for fresh input. Slow frames and the first
    // animation update after a pause need no wait.
    class CameraAnimationCadence {
    public:
        using Clock = std::chrono::steady_clock;

        [[nodiscard]] double secondsUntilReady(Clock::time_point now, double display_interval) const {
            if (!last_frame_)
                return 0.0;
            return std::max(0.0, display_interval - std::chrono::duration<double>(now - *last_frame_).count());
        }

        void noteFrame(Clock::time_point started_at) { last_frame_ = started_at; }

    private:
        std::optional<Clock::time_point> last_frame_;
    };

} // namespace lfs::vis

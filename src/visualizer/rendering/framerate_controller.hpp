/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include "core/export.hpp"

#include <chrono>
#include <deque>
#include <mutex>

namespace lfs::vis {

    struct FramerateSettings {
        float time_window_seconds = 1.0f; // Trailing timestamp window (seconds)
        // Passive live-preview splat re-render cadence while training (UI panels keep
        // full rate; retained last splat image is shown between ticks). A few Hz:
        // enough to feel live, low enough that step-boundary lock cost stays <<5%.
        // Derived as a fixed runtime constant — no dataset/resolution knobs.
        float training_frame_refresh_time_sec = 0.25f; // 4 Hz
    };

    class FramerateController {
    public:
        using Clock = std::chrono::steady_clock;

        const FramerateSettings& getSettings() const { return settings_; }
        LFS_VIS_API void beginFrame(Clock::time_point now = Clock::now());
        LFS_VIS_API float getAverageFPS(Clock::time_point now = Clock::now()) const;

    private:
        void prune(Clock::time_point now) const;

        FramerateSettings settings_;
        mutable std::mutex mutex_;
        mutable std::deque<Clock::time_point> frames_;
    };

} // namespace lfs::vis

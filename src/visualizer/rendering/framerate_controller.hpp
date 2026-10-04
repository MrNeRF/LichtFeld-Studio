/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include "core/export.hpp"
#include "frame_demand.hpp"

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

    struct FrameRates {
        float ui = 0.0f;
        float view = 0.0f;
    };

    // The idle-clear presentation updates the readouts without counting itself
    // as activity. Keep the deadline until a presentation actually succeeds.
    class FrameRateTracker {
    public:
        using Clock = FramerateController::Clock;
        FrameRates sample(Clock::time_point now = Clock::now()) const {
            return {ui_.getAverageFPS(now), view_.getAverageFPS(now)};
        }
        void countView(Clock::time_point now = Clock::now()) { view_.beginFrame(now); }
        void countPresented(const FramePlan& plan, Clock::time_point now = Clock::now()) {
            auto activity = plan.reasons;
            activity.reset(static_cast<std::size_t>(FrameReason::FpsIdle));
            if (activity.any()) {
                ui_.beginFrame(now);
                idle_due_ = now + std::chrono::duration_cast<Clock::duration>(
                                      std::chrono::duration<float>(ui_.getSettings().time_window_seconds));
            } else if (plan.reasons.test(static_cast<std::size_t>(FrameReason::FpsIdle))) {
                idle_due_.reset();
            }
        }
        std::optional<Clock::time_point> idleDeadline() const { return idle_due_; }
        bool idleDue(Clock::time_point now) const { return idle_due_ && now >= *idle_due_; }

    private:
        FramerateController ui_, view_;
        std::optional<Clock::time_point> idle_due_;
    };

} // namespace lfs::vis

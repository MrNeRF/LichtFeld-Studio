/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "framerate_controller.hpp"

namespace lfs::vis {

    void FramerateController::beginFrame(const Clock::time_point now) {
        std::lock_guard lock(mutex_);
        prune(now);
        frames_.push_back(now);
    }

    float FramerateController::getAverageFPS(const Clock::time_point now) const {
        std::lock_guard lock(mutex_);
        prune(now);
        return static_cast<float>(frames_.size()) / settings_.time_window_seconds;
    }

    void FramerateController::prune(const Clock::time_point now) const {
        const auto window = std::chrono::duration<float>(settings_.time_window_seconds);
        while (!frames_.empty() && now - frames_.front() >= window)
            frames_.pop_front();
    }

} // namespace lfs::vis

/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#pragma once

#include "visualizer/preferences.hpp"
#include <algorithm>
#include <cmath>

namespace lfs::vis::input {
    inline float trackpadSpeedFactor(const float level) {
        return std::exp2((level - 50.0f) / 25.0f);
    }

    inline bool trackpadSwipe(const TrackpadPreferenceState& preferences, const int touches) {
        return preferences.device == NavigationDevice::Trackpad ||
               (preferences.device == NavigationDevice::Automatic && touches >= 2);
    }

    inline float swipePixels(const float speed) { return 10.0f * trackpadSpeedFactor(speed); }
    inline float swipeZoomFactor(const float delta, const float speed) {
        return std::exp(delta * 0.05f * trackpadSpeedFactor(speed));
    }
    inline float pinchZoomFactor(const float scale, const float speed) {
        return std::pow(scale, 2.0f * trackpadSpeedFactor(speed));
    }
    inline float wheelZoomFraction(const float delta, const float speed) { return delta * speed * 0.01f; }
    inline float wheelZoomFactor(const float delta, const float speed) {
        return 1.0f / std::max(0.01f, 1.0f - wheelZoomFraction(delta, speed));
    }
} // namespace lfs::vis::input

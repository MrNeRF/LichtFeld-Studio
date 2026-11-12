/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#pragma once

#include "visualizer/preferences.hpp"
#include <algorithm>
#include <cmath>
#include <glm/glm.hpp>
#include <utility>

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

    // Turns an OS trackpad gesture's running transform (total scale and
    // translation since the gesture began) into the pan and pinch since the
    // last take(). A pinch is recognized a few updates in and its translation
    // is not a pan, so once a gesture pinches only its scale counts.
    class TrackpadGestureTracker {
    public:
        struct Motion {
            glm::vec2 pan{0.0f}; // content movement, y down
            float pinch = 1.0f;  // scale change, > 1 zooms in
        };

        void update(const float scale, const glm::vec2 translation) {
            if (!(scale > 0.0f) || !std::isfinite(scale) || !std::isfinite(translation.x) ||
                !std::isfinite(translation.y))
                return;
            if (!pinching_ && std::abs(scale - 1.0f) > kPinchEpsilon)
                pinching_ = true;
            if (pinching_)
                pending_.pinch *= scale / last_scale_;
            else
                pending_.pan += translation - last_translation_;
            last_scale_ = scale;
            last_translation_ = translation;
        }

        // The next gesture starts again from the identity transform.
        void endGesture() {
            pinching_ = false;
            last_scale_ = 1.0f;
            last_translation_ = glm::vec2(0.0f);
        }

        [[nodiscard]] Motion take() { return std::exchange(pending_, Motion{}); }

    private:
        static constexpr float kPinchEpsilon = 1e-3f;
        bool pinching_ = false;
        float last_scale_ = 1.0f;
        glm::vec2 last_translation_{0.0f};
        Motion pending_{};
    };
} // namespace lfs::vis::input

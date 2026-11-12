/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include "input/navigation_gestures.hpp"
#include <functional>
#include <memory>

struct SDL_Window;

namespace lfs::vis {

    // Reads precision-touchpad pan and pinch through Windows Direct Manipulation.
    // Without it, Windows delivers two-finger swipes as emulated scroll-wheel
    // messages that lock to the horizontal or vertical axis until the motion
    // strays far enough; Direct Manipulation reports both axes from the start.
    // Elsewhere every call is a no-op.
    class Win32Trackpad {
    public:
        // pan is in logical pixels.
        using Motion = input::TrackpadGestureTracker::Motion;

        Win32Trackpad();
        ~Win32Trackpad();
        Win32Trackpad(const Win32Trackpad&) = delete;
        Win32Trackpad& operator=(const Win32Trackpad&) = delete;

        // wants_gesture runs when a touchpad gesture starts: true captures it
        // for navigation, false leaves it to arrive as scroll-wheel input.
        // wake must make a blocked event wait return.
        bool init(SDL_Window* window, std::function<bool()> wants_gesture, std::function<void()> wake);
        void shutdown();

        // Advances Direct Manipulation. Call on the window's thread after pumping events.
        void update();
        // A captured gesture is in progress, so the event loop must keep calling update().
        [[nodiscard]] bool active() const;
        Motion takeMotion();

    private:
        struct Impl;
        std::unique_ptr<Impl> impl_;
    };

} // namespace lfs::vis

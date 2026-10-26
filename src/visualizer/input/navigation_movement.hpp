/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include <cmath>
#include <glm/glm.hpp>

namespace lfs::vis::input {

    // Local axes are right, up, forward (positive forward follows the view).
    // Analog input is bounded without promoting partial deflection to full speed.
    // Keyboard input retains its historical diagonal and held-key semantics.
    class NavigationMovement {
    public:
        NavigationMovement() = default;

        [[nodiscard]] static NavigationMovement fromKeys(bool forward, bool backward, bool left,
                                                         bool right, bool up, bool down) {
            return NavigationMovement({float(right) - float(left), float(up) - float(down),
                                       float(forward) - float(backward)},
                                      forward || backward || left || right || up || down);
        }

        [[nodiscard]] static NavigationMovement fromAnalog(glm::vec3 axes) {
            if (!std::isfinite(axes.x) || !std::isfinite(axes.y) || !std::isfinite(axes.z))
                return {};
            // Clamp before computing the length so even oversized finite values
            // cannot overflow normalization.
            axes = glm::clamp(axes, glm::vec3(-1), glm::vec3(1));
            const float length = glm::length(axes);
            if (length > 1.0f)
                axes /= length;
            return NavigationMovement(axes, axes != glm::vec3(0));
        }

        [[nodiscard]] float right() const { return axes_.x; }
        [[nodiscard]] float up() const { return axes_.y; }
        [[nodiscard]] float forward() const { return axes_.z; }
        [[nodiscard]] bool held() const { return held_; }
        [[nodiscard]] bool active() const { return axes_ != glm::vec3(0); }

    private:
        NavigationMovement(glm::vec3 axes, bool held) : axes_(axes), held_(held) {}
        glm::vec3 axes_{0};
        bool held_ = false;
    };

} // namespace lfs::vis::input

/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include "rendering/render_constants.hpp"
#include <cstdint>
#include <glm/glm.hpp>
#include <glm/gtc/quaternion.hpp>

namespace lfs::sequencer {

    inline constexpr glm::quat IDENTITY_ROTATION{1, 0, 0, 0};

    enum class EasingType : uint8_t {
        LINEAR,
        EASE_IN,
        EASE_OUT,
        EASE_IN_OUT
    };

    struct Keyframe {
        float time = 0.0f;
        glm::vec3 position{0.0f};
        glm::quat rotation = IDENTITY_ROTATION;
        float focal_length_mm = rendering::DEFAULT_FOCAL_LENGTH_MM;
        EasingType easing = EasingType::LINEAR;
        bool is_loop_point = false; // mirrors first keyframe for seamless looping

        [[nodiscard]] bool operator<(const Keyframe& other) const { return time < other.time; }
    };

    struct CameraState {
        glm::vec3 position{0.0f};
        glm::quat rotation = IDENTITY_ROTATION;
        float focal_length_mm = rendering::DEFAULT_FOCAL_LENGTH_MM;
    };

} // namespace lfs::sequencer

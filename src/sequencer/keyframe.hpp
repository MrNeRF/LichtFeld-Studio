/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include <cstdint>
#include <glm/glm.hpp>
#include <glm/gtc/quaternion.hpp>

namespace lfs::sequencer {

    inline constexpr float DEFAULT_FOV = 45.0f;
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
        float fov = DEFAULT_FOV;
        EasingType easing = EasingType::LINEAR;
        bool is_loop_point = false; // Loop closure keyframe (copies first keyframe)

        [[nodiscard]] bool operator<(const Keyframe& other) const { return time < other.time; }
    };

    struct CameraState {
        glm::vec3 position{0.0f};
        glm::quat rotation = IDENTITY_ROTATION;
        float fov = DEFAULT_FOV;
    };

} // namespace lfs::sequencer

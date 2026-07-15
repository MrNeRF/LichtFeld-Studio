/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include "rendering/render_constants.hpp"
#include <cstdint>
#include <glm/glm.hpp>
#include <glm/gtc/quaternion.hpp>

namespace lfs::sequencer {

    using KeyframeId = uint64_t;
    inline constexpr KeyframeId INVALID_KEYFRAME_ID = 0;
    inline constexpr glm::quat IDENTITY_ROTATION{1, 0, 0, 0};
    inline constexpr float MAX_SEQUENCER_TIME_SECONDS = 7.0f * 24.0f * 60.0f * 60.0f;

    enum class EasingType : uint8_t {
        LINEAR,
        EASE_IN,
        EASE_OUT,
        EASE_IN_OUT
    };

    [[nodiscard]] inline constexpr bool isValidEasingType(const EasingType easing) noexcept {
        return easing >= EasingType::LINEAR && easing <= EasingType::EASE_IN_OUT;
    }

    struct Keyframe {
        KeyframeId id = INVALID_KEYFRAME_ID;
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

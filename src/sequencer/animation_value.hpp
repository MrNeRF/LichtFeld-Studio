/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include <cstdint>
#include <glm/glm.hpp>
#include <glm/gtc/quaternion.hpp>
#include <variant>

namespace lfs::sequencer {

    inline constexpr float MAX_ANIMATION_VALUE_MAGNITUDE = 1.0e12f;
    inline constexpr float MIN_ANIMATION_QUATERNION_NORM_SQUARED = 1.0e-12f;

    enum class ValueType : uint8_t { Bool,
                                     Int,
                                     Float,
                                     Vec2,
                                     Vec3,
                                     Vec4,
                                     Quat,
                                     Mat4 };

    using AnimationValue = std::variant<bool, int, float, glm::vec2, glm::vec3, glm::vec4, glm::quat, glm::mat4>;

    [[nodiscard]] inline ValueType getValueType(const AnimationValue& value) {
        return static_cast<ValueType>(value.index());
    }

    [[nodiscard]] AnimationValue interpolateValue(const AnimationValue& a, const AnimationValue& b, float t);

} // namespace lfs::sequencer

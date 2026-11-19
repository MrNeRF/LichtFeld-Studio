/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include <array>
#include <cmath>
#include <cstddef>
#include <glm/glm.hpp>

namespace lfs::vis::vr {

    // The joints of XR_EXT_hand_tracking, in its order: palm, wrist, four of
    // the thumb, then five each of the index, middle, ring and little fingers,
    // from the wrist out to the tip.
    inline constexpr std::size_t kHandJointCount = 26;

    // Joint positions of one hand, in meters.
    using HandJoints = std::array<glm::vec3, kHandJointCount>;

    // A corner of a triangle of a hand, as it is drawn.
    struct HandVertex {
        glm::vec3 position{0.0f};
        glm::vec3 normal{0.0f};
    };

    namespace hand_detail {
        inline constexpr std::size_t kWrist = 1;
        // First joint of the thumb, its metacarpal.
        inline constexpr std::size_t kThumb = 2;
        // Metacarpal joint of the index, middle, ring and little fingers.
        inline constexpr std::array<std::size_t, 4> kFingers = {6, 11, 16, 21};

        // Two unit vectors perpendicular to a unit axis and to each other.
        inline void basis(const glm::vec3 axis, glm::vec3& u, glm::vec3& v) {
            const glm::vec3 other = std::abs(axis.y) < 0.9f ? glm::vec3(0.0f, 1.0f, 0.0f) : glm::vec3(1.0f, 0.0f, 0.0f);
            u = glm::normalize(glm::cross(axis, other));
            v = glm::cross(axis, u);
        }
    } // namespace hand_detail

    [[nodiscard]] inline bool handJointsValid(const HandJoints& joints) {
        for (const glm::vec3& joint : joints) {
            if (!std::isfinite(joint.x) || !std::isfinite(joint.y) || !std::isfinite(joint.z))
                return false;
        }
        return true;
    }

} // namespace lfs::vis::vr

/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include <algorithm>
#include <cmath>
#include <glm/gtc/quaternion.hpp>
#include <rendering/coordinate_conventions.hpp>
#include <stdexcept>

namespace lfs::vis::vr {

    // Signed angles in radians: left/down < right/up, all strictly within
    // (-pi/2, pi/2). Pixel coordinates have their origin at the top left.
    struct EyeFov {
        float left, right, up, down;
    };

    [[nodiscard]] inline rendering::CameraIntrinsics eyeIntrinsics(const EyeFov fov,
                                                                   const glm::ivec2 size) {
        const auto valid_angle = [](const float angle) {
            return std::isfinite(angle) && std::abs(angle) < glm::half_pi<float>();
        };
        if (size.x <= 0 || size.y <= 0 ||
            !valid_angle(fov.left) || !valid_angle(fov.right) ||
            !valid_angle(fov.up) || !valid_angle(fov.down) ||
            fov.left >= fov.right || fov.down >= fov.up) {
            throw std::invalid_argument("Invalid VR eye FOV or image size");
        }
        const float left = std::tan(fov.left);
        const float right = std::tan(fov.right);
        const float up = std::tan(fov.up);
        const float down = std::tan(fov.down);
        const float fx = static_cast<float>(size.x) / (right - left);
        const float fy = static_cast<float>(size.y) / (up - down);
        return {.focal_x = fx, .focal_y = fy, .center_x = -left * fx, .center_y = up * fy};
    }

    namespace detail {
        [[nodiscard]] inline glm::quat normalizedOrientation(const glm::quat orientation) {
            const float length = glm::length(orientation);
            return std::isfinite(length) && length > 1e-6f
                       ? orientation / length
                       : glm::quat(1.0f, 0.0f, 0.0f, 0.0f);
        }
    } // namespace detail

    // Camera-to-world matrix, RH +X right, +Y up, -Z forward (also tracking axes).
    // GLM quaternion construction is (w, x, y, z), unlike XrQuaternionf.
    [[nodiscard]] inline glm::mat4 poseMatrix(const glm::quat orientation, const glm::vec3 position) {
        glm::mat4 pose = glm::mat4_cast(detail::normalizedOrientation(orientation));
        pose[3] = glm::vec4(position, 1.0f);
        return pose;
    }

    // Remove pitch/roll, preserving projected forward heading about world +Y.
    // Looking straight up/down uses the projected right axis; invalid/zero input
    // yields identity. The result is always a normalized quaternion.
    [[nodiscard]] inline glm::quat yawOnlyOrientation(const glm::quat orientation) {
        const glm::quat normalized = detail::normalizedOrientation(orientation);
        const glm::vec3 forward = normalized * glm::vec3(0.0f, 0.0f, -1.0f);
        float yaw;
        if (forward.x * forward.x + forward.z * forward.z > 1e-12f) {
            yaw = std::atan2(-forward.x, -forward.z);
        } else {
            const glm::vec3 right = normalized * glm::vec3(1.0f, 0.0f, 0.0f);
            yaw = std::atan2(-right.z, right.x);
        }
        return glm::angleAxis(yaw, glm::vec3(0.0f, 1.0f, 0.0f));
    }

    // Tracking-to-visualizer-world rig. Position is in scene units; tracked
    // positions are in meters. Keep orientation yaw-only for level navigation
    // (arbitrary rig orientation is supported). Scale must be finite and > 0.
    struct NavigationRig {
        glm::vec3 position{0.0f};
        glm::quat orientation{1.0f, 0.0f, 0.0f, 0.0f};
        float scene_units_per_meter = 1.0f;

        [[nodiscard]] rendering::CameraPose cameraPose(const glm::quat tracked_orientation,
                                                       const glm::vec3 tracked_position) const {
            const glm::quat rig = detail::normalizedOrientation(orientation);
            return {.rotation = glm::mat3_cast(rig * detail::normalizedOrientation(tracked_orientation)),
                    .translation = position + rig * (tracked_position * scene_units_per_meter)};
        }

        // Positive radians turn left about WORLD +Y. Compensate rig translation
        // so the actual tracked head center remains fixed, even away from origin.
        void snapTurn(const float radians, const glm::vec3 tracked_head_position) {
            const glm::vec3 head_world = cameraPose(glm::quat(1.0f, 0.0f, 0.0f, 0.0f),
                                                    tracked_head_position)
                                             .translation;
            orientation = detail::normalizedOrientation(
                glm::angleAxis(radians, glm::vec3(0.0f, 1.0f, 0.0f)) *
                detail::normalizedOrientation(orientation));
            position = head_world - orientation * (tracked_head_position * scene_units_per_meter);
        }
    };

    // Rescaled deadzones: threshold in [0,1), output in [-1,1] or unit disk.
    // Out-of-range finite inputs saturate; nonfinite inputs yield zero.
    [[nodiscard]] inline float deadzone(const float value, const float threshold) {
        if (!std::isfinite(value) || !std::isfinite(threshold) || threshold >= 1.0f)
            return 0.0f;
        const float zone = std::max(0.0f, threshold);
        const float magnitude = std::min(std::abs(value), 1.0f);
        if (magnitude <= zone)
            return 0.0f;
        return std::copysign((magnitude - zone) / (1.0f - zone), value);
    }

    [[nodiscard]] inline glm::vec2 deadzone(const glm::vec2 value, const float threshold) {
        const float magnitude = glm::length(value);
        if (!std::isfinite(magnitude) || magnitude <= 1e-6f)
            return glm::vec2(0.0f);
        return value / magnitude * deadzone(magnitude, threshold);
    }

    // One signed pulse at engagement, then no pulse until |axis| <= release.
    // Switching directly to the opposite side does not re-arm. Invalid input
    // behaves as neutral. Throws for thresholds not satisfying 0<=release<engage<=1.
    class SnapTurnHysteresis {
    public:
        explicit SnapTurnHysteresis(const float engage = 0.7f, const float release = 0.3f)
            : engage_(engage),
              release_(release) {
            if (!std::isfinite(engage) || !std::isfinite(release) ||
                release < 0.0f || engage > 1.0f || release >= engage)
                throw std::invalid_argument("Invalid snap-turn hysteresis thresholds");
        }

        [[nodiscard]] int update(const float axis) {
            if (!std::isfinite(axis) || std::abs(axis) <= release_) {
                reset();
                return 0;
            }
            if (!armed_ || std::abs(axis) < engage_)
                return 0;
            armed_ = false;
            return axis > 0.0f ? 1 : -1;
        }

        void reset() { armed_ = true; }

    private:
        float engage_;
        float release_;
        bool armed_ = true;
    };

} // namespace lfs::vis::vr

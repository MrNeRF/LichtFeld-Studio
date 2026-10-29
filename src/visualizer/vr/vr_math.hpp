/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include <algorithm>
#include <cmath>
#include <glm/gtc/quaternion.hpp>
#include <optional>
#include <rendering/coordinate_conventions.hpp>

namespace lfs::vis::vr {

    // Signed angles in radians: left/down < right/up, all strictly within
    // (-pi/2, pi/2). Pixel coordinates have their origin at the top left.
    struct EyeFov {
        float left, right, up, down;
    };

    [[nodiscard]] inline std::optional<rendering::CameraIntrinsics> eyeIntrinsics(const EyeFov fov,
                                                                                  const glm::ivec2 size) {
        const auto valid_angle = [](const float angle) {
            return std::isfinite(angle) && std::abs(angle) < glm::half_pi<float>();
        };
        if (size.x <= 0 || size.y <= 0 ||
            !valid_angle(fov.left) || !valid_angle(fov.right) ||
            !valid_angle(fov.up) || !valid_angle(fov.down) ||
            fov.left >= fov.right || fov.down >= fov.up) {
            return std::nullopt;
        }
        const float left = std::tan(fov.left);
        const float right = std::tan(fov.right);
        const float up = std::tan(fov.up);
        const float down = std::tan(fov.down);
        const float fx = static_cast<float>(size.x) / (right - left);
        const float fy = static_cast<float>(size.y) / (up - down);
        return rendering::CameraIntrinsics{.focal_x = fx, .focal_y = fy, .center_x = -left * fx, .center_y = up * fy};
    }

    namespace detail {
        [[nodiscard]] inline glm::quat normalizedOrientation(const glm::quat orientation) {
            const float length = glm::length(orientation);
            return std::isfinite(length) && length > 1e-6f
                       ? orientation / length
                       : glm::quat(1.0f, 0.0f, 0.0f, 0.0f);
        }
    } // namespace detail

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

    // Tracking-to-visualizer-world rig, RH +X right, +Y up, -Z forward. Position
    // is in scene units; tracked positions are in meters. Scale must be > 0.
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

        // Places the tracked head at the target position facing the target's
        // heading; tracked pitch/roll stay physical head motion.
        void recenter(const glm::quat head_orientation, const glm::vec3 head_position,
                      const glm::quat target_orientation, const glm::vec3 target_position) {
            orientation = yawOnlyOrientation(target_orientation) * glm::inverse(yawOnlyOrientation(head_orientation));
            position = target_position - orientation * (head_position * scene_units_per_meter);
        }

        // Positive radians turn left about world +Y around the tracked head.
        void snapTurn(const float radians, const glm::vec3 tracked_head_position) {
            const glm::vec3 head_world = position + detail::normalizedOrientation(orientation) *
                                                        (tracked_head_position * scene_units_per_meter);
            orientation = detail::normalizedOrientation(
                glm::angleAxis(radians, glm::vec3(0.0f, 1.0f, 0.0f)) * detail::normalizedOrientation(orientation));
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

    // One signed pulse when |axis| reaches 0.7, then none until |axis| <= 0.3.
    // Switching directly to the opposite side does not re-arm. Invalid input
    // behaves as neutral.
    class SnapTurnHysteresis {
    public:
        [[nodiscard]] int update(const float axis) {
            if (!std::isfinite(axis) || std::abs(axis) <= kRelease) {
                reset();
                return 0;
            }
            if (!armed_ || std::abs(axis) < kEngage)
                return 0;
            armed_ = false;
            return axis > 0.0f ? 1 : -1;
        }

        void reset() { armed_ = true; }

    private:
        static constexpr float kEngage = 0.7f;
        static constexpr float kRelease = 0.3f;
        bool armed_ = true;
    };

} // namespace lfs::vis::vr

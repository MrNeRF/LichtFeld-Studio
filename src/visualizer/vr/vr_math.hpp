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

    // Tracking space to Vulkan clip space for an eye at this pose: X right, Y
    // down, depth from 0 at the near plane to 1 at the far plane. For what is
    // drawn straight onto an eye image, in the space the eye is tracked in.
    [[nodiscard]] inline glm::mat4 eyeViewProjection(const EyeFov fov, const glm::quat orientation,
                                                     const glm::vec3 position, const float near_plane,
                                                     const float far_plane) {
        const float left = std::tan(fov.left);
        const float right = std::tan(fov.right);
        const float up = std::tan(fov.up);
        const float down = std::tan(fov.down);
        const float width = right - left;
        const float height = down - up; // negative: up on the image is -Y
        glm::mat4 projection(0.0f);
        projection[0][0] = 2.0f / width;
        projection[1][1] = 2.0f / height;
        projection[2][0] = (right + left) / width;
        projection[2][1] = (up + down) / height;
        projection[2][2] = -far_plane / (far_plane - near_plane);
        projection[2][3] = -1.0f;
        projection[3][2] = -(far_plane * near_plane) / (far_plane - near_plane);
        const glm::mat3 to_eye = glm::transpose(glm::mat3_cast(detail::normalizedOrientation(orientation)));
        glm::mat4 view(to_eye);
        view[3] = glm::vec4(-(to_eye * position), 1.0f);
        return projection * view;
    }

    // Whether an eye at this pose sees a point: inside its field of view,
    // narrowed on every side by `margin` radians, or widened when negative.
    [[nodiscard]] inline bool eyeSees(const EyeFov fov, const glm::quat orientation, const glm::vec3 position,
                                      const glm::vec3 point, const float margin = 0.0f) {
        const glm::vec3 local = glm::conjugate(detail::normalizedOrientation(orientation)) * (point - position);
        if (!(local.z < 0.0f))
            return false;
        const float across = std::atan2(local.x, -local.z);
        const float upward = std::atan2(local.y, -local.z);
        return across >= fov.left + margin && across <= fov.right - margin &&
               upward >= fov.down + margin && upward <= fov.up - margin;
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

    // A tracked hand holding its grip: position in meters, in the tracking space.
    struct GrabHand {
        glm::vec3 position{0.0f};
        glm::quat orientation{1.0f, 0.0f, 0.0f, 0.0f};
    };

    // Steadies a hand that holds the scene, so tremor and tracking jitter do
    // not shake it. A speed-adaptive low-pass (the One Euro filter of Casiez,
    // Roussel and Vogel, 2012): a hand at rest is smoothed hard, a moving hand
    // is followed closely, which keeps the lag small. Time is in seconds.
    class HandSmoother {
    public:
        // Cutoff of a hand at rest, in Hz.
        static constexpr float kRestCutoff = 1.5f;
        // Hz added to the cutoff per m/s, and per rad/s.
        static constexpr float kPositionGain = 40.0f;
        static constexpr float kOrientationGain = 8.0f;
        // Cutoff of the speed estimates, in Hz.
        static constexpr float kSpeedCutoff = 1.0f;

        HandSmoother() = default;
        // A higher cutoff at rest follows more closely and steadies less.
        explicit HandSmoother(const float rest_cutoff) : rest_cutoff_(rest_cutoff) {}

        // The first hand after a reset, a time step that is not positive and
        // a non-finite hand all pass through and restart the filter.
        [[nodiscard]] GrabHand update(const GrabHand& hand, const float dt) {
            const GrabHand raw{hand.position, detail::normalizedOrientation(hand.orientation)};
            const bool finite = std::isfinite(raw.position.x) && std::isfinite(raw.position.y) &&
                                std::isfinite(raw.position.z);
            if (!steady_ || !finite || !(dt > 0.0f)) {
                reset();
                if (!finite)
                    return hand;
                steady_ = raw_ = raw;
                return raw;
            }
            // Averaging the signed velocities cancels a tremor, which goes back
            // and forth, and keeps a deliberate motion, which goes one way.
            const float speed_blend = blend(kSpeedCutoff, dt);
            velocity_ = glm::mix(velocity_, (raw.position - raw_->position) / dt, speed_blend);
            spin_ = glm::mix(spin_, rotationVector(raw.orientation * glm::inverse(raw_->orientation)) / dt, speed_blend);
            steady_->position = glm::mix(steady_->position, raw.position,
                                         blend(rest_cutoff_ + kPositionGain * glm::length(velocity_), dt));
            steady_->orientation = detail::normalizedOrientation(
                glm::slerp(steady_->orientation, raw.orientation,
                           blend(rest_cutoff_ + kOrientationGain * glm::length(spin_), dt)));
            raw_ = raw;
            return *steady_;
        }

        void reset() {
            steady_.reset();
            raw_.reset();
            velocity_ = spin_ = glm::vec3(0.0f);
        }

    private:
        // Share of the new sample in a first-order low-pass with this cutoff.
        [[nodiscard]] static float blend(const float cutoff_hz, const float dt) {
            const float rate = glm::two_pi<float>() * cutoff_hz * dt;
            return rate / (rate + 1.0f);
        }

        // Axis times angle in radians, along the shorter way round.
        [[nodiscard]] static glm::vec3 rotationVector(glm::quat rotation) {
            if (rotation.w < 0.0f)
                rotation = -rotation;
            const glm::vec3 axis(rotation.x, rotation.y, rotation.z);
            const float sine = glm::length(axis);
            return sine > 1e-6f ? axis * (2.0f * std::atan2(sine, rotation.w) / sine) : axis * 2.0f;
        }

        float rest_cutoff_ = kRestCutoff;
        std::optional<GrabHand> steady_;
        std::optional<GrabHand> raw_;
        glm::vec3 velocity_{0.0f}; // m/s
        glm::vec3 spin_{0.0f};     // rad/s
    };

    // Grip manipulation: the scene behaves like an object held in the hands.
    // One gripped hand drags and turns it; two also scale it about their
    // midpoint and turn it with the line between them. Only the rig moves, so
    // the scene itself is never edited. Each update applies the hands' motion
    // since the previous one, which composes with walking, snap turns and
    // recentering; gripping or releasing a hand only changes what is followed
    // next, so the view never jumps.
    class GrabManipulation {
    public:
        // Scene units per meter, the range of the VR scale preference.
        static constexpr float kMinScale = 0.001f;
        static constexpr float kMaxScale = 1000.0f;
        // Meters. Closer hands follow their midpoint without scaling or turning.
        static constexpr float kMinHandDistance = 0.05f;

        void update(NavigationRig& rig, const std::optional<GrabHand>& left, const std::optional<GrabHand>& right) {
            const std::optional<Grip> grip = gripOf(left, right);
            if (grip && previous_ && grip->hands == previous_->hands) {
                const Grip& from = *previous_;
                const glm::quat rig_orientation = detail::normalizedOrientation(rig.orientation);
                // The world point under the previous pivot stays under the new one.
                const glm::vec3 anchor = rig.position + rig_orientation * (from.pivot * rig.scene_units_per_meter);
                glm::quat orientation = rig_orientation;
                float scale = rig.scene_units_per_meter;
                if (from.frame && grip->frame) {
                    orientation = detail::normalizedOrientation(rig_orientation * *from.frame * glm::inverse(*grip->frame));
                    if (from.span > 0.0f && grip->span > 0.0f)
                        scale = std::clamp(scale * from.span / grip->span, kMinScale, kMaxScale);
                }
                rig.orientation = orientation;
                rig.scene_units_per_meter = scale;
                rig.position = anchor - orientation * (grip->pivot * scale);
            }
            previous_ = grip;
        }

        void reset() { previous_.reset(); }
        [[nodiscard]] bool active() const { return previous_.has_value(); }

    private:
        struct Grip {
            int hands = 0;                  // bit 0 left, bit 1 right
            glm::vec3 pivot{0.0f};          // tracked point that keeps its world point
            std::optional<glm::quat> frame; // tracked orientation the world turns with
            float span = 0.0f;              // meters between the hands; 0 with one hand
        };

        [[nodiscard]] static bool tracked(const std::optional<GrabHand>& hand) {
            return hand && std::isfinite(hand->position.x) && std::isfinite(hand->position.y) &&
                   std::isfinite(hand->position.z);
        }

        [[nodiscard]] static std::optional<Grip> gripOf(const std::optional<GrabHand>& left,
                                                        const std::optional<GrabHand>& right) {
            const bool has_left = tracked(left);
            const bool has_right = tracked(right);
            if (!has_left && !has_right)
                return std::nullopt;
            Grip grip;
            grip.hands = (has_left ? 1 : 0) | (has_right ? 2 : 0);
            if (!(has_left && has_right)) {
                const GrabHand& hand = has_left ? *left : *right;
                grip.pivot = hand.position;
                grip.frame = detail::normalizedOrientation(hand.orientation);
                return grip;
            }
            grip.pivot = (left->position + right->position) * 0.5f;
            const glm::vec3 across = right->position - left->position;
            const float distance = glm::length(across);
            if (!(distance >= kMinHandDistance))
                return grip;
            // The line between the hands aims the scene; their shared up axis
            // rolls it about that line, so twisting both wrists rolls the scene.
            const glm::vec3 x = across / distance;
            const glm::vec3 shared_up = detail::normalizedOrientation(left->orientation) * glm::vec3(0.0f, 1.0f, 0.0f) +
                                        detail::normalizedOrientation(right->orientation) * glm::vec3(0.0f, 1.0f, 0.0f);
            const glm::vec3 up = shared_up - x * glm::dot(shared_up, x);
            const float up_length = glm::length(up);
            if (!(up_length > 1e-3f))
                return grip;
            const glm::vec3 y = up / up_length;
            grip.frame = detail::normalizedOrientation(glm::quat_cast(glm::mat3(x, y, glm::cross(x, y))));
            grip.span = distance;
            return grip;
        }

        std::optional<Grip> previous_;
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

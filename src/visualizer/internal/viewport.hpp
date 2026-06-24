/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once
#include "rendering/coordinate_conventions.hpp"
#include "rendering/render_constants.hpp"
#include <algorithm>
#include <chrono>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>
#define GLM_ENABLE_EXPERIMENTAL
#include <cmath>
#include <glm/gtc/constants.hpp>
#include <glm/gtx/norm.hpp>
#include <iostream>
#include <optional>

class Viewport {
    class CameraMotion {
    public:
        glm::vec2 prePos;
        float zoomSpeed = 11.0f;
        float maxZoomSpeed = 100.0f;
        float rotateSpeed = 0.001f;
        float rotateCenterSpeed = 0.002f;
        float rotateRollSpeed = 0.01f;
        float translateSpeed = 0.0005f;
        float wasdSpeed = 8.0f;
        float maxWasdSpeed = 100.0f;
        bool isOrbiting = false;

        // Multiplicative steps: the ranges span three orders of magnitude, so
        // linear increments would need hundreds of keypresses to traverse them.
        static constexpr float kSpeedStepFactor = 1.2f;

        void increaseWasdSpeed() { wasdSpeed = std::min(wasdSpeed * kSpeedStepFactor, maxWasdSpeed); }
        void decreaseWasdSpeed() { wasdSpeed = std::max(wasdSpeed / kSpeedStepFactor, 1.0f); }
        float getWasdSpeed() const { return wasdSpeed; }
        float getMaxWasdSpeed() const { return maxWasdSpeed; }

        void increaseZoomSpeed() { zoomSpeed = std::min(zoomSpeed * kSpeedStepFactor, maxZoomSpeed); }
        void decreaseZoomSpeed() { zoomSpeed = std::max(zoomSpeed / kSpeedStepFactor, 1.0f); }
        float getZoomSpeed() const { return zoomSpeed; }
        float getMaxZoomSpeed() const { return maxZoomSpeed; }

        // Camera state
        glm::vec3 t = glm::vec3(-5.657f, 3.0f, -5.657f);
        glm::vec3 pivot = glm::vec3(0.0f);
        glm::mat3 R = computeLookAtRotation(t, pivot); // Look at pivot from t
        std::chrono::steady_clock::time_point pivot_set_time{};

        // Home position
        glm::vec3 home_t = glm::vec3(-5.657f, 3.0f, -5.657f);
        glm::vec3 home_pivot = glm::vec3(0.0f);
        glm::mat3 home_R = computeLookAtRotation(home_t, home_pivot);
        bool home_saved = true;

        CameraMotion() = default;

        // Compute camera-to-world rotation that looks from 'from' toward 'to'
        static glm::mat3 computeLookAtRotation(const glm::vec3& from, const glm::vec3& to) {
            return lfs::rendering::makeVisualizerLookAtRotation(from, to);
        }

        void saveHomePosition() {
            home_R = R;
            home_t = t;
            home_pivot = pivot;
            home_saved = true;
        }

        void resetToHome() {
            R = home_R;
            t = home_t;
            pivot = home_pivot;
            resetRollTarget();
            clearTransientMotion();
        }

        void resetRollTarget() { roll_target = 0.0f; }

        // Focus camera on bounding box (accepts focal length in mm)
        void focusOnBounds(const glm::vec3& bounds_min, const glm::vec3& bounds_max,
                           float focal_length_mm = lfs::rendering::DEFAULT_FOCAL_LENGTH_MM,
                           float padding = 1.2f) {
            static constexpr float MIN_BOUNDS_DIAGONAL = 0.001f;

            const glm::vec3 center = (bounds_min + bounds_max) * 0.5f;
            const float diagonal = glm::length(bounds_max - bounds_min);
            if (diagonal < MIN_BOUNDS_DIAGONAL)
                return;

            const float vfov_rad = lfs::rendering::focalLengthToVFovRad(focal_length_mm);
            const float half_fov = vfov_rad * 0.5f;
            const float distance = (diagonal * 0.5f * padding) / std::tan(half_fov);

            const glm::vec3 backward = lfs::rendering::cameraBackward(R);
            t = center + backward * distance;
            pivot = center;
            R = computeLookAtRotation(t, pivot);
            resetRollTarget();
            clearTransientMotion();
        }

        // Record the whole-scene radius (half the bounds diagonal). It scales
        // WASD speed and caps pan distance so navigation tracks splat size; the
        // controller feeds it the trimmed whole-scene extent. 0 clears the cache.
        void setSceneExtent(float radius) {
            if (std::isfinite(radius) && radius >= 0.0f)
                scene_extent_ = radius;
        }

        void rotate(const glm::vec2& pos, bool enforceUpright = false) {
            glm::vec2 delta = pos - prePos;

            float y = -delta.x * rotateSpeed;
            float p = -delta.y * rotateSpeed;
            glm::vec3 upVec = enforceUpright ? glm::vec3(0.0f, 1.0f, 0.0f) : R[1];

            if (enforceUpright) {
                // Clamp pitch short of vertical: at the poles the upright basis
                // (right = forward x world-up) degenerates and the view flips.
                constexpr float MAX_PITCH = glm::radians(89.0f);
                const glm::vec3 fwd = lfs::rendering::cameraForward(R);
                const float current_pitch = std::asin(glm::clamp(fwd.y, -1.0f, 1.0f));
                p = glm::clamp(p, -MAX_PITCH - current_pitch, MAX_PITCH - current_pitch);
            }

            glm::mat3 Ry = glm::mat3(glm::rotate(glm::mat4(1.0f), y, upVec));
            glm::mat3 Rp = glm::mat3(glm::rotate(glm::mat4(1.0f), p, R[0]));
            R = Rp * Ry * R;

            if (enforceUpright) {
                const glm::vec3 forward = lfs::rendering::cameraForward(R);
                const glm::vec3 right = normalizedOr(glm::cross(forward, upVec), R[0]);
                const glm::vec3 up = normalizedOr(glm::cross(-forward, right), R[1]);
                R[0] = right;
                R[1] = up;
                R[2] = -forward;
                roll_target = 0.0f;
            }

            prePos = pos;
        }

        void rotateFpv(const glm::vec2& pos) {
            float distance_to_pivot = glm::length(pivot - t);
            if (!std::isfinite(distance_to_pivot) || distance_to_pivot < 0.1f)
                distance_to_pivot = 5.0f;

            rotate(pos, true);
            updatePivotFromCamera(distance_to_pivot);
        }

        void rotate_roll(float diff) {
            const float ang_rad = diff * rotateRollSpeed;
            R = R * glm::mat3(glm::rotate(glm::mat4(1.0f), ang_rad, glm::vec3(0.0f, 0.0f, 1.0f)));
            // Free-orbit leveling targets this angle instead of zero so a
            // deliberate roll survives subsequent orbiting.
            roll_target = wrapAngle(roll_target + ang_rad);
        }

        void translate(const glm::vec2& pos) { applyPanDrag(pos); }

        void zoom(float delta, bool carry_pivot = false) {
            const glm::vec3 forward = lfs::rendering::cameraForward(R);
            const float distToPivot = glm::length(pivot - t);
            // zoomSpeed is a 0..100 level (default 11) mapped linearly to the
            // fraction of the camera-to-pivot distance covered per scroll unit;
            // level 100 matches the previous fastest setting (full distance).
            constexpr float kZoomFractionPerLevel = 0.01f;
            const float adaptiveSpeed = zoomSpeed * kZoomFractionPerLevel * distToPivot;
            const glm::vec3 movement = delta * adaptiveSpeed * forward;

            t += movement;
            if (carry_pivot) {
                pivot += movement;
                return;
            }

            // Zooming in must never deadlock at the pivot: once the camera
            // reaches the minimum distance, push the pivot ahead instead of
            // stopping. This also recovers a pivot that ended up behind the
            // camera.
            constexpr float kMinDistance = 0.1f;
            if (delta > 0.0f && glm::dot(pivot - t, forward) < kMinDistance) {
                pivot = t + forward * kMinDistance;
            }
        }

        // WASD movement carried by an exponentially-damped velocity: holding a
        // key eases the camera up to wasdSpeed, releasing lets it glide to rest
        // instead of stopping dead, giving a light sense of inertia. Call every
        // frame while hasWasdMomentum() so a released key can decay to zero.
        void advanceWasd(float deltaTime, bool forward, bool backward, bool left,
                         bool right, bool up, bool down, float additional_speed = 0.0f) {
            glm::vec3 dir(0.0f);
            if (forward != backward) {
                const glm::vec3 f = lfs::rendering::cameraForward(R);
                dir += forward ? f : -f;
            }
            if (left != right) {
                const glm::vec3 r = glm::normalize(R * glm::vec3(1, 0, 0));
                dir += right ? r : -r;
            }
            if (up != down) {
                const glm::vec3 u = glm::normalize(R * glm::vec3(0, 1, 0));
                dir += up ? u : -u;
            }

            const float effective_speed = (wasdSpeed + additional_speed) * wasdMoveScale();
            const glm::vec3 target_velocity = dir * effective_speed;
            const float blend = 1.0f - std::exp(-deltaTime * kWasdInertiaRate);
            wasd_velocity = glm::mix(wasd_velocity, target_velocity, blend);

            const float stop_speed = kWasdStopFraction * effective_speed;
            if (glm::length2(dir) < 1e-8f && glm::length2(wasd_velocity) < stop_speed * stop_speed) {
                wasd_velocity = glm::vec3(0.0f);
                return;
            }

            const glm::vec3 movement = wasd_velocity * deltaTime;
            t += movement;
            pivot += movement;
        }

        [[nodiscard]] bool hasWasdMomentum() const { return glm::length2(wasd_velocity) > 0.0f; }
        void clearWasdMomentum() { wasd_velocity = glm::vec3(0.0f); }

        void initScreenPos(const glm::vec2& pos) { prePos = pos; }

        void setPivot(const glm::vec3& new_pivot) {
            pivot = new_pivot;
            pivot_set_time = std::chrono::steady_clock::now();
        }

        glm::vec3 getPivot() const { return pivot; }

        float getSecondsSincePivotSet() const {
            return std::chrono::duration<float>(
                       std::chrono::steady_clock::now() - pivot_set_time)
                .count();
        }

        void updatePivotFromCamera(float distance = 5.0f) {
            const glm::vec3 forward = lfs::rendering::cameraForward(R);
            pivot = t + forward * distance;
        }

        void startRotateAroundCenter(const glm::vec2& pos, float time) {
            prePos = pos;
            orbit_last_time = time;
            isOrbiting = true;
            clearOrbitMomentum();
        }

        void updateRotateAroundCenter(const glm::vec2& pos, float /*time*/) {
            if (isOrbiting)
                applyOrbitDrag(pos, false);
        }

        void updateTrackballRotateAroundCenter(const glm::vec2& pos, float /*time*/) {
            if (isOrbiting)
                applyOrbitDrag(pos, true);
        }

        // Release intentionally keeps the momentum so updateOrbitCoast can ease
        // the view to a stop rather than halting it dead.
        void endRotateAroundCenter() { isOrbiting = false; }

        // Like updateRotateAroundCenter, but also remembers the recent angular
        // motion so the view can coast on release. The rotation still applies
        // immediately; only the post-release coast is new.
        void orbitDrag(const glm::vec2& pos, bool trackball, float time) {
            if (!isOrbiting)
                return;
            const glm::vec2 rotation = applyOrbitDrag(pos, trackball);
            const float sample_time = std::max(time - orbit_last_time, kOrbitMinSampleTime);
            orbit_last_time = time;
            orbit_coast_trackball = trackball;
            orbit_vel_yaw = glm::mix(orbit_vel_yaw, rotation.x / sample_time, kOrbitVelBlend);
            orbit_vel_pitch = glm::mix(orbit_vel_pitch, rotation.y / sample_time, kOrbitVelBlend);
        }

        // Fade stored motion while the button is still held, so pausing before
        // release lets the coast die down rather than flinging on a stale drag.
        void decayOrbitMomentum(float deltaTime) {
            const float decay = std::exp(-deltaTime * kOrbitCoastRate);
            orbit_vel_yaw *= decay;
            orbit_vel_pitch *= decay;
            snapOrbitMomentumToRest();
        }

        // Coast after release by integrating remembered angular velocity while
        // decaying it to zero.
        void updateOrbitCoast(float deltaTime) {
            if (!hasOrbitMomentum())
                return;
            const float decay = std::exp(-deltaTime * kOrbitCoastRate);
            const float coast_time = (1.0f - decay) / kOrbitCoastRate;
            applyOrbitRotation(
                orbit_vel_yaw * coast_time,
                orbit_vel_pitch * coast_time,
                orbit_coast_trackball);
            orbit_vel_yaw *= decay;
            orbit_vel_pitch *= decay;
            snapOrbitMomentumToRest();
        }

        [[nodiscard]] bool hasOrbitMomentum() const {
            return orbit_vel_yaw != 0.0f || orbit_vel_pitch != 0.0f;
        }

        void clearOrbitMomentum() {
            orbit_vel_yaw = 0.0f;
            orbit_vel_pitch = 0.0f;
        }

        // Panning carries a world-space velocity so a released drag glides to
        // rest instead of stopping dead, mirroring the orbit ease-out.
        void startPan(const glm::vec2& pos, float time) {
            prePos = pos;
            pan_last_time = time;
            clearPanMomentum();
        }

        // Applies a world-space pan from the drag delta and remembers the recent
        // motion so the view can coast to rest on release.
        void panDrag(const glm::vec2& pos, float time) {
            const glm::vec3 movement = applyPanDrag(pos);
            const float sample_time = std::max(time - pan_last_time, kPanMinSampleTime);
            pan_last_time = time;
            pan_velocity = glm::mix(pan_velocity, movement / sample_time, kPanVelBlend);
        }

        // Fade stored motion while the button is still held, so pausing before
        // release lets the coast die down rather than flinging on a stale drag.
        void decayPanMomentum(float deltaTime) {
            pan_velocity *= std::exp(-deltaTime * kPanCoastRate);
            snapPanMomentumToRest();
        }

        // Coast after release by integrating remembered velocity while decaying
        // it to zero.
        void updatePanCoast(float deltaTime) {
            if (!hasPanMomentum())
                return;
            const float decay = std::exp(-deltaTime * kPanCoastRate);
            const float coast_time = (1.0f - decay) / kPanCoastRate;
            const glm::vec3 movement = pan_velocity * coast_time;
            t += movement;
            pivot += movement;
            pan_velocity *= decay;
            snapPanMomentumToRest();
        }

        [[nodiscard]] bool hasPanMomentum() const { return glm::length2(pan_velocity) > 0.0f; }
        void clearPanMomentum() { pan_velocity = glm::vec3(0.0f); }

        // Short eased translation toward a target position; orientation and
        // pivot are not touched, so orbiting mid-glide stays consistent.
        void startGlide(const glm::vec3& target) {
            glide_target_t = target;
            glide_time_left = kGlideDuration;
        }

        [[nodiscard]] bool isGliding() const { return glide_time_left > 0.0f; }

        void finishGlide() {
            if (isGliding()) {
                t = glide_target_t;
                glide_time_left = 0.0f;
            }
        }

        void updateGlide(float delta_time) {
            if (!isGliding())
                return;
            glide_time_left = std::max(glide_time_left - delta_time, 0.0f);
            const float blend = 1.0f - std::exp(-delta_time * kGlideRate);
            t = glm::mix(t, glide_target_t, blend);
            if (glide_time_left == 0.0f || glm::length2(glide_target_t - t) < 1e-8f) {
                t = glide_target_t;
                glide_time_left = 0.0f;
            }
        }

        void clearTransientMotion() {
            clearWasdMomentum();
            clearOrbitMomentum();
            clearPanMomentum();
            glide_time_left = 0.0f;
        }

        void setAxisAlignedView(int axis, bool negative) {
            float dist_to_pivot = glm::length(pivot - t);
            if (!std::isfinite(dist_to_pivot) || dist_to_pivot < 0.1f)
                dist_to_pivot = 5.0f;

            R = axisViewRotation(axis, negative);
            const glm::vec3 forward = lfs::rendering::cameraForward(R);
            t = pivot - forward * dist_to_pivot;
            resetRollTarget();
            clearTransientMotion();
        }

        [[nodiscard]] bool snapToNearestAxisView(const float max_angle_degrees,
                                                 int* snapped_axis = nullptr,
                                                 bool* snapped_negative = nullptr) {
            const glm::vec3 forward = glm::normalize(lfs::rendering::cameraForward(R));
            if (!std::isfinite(forward.x) || !std::isfinite(forward.y) || !std::isfinite(forward.z)) {
                return false;
            }

            float best_dot = -1.0f;
            int best_axis = -1;
            bool best_negative = false;

            for (int axis = 0; axis < 3; ++axis) {
                for (const bool negative : {false, true}) {
                    const float dot = glm::dot(forward, axisViewForward(axis, negative));
                    if (dot > best_dot) {
                        best_dot = dot;
                        best_axis = axis;
                        best_negative = negative;
                    }
                }
            }

            const float snap_dot = std::cos(glm::radians(max_angle_degrees));
            if (best_axis < 0 || best_dot < snap_dot) {
                return false;
            }

            setAxisAlignedView(best_axis, best_negative);
            if (snapped_axis) {
                *snapped_axis = best_axis;
            }
            if (snapped_negative) {
                *snapped_negative = best_negative;
            }
            return true;
        }

    private:
        float roll_target = 0.0f;
        glm::vec3 glide_target_t{0.0f};
        float glide_time_left = 0.0f;
        static constexpr float kGlideDuration = 0.35f;
        static constexpr float kGlideRate = 15.0f;

        glm::vec3 wasd_velocity{0.0f};
        static constexpr float kWasdInertiaRate = 12.0f;
        static constexpr float kWasdStopFraction = 0.02f;

        // Whole-scene radius (half the bounds diagonal), fed by the controller.
        // WASD scales by it and panning is capped by it. 0 = unknown, in
        // which case movement keeps its distance-based behavior. Scenes load in
        // arbitrary world units (no normalization), so an absolute speed feels right
        // on one scene and wrong on the next; scaling by this removes that.
        //
        // WASD effective speed is wasdSpeed * radius / kWasdReferenceExtent, with
        // wasdSpeed a 1..100 level (default 8). kWasdReferenceExtent is set so level
        // 50 crosses one scene radius per second; the level-8 default is a calm
        // exploration pace and level 100 covers two radii per second.
        float scene_extent_ = 0.0f;
        static constexpr float kWasdReferenceExtent = 50.0f;
        static constexpr float kMinMoveScale = 0.05f;
        static constexpr float kMaxMoveScale = 100.0f;
        // Cap pan distance against the trimmed scene radius. A minimum tied to
        // scene size makes large scenes pan too fast when the camera/pivot is
        // still close to the origin.
        static constexpr float kPanMaxDistanceFraction = 1.0f;

        // WASD multiplier from the scene radius, clamped so degenerate or enormous
        // bounds can't produce unusable speeds; 1.0 when the extent is unknown.
        float wasdMoveScale() const {
            if (scene_extent_ <= 0.0f)
                return 1.0f;
            return std::clamp(scene_extent_ / kWasdReferenceExtent, kMinMoveScale, kMaxMoveScale);
        }

        float orbit_vel_yaw = 0.0f;
        float orbit_vel_pitch = 0.0f;
        float orbit_last_time = 0.0f;
        bool orbit_coast_trackball = false;
        static constexpr float kOrbitMinSampleTime = 1.0f / 240.0f;
        static constexpr float kOrbitVelBlend = 0.35f;
        static constexpr float kOrbitCoastRate = 18.0f;
        static constexpr float kOrbitStopVelocity = 1e-4f;

        void snapOrbitMomentumToRest() {
            if (std::abs(orbit_vel_yaw) < kOrbitStopVelocity &&
                std::abs(orbit_vel_pitch) < kOrbitStopVelocity) {
                orbit_vel_yaw = 0.0f;
                orbit_vel_pitch = 0.0f;
            }
        }

        void applyOrbitRotation(float yaw, float pitch, bool trackball) {
            if (trackball) {
                applyTrackballRotationAroundCenter(yaw, pitch);
            } else {
                applyRotationAroundCenter(yaw, pitch);
            }
        }

        // Maps a drag delta to an orbit rotation, applies it, and returns the
        // applied (yaw, pitch) so callers can track angular momentum.
        glm::vec2 applyOrbitDrag(const glm::vec2& pos, bool trackball) {
            const glm::vec2 delta = pos - prePos;
            const glm::vec2 rotation(-delta.x * rotateCenterSpeed, -delta.y * rotateCenterSpeed);
            prePos = pos;
            applyOrbitRotation(rotation.x, rotation.y, trackball);
            return rotation;
        }

        glm::vec3 pan_velocity{0.0f};
        float pan_last_time = 0.0f;
        static constexpr float kPanMinSampleTime = 1.0f / 240.0f;
        static constexpr float kPanVelBlend = 0.35f;
        static constexpr float kPanCoastRate = 18.0f;
        static constexpr float kPanStopVelocity = 1e-4f;

        void snapPanMomentumToRest() {
            if (glm::length2(pan_velocity) < kPanStopVelocity * kPanStopVelocity)
                pan_velocity = glm::vec3(0.0f);
        }

        // Maps a drag delta to a world-space translation, applies it to camera
        // and pivot, advances prePos, and returns the applied movement so
        // callers can track momentum.
        glm::vec3 applyPanDrag(const glm::vec2& pos) {
            const glm::vec2 delta = pos - prePos;
            const float dist_to_pivot = glm::length(pivot - t);
            float pan_dist = dist_to_pivot;
            if (scene_extent_ > 0.0f) {
                pan_dist = std::min(pan_dist, scene_extent_ * kPanMaxDistanceFraction);
            }
            const float adaptive_speed = translateSpeed * pan_dist;
            const glm::vec3 movement = -(delta.x * adaptive_speed) * R[0] + (delta.y * adaptive_speed) * R[1];
            t += movement;
            pivot += movement;
            prePos = pos;
            return movement;
        }

        [[nodiscard]] static float wrapAngle(float angle) {
            while (angle > glm::pi<float>())
                angle -= glm::two_pi<float>();
            while (angle < -glm::pi<float>())
                angle += glm::two_pi<float>();
            return angle;
        }

        [[nodiscard]] static glm::vec3 axisViewForward(const int axis, const bool negative) {
            const float sign = negative ? -1.0f : 1.0f;
            switch (axis) {
            case 0: return glm::vec3(-sign, 0.0f, 0.0f);
            case 1: return glm::vec3(0.0f, -sign, 0.0f);
            case 2: return glm::vec3(0.0f, 0.0f, -sign);
            default: return glm::vec3(0.0f, 0.0f, -1.0f);
            }
        }

        [[nodiscard]] static glm::vec3 axisViewUp(const int axis, const bool negative) {
            const float sign = negative ? -1.0f : 1.0f;
            switch (axis) {
            case 0: return glm::vec3(0.0f, 1.0f, 0.0f);
            case 1: return glm::vec3(0.0f, 0.0f, sign);
            case 2: return glm::vec3(0.0f, 1.0f, 0.0f);
            default: return glm::vec3(0.0f, 1.0f, 0.0f);
            }
        }

        [[nodiscard]] static glm::mat3 axisViewRotation(const int axis, const bool negative) {
            return lfs::rendering::makeVisualizerLookAtRotation(
                glm::vec3(0.0f),
                axisViewForward(axis, negative),
                axisViewUp(axis, negative));
        }

        [[nodiscard]] static glm::mat3 orthonormalizeRotation(const glm::mat3& rotation) {
            glm::vec3 right = rotation[0];
            glm::vec3 up = rotation[1];

            if (glm::length2(right) < 1e-10f) {
                right = glm::vec3(1.0f, 0.0f, 0.0f);
            }
            right = glm::normalize(right);

            up -= right * glm::dot(up, right);
            if (glm::length2(up) < 1e-10f) {
                const glm::vec3 fallback_forward = glm::normalize(-rotation[2]);
                const glm::vec3 fallback_up =
                    std::abs(fallback_forward.y) < 0.99f ? glm::vec3(0.0f, 1.0f, 0.0f)
                                                         : glm::vec3(1.0f, 0.0f, 0.0f);
                right = glm::normalize(glm::cross(fallback_forward, fallback_up));
                up = glm::normalize(glm::cross(right, fallback_forward));
                return glm::mat3(right, up, -fallback_forward);
            }
            up = glm::normalize(up);

            glm::vec3 back = glm::cross(right, up);
            if (glm::length2(back) < 1e-10f) {
                back = glm::vec3(0.0f, 0.0f, 1.0f);
            } else {
                back = glm::normalize(back);
            }

            return glm::mat3(right, up, back);
        }

        [[nodiscard]] static glm::vec3 normalizedOr(const glm::vec3& value, const glm::vec3& fallback) {
            const float length = glm::length(value);
            if (!std::isfinite(length) || length <= 1e-6f) {
                return fallback;
            }
            return value / length;
        }

        [[nodiscard]] static glm::mat3 makeRollStableOrbitRotation(const glm::vec3& eye,
                                                                   const glm::vec3& target,
                                                                   const glm::vec3& transported_right,
                                                                   const glm::mat3& fallback_rotation,
                                                                   const float step_angle,
                                                                   const float roll_target_angle) {
            constexpr glm::vec3 WORLD_UP(0.0f, 1.0f, 0.0f);

            const glm::vec3 view = target - eye;
            const float view_length = glm::length(view);
            if (!std::isfinite(view_length) || view_length <= 1e-6f) {
                return orthonormalizeRotation(fallback_rotation);
            }

            const glm::vec3 forward = view / view_length;

            glm::vec3 right = transported_right - forward * glm::dot(transported_right, forward);
            const float transported_length = glm::length(right);
            if (std::isfinite(transported_length) && transported_length > 1e-4f) {
                right /= transported_length;
            } else {
                const glm::vec3 fallback_up = lfs::rendering::chooseFallbackUp(forward);
                right = normalizedOr(glm::cross(forward, fallback_up), glm::vec3(1.0f, 0.0f, 0.0f));
            }

            // Level out roll gradually, at most as fast as the orbit step itself,
            // so leveling is independent of mouse-event granularity and never
            // snaps. cross(forward, WORLD_UP) is ill-conditioned near the poles
            // (its direction swings arbitrarily fast as forward approaches
            // vertical), so the correction fades to pure parallel transport
            // there.
            glm::vec3 level_right = glm::cross(forward, WORLD_UP);
            const float level_length = glm::length(level_right);
            if (level_length > 1e-4f && std::isfinite(step_angle) && step_angle > 0.0f) {
                level_right /= level_length;
                if (glm::dot(level_right, right) < 0.0f) {
                    level_right = -level_right;
                }
                if (roll_target_angle != 0.0f) {
                    // Rolling the camera by +a rotates its right vector by -a
                    // about forward; aim the leveler at the rolled frame.
                    level_right = level_right * std::cos(roll_target_angle) -
                                  glm::cross(forward, level_right) * std::sin(roll_target_angle);
                }

                const float pole_blend = glm::smoothstep(
                    0.95f, 0.999f, std::abs(glm::dot(forward, WORLD_UP)));
                const float max_correction = step_angle * (1.0f - pole_blend);

                const float cos_err = glm::clamp(glm::dot(right, level_right), -1.0f, 1.0f);
                const float sin_err = glm::clamp(glm::dot(forward, glm::cross(right, level_right)), -1.0f, 1.0f);
                const float roll_error = std::atan2(sin_err, cos_err);
                const float correction = glm::clamp(roll_error, -max_correction, max_correction);

                right = right * std::cos(correction) + glm::cross(forward, right) * std::sin(correction);
                right = normalizedOr(right, level_right);
            }

            const glm::vec3 backward = -forward;
            const glm::vec3 up = normalizedOr(glm::cross(backward, right), glm::vec3(0.0f, 1.0f, 0.0f));
            return glm::mat3(right, up, backward);
        }

        void applyRotationAroundCenter(const float yaw, const float pitch) {
            constexpr glm::vec3 WORLD_UP(0.0f, 1.0f, 0.0f);
            // Just short of vertical: keeps the upright re-orthogonalization
            // below well-conditioned (right length >= 0.014) while allowing a
            // near-top-down view.
            constexpr float MAX_VERTICAL_DOT = 0.9999f;
            constexpr float HORIZONTAL_COMPONENT = 0.01414178f; // sqrt(1 - 0.9999^2)

            // Saturate the pitch input at the elevation limit. A fast drag
            // event could otherwise step across the pole in one event and the
            // upright rebuild below would flip the view 180 degrees. From an
            // exact axis pole view (|elevation| > limit) the allowed range
            // keeps only the exit direction whose rebuilt right stays aligned
            // with the current frame, so leaving the pole never flips either.
            const float max_elevation = std::asin(MAX_VERTICAL_DOT);
            const glm::vec3 fwd = lfs::rendering::cameraForward(R);
            const float elevation = std::asin(glm::clamp(fwd.y, -1.0f, 1.0f));
            const float limit = std::max(max_elevation, std::abs(elevation));
            const float clamped_pitch = glm::clamp(pitch, -limit - elevation, limit - elevation);

            // Apply yaw (world Y) and pitch (local right)
            const glm::mat3 Ry = glm::mat3(glm::rotate(glm::mat4(1.0f), yaw, WORLD_UP));
            const glm::mat3 Rp = glm::mat3(glm::rotate(glm::mat4(1.0f), clamped_pitch, R[0]));
            const glm::mat3 U = Rp * Ry;

            // Transform position and orientation
            const float dist = glm::length(t - pivot);
            t = pivot + U * (t - pivot);
            R = U * R;

            // Clamp forward to prevent gimbal lock
            glm::vec3 forward = lfs::rendering::cameraForward(R);
            const float upDot = glm::dot(forward, WORLD_UP);

            if (std::abs(upDot) > MAX_VERTICAL_DOT) {
                const glm::vec3 horizontal = forward - WORLD_UP * upDot;
                const float horizLen = glm::length(horizontal);

                if (horizLen > 1e-4f) {
                    const float sign = upDot > 0.0f ? 1.0f : -1.0f;
                    forward = (horizontal / horizLen) * HORIZONTAL_COMPONENT + WORLD_UP * (sign * MAX_VERTICAL_DOT);
                    t = pivot - forward * dist;
                }
            }

            // Re-orthogonalize to prevent roll drift
            glm::vec3 right = glm::cross(forward, WORLD_UP);
            const float rightLen = glm::length(right);
            right = (rightLen > 1e-2f) ? right / rightLen
                                       : glm::normalize(R[0] - forward * glm::dot(R[0], forward));

            R[0] = right;
            R[1] = glm::normalize(glm::cross(-forward, right));
            R[2] = -forward;
            resetRollTarget();
        }

        void applyTrackballRotationAroundCenter(const float yaw, const float pitch) {
            const glm::vec3 orbit_offset = t - pivot;
            const float orbit_distance = glm::length(orbit_offset);
            if (!std::isfinite(orbit_distance) || orbit_distance <= 1e-6f) {
                R = orthonormalizeRotation(R);
                return;
            }

            const glm::vec3 local_up = normalizedOr(R[1], glm::vec3(0.0f, 1.0f, 0.0f));
            const glm::vec3 local_right = normalizedOr(R[0], glm::vec3(1.0f, 0.0f, 0.0f));
            const glm::mat3 Ry = glm::mat3(glm::rotate(glm::mat4(1.0f), yaw, local_up));
            const glm::vec3 pitch_axis = normalizedOr(Ry * local_right, local_right);
            const glm::mat3 Rp = glm::mat3(glm::rotate(glm::mat4(1.0f), pitch, pitch_axis));
            const glm::mat3 U = Rp * Ry;

            const glm::vec3 rotated_offset = U * orbit_offset;
            const float rotated_distance = glm::length(rotated_offset);
            if (!std::isfinite(rotated_distance) || rotated_distance <= 1e-6f) {
                R = orthonormalizeRotation(R);
                return;
            }

            t = pivot + (rotated_offset / rotated_distance) * orbit_distance;
            R = makeRollStableOrbitRotation(
                t,
                pivot,
                U * local_right,
                U * R,
                std::abs(yaw) + std::abs(pitch),
                roll_target);
        }
    };

public:
    static constexpr float INVALID_WORLD_POS = -1e10f;

    glm::ivec2 windowSize;
    glm::ivec2 frameBufferSize;
    CameraMotion camera;
    std::optional<float> ortho_scale_override;

    Viewport(size_t width = 1280, size_t height = 720) {
        windowSize = glm::ivec2(width, height);
        camera = CameraMotion();
    }

    void setViewMatrix(const glm::mat3& R, const glm::vec3& t) {
        camera.R = R;
        camera.t = t;
        camera.resetRollTarget();
        camera.clearTransientMotion();
    }

    glm::mat3 getRotationMatrix() const {
        return camera.R;
    }

    glm::vec3 getTranslation() const {
        return camera.t;
    }

    glm::mat4 getViewMatrix() const {
        return lfs::rendering::makeViewMatrix(camera.R, camera.t);
    }

    glm::mat4 getProjectionMatrix(float focal_length_mm = lfs::rendering::DEFAULT_FOCAL_LENGTH_MM,
                                  float near_plane = lfs::rendering::DEFAULT_NEAR_PLANE,
                                  float far_plane = lfs::rendering::DEFAULT_FAR_PLANE) const {
        float aspect_ratio = static_cast<float>(windowSize.x) / static_cast<float>(windowSize.y);
        float fov_radians = lfs::rendering::focalLengthToVFovRad(focal_length_mm);
        return glm::perspective(fov_radians, aspect_ratio, near_plane, far_plane);
    }

    [[nodiscard]] static bool isValidWorldPosition(const glm::vec3& world_pos) {
        return std::isfinite(world_pos.x) &&
               std::isfinite(world_pos.y) &&
               std::isfinite(world_pos.z) &&
               (world_pos.x != INVALID_WORLD_POS ||
                world_pos.y != INVALID_WORLD_POS ||
                world_pos.z != INVALID_WORLD_POS);
    }

    // Unproject screen pixel to world position (returns INVALID_WORLD_POS if invalid)
    [[nodiscard]] glm::vec3 unprojectPixel(float screen_x, float screen_y, float depth,
                                           float focal_length_mm = lfs::rendering::DEFAULT_FOCAL_LENGTH_MM,
                                           bool orthographic = false,
                                           float ortho_scale = lfs::rendering::DEFAULT_ORTHO_SCALE) const {
        constexpr float MAX_DEPTH = 1e9f;

        if (depth <= 0.0f || depth > MAX_DEPTH ||
            (orthographic && (!std::isfinite(ortho_scale) || ortho_scale <= 0.0f))) {
            return glm::vec3(INVALID_WORLD_POS);
        }

        return lfs::rendering::unprojectScreenPoint(
            camera.R,
            camera.t,
            windowSize,
            screen_x,
            screen_y,
            depth,
            focal_length_mm,
            orthographic,
            ortho_scale);
    }
};

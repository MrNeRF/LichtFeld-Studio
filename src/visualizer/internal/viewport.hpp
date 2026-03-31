/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once
#include "rendering/coordinate_conventions.hpp"
#include "rendering/render_constants.hpp"
#include <chrono>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>
#define GLM_ENABLE_EXPERIMENTAL
#include <cmath>
#include <glm/gtx/norm.hpp>
#include <iostream>

class Viewport {
    class CameraMotion {
    public:
        glm::vec2 prePos;
        float zoomSpeed = 5.0f;
        float maxZoomSpeed = 100.0f;
        float rotateSpeed = 0.001f;
        float rotateCenterSpeed = 0.002f;
        float rotateRollSpeed = 0.01f;
        float translateSpeed = 0.001f;
        float wasdSpeed = 6.0f;
        float maxWasdSpeed = 100.0f;
        bool isOrbiting = false;

        void increaseWasdSpeed() { wasdSpeed = std::min(wasdSpeed + 1.0f, maxWasdSpeed); }
        void decreaseWasdSpeed() { wasdSpeed = std::max(wasdSpeed - 1.0f, 1.0f); }
        float getWasdSpeed() const { return wasdSpeed; }
        float getMaxWasdSpeed() const { return maxWasdSpeed; }

        void increaseZoomSpeed() { zoomSpeed = std::min(zoomSpeed + 0.1f, maxZoomSpeed); }
        void decreaseZoomSpeed() { zoomSpeed = std::max(zoomSpeed - 0.1f, 0.1f); }
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
        }

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
        }

        void rotate(const glm::vec2& pos, bool enforceUpright = false) {
            glm::vec2 delta = pos - prePos;

            float y = -delta.x * rotateSpeed;
            float p = -delta.y * rotateSpeed;
            glm::vec3 upVec = enforceUpright ? glm::vec3(0.0f, 1.0f, 0.0f) : R[1];

            glm::mat3 Ry = glm::mat3(glm::rotate(glm::mat4(1.0f), y, upVec));
            glm::mat3 Rp = glm::mat3(glm::rotate(glm::mat4(1.0f), p, R[0]));
            R = Rp * Ry * R;

            if (enforceUpright) {
                const glm::vec3 forward = lfs::rendering::cameraForward(R);
                glm::vec3 right = glm::normalize(glm::cross(forward, upVec));
                glm::vec3 up = glm::normalize(glm::cross(-forward, right));
                R[0] = right;
                R[1] = up;
                R[2] = -forward;
            }

            prePos = pos;
        }

        void rotate_roll(float diff) {
            float ang_rad = diff * rotateRollSpeed;
            glm::mat3 rot_z = glm::mat3(
                glm::cos(ang_rad), -glm::sin(ang_rad), 0.0f,
                glm::sin(ang_rad), glm::cos(ang_rad), 0.0f,
                0.0f, 0.0f, 1.0f);
            R = R * rot_z;
        }

        void translate(const glm::vec2& pos) {
            const glm::vec2 delta = pos - prePos;
            const float dist_to_pivot = glm::length(pivot - t);
            const float adaptive_speed = translateSpeed * dist_to_pivot;
            const glm::vec3 movement = -(delta.x * adaptive_speed) * R[0] + (delta.y * adaptive_speed) * R[1];
            t += movement;
            pivot += movement;
            prePos = pos;
        }

        void zoom(float delta) {
            const glm::vec3 forward = lfs::rendering::cameraForward(R);
            const float distToPivot = glm::length(pivot - t);
            const float adaptiveSpeed = zoomSpeed * 0.01f * distToPivot;
            glm::vec3 movement = delta * adaptiveSpeed * forward;

            // Prevent zooming past pivot
            if (delta > 0.0f) {
                const float current_dist = glm::length(pivot - t);
                const float move_dist = glm::length(movement);
                constexpr float kMinDistance = 0.1f;
                if (current_dist - move_dist < kMinDistance) {
                    const float allowed = std::max(0.0f, current_dist - kMinDistance);
                    movement = glm::normalize(forward) * allowed;
                }
            }
            t += movement;
        }

        void advance_forward(float deltaTime, float additional_speed = 0.0f) {
            const glm::vec3 forward = lfs::rendering::cameraForward(R);
            const glm::vec3 movement = forward * deltaTime * (wasdSpeed + additional_speed);
            t += movement;
            pivot += movement;
        }

        void advance_backward(float deltaTime, float additional_speed = 0.0f) {
            const glm::vec3 forward = lfs::rendering::cameraForward(R);
            const glm::vec3 movement = -forward * deltaTime * (wasdSpeed + additional_speed);
            t += movement;
            pivot += movement;
        }

        void advance_left(float deltaTime, float additional_speed = 0.0f) {
            const glm::vec3 right = glm::normalize(R * glm::vec3(1, 0, 0));
            const glm::vec3 movement = -right * deltaTime * (wasdSpeed + additional_speed);
            t += movement;
            pivot += movement;
        }

        void advance_right(float deltaTime, float additional_speed = 0.0f) {
            const glm::vec3 right = glm::normalize(R * glm::vec3(1, 0, 0));
            const glm::vec3 movement = right * deltaTime * (wasdSpeed + additional_speed);
            t += movement;
            pivot += movement;
        }

        void advance_up(float deltaTime, float additional_speed = 0.0f) {
            const glm::vec3 up = glm::normalize(R * glm::vec3(0, 1, 0));
            const glm::vec3 movement = up * deltaTime * (wasdSpeed + additional_speed);
            t += movement;
            pivot += movement;
        }

        void advance_down(float deltaTime, float additional_speed = 0.0f) {
            const glm::vec3 up = glm::normalize(R * glm::vec3(0, 1, 0));
            const glm::vec3 movement = -up * deltaTime * (wasdSpeed + additional_speed);
            t += movement;
            pivot += movement;
        }

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

        // Simplified orbit methods - no velocity tracking
        void startRotateAroundCenter(const glm::vec2& pos, float /*time*/) {
            prePos = pos;
            isOrbiting = true;
        }

        void updateRotateAroundCenter(const glm::vec2& pos, float /*time*/) {
            if (!isOrbiting)
                return;

            glm::vec2 delta = pos - prePos;
            float yaw = -delta.x * rotateCenterSpeed;
            float pitch = -delta.y * rotateCenterSpeed;

            applyRotationAroundCenter(yaw, pitch);
            prePos = pos;
        }

        void endRotateAroundCenter() {
            isOrbiting = false;
            // No velocity to clear
        }

        // No-op since we removed inertia
        void updateInertia(float /*deltaTime*/) {
            // Inertia disabled - do nothing
        }

    private:
        void applyRotationAroundCenter(const float yaw, const float pitch) {
            constexpr glm::vec3 WORLD_UP(0.0f, 1.0f, 0.0f);
            constexpr float MAX_VERTICAL_DOT = 0.98f;
            constexpr float HORIZONTAL_COMPONENT = 0.19899749f; // sqrt(1 - 0.98^2)

            // Apply yaw (world Y) and pitch (local right)
            const glm::mat3 Ry = glm::mat3(glm::rotate(glm::mat4(1.0f), yaw, WORLD_UP));
            const glm::mat3 Rp = glm::mat3(glm::rotate(glm::mat4(1.0f), pitch, R[0]));
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
        }
    };

public:
    static constexpr float INVALID_WORLD_POS = -1e10f;

    glm::ivec2 windowSize;
    glm::ivec2 frameBufferSize;
    CameraMotion camera;

    Viewport(size_t width = 1280, size_t height = 720) {
        windowSize = glm::ivec2(width, height);
        camera = CameraMotion();
    }

    void setViewMatrix(const glm::mat3& R, const glm::vec3& t) {
        camera.R = R;
        camera.t = t;
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
        return world_pos.x != INVALID_WORLD_POS ||
               world_pos.y != INVALID_WORLD_POS ||
               world_pos.z != INVALID_WORLD_POS;
    }

    // Unproject screen pixel to world position (returns INVALID_WORLD_POS if invalid)
    [[nodiscard]] glm::vec3 unprojectPixel(float screen_x, float screen_y, float depth,
                                           float focal_length_mm = lfs::rendering::DEFAULT_FOCAL_LENGTH_MM) const {
        constexpr float MAX_DEPTH = 1e9f;

        if (depth <= 0.0f || depth > MAX_DEPTH) {
            return glm::vec3(INVALID_WORLD_POS);
        }

        return lfs::rendering::unprojectScreenPoint(
            camera.R,
            camera.t,
            windowSize,
            screen_x,
            screen_y,
            depth,
            focal_length_mm);
    }
};

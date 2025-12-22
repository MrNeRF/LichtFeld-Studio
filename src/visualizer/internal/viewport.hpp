/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once
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
        float wasdSpeed = 50.0f;
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
        glm::vec3 t = glm::vec3(-5.657f, -3.0f, -5.657f);
        glm::vec3 pivot = glm::vec3(0.0f);
        glm::mat3 R = computeLookAtRotation(t, pivot); // Look at pivot from t
        std::chrono::steady_clock::time_point pivot_set_time{};

        // Home position
        glm::vec3 home_t = glm::vec3(-5.657f, -3.0f, -5.657f);
        glm::vec3 home_pivot = glm::vec3(0.0f);
        glm::mat3 home_R = computeLookAtRotation(home_t, home_pivot);
        bool home_saved = true;

        CameraMotion() = default;

        // Compute camera-to-world rotation that looks from 'from' toward 'to'
        static glm::mat3 computeLookAtRotation(const glm::vec3& from, const glm::vec3& to) {
            const glm::vec3 forward = glm::normalize(to - from);
            const glm::vec3 world_up(0.0f, 1.0f, 0.0f);
            const glm::vec3 right = glm::normalize(glm::cross(world_up, forward));
            const glm::vec3 up = glm::cross(forward, right);
            return glm::mat3(right, up, forward); // Columns: right, up, forward
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

        // Focus camera on bounding box
        void focusOnBounds(const glm::vec3& bounds_min, const glm::vec3& bounds_max,
                           float fov_degrees = lfs::rendering::DEFAULT_FOV,
                           float padding = 1.2f) {
            static constexpr float MIN_BOUNDS_DIAGONAL = 0.001f;

            const glm::vec3 center = (bounds_min + bounds_max) * 0.5f;
            const float diagonal = glm::length(bounds_max - bounds_min);
            if (diagonal < MIN_BOUNDS_DIAGONAL)
                return;

            const float half_fov = glm::radians(fov_degrees) * 0.5f;
            const float distance = (diagonal * 0.5f * padding) / std::tan(half_fov);

            const glm::vec3 backward = -R[2];
            t = center + backward * distance;
            pivot = center;
            R = computeLookAtRotation(t, pivot);
        }

        void rotate(const glm::vec2& pos, bool enforceUpright = false) {
            glm::vec2 delta = pos - prePos;

            float y = +delta.x * rotateSpeed;
            float p = -delta.y * rotateSpeed;
            glm::vec3 upVec = enforceUpright ? glm::vec3(0.0f, 1.0f, 0.0f) : R[1];

            glm::mat3 Ry = glm::mat3(glm::rotate(glm::mat4(1.0f), y, upVec));
            glm::mat3 Rp = glm::mat3(glm::rotate(glm::mat4(1.0f), p, R[0]));
            R = Rp * Ry * R;

            if (enforceUpright) {
                glm::vec3 forward = glm::normalize(R[2]);
                glm::vec3 right = glm::normalize(glm::cross(upVec, forward));
                glm::vec3 up = glm::normalize(glm::cross(forward, right));
                R[0] = right;
                R[1] = up;
                R[2] = forward;
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
            const glm::vec3 movement = -(delta.x * adaptive_speed) * R[0] - (delta.y * adaptive_speed) * R[1];
            t += movement;
            pivot += movement;
            prePos = pos;
        }

        void zoom(float delta) {
            const glm::vec3 forward = R[2];
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

        void advance_forward(float deltaTime) {
            const glm::vec3 forward = glm::normalize(R * glm::vec3(0, 0, 1));
            const glm::vec3 movement = forward * deltaTime * wasdSpeed;
            t += movement;
            pivot += movement;
        }

        void advance_backward(float deltaTime) {
            const glm::vec3 forward = glm::normalize(R * glm::vec3(0, 0, 1));
            const glm::vec3 movement = -forward * deltaTime * wasdSpeed;
            t += movement;
            pivot += movement;
        }

        void advance_left(float deltaTime) {
            const glm::vec3 right = glm::normalize(R * glm::vec3(1, 0, 0));
            const glm::vec3 movement = -right * deltaTime * wasdSpeed;
            t += movement;
            pivot += movement;
        }

        void advance_right(float deltaTime) {
            const glm::vec3 right = glm::normalize(R * glm::vec3(1, 0, 0));
            const glm::vec3 movement = right * deltaTime * wasdSpeed;
            t += movement;
            pivot += movement;
        }

        void advance_up(float deltaTime) {
            const glm::vec3 up = glm::normalize(R * glm::vec3(0, 1, 0));
            const glm::vec3 movement = -up * deltaTime * wasdSpeed;
            t += movement;
            pivot += movement;
        }

        void advance_down(float deltaTime) {
            const glm::vec3 up = glm::normalize(R * glm::vec3(0, 1, 0));
            const glm::vec3 movement = up * deltaTime * wasdSpeed;
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
            const glm::vec3 forward = R * glm::vec3(0, 0, 1);
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
            float yaw = +delta.x * rotateCenterSpeed;
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
        void applyRotationAroundCenter(float yaw, float pitch) {
            // Rotate around the pivot point
            // Use world Y-axis for yaw rotation to maintain height
            // and camera's local right axis for pitch
            glm::vec3 worldUp(0.0f, 1.0f, 0.0f);

            // Yaw rotation around world Y-axis
            glm::mat3 Ry = glm::mat3(glm::rotate(glm::mat4(1.0f), yaw, worldUp));

            // Pitch rotation around camera's local right axis
            glm::mat3 Rp = glm::mat3(glm::rotate(glm::mat4(1.0f), pitch, R[0]));

            // Apply rotations: first yaw (around world), then pitch (around local)
            // This order maintains the horizon level during pure horizontal orbiting
            glm::mat3 U = Rp * Ry;

            // Transform position relative to pivot
            glm::vec3 relative_pos = t - pivot;
            relative_pos = U * relative_pos;
            t = pivot + relative_pos;

            // Transform orientation
            R = U * R;

            // Re-orthogonalize to prevent roll accumulation from numerical drift
            glm::vec3 forward = glm::normalize(R[2]);
            glm::vec3 right = glm::normalize(glm::cross(worldUp, forward));
            glm::vec3 up = glm::normalize(glm::cross(forward, right));
            R[0] = right;
            R[1] = up;
            R[2] = forward;
        }
    };

public:
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
        // Convert R (3x3) and t (3x1) to a 4x4 view matrix
        // View matrix = FLIP_YZ * inverse(camera transform)

        const glm::mat3 R_inv = lfs::rendering::computeViewRotation(camera.R);
        const glm::vec3 t_inv = lfs::rendering::FLIP_YZ * (-glm::transpose(camera.R) * camera.t);

        glm::mat4 view(1.0f);
        for (int i = 0; i < 3; ++i)
            for (int j = 0; j < 3; ++j)
                view[i][j] = R_inv[i][j];

        view[3][0] = t_inv.x;
        view[3][1] = t_inv.y;
        view[3][2] = t_inv.z;

        return view;
    }

    glm::mat4 getProjectionMatrix(float fov_degrees = lfs::rendering::DEFAULT_FOV,
                                  float near_plane = lfs::rendering::DEFAULT_NEAR_PLANE,
                                  float far_plane = lfs::rendering::DEFAULT_FAR_PLANE) const {
        float aspect_ratio = static_cast<float>(windowSize.x) / static_cast<float>(windowSize.y);
        float fov_radians = glm::radians(fov_degrees);
        return glm::perspective(fov_radians, aspect_ratio, near_plane, far_plane);
    }

    // Unproject screen pixel to world position (returns INVALID_WORLD_POS if invalid)
    [[nodiscard]] glm::vec3 unprojectPixel(float screen_x, float screen_y, float depth,
                                           float fov_degrees = lfs::rendering::DEFAULT_FOV) const {
        constexpr float INVALID_WORLD_POS = -1e10f;
        constexpr float MAX_DEPTH = 1e9f;

        if (depth <= 0.0f || depth > MAX_DEPTH) {
            return glm::vec3(INVALID_WORLD_POS);
        }

        const float width = static_cast<float>(windowSize.x);
        const float height = static_cast<float>(windowSize.y);
        const float fov_y = glm::radians(fov_degrees);
        const float aspect = width / height;
        const float fov_x = 2.0f * std::atan(std::tan(fov_y * 0.5f) * aspect);

        const float fx = width / (2.0f * std::tan(fov_x * 0.5f));
        const float fy = height / (2.0f * std::tan(fov_y * 0.5f));
        const float cx = width * 0.5f;
        const float cy = height * 0.5f;

        const glm::vec4 view_pos(
            (screen_x - cx) * depth / fx,
            (screen_y - cy) * depth / fy,
            depth,
            1.0f);

        const glm::mat3 R_inv = glm::transpose(camera.R);
        const glm::vec3 t_inv = -R_inv * camera.t;

        glm::mat4 w2c(1.0f);
        for (int i = 0; i < 3; ++i)
            for (int j = 0; j < 3; ++j)
                w2c[i][j] = R_inv[i][j];
        w2c[3][0] = t_inv.x;
        w2c[3][1] = t_inv.y;
        w2c[3][2] = t_inv.z;

        return glm::vec3(glm::inverse(w2c) * view_pos);
    }
};
/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include "rendering_types.hpp"
#include "viewport_interaction_context.hpp"
#include <chrono>

namespace lfs::rendering {
    class RenderingEngine;
}

namespace lfs::vis {
    class SceneManager;

    class CameraInteractionService {
    public:
        static constexpr auto PICK_THROTTLE_INTERVAL = std::chrono::milliseconds(50);

        void setCurrentCameraId(int cam_id) { current_camera_id_ = cam_id; }
        [[nodiscard]] int currentCameraId() const { return current_camera_id_; }

        [[nodiscard]] int hoveredCameraId() const { return hovered_camera_id_; }

        void clearCurrentCamera() { current_camera_id_ = -1; }
        void clearHoveredCamera() { hovered_camera_id_ = -1; }

        [[nodiscard]] int pickCameraFrustum(lfs::rendering::RenderingEngine* engine,
                                            SceneManager* scene_manager,
                                            const ViewportInteractionContext& viewport_context,
                                            const RenderSettings& settings,
                                            const glm::vec2& mouse_pos,
                                            bool& hover_changed);

    private:
        [[nodiscard]] bool shouldThrottlePick(std::chrono::steady_clock::time_point now) const {
            return now - last_pick_time_ < PICK_THROTTLE_INTERVAL;
        }

        void notePick(std::chrono::steady_clock::time_point now) {
            last_pick_time_ = now;
        }

        [[nodiscard]] bool updateHoveredCamera(int cam_id) {
            if (cam_id == hovered_camera_id_) {
                return false;
            }
            hovered_camera_id_ = cam_id;
            return true;
        }

        int current_camera_id_ = -1;
        int hovered_camera_id_ = -1;
        std::chrono::steady_clock::time_point last_pick_time_{};
    };

} // namespace lfs::vis

/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "camera_interaction_service.hpp"
#include "rendering/coordinate_conventions.hpp"
#include "rendering/rendering.hpp"
#include "scene/scene_manager.hpp"

namespace lfs::vis {

    int CameraInteractionService::pickCameraFrustum(
        lfs::rendering::RenderingEngine* const engine,
        SceneManager* const scene_manager,
        const ViewportInteractionContext& viewport_context,
        const RenderSettings& settings,
        const glm::vec2& mouse_pos,
        bool& hover_changed) {
        hover_changed = false;

        if (!settings.show_camera_frustums) {
            return -1;
        }

        const auto now = std::chrono::steady_clock::now();
        if (shouldThrottlePick(now)) {
            return hovered_camera_id_;
        }
        notePick(now);

        if (!engine || !scene_manager || !viewport_context.pick_context_valid) {
            return hovered_camera_id_;
        }

        auto cameras = scene_manager->getScene().getVisibleCameras();
        if (cameras.empty()) {
            return -1;
        }

        const auto* panel = viewport_context.resolvePanel(mouse_pos);
        if (!panel) {
            return hovered_camera_id_;
        }

        auto scene_transforms = scene_manager->getScene().getVisibleCameraSceneTransforms();
        for (auto& transform : scene_transforms) {
            transform = lfs::rendering::dataWorldTransformToVisualizerWorld(transform);
        }

        const lfs::rendering::CameraFrustumPickRequest request{
            .mouse_pos = mouse_pos,
            .viewport_pos = panel->viewport_pos,
            .viewport_size = panel->viewport_size,
            .viewport = panel->viewport_data,
            .scale = settings.camera_frustum_scale,
            .scene_transforms = std::move(scene_transforms)};

        const auto pick_result = engine->pickCameraFrustum(cameras, request);

        int cam_id = -1;
        if (pick_result) {
            cam_id = *pick_result;
        }

        hover_changed = updateHoveredCamera(cam_id);
        return hovered_camera_id_;
    }

} // namespace lfs::vis

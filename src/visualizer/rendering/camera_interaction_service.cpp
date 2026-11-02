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

        const auto& scene = scene_manager->getScene();
        const auto cameras = scene.getVisibleCamerasCached();
        if (!cameras || cameras->empty()) {
            return -1;
        }

        const auto* panel = viewport_context.resolvePanel(mouse_pos);
        if (!panel) {
            return hovered_camera_id_;
        }

        if (cameras != camera_snapshot_.lock()) {
            scene_transforms_ = scene.getVisibleCameraSceneTransforms();
            for (auto& transform : scene_transforms_) {
                transform = lfs::rendering::dataWorldTransformToVisualizerWorld(transform);
            }
            camera_snapshot_ = cameras;
        }

        const auto pick = [&](const auto& pick_cameras, const auto& transforms) {
            const lfs::rendering::CameraFrustumPickRequest request{
                .mouse_pos = mouse_pos,
                .viewport_pos = panel->viewport_pos,
                .viewport_size = panel->viewport_size,
                .viewport = panel->viewport_data,
                .scale = settings.camera_frustum_scale,
                .scene_transforms = transforms};
            return engine->pickCameraFrustum(pick_cameras, request);
        };

        int cam_id = -1;
        if (settings.hide_disabled_camera_frustums) {
            const auto disabled = scene.getTrainingDisabledCameraUids();
            std::vector<std::shared_ptr<const lfs::core::Camera>> enabled_cameras;
            std::vector<glm::mat4> enabled_transforms;
            enabled_cameras.reserve(cameras->size());
            enabled_transforms.reserve(scene_transforms_.size());
            for (size_t index = 0; index < cameras->size() && index < scene_transforms_.size(); ++index) {
                if (!disabled.contains((*cameras)[index]->uid())) {
                    enabled_cameras.push_back((*cameras)[index]);
                    enabled_transforms.push_back(scene_transforms_[index]);
                }
            }
            if (enabled_cameras.empty())
                return -1;
            if (const auto pick_result = pick(enabled_cameras, enabled_transforms))
                cam_id = *pick_result;
        } else if (const auto pick_result = pick(*cameras, scene_transforms_)) {
            cam_id = *pick_result;
        }

        hover_changed = updateHoveredCamera(cam_id);
        return hovered_camera_id_;
    }

} // namespace lfs::vis

/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "overlay_pass.hpp"
#include "../rendering_manager.hpp"
#include "core/logger.hpp"
#include "scene/scene_manager.hpp"
#include <cassert>
#include <glad/glad.h>

namespace lfs::vis {

    void OverlayPass::execute(lfs::rendering::RenderingEngine& engine,
                              const FrameContext& ctx,
                              FrameResources& res) {
        const auto& settings = ctx.settings;

        if (ctx.render_size.x <= 0 || ctx.render_size.y <= 0)
            return;

        const lfs::rendering::ViewportData viewport{
            .rotation = ctx.viewport.getRotationMatrix(),
            .translation = ctx.viewport.getTranslation(),
            .size = ctx.render_size,
            .focal_length_mm = settings.focal_length_mm,
            .orthographic = settings.orthographic,
            .ortho_scale = settings.ortho_scale};

        if (settings.show_crop_box && ctx.scene_manager) {
            const auto visible_cropboxes = ctx.scene_manager->getScene().getVisibleCropBoxes();
            const core::NodeId selected_cropbox_id = ctx.scene_manager->getSelectedNodeCropBoxId();

            for (const auto& cb : visible_cropboxes) {
                if (!cb.data)
                    continue;

                const bool is_selected = (cb.node_id == selected_cropbox_id);

                const bool use_pending = is_selected && ctx.gizmo.cropbox_active;
                const glm::vec3 box_min = use_pending ? ctx.gizmo.cropbox_min : cb.data->min;
                const glm::vec3 box_max = use_pending ? ctx.gizmo.cropbox_max : cb.data->max;
                const glm::mat4 box_transform = use_pending ? ctx.gizmo.cropbox_transform : cb.world_transform;

                const lfs::rendering::BoundingBox box{
                    .min = box_min,
                    .max = box_max,
                    .transform = glm::inverse(box_transform)};

                const glm::vec3 base_color = cb.data->inverse
                                                 ? glm::vec3(1.0f, 0.2f, 0.2f)
                                                 : cb.data->color;
                const float flash = is_selected ? cb.data->flash_intensity : 0.0f;
                constexpr float FLASH_LINE_BOOST = 4.0f;
                const glm::vec3 color = glm::mix(base_color, glm::vec3(1.0f), flash);
                const float line_width = cb.data->line_width + flash * FLASH_LINE_BOOST;

                auto bbox_result = engine.renderBoundingBox(box, viewport, color, line_width);
                if (!bbox_result) {
                    LOG_WARN("Failed to render bounding box: {}", bbox_result.error());
                }
            }
        }

        if (settings.show_ellipsoid && ctx.scene_manager) {
            const auto visible_ellipsoids = ctx.scene_manager->getScene().getVisibleEllipsoids();
            const core::NodeId selected_ellipsoid_id = ctx.scene_manager->getSelectedNodeEllipsoidId();

            for (const auto& el : visible_ellipsoids) {
                if (!el.data)
                    continue;

                const bool is_selected = (el.node_id == selected_ellipsoid_id);

                const glm::vec3 radii = (is_selected && ctx.gizmo.ellipsoid_active)
                                            ? ctx.gizmo.ellipsoid_radii
                                            : el.data->radii;
                const glm::mat4 transform = (is_selected && ctx.gizmo.ellipsoid_active)
                                                ? ctx.gizmo.ellipsoid_transform
                                                : el.world_transform;

                const lfs::rendering::Ellipsoid ellipsoid{
                    .radii = radii,
                    .transform = transform};

                const glm::vec3 base_color = el.data->inverse
                                                 ? glm::vec3(1.0f, 0.2f, 0.2f)
                                                 : el.data->color;
                const float flash = is_selected ? el.data->flash_intensity : 0.0f;
                constexpr float FLASH_LINE_BOOST = 4.0f;
                const glm::vec3 color = glm::mix(base_color, glm::vec3(1.0f), flash);
                const float line_width = el.data->line_width + flash * FLASH_LINE_BOOST;

                auto ellipsoid_result = engine.renderEllipsoid(ellipsoid, viewport, color, line_width);
                if (!ellipsoid_result) {
                    LOG_WARN("Failed to render ellipsoid: {}", ellipsoid_result.error());
                }
            }
        }

        if (settings.show_coord_axes) {
            auto axes_result = engine.renderCoordinateAxes(viewport, settings.axes_size, settings.axes_visibility, settings.equirectangular);
            if (!axes_result) {
                LOG_WARN("Failed to render coordinate axes: {}", axes_result.error());
            }
        }

        {
            constexpr float PIVOT_DURATION_SEC = 0.5f;
            constexpr float PIVOT_SIZE_PX = 50.0f;

            const float time_since_set = ctx.viewport.camera.getSecondsSincePivotSet();
            const bool animation_active = time_since_set < PIVOT_DURATION_SEC;

            if (animation_active && res.manager) {
                const auto remaining_ms = static_cast<int>((PIVOT_DURATION_SEC - time_since_set) * 1000.0f);
                res.manager->setPivotAnimationEndTime(std::chrono::steady_clock::now() +
                                                      std::chrono::milliseconds(remaining_ms));
            }

            if (settings.show_pivot || animation_active) {
                const float opacity = settings.show_pivot ? 1.0f : 1.0f - std::clamp(time_since_set / PIVOT_DURATION_SEC, 0.0f, 1.0f);

                if (auto result = engine.renderPivot(viewport, ctx.viewport.camera.getPivot(),
                                                     PIVOT_SIZE_PX, opacity);
                    !result) {
                    LOG_WARN("Pivot render failed: {}", result.error());
                }
            }
        }

        if (settings.show_camera_frustums && ctx.scene_manager) {
            auto cameras = ctx.scene_manager->getScene().getVisibleCameras();

            if (!cameras.empty()) {
                int highlight_index = -1;
                if (ctx.pick.hovered_camera_id >= 0) {
                    for (size_t i = 0; i < cameras.size(); ++i) {
                        if (cameras[i]->uid() == ctx.pick.hovered_camera_id) {
                            highlight_index = static_cast<int>(i);
                            break;
                        }
                    }
                }

                glm::mat4 scene_transform(1.0f);
                auto visible_transforms = ctx.scene_manager->getScene().getVisibleNodeTransforms();
                if (!visible_transforms.empty()) {
                    scene_transform = visible_transforms[0];
                }

                LOG_TRACE("Rendering {} camera frustums with scale {}, highlighted index: {} (ID: {})",
                          cameras.size(), settings.camera_frustum_scale, highlight_index, ctx.pick.hovered_camera_id);

                auto disabled_uids = ctx.scene_manager->getScene().getTrainingDisabledCameraUids();

                std::unordered_set<int> selected_uids;
                for (const auto& name : ctx.scene_manager->getSelectedNodeNames()) {
                    const auto* node = ctx.scene_manager->getScene().getNode(name);
                    if (node && node->type == core::NodeType::CAMERA && node->camera_uid >= 0)
                        selected_uids.insert(node->camera_uid);
                }

                auto frustum_result = engine.renderCameraFrustumsWithHighlight(
                    cameras, viewport,
                    settings.camera_frustum_scale,
                    settings.train_camera_color,
                    settings.eval_camera_color,
                    highlight_index,
                    scene_transform,
                    settings.equirectangular,
                    disabled_uids,
                    selected_uids);

                if (!frustum_result) {
                    LOG_ERROR("Failed to render camera frustums: {}", frustum_result.error());
                }

                if (ctx.pick.requested && ctx.viewport_region) {
                    res.pick_consumed = true;

                    auto pick_result = engine.pickCameraFrustum(
                        cameras,
                        ctx.pick.pos,
                        glm::vec2(ctx.viewport_region->x, ctx.viewport_region->y),
                        glm::vec2(ctx.viewport_region->width, ctx.viewport_region->height),
                        viewport,
                        settings.camera_frustum_scale,
                        scene_transform);

                    if (pick_result) {
                        int cam_id = *pick_result;
                        if (cam_id != ctx.pick.hovered_camera_id) {
                            LOG_DEBUG("Camera hover changed: {} -> {}", ctx.pick.hovered_camera_id, cam_id);
                            res.hovered_camera_id = cam_id;
                            if (res.manager)
                                res.manager->markDirty(DirtyFlag::OVERLAY);
                        }
                    } else if (ctx.pick.hovered_camera_id != -1) {
                        LOG_DEBUG("Camera hover lost (was ID: {})", ctx.pick.hovered_camera_id);
                        res.hovered_camera_id = -1;
                        if (res.manager)
                            res.manager->markDirty(DirtyFlag::OVERLAY);
                    }
                }
            }
        }

        if (settings.show_grid && settings.split_view_mode == SplitViewMode::Disabled && !settings.equirectangular) {
            if (const auto result = engine.renderGrid(
                    viewport,
                    static_cast<lfs::rendering::GridPlane>(settings.grid_plane),
                    settings.grid_opacity);
                !result) {
                LOG_WARN("Grid render failed: {}", result.error());
            }
        }
    }

} // namespace lfs::vis

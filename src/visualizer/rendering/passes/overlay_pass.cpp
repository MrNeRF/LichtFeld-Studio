/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "overlay_pass.hpp"
#include "../rendering_manager.hpp"
#include "core/logger.hpp"
#include "scene/scene_manager.hpp"
#include <glad/glad.h>

namespace lfs::vis {

    void OverlayPass::execute(lfs::rendering::RenderingEngine& /*engine*/,
                              const FrameContext& ctx,
                              FrameResources& /*res*/) {
        renderOverlays(ctx.ctx);
    }

    void OverlayPass::renderOverlays(const RenderingManager::RenderContext& context) {
        const auto& settings = mgr_.settings_;
        auto& engine = *mgr_.engine_;

        glm::ivec2 render_size = context.viewport.windowSize;
        if (context.viewport_region) {
            render_size = glm::ivec2(
                static_cast<int>(context.viewport_region->width),
                static_cast<int>(context.viewport_region->height));
        }

        if (render_size.x <= 0 || render_size.y <= 0) {
            return;
        }

        lfs::rendering::ViewportData viewport{
            .rotation = context.viewport.getRotationMatrix(),
            .translation = context.viewport.getTranslation(),
            .size = render_size,
            .focal_length_mm = settings.focal_length_mm,
            .orthographic = settings.orthographic,
            .ortho_scale = settings.ortho_scale};

        if (settings.show_crop_box && context.scene_manager) {
            const auto visible_cropboxes = context.scene_manager->getScene().getVisibleCropBoxes();
            const core::NodeId selected_cropbox_id = context.scene_manager->getSelectedNodeCropBoxId();

            for (const auto& cb : visible_cropboxes) {
                if (!cb.data)
                    continue;

                const bool is_selected = (cb.node_id == selected_cropbox_id);

                const bool use_pending = is_selected && mgr_.cropbox_gizmo_active_;
                const glm::vec3 box_min = use_pending ? mgr_.pending_cropbox_min_ : cb.data->min;
                const glm::vec3 box_max = use_pending ? mgr_.pending_cropbox_max_ : cb.data->max;
                const glm::mat4 box_transform = use_pending ? mgr_.pending_cropbox_transform_ : cb.world_transform;

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

        if (settings.show_ellipsoid && context.scene_manager) {
            const auto visible_ellipsoids = context.scene_manager->getScene().getVisibleEllipsoids();
            const core::NodeId selected_ellipsoid_id = context.scene_manager->getSelectedNodeEllipsoidId();

            for (const auto& el : visible_ellipsoids) {
                if (!el.data)
                    continue;

                const bool is_selected = (el.node_id == selected_ellipsoid_id);

                const glm::vec3 radii = (is_selected && mgr_.ellipsoid_gizmo_active_)
                                            ? mgr_.pending_ellipsoid_radii_
                                            : el.data->radii;
                const glm::mat4 transform = (is_selected && mgr_.ellipsoid_gizmo_active_)
                                                ? mgr_.pending_ellipsoid_transform_
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

            const float time_since_set = context.viewport.camera.getSecondsSincePivotSet();
            const bool animation_active = time_since_set < PIVOT_DURATION_SEC;

            if (animation_active) {
                const auto remaining_ms = static_cast<int>((PIVOT_DURATION_SEC - time_since_set) * 1000.0f);
                mgr_.setPivotAnimationEndTime(std::chrono::steady_clock::now() +
                                              std::chrono::milliseconds(remaining_ms));
            }

            if (settings.show_pivot || animation_active) {
                const float opacity = settings.show_pivot ? 1.0f : 1.0f - std::clamp(time_since_set / PIVOT_DURATION_SEC, 0.0f, 1.0f);

                if (auto result = engine.renderPivot(viewport, context.viewport.camera.getPivot(),
                                                     PIVOT_SIZE_PX, opacity);
                    !result) {
                    LOG_WARN("Pivot render failed: {}", result.error());
                }
            }
        }

        if (settings.show_camera_frustums && context.scene_manager) {
            auto cameras = context.scene_manager->getScene().getVisibleCameras();

            if (!cameras.empty()) {
                int highlight_index = -1;
                if (mgr_.hovered_camera_id_ >= 0) {
                    for (size_t i = 0; i < cameras.size(); ++i) {
                        if (cameras[i]->uid() == mgr_.hovered_camera_id_) {
                            highlight_index = static_cast<int>(i);
                            break;
                        }
                    }
                }

                glm::mat4 scene_transform(1.0f);
                auto visible_transforms = context.scene_manager->getScene().getVisibleNodeTransforms();
                if (!visible_transforms.empty()) {
                    scene_transform = visible_transforms[0];
                }

                LOG_TRACE("Rendering {} camera frustums with scale {}, highlighted index: {} (ID: {})",
                          cameras.size(), settings.camera_frustum_scale, highlight_index, mgr_.hovered_camera_id_);

                auto disabled_uids = context.scene_manager->getScene().getTrainingDisabledCameraUids();

                std::unordered_set<int> selected_uids;
                for (const auto& name : context.scene_manager->getSelectedNodeNames()) {
                    const auto* node = context.scene_manager->getScene().getNode(name);
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

                if (mgr_.pick_requested_ && context.viewport_region) {
                    mgr_.pick_requested_ = false;

                    auto pick_result = engine.pickCameraFrustum(
                        cameras,
                        mgr_.pending_pick_pos_,
                        glm::vec2(context.viewport_region->x, context.viewport_region->y),
                        glm::vec2(context.viewport_region->width, context.viewport_region->height),
                        viewport,
                        settings.camera_frustum_scale,
                        scene_transform);

                    if (pick_result) {
                        int cam_id = *pick_result;
                        if (cam_id != mgr_.hovered_camera_id_) {
                            int old_hover = mgr_.hovered_camera_id_;
                            mgr_.hovered_camera_id_ = cam_id;
                            mgr_.markDirty(DirtyFlag::OVERLAY);
                            LOG_DEBUG("Camera hover changed: {} -> {}", old_hover, cam_id);
                        }
                    } else if (mgr_.hovered_camera_id_ != -1) {
                        int old_hover = mgr_.hovered_camera_id_;
                        mgr_.hovered_camera_id_ = -1;
                        mgr_.markDirty(DirtyFlag::OVERLAY);
                        LOG_DEBUG("Camera hover lost (was ID: {})", old_hover);
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

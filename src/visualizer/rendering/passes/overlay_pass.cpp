/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "overlay_pass.hpp"
#include "core/logger.hpp"
#include "rendering/gl_state_guard.hpp"
#include "scene/scene_manager.hpp"
#include "theme/theme.hpp"
#include "training/training_manager.hpp"
#include <algorithm>
#include <cassert>
#include <glad/glad.h>
#include <unordered_set>

namespace lfs::vis {

    namespace {
        [[nodiscard]] lfs::rendering::ScreenSpaceVignette makeViewportVignette() {
            const auto& vignette = theme().vignette;
            return {
                .enabled = vignette.enabled,
                .intensity = vignette.intensity,
                .radius = vignette.radius,
                .softness = vignette.softness,
            };
        }
    } // namespace

    void OverlayPass::execute(lfs::rendering::RenderingEngine& engine,
                              const FrameContext& ctx,
                              FrameResources& res) {
        const auto& settings = ctx.settings;

        if (ctx.render_size.x <= 0 || ctx.render_size.y <= 0)
            return;

        const auto render_overlays = [&](const Viewport& source_viewport,
                                         const lfs::rendering::ViewportData& viewport,
                                         const int grid_plane) {
            if (settings.depth_filter_enabled) {
                const lfs::rendering::BoundingBox depth_box{
                    .min = settings.depth_filter_min,
                    .max = settings.depth_filter_max,
                    .transform = settings.depth_filter_transform.inv().toMat4()};

                constexpr glm::vec3 DEPTH_BOX_OUTLINE_COLOR{0.0f, 0.0f, 0.0f};
                constexpr float DEPTH_BOX_OUTLINE_WIDTH = 9.0f;
                if (auto result =
                        engine.renderBoundingBox(depth_box, viewport, DEPTH_BOX_OUTLINE_COLOR, DEPTH_BOX_OUTLINE_WIDTH);
                    !result) {
                    LOG_WARN("Failed to render depth selection box outline: {}", result.error());
                }

                constexpr glm::vec3 DEPTH_BOX_GLOW_COLOR{1.0f, 1.0f, 1.0f};
                constexpr float DEPTH_BOX_GLOW_WIDTH = 6.0f;
                if (auto result = engine.renderBoundingBox(depth_box, viewport, DEPTH_BOX_GLOW_COLOR, DEPTH_BOX_GLOW_WIDTH);
                    !result) {
                    LOG_WARN("Failed to render depth selection box glow: {}", result.error());
                }

                constexpr glm::vec3 DEPTH_BOX_COLOR{1.0f, 1.0f, 1.0f};
                constexpr float DEPTH_BOX_LINE_WIDTH = 4.5f;
                if (auto result = engine.renderBoundingBox(depth_box, viewport, DEPTH_BOX_COLOR, DEPTH_BOX_LINE_WIDTH);
                    !result) {
                    LOG_WARN("Failed to render depth selection box: {}", result.error());
                }
            }

            if (settings.show_crop_box && ctx.scene_manager) {
                const auto& visible_cropboxes = ctx.scene_state.cropboxes;
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

                    if (auto bbox_result = engine.renderBoundingBox(box, viewport, color, line_width);
                        !bbox_result) {
                        LOG_WARN("Failed to render bounding box: {}", bbox_result.error());
                    }
                }
            }

            if (settings.show_ellipsoid && ctx.scene_manager) {
                const auto& visible_ellipsoids = ctx.scene_state.ellipsoids;
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

                    if (auto ellipsoid_result = engine.renderEllipsoid(ellipsoid, viewport, color, line_width);
                        !ellipsoid_result) {
                        LOG_WARN("Failed to render ellipsoid: {}", ellipsoid_result.error());
                    }
                }
            }

            if (settings.show_coord_axes) {
                if (auto axes_result = engine.renderCoordinateAxes(
                        viewport, settings.axes_size, settings.axes_visibility, settings.equirectangular);
                    !axes_result) {
                    LOG_WARN("Failed to render coordinate axes: {}", axes_result.error());
                }
            }

            {
                constexpr float PIVOT_DURATION_SEC = 0.5f;
                constexpr float PIVOT_SIZE_PX = 50.0f;

                const float time_since_set = source_viewport.camera.getSecondsSincePivotSet();
                const bool animation_active = time_since_set < PIVOT_DURATION_SEC;

                if (animation_active) {
                    const auto remaining_ms = static_cast<int>((PIVOT_DURATION_SEC - time_since_set) * 1000.0f);
                    const auto animation_end = std::chrono::steady_clock::now() +
                                               std::chrono::milliseconds(remaining_ms);
                    if (!res.pivot_animation_end || animation_end > *res.pivot_animation_end) {
                        res.pivot_animation_end = animation_end;
                    }
                }

                if (settings.show_pivot || animation_active) {
                    const float opacity =
                        settings.show_pivot
                            ? 1.0f
                            : 1.0f - std::clamp(time_since_set / PIVOT_DURATION_SEC, 0.0f, 1.0f);

                    if (auto result = engine.renderPivot(
                            viewport, source_viewport.camera.getPivot(), PIVOT_SIZE_PX, opacity);
                        !result) {
                        LOG_WARN("Pivot render failed: {}", result.error());
                    }
                }
            }

            if (settings.show_camera_frustums && ctx.scene_manager) {
                auto cameras = ctx.scene_manager->getScene().getVisibleCameras();

                if (!cameras.empty()) {
                    int focused_index = -1;
                    if (ctx.hovered_camera_id >= 0) {
                        for (size_t i = 0; i < cameras.size(); ++i) {
                            if (cameras[i]->uid() == ctx.hovered_camera_id) {
                                focused_index = static_cast<int>(i);
                                break;
                            }
                        }
                    }

                    LOG_TRACE("Rendering {} camera frustums with scale {}, focused index: {} (ID: {})",
                              cameras.size(), settings.camera_frustum_scale, focused_index, ctx.hovered_camera_id);

                    auto disabled_uids = ctx.scene_manager->getScene().getTrainingDisabledCameraUids();

                    std::unordered_set<int> emphasized_uids;
                    for (const auto& name : ctx.scene_manager->getSelectedNodeNames()) {
                        const auto* node = ctx.scene_manager->getScene().getNode(name);
                        if (node && node->type == core::NodeType::CAMERA && node->camera_uid >= 0) {
                            emphasized_uids.insert(node->camera_uid);
                        }
                    }

                    lfs::rendering::CameraFrustumRenderRequest request{
                        .viewport = viewport,
                        .scale = settings.camera_frustum_scale,
                        .train_color = settings.train_camera_color,
                        .eval_color = settings.eval_camera_color,
                        .per_camera_colors = {},
                        .focused_index = focused_index,
                        .scene_transforms = ctx.scene_state.camera_scene_transforms,
                        .equirectangular_view = settings.equirectangular,
                        .disabled_uids = std::move(disabled_uids),
                        .emphasized_uids = std::move(emphasized_uids)};

                    if (const auto* trainer_manager = ctx.scene_manager->getTrainerManager()) {
                        if (const auto* trainer = trainer_manager->getTrainer()) {
                            std::vector<std::array<float, 3>> loss_colors;
                            if (trainer->fillCameraLossColors(cameras, loss_colors)) {
                                request.per_camera_colors.reserve(loss_colors.size());
                                for (const auto& color : loss_colors) {
                                    request.per_camera_colors.emplace_back(color[0], color[1], color[2]);
                                }
                            }
                        }
                    }

                    if (auto frustum_result = engine.renderCameraFrustums(cameras, request);
                        !frustum_result) {
                        LOG_ERROR("Failed to render camera frustums: {}", frustum_result.error());
                    }
                }
            }

            if (settings.show_grid &&
                (!splitViewUsesComparisonPanels(settings.split_view_mode)) &&
                !settings.equirectangular) {
                if (const auto result = engine.renderGrid(
                        viewport,
                        static_cast<lfs::rendering::GridPlane>(grid_plane),
                        settings.grid_opacity);
                    !result) {
                    LOG_WARN("Grid render failed: {}", result.error());
                }
            }
        };

        if (ctx.view_panels.size() > 1 && ctx.render_size.x > 1 && ctx.render_size.y > 0) {
            {
                lfs::rendering::GLViewportGuard viewport_guard;
                lfs::rendering::GLScissorEnableGuard scissor_guard;
                glEnable(GL_SCISSOR_TEST);

                for (const auto& panel : ctx.view_panels) {
                    if (!panel.valid()) {
                        continue;
                    }

                    glViewport(ctx.viewport_pos.x + panel.viewport_offset.x,
                               ctx.viewport_pos.y + panel.viewport_offset.y,
                               panel.render_size.x,
                               panel.render_size.y);
                    glScissor(ctx.viewport_pos.x + panel.viewport_offset.x,
                              ctx.viewport_pos.y + panel.viewport_offset.y,
                              panel.render_size.x,
                              panel.render_size.y);
                    render_overlays(*panel.viewport, ctx.makeViewportData(panel), panel.grid_plane);
                }
            }
        } else {
            render_overlays(ctx.viewport, ctx.makeViewportData(), settings.grid_plane);
        }

        const auto vignette = makeViewportVignette();
        if (vignette.active()) {
            glViewport(ctx.viewport_pos.x, ctx.viewport_pos.y, ctx.render_size.x, ctx.render_size.y);
            if (auto result = engine.renderScreenSpaceVignette(ctx.render_size, vignette); !result) {
                LOG_WARN("Screen-space vignette render failed: {}", result.error());
            }
        }
    }

} // namespace lfs::vis

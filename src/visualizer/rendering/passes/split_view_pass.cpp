/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "split_view_pass.hpp"
#include "core/logger.hpp"
#include "scene/scene_manager.hpp"
#include "training/training_manager.hpp"
#include <cassert>
#include <shared_mutex>

namespace lfs::vis {

    bool SplitViewPass::shouldExecute(DirtyMask frame_dirty, const FrameContext& ctx) const {
        if (ctx.settings.split_view_mode == SplitViewMode::Disabled)
            return false;
        return (frame_dirty & sensitivity()) != 0;
    }

    void SplitViewPass::execute(lfs::rendering::RenderingEngine& engine,
                                const FrameContext& ctx,
                                FrameResources& res) {
        auto split_request = buildSplitViewRequest(ctx, res);
        if (!split_request) {
            res.split_view_executed = false;
            return;
        }

        res.split_info.enabled = true;
        if (split_request->panels.size() >= 2) {
            res.split_info.left_name = split_request->panels[0].label;
            res.split_info.right_name = split_request->panels[1].label;
        }

        std::optional<std::shared_lock<std::shared_mutex>> render_lock;
        if (const auto* tm = ctx.scene_manager ? ctx.scene_manager->getTrainerManager() : nullptr) {
            if (const auto* trainer = tm->getTrainer()) {
                render_lock.emplace(trainer->getRenderMutex());
            }
        }

        auto result = engine.renderSplitView(*split_request);
        render_lock.reset();

        if (result) {
            res.cached_result = *result;
            res.cached_result_size = ctx.render_size;
            res.split_view_executed = true;
        } else {
            LOG_ERROR("Failed to render split view: {}", result.error());
            res.cached_result_size = {0, 0};
        }
    }

    std::optional<lfs::rendering::SplitViewRequest>
    SplitViewPass::buildSplitViewRequest(const FrameContext& ctx, const FrameResources& res) {
        const auto& settings = ctx.settings;
        if (settings.split_view_mode == SplitViewMode::Disabled || !ctx.scene_manager)
            return std::nullopt;

        const lfs::rendering::ViewportData viewport_data{
            .rotation = ctx.viewport.getRotationMatrix(),
            .translation = ctx.viewport.getTranslation(),
            .size = ctx.render_size,
            .focal_length_mm = settings.focal_length_mm,
            .orthographic = settings.orthographic,
            .ortho_scale = settings.ortho_scale};

        std::optional<lfs::rendering::BoundingBox> crop_box;
        if (settings.use_crop_box || settings.show_crop_box) {
            const auto& cropboxes = ctx.scene_manager->getScene().getVisibleCropBoxes();
            if (!cropboxes.empty() && cropboxes[0].data) {
                const auto& cb = cropboxes[0];
                crop_box = lfs::rendering::BoundingBox{
                    .min = cb.data->min,
                    .max = cb.data->max,
                    .transform = glm::inverse(cb.world_transform)};
            }
        }

        if (settings.split_view_mode == SplitViewMode::GTComparison) {
            if (!res.gt_context || !res.gt_context->valid() || !res.render_texture_valid)
                return std::nullopt;

            auto letterbox_viewport = viewport_data;
            letterbox_viewport.size = ctx.render_size;

            const auto disabled_uids = ctx.scene_manager->getScene().getTrainingDisabledCameraUids();
            const bool cam_disabled = ctx.current_camera_id >= 0 && disabled_uids.count(ctx.current_camera_id) > 0;
            std::string gt_label = cam_disabled ? "Ground Truth (Excluded from Training)" : "Ground Truth";

            return lfs::rendering::SplitViewRequest{
                .panels = {{.content_type = lfs::rendering::PanelContentType::Image2D,
                            .texture_id = res.gt_context->gt_texture_id,
                            .label = std::move(gt_label),
                            .start_position = 0.0f,
                            .end_position = settings.split_position},
                           {.content_type = lfs::rendering::PanelContentType::CachedRender,
                            .texture_id = ctx.cached_render_texture,
                            .label = "Rendered",
                            .start_position = settings.split_position,
                            .end_position = 1.0f}},
                .viewport = letterbox_viewport,
                .scaling_modifier = settings.scaling_modifier,
                .antialiasing = settings.antialiasing,
                .mip_filter = settings.mip_filter,
                .sh_degree = settings.sh_degree,
                .background_color = settings.background_color,
                .crop_box = crop_box,
                .point_cloud_mode = settings.point_cloud_mode,
                .voxel_size = settings.voxel_size,
                .gut = settings.gut,
                .equirectangular = settings.equirectangular,
                .show_rings = settings.show_rings,
                .ring_width = settings.ring_width,
                .show_dividers = true,
                .divider_color = glm::vec4(1.0f, 0.85f, 0.0f, 1.0f),
                .show_labels = true,
                .left_texcoord_scale = res.gt_context->gt_texcoord_scale,
                .right_texcoord_scale = res.gt_context->render_texcoord_scale,
                .flip_left_y = res.gt_context->gt_needs_flip,
                .letterbox = true,
                .content_size = res.gt_context->dimensions};
        }

        if (settings.split_view_mode == SplitViewMode::PLYComparison) {
            const auto& scene = ctx.scene_manager->getScene();
            const auto visible_nodes = scene.getVisibleNodes();
            if (visible_nodes.size() < 2) {
                LOG_TRACE("PLY comparison needs at least 2 visible nodes, have {}", visible_nodes.size());
                return std::nullopt;
            }

            size_t left_idx = settings.split_view_offset % visible_nodes.size();
            size_t right_idx = (settings.split_view_offset + 1) % visible_nodes.size();
            assert(visible_nodes[left_idx]->model);
            assert(visible_nodes[right_idx]->model);

            LOG_TRACE("Creating PLY comparison split view: {} vs {}",
                      visible_nodes[left_idx]->name, visible_nodes[right_idx]->name);

            const glm::vec2 texcoord_scale(1.0f, 1.0f);

            return lfs::rendering::SplitViewRequest{
                .panels = {
                    {.content_type = lfs::rendering::PanelContentType::Model3D,
                     .model = visible_nodes[left_idx]->model.get(),
                     .model_transform = scene.getWorldTransform(visible_nodes[left_idx]->id),
                     .texture_id = 0,
                     .label = visible_nodes[left_idx]->name,
                     .start_position = 0.0f,
                     .end_position = settings.split_position},
                    {.content_type = lfs::rendering::PanelContentType::Model3D,
                     .model = visible_nodes[right_idx]->model.get(),
                     .model_transform = scene.getWorldTransform(visible_nodes[right_idx]->id),
                     .texture_id = 0,
                     .label = visible_nodes[right_idx]->name,
                     .start_position = settings.split_position,
                     .end_position = 1.0f}},
                .viewport = viewport_data,
                .scaling_modifier = settings.scaling_modifier,
                .antialiasing = settings.antialiasing,
                .mip_filter = settings.mip_filter,
                .sh_degree = settings.sh_degree,
                .background_color = settings.background_color,
                .crop_box = crop_box,
                .point_cloud_mode = settings.point_cloud_mode,
                .voxel_size = settings.voxel_size,
                .gut = settings.gut,
                .equirectangular = settings.equirectangular,
                .show_rings = settings.show_rings,
                .ring_width = settings.ring_width,
                .show_dividers = true,
                .divider_color = glm::vec4(1.0f, 0.85f, 0.0f, 1.0f),
                .show_labels = true,
                .left_texcoord_scale = texcoord_scale,
                .right_texcoord_scale = texcoord_scale};
        }

        return std::nullopt;
    }

} // namespace lfs::vis

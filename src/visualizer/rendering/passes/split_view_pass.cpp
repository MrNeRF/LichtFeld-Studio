/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "split_view_pass.hpp"
#include "../viewport_request_builder.hpp"
#include "core/logger.hpp"
#include "scene/scene_manager.hpp"
#include <cassert>

namespace lfs::vis {

    namespace {
        [[nodiscard]] lfs::rendering::SplitViewPanelContent buildModelPanelContent(
            const FrameContext& ctx,
            const lfs::core::SplatData& model,
            const glm::mat4& model_transform) {
            lfs::rendering::SplitViewPanelContent content{
                .type = lfs::rendering::PanelContentType::Model3D,
                .model = &model,
                .model_transform = model_transform,
                .gaussian_render = std::nullopt,
                .point_cloud_render = std::nullopt,
                .texture_id = 0};

            if (ctx.settings.point_cloud_mode) {
                content.point_cloud_render = buildSplitViewPointCloudPanelRenderState(ctx, ctx.render_size);
                return content;
            }

            content.gaussian_render = buildSplitViewGaussianPanelRenderState(ctx, ctx.render_size);
            return content;
        }
    } // namespace

    bool SplitViewPass::shouldExecute(DirtyMask frame_dirty, const FrameContext& ctx) const {
        if (ctx.settings.split_view_mode == SplitViewMode::Disabled)
            return false;
        return (frame_dirty & sensitivity()) != 0;
    }

    void SplitViewPass::execute(lfs::rendering::RenderingEngine& engine,
                                const FrameContext& ctx,
                                FrameResources& res) {
        SplitViewInfo split_info{};
        auto split_request = buildSplitViewRequest(ctx, res, split_info);
        if (!split_request) {
            res.split_view_executed = false;
            return;
        }

        res.split_info = std::move(split_info);
        res.split_info.enabled = true;

        auto render_lock = acquireRenderLock(ctx);

        auto result = engine.renderSplitViewGpuFrame(*split_request);
        render_lock.reset();

        if (result) {
            res.cached_metadata = makeCachedRenderMetadata(result->metadata);
            res.cached_gpu_frame = result->frame;
            res.cached_result_size = ctx.render_size;
            res.split_view_executed = true;
        } else {
            LOG_ERROR("Failed to render split view: {}", result.error());
            res.cached_metadata = {};
            res.cached_gpu_frame.reset();
            res.cached_result_size = {0, 0};
        }
    }

    std::optional<lfs::rendering::SplitViewRequest>
    SplitViewPass::buildSplitViewRequest(const FrameContext& ctx, const FrameResources& res, SplitViewInfo& split_info) {
        const auto& settings = ctx.settings;
        if (settings.split_view_mode == SplitViewMode::Disabled || !ctx.scene_manager)
            return std::nullopt;

        const auto viewport_data = ctx.makeViewportData();

        if (settings.split_view_mode == SplitViewMode::GTComparison) {
            if (!res.gt_context || !res.gt_context->valid() ||
                !res.cached_gpu_frame || !res.cached_gpu_frame->valid())
                return std::nullopt;

            const unsigned int rendered_texture = res.cached_gpu_frame->color.id;
            // GT compare consumes the pre-rendered GpuFrame directly. Match the old cached-render
            // contract by sampling only the GT-sized content region, but keep the cached-render
            // orientation unchanged so the split-view compositor does not apply an extra Y flip.
            const glm::vec2 rendered_texcoord_scale = res.gt_context->render_texcoord_scale;

            auto letterbox_viewport = viewport_data;
            letterbox_viewport.size = ctx.render_size;

            const auto disabled_uids = ctx.scene_manager->getScene().getTrainingDisabledCameraUids();
            const bool cam_disabled = ctx.current_camera_id >= 0 && disabled_uids.count(ctx.current_camera_id) > 0;
            std::string gt_label = cam_disabled ? "Ground Truth (Excluded from Training)" : "Ground Truth";
            split_info.left_name = gt_label;
            split_info.right_name = "Rendered";

            return lfs::rendering::SplitViewRequest{
                .panels =
                    std::array<lfs::rendering::SplitViewPanel, 2>{
                        lfs::rendering::SplitViewPanel{
                            .content =
                                {.type = lfs::rendering::PanelContentType::Image2D,
                                 .model = nullptr,
                                 .model_transform = glm::mat4(1.0f),
                                 .gaussian_render = std::nullopt,
                                 .point_cloud_render = std::nullopt,
                                 .texture_id = res.gt_context->gt_texture_id},
                            .presentation =
                                {.start_position = 0.0f,
                                 .end_position = settings.split_position,
                                 .texcoord_scale = res.gt_context->gt_texcoord_scale,
                                 .flip_y = res.gt_context->gt_needs_flip}},
                        lfs::rendering::SplitViewPanel{
                            .content =
                                {.type = lfs::rendering::PanelContentType::CachedRender,
                                 .model = nullptr,
                                 .model_transform = glm::mat4(1.0f),
                                 .gaussian_render = std::nullopt,
                                 .point_cloud_render = std::nullopt,
                                 .texture_id = rendered_texture},
                            .presentation =
                                {.start_position = settings.split_position,
                                 .end_position = 1.0f,
                                 .texcoord_scale = rendered_texcoord_scale,
                                 .flip_y = std::nullopt}}},
                .composite =
                    {.output_size = letterbox_viewport.size,
                     .background_color = settings.background_color},
                .presentation =
                    {.divider_color = glm::vec4(1.0f, 0.85f, 0.0f, 1.0f),
                     .letterbox = true,
                     .content_size = res.gt_context->dimensions}};
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
            split_info.left_name = visible_nodes[left_idx]->name;
            split_info.right_name = visible_nodes[right_idx]->name;

            return lfs::rendering::SplitViewRequest{
                .panels = std::array<lfs::rendering::SplitViewPanel, 2>{
                    lfs::rendering::SplitViewPanel{
                        .content =
                            buildModelPanelContent(
                                ctx,
                                *visible_nodes[left_idx]->model,
                                scene.getWorldTransform(visible_nodes[left_idx]->id)),
                        .presentation =
                            {.start_position = 0.0f,
                             .end_position = settings.split_position,
                             .texcoord_scale = texcoord_scale,
                             .flip_y = std::nullopt}},
                    lfs::rendering::SplitViewPanel{
                        .content =
                            buildModelPanelContent(
                                ctx,
                                *visible_nodes[right_idx]->model,
                                scene.getWorldTransform(visible_nodes[right_idx]->id)),
                        .presentation =
                            {.start_position = settings.split_position,
                             .end_position = 1.0f,
                             .texcoord_scale = texcoord_scale,
                             .flip_y = std::nullopt}}},
                .composite = {.output_size = viewport_data.size, .background_color = settings.background_color},
                .presentation = {.divider_color = glm::vec4(1.0f, 0.85f, 0.0f, 1.0f)}};
        }

        return std::nullopt;
    }

} // namespace lfs::vis

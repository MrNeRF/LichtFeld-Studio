/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "render_frame_coordinator.hpp"
#include "core/logger.hpp"
#include "core/splat_data.hpp"
#include "core/tensor.hpp"
#include "model_renderability.hpp"
#include "passes/point_cloud_pass.hpp"
#include "passes/splat_raster_pass.hpp"
#include "render_pass.hpp"
#include "scene/scene_manager.hpp"
#include "scene/scene_render_state.hpp"
#include <glad/glad.h>
#include <string_view>
#include <vector>

namespace lfs::vis {

    namespace {

        void executeTimedPass(RenderPass& pass,
                              lfs::rendering::RenderingEngine& engine,
                              const FrameContext& frame_ctx,
                              FrameResources& resources,
                              const std::string_view phase = {}) {
            std::string timer_name = "RenderPass::";
            timer_name += pass.name();
            if (!phase.empty()) {
                timer_name += "[";
                timer_name += phase;
                timer_name += "]";
            }

            lfs::core::ScopedTimer timer(std::move(timer_name));
            pass.execute(engine, frame_ctx, resources);
        }

        [[nodiscard]] std::vector<FrameViewPanel> buildFrameViewPanels(
            const RenderFrameCoordinator::Context& context,
            const SplitViewService& split_view_service,
            const glm::ivec2& render_size) {
            std::vector<FrameViewPanel> panels;
            if (render_size.x <= 0 || render_size.y <= 0) {
                return panels;
            }

            panels.push_back({
                .panel = SplitViewPanelId::Left,
                .viewport = &context.viewport,
                .render_size = render_size,
                .viewport_offset = {0, 0},
                .start_position = 0.0f,
                .end_position = 1.0f,
                .grid_plane = context.grid_planes[splitViewPanelIndex(SplitViewPanelId::Left)],
            });

            const auto layouts = split_view_service.panelLayouts(context.settings, render_size.x);
            if (!layouts || render_size.x <= 1 || render_size.y <= 0) {
                return panels;
            }

            panels.clear();
            panels.reserve(layouts->size());

            panels.push_back({
                .panel = SplitViewPanelId::Left,
                .viewport = &context.viewport,
                .render_size = {std::max((*layouts)[0].width, 1), render_size.y},
                .viewport_offset = {(*layouts)[0].x, 0},
                .start_position = (*layouts)[0].start_position,
                .end_position = (*layouts)[0].end_position,
                .grid_plane = context.grid_planes[splitViewPanelIndex(SplitViewPanelId::Left)],
            });
            panels.push_back({
                .panel = SplitViewPanelId::Right,
                .viewport = &split_view_service.secondaryViewport(),
                .render_size = {std::max((*layouts)[1].width, 1), render_size.y},
                .viewport_offset = {(*layouts)[1].x, 0},
                .start_position = (*layouts)[1].start_position,
                .end_position = (*layouts)[1].end_position,
                .grid_plane = context.grid_planes[splitViewPanelIndex(SplitViewPanelId::Right)],
            });
            return panels;
        }

    } // namespace

    RenderFrameCoordinator::Result RenderFrameCoordinator::execute(const Context& context) {
        LOG_TIMER_TRACE("RenderFrameCoordinator::execute");

        dependencies_.render_count++;
        LOG_TRACE("Render #{}", dependencies_.render_count);

        glm::ivec2 render_size = context.viewport.windowSize;
        glm::ivec2 viewport_pos(0, 0);
        if (context.viewport_region) {
            render_size = glm::ivec2(
                static_cast<int>(context.viewport_region->width),
                static_cast<int>(context.viewport_region->height));
            const int gl_y = context.viewport.frameBufferSize.y -
                             static_cast<int>(context.viewport_region->y) -
                             static_cast<int>(context.viewport_region->height);
            viewport_pos = glm::ivec2(static_cast<int>(context.viewport_region->x), gl_y);
        }

        glClearColor(context.settings.background_color.r, context.settings.background_color.g,
                     context.settings.background_color.b, 1.0f);
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

        const bool count_frame = context.frame_dirty != 0;
        if (count_frame) {
            dependencies_.framerate_controller.beginFrame();
        }

        SceneRenderState scene_state;
        if (context.scene_manager) {
            scene_state = context.scene_manager->buildRenderState();
        }

        const bool has_splats = hasRenderableGaussians(context.model);
        const bool has_point_cloud = scene_state.point_cloud && scene_state.point_cloud->size() > 0;
        if (!has_point_cloud) {
            dependencies_.pass_graph.resetPointCloudCache();
        }

        const auto view_panels = buildFrameViewPanels(context, dependencies_.split_view_service, render_size);
        if (view_panels.empty()) {
            return Result{};
        }

        std::vector<ViewportInteractionPanel> interaction_panels;
        interaction_panels.reserve(view_panels.size());
        for (const auto& panel : view_panels) {
            if (!panel.valid()) {
                continue;
            }
            interaction_panels.push_back({
                .panel = panel.panel,
                .viewport_data =
                    {.rotation = panel.viewport->getRotationMatrix(),
                     .translation = panel.viewport->getTranslation(),
                     .size = panel.render_size,
                     .focal_length_mm = context.settings.focal_length_mm,
                     .orthographic = context.settings.orthographic,
                     .ortho_scale = context.settings.ortho_scale},
                .viewport_pos =
                    glm::vec2(static_cast<float>(viewport_pos.x + panel.viewport_offset.x),
                              static_cast<float>(viewport_pos.y + panel.viewport_offset.y)),
                .viewport_size =
                    glm::vec2(static_cast<float>(panel.render_size.x), static_cast<float>(panel.render_size.y)),
            });
        }
        dependencies_.viewport_interaction_context.updatePickContext(interaction_panels);

        const FrameContext frame_ctx{
            .viewport = context.viewport,
            .viewport_region = context.viewport_region,
            .render_lock_held = context.render_lock_held,
            .scene_manager = context.scene_manager,
            .model = context.model,
            .scene_state = std::move(scene_state),
            .settings = context.settings,
            .render_size = render_size,
            .viewport_pos = viewport_pos,
            .frame_dirty = context.frame_dirty,
            .cursor_preview = dependencies_.viewport_overlay.cursorPreview(),
            .gizmo = dependencies_.viewport_overlay.makeFrameGizmoState(),
            .hovered_camera_id = context.hovered_camera_id,
            .current_camera_id = context.current_camera_id,
            .hovered_gaussian_id = dependencies_.viewport_overlay.hoveredGaussianId(),
            .selection_flash_intensity = context.selection_flash_intensity,
            .view_panels = view_panels};

        FrameResources resources{
            .cached_metadata = dependencies_.viewport_artifacts.cachedMetadata(),
            .cached_gpu_frame = dependencies_.viewport_artifacts.gpuFrame(),
            .cached_result_size = dependencies_.viewport_artifacts.renderedSize(),
            .gt_context = dependencies_.split_view_service.gtContext(),
            .hovered_gaussian_id = dependencies_.viewport_overlay.hoveredGaussianId(),
            .split_info = {},
            .additional_dirty = 0,
            .pivot_animation_end = std::nullopt};

        if (!has_splats && !has_point_cloud) {
            const bool had_cached_output = dependencies_.viewport_artifacts.hasOutputArtifacts();
            if (had_cached_output) {
                resources.cached_metadata = {};
                resources.cached_gpu_frame.reset();
                resources.cached_result_size = {0, 0};
                resources.hovered_gaussian_id = -1;
                lfs::core::Tensor::trim_memory_pool();
            }
        }

        if (splitViewUsesGTComparison(frame_ctx.settings.split_view_mode) &&
            resources.gt_context && resources.gt_context->valid()) {
            auto* const splat_raster_pass = dependencies_.pass_graph.splatRasterPass();
            auto* const point_cloud_pass = dependencies_.pass_graph.pointCloudPass();
            const bool needs_gt_pre_render =
                !(resources.cached_gpu_frame && resources.cached_gpu_frame->valid()) ||
                (has_splats && splat_raster_pass && (context.frame_dirty & splat_raster_pass->sensitivity())) ||
                (has_point_cloud && point_cloud_pass && (context.frame_dirty & point_cloud_pass->sensitivity()));

            if (needs_gt_pre_render) {
                if (has_splats && splat_raster_pass) {
                    executeTimedPass(*splat_raster_pass, dependencies_.engine, frame_ctx, resources, "gt_pre");
                    resources.splat_pre_rendered = true;
                } else if (has_point_cloud && point_cloud_pass) {
                    executeTimedPass(*point_cloud_pass, dependencies_.engine, frame_ctx, resources, "gt_pre");
                    resources.splat_pre_rendered = true;
                }
            }
        }

        for (auto& pass : dependencies_.pass_graph.passes()) {
            if (pass->shouldExecute(context.frame_dirty, frame_ctx)) {
                executeTimedPass(*pass, dependencies_.engine, frame_ctx, resources);
            }
        }

        const bool viewport_output_updated =
            (context.frame_dirty & (DirtyFlag::SPLATS | DirtyFlag::CAMERA | DirtyFlag::VIEWPORT |
                                    DirtyFlag::SELECTION | DirtyFlag::BACKGROUND | DirtyFlag::PPISP |
                                    DirtyFlag::SPLIT_VIEW)) != 0;
        dependencies_.viewport_artifacts.updateFromFrameResources(resources, viewport_output_updated);
        dependencies_.viewport_overlay.setHoveredGaussianId(resources.hovered_gaussian_id);
        dependencies_.split_view_service.updateInfo(resources);

        if (count_frame) {
            dependencies_.framerate_controller.endFrame();
        }

        return Result{
            .additional_dirty = resources.additional_dirty,
            .pivot_animation_end = resources.pivot_animation_end};
    }

} // namespace lfs::vis

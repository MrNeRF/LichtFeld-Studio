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

        dependencies_.viewport_interaction_context.viewport_data = lfs::rendering::ViewportData{
            .rotation = context.viewport.getRotationMatrix(),
            .translation = context.viewport.getTranslation(),
            .size = render_size,
            .focal_length_mm = context.settings.focal_length_mm,
            .orthographic = context.settings.orthographic,
            .ortho_scale = context.settings.ortho_scale};

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
            .selection_flash_intensity = context.selection_flash_intensity};

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

        if (frame_ctx.settings.split_view_mode == SplitViewMode::GTComparison &&
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

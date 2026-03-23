/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "splat_raster_pass.hpp"
#include "../model_renderability.hpp"
#include "../viewport_appearance_correction.hpp"
#include "../viewport_request_builder.hpp"
#include "core/logger.hpp"
#include "core/splat_data.hpp"
#include <cassert>

namespace lfs::vis {
    bool SplatRasterPass::shouldExecute(DirtyMask frame_dirty, const FrameContext& ctx) const {
        if (!hasRenderableGaussians(ctx.model))
            return false;
        return (frame_dirty & sensitivity()) != 0;
    }

    void SplatRasterPass::execute(lfs::rendering::RenderingEngine& engine,
                                  const FrameContext& ctx,
                                  FrameResources& res) {
        if (res.split_view_executed || res.splat_pre_rendered)
            return;

        renderToTexture(engine, ctx, res);
    }

    void SplatRasterPass::renderToTexture(lfs::rendering::RenderingEngine& engine,
                                          const FrameContext& ctx, FrameResources& res) {
        LOG_TIMER_TRACE("SplatRasterPass::renderToTexture");
        assert(hasRenderableGaussians(ctx.model));

        const auto& settings = ctx.settings;

        glm::ivec2 viewport_size = ctx.viewport.windowSize;
        if (ctx.viewport_region) {
            viewport_size = glm::ivec2(
                static_cast<int>(ctx.viewport_region->width),
                static_cast<int>(ctx.viewport_region->height));
        }

        const float scale = std::clamp(settings.render_scale, 0.25f, 1.0f);
        glm::ivec2 render_size(
            static_cast<int>(viewport_size.x * scale),
            static_cast<int>(viewport_size.y * scale));

        if (settings.split_view_mode == SplitViewMode::GTComparison && res.gt_context && res.gt_context->valid()) {
            render_size = res.gt_context->dimensions;
        }

        std::optional<lfs::rendering::ViewportRenderRequest> gaussian_request;
        std::optional<lfs::rendering::PointCloudRenderRequest> point_cloud_request;
        if (settings.point_cloud_mode) {
            point_cloud_request = buildPointCloudRenderRequest(ctx, ctx.render_size, ctx.scene_state.model_transforms);
            point_cloud_request->frame_view.size = render_size;
            point_cloud_request->frame_view.background_color = ctx.makeFrameView().background_color;
        } else {
            gaussian_request = buildViewportRenderRequest(ctx, render_size);
        }

        const bool need_hovered_output =
            (ctx.cursor_preview.selection_mode == SelectionPreviewMode::Rings) && ctx.cursor_preview.active;
        std::optional<lfs::rendering::HoveredGaussianQueryRequest> hovered_query_request;
        if (need_hovered_output) {
            hovered_query_request = buildHoveredGaussianQueryRequest(ctx, render_size);
        }

        auto render_lock = acquireRenderLock(ctx);

        std::optional<lfs::rendering::GaussianGpuFrameResult> viewport_frame;
        std::string render_error;

        if (settings.point_cloud_mode) {
            if (settings.apply_appearance_correction) {
                auto image_result = engine.renderPointCloudImage(*ctx.model, *point_cloud_request);
                if (!image_result) {
                    render_error = image_result.error();
                } else {
                    if (image_result->image) {
                        image_result->image = applyViewportAppearanceCorrection(
                            std::move(image_result->image),
                            ctx.scene_manager,
                            settings,
                            ctx.current_camera_id);
                    }

                    if (!image_result->image) {
                        render_error = "Point-cloud image render returned no image payload";
                    } else {
                        auto refreshed_gpu_frame = engine.materializeGpuFrame(
                            image_result->image, image_result->metadata, render_size);
                        if (!refreshed_gpu_frame) {
                            render_error = refreshed_gpu_frame.error();
                        } else {
                            viewport_frame = lfs::rendering::GaussianGpuFrameResult{
                                .frame = *refreshed_gpu_frame,
                                .metadata = std::move(image_result->metadata)};
                        }
                    }
                }
            } else {
                auto gpu_frame = engine.renderPointCloudGpuFrame(*ctx.model, *point_cloud_request);
                if (!gpu_frame) {
                    render_error = gpu_frame.error();
                } else {
                    viewport_frame = lfs::rendering::GaussianGpuFrameResult{
                        .frame = *gpu_frame,
                        .metadata =
                            {.depth = nullptr,
                             .depth_right = nullptr,
                             .valid = true,
                             .depth_is_ndc = gpu_frame->depth_is_ndc,
                             .external_depth_texture = gpu_frame->depth.valid() ? gpu_frame->depth.id : 0,
                             .near_plane = gpu_frame->near_plane,
                             .far_plane = gpu_frame->far_plane,
                             .orthographic = gpu_frame->orthographic}};
                }
            }
        } else if (settings.apply_appearance_correction) {
            auto image_result = engine.renderGaussiansImage(*ctx.model, *gaussian_request);
            if (!image_result) {
                render_error = image_result.error();
            } else {
                if (image_result->image) {
                    image_result->image = applyViewportAppearanceCorrection(
                        std::move(image_result->image),
                        ctx.scene_manager,
                        settings,
                        ctx.current_camera_id);
                }

                if (!image_result->image) {
                    render_error = "Gaussian image render returned no image payload";
                } else {
                    auto refreshed_gpu_frame = engine.materializeGpuFrame(
                        image_result->image, image_result->metadata, render_size);
                    if (!refreshed_gpu_frame) {
                        render_error = refreshed_gpu_frame.error();
                    } else {
                        viewport_frame = lfs::rendering::GaussianGpuFrameResult{
                            .frame = *refreshed_gpu_frame,
                            .metadata = std::move(image_result->metadata)};
                    }
                }
            }
        } else {
            auto gpu_frame = engine.renderGaussiansGpuFrame(*ctx.model, *gaussian_request);
            if (!gpu_frame) {
                render_error = gpu_frame.error();
            } else {
                viewport_frame = std::move(*gpu_frame);
            }
        }

        std::optional<int> hovered_gaussian_id;
        if (viewport_frame && hovered_query_request) {
            auto hovered_result = engine.queryHoveredGaussianId(*ctx.model, *hovered_query_request);
            if (!hovered_result) {
                LOG_WARN("Failed to query hovered gaussian id: {}", hovered_result.error());
            } else {
                hovered_gaussian_id = *hovered_result;
            }
        }

        render_lock.reset();

        if (viewport_frame) {
            res.cached_metadata = makeCachedRenderMetadata(viewport_frame->metadata);
            res.cached_gpu_frame = viewport_frame->frame;
            res.hovered_gaussian_id = hovered_gaussian_id.value_or(-1);
            res.cached_result_size = render_size;
        } else {
            LOG_ERROR("Failed to render gaussians: {}", render_error);
            res.cached_metadata = {};
            res.cached_gpu_frame.reset();
            res.cached_result_size = {0, 0};
        }
    }

} // namespace lfs::vis

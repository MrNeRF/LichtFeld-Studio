/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "point_cloud_pass.hpp"
#include "../model_renderability.hpp"
#include "../viewport_request_builder.hpp"
#include "core/logger.hpp"
#include "core/point_cloud.hpp"
#include "core/splat_data.hpp"
#include "scene/scene_manager.hpp"
#include <cassert>

namespace lfs::vis {

    bool PointCloudPass::shouldExecute(DirtyMask frame_dirty, const FrameContext& ctx) const {
        if (hasRenderableGaussians(ctx.model))
            return false;
        if (!ctx.scene_manager)
            return false;
        return (frame_dirty & sensitivity()) != 0;
    }

    void PointCloudPass::execute(lfs::rendering::RenderingEngine& engine,
                                 const FrameContext& ctx,
                                 FrameResources& res) {
        if (res.split_view_executed)
            return;

        assert(ctx.scene_manager);

        const auto& scene_state = ctx.scene_state;

        if (!scene_state.point_cloud && cached_source_point_cloud_) {
            cached_filtered_point_cloud_.reset();
            cached_source_point_cloud_ = nullptr;
        }

        if (!scene_state.point_cloud || scene_state.point_cloud->size() == 0)
            return;

        const lfs::core::PointCloud* point_cloud_to_render = scene_state.point_cloud;

        for (const auto& cb : scene_state.cropboxes) {
            if (!cb.data || (!cb.data->enabled && !ctx.settings.show_crop_box))
                continue;

            const bool cache_valid = cached_filtered_point_cloud_ &&
                                     cached_source_point_cloud_ == scene_state.point_cloud &&
                                     cached_cropbox_transform_ == cb.world_transform &&
                                     cached_cropbox_min_ == cb.data->min &&
                                     cached_cropbox_max_ == cb.data->max &&
                                     cached_cropbox_inverse_ == cb.data->inverse;

            if (!cache_valid) {
                const auto& means = scene_state.point_cloud->means;
                const auto& colors = scene_state.point_cloud->colors;
                const glm::mat4 m = glm::inverse(cb.world_transform);
                const auto device = means.device();

                // R (3x3) and t (3,) from the inverse transform — avoids homogeneous expansion
                const auto R = lfs::core::Tensor::from_vector(
                    {m[0][0], m[1][0], m[2][0],
                     m[0][1], m[1][1], m[2][1],
                     m[0][2], m[1][2], m[2][2]},
                    {3, 3}, device);
                const auto t = lfs::core::Tensor::from_vector(
                    {m[3][0], m[3][1], m[3][2]}, {1, 3}, device);

                // local_pos = means @ R + t  — shape [N, 3], no homogeneous coords
                const auto local_pos = means.mm(R) + t;

                const auto x = local_pos.slice(1, 0, 1).squeeze(1);
                const auto y = local_pos.slice(1, 1, 2).squeeze(1);
                const auto z = local_pos.slice(1, 2, 3).squeeze(1);

                auto mask = (x >= cb.data->min.x) && (x <= cb.data->max.x) &&
                            (y >= cb.data->min.y) && (y <= cb.data->max.y) &&
                            (z >= cb.data->min.z) && (z <= cb.data->max.z);
                if (cb.data->inverse)
                    mask = mask.logical_not();

                const auto indices = mask.nonzero().squeeze(1);
                if (indices.size(0) > 0) {
                    cached_filtered_point_cloud_ = std::make_unique<lfs::core::PointCloud>(
                        means.index_select(0, indices), colors.index_select(0, indices));
                } else {
                    cached_filtered_point_cloud_.reset();
                }

                cached_source_point_cloud_ = scene_state.point_cloud;
                cached_cropbox_transform_ = cb.world_transform;
                cached_cropbox_min_ = cb.data->min;
                cached_cropbox_max_ = cb.data->max;
                cached_cropbox_inverse_ = cb.data->inverse;
            }

            if (cached_filtered_point_cloud_) {
                point_cloud_to_render = cached_filtered_point_cloud_.get();
            } else {
                return;
            }
            break;
        }

        LOG_TRACE("Rendering point cloud with {} points", point_cloud_to_render->size());

        glm::mat4 point_cloud_transform(1.0f);
        if (!scene_state.model_transforms.empty()) {
            point_cloud_transform = scene_state.model_transforms[0];
        }
        const std::vector<glm::mat4> pc_transforms = {point_cloud_transform};
        const auto pc_request = buildPointCloudRenderRequest(ctx, ctx.render_size, pc_transforms);

        if (ctx.settings.split_view_mode == SplitViewMode::GTComparison &&
            res.gt_context && res.gt_context->valid()) {
            renderToTexture(engine, ctx, res, *point_cloud_to_render, pc_transforms, pc_request,
                            res.gt_context->dimensions);
            return;
        }

        renderToTexture(engine, ctx, res, *point_cloud_to_render, pc_transforms, pc_request,
                        ctx.render_size);
    }

    void PointCloudPass::renderToTexture(lfs::rendering::RenderingEngine& engine,
                                         const FrameContext& /*ctx*/,
                                         FrameResources& res,
                                         const lfs::core::PointCloud& point_cloud,
                                         const std::vector<glm::mat4>& pc_transforms,
                                         const lfs::rendering::PointCloudRenderRequest& request,
                                         const glm::ivec2 render_size) {
        auto request_for_texture = request;
        request_for_texture.frame_view.size = render_size;
        request_for_texture.scene.model_transforms = &pc_transforms;

        auto gpu_frame_result = engine.renderPointCloudGpuFrame(point_cloud, request_for_texture);
        if (gpu_frame_result) {
            res.cached_metadata = {};
            res.cached_gpu_frame = *gpu_frame_result;
            res.cached_result_size = render_size;
        } else {
            LOG_ERROR("Failed to render point cloud GPU frame: {}", gpu_frame_result.error());
            res.cached_metadata = {};
            res.cached_gpu_frame.reset();
            res.cached_result_size = {0, 0};
        }
    }

    void PointCloudPass::resetCache() {
        cached_filtered_point_cloud_.reset();
        cached_source_point_cloud_ = nullptr;
    }

} // namespace lfs::vis

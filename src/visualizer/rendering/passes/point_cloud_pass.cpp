/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "point_cloud_pass.hpp"
#include "core/logger.hpp"
#include "core/point_cloud.hpp"
#include "core/splat_data.hpp"
#include "scene/scene_manager.hpp"
#include <cassert>
#include <glad/glad.h>

namespace lfs::vis {

    bool PointCloudPass::shouldExecute(DirtyMask frame_dirty, const FrameContext& ctx) const {
        if (ctx.model && ctx.model->size() > 0)
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

        const lfs::rendering::ViewportData viewport_data{
            .rotation = ctx.viewport.getRotationMatrix(),
            .translation = ctx.viewport.getTranslation(),
            .size = ctx.render_size,
            .focal_length_mm = ctx.settings.focal_length_mm,
            .orthographic = ctx.settings.orthographic,
            .ortho_scale = ctx.settings.ortho_scale};

        std::optional<lfs::rendering::BoundingBox> crop_box;
        bool crop_inverse = false;
        bool crop_desaturate = false;
        for (const auto& cb : scene_state.cropboxes) {
            if (!cb.data || (!cb.data->enabled && !ctx.settings.show_crop_box))
                continue;
            crop_box = lfs::rendering::BoundingBox{
                .min = cb.data->min,
                .max = cb.data->max,
                .transform = glm::inverse(cb.world_transform)};
            crop_inverse = cb.data->inverse;
            crop_desaturate = ctx.settings.show_crop_box && !ctx.settings.use_crop_box && ctx.settings.desaturate_cropping;
            break;
        }

        const lfs::rendering::RenderRequest pc_request{
            .viewport = viewport_data,
            .scaling_modifier = ctx.settings.scaling_modifier,
            .mip_filter = ctx.settings.mip_filter,
            .sh_degree = 0,
            .background_color = ctx.settings.background_color,
            .crop_box = crop_box,
            .point_cloud_mode = true,
            .voxel_size = ctx.settings.voxel_size,
            .equirectangular = ctx.settings.equirectangular,
            .model_transforms = {point_cloud_transform},
            .crop_inverse = crop_inverse,
            .crop_desaturate = crop_desaturate};

        auto render_result = engine.renderPointCloud(*point_cloud_to_render, pc_request);
        if (render_result) {
            res.cached_result = *render_result;
            res.cached_result_size = ctx.render_size;

            glViewport(ctx.viewport_pos.x, ctx.viewport_pos.y, ctx.render_size.x, ctx.render_size.y);
            glClearColor(ctx.settings.background_color.r, ctx.settings.background_color.g,
                         ctx.settings.background_color.b, 1.0f);
            glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

            const auto present_result = engine.presentToScreen(
                res.cached_result, ctx.viewport_pos, res.cached_result_size);
            if (present_result) {
                res.splats_presented = true;
            } else {
                LOG_ERROR("Failed to present point cloud: {}", present_result.error());
            }
        } else {
            LOG_ERROR("Failed to render point cloud: {}", render_result.error());
        }
    }

    void PointCloudPass::resetCache() {
        cached_filtered_point_cloud_.reset();
        cached_source_point_cloud_ = nullptr;
    }

} // namespace lfs::vis

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

    PointCloudPass::~PointCloudPass() {
        if (render_depth_rbo_ != 0) {
            glDeleteRenderbuffers(1, &render_depth_rbo_);
        }
        if (render_fbo_ != 0) {
            glDeleteFramebuffers(1, &render_fbo_);
        }
    }

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
        const std::vector<glm::mat4> pc_transforms = {point_cloud_transform};

        const auto viewport_data = ctx.makeViewportData();

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
            .model_transforms = &pc_transforms,
            .crop_inverse = crop_inverse,
            .crop_desaturate = crop_desaturate};

        if (ctx.settings.split_view_mode == SplitViewMode::GTComparison &&
            res.gt_context && res.gt_context->valid()) {
            renderToTexture(engine, ctx, res, *point_cloud_to_render, pc_transforms, pc_request);
            return;
        }

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

    void PointCloudPass::renderToTexture(lfs::rendering::RenderingEngine& engine,
                                         const FrameContext& ctx,
                                         FrameResources& res,
                                         const lfs::core::PointCloud& point_cloud,
                                         const std::vector<glm::mat4>& pc_transforms,
                                         const lfs::rendering::RenderRequest& request) {
        if (!res.gt_context || !res.gt_context->valid()) {
            return;
        }

        const glm::ivec2 render_size = res.gt_context->dimensions;
        const glm::ivec2 alloc_size(
            ((render_size.x + GPU_ALIGNMENT - 1) / GPU_ALIGNMENT) * GPU_ALIGNMENT,
            ((render_size.y + GPU_ALIGNMENT - 1) / GPU_ALIGNMENT) * GPU_ALIGNMENT);

        if (alloc_size != texture_size_) {
            glBindTexture(GL_TEXTURE_2D, ctx.cached_render_texture);
            glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, alloc_size.x, alloc_size.y,
                         0, GL_RGBA, GL_UNSIGNED_BYTE, nullptr);
            texture_size_ = alloc_size;
        }

        if (render_fbo_ == 0) {
            glGenFramebuffers(1, &render_fbo_);
            glGenRenderbuffers(1, &render_depth_rbo_);
        }

        GLint current_fbo;
        glGetIntegerv(GL_FRAMEBUFFER_BINDING, &current_fbo);
        GLint saved_viewport[4];
        glGetIntegerv(GL_VIEWPORT, saved_viewport);
        const GLboolean scissor_was_enabled = glIsEnabled(GL_SCISSOR_TEST);

        glBindFramebuffer(GL_FRAMEBUFFER, render_fbo_);
        glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, ctx.cached_render_texture, 0);
        glDisable(GL_SCISSOR_TEST);

        if (alloc_size != depth_buffer_size_) {
            glBindRenderbuffer(GL_RENDERBUFFER, render_depth_rbo_);
            glRenderbufferStorage(GL_RENDERBUFFER, GL_DEPTH_COMPONENT24, alloc_size.x, alloc_size.y);
            depth_buffer_size_ = alloc_size;
        }
        glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_RENDERBUFFER, render_depth_rbo_);

        if (const GLenum fb_status = glCheckFramebufferStatus(GL_FRAMEBUFFER);
            fb_status != GL_FRAMEBUFFER_COMPLETE) {
            LOG_ERROR("Point cloud GT FBO incomplete: 0x{:x}", fb_status);
            glBindFramebuffer(GL_FRAMEBUFFER, current_fbo);
            res.render_texture_valid = false;
            return;
        }

        glViewport(0, 0, render_size.x, render_size.y);
        glClearColor(ctx.settings.background_color.r, ctx.settings.background_color.g,
                     ctx.settings.background_color.b, 1.0f);
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

        auto request_for_texture = request;
        request_for_texture.viewport.size = render_size;
        request_for_texture.model_transforms = &pc_transforms;

        auto render_result = engine.renderPointCloud(point_cloud, request_for_texture);
        if (render_result) {
            res.cached_result = *render_result;
            res.cached_result_size = render_size;

            const auto present_result = engine.presentToScreen(res.cached_result, glm::ivec2(0), render_size);
            res.render_texture_valid = present_result.has_value();
            if (!present_result) {
                LOG_ERROR("Failed to present point cloud GT render: {}", present_result.error());
            }
        } else {
            LOG_ERROR("Failed to render point cloud: {}", render_result.error());
            res.render_texture_valid = false;
            res.cached_result_size = {0, 0};
        }

        glBindFramebuffer(GL_FRAMEBUFFER, current_fbo);
        glViewport(saved_viewport[0], saved_viewport[1], saved_viewport[2], saved_viewport[3]);
        if (scissor_was_enabled) {
            glEnable(GL_SCISSOR_TEST);
        }
    }

    void PointCloudPass::resetCache() {
        cached_filtered_point_cloud_.reset();
        cached_source_point_cloud_ = nullptr;
    }

} // namespace lfs::vis

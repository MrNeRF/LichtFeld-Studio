/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "splat_raster_pass.hpp"
#include "core/cuda_debug.hpp"
#include "core/logger.hpp"
#include "core/splat_data.hpp"
#include "geometry/euclidean_transform.hpp"
#include "scene/scene_manager.hpp"
#include "training/components/ppisp.hpp"
#include "training/components/ppisp_controller.hpp"
#include "training/trainer.hpp"
#include "training/training_manager.hpp"
#include <cassert>
#include <cuda_runtime.h>
#include <glad/glad.h>

namespace lfs::vis {

    namespace {
        lfs::training::PPISPRenderOverrides toRenderOverrides(const PPISPOverrides& ov) {
            lfs::training::PPISPRenderOverrides r;
            r.exposure_offset = ov.exposure_offset;
            r.vignette_enabled = ov.vignette_enabled;
            r.vignette_strength = ov.vignette_strength;
            r.wb_temperature = ov.wb_temperature;
            r.wb_tint = ov.wb_tint;
            r.color_red_x = ov.color_red_x;
            r.color_red_y = ov.color_red_y;
            r.color_green_x = ov.color_green_x;
            r.color_green_y = ov.color_green_y;
            r.color_blue_x = ov.color_blue_x;
            r.color_blue_y = ov.color_blue_y;
            r.gamma_multiplier = ov.gamma_multiplier;
            r.gamma_red = ov.gamma_red;
            r.gamma_green = ov.gamma_green;
            r.gamma_blue = ov.gamma_blue;
            r.crf_toe = ov.crf_toe;
            r.crf_shoulder = ov.crf_shoulder;
            return r;
        }

        lfs::core::Tensor applyStandaloneAppearance(const lfs::core::Tensor& rgb, SceneManager& scene_mgr,
                                                    const int camera_uid, const PPISPOverrides& overrides,
                                                    const bool use_controller = true) {
            auto* ppisp = scene_mgr.getAppearancePPISP();
            if (!ppisp) {
                return rgb;
            }

            const bool was_hwc = (rgb.ndim() == 3 && rgb.shape()[2] == 3);
            const auto input = was_hwc ? rgb.permute({2, 0, 1}).contiguous() : rgb;
            const bool is_training_camera = (camera_uid >= 0 && camera_uid < ppisp->num_frames());
            const bool has_controller = use_controller && scene_mgr.hasAppearanceController();

            lfs::core::Tensor result;

            if (has_controller) {
                auto* pool = scene_mgr.getAppearanceControllerPool();
                const int controller_idx = camera_uid >= 0 ? camera_uid % pool->num_cameras() : 0;
                const auto params = pool->predict(controller_idx, input.unsqueeze(0), 1.0f);
                result = overrides.isIdentity()
                             ? ppisp->apply_with_controller_params(input, params, 0)
                             : ppisp->apply_with_controller_params_and_overrides(input, params, 0,
                                                                                 toRenderOverrides(overrides));
            } else if (is_training_camera) {
                result = overrides.isIdentity() ? ppisp->apply(input, camera_uid, camera_uid)
                                                : ppisp->apply_with_overrides(input, camera_uid, camera_uid,
                                                                              toRenderOverrides(overrides));
            } else {
                const int fallback_camera = ppisp->any_camera_id();
                const int fallback_frame = ppisp->any_frame_uid();
                result = overrides.isIdentity() ? ppisp->apply(input, fallback_camera, fallback_frame)
                                                : ppisp->apply_with_overrides(input, fallback_camera, fallback_frame,
                                                                              toRenderOverrides(overrides));
            }

            return (was_hwc && result.is_valid()) ? result.permute({1, 2, 0}).contiguous() : result;
        }
    } // namespace

    SplatRasterPass::~SplatRasterPass() {
        if (render_fbo_)
            glDeleteFramebuffers(1, &render_fbo_);
        if (render_depth_rbo_)
            glDeleteRenderbuffers(1, &render_depth_rbo_);
        if (d_hovered_depth_id_)
            cudaFree(d_hovered_depth_id_);
        if (h_hovered_depth_id_)
            cudaFreeHost(h_hovered_depth_id_);
        if (readback_event_)
            cudaEventDestroy(readback_event_);
    }

    bool SplatRasterPass::shouldExecute(DirtyMask frame_dirty, const FrameContext& ctx) const {
        if (!ctx.model || ctx.model->size() == 0)
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
        assert(ctx.model && ctx.model->size() > 0);

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

        const glm::ivec2 alloc_size(
            ((render_size.x + GPU_ALIGNMENT - 1) / GPU_ALIGNMENT) * GPU_ALIGNMENT,
            ((render_size.y + GPU_ALIGNMENT - 1) / GPU_ALIGNMENT) * GPU_ALIGNMENT);

        if (alloc_size != texture_size_) {
            glBindTexture(GL_TEXTURE_2D, ctx.cached_render_texture);
            glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, alloc_size.x, alloc_size.y,
                         0, GL_RGBA, GL_UNSIGNED_BYTE, nullptr);
            LOG_DEBUG("Render texture resize: {}x{} -> {}x{}", texture_size_.x, texture_size_.y, alloc_size.x, alloc_size.y);
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
            LOG_DEBUG("Depth buffer resize: {}x{}", alloc_size.x, alloc_size.y);
            depth_buffer_size_ = alloc_size;
        }
        glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_RENDERBUFFER, render_depth_rbo_);

        const GLenum fb_status = glCheckFramebufferStatus(GL_FRAMEBUFFER);
        if (fb_status != GL_FRAMEBUFFER_COMPLETE) {
            LOG_ERROR("FBO incomplete: 0x{:x}", fb_status);
            glBindFramebuffer(GL_FRAMEBUFFER, current_fbo);
            res.render_texture_valid = false;
            return;
        }

        glViewport(0, 0, render_size.x, render_size.y);
        glClearColor(settings.background_color.r, settings.background_color.g, settings.background_color.b, 1.0f);
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

        auto viewport_data = ctx.makeViewportData();
        viewport_data.size = render_size;

        const auto& scene_state = ctx.scene_state;

        lfs::rendering::RenderRequest request{
            .viewport = viewport_data,
            .scaling_modifier = settings.scaling_modifier,
            .antialiasing = settings.antialiasing,
            .mip_filter = settings.mip_filter,
            .sh_degree = settings.sh_degree,
            .background_color = settings.background_color,
            .crop_box = std::nullopt,
            .point_cloud_mode = settings.point_cloud_mode,
            .voxel_size = settings.voxel_size,
            .gut = settings.gut,
            .equirectangular = settings.equirectangular,
            .show_rings = settings.show_rings,
            .ring_width = settings.ring_width,
            .show_center_markers = settings.show_center_markers,
            .model_transforms = &scene_state.model_transforms,
            .transform_indices = scene_state.transform_indices,
            .selection_mask = scene_state.selection_mask,
            .output_screen_positions = ctx.brush.output_screen_positions,
            .brush_active = ctx.brush.active,
            .brush_x = ctx.brush.x,
            .brush_y = ctx.brush.y,
            .brush_radius = ctx.brush.radius,
            .brush_add_mode = ctx.brush.add_mode,
            .brush_selection_tensor = ctx.brush.preview_selection ? ctx.brush.preview_selection : ctx.brush.selection_tensor,
            .brush_saturation_mode = ctx.brush.saturation_mode,
            .brush_saturation_amount = ctx.brush.saturation_amount,
            .selection_mode_rings = (ctx.brush.selection_mode == lfs::rendering::SelectionMode::Rings),
            .selected_node_mask = (settings.desaturate_unselected || ctx.selection_flash_intensity > 0.0f)
                                      ? scene_state.selected_node_mask
                                      : std::vector<bool>{},
            .node_visibility_mask = scene_state.node_visibility_mask,
            .desaturate_unselected = settings.desaturate_unselected,
            .selection_flash_intensity = ctx.selection_flash_intensity,
            .hovered_depth_id = nullptr,
            .highlight_gaussian_id = (ctx.brush.selection_mode == lfs::rendering::SelectionMode::Rings) ? ctx.hovered_gaussian_id : -1,
            .far_plane = settings.depth_clip_enabled ? settings.depth_clip_far : lfs::rendering::DEFAULT_FAR_PLANE,
            .orthographic = settings.orthographic,
            .ortho_scale = settings.ortho_scale};

        const bool need_hovered_output = (ctx.brush.selection_mode == lfs::rendering::SelectionMode::Rings) && ctx.brush.active;
        if (need_hovered_output) {
            if (d_hovered_depth_id_ == nullptr) {
                CHECK_CUDA(cudaMalloc(&d_hovered_depth_id_, sizeof(unsigned long long)));
            }
            if (h_hovered_depth_id_ == nullptr) {
                CHECK_CUDA(cudaMallocHost(&h_hovered_depth_id_, sizeof(unsigned long long)));
            }
            if (readback_event_ == nullptr) {
                CHECK_CUDA(cudaEventCreate(&readback_event_));
            }

            // Poll previous async readback
            if (readback_pending_) {
                if (cudaEventQuery(readback_event_) == cudaSuccess) {
                    last_hovered_result_ = *h_hovered_depth_id_;
                    readback_pending_ = false;
                }
            }

            CHECK_CUDA(cudaMemsetAsync(d_hovered_depth_id_, 0xFF, sizeof(unsigned long long)));
            request.hovered_depth_id = d_hovered_depth_id_;
        }

        if ((settings.use_crop_box || settings.show_crop_box) && ctx.scene_manager) {
            const auto& cropboxes = scene_state.cropboxes;
            const size_t idx = (scene_state.selected_cropbox_index >= 0)
                                   ? static_cast<size_t>(scene_state.selected_cropbox_index)
                                   : 0;

            if (idx < cropboxes.size() && cropboxes[idx].data) {
                const auto& cb = cropboxes[idx];
                request.crop_box = lfs::rendering::BoundingBox{
                    .min = cb.data->min,
                    .max = cb.data->max,
                    .transform = glm::inverse(cb.world_transform)};
                request.crop_inverse = cb.data->inverse;
                request.crop_desaturate = settings.show_crop_box && !settings.use_crop_box && settings.desaturate_cropping;
                request.crop_parent_node_index = ctx.scene_manager->getScene().getVisibleNodeIndex(cb.parent_splat_id);
            }
        }

        if ((settings.use_ellipsoid || settings.show_ellipsoid) && ctx.scene_manager) {
            const auto& scene = ctx.scene_manager->getScene();
            const auto visible_ellipsoids = scene.getVisibleEllipsoids();
            const core::NodeId selected_ellipsoid_id = ctx.scene_manager->getSelectedNodeEllipsoidId();
            for (const auto& el : visible_ellipsoids) {
                if (!el.data)
                    continue;
                if (selected_ellipsoid_id != core::NULL_NODE && el.node_id != selected_ellipsoid_id)
                    continue;
                request.ellipsoid = lfs::rendering::Ellipsoid{
                    .radii = el.data->radii,
                    .transform = glm::inverse(el.world_transform)};
                request.ellipsoid_inverse = el.data->inverse;
                request.ellipsoid_desaturate = settings.show_ellipsoid && !settings.use_ellipsoid && settings.desaturate_cropping;
                request.ellipsoid_parent_node_index = scene.getVisibleNodeIndex(el.parent_splat_id);
                break;
            }
        }

        if (settings.depth_filter_enabled) {
            request.depth_filter = lfs::rendering::BoundingBox{
                .min = settings.depth_filter_min,
                .max = settings.depth_filter_max,
                .transform = settings.depth_filter_transform.inv().toMat4()};
        }

        auto render_lock = acquireRenderLock(ctx);

        auto render_result = engine.renderGaussians(*ctx.model, request);

        if (render_result && render_result->image && settings.apply_appearance_correction) {
            bool applied = false;

            if (const auto* tm = ctx.scene_manager ? ctx.scene_manager->getTrainerManager() : nullptr) {
                if (const auto* trainer = tm->getTrainer(); trainer && trainer->hasPPISP()) {
                    lfs::training::PPISPViewportOverrides trainer_overrides{};
                    if (settings.ppisp_mode == RenderSettings::PPISPMode::MANUAL) {
                        trainer_overrides.exposure_offset = settings.ppisp_overrides.exposure_offset;
                        trainer_overrides.vignette_enabled = settings.ppisp_overrides.vignette_enabled;
                        trainer_overrides.vignette_strength = settings.ppisp_overrides.vignette_strength;
                        trainer_overrides.wb_temperature = settings.ppisp_overrides.wb_temperature;
                        trainer_overrides.wb_tint = settings.ppisp_overrides.wb_tint;
                        trainer_overrides.gamma_multiplier = settings.ppisp_overrides.gamma_multiplier;
                    }
                    const bool use_controller = (settings.ppisp_mode == RenderSettings::PPISPMode::AUTO);
                    auto corrected = trainer->applyPPISPForViewport(
                        *render_result->image, ctx.current_camera_id, trainer_overrides, use_controller);
                    render_result->image = std::make_shared<lfs::core::Tensor>(std::move(corrected));
                    applied = true;
                }
            }

            if (!applied && ctx.scene_manager) {
                if (ctx.scene_manager->hasAppearanceModel()) {
                    const auto& overrides = (settings.ppisp_mode == RenderSettings::PPISPMode::MANUAL)
                                                ? settings.ppisp_overrides
                                                : PPISPOverrides{};
                    const bool use_controller = (settings.ppisp_mode == RenderSettings::PPISPMode::AUTO);
                    auto corrected = applyStandaloneAppearance(
                        *render_result->image, *ctx.scene_manager, ctx.current_camera_id, overrides, use_controller);
                    if (corrected.is_valid()) {
                        render_result->image = std::make_shared<lfs::core::Tensor>(std::move(corrected));
                    }
                }
            }
        }

        render_lock.reset();

        if (render_result) {
            res.cached_result = *render_result;

            if (need_hovered_output) {
                // Start async readback — result available next frame
                CHECK_CUDA(cudaMemcpyAsync(h_hovered_depth_id_, d_hovered_depth_id_,
                                           sizeof(unsigned long long), cudaMemcpyDeviceToHost));
                CHECK_CUDA(cudaEventRecord(readback_event_));
                readback_pending_ = true;

                // Use previous frame's result
                if (last_hovered_result_ == 0xFFFFFFFFFFFFFFFFULL) {
                    res.hovered_gaussian_id = -1;
                } else {
                    res.hovered_gaussian_id = static_cast<int>(last_hovered_result_ & 0xFFFFFFFF);
                }
            }

            res.cached_result_size = render_size;

            if (settings.split_view_mode == SplitViewMode::GTComparison) {
                const auto present_result = engine.presentToScreen(res.cached_result, glm::ivec2(0), render_size);
                res.render_texture_valid = present_result.has_value();
            } else {
                res.render_texture_valid = true;
            }
        } else {
            LOG_ERROR("Failed to render gaussians: {}", render_result.error());
            res.render_texture_valid = false;
            res.cached_result_size = {0, 0};
        }

        glBindFramebuffer(GL_FRAMEBUFFER, current_fbo);
        glViewport(saved_viewport[0], saved_viewport[1], saved_viewport[2], saved_viewport[3]);
        if (scissor_was_enabled)
            glEnable(GL_SCISSOR_TEST);
    }

} // namespace lfs::vis

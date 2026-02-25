/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "splat_raster_pass.hpp"
#include "../rendering_manager.hpp"
#include "core/logger.hpp"
#include "core/splat_data.hpp"
#include "geometry/euclidean_transform.hpp"
#include "scene/scene_manager.hpp"
#include "training/components/ppisp.hpp"
#include "training/components/ppisp_controller.hpp"
#include "training/trainer.hpp"
#include "training/training_manager.hpp"
#include <cuda_runtime.h>
#include <glad/glad.h>
#include <shared_mutex>

namespace lfs::vis {

    namespace {
        constexpr int GPU_ALIGNMENT = 16;

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

    bool SplatRasterPass::shouldExecute(DirtyMask frame_dirty, const FrameContext& ctx) const {
        if (!ctx.model || ctx.model->size() == 0)
            return false;
        return (frame_dirty & sensitivity()) != 0;
    }

    void SplatRasterPass::execute(lfs::rendering::RenderingEngine& /*engine*/,
                                  const FrameContext& ctx,
                                  FrameResources& res) {
        if (res.split_view_executed)
            return;

        renderToTexture(ctx.ctx, ctx.scene_manager, ctx.model);
        res.cached_result = mgr_.cached_result_;
        res.cached_result_size = mgr_.cached_result_size_;
        res.render_texture_valid = mgr_.render_texture_valid_;
    }

    void SplatRasterPass::renderToTexture(const RenderingManager::RenderContext& context,
                                          SceneManager* scene_manager,
                                          const lfs::core::SplatData* model) {
        LOG_TIMER_TRACE("SplatRasterPass::renderToTexture");
        if (!model || model->size() == 0) {
            mgr_.render_texture_valid_ = false;
            return;
        }

        const auto& settings = mgr_.settings_;

        glm::ivec2 viewport_size = context.viewport.windowSize;
        if (context.viewport_region) {
            viewport_size = glm::ivec2(
                static_cast<int>(context.viewport_region->width),
                static_cast<int>(context.viewport_region->height));
        }

        const float scale = std::clamp(settings.render_scale, 0.25f, 1.0f);
        glm::ivec2 render_size(
            static_cast<int>(viewport_size.x * scale),
            static_cast<int>(viewport_size.y * scale));

        if (settings.split_view_mode == SplitViewMode::GTComparison && mgr_.gt_context_ && mgr_.gt_context_->valid()) {
            render_size = mgr_.gt_context_->dimensions;
        }

        const glm::ivec2 alloc_size(
            ((render_size.x + GPU_ALIGNMENT - 1) / GPU_ALIGNMENT) * GPU_ALIGNMENT,
            ((render_size.y + GPU_ALIGNMENT - 1) / GPU_ALIGNMENT) * GPU_ALIGNMENT);

        static glm::ivec2 texture_size{0, 0};
        if (alloc_size != texture_size) {
            glBindTexture(GL_TEXTURE_2D, mgr_.cached_render_texture_);
            glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, alloc_size.x, alloc_size.y,
                         0, GL_RGBA, GL_UNSIGNED_BYTE, nullptr);
            LOG_DEBUG("Render texture resize: {}x{} -> {}x{}", texture_size.x, texture_size.y, alloc_size.x, alloc_size.y);
            texture_size = alloc_size;
        }

        static GLuint render_fbo = 0;
        static GLuint render_depth_rbo = 0;
        static glm::ivec2 depth_buffer_size{0, 0};

        if (render_fbo == 0) {
            glGenFramebuffers(1, &render_fbo);
            glGenRenderbuffers(1, &render_depth_rbo);
        }

        GLint current_fbo;
        glGetIntegerv(GL_FRAMEBUFFER_BINDING, &current_fbo);

        glBindFramebuffer(GL_FRAMEBUFFER, render_fbo);
        glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, mgr_.cached_render_texture_, 0);

        if (alloc_size != depth_buffer_size) {
            glBindRenderbuffer(GL_RENDERBUFFER, render_depth_rbo);
            glRenderbufferStorage(GL_RENDERBUFFER, GL_DEPTH_COMPONENT24, alloc_size.x, alloc_size.y);
            LOG_DEBUG("Depth buffer resize: {}x{}", alloc_size.x, alloc_size.y);
            depth_buffer_size = alloc_size;
        }
        glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_RENDERBUFFER, render_depth_rbo);

        const GLenum fb_status = glCheckFramebufferStatus(GL_FRAMEBUFFER);
        if (fb_status != GL_FRAMEBUFFER_COMPLETE) {
            LOG_ERROR("FBO incomplete: 0x{:x}", fb_status);
            glBindFramebuffer(GL_FRAMEBUFFER, current_fbo);
            mgr_.render_texture_valid_ = false;
            return;
        }

        glViewport(0, 0, render_size.x, render_size.y);
        glClearColor(settings.background_color.r, settings.background_color.g, settings.background_color.b, 1.0f);
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

        lfs::rendering::ViewportData viewport_data{
            .rotation = context.viewport.getRotationMatrix(),
            .translation = context.viewport.getTranslation(),
            .size = render_size,
            .focal_length_mm = settings.focal_length_mm,
            .orthographic = settings.orthographic,
            .ortho_scale = settings.ortho_scale};

        lfs::vis::SceneRenderState scene_state;
        if (scene_manager) {
            scene_state = scene_manager->buildRenderState();
        }

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
            .model_transforms = std::move(scene_state.model_transforms),
            .transform_indices = scene_state.transform_indices,
            .selection_mask = scene_state.selection_mask,
            .output_screen_positions = mgr_.output_screen_positions_,
            .brush_active = mgr_.brush_active_,
            .brush_x = mgr_.brush_x_,
            .brush_y = mgr_.brush_y_,
            .brush_radius = mgr_.brush_radius_,
            .brush_add_mode = mgr_.brush_add_mode_,
            .brush_selection_tensor = mgr_.preview_selection_ ? mgr_.preview_selection_ : mgr_.brush_selection_tensor_,
            .brush_saturation_mode = mgr_.brush_saturation_mode_,
            .brush_saturation_amount = mgr_.brush_saturation_amount_,
            .selection_mode_rings = (mgr_.selection_mode_ == lfs::rendering::SelectionMode::Rings),
            .selected_node_mask = (settings.desaturate_unselected || mgr_.getSelectionFlashIntensity() > 0.0f)
                                      ? std::move(scene_state.selected_node_mask)
                                      : std::vector<bool>{},
            .node_visibility_mask = std::move(scene_state.node_visibility_mask),
            .desaturate_unselected = settings.desaturate_unselected,
            .selection_flash_intensity = mgr_.getSelectionFlashIntensity(),
            .hovered_depth_id = nullptr,
            .highlight_gaussian_id = (mgr_.selection_mode_ == lfs::rendering::SelectionMode::Rings) ? mgr_.hovered_gaussian_id_ : -1,
            .far_plane = settings.depth_clip_enabled ? settings.depth_clip_far : lfs::rendering::DEFAULT_FAR_PLANE,
            .orthographic = settings.orthographic,
            .ortho_scale = settings.ortho_scale};

        const bool need_hovered_output = (mgr_.selection_mode_ == lfs::rendering::SelectionMode::Rings) && mgr_.brush_active_;
        if (need_hovered_output) {
            if (mgr_.d_hovered_depth_id_ == nullptr) {
                cudaMalloc(&mgr_.d_hovered_depth_id_, sizeof(unsigned long long));
            }
            constexpr unsigned long long init_val = 0xFFFFFFFFFFFFFFFFULL;
            cudaMemcpy(mgr_.d_hovered_depth_id_, &init_val, sizeof(unsigned long long), cudaMemcpyHostToDevice);
            request.hovered_depth_id = mgr_.d_hovered_depth_id_;
        }

        if (settings.use_crop_box || settings.show_crop_box) {
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
                request.crop_parent_node_index = scene_manager->getScene().getVisibleNodeIndex(cb.parent_splat_id);
            }
        }

        if (settings.use_ellipsoid || settings.show_ellipsoid) {
            const auto& scene = scene_manager->getScene();
            const auto visible_ellipsoids = scene.getVisibleEllipsoids();
            const core::NodeId selected_ellipsoid_id = scene_manager->getSelectedNodeEllipsoidId();
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

        std::optional<std::shared_lock<std::shared_mutex>> render_lock;
        if (const auto* tm = scene_manager ? scene_manager->getTrainerManager() : nullptr) {
            if (const auto* trainer = tm->getTrainer()) {
                render_lock.emplace(trainer->getRenderMutex());
            }
        }

        auto render_result = mgr_.engine_->renderGaussians(*model, request);

        if (render_result && render_result->image && settings.apply_appearance_correction) {
            bool applied = false;

            if (const auto* tm = scene_manager ? scene_manager->getTrainerManager() : nullptr) {
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
                        *render_result->image, mgr_.current_camera_id_, trainer_overrides, use_controller);
                    render_result->image = std::make_shared<lfs::core::Tensor>(std::move(corrected));
                    applied = true;
                }
            }

            if (!applied && scene_manager) {
                if (scene_manager->hasAppearanceModel()) {
                    const auto& overrides = (settings.ppisp_mode == RenderSettings::PPISPMode::MANUAL)
                                                ? settings.ppisp_overrides
                                                : PPISPOverrides{};
                    const bool use_controller = (settings.ppisp_mode == RenderSettings::PPISPMode::AUTO);
                    auto corrected = applyStandaloneAppearance(
                        *render_result->image, *scene_manager, mgr_.current_camera_id_, overrides, use_controller);
                    if (corrected.is_valid()) {
                        render_result->image = std::make_shared<lfs::core::Tensor>(std::move(corrected));
                    }
                }
            }
        }

        render_lock.reset();

        if (render_result) {
            mgr_.cached_result_ = *render_result;

            if (need_hovered_output) {
                cudaMemcpy(&mgr_.hovered_depth_id_, mgr_.d_hovered_depth_id_, sizeof(unsigned long long), cudaMemcpyDeviceToHost);
                if (mgr_.hovered_depth_id_ == 0xFFFFFFFFFFFFFFFFULL) {
                    mgr_.hovered_gaussian_id_ = -1;
                } else {
                    mgr_.hovered_gaussian_id_ = static_cast<int>(mgr_.hovered_depth_id_ & 0xFFFFFFFF);
                }
            }

            mgr_.cached_result_size_ = render_size;

            if (settings.split_view_mode == SplitViewMode::GTComparison) {
                const auto present_result = mgr_.engine_->presentToScreen(mgr_.cached_result_, glm::ivec2(0), render_size);
                mgr_.render_texture_valid_ = present_result.has_value();
            } else {
                mgr_.render_texture_valid_ = true;
            }
        } else {
            LOG_ERROR("Failed to render gaussians: {}", render_result.error());
            mgr_.render_texture_valid_ = false;
            mgr_.cached_result_size_ = {0, 0};
        }

        glBindFramebuffer(GL_FRAMEBUFFER, current_fbo);
    }

} // namespace lfs::vis

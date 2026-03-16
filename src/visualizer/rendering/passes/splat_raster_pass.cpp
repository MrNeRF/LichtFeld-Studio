/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "splat_raster_pass.hpp"
#include "../legacy_render_request_adapter.hpp"
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

        auto request = buildLegacyGaussianRenderRequest(ctx, render_size);

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
            res.cached_gpu_frame.reset();

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
            const auto gpu_frame_result = engine.materializeGpuFrame(res.cached_result, render_size);
            res.render_texture_valid = gpu_frame_result.has_value();
            if (gpu_frame_result) {
                res.cached_gpu_frame = *gpu_frame_result;
            } else {
                LOG_ERROR("Failed to materialize gaussian GPU frame: {}", gpu_frame_result.error());
            }
        } else {
            LOG_ERROR("Failed to render gaussians: {}", render_result.error());
            res.render_texture_valid = false;
            res.cached_gpu_frame.reset();
            res.cached_result_size = {0, 0};
        }
    }

} // namespace lfs::vis

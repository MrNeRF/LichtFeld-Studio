/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "split_view_service.hpp"
#include "gt_texture_cache.hpp"
#include "render_pass.hpp"
#include "scene/scene_manager.hpp"
#include "training/trainer.hpp"
#include "training/training_manager.hpp"

namespace lfs::vis {

    bool SplitViewService::hasValidGTContext() const {
        return gt_context_ && gt_context_->valid();
    }

    std::optional<glm::ivec2> SplitViewService::gtContentDimensions() const {
        if (!hasValidGTContext()) {
            return std::nullopt;
        }
        return gt_context_->dimensions;
    }

    void SplitViewService::clear() {
        clearGTContext();
        pre_gt_equirectangular_ = false;
        std::lock_guard<std::mutex> lock(info_mutex_);
        current_info_ = {};
    }

    void SplitViewService::clearGTContext() {
        gt_context_.reset();
    }

    bool SplitViewService::togglePLYComparison(RenderSettings& settings) {
        const bool enabled = settings.split_view_mode != SplitViewMode::PLYComparison;
        settings.split_view_mode = enabled ? SplitViewMode::PLYComparison : SplitViewMode::Disabled;
        settings.split_view_offset = 0;
        return enabled;
    }

    SplitViewService::GTToggleResult SplitViewService::toggleGTComparison(RenderSettings& settings) {
        GTToggleResult result;
        if (settings.split_view_mode == SplitViewMode::GTComparison) {
            settings.split_view_mode = SplitViewMode::Disabled;
            settings.equirectangular = pre_gt_equirectangular_;
            result.restore_equirectangular = pre_gt_equirectangular_;
            clearGTContext();
            return result;
        }

        pre_gt_equirectangular_ = settings.equirectangular;
        settings.split_view_mode = SplitViewMode::GTComparison;
        result.enabled = true;
        return result;
    }

    void SplitViewService::handleSceneLoaded(RenderSettings& settings) {
        clearGTContext();
        {
            std::lock_guard<std::mutex> lock(info_mutex_);
            current_info_ = {};
        }
        if (settings.split_view_mode == SplitViewMode::GTComparison) {
            settings.split_view_mode = SplitViewMode::Disabled;
        }
    }

    void SplitViewService::handleSceneCleared(RenderSettings& settings) {
        clear();
        settings.split_view_mode = SplitViewMode::Disabled;
        settings.split_view_offset = 0;
    }

    bool SplitViewService::handlePLYRemoved(RenderSettings& settings, SceneManager* scene_manager) {
        if (settings.split_view_mode != SplitViewMode::PLYComparison || !scene_manager) {
            return false;
        }

        const auto visible_nodes = scene_manager->getScene().getVisibleNodes();
        if (visible_nodes.size() >= 2) {
            return false;
        }

        settings.split_view_mode = SplitViewMode::Disabled;
        settings.split_view_offset = 0;
        return true;
    }

    void SplitViewService::advanceSplitOffset(RenderSettings& settings) {
        ++settings.split_view_offset;
    }

    SplitViewInfo SplitViewService::getInfo() const {
        std::lock_guard<std::mutex> lock(info_mutex_);
        return current_info_;
    }

    void SplitViewService::updateInfo(const FrameResources& resources) {
        std::lock_guard<std::mutex> lock(info_mutex_);
        current_info_ = resources.split_view_executed ? resources.split_info : SplitViewInfo{};
    }

    void SplitViewService::prepareGTComparisonContext(SceneManager* scene_manager,
                                                      const RenderSettings& settings,
                                                      const int current_camera_id,
                                                      const bool has_renderable_content,
                                                      const bool has_viewport_output,
                                                      GTTextureCache& texture_cache,
                                                      bool& request_viewport_prerender) {
        request_viewport_prerender = false;

        if (settings.split_view_mode != SplitViewMode::GTComparison ||
            current_camera_id < 0 ||
            !has_renderable_content ||
            !scene_manager) {
            clearGTContext();
            return;
        }

        clearGTContext();

        auto* trainer_manager = scene_manager->getTrainerManager();
        if (!trainer_manager || !trainer_manager->hasTrainer()) {
            return;
        }

        const auto* trainer = trainer_manager->getTrainer();
        if (!trainer) {
            return;
        }

        const auto loader_owner = trainer->getActiveImageLoader();
        const auto cam = trainer_manager->getCamById(current_camera_id);
        if (!cam) {
            return;
        }

        lfs::io::LoadParams gt_load_params;
        const lfs::io::LoadParams* gt_load_params_ptr = nullptr;
        if (loader_owner) {
            const auto gt_load_config = trainer->getGTLoadConfigSnapshot();
            gt_load_params.resize_factor = gt_load_config.resize_factor;
            gt_load_params.max_width = gt_load_config.max_width;
            if (gt_load_config.undistort && cam->is_undistort_prepared()) {
                gt_load_params.undistort = &cam->undistort_params();
            }
            gt_load_params_ptr = &gt_load_params;
        }

        const auto gt_info = texture_cache.getGTTexture(
            current_camera_id,
            cam->image_path(),
            loader_owner.get(),
            gt_load_params_ptr);
        if (gt_info.texture_id == 0) {
            return;
        }

        const glm::ivec2 dims(gt_info.width, gt_info.height);
        const glm::ivec2 aligned(
            ((dims.x + GPU_ALIGNMENT - 1) / GPU_ALIGNMENT) * GPU_ALIGNMENT,
            ((dims.y + GPU_ALIGNMENT - 1) / GPU_ALIGNMENT) * GPU_ALIGNMENT);

        gt_context_ = GTComparisonContext{
            .gt_texture_id = gt_info.texture_id,
            .dimensions = dims,
            .gpu_aligned_dims = aligned,
            .render_texcoord_scale = glm::vec2(dims) / glm::vec2(aligned),
            .gt_texcoord_scale = gt_info.texcoord_scale,
            .gt_needs_flip = gt_info.needs_flip};

        request_viewport_prerender = hasValidGTContext() && !has_viewport_output;
    }

} // namespace lfs::vis

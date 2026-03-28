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

    bool SplitViewService::isActive(const RenderSettings& settings) const {
        return splitViewEnabled(settings.split_view_mode);
    }

    bool SplitViewService::isGTComparisonActive(const RenderSettings& settings) const {
        return splitViewUsesGTComparison(settings.split_view_mode);
    }

    bool SplitViewService::isIndependentDualActive(const RenderSettings& settings) const {
        return splitViewUsesIndependentPanels(settings.split_view_mode);
    }

    std::optional<std::array<SplitViewPanelLayout, 2>>
    SplitViewService::panelLayouts(const RenderSettings& settings, const int total_width) const {
        if (!isIndependentDualActive(settings) || total_width <= 0) {
            return std::nullopt;
        }
        return makeSplitViewPanelLayouts(total_width, settings.split_position);
    }

    std::optional<int> SplitViewService::dividerPixel(const RenderSettings& settings, const int total_width) const {
        if (!isActive(settings) || total_width <= 0) {
            return std::nullopt;
        }
        return splitViewDividerPixel(total_width, settings.split_position);
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
        focused_panel_ = SplitViewPanelId::Left;
        std::lock_guard<std::mutex> lock(info_mutex_);
        current_info_ = {};
    }

    void SplitViewService::clearGTContext() {
        gt_context_.reset();
    }

    SplitViewService::ModeChangeResult SplitViewService::transitionToMode(RenderSettings& settings,
                                                                          const SplitViewMode target_mode,
                                                                          const Viewport* const primary_viewport,
                                                                          const GTExitBehavior gt_exit_behavior) {
        const SplitViewMode previous_mode = settings.split_view_mode;
        ModeChangeResult result{
            .previous_mode = previous_mode,
            .current_mode = previous_mode,
            .mode_changed = false,
            .clear_viewport_output = false,
            .restore_equirectangular = std::nullopt,
        };

        if (previous_mode == target_mode) {
            return result;
        }

        const bool previous_gt = splitViewUsesGTComparison(previous_mode);
        const bool target_gt = splitViewUsesGTComparison(target_mode);
        if (!previous_gt && target_gt) {
            pre_gt_equirectangular_ = settings.equirectangular;
        } else if (previous_gt && !target_gt && gt_exit_behavior == GTExitBehavior::RestorePrevious) {
            settings.equirectangular = pre_gt_equirectangular_;
            result.restore_equirectangular = pre_gt_equirectangular_;
        }

        clearGTContext();
        settings.split_view_mode = target_mode;
        result.current_mode = target_mode;
        result.mode_changed = true;
        result.clear_viewport_output = splitViewEnabled(previous_mode) && !splitViewEnabled(target_mode);

        if (splitViewUsesPLYComparison(target_mode) || splitViewUsesPLYComparison(previous_mode)) {
            settings.split_view_offset = 0;
        }

        if (target_mode == SplitViewMode::IndependentDual) {
            if (primary_viewport) {
                secondary_viewport_ = *primary_viewport;
            }
            focused_panel_ = SplitViewPanelId::Left;
        } else if (previous_mode == SplitViewMode::IndependentDual) {
            focused_panel_ = SplitViewPanelId::Left;
        }

        return result;
    }

    SplitViewService::ModeChangeResult SplitViewService::toggleMode(RenderSettings& settings,
                                                                    const SplitViewMode target_mode,
                                                                    const Viewport* const primary_viewport) {
        const SplitViewMode next_mode =
            settings.split_view_mode == target_mode ? SplitViewMode::Disabled : target_mode;
        return transitionToMode(settings, next_mode, primary_viewport, GTExitBehavior::RestorePrevious);
    }

    SplitViewService::ModeChangeResult SplitViewService::handleSceneLoaded(RenderSettings& settings) {
        auto result = transitionToMode(
            settings,
            isGTComparisonActive(settings) ? SplitViewMode::Disabled : settings.split_view_mode,
            nullptr,
            GTExitBehavior::PreserveCurrent);
        {
            std::lock_guard<std::mutex> lock(info_mutex_);
            current_info_ = {};
        }
        clearGTContext();
        return result;
    }

    SplitViewService::ModeChangeResult SplitViewService::handleSceneCleared(RenderSettings& settings) {
        auto result = transitionToMode(
            settings,
            SplitViewMode::Disabled,
            nullptr,
            GTExitBehavior::PreserveCurrent);
        clear();
        settings.split_view_offset = 0;
        result.current_mode = SplitViewMode::Disabled;
        result.clear_viewport_output = result.clear_viewport_output || splitViewEnabled(result.previous_mode);
        return result;
    }

    SplitViewService::ModeChangeResult SplitViewService::handlePLYRemoved(RenderSettings& settings,
                                                                          SceneManager* scene_manager) {
        if (!splitViewUsesPLYComparison(settings.split_view_mode) || !scene_manager) {
            return {};
        }

        const auto visible_nodes = scene_manager->getScene().getVisibleNodes();
        if (visible_nodes.size() >= 2) {
            return {};
        }

        auto result = transitionToMode(
            settings,
            SplitViewMode::Disabled,
            nullptr,
            GTExitBehavior::PreserveCurrent);
        settings.split_view_offset = 0;
        return result;
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

        if (!isGTComparisonActive(settings) ||
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

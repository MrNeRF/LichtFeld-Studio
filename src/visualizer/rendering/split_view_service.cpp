/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "split_view_service.hpp"
#include "gt_texture_cache.hpp"
#include "render_pass.hpp"
#include "scene/scene_manager.hpp"
#include "training/trainer.hpp"
#include "training/training_manager.hpp"
#include <algorithm>

namespace lfs::vis {

    namespace {
        [[nodiscard]] glm::mat4 currentSceneTransform(SceneManager* const scene_manager) {
            if (!scene_manager) {
                return glm::mat4(1.0f);
            }

            const auto visible_transforms = scene_manager->getScene().getVisibleNodeTransforms();
            if (visible_transforms.empty()) {
                return glm::mat4(1.0f);
            }
            return visible_transforms[0];
        }

        [[nodiscard]] bool equalVec2(const glm::vec2& a, const glm::vec2& b) {
            return a.x == b.x && a.y == b.y;
        }

        [[nodiscard]] bool equalMat4(const glm::mat4& a, const glm::mat4& b) {
            for (int col = 0; col < 4; ++col) {
                for (int row = 0; row < 4; ++row) {
                    if (a[col][row] != b[col][row]) {
                        return false;
                    }
                }
            }
            return true;
        }

        [[nodiscard]] std::optional<GTRenderCamera> buildGTRenderCamera(
            const lfs::core::Camera& cam,
            const glm::ivec2 render_size,
            const glm::mat4& scene_transform) {
            if (render_size.x <= 0 || render_size.y <= 0) {
                return std::nullopt;
            }

            auto R_tensor = cam.R().cpu();
            auto T_tensor = cam.T().cpu();
            const float* const R_data = R_tensor.ptr<float>();
            const float* const T_data = T_tensor.ptr<float>();
            if (!R_data || !T_data) {
                return std::nullopt;
            }

            glm::mat3 world_to_cam_R(1.0f);
            for (int row = 0; row < 3; ++row) {
                for (int col = 0; col < 3; ++col) {
                    world_to_cam_R[col][row] = R_data[row * 3 + col];
                }
            }

            const glm::vec3 world_to_cam_T(T_data[0], T_data[1], T_data[2]);
            const glm::mat3 cam_to_world_R = glm::transpose(world_to_cam_R);
            const glm::vec3 cam_to_world_T = -cam_to_world_R * world_to_cam_T;

            GTRenderCamera render_camera;
            render_camera.rotation = glm::mat3(scene_transform) * cam_to_world_R;
            render_camera.translation = glm::mat3(scene_transform) * cam_to_world_T + glm::vec3(scene_transform[3]);
            render_camera.equirectangular =
                cam.camera_model_type() == lfs::core::CameraModelType::EQUIRECTANGULAR;

            if (!render_camera.equirectangular) {
                const float x_scale =
                    static_cast<float>(render_size.x) / static_cast<float>(std::max(cam.camera_width(), 1));
                const float y_scale =
                    static_cast<float>(render_size.y) / static_cast<float>(std::max(cam.camera_height(), 1));
                render_camera.intrinsics = lfs::rendering::CameraIntrinsics{
                    .focal_x = cam.focal_x() * x_scale,
                    .focal_y = cam.focal_y() * y_scale,
                    .center_x = cam.center_x() * x_scale,
                    .center_y = cam.center_y() * y_scale};
            }

            return render_camera;
        }
    } // namespace

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

        auto* trainer_manager = scene_manager->getTrainerManager();
        if (!trainer_manager || !trainer_manager->hasTrainer()) {
            clearGTContext();
            return;
        }

        const auto* trainer = trainer_manager->getTrainer();
        if (!trainer) {
            clearGTContext();
            return;
        }

        const auto loader_owner = trainer->getActiveImageLoader();
        const auto cam = trainer_manager->getCamById(current_camera_id);
        if (!cam) {
            clearGTContext();
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
            clearGTContext();
            return;
        }

        const glm::ivec2 dims(gt_info.width, gt_info.height);
        const glm::ivec2 aligned(
            ((dims.x + GPU_ALIGNMENT - 1) / GPU_ALIGNMENT) * GPU_ALIGNMENT,
            ((dims.y + GPU_ALIGNMENT - 1) / GPU_ALIGNMENT) * GPU_ALIGNMENT);
        const glm::mat4 scene_transform = currentSceneTransform(scene_manager);

        if (gt_context_ &&
            gt_context_->camera_id == current_camera_id &&
            gt_context_->gt_texture_id == gt_info.texture_id &&
            gt_context_->dimensions == dims &&
            gt_context_->gpu_aligned_dims == aligned &&
            equalVec2(gt_context_->gt_texcoord_scale, gt_info.texcoord_scale) &&
            gt_context_->gt_needs_flip == gt_info.needs_flip &&
            equalMat4(gt_context_->scene_transform, scene_transform)) {
            request_viewport_prerender = hasValidGTContext() && !has_viewport_output;
            return;
        }

        gt_context_ = GTComparisonContext{
            .gt_texture_id = gt_info.texture_id,
            .camera_id = current_camera_id,
            .dimensions = dims,
            .gpu_aligned_dims = aligned,
            .render_texcoord_scale = glm::vec2(dims) / glm::vec2(aligned),
            .gt_texcoord_scale = gt_info.texcoord_scale,
            .gt_needs_flip = gt_info.needs_flip,
            .scene_transform = scene_transform,
            .render_camera = buildGTRenderCamera(*cam, dims, scene_transform)};

        request_viewport_prerender = hasValidGTContext() && !has_viewport_output;
    }

} // namespace lfs::vis

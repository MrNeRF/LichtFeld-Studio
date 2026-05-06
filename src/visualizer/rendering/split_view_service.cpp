/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "split_view_service.hpp"
#include "render_pass.hpp"
#include "rendering/coordinate_conventions.hpp"
#include "scene/scene_manager.hpp"
#include <algorithm>

namespace lfs::vis {

    namespace detail {
        [[nodiscard]] glm::mat4 currentSceneTransform(SceneManager* const scene_manager,
                                                      const int camera_uid) {
            if (!scene_manager) {
                return glm::mat4(1.0f);
            }

            const auto& scene = scene_manager->getScene();
            if (const auto transform = scene.getCameraSceneTransformByUid(camera_uid)) {
                return lfs::rendering::dataWorldTransformToVisualizerWorld(*transform);
            }

            const auto visible_nodes = scene.getNodes();
            const lfs::core::SceneNode* single_visible_point_cloud = nullptr;
            for (const auto* node : visible_nodes) {
                if (!node || node->type != lfs::core::NodeType::POINTCLOUD || !node->point_cloud) {
                    continue;
                }
                if (!scene.isNodeEffectivelyVisible(node->id)) {
                    continue;
                }
                if (single_visible_point_cloud) {
                    return lfs::rendering::dataWorldTransformToVisualizerWorld(glm::mat4(1.0f));
                }
                single_visible_point_cloud = node;
            }
            if (single_visible_point_cloud) {
                return lfs::rendering::dataWorldTransformToVisualizerWorld(
                    scene.getWorldTransform(single_visible_point_cloud->id));
            }

            const auto visible_transforms = scene.getVisibleNodeTransforms();
            return visible_transforms.empty()
                       ? glm::mat4(1.0f)
                       : lfs::rendering::dataWorldTransformToVisualizerWorld(visible_transforms[0]);
        }
    } // namespace detail

    namespace detail {

        std::optional<GTRenderCamera> buildGTRenderCamera(const lfs::core::Camera& cam,
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

            const auto pose = lfs::rendering::visualizerCameraPoseFromDataWorldToCamera(
                lfs::rendering::mat3FromRowMajor3x3(R_data),
                glm::vec3(T_data[0], T_data[1], T_data[2]),
                scene_transform);

            GTRenderCamera render_camera;
            render_camera.rotation = pose.rotation;
            render_camera.translation = pose.translation;
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
    } // namespace detail

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

} // namespace lfs::vis

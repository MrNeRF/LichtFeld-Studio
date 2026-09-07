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
                                                          const glm::mat4& scene_transform,
                                                          const std::optional<GTComparisonPixelRegion> pixel_region) {
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
                const glm::ivec2 calibrated_extent =
                    pixel_region ? pixel_region->full_extent : render_size;
                if (calibrated_extent.x <= 0 || calibrated_extent.y <= 0) {
                    return std::nullopt;
                }
                lfs::rendering::CameraIntrinsics full_intrinsics;
                if (pixel_region && pixel_region->full_intrinsics) {
                    full_intrinsics = *pixel_region->full_intrinsics;
                } else {
                    float base_fx = cam.focal_x();
                    float base_fy = cam.focal_y();
                    float base_cx = cam.center_x();
                    float base_cy = cam.center_y();
                    int base_width = cam.camera_width();
                    int base_height = cam.camera_height();
                    if (cam.is_undistort_precomputed()) {
                        const auto& undistort = cam.undistort_params();
                        base_fx = undistort.dst_fx;
                        base_fy = undistort.dst_fy;
                        base_cx = undistort.dst_cx;
                        base_cy = undistort.dst_cy;
                        base_width = undistort.dst_width;
                        base_height = undistort.dst_height;
                    }
                    const float x_scale =
                        static_cast<float>(calibrated_extent.x) /
                        static_cast<float>(std::max(base_width, 1));
                    const float y_scale =
                        static_cast<float>(calibrated_extent.y) /
                        static_cast<float>(std::max(base_height, 1));
                    full_intrinsics = {
                        .focal_x = base_fx * x_scale,
                        .focal_y = base_fy * y_scale,
                        .center_x = base_cx * x_scale,
                        .center_y = base_cy * y_scale};
                }
                const glm::vec2 crop_origin =
                    pixel_region ? glm::vec2(pixel_region->origin) : glm::vec2(0.0f);
                render_camera.intrinsics = lfs::rendering::CameraIntrinsics{
                    .focal_x = full_intrinsics.focal_x,
                    .focal_y = full_intrinsics.focal_y,
                    .center_x = full_intrinsics.center_x - crop_origin.x,
                    .center_y = full_intrinsics.center_y - crop_origin.y};
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

    std::optional<glm::ivec2> SplitViewService::gtContentDimensions() const {
        if (!hasValidGTContext()) {
            return std::nullopt;
        }
        return gt_context_->dimensions;
    }

    void SplitViewService::clear() {
        clearGTContext();
        pre_gt_equirectangular_ = false;
        pre_gt_show_camera_frustums_ = false;
        gt_forced_camera_frustums_off_ = false;
        focused_panel_ = SplitViewPanelId::Left;
        std::lock_guard<std::mutex> lock(info_mutex_);
        const SplitViewInfo empty_info{};
        if (current_info_ != empty_info) {
            current_info_ = empty_info;
            ++info_generation_;
        }
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
            .render_settings_changed = false,
            .restore_equirectangular = std::nullopt,
        };

        if (previous_mode == target_mode) {
            return result;
        }

        const bool previous_gt = splitViewUsesGTComparison(previous_mode);
        const bool target_gt = splitViewUsesGTComparison(target_mode);
        if (!previous_gt && target_gt) {
            pre_gt_equirectangular_ = settings.equirectangular;
            pre_gt_show_camera_frustums_ = settings.show_camera_frustums;
            gt_forced_camera_frustums_off_ = settings.show_camera_frustums;
            if (gt_forced_camera_frustums_off_) {
                settings.show_camera_frustums = false;
                result.render_settings_changed = true;
            }
        } else if (previous_gt && !target_gt) {
            if (gt_exit_behavior == GTExitBehavior::RestorePrevious) {
                settings.equirectangular = pre_gt_equirectangular_;
                result.restore_equirectangular = pre_gt_equirectangular_;
                result.render_settings_changed = true;

                if (gt_forced_camera_frustums_off_ &&
                    settings.show_camera_frustums != pre_gt_show_camera_frustums_) {
                    settings.show_camera_frustums = pre_gt_show_camera_frustums_;
                    result.render_settings_changed = true;
                }
            }
            gt_forced_camera_frustums_off_ = false;
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
            secondary_viewport_.ortho_scale_override = settings.ortho_scale;
            focused_panel_ = SplitViewPanelId::Left;
        } else if (previous_mode == SplitViewMode::IndependentDual) {
            secondary_viewport_.ortho_scale_override.reset();
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
            const SplitViewInfo empty_info{};
            if (current_info_ != empty_info) {
                current_info_ = empty_info;
                ++info_generation_;
            }
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

        const auto visible_nodes = scene_manager->getScene().getVisibleSplatNodeSlots();
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

    std::optional<SplitViewInfo> SplitViewService::getInfoIfChanged(std::uint64_t& generation) const {
        std::lock_guard<std::mutex> lock(info_mutex_);
        if (generation == info_generation_) {
            return std::nullopt;
        }
        generation = info_generation_;
        return current_info_;
    }

    void SplitViewService::updateInfo(const FrameResources& resources) {
        std::lock_guard<std::mutex> lock(info_mutex_);
        const SplitViewInfo next_info = resources.split_view_executed ? resources.split_info : SplitViewInfo{};
        if (current_info_ != next_info) {
            current_info_ = next_info;
            ++info_generation_;
        }
    }

} // namespace lfs::vis

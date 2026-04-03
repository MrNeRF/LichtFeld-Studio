/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "core/events.hpp"
#include "core/logger.hpp"
#include "core/services.hpp"
#include "rendering_manager.hpp"

namespace lfs::vis {

    using namespace lfs::core::events;

    void RenderingManager::setupEventHandlers() {
        cmd::ToggleSplitView::when([this](const auto&) { handleToggleSplitView(); });
        cmd::ToggleIndependentSplitView::when([this](const auto& event) { handleToggleIndependentSplitView(event); });
        cmd::ToggleGTComparison::when([this](const auto&) { handleToggleGTComparison(); });
        cmd::GoToCamView::when([this](const auto& event) { handleGoToCamView(event.cam_id); });
        ui::SplitPositionChanged::when([this](const auto& event) { handleSplitPositionChanged(event.position); });
        ui::RenderSettingsChanged::when([this](const auto& event) { handleRenderSettingsChanged(event); });
        ui::WindowResized::when([this](const auto&) { handleWindowResized(); });
        ui::GridSettingsChanged::when([this](const auto& event) { handleGridSettingsChanged(event); });
        ui::NodeSelected::when([this](const auto&) { triggerSelectionFlash(); });
        state::SceneLoaded::when([this](const auto&) { handleSceneLoaded(); });
        state::SceneChanged::when([this](const auto&) { handleSceneChanged(); });
        state::SceneCleared::when([this](const auto&) { handleSceneCleared(); });
        cmd::SetPLYVisibility::when([this](const auto&) { handlePLYVisibilityChanged(); });
        state::PLYAdded::when([this](const auto&) { handlePLYAdded(); });
        state::PLYRemoved::when([this](const auto&) { handlePLYRemoved(); });
        ui::CropBoxChanged::when([this](const auto& event) { handleCropBoxChanged(event.enabled); });
        ui::EllipsoidChanged::when([this](const auto& event) { handleEllipsoidChanged(event.enabled); });
        ui::PointCloudModeChanged::when([this](const auto& event) { handlePointCloudModeChanged(event); });
    }

    void RenderingManager::handleToggleSplitView() {
        SplitViewService::ModeChangeResult result;
        {
            std::lock_guard<std::mutex> lock(settings_mutex_);
            result = split_view_service_.toggleMode(settings_, SplitViewMode::PLYComparison);
            markDirty(DirtyFlag::SPLIT_VIEW);
        }
        applySplitModeChange(result);
        LOG_INFO("Split view: {}", result.current_mode == SplitViewMode::PLYComparison ? "PLY comparison mode" : "disabled");
    }

    void RenderingManager::handleToggleIndependentSplitView(const cmd::ToggleIndependentSplitView& event) {
        if (!event.viewport) {
            return;
        }

        SplitViewService::ModeChangeResult result;
        {
            std::lock_guard<std::mutex> lock(settings_mutex_);
            result = split_view_service_.toggleMode(settings_, SplitViewMode::IndependentDual, event.viewport);
            markDirty(DirtyFlag::SPLIT_VIEW | DirtyFlag::CAMERA);
        }
        applySplitModeChange(result);
        LOG_INFO("Split view: {}", result.current_mode == SplitViewMode::IndependentDual
                                       ? "independent dual-camera mode"
                                       : "disabled");
    }

    void RenderingManager::handleToggleGTComparison() {
        SplitViewService::ModeChangeResult result;

        {
            std::lock_guard<std::mutex> lock(settings_mutex_);
            result = split_view_service_.toggleMode(settings_, SplitViewMode::GTComparison);
            markDirty(DirtyFlag::SPLIT_VIEW | DirtyFlag::SPLATS);
        }

        applySplitModeChange(result);
        if (!splitViewUsesGTComparison(result.current_mode)) {
            invalidateCameraMetricsRequests(true);
        }
    }

    void RenderingManager::handleGoToCamView(const int cam_id) {
        setCurrentCameraId(cam_id);
        LOG_DEBUG("Current camera ID set to: {}", cam_id);

        if (isGTComparisonActive() && cam_id >= 0) {
            markDirty(DirtyFlag::SPLIT_VIEW);
        }
    }

    void RenderingManager::handleSplitPositionChanged(const float position) {
        std::lock_guard<std::mutex> lock(settings_mutex_);
        settings_.split_position = position;
        LOG_TRACE("Split position changed to: {}", position);
        markDirty(DirtyFlag::SPLIT_VIEW);
    }

    void RenderingManager::handleRenderSettingsChanged(const ui::RenderSettingsChanged& event) {
        std::lock_guard<std::mutex> lock(settings_mutex_);
        if (event.sh_degree) {
            settings_.sh_degree = *event.sh_degree;
            LOG_TRACE("SH_DEGREE changed to: {}", settings_.sh_degree);
        }
        if (event.focal_length_mm) {
            settings_.focal_length_mm = *event.focal_length_mm;
            LOG_TRACE("Focal length changed to: {} mm", settings_.focal_length_mm);
        }
        if (event.scaling_modifier) {
            settings_.scaling_modifier = *event.scaling_modifier;
            LOG_TRACE("Scaling modifier changed to: {}", settings_.scaling_modifier);
        }
        if (event.antialiasing) {
            settings_.antialiasing = *event.antialiasing;
            LOG_TRACE("Antialiasing: {}", settings_.antialiasing ? "enabled" : "disabled");
        }
        if (event.background_color) {
            settings_.background_color = *event.background_color;
            LOG_TRACE("Background color changed");
        }
        if (event.equirectangular) {
            settings_.equirectangular = *event.equirectangular;
            LOG_TRACE("Equirectangular rendering: {}", settings_.equirectangular ? "enabled" : "disabled");
        }
        markDirty(DirtyFlag::SPLATS | DirtyFlag::CAMERA | DirtyFlag::BACKGROUND);
    }

    void RenderingManager::handleWindowResized() {
        LOG_DEBUG("Window resized, clearing render cache");
        markDirty(DirtyFlag::VIEWPORT | DirtyFlag::CAMERA);
        viewport_artifact_service_.clearViewportOutput();
        frame_lifecycle_service_.resetViewportSize();
        gt_texture_cache_.clear();
    }

    void RenderingManager::handleGridSettingsChanged(const ui::GridSettingsChanged& event) {
        std::lock_guard<std::mutex> lock(settings_mutex_);
        settings_.show_grid = event.enabled;
        settings_.grid_plane = event.plane;
        settings_.grid_opacity = event.opacity;
        LOG_TRACE("Grid settings updated - enabled: {}, plane: {}, opacity: {}",
                  event.enabled, event.plane, event.opacity);
        markDirty(DirtyFlag::OVERLAY);
    }

    void RenderingManager::handleSceneLoaded() {
        LOG_DEBUG("Scene loaded, marking render dirty");
        markDirty();
        gt_texture_cache_.clear();
        invalidateCameraMetricsRequests(true);
        camera_interaction_service_.clearCurrentCamera();
        camera_interaction_service_.clearHoveredCamera();

        SplitViewService::ModeChangeResult result;
        {
            std::lock_guard<std::mutex> lock(settings_mutex_);
            result = split_view_service_.handleSceneLoaded(settings_);
        }
        applySplitModeChange(result);
        if (splitViewUsesGTComparison(result.previous_mode) && !splitViewUsesGTComparison(result.current_mode)) {
            LOG_INFO("Scene loaded, disabling GT comparison (camera selection reset)");
        }
    }

    void RenderingManager::handleSceneChanged() {
        pass_graph_.resetPointCloudCache();
        markDirty();
    }

    void RenderingManager::handleSceneCleared() {
        viewport_artifact_service_.clearViewportOutput();
        pass_graph_.resetPointCloudCache();
        gt_texture_cache_.clear();
        invalidateCameraMetricsRequests(true);
        SplitViewService::ModeChangeResult result;
        {
            std::lock_guard<std::mutex> lock(settings_mutex_);
            result = split_view_service_.handleSceneCleared(settings_);
        }
        if (engine_) {
            engine_->clearFrustumCache();
        }
        camera_interaction_service_.clearCurrentCamera();
        camera_interaction_service_.clearHoveredCamera();
        frame_lifecycle_service_.resetModelTracking();
        applySplitModeChange(result);
        markDirty();
    }

    void RenderingManager::handlePLYVisibilityChanged() {
        markDirty(DirtyFlag::SPLATS | DirtyFlag::MESH | DirtyFlag::OVERLAY);
    }

    void RenderingManager::handlePLYAdded() {
        LOG_DEBUG("PLY added, marking render dirty");
        markDirty(DirtyFlag::SPLATS | DirtyFlag::MESH | DirtyFlag::OVERLAY);
    }

    void RenderingManager::handlePLYRemoved() {
        SplitViewService::ModeChangeResult result;
        {
            std::lock_guard<std::mutex> lock(settings_mutex_);
            result = split_view_service_.handlePLYRemoved(settings_, services().sceneOrNull());
            markDirty(DirtyFlag::SPLATS | DirtyFlag::MESH | DirtyFlag::OVERLAY | DirtyFlag::SPLIT_VIEW);
        }
        applySplitModeChange(result);
        if (result.mode_changed) {
            LOG_DEBUG("PLY removed, disabling split view (not enough PLYs)");
        }
    }

    void RenderingManager::handleCropBoxChanged(const bool enabled) {
        std::lock_guard<std::mutex> lock(settings_mutex_);
        settings_.use_crop_box = enabled;
        markDirty(DirtyFlag::SPLATS | DirtyFlag::OVERLAY);
    }

    void RenderingManager::handleEllipsoidChanged(const bool enabled) {
        std::lock_guard<std::mutex> lock(settings_mutex_);
        settings_.use_ellipsoid = enabled;
        markDirty(DirtyFlag::SPLATS | DirtyFlag::OVERLAY);
    }

    void RenderingManager::handlePointCloudModeChanged(const ui::PointCloudModeChanged& event) {
        std::lock_guard<std::mutex> lock(settings_mutex_);
        settings_.point_cloud_mode = event.enabled;
        settings_.voxel_size = event.voxel_size;
        LOG_DEBUG("Point cloud mode: {}, voxel size: {}",
                  event.enabled ? "enabled" : "disabled", event.voxel_size);
        viewport_artifact_service_.clearViewportOutput();
        markDirty(DirtyFlag::SPLATS);
    }

} // namespace lfs::vis

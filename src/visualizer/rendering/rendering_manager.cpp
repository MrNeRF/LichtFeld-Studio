/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "rendering_manager.hpp"
#include "core/events.hpp"
#include "core/logger.hpp"
#include "rendering/ppisp_overrides_utils.hpp"
#include "rendering/rasterizer/rasterization/include/rasterization_api_tensor.h"
#include "rendering/rasterizer/rasterization/include/rasterization_config.h"
#include "rendering/rendering.hpp"
#include "rendering/rendering_pipeline.hpp"
#include "scene/scene_manager.hpp"
#include "theme/theme.hpp"
#include "training/trainer.hpp"
#include "training/training_manager.hpp"
#include <stdexcept>

namespace lfs::vis {

    namespace {
        [[nodiscard]] bool shouldRefreshCameraMetricsForSettings(
            const RenderSettings& old_settings,
            const RenderSettings& new_settings) {
            if (new_settings.camera_metrics_mode == RenderSettings::CameraMetricsMode::Off) {
                return false;
            }

            return old_settings.camera_metrics_mode != new_settings.camera_metrics_mode ||
                   old_settings.apply_appearance_correction != new_settings.apply_appearance_correction ||
                   old_settings.ppisp_mode != new_settings.ppisp_mode ||
                   !ppispOverridesEqual(old_settings.ppisp_overrides, new_settings.ppisp_overrides);
        }

        [[nodiscard]] std::expected<RenderingManager::CameraMetricsOverlayState, std::string>
        computeCameraMetricsForCurrentView(TrainerManager& trainer_mgr,
                                           const int camera_id,
                                           const int iteration,
                                           const RenderSettings& settings) {
            const bool include_ssim =
                settings.camera_metrics_mode == RenderSettings::CameraMetricsMode::PSNRSSIM;
            lfs::training::Trainer::CameraMetricsAppearanceConfig appearance{};
            appearance.enabled = settings.apply_appearance_correction;
            appearance.use_controller =
                settings.ppisp_mode == RenderSettings::PPISPMode::AUTO;
            appearance.overrides = toTrainerPPISPOverrides(settings.ppisp_overrides);

            auto metrics =
                trainer_mgr.computeCameraMetricsForCameraId(camera_id, include_ssim, appearance);
            if (!metrics) {
                return std::unexpected(metrics.error());
            }

            return RenderingManager::CameraMetricsOverlayState{
                .camera_id = camera_id,
                .iteration = iteration,
                .psnr = metrics->psnr,
                .ssim = metrics->ssim,
                .used_mask = metrics->used_mask};
        }
    } // namespace

    int RenderingManager::clampGridPlane(const int plane) {
        return std::clamp(plane, 0, 2);
    }

    void RenderingManager::syncGridPlanesLocked(const int plane) {
        panel_grid_planes_.fill(clampGridPlane(plane));
    }

    // RenderingManager Implementation
    RenderingManager::RenderingManager() {
        camera_metrics_worker_ = std::jthread([this](std::stop_token stop_token) {
            cameraMetricsWorkerLoop(stop_token);
        });
        setupEventHandlers();
    }

    RenderingManager::~RenderingManager() {
        camera_metrics_worker_.request_stop();
        camera_metrics_cv_.notify_all();
    }

    void RenderingManager::initialize() {
        if (initialized_)
            return;

        LOG_TIMER("RenderingEngine initialization");

        engine_ = lfs::rendering::RenderingEngine::create();
        auto init_result = engine_->initialize();
        if (!init_result) {
            LOG_ERROR("Failed to initialize rendering engine: {}", init_result.error());
            throw std::runtime_error("Failed to initialize rendering engine: " + init_result.error());
        }

        initialized_ = true;
        LOG_INFO("Rendering engine initialized successfully");
    }

    void RenderingManager::markDirty() {
        markDirty(DirtyFlag::ALL);
    }

    void RenderingManager::markDirty(const DirtyMask flags) {
        dirty_mask_.fetch_or(flags, std::memory_order_relaxed);

        LOG_TRACE("Render marked dirty (flags: 0x{:x})", flags);
    }

    void RenderingManager::setViewportResizeActive(bool active) {
        if (const DirtyMask dirty = frame_lifecycle_service_.setViewportResizeActive(active); dirty) {
            markDirty(dirty);
        }
    }

    void RenderingManager::updateSettings(const RenderSettings& new_settings) {
        bool clear_metrics = false;
        bool clear_frustum_thumbnails = false;
        bool frustum_visibility_changed = false;
        {
            std::lock_guard<std::mutex> lock(settings_mutex_);
            const int focused_panel_index =
                static_cast<int>(splitViewPanelIndex(split_view_service_.focusedPanel()));
            const bool grid_plane_changed = settings_.grid_plane != new_settings.grid_plane;

            // Update preview color if changed
            if (settings_.selection_color_preview != new_settings.selection_color_preview) {
                const auto& p = new_settings.selection_color_preview;
                lfs::rendering::config::setSelectionPreviewColor(make_float3(p.x, p.y, p.z));
            }

            // Update center marker color (group 0) if changed
            if (settings_.selection_color_center_marker != new_settings.selection_color_center_marker) {
                const auto& m = new_settings.selection_color_center_marker;
                lfs::rendering::config::setSelectionGroupColor(0, make_float3(m.x, m.y, m.z));
            }

            if (new_settings.camera_metrics_mode == RenderSettings::CameraMetricsMode::Off) {
                clear_metrics = true;
            } else if (camera_interaction_service_.currentCameraId() >= 0 &&
                       shouldRefreshCameraMetricsForSettings(settings_, new_settings)) {
                clear_metrics = true;
            }

            if (settings_.show_camera_frustums && !new_settings.show_camera_frustums) {
                clear_frustum_thumbnails = true;
            }
            frustum_visibility_changed = settings_.show_camera_frustums != new_settings.show_camera_frustums;

            settings_ = new_settings;
            settings_.grid_plane = clampGridPlane(settings_.grid_plane);
            if (split_view_service_.isIndependentDualActive(settings_)) {
                if (grid_plane_changed) {
                    panel_grid_planes_[focused_panel_index] = settings_.grid_plane;
                }
            } else {
                syncGridPlanesLocked(settings_.grid_plane);
            }
            markDirty();
        }

        if (clear_metrics) {
            invalidateCameraMetricsRequests(true);
        }
        if (frustum_visibility_changed) {
            invalidateFrustumImageLoaderSync();
        }
        if (clear_frustum_thumbnails) {
            clearFrustumThumbnailState();
            syncFrustumImageLoader(viewport_interaction_context_.scene_manager);
        }
    }

    RenderSettings RenderingManager::getSettings() const {
        std::lock_guard<std::mutex> lock(settings_mutex_);
        return settings_;
    }

    void RenderingManager::setOrthographic(const bool enabled, const float viewport_height, const float distance_to_pivot) {
        std::lock_guard<std::mutex> lock(settings_mutex_);

        // Calculate ortho_scale to preserve apparent size at pivot distance
        if (enabled && !settings_.orthographic) {
            constexpr float MIN_DISTANCE = 0.01f;
            constexpr float MIN_SCALE = 1.0f;
            constexpr float MAX_SCALE = 10000.0f;
            constexpr float DEFAULT_SCALE = 100.0f;

            if (viewport_height <= 0.0f || distance_to_pivot <= MIN_DISTANCE) {
                LOG_WARN("setOrthographic: invalid viewport_height={} or distance={}", viewport_height, distance_to_pivot);
                settings_.ortho_scale = DEFAULT_SCALE;
            } else {
                const float vfov = lfs::rendering::focalLengthToVFov(settings_.focal_length_mm);
                const float half_tan_fov = std::tan(glm::radians(vfov) * 0.5f);
                settings_.ortho_scale = std::clamp(
                    viewport_height / (2.0f * distance_to_pivot * half_tan_fov),
                    MIN_SCALE, MAX_SCALE);
            }
        }

        settings_.orthographic = enabled;
        markDirty(DirtyFlag::CAMERA);
    }

    float RenderingManager::getFovDegrees() const {
        std::lock_guard<std::mutex> lock(settings_mutex_);
        return lfs::rendering::focalLengthToVFov(settings_.focal_length_mm);
    }

    float RenderingManager::getFocalLengthMm() const {
        std::lock_guard<std::mutex> lock(settings_mutex_);
        return settings_.focal_length_mm;
    }

    void RenderingManager::setFocalLength(const float focal_mm) {
        std::lock_guard<std::mutex> lock(settings_mutex_);
        settings_.focal_length_mm = std::clamp(focal_mm,
                                               lfs::rendering::MIN_FOCAL_LENGTH_MM,
                                               lfs::rendering::MAX_FOCAL_LENGTH_MM);
        markDirty(DirtyFlag::CAMERA);
    }

    float RenderingManager::getScalingModifier() const {
        std::lock_guard<std::mutex> lock(settings_mutex_);
        return settings_.scaling_modifier;
    }

    void RenderingManager::setScalingModifier(const float s) {
        std::lock_guard<std::mutex> lock(settings_mutex_);
        settings_.scaling_modifier = s;
        markDirty(DirtyFlag::SPLATS);
    }

    void RenderingManager::syncSelectionGroupColor(const int group_id, const glm::vec3& color) {
        lfs::rendering::config::setSelectionGroupColor(group_id, make_float3(color.x, color.y, color.z));
        markDirty(DirtyFlag::SELECTION);
    }

    void RenderingManager::clearFrustumThumbnailState() {
        if (!engine_) {
            return;
        }

        engine_->clearFrustumCache();
    }

    void RenderingManager::invalidateFrustumImageLoaderSync(const bool poll_until_ready) {
        frustum_loader_dirty_.store(true, std::memory_order_relaxed);
        if (poll_until_ready) {
            frustum_loader_poll_until_ready_.store(true, std::memory_order_relaxed);
        }
    }

    void RenderingManager::storeFrustumImageLoaderSyncState(
        std::shared_ptr<lfs::io::PipelinedImageLoader> loader,
        const bool allow_fallback,
        const bool wait_for_active_loader) {
        {
            std::lock_guard<std::mutex> lock(frustum_loader_sync_mutex_);
            synced_frustum_loader_ = std::move(loader);
            synced_frustum_allow_fallback_ = allow_fallback;
            frustum_loader_sync_initialized_ = true;
        }
        frustum_loader_poll_until_ready_.store(wait_for_active_loader, std::memory_order_relaxed);
        frustum_loader_dirty_.store(false, std::memory_order_relaxed);
    }

    void RenderingManager::syncFrustumImageLoader(SceneManager* const scene_manager) {
        if (!engine_) {
            return;
        }

        std::shared_ptr<lfs::io::PipelinedImageLoader> frustum_loader;
        bool allow_fallback_loader = true;
        bool wait_for_active_loader = false;
        bool show_camera_frustums = false;
        {
            std::lock_guard<std::mutex> lock(settings_mutex_);
            show_camera_frustums = settings_.show_camera_frustums;
        }

        if (!show_camera_frustums) {
            allow_fallback_loader = false;
        } else if (const auto* tm = scene_manager ? scene_manager->getTrainerManager() : nullptr) {
            if (const auto* trainer = tm->getTrainer()) {
                frustum_loader = trainer->getActiveImageLoader();
            }
            if (tm->isRunning() && !frustum_loader) {
                allow_fallback_loader = false;
                wait_for_active_loader = true;
            }
        }

        {
            std::lock_guard<std::mutex> lock(frustum_loader_sync_mutex_);
            if (frustum_loader_sync_initialized_ &&
                synced_frustum_loader_ == frustum_loader &&
                synced_frustum_allow_fallback_ == allow_fallback_loader) {
                frustum_loader_poll_until_ready_.store(wait_for_active_loader, std::memory_order_relaxed);
                frustum_loader_dirty_.store(false, std::memory_order_relaxed);
                return;
            }
        }

        engine_->setFrustumImageLoader(frustum_loader, allow_fallback_loader);
        storeFrustumImageLoaderSyncState(std::move(frustum_loader),
                                         allow_fallback_loader,
                                         wait_for_active_loader);
    }

    void RenderingManager::advanceSplitOffset() {
        std::lock_guard<std::mutex> lock(settings_mutex_);
        split_view_service_.advanceSplitOffset(settings_);
        markDirty(DirtyFlag::SPLIT_VIEW | DirtyFlag::SPLATS);
    }

    SplitViewInfo RenderingManager::getSplitViewInfo() const {
        return split_view_service_.getInfo();
    }

    bool RenderingManager::isSplitViewActive() const {
        std::lock_guard<std::mutex> lock(settings_mutex_);
        return split_view_service_.isActive(settings_);
    }

    bool RenderingManager::isGTComparisonActive() const {
        std::lock_guard<std::mutex> lock(settings_mutex_);
        return split_view_service_.isGTComparisonActive(settings_);
    }

    bool RenderingManager::isIndependentSplitViewActive() const {
        std::lock_guard<std::mutex> lock(settings_mutex_);
        return split_view_service_.isIndependentDualActive(settings_);
    }

    float RenderingManager::getSplitPosition() const {
        std::lock_guard<std::mutex> lock(settings_mutex_);
        return settings_.split_position;
    }

    void RenderingManager::setFocusedSplitPanel(const SplitViewPanelId panel) {
        std::lock_guard<std::mutex> lock(settings_mutex_);
        split_view_service_.setFocusedPanel(panel);
        if (split_view_service_.isIndependentDualActive(settings_)) {
            settings_.grid_plane = panel_grid_planes_[splitViewPanelIndex(panel)];
        }
    }

    int RenderingManager::getGridPlaneForPanel(const SplitViewPanelId panel) const {
        std::lock_guard<std::mutex> lock(settings_mutex_);
        return panel_grid_planes_[splitViewPanelIndex(panel)];
    }

    void RenderingManager::setGridPlaneForPanel(const SplitViewPanelId panel, const int plane) {
        std::lock_guard<std::mutex> lock(settings_mutex_);
        const int clamped_plane = clampGridPlane(plane);
        const bool independent_split_active = split_view_service_.isIndependentDualActive(settings_);
        if (independent_split_active) {
            panel_grid_planes_[splitViewPanelIndex(panel)] = clamped_plane;
        } else {
            syncGridPlanesLocked(clamped_plane);
        }
        if (!independent_split_active || split_view_service_.focusedPanel() == panel) {
            settings_.grid_plane = clamped_plane;
        }
        markDirty(DirtyFlag::OVERLAY);
    }

    void RenderingManager::setLatestCameraMetrics(CameraMetricsOverlayState metrics) {
        std::lock_guard<std::mutex> lock(camera_metrics_mutex_);
        latest_camera_metrics_ = std::move(metrics);
        last_camera_metrics_refresh_time_ = std::chrono::steady_clock::now();
    }

    void RenderingManager::clearLatestCameraMetrics() {
        std::lock_guard<std::mutex> lock(camera_metrics_mutex_);
        latest_camera_metrics_.reset();
    }

    std::optional<RenderingManager::CameraMetricsOverlayState> RenderingManager::getLatestCameraMetrics() const {
        std::lock_guard<std::mutex> lock(camera_metrics_mutex_);
        return latest_camera_metrics_;
    }

    void RenderingManager::invalidateCameraMetricsRequests(const bool clear_latest) {
        std::lock_guard<std::mutex> lock(camera_metrics_mutex_);
        ++camera_metrics_request_generation_;
        pending_camera_metrics_request_.reset();
        last_camera_metrics_refresh_time_ = {};
        if (clear_latest) {
            latest_camera_metrics_.reset();
        }
    }

    void RenderingManager::queueCameraMetricsRefreshIfStale(SceneManager* const scene_manager) {
        if (!scene_manager) {
            return;
        }

        auto* const trainer_mgr = scene_manager->getTrainerManager();
        if (!trainer_mgr || !trainer_mgr->getTrainer()) {
            return;
        }

        const auto settings = getSettings();
        if (!splitViewUsesGTComparison(settings.split_view_mode) ||
            settings.camera_metrics_mode == RenderSettings::CameraMetricsMode::Off) {
            return;
        }

        const int current_camera_id = camera_interaction_service_.currentCameraId();
        if (current_camera_id < 0) {
            return;
        }

        const int current_iteration = trainer_mgr->getCurrentIteration();
        const bool include_ssim =
            settings.camera_metrics_mode == RenderSettings::CameraMetricsMode::PSNRSSIM;
        const auto now = std::chrono::steady_clock::now();

        bool should_queue = false;
        CameraMetricsJobRequest request{
            .trainer_manager = trainer_mgr,
            .camera_id = current_camera_id,
            .iteration = current_iteration,
            .settings = settings};

        auto request_matches = [](const CameraMetricsJobRequest& lhs,
                                  const CameraMetricsJobRequest& rhs) {
            return lhs.trainer_manager == rhs.trainer_manager &&
                   lhs.camera_id == rhs.camera_id &&
                   lhs.iteration == rhs.iteration &&
                   lhs.settings.camera_metrics_mode == rhs.settings.camera_metrics_mode &&
                   lhs.settings.apply_appearance_correction == rhs.settings.apply_appearance_correction &&
                   lhs.settings.ppisp_mode == rhs.settings.ppisp_mode &&
                   ppispOverridesEqual(lhs.settings.ppisp_overrides, rhs.settings.ppisp_overrides);
        };

        {
            std::lock_guard<std::mutex> lock(camera_metrics_mutex_);

            const bool missing_metrics = !latest_camera_metrics_.has_value();
            const bool wrong_camera = latest_camera_metrics_ &&
                                      latest_camera_metrics_->camera_id != current_camera_id;
            const bool stale_iteration = latest_camera_metrics_ &&
                                         latest_camera_metrics_->camera_id == current_camera_id &&
                                         latest_camera_metrics_->iteration != current_iteration;
            const bool missing_ssim = include_ssim && latest_camera_metrics_ &&
                                      latest_camera_metrics_->camera_id == current_camera_id &&
                                      !latest_camera_metrics_->ssim.has_value();
            const bool immediate_refresh = missing_metrics || wrong_camera || missing_ssim;
            const bool refresh_interval_elapsed =
                last_camera_metrics_refresh_time_.time_since_epoch().count() == 0 ||
                (now - last_camera_metrics_refresh_time_) >= CAMERA_METRICS_REFRESH_INTERVAL;
            const bool same_as_pending =
                pending_camera_metrics_request_ &&
                request_matches(*pending_camera_metrics_request_, request);
            const bool same_as_active =
                active_camera_metrics_request_ &&
                request_matches(*active_camera_metrics_request_, request);

            if ((immediate_refresh || stale_iteration) &&
                refresh_interval_elapsed &&
                !same_as_pending &&
                !same_as_active) {
                request.generation = ++camera_metrics_request_generation_;
                pending_camera_metrics_request_ = request;
                last_camera_metrics_refresh_time_ = now;
                should_queue = true;
            }
        }

        if (!should_queue) {
            return;
        }

        camera_metrics_cv_.notify_one();
    }

    void RenderingManager::cameraMetricsWorkerLoop(const std::stop_token stop_token) {
        while (true) {
            CameraMetricsJobRequest request;
            {
                std::unique_lock<std::mutex> lock(camera_metrics_mutex_);
                camera_metrics_cv_.wait(lock, stop_token, [this] {
                    return pending_camera_metrics_request_.has_value();
                });
                if (stop_token.stop_requested()) {
                    return;
                }

                request = *pending_camera_metrics_request_;
                active_camera_metrics_request_ = request;
                pending_camera_metrics_request_.reset();
            }

            auto metrics = computeCameraMetricsForCurrentView(
                *request.trainer_manager,
                request.camera_id,
                request.iteration,
                request.settings);

            bool applied = false;
            {
                std::lock_guard<std::mutex> lock(camera_metrics_mutex_);
                if (active_camera_metrics_request_ &&
                    active_camera_metrics_request_->generation == request.generation) {
                    active_camera_metrics_request_.reset();
                }

                if (request.generation == camera_metrics_request_generation_) {
                    if (metrics) {
                        latest_camera_metrics_ = *metrics;
                    } else {
                        latest_camera_metrics_.reset();
                    }
                    last_camera_metrics_refresh_time_ = std::chrono::steady_clock::now();
                    applied = true;
                }
            }

            if (applied) {
                markDirty(DirtyFlag::OVERLAY);
            }
        }
    }

    std::optional<float> RenderingManager::getSplitDividerScreenX(const glm::vec2& viewport_pos,
                                                                  const glm::vec2& viewport_size) const {
        std::lock_guard<std::mutex> lock(settings_mutex_);
        if (!split_view_service_.isActive(settings_)) {
            return std::nullopt;
        }

        const auto content_bounds = getContentBounds(glm::ivec2(
            std::max(static_cast<int>(viewport_size.x), 0),
            std::max(static_cast<int>(viewport_size.y), 0)));
        return viewport_pos.x + content_bounds.x + content_bounds.width * settings_.split_position;
    }

    Viewport& RenderingManager::resolvePanelViewport(Viewport& primary_viewport, const SplitViewPanelId panel) {
        std::lock_guard<std::mutex> lock(settings_mutex_);
        if (split_view_service_.isIndependentDualActive(settings_) &&
            panel == SplitViewPanelId::Right) {
            return split_view_service_.secondaryViewport();
        }
        return primary_viewport;
    }

    const Viewport& RenderingManager::resolvePanelViewport(
        const Viewport& primary_viewport,
        const SplitViewPanelId panel) const {
        std::lock_guard<std::mutex> lock(settings_mutex_);
        if (split_view_service_.isIndependentDualActive(settings_) &&
            panel == SplitViewPanelId::Right) {
            return split_view_service_.secondaryViewport();
        }
        return primary_viewport;
    }

    void RenderingManager::applySplitModeChange(const SplitViewService::ModeChangeResult& result) {
        if (!result.mode_changed) {
            return;
        }

        if (result.clear_viewport_output) {
            viewport_artifact_service_.clearViewportOutput();
        }

        if (result.restore_equirectangular) {
            auto event = lfs::core::events::ui::RenderSettingsChanged{};
            event.equirectangular = *result.restore_equirectangular;
            event.emit();
        }
    }

    Viewport& RenderingManager::resolveFocusedViewport(Viewport& primary_viewport) {
        return resolvePanelViewport(primary_viewport, split_view_service_.focusedPanel());
    }

    const Viewport& RenderingManager::resolveFocusedViewport(const Viewport& primary_viewport) const {
        return resolvePanelViewport(primary_viewport, split_view_service_.focusedPanel());
    }

    void RenderingManager::setCursorPreviewState(const bool active, const float x, const float y, const float radius,
                                                 const bool add_mode, lfs::core::Tensor* selection_tensor,
                                                 const bool saturation_mode, const float saturation_amount,
                                                 const std::optional<SplitViewPanelId> panel,
                                                 const int focused_gaussian_id) {
        viewport_overlay_service_.setCursorPreview(active, x, y, radius, add_mode, selection_tensor,
                                                   saturation_mode, saturation_amount, panel, focused_gaussian_id);
        markDirty(DirtyFlag::SELECTION);
    }

    void RenderingManager::clearCursorPreviewState() {
        viewport_overlay_service_.clearCursorPreview();
        markDirty(DirtyFlag::SELECTION);
    }

    void RenderingManager::setRectPreview(float x0, float y0, float x1, float y1, bool add_mode,
                                          const std::optional<SplitViewPanelId> panel) {
        viewport_overlay_service_.setRect(x0, y0, x1, y1, add_mode, panel);
    }

    void RenderingManager::clearRectPreview() {
        viewport_overlay_service_.clearRect();
    }

    void RenderingManager::setPolygonPreview(const std::vector<std::pair<float, float>>& points, bool closed,
                                             bool add_mode, const std::optional<SplitViewPanelId> panel) {
        viewport_overlay_service_.setPolygon(points, closed, add_mode, panel);
    }

    void RenderingManager::setPolygonPreviewWorldSpace(const std::vector<glm::vec3>& world_points,
                                                       const bool closed, const bool add_mode,
                                                       const std::optional<SplitViewPanelId> panel) {
        viewport_overlay_service_.setPolygonWorldSpace(world_points, closed, add_mode, panel);
    }

    void RenderingManager::clearPolygonPreview() {
        viewport_overlay_service_.clearPolygon();
    }

    void RenderingManager::setLassoPreview(const std::vector<std::pair<float, float>>& points, bool add_mode,
                                           const std::optional<SplitViewPanelId> panel) {
        viewport_overlay_service_.setLasso(points, add_mode, panel);
    }

    void RenderingManager::clearLassoPreview() {
        viewport_overlay_service_.clearLasso();
    }

    void RenderingManager::clearSelectionPreviews() {
        viewport_overlay_service_.clearSelectionPreviews();
        markDirty(DirtyFlag::SELECTION);
    }

} // namespace lfs::vis

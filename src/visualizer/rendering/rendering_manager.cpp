/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "rendering_manager.hpp"
#include "core/events.hpp"
#include "core/logger.hpp"
#include "point_cloud_vulkan_renderer.hpp"
#include "rendering/ppisp_overrides_utils.hpp"
#include "rendering/rendering.hpp"
#include "rendering/selection_ops.hpp"
#include "scene/scene_manager.hpp"
#include "theme/theme.hpp"
#include "training/trainer.hpp"
#include "training/training_manager.hpp"
#include "visualizer/app_store.hpp"
#include "vksplat_viewport_renderer.hpp"

#include <algorithm>
#include <cmath>
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

        [[nodiscard]] bool applySparkLodViewerDefaults(RenderSettings& settings) {
            bool changed = false;

            if (settings.lod_max_splats == 1'500'000) {
                settings.lod_max_splats = DEFAULT_LOD_MAX_SPLATS;
                changed = true;
            }

            if (settings.lod_behind_camera_penalty == 2.0f) {
                settings.lod_behind_camera_penalty = DEFAULT_LOD_BEHIND_CAMERA_FOVEATION;
                changed = true;
            }

            if (settings.lod_cone_inner_degrees == 0.0f &&
                settings.lod_cone_outer_degrees == 0.0f) {
                settings.lod_cone_foveation = DEFAULT_LOD_CONE_FOVEATION;
                settings.lod_cone_inner_degrees = DEFAULT_LOD_CONE_INNER_DEGREES;
                settings.lod_cone_outer_degrees = DEFAULT_LOD_CONE_OUTER_DEGREES;
                changed = true;
            }

            return changed;
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

        [[nodiscard]] AppStore::CameraMetrics toAppCameraMetrics(
            const RenderingManager::CameraMetricsOverlayState& metrics) {
            return AppStore::CameraMetrics{
                .camera_id = metrics.camera_id,
                .iteration = metrics.iteration,
                .psnr = metrics.psnr,
                .ssim = metrics.ssim,
                .used_mask = metrics.used_mask};
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
        if (lod_controller_) {
            lod_controller_->setReadyCallback(nullptr);
        }
        camera_metrics_worker_.request_stop();
        camera_metrics_cv_.notify_all();
    }

    void RenderingManager::setWakeCallback(std::function<void()> callback) {
        std::scoped_lock lock(wake_callback_mutex_);
        wake_callback_ = std::move(callback);
    }

    void RenderingManager::initialize() {
        // Gate on engine_ rather than initialized_: the Vulkan path flips
        // initialized_ on first frame without building the auxiliary engine,
        // and getRenderingEngine() relies on this to lazy-create it on demand.
        if (engine_)
            return;

        LOG_TIMER("RenderingEngine initialization");

        engine_ = lfs::rendering::RenderingEngine::create();
        auto init_result = engine_->initialize();
        if (!init_result) {
            LOG_ERROR("Failed to initialize rendering engine: {}", init_result.error());
            throw std::runtime_error("Failed to initialize rendering engine: " + init_result.error());
        }

        initialized_ = true;
        LOG_INFO("Auxiliary rendering engine initialized successfully");
    }

    void RenderingManager::markDirty() {
        markDirty(DirtyFlag::ALL);
    }

    void RenderingManager::markDirty(const DirtyMask flags) {
        dirty_mask_.fetch_or(flags, std::memory_order_relaxed);

        LOG_TRACE("Render marked dirty (flags: 0x{:x})", flags);
    }

    void RenderingManager::markCameraPoseChanged() {
        camera_pose_dirty_.store(true, std::memory_order_release);
        markDirty(DirtyFlag::CAMERA);
    }

    bool RenderingManager::pollDirtyState() {
        if (const DirtyMask animation_dirty = animation_state_.pollDirtyState(); animation_dirty) {
            dirty_mask_.fetch_or(animation_dirty, std::memory_order_relaxed);
            return true;
        }
        if (lod_controller_ && lod_controller_->hasReadyResults()) {
            dirty_mask_.fetch_or(DirtyFlag::CAMERA, std::memory_order_relaxed);
            return true;
        }
        return dirty_mask_.load(std::memory_order_relaxed) != 0;
    }

    void RenderingManager::requestRenderFollowUp() {
        dirty_mask_.fetch_or(DirtyFlag::CAMERA, std::memory_order_relaxed);

        std::function<void()> wake_callback;
        {
            std::scoped_lock lock(wake_callback_mutex_);
            wake_callback = wake_callback_;
        }
        if (wake_callback) {
            wake_callback();
        }
    }

    void RenderingManager::notifyAsyncLodResultsReady() {
        requestRenderFollowUp();
    }

    void RenderingManager::setViewportResizeActive(bool active) {
        if (const DirtyMask dirty = frame_lifecycle_service_.setViewportResizeActive(active); dirty) {
            markDirty(dirty);
            std::function<void()> wake_callback;
            {
                std::scoped_lock lock(wake_callback_mutex_);
                wake_callback = wake_callback_;
            }
            if (wake_callback) {
                wake_callback();
            }
        }
    }

    void RenderingManager::requestResizeTrainingPause(TrainerManager* const trainer_manager) {
        if (resize_training_pause_active_ || !trainer_manager || !trainer_manager->isRunning()) {
            return;
        }

        trainer_manager->pauseTrainingTemporary();
        resize_training_pause_trainer_ = trainer_manager;
        resize_training_pause_active_ = true;
    }

    void RenderingManager::releaseResizeTrainingPause() {
        if (!resize_training_pause_active_) {
            return;
        }

        if (resize_training_pause_trainer_) {
            resize_training_pause_trainer_->resumeTrainingTemporary();
        }
        resize_training_pause_trainer_ = nullptr;
        resize_training_pause_active_ = false;
    }

    void RenderingManager::setLodAvailable(bool available) {
        lod_available_ = available;
        if (available) {
            auto settings = getSettings();
            if (applySparkLodViewerDefaults(settings)) {
                updateSettings(settings, DirtyFlag::ALL);
            }
        }
    }

    void RenderingManager::setLodEnabled(bool enabled) {
        auto settings = getSettings();
        settings.lod_enabled = enabled;
        const bool changed = enabled && applySparkLodViewerDefaults(settings);
        updateSettings(settings, changed ? DirtyFlag::ALL : DirtyFlag::SPLATS);
    }

    bool RenderingManager::isLodEnabled() const {
        std::lock_guard<std::mutex> lock(settings_mutex_);
        return settings_.lod_enabled;
    }

    SparkLodController::Stats RenderingManager::getLodStats() const {
        SparkLodController::Stats stats;
        if (lod_controller_) {
            stats = lod_controller_->stats();
        }

        bool gpu_selection_eligible = false;
        float render_scale_setting = 1.0f;
        {
            std::lock_guard<std::mutex> lock(settings_mutex_);
            stats.enabled = settings_.lod_enabled;
            stats.requested_max_splats = settings_.lod_max_splats;
            if (stats.max_splats == 0) {
                stats.max_splats = settings_.lod_max_splats;
            }
            if (stats.lod_render_scale == 0.0f) {
                stats.lod_render_scale = settings_.lod_render_scale;
            }
            stats.behind_camera_penalty = settings_.lod_behind_camera_penalty;
            stats.cone_foveation = settings_.lod_cone_foveation;
            stats.cone_inner_degrees = settings_.lod_cone_inner_degrees;
            stats.cone_outer_degrees = settings_.lod_cone_outer_degrees;
            gpu_selection_eligible = settings_.lod_enabled;
            render_scale_setting = settings_.lod_render_scale;
        }

        if (gpu_selection_eligible && vksplat_viewport_renderer_) {
            const auto gpu = vksplat_viewport_renderer_->gpuLodSelectionStatus();
            if (gpu.active) {
                // The CPU controller is frozen at its bootstrap cut in GPU
                // mode; report the selector's live numbers instead.
                stats.gpu_selection = true;
                stats.selected_splats = gpu.selected;
                stats.output_size = gpu.selected;
                // Effective target = LOD Budget x Render Scale (Spark-style
                // quality scaler); the overlay shows both when they differ.
                stats.max_splats = std::max<size_t>(
                    1,
                    static_cast<size_t>(
                        std::llround(static_cast<double>(stats.requested_max_splats) *
                                     std::max(render_scale_setting, 0.1f))));
                stats.budget_repair_active = false;
                stats.budget_fill_active = false;
                stats.budget_limited = gpu.overflow > 0;
                stats.threshold_limited = gpu.overflow == 0;
                stats.output_limited = false;
                if (stats.pixel_scale_limit > 0.0f) {
                    stats.pixel_scale_limit *= gpu.pixel_scale_feedback;
                }
                if (gpu.chunk_count > 0) {
                    stats.chunk_count = gpu.chunk_count;
                    stats.resident_chunks = gpu.resident_chunks;
                    stats.touched_chunks = gpu.touched_chunks;
                }
                stats.gpu_output_capacity = gpu.capacity;
                stats.gpu_overflow = gpu.overflow;
                stats.gpu_pixel_scale_feedback = gpu.pixel_scale_feedback;
                stats.pool_pages = gpu.pool_pages;
                stats.streaming_jobs = gpu.streaming_jobs;
                stats.miss_chunks = gpu.miss_chunks;
                stats.deferred_requests = gpu.deferred_requests;
                stats.admission_frozen = gpu.admission_frozen;
            }
        }

        stats.available = lod_available_ || stats.has_tree;
        stats.active = stats.has_tree && lod_controller_ != nullptr &&
                       (stats.enabled || stats.full_quality_reference);
        return stats;
    }

    void RenderingManager::releaseSceneModelResources() {
        clearVulkanMeshFrame();

        point_cloud_colors_cache_ = {};
        point_cloud_colors_cache_key_ = nullptr;
        point_cloud_colors_cache_size_ = 0;
        ++point_cloud_data_revision_;
        ++point_cloud_preview_selection_revision_;

        if (vksplat_viewport_renderer_) {
            vksplat_viewport_renderer_->releaseSceneResources();
        }
        if (point_cloud_vulkan_renderer_) {
            point_cloud_vulkan_renderer_->reset();
        }
        frame_lifecycle_service_.resetModelTracking();
    }

    void RenderingManager::releaseSceneRenderResources() {
        viewport_artifact_service_.clearViewportOutput();
        vulkan_viewport_image_.reset();
        vulkan_viewport_image_generation_ = 0;
        vulkan_external_viewport_image_ = VK_NULL_HANDLE;
        vulkan_external_viewport_image_view_ = VK_NULL_HANDLE;
        vulkan_external_viewport_image_layout_ = VK_IMAGE_LAYOUT_UNDEFINED;
        vulkan_external_viewport_image_generation_ = 0;
        split_view_image_generation_ = 0;
        vulkan_viewport_image_size_ = {0, 0};
        vulkan_viewport_image_flip_y_ = false;

        clearVulkanMeshFrame();

        point_cloud_colors_cache_ = {};
        point_cloud_colors_cache_key_ = nullptr;
        point_cloud_colors_cache_size_ = 0;
        ++point_cloud_data_revision_;
        ++point_cloud_preview_selection_revision_;

        if (vksplat_viewport_renderer_) {
            vksplat_viewport_renderer_->reset();
        }
        if (point_cloud_vulkan_renderer_) {
            point_cloud_vulkan_renderer_->reset();
        }
        frame_lifecycle_service_.resetModelTracking();
        lfs::core::Tensor::trim_memory_pool();
    }

    void RenderingManager::updateSettings(const RenderSettings& new_settings) {
        updateSettings(new_settings, DirtyFlag::ALL);
    }

    void RenderingManager::updateSettings(const RenderSettings& new_settings,
                                          const DirtyMask dirty_flags) {
        RenderSettings sanitized_settings = new_settings;
        bool clear_metrics = false;
        bool lod_request_changed = false;
        bool lod_enabled_turned_on = false;
        {
            std::lock_guard<std::mutex> lock(settings_mutex_);
            if (split_view_service_.isGTComparisonActive(settings_) ||
                split_view_service_.isGTComparisonActive(sanitized_settings)) {
                sanitized_settings.show_camera_frustums = false;
            }
            const int focused_panel_index =
                static_cast<int>(splitViewPanelIndex(split_view_service_.focusedPanel()));
            const bool grid_plane_changed = settings_.grid_plane != sanitized_settings.grid_plane;
            lod_enabled_turned_on = !settings_.lod_enabled && sanitized_settings.lod_enabled;
            lod_request_changed =
                settings_.lod_enabled != sanitized_settings.lod_enabled ||
                settings_.lod_max_splats != sanitized_settings.lod_max_splats ||
                settings_.lod_render_scale != sanitized_settings.lod_render_scale ||
                settings_.lod_behind_camera_penalty != sanitized_settings.lod_behind_camera_penalty ||
                settings_.lod_cone_foveation != sanitized_settings.lod_cone_foveation ||
                settings_.lod_cone_inner_degrees != sanitized_settings.lod_cone_inner_degrees ||
                settings_.lod_cone_outer_degrees != sanitized_settings.lod_cone_outer_degrees;

            // Update preview color if changed
            if (settings_.selection_color_preview != sanitized_settings.selection_color_preview) {
                const auto& p = sanitized_settings.selection_color_preview;
                lfs::rendering::config::setSelectionPreviewColor(make_float3(p.x, p.y, p.z));
            }

            // Update center marker color (group 0) if changed
            if (settings_.selection_color_center_marker != sanitized_settings.selection_color_center_marker) {
                const auto& m = sanitized_settings.selection_color_center_marker;
                lfs::rendering::config::setSelectionGroupColor(0, make_float3(m.x, m.y, m.z));
            }

            if (sanitized_settings.camera_metrics_mode == RenderSettings::CameraMetricsMode::Off) {
                clear_metrics = true;
            } else if (camera_interaction_service_.currentCameraId() >= 0 &&
                       shouldRefreshCameraMetricsForSettings(settings_, sanitized_settings)) {
                clear_metrics = true;
            }

            const auto previous_backend = settings_.raster_backend;
            const bool previous_gut = settings_.gut;
            settings_ = sanitized_settings;
            const bool gut_toggle_only =
                settings_.raster_backend == previous_backend && settings_.gut != previous_gut;
            settings_.raster_backend = gut_toggle_only
                                           ? lfs::rendering::viewerRasterBackendForGutMode(settings_.gut)
                                           : lfs::rendering::normalizeViewerRasterBackend(
                                                 settings_.raster_backend, settings_.gut);
            settings_.gut = lfs::rendering::isGutBackend(settings_.raster_backend);
            enforceProjectionBackend(settings_);
            sanitizeDepthViewSettings(settings_);
            sanitizeGTComparisonSettings(settings_);
            settings_.grid_plane = clampGridPlane(settings_.grid_plane);
            if (split_view_service_.isIndependentDualActive(settings_)) {
                if (grid_plane_changed) {
                    panel_grid_planes_[focused_panel_index] = settings_.grid_plane;
                }
            } else {
                syncGridPlanesLocked(settings_.grid_plane);
            }
            markDirty(dirty_flags);
        }

        if (lod_request_changed && lod_controller_) {
            lod_controller_->invalidatePendingWork();
        }
        if (lod_enabled_turned_on) {
            lod_controller_needs_sync_traversal_ = true;
        }

        auto& render_settings_generation = app_store().render_settings_generation;
        render_settings_generation.set(render_settings_generation.get() + 1);

        if (clear_metrics) {
            invalidateCameraMetricsRequests(true);
        }
    }

    RenderSettings RenderingManager::getSettings() const {
        std::lock_guard<std::mutex> lock(settings_mutex_);
        return settings_;
    }

    void RenderingManager::setOrthographic(const bool enabled, const float viewport_height, const float distance_to_pivot) {
        std::lock_guard<std::mutex> lock(settings_mutex_);

        constexpr float MIN_DISTANCE = 0.01f;
        constexpr float MIN_SCALE = 1.0f;
        constexpr float MAX_SCALE = 10000.0f;
        constexpr float DEFAULT_SCALE = 100.0f;

        if (viewport_height <= 0.0f || distance_to_pivot <= MIN_DISTANCE) {
            LOG_WARN("setOrthographic: invalid viewport_height={} or distance={}", viewport_height, distance_to_pivot);
            if (enabled && !settings_.orthographic) {
                settings_.ortho_scale = DEFAULT_SCALE;
            }
            settings_.orthographic = enabled;
            markDirty(DirtyFlag::CAMERA);
            return;
        }

        if (enabled && !settings_.orthographic) {
            const float vfov = lfs::rendering::focalLengthToVFov(settings_.focal_length_mm);
            const float half_tan_fov = std::tan(glm::radians(vfov) * 0.5f);
            settings_.ortho_scale = std::clamp(
                viewport_height / (2.0f * distance_to_pivot * half_tan_fov),
                MIN_SCALE, MAX_SCALE);
        } else if (!enabled && settings_.orthographic) {
            const float half_tan_fov = viewport_height / (2.0f * distance_to_pivot * settings_.ortho_scale);
            const float vfov = glm::degrees(2.0f * std::atan(half_tan_fov));
            settings_.focal_length_mm = std::clamp(
                lfs::rendering::vFovToFocalLength(vfov),
                lfs::rendering::MIN_FOCAL_LENGTH_MM,
                lfs::rendering::MAX_FOCAL_LENGTH_MM);
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
        const auto app_metrics = toAppCameraMetrics(metrics);
        {
            std::lock_guard<std::mutex> lock(camera_metrics_mutex_);
            latest_camera_metrics_ = std::move(metrics);
            last_camera_metrics_refresh_time_ = std::chrono::steady_clock::now();
        }
        app_store().camera_metrics.set(app_metrics);
    }

    void RenderingManager::clearLatestCameraMetrics() {
        {
            std::lock_guard<std::mutex> lock(camera_metrics_mutex_);
            latest_camera_metrics_.reset();
        }
        app_store().camera_metrics.set(std::optional<AppStore::CameraMetrics>{});
    }

    std::optional<RenderingManager::CameraMetricsOverlayState> RenderingManager::getLatestCameraMetrics() const {
        std::lock_guard<std::mutex> lock(camera_metrics_mutex_);
        return latest_camera_metrics_;
    }

    void RenderingManager::invalidateCameraMetricsRequests(const bool clear_latest) {
        {
            std::lock_guard<std::mutex> lock(camera_metrics_mutex_);
            ++camera_metrics_request_generation_;
            pending_camera_metrics_request_.reset();
            last_camera_metrics_refresh_time_ = {};
            if (clear_latest) {
                latest_camera_metrics_.reset();
            }
        }
        if (clear_latest)
            app_store().camera_metrics.set(std::optional<AppStore::CameraMetrics>{});
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
            std::optional<AppStore::CameraMetrics> app_metrics;
            {
                std::lock_guard<std::mutex> lock(camera_metrics_mutex_);
                if (active_camera_metrics_request_ &&
                    active_camera_metrics_request_->generation == request.generation) {
                    active_camera_metrics_request_.reset();
                }

                if (request.generation == camera_metrics_request_generation_) {
                    if (metrics) {
                        latest_camera_metrics_ = *metrics;
                        app_metrics = toAppCameraMetrics(*metrics);
                    } else {
                        latest_camera_metrics_.reset();
                    }
                    last_camera_metrics_refresh_time_ = std::chrono::steady_clock::now();
                    applied = true;
                }
            }

            if (applied) {
                app_store().camera_metrics.set(std::move(app_metrics));
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
        const int content_width = std::max(static_cast<int>(std::lround(content_bounds.width)), 0);
        if (content_width <= 0) {
            return std::nullopt;
        }

        return viewport_pos.x + content_bounds.x +
               static_cast<float>(splitViewDividerPixel(content_width, settings_.split_position));
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
        if (result.render_settings_changed) {
            markDirty(DirtyFlag::OVERLAY);
            auto& render_settings_generation = app_store().render_settings_generation;
            render_settings_generation.set(render_settings_generation.get() + 1);
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
                                          const std::optional<SplitViewPanelId> panel,
                                          const bool track_cursor) {
        viewport_overlay_service_.setRect(x0, y0, x1, y1, add_mode, panel, track_cursor);
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
                                           const std::optional<SplitViewPanelId> panel,
                                           const bool track_cursor) {
        viewport_overlay_service_.setLasso(points, add_mode, panel, track_cursor);
    }

    void RenderingManager::clearLassoPreview() {
        viewport_overlay_service_.clearLasso();
    }

    void RenderingManager::clearSelectionPreviews() {
        viewport_overlay_service_.clearSelectionPreviews();
        markDirty(DirtyFlag::SELECTION);
    }

} // namespace lfs::vis

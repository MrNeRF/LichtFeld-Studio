/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "rendering_manager.hpp"
#include "core/cuda/memory_arena.hpp"
#include "core/events.hpp"
#include "core/logger.hpp"
#include "operation/undo_entry.hpp"
#include "operation/undo_history.hpp"
#include "point_cloud_vulkan_renderer.hpp"
#include "preferences.hpp"
#include "rendering/export_post_process.hpp"
#include "rendering/ppisp_overrides_utils.hpp"
#include "rendering/rendering.hpp"
#include "rendering/scene_upscaler_registry.hpp"
#include "rendering/selection_ops.hpp"
#include "scene/scene_manager.hpp"
#include "theme/theme.hpp"
#include "training/trainer.hpp"
#include "training/training_manager.hpp"
#include "visualizer/app_store.hpp"
#include "vksplat_viewport_renderer.hpp"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <mutex>
#include <stdexcept>
#include <string>
#include <unordered_set>

namespace lfs::vis {

    namespace {
        void warnUnavailableSceneUpscalerOnce(const std::string& attempted_id) {
            static std::mutex mutex;
            static std::unordered_set<std::string> warned_ids;
            std::lock_guard lock(mutex);
            if (!warned_ids.insert(attempted_id).second)
                return;
            LOG_WARN("Scene reconstruction backend '{}' is not available in this process; "
                     "keeping the previous backend",
                     attempted_id);
        }

        // Sanitizes the screen-window fields, plus a one-shot migration of the
        // legacy positive-Z depth box. General near/far normalization still does
        // not belong here: the tool and GUI bindings rewrite depth_filter_min/max
        // every frame, and a normalizer that keeps transforming those values
        // shifts that loop's equilibrium. The migration below is exempt because
        // it is an exact sentinel match that cannot fire twice - once (0, 100)
        // becomes (-100, 0) it no longer matches, and a band the screen-window
        // path writes always satisfies min.z = -far <= max.z = -near <= 0.

        constexpr float kDepthNearMin = 0.01f;
        constexpr float kDepthNearMax = 999.99f;
        constexpr float kDepthMax = 1000.0f;

        [[nodiscard]] DepthWindowState depthWindowFromProjection(const RenderSettings& settings) {
            return {
                .near_plane = -settings.depth_filter_max.z,
                .far_plane = -settings.depth_filter_min.z,
                .scale_x = settings.depth_filter_scale_x,
                .scale_y = settings.depth_filter_scale_y,
                .offset_x = settings.depth_filter_offset_x,
                .offset_y = settings.depth_filter_offset_y,
            };
        }

        void clampDepthWindowState(DepthWindowState& state) {
            if (!std::isfinite(state.scale_x)) {
                state.scale_x = 0.35f;
            }
            if (!std::isfinite(state.scale_y)) {
                state.scale_y = 0.35f;
            }
            if (!std::isfinite(state.offset_x)) {
                state.offset_x = 0.0f;
            }
            if (!std::isfinite(state.offset_y)) {
                state.offset_y = 0.0f;
            }
            state.scale_x = std::clamp(state.scale_x, 0.05f, 1.0f);
            state.scale_y = std::clamp(state.scale_y, 0.05f, 1.0f);
            state.offset_x = std::clamp(state.offset_x, -1.0f, 1.0f);
            state.offset_y = std::clamp(state.offset_y, -1.0f, 1.0f);
            state.near_plane = std::clamp(state.near_plane, 0.0f, kDepthNearMax);
            state.far_plane = std::clamp(state.far_plane, state.near_plane + kDepthNearMin, kDepthMax);
        }

        void applyDepthWindowToProjection(RenderSettings& settings, const DepthWindowState& state) {
            settings.depth_filter_scale_x = state.scale_x;
            settings.depth_filter_scale_y = state.scale_y;
            settings.depth_filter_offset_x = state.offset_x;
            settings.depth_filter_offset_y = state.offset_y;
            settings.depth_filter_min.z = -state.far_plane;
            settings.depth_filter_max.z = -state.near_plane;
        }

        void sanitizeSelectionWindowSettings(RenderSettings& settings) {
            constexpr float kDefaultScale = 0.35f;
            constexpr float kDefaultOffset = 0.0f;

            // The constructor defaults predate the screen-window contract and
            // still carry the old positive-Z box. Decoded as a negated near/far
            // band they yield [-100, 0], which no positive depth satisfies, so
            // enabling the filter without a writer selects nothing. Only Z is
            // tested: legacy callers could have moved X/Y while leaving the
            // default near/far in place. Deliberately independent of the
            // previous enabled state - session restore applies an enabled box
            // on top of an already-enabled filter, so a false->true edge would
            // miss it.
            if (settings.depth_filter_enabled &&
                settings.depth_filter_min.z == 0.0f &&
                settings.depth_filter_max.z == 100.0f) {
                settings.depth_filter_min.z = -100.0f;
                settings.depth_filter_max.z = 0.0f;
            }

            if (!std::isfinite(settings.depth_filter_scale_x)) {
                settings.depth_filter_scale_x = kDefaultScale;
            }
            if (!std::isfinite(settings.depth_filter_scale_y)) {
                settings.depth_filter_scale_y = kDefaultScale;
            }
            if (!std::isfinite(settings.depth_filter_offset_x)) {
                settings.depth_filter_offset_x = kDefaultOffset;
            }
            if (!std::isfinite(settings.depth_filter_offset_y)) {
                settings.depth_filter_offset_y = kDefaultOffset;
            }
            settings.depth_filter_scale_x = std::clamp(settings.depth_filter_scale_x, 0.05f, 1.0f);
            settings.depth_filter_scale_y = std::clamp(settings.depth_filter_scale_y, 0.05f, 1.0f);
            settings.depth_filter_offset_x = std::clamp(settings.depth_filter_offset_x, -1.0f, 1.0f);
            settings.depth_filter_offset_y = std::clamp(settings.depth_filter_offset_y, -1.0f, 1.0f);
            settings.depth_filter_viz_mode = std::clamp(settings.depth_filter_viz_mode, 0, 2);
        }

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

        constexpr std::uint32_t kVksplatIdleScratchReleaseFrames = 30;

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
        viewport_interop_ = std::make_unique<ViewportInteropService>();
        gt_comparison_image_worker_ = std::jthread([this](std::stop_token stop_token) {
            gtComparisonImageWorkerLoop(stop_token);
        });
        camera_metrics_worker_ = std::jthread([this](std::stop_token stop_token) {
            cameraMetricsWorkerLoop(stop_token);
        });
        const auto initial_depth_window = depthWindowFromProjection(settings_);
        panel_depth_windows_ = {initial_depth_window, initial_depth_window};
        setupEventHandlers();
    }

    RenderingManager::~RenderingManager() {
        event_handlers_ = lfs::event::ScopedHandler{};
        invalidateGTComparisonImageCache();
        gt_comparison_image_worker_.request_stop();
        gt_comparison_image_cv_.notify_all();
        if (gt_comparison_image_worker_.joinable()) {
            gt_comparison_image_worker_.join();
        }
        shutdownViewportInterop();
        if (lod_controller_) {
            lod_controller_->setReadyCallback(nullptr);
        }
        camera_metrics_worker_.request_stop();
        camera_metrics_cv_.notify_all();
        lfs::rendering::releaseEnvironmentMapCaches();
    }

    ViewportInteropService& RenderingManager::viewportInterop() {
        assert(viewport_interop_ && "ViewportInteropService not initialized");
        return *viewport_interop_;
    }

    const ViewportInteropService& RenderingManager::viewportInterop() const {
        assert(viewport_interop_ && "ViewportInteropService not initialized");
        return *viewport_interop_;
    }

    void RenderingManager::prepareViewportInterop(VulkanContext& context) {
        viewportInterop().prepareFrame(context, isViewportResizeDeferring());
    }

    void RenderingManager::bindViewportInteropParams(VulkanViewportPassParams& params,
                                                     const std::size_t frame_slot,
                                                     const bool export_locked) {
        viewportInterop().bindViewportParams(params, frame_slot, export_locked,
                                             isViewportResizeDeferring());
    }

    void RenderingManager::shutdownViewportInterop(VulkanContext* context) {
        if (viewport_interop_) {
            viewport_interop_->shutdown(context);
        }
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
        markDirty(DirtyFlag::CAMERA);
    }

    void RenderingManager::markCameraCut() {
        temporal_camera_cut_generation_.fetch_add(1, std::memory_order_release);
        markCameraPoseChanged();
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

    void RenderingManager::requestTemporalFollowUp() {
        dirty_mask_.fetch_or(DirtyFlag::TEMPORAL, std::memory_order_relaxed);

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

    void RenderingManager::setViewportResizeActive(
        const bool active,
        const ViewportResizeRenderPolicy render_policy) {
        if (const DirtyMask dirty = frame_lifecycle_service_.setViewportResizeActive(active, render_policy); dirty) {
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

    void RenderingManager::clearVulkanViewportImageState(const glm::ivec2 size,
                                                         const bool flip_y,
                                                         const glm::ivec2 alloc_size) {
        vulkan_viewport_image_.reset();
        vulkan_external_viewport_image_ = VK_NULL_HANDLE;
        vulkan_external_viewport_image_view_ = VK_NULL_HANDLE;
        vulkan_external_viewport_image_layout_ = VK_IMAGE_LAYOUT_UNDEFINED;
        vulkan_external_viewport_image_generation_ = 0;
        vulkan_viewport_image_size_ = size;
        vulkan_viewport_image_alloc_size_ = alloc_size.x > 0 && alloc_size.y > 0 ? alloc_size : size;
        vulkan_viewport_image_flip_y_ = flip_y;
        vulkan_gt_comparison_content_size_ = {0, 0};
        vulkan_gt_comparison_selection_view_.reset();
    }

    void RenderingManager::releaseSceneRenderResources() {
        vksplat_stale_frame_guard_.onSuccess();
        viewport_artifact_service_.clearViewportOutput();
        invalidateGTComparisonImageCache();
        clearVulkanViewportImageState();
        vulkan_viewport_coordinate_size_ = {0, 0};
        last_logged_vksplat_render_error_.clear();
        vulkan_viewport_image_generation_ = 0;
        split_view_image_generation_ = 0;

        clearVulkanMeshFrame();

        point_cloud_colors_cache_ = {};
        point_cloud_colors_cache_key_ = nullptr;
        point_cloud_colors_cache_size_ = 0;
        ++point_cloud_data_revision_;
        ++point_cloud_preview_selection_revision_;

        if (vksplat_viewport_renderer_) {
            vksplat_viewport_renderer_->reset();
        }
        // Renderer reset frees ring cells; clear manager GT ticket state so the
        // next frame does not poll a stale ticket id against a fresh ring.
        gt_async_depth_ticket_ = 0;
        gt_async_depth_dest_ = {};
        gt_async_ticket_mode_ = GTComparisonMode::RGB;
        gt_async_ticket_intrinsics_.reset();
        gt_async_ticket_flip_y_ = false;
        gt_async_ticket_metadata_ = {};
        gt_async_held_display_.reset();
        gt_async_held_flip_y_ = false;
        gt_async_held_metadata_ = {};
        gt_async_ticket_view_.reset();
        gt_async_held_view_.reset();
        if (point_cloud_vulkan_renderer_) {
            point_cloud_vulkan_renderer_->reset();
        }
        frame_lifecycle_service_.resetModelTracking();
        lfs::core::Tensor::trim_memory_pool();
    }

    void RenderingManager::noteVksplatIdleFrame(const bool training_active) {
        if (!vksplat_viewport_renderer_) {
            vksplat_idle_frame_count_ = 0;
            return;
        }

        auto* const arena = lfs::core::GlobalArenaManager::instance().try_get_arena();
        const bool under_pressure = arena != nullptr && arena->is_under_memory_pressure();

        if (!training_active) {
            vksplat_idle_frame_count_ = 0;
            if (under_pressure) {
                vksplat_viewport_renderer_->releaseScratchOnIdle(true);
            }
            return;
        }

        if (vksplat_idle_frame_count_ < kVksplatIdleScratchReleaseFrames) {
            ++vksplat_idle_frame_count_;
        }
        if (under_pressure || vksplat_idle_frame_count_ >= kVksplatIdleScratchReleaseFrames) {
            // During training the shared arena is owned by FastGS. Only release
            // private viewer allocations here; the terminal callback below is
            // the point at which the shared import may be relinquished.
            vksplat_viewport_renderer_->releaseScratchOnIdle(
                false,
                vksplat_idle_frame_count_ >= kVksplatIdleScratchReleaseFrames);
            vksplat_idle_frame_count_ = 0;
        }
    }

    void RenderingManager::updateSettings(const RenderSettings& new_settings) {
        updateSettings(new_settings, DirtyFlag::ALL);
    }

    void RenderingManager::updateSettings(const RenderSettings& new_settings,
                                          const DirtyMask dirty_flags,
                                          const SceneUpscalerPresetUpdate preset_update) {
        RenderSettings sanitized_settings = new_settings;
        if (const auto requested = sceneUpscalerBackendFromId(sanitized_settings.scene_upscaler);
            requested && !sceneUpscalerBackendAvailable(*requested)) {
            std::string previous_backend_id;
            {
                std::lock_guard lock(settings_mutex_);
                previous_backend_id = settings_.scene_upscaler;
            }
            if (sanitized_settings.scene_upscaler != previous_backend_id) {
                warnUnavailableSceneUpscalerOnce(sanitized_settings.scene_upscaler);
                sanitized_settings.scene_upscaler = std::move(previous_backend_id);
            }
        }
        const auto backend = sceneUpscalerBackendFromId(sanitized_settings.scene_upscaler)
                                 .value_or(SceneUpscalerBackend::Native);
        const std::string backend_id(sceneUpscalerBackendId(backend));
        if (preset_update == SceneUpscalerPresetUpdate::RestoreRememberedForBackend) {
            const auto restored = resolveSceneUpscalerPresetUpdate(
                                      backend,
                                      std::nullopt,
                                      loadSceneUpscalerPresetPreference(backend_id))
                                      .value_or(defaultSceneUpscalerPreset(backend));
            sanitized_settings.scene_upscaler_preset = std::string(restored.id);
        } else if (!sceneUpscalerPreset(backend, sanitized_settings.scene_upscaler_preset)) {
            sanitized_settings.scene_upscaler_preset = loadSceneUpscalerPresetPreference(backend_id);
        }
        const auto preset = sceneUpscalerPreset(backend, sanitized_settings.scene_upscaler_preset)
                                .value_or(defaultSceneUpscalerPreset(backend));
        sanitized_settings.scene_upscaler = backend_id;
        sanitized_settings.scene_upscaler_preset = std::string(preset.id);
        sanitized_settings.scene_upscaler_scale = preset.input_scale;
        bool clear_metrics = false;
        bool lod_request_changed = false;
        bool lod_enabled_turned_on = false;
        // Equal-mode writes can re-enter from latch release and take only
        // settings_mutex_. A mode change releases it before acquiring the
        // transition mutex, then rechecks the mode under both locks.
        std::unique_lock<std::mutex> transition_lock;
        for (;;) {
            std::unique_lock<std::mutex> lock(settings_mutex_);
            const bool split_mode_changes =
                settings_.split_view_mode != sanitized_settings.split_view_mode;
            if (split_mode_changes && !transition_lock.owns_lock()) {
                lock.unlock();
                transition_lock = std::unique_lock<std::mutex>(depth_window_transition_mutex_);
                continue;
            }
            const SplitViewPanelId pre_transition_focus = split_view_service_.focusedPanel();
            const SplitViewMode previous_split_mode = settings_.split_view_mode;
            if (split_view_service_.isGTComparisonActive(settings_) ||
                split_view_service_.isGTComparisonActive(sanitized_settings)) {
                sanitized_settings.show_camera_frustums = false;
            }
            const int focused_panel_index =
                static_cast<int>(splitViewPanelIndex(split_view_service_.focusedPanel()));

            // Without selection intent, a same-mode unsynced write must retain
            // the current projection, including any active drag preview.
            if (!split_mode_changes && !depth_window_sync_ &&
                split_view_service_.isIndependentDualActive(settings_) &&
                (dirty_flags & DirtyFlag::SELECTION) == 0) {
                applyDepthWindowToProjection(sanitized_settings, depthWindowFromProjection(settings_));
            }

            const float previous_depth_filter_scale_x = settings_.depth_filter_scale_x;
            const float previous_depth_filter_scale_y = settings_.depth_filter_scale_y;
            const float previous_depth_filter_offset_x = settings_.depth_filter_offset_x;
            const float previous_depth_filter_offset_y = settings_.depth_filter_offset_y;
            const float previous_depth_filter_min_z = settings_.depth_filter_min.z;
            const float previous_depth_filter_max_z = settings_.depth_filter_max.z;
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

            const float previous_depth_min_z = settings_.depth_filter_min.z;
            const float previous_depth_max_z = settings_.depth_filter_max.z;
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
            sanitizeSelectionWindowSettings(settings_);
            settings_.grid_plane = clampGridPlane(settings_.grid_plane);

            const bool depth_window_projection_changed =
                previous_depth_filter_scale_x != settings_.depth_filter_scale_x ||
                previous_depth_filter_scale_y != settings_.depth_filter_scale_y ||
                previous_depth_filter_offset_x != settings_.depth_filter_offset_x ||
                previous_depth_filter_offset_y != settings_.depth_filter_offset_y ||
                previous_depth_filter_min_z != settings_.depth_filter_min.z ||
                previous_depth_filter_max_z != settings_.depth_filter_max.z;

            if (settings_.split_view_mode == SplitViewMode::Disabled && depth_window_projection_changed) {
                discardRetainedDepthWindowPairLocked(pre_transition_focus);
            }

            if (split_view_service_.isIndependentDualActive(settings_)) {
                if (grid_plane_changed) {
                    panel_grid_planes_[focused_panel_index] = settings_.grid_plane;
                }
                if (depth_window_projection_changed) {
                    const auto projection_window = depthWindowFromProjection(settings_);
                    // These non-drag projection writes supersede backups and ownership for
                    // every slot written.
                    if (depth_window_sync_) {
                        panel_depth_windows_ = {projection_window, projection_window};
                        releaseDepthWindowBackupsLocked(split_view_service_.focusedPanel(),
                                                        /*fan_out=*/true);
                    } else if ((dirty_flags & DirtyFlag::SELECTION) != 0) {
                        panel_depth_windows_[focused_panel_index] = projection_window;
                        releaseDepthWindowBackupsLocked(split_view_service_.focusedPanel(),
                                                        /*fan_out=*/false);
                    }
                }
            } else {
                syncGridPlanesLocked(settings_.grid_plane);
                // settings_ already has the new mode. A write that enters GT from independent-dual
                // and changes projection must not copy that projection into both slots before
                // they are parked below. Treat its depth change as a GT-time global write:
                // preserve the dormant pair.
                const bool entering_gt_from_independent_panels =
                    split_mode_changes &&
                    splitViewUsesIndependentPanels(previous_split_mode) &&
                    splitViewUsesGTComparison(settings_.split_view_mode);
                if (depth_window_projection_changed && !entering_gt_from_independent_panels) {
                    const auto projection_window = depthWindowFromProjection(settings_);
                    panel_depth_windows_ = {projection_window, projection_window};
                    releaseDepthWindowBackupsLocked(split_view_service_.focusedPanel(),
                                                    /*fan_out=*/true);
                }
            }

            if (settings_.depth_filter_min.z != previous_depth_min_z ||
                settings_.depth_filter_max.z != previous_depth_max_z) {
                ++depth_window_projection_generation_;
            }
            // After applying settings, run the event paths' transition logic: epoch bump,
            // backup restoration and seed/collapse. On independent entry, sync grid planes
            // as handleToggleIndependentSplitView and restoreSplitViewMode do.
            // The non-independent branch above already handles grid planes on exit.
            if (split_mode_changes) {
                // Only this call can combine a mode change with a global depth write.
                // Pass the change flag so GT parking preserves the incoming projection in settings_.
                applyDepthWindowModeTransitionLocked(
                    previous_split_mode, settings_.split_view_mode, pre_transition_focus,
                    depth_window_projection_changed);
                if (splitViewUsesIndependentPanels(settings_.split_view_mode)) {
                    syncGridPlanesLocked(settings_.grid_plane);
                }
            }
            markDirty(dirty_flags);
            break;
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

    void RenderingManager::reportSceneUpscalerRuntimeSelection(
        const SceneUpscalerSelection selection) {
        bool changed = false;
        {
            std::lock_guard lock(settings_mutex_);
            changed = scene_upscaler_runtime_selection_ != selection;
            scene_upscaler_runtime_selection_ = selection;
        }
        // The renderer chooses its source resolution before the presentation pass
        // proves whether reconstruction is available. A real active/fallback
        // transition therefore needs one feedback frame: active may adopt the
        // preset scale, while fallback must replace any cached reduced source with
        // a full-resolution native frame. TEMPORAL deliberately avoids restarting
        // the convergence sequence as CAMERA would.
        if (changed)
            requestTemporalFollowUp();
    }

    SceneUpscalerSelection RenderingManager::sceneUpscalerRuntimeSelection() const {
        std::lock_guard lock(settings_mutex_);
        return scene_upscaler_runtime_selection_;
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

    void RenderingManager::advanceSplitOffset() {
        std::lock_guard<std::mutex> lock(settings_mutex_);
        split_view_service_.advanceSplitOffset(settings_);
        markDirty(DirtyFlag::SPLIT_VIEW);
    }

    SplitViewInfo RenderingManager::getSplitViewInfo() const {
        return split_view_service_.getInfo();
    }

    std::optional<SplitViewInfo> RenderingManager::getSplitViewInfoIfChanged(
        std::uint64_t& generation) const {
        return split_view_service_.getInfoIfChanged(generation);
    }

    bool RenderingManager::isSplitViewActive() const {
        std::lock_guard<std::mutex> lock(settings_mutex_);
        return split_view_service_.isActive(settings_);
    }

    bool RenderingManager::isGTComparisonActive() const {
        std::lock_guard<std::mutex> lock(settings_mutex_);
        return split_view_service_.isGTComparisonActive(settings_);
    }

    bool RenderingManager::depthWindowDragActiveLocked() const {
        // Preview counters track latched drags that crossed the draw threshold.
        // Frame capture uses this signal; the sync gate uses ownership instead.
        return depth_window_preview_counts_[0] > 0 || depth_window_preview_counts_[1] > 0;
    }

    bool RenderingManager::depthWindowDragOwnedLocked() const {
        // Ownership lasts from invoke to destruction, including subthreshold presses.
        // The sync gate uses this lifetime so before_ capture cannot straddle a sync change.
        return depth_window_drag_counts_[0] > 0 || depth_window_drag_counts_[1] > 0;
    }

    bool RenderingManager::beginDepthWindowDrag(const SplitViewPanelId panel,
                                                uint64_t& out_drag_token) {
        std::lock_guard<std::mutex> lock(settings_mutex_);
        const size_t index = splitViewPanelIndex(panel);
        // Mint a unique, monotonic token under the same lock as its slot claims.
        const uint64_t token = ++depth_window_last_drag_token_;
        out_drag_token = token;
        // CLAIM RULE: retain the original backup when taking over an owned slot.
        // For an unowned slot, refresh from live state so an old backup cannot
        // become the new drag's undo baseline. Until the next claim, that old
        // backup remains available for transition folding.
        if (!depth_window_pin_owners_[index]) {
            depth_window_drag_backups_[index] = panel_depth_windows_[index];
        }
        // Take ownership while retaining the first backup. Ownership keeps it
        // non-idle regardless of drag count and excludes other tokens' writes,
        // restores and per-slot releases until takeover or supersession.
        depth_window_pin_owners_[index] = token;
        // Sync or a non-independent mode makes previews write both slots.
        // Both need pre-drag backups before the shared value changes; otherwise a
        // replacement drag on the other panel could save the preview as its backup
        // and restore it at the next mode transition.
        const bool fans_out =
            !split_view_service_.isIndependentDualActive(settings_) || depth_window_sync_;
        if (fans_out) {
            const size_t other_index = index == 0 ? 1 : 0;
            // Same claim rule, asked independently of the own slot.
            if (!depth_window_pin_owners_[other_index]) {
                depth_window_drag_backups_[other_index] = panel_depth_windows_[other_index];
            }
            // The other slot has no count for this drag; ownership preserves its
            // backup even through a same-epoch history restore.
            depth_window_pin_owners_[other_index] = token;
        }
        ++depth_window_drag_counts_[index];
        // Report ownership of the other slot so teardown restores every slot
        // this drag's writes may have touched.
        return fans_out;
    }

    void RenderingManager::endDepthWindowDrag(const SplitViewPanelId panel,
                                              const uint64_t drag_token) {
        std::lock_guard<std::mutex> lock(settings_mutex_);
        const size_t index = splitViewPanelIndex(panel);
        if (depth_window_drag_counts_[index] == 0) {
            return;
        }
        --depth_window_drag_counts_[index];
        // IDENTITY GATE: release only slots this drag still owns.
        // Mode transitions, project restore and non-drag writes can clear ownership;
        // later drags can take it over. Per-slot checks protect new owners while
        // ensuring this drag releases every slot it still owns.
        for (size_t slot = 0; slot < depth_window_pin_owners_.size(); ++slot) {
            if (depth_window_pin_owners_[slot] != drag_token) {
                continue;
            }
            depth_window_pin_owners_[slot].reset();
            // Keep the backup after ownership ends. Mode transitions and project restore
            // cancel drags before folding backups; clearing it here would leave only the
            // drag's preview. The next claim refreshes an unowned slot's backup from live
            // state, preventing a stale undo baseline for the new drag.
        }
    }

    void RenderingManager::beginDepthWindowPreview(const SplitViewPanelId panel) {
        {
            std::lock_guard<std::mutex> lock(settings_mutex_);
            ++depth_window_preview_counts_[splitViewPanelIndex(panel)];
        }
        markDirty(DirtyFlag::OVERLAY);
    }

    void RenderingManager::endDepthWindowPreview(const SplitViewPanelId panel) {
        {
            std::lock_guard<std::mutex> lock(settings_mutex_);
            const size_t index = splitViewPanelIndex(panel);
            if (depth_window_preview_counts_[index] == 0) {
                return;
            }
            --depth_window_preview_counts_[index];
        }
        markDirty(DirtyFlag::OVERLAY);
    }

    bool RenderingManager::depthWindowDragPreview() const {
        std::lock_guard<std::mutex> lock(settings_mutex_);
        return depthWindowDragActiveLocked();
    }

    GTComparisonMode RenderingManager::getGTComparisonMode() const {
        std::lock_guard<std::mutex> lock(settings_mutex_);
        return settings_.gt_comparison_mode;
    }

    SplitViewMode RenderingManager::getSplitViewMode() const {
        std::lock_guard<std::mutex> lock(settings_mutex_);
        return settings_.split_view_mode;
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
            applyDepthWindowProjectionLocked(panel_depth_windows_[splitViewPanelIndex(panel)]);
        }
    }

    int RenderingManager::getGridPlaneForPanel(const SplitViewPanelId panel) const {
        std::lock_guard<std::mutex> lock(settings_mutex_);
        return panel_grid_planes_[splitViewPanelIndex(panel)];
    }

    DepthWindowState RenderingManager::getDepthWindowForPanel(const SplitViewPanelId panel) const {
        std::lock_guard<std::mutex> lock(settings_mutex_);
        return panel_depth_windows_[splitViewPanelIndex(panel)];
    }

    RenderingManager::DepthWindowOverlaySnapshot RenderingManager::getDepthWindowOverlaySnapshot() const {
        std::lock_guard<std::mutex> lock(settings_mutex_);
        return {
            .independent_dual_active = split_view_service_.isIndependentDualActive(settings_),
            .panel_windows = panel_depth_windows_,
        };
    }

    void RenderingManager::releaseDepthWindowBackupsLocked(const SplitViewPanelId panel,
                                                           const bool fan_out) {
        // Callers have replaced these slots with a commit or deliberate non-drag write
        // (settings, panel setter or sync copy). Clear stale backups and ownership,
        // regardless of drag counts, so the old drag cannot overwrite the new value
        // through preview or teardown (restorePinnedDepthWindowSlots).
        // Drag preview writes never call this helper.
        const auto supersede = [this](const size_t slot) {
            depth_window_drag_backups_[slot].reset();
            depth_window_pin_owners_[slot].reset();
        };
        const size_t index = splitViewPanelIndex(panel);
        supersede(index);
        if (!fan_out) {
            return;
        }
        supersede(index == 0 ? 1 : 0);
    }

    bool RenderingManager::restorePinnedDepthWindowSlots(
        const SplitViewPanelId panel,
        const DepthWindowState& own_state,
        const std::optional<DepthWindowState>& other_state,
        const uint64_t expected_epoch,
        const uint64_t drag_token) {
        DepthWindowState clamped_own = own_state;
        clampDepthWindowState(clamped_own);
        std::lock_guard<std::mutex> lock(settings_mutex_);
        if (depth_window_mode_epoch_ != expected_epoch) {
            return false;
        }
        const size_t index = splitViewPanelIndex(panel);
        const size_t other_index = index == 0 ? 1 : 0;
        const bool owns_addressed_slot = depth_window_pin_owners_[index] == drag_token;
        bool wrote = false;
        // Under one lock, restore only slots still owned by this drag.
        // Skip slots whose ownership was cleared by a newer write or taken by another
        // drag, preserving their newer values.
        if (owns_addressed_slot) {
            applyDepthWindowForPanelLocked(panel, clamped_own, /*restore_mode=*/true);
            wrote = true;
        }
        if (other_state && depth_window_pin_owners_[other_index] == drag_token) {
            DepthWindowState clamped_other = *other_state;
            clampDepthWindowState(clamped_other);
            applyDepthWindowForPanelLocked(
                other_index == 0 ? SplitViewPanelId::Left : SplitViewPanelId::Right,
                clamped_other, /*restore_mode=*/true);
            wrote = true;
        }
        if (wrote) {
            markDirty(DirtyFlag::ALL);
        }
        return owns_addressed_slot;
    }

    void RenderingManager::releaseIdleDepthWindowBackupsLocked() {
        for (size_t index = 0; index < depth_window_drag_backups_.size(); ++index) {
            // A drag writing both slots can own a backup where the drag count is zero.
            // Keep owned backups so the next transition can restore pre-drag state.
            if (depth_window_drag_counts_[index] == 0 &&
                !depth_window_pin_owners_[index]) {
                depth_window_drag_backups_[index].reset();
            }
        }
    }

    bool RenderingManager::applyDepthWindowForPanelLocked(const SplitViewPanelId panel,
                                                          const DepthWindowState& clamped,
                                                          const bool restore_mode) {
        const bool independent_dual = split_view_service_.isIndependentDualActive(settings_);
        // Restore one slot regardless of sync: this undoes the drag's preview,
        // not a user edit.
        const bool fan_out = !restore_mode && (!independent_dual || depth_window_sync_);
        const size_t panel_index = splitViewPanelIndex(panel);
        if (fan_out) {
            panel_depth_windows_.fill(clamped);
        } else {
            panel_depth_windows_[panel_index] = clamped;
        }
        // Outside independent-dual, the projection is the window.
        // Even a single-slot restore must update it.
        const bool updates_projection = fan_out ||
                                        split_view_service_.focusedPanel() == panel ||
                                        (restore_mode && !independent_dual);
        if (updates_projection) {
            applyDepthWindowProjectionLocked(clamped);
        }
        return fan_out;
    }

    // Depth-window writes need DirtyFlag::ALL, as in one-argument updateSettings.
    // SELECTION re-rasterizes cached per-splat containment, leaving stale
    // classifications on screen until a full render.
    bool RenderingManager::setDepthWindowForPanel(const SplitViewPanelId panel, const DepthWindowState& state) {
        DepthWindowState clamped = state;
        clampDepthWindowState(clamped);
        std::lock_guard<std::mutex> lock(settings_mutex_);
        if (depth_window_dormant_panels_ && splitViewUsesGTComparison(settings_.split_view_mode)) {
            return false;
        }
        if (settings_.split_view_mode == SplitViewMode::Disabled && clamped != depthWindowFromProjection(settings_)) {
            discardRetainedDepthWindowPairLocked(split_view_service_.focusedPanel());
        }
        const bool fan_out = applyDepthWindowForPanelLocked(panel, clamped);
        releaseDepthWindowBackupsLocked(panel, fan_out);
        markDirty(DirtyFlag::ALL);
        return true;
    }

    bool RenderingManager::applyDepthWindowForPanelIfEpoch(const SplitViewPanelId panel,
                                                           const DepthWindowState& state,
                                                           const uint64_t expected_epoch,
                                                           const uint64_t drag_token) {
        DepthWindowState clamped = state;
        clampDepthWindowState(clamped);
        std::lock_guard<std::mutex> lock(settings_mutex_);
        if (depth_window_mode_epoch_ != expected_epoch) {
            return false;
        }
        // A drag may write only while it owns its panel's slot. Non-drag writes clear
        // ownership; replacement drags take it over. Refuse superseded writes without
        // changing state, just as for a stale epoch.
        if (depth_window_pin_owners_[splitViewPanelIndex(panel)] != drag_token) {
            return false;
        }
        applyDepthWindowForPanelLocked(panel, clamped);
        markDirty(DirtyFlag::ALL);
        return true;
    }

    bool RenderingManager::commitDepthWindowForPanelIfEpoch(
        const SplitViewPanelId panel,
        const DepthWindowState& state,
        const uint64_t expected_epoch,
        const uint64_t drag_token,
        op::DepthWindowModeSnapshot& out_snapshot) {
        DepthWindowState clamped = state;
        clampDepthWindowState(clamped);
        std::lock_guard<std::mutex> lock(settings_mutex_);
        if (depth_window_mode_epoch_ != expected_epoch) {
            return false;
        }
        // Like preview writes, commits require this drag to own the panel's slot.
        // Refuse a superseded drag's release without side effects, as for a stale epoch.
        if (depth_window_pin_owners_[splitViewPanelIndex(panel)] != drag_token) {
            return false;
        }
        // Compare a commit with committed pre-drag geometry, not its preview.
        const auto& backup = depth_window_drag_backups_[splitViewPanelIndex(panel)];
        if (settings_.split_view_mode == SplitViewMode::Disabled &&
            clamped != backup.value_or(depthWindowFromProjection(settings_))) {
            discardRetainedDepthWindowPairLocked(split_view_service_.focusedPanel());
        }
        const bool fan_out = applyDepthWindowForPanelLocked(panel, clamped);
        // The value is now committed. Discard its pre-drag backup so a later mode
        // transition cannot roll it back as an uncommitted preview.
        releaseDepthWindowBackupsLocked(panel, fan_out);
        // Keep the epoch check, write and undo snapshot under one lock so no mode
        // transition can intervene.
        out_snapshot = depthWindowSnapshotLocked();
        markDirty(DirtyFlag::ALL);
        return true;
    }

    void RenderingManager::setDepthWindowSync(const bool sync) {
        std::optional<std::pair<op::DepthWindowModeSnapshot, op::DepthWindowModeSnapshot>> undo_snapshots;
        {
            std::lock_guard<std::mutex> lock(settings_mutex_);
            if (depth_window_sync_ == sync) {
                return;
            }
            // Ignore sync changes from drag invoke to destruction, including subthreshold
            // presses. Otherwise the toggle's undo snapshot could capture preview geometry,
            // and the drag's before_ baseline could disagree with the new sync state.
            if (depthWindowDragOwnedLocked()) {
                return;
            }
            if (settings_.split_view_mode == SplitViewMode::Disabled) {
                // An actual global sync edit ends retention before the GT guard.
                discardRetainedDepthWindowPairLocked(split_view_service_.focusedPanel());
            }
            // GT has no per-panel edit target. Refuse while it parks a pair: sync undo
            // restores live slots, not that pair, and could leave unequal windows with
            // sync on. GT without a parked pair is unaffected.
            if (depth_window_dormant_panels_) {
                return;
            }
            bool slots_changed = false;
            if (sync && split_view_service_.isIndependentDualActive(settings_)) {
                const size_t focused_index =
                    splitViewPanelIndex(split_view_service_.focusedPanel());
                const size_t other_index = focused_index == 0 ? 1 : 0;
                if (panel_depth_windows_[focused_index] != panel_depth_windows_[other_index]) {
                    const op::DepthWindowModeSnapshot before_snapshot = depthWindowSnapshotLocked();
                    panel_depth_windows_[other_index] = panel_depth_windows_[focused_index];
                    // Discard stale backups after this sync copy, as setDepthWindowForPanel does,
                    // so a later transition cannot restore the pre-drag window over the copied value.
                    releaseIdleDepthWindowBackupsLocked();
                    // Like collapse on leaving independent-dual, this copy replaces the other
                    // panel's window with the focused one. Stamp lineage under the copy's lock
                    // so cached per-panel state cannot reuse the discarded window after a hidden
                    // ON -> OFF -> focus-change -> ON cycle. Equal slots need no copy or stamp.
                    stampDepthWindowLineageLocked(split_view_service_.focusedPanel(),
                                                  DepthWindowLineageKind::SyncCopy);
                    slots_changed = true;
                    op::DepthWindowModeSnapshot after_snapshot = depthWindowSnapshotLocked();
                    after_snapshot.sync = true;
                    undo_snapshots = {before_snapshot, after_snapshot};
                }
            }
            depth_window_sync_ = sync;
            if (slots_changed) {
                markDirty(DirtyFlag::ALL);
            }
        }
        if (undo_snapshots) {
            op::undoHistory().push(std::make_unique<op::DepthWindowSyncUndoEntry>(
                *this, undo_snapshots->first, undo_snapshots->second));
        }
    }

    bool RenderingManager::getDepthWindowSync() const {
        std::lock_guard<std::mutex> lock(settings_mutex_);
        return depth_window_sync_;
    }

    SplitViewPanelId RenderingManager::getDepthWindowCollapseSource() const {
        std::lock_guard<std::mutex> lock(settings_mutex_);
        return depth_window_collapse_source_;
    }

    RenderingManager::DepthWindowCollapseRecord
    RenderingManager::getDepthWindowCollapseRecord() const {
        std::lock_guard<std::mutex> lock(settings_mutex_);
        return {depth_window_collapse_source_,
                depth_window_collapse_generation_,
                depth_window_collapse_kind_};
    }

    void RenderingManager::discardRetainedDepthWindowPairLocked(const SplitViewPanelId source) {
        if (!depth_window_dormant_panels_) {
            return;
        }
        depth_window_dormant_panels_.reset();
        stampDepthWindowLineageLocked(source, DepthWindowLineageKind::RetainedPairDiscard);
    }

    void RenderingManager::stampDepthWindowLineageLocked(
        const SplitViewPanelId source,
        const DepthWindowLineageKind kind) {
        // Sole writer of source, generation and kind; settings_mutex_ keeps them consistent.
        // Invalidating writes stamp under the same lock as slot changes, except sync
        // undo/redo, which stamps under a second acquisition (DepthWindowSyncUndoEntry::apply).
        // The record is self-consistent, but not atomic with the slots. Consumers reading
        // them separately must revalidate generation around their reads.
        depth_window_collapse_source_ = source;
        depth_window_collapse_kind_ = kind;
        ++depth_window_collapse_generation_;
    }

    op::DepthWindowModeSnapshot RenderingManager::depthWindowSnapshotLocked() const {
        return {
            .panels = panel_depth_windows_,
            .sync = depth_window_sync_,
            .projection = depthWindowFromProjection(settings_),
            .mode_epoch = depth_window_mode_epoch_,
            .independent_dual = split_view_service_.isIndependentDualActive(settings_),
        };
    }

    op::DepthWindowModeSnapshot RenderingManager::depthWindowSnapshot() const {
        std::lock_guard<std::mutex> lock(settings_mutex_);
        return depthWindowSnapshotLocked();
    }

    op::DepthWindowModeSnapshot
    RenderingManager::depthWindowBaselineSnapshotForDrag(const uint64_t drag_token) const {
        std::lock_guard<std::mutex> lock(settings_mutex_);
        auto snapshot = depthWindowSnapshotLocked();
        if (drag_token == 0) {
            return snapshot;
        }
        for (size_t i = 0; i < snapshot.panels.size(); ++i) {
            if (depth_window_pin_owners_[i] == drag_token && depth_window_drag_backups_[i]) {
                snapshot.panels[i] = *depth_window_drag_backups_[i];
            }
        }
        return snapshot;
    }

    void RenderingManager::restoreDepthWindowStateFromProject() {
        // Project restore advances the epoch; serialize it with drag release sequences.
        // Acquire transition before settings/history locks; release settings before
        // pushing history.
        const auto transition_lock = acquireDepthWindowTransitionLock();
        std::lock_guard<std::mutex> lock(settings_mutex_);
        const auto projection_window = depthWindowFromProjection(settings_);
        restoreDepthWindowStateLocked({projection_window, projection_window}, false, projection_window);
        ++depth_window_projection_generation_;
        // Project restore advances the epoch, expiring drags and undo entries from
        // the previous session. Clear their pre-drag backups so later mode transitions
        // cannot restore old windows over the loaded state.
        depth_window_drag_backups_ = {};
        // Clear ownership with the backups. endDepthWindowDrag releases only slots
        // it still owns, so surviving drags leave these slots untouched.
        depth_window_pin_owners_ = {};
        // Like the backups, the parked pair belongs to the previous session.
        // Discard it so leaving GT cannot overwrite the restored windows.
        depth_window_dormant_panels_.reset();
        // Restore seeds both slots from the project's projection, invalidating cached
        // panel references. Stamp this even if independent-dual, sync OFF and focus
        // stay unchanged, so polling detects the reset. The source names the focused
        // panel only to complete the record; consumers ignore it for ProjectRestore.
        stampDepthWindowLineageLocked(split_view_service_.focusedPanel(),
                                      DepthWindowLineageKind::ProjectRestore);
        ++depth_window_mode_epoch_;
    }

    bool RenderingManager::restoreDepthWindowSnapshotIfEpoch(
        const op::DepthWindowModeSnapshot& snapshot,
        const uint64_t expected_epoch,
        const bool restore_sync) {
        std::lock_guard<std::mutex> lock(settings_mutex_);
        if (depth_window_mode_epoch_ != expected_epoch) {
            return false;
        }
        // Focus may have changed: restore projection from the currently focused
        // panel's snapshot slot. restore_sync restores the saved sync flag for sync
        // entries; drag entries restore slots only and preserve the current flag.
        const auto& focused_slot =
            snapshot.panels[splitViewPanelIndex(split_view_service_.focusedPanel())];
        restoreDepthWindowStateLocked(snapshot.panels,
                                      restore_sync ? snapshot.sync : depth_window_sync_,
                                      focused_slot);
        markDirty(DirtyFlag::ALL);
        return true;
    }

    void RenderingManager::stampDepthWindowSyncRestoreLineage() {
        std::lock_guard<std::mutex> lock(settings_mutex_);
        // ProjectRestore means "fresh baseline required" here: sync undo/redo restores
        // both absolute window snapshots, invalidating cached panel references.
        // This is not a project load. The focused source only completes the record;
        // consumers ignore it for this kind.
        stampDepthWindowLineageLocked(split_view_service_.focusedPanel(),
                                      DepthWindowLineageKind::ProjectRestore);
    }

    uint64_t RenderingManager::depthWindowProjectionGeneration() const {
        std::lock_guard<std::mutex> lock(settings_mutex_);
        return depth_window_projection_generation_;
    }

    uint64_t RenderingManager::depthWindowModeEpoch() const {
        std::lock_guard<std::mutex> lock(settings_mutex_);
        return depth_window_mode_epoch_;
    }

    void RenderingManager::applyDepthWindowProjectionLocked(const DepthWindowState& state) {
        const float previous_depth_min_z = settings_.depth_filter_min.z;
        const float previous_depth_max_z = settings_.depth_filter_max.z;
        applyDepthWindowToProjection(settings_, state);
        if (settings_.depth_filter_min.z != previous_depth_min_z ||
            settings_.depth_filter_max.z != previous_depth_max_z) {
            ++depth_window_projection_generation_;
        }
    }

    void RenderingManager::restoreDepthWindowStateLocked(
        const std::array<DepthWindowState, 2>& panels,
        const bool sync,
        const DepthWindowState& projection) {
        panel_depth_windows_ = panels;
        depth_window_sync_ = sync;
        // Undo/redo and project restore replace both slots with valid state.
        // Discard stale backups only for unowned slots with a zero drag count.
        releaseIdleDepthWindowBackupsLocked();
        applyDepthWindowProjectionLocked(projection);
    }

    void RenderingManager::applyDepthWindowModeTransitionLocked(
        const SplitViewMode previous_mode,
        const SplitViewMode new_mode,
        const SplitViewPanelId pre_transition_focus,
        const bool boundary_carries_global_depth_write) {
        // Other comparison modes first invalidate any retained pair. Only an
        // independent/GT boundary expires drags and folds their previews;
        // Disabled <-> PLY otherwise preserves the global drag lifetime.
        const bool was_independent = splitViewUsesIndependentPanels(previous_mode);
        const bool is_independent = splitViewUsesIndependentPanels(new_mode);
        if (!is_independent && !splitViewUsesGTComparison(new_mode) && new_mode != SplitViewMode::Disabled) {
            discardRetainedDepthWindowPairLocked(pre_transition_focus);
        }
        const bool independent_boundary = was_independent != is_independent;
        const bool gt_boundary =
            splitViewUsesGTComparison(previous_mode) != splitViewUsesGTComparison(new_mode);
        if (!independent_boundary && !gt_boundary) {
            return;
        }

        // GT suspends filtering. Park and restore the panel pair separately
        // from ordinary collapse/seed transitions, which would homogenize it.
        const bool is_gt = splitViewUsesGTComparison(new_mode);
        const bool park_dormant_panels = was_independent && is_gt;
        const bool restore_dormant_panels =
            is_independent && depth_window_dormant_panels_.has_value();

        // A modal checked out before cancellation can race this transition and write
        // a preview. Restore all recorded backups first so both panels collapse from
        // pre-drag state whichever operation won the mutex. Discard the backups so no
        // surviving or replacement drag inherits them across the epoch boundary.
        const size_t focused_index = splitViewPanelIndex(split_view_service_.focusedPanel());
        for (size_t index = 0; index < depth_window_drag_backups_.size(); ++index) {
            if (!depth_window_drag_backups_[index]) {
                continue;
            }
            const DepthWindowState backup_window = *depth_window_drag_backups_[index];
            panel_depth_windows_[index] = backup_window;
            // Parking sets projection below from pre-transition focus or an explicit boundary
            // write. Events may have reset current focus; direct updateSettings keeps it.
            if (index == focused_index && !park_dormant_panels) {
                applyDepthWindowProjectionLocked(backup_window);
            }
            depth_window_drag_backups_[index].reset();
        }
        // All backups are consumed; clear all slot ownership. Surviving drags have
        // no slots to release, and further writes fail both ownership and epoch checks.
        depth_window_pin_owners_ = {};

        if (park_dormant_panels) {
            // Backups are restored and ownership cleared. Park both pre-drag windows;
            // their references survive, so no lineage stamp is needed.
            depth_window_dormant_panels_ = panel_depth_windows_;
            // Preserve an explicit GT-boundary projection. Otherwise remove
            // any abandoned preview using the clean pre-transition focused slot.
            if (!boundary_carries_global_depth_write) {
                const auto parked_focused_window =
                    panel_depth_windows_[splitViewPanelIndex(pre_transition_focus)];
                applyDepthWindowProjectionLocked(parked_focused_window);
            }
        } else if (restore_dormant_panels) {
            // On independent entry, restore the exact retained pair over slot values from
            // GT-time global writes. Projection follows current focus: event-driven
            // transitions reset it to Left; direct settings writes keep it.
            // Neither restores pre-GT focus.
            panel_depth_windows_ = *depth_window_dormant_panels_;
            const auto focused_window =
                panel_depth_windows_[splitViewPanelIndex(split_view_service_.focusedPanel())];
            applyDepthWindowProjectionLocked(focused_window);
        } else if (splitViewUsesGTComparison(previous_mode) && new_mode == SplitViewMode::Disabled) {
            // Disabled drags back up live slots, so align both with the global projection.
            // Keep the retained pair separate for the next independent entry.
            panel_depth_windows_.fill(depthWindowFromProjection(settings_));
        } else if (!was_independent && is_independent) {
            const auto projection_window = depthWindowFromProjection(settings_);
            panel_depth_windows_ = {projection_window, projection_window};
        } else if (was_independent && !is_independent) {
            // Record the source from before the focus reset to Left so toolbar Size references
            // follow its window. Stamp source, generation and kind under the collapse lock.
            // Source alone hides intermediate transitions in a leave -> enter -> leave cycle.
            // Comparing the generation delta with the observed transition reveals missed
            // transitions; kind tells the poller how to recover.
            stampDepthWindowLineageLocked(pre_transition_focus,
                                          DepthWindowLineageKind::LeaveCollapse);
            const auto collapsed =
                panel_depth_windows_[splitViewPanelIndex(pre_transition_focus)];
            applyDepthWindowProjectionLocked(collapsed);
            panel_depth_windows_ = {collapsed, collapsed};
        }

        // GT -> Disabled and repeated GT entry retain the original pair until
        // an invalidating edit or comparison mode. Independent entry consumes it.
        if (restore_dormant_panels) {
            depth_window_dormant_panels_.reset();
        }

        // For the boundary detected above, advance the epoch to reject further writes
        // and commits from drags that survive cancellation, and expire pre-transition
        // undo entries.
        ++depth_window_mode_epoch_;
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

    void RenderingManager::clearLatestCameraMetrics() {
        {
            std::lock_guard<std::mutex> lock(camera_metrics_mutex_);
            latest_camera_metrics_.reset();
        }
        app_store().camera_metrics.set(std::optional<AppStore::CameraMetrics>{});
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

        std::optional<AppStore::CameraMetrics> cached_app_metrics;
        {
            std::lock_guard<std::mutex> lock(camera_metrics_mutex_);
            const auto cached = std::find_if(
                camera_metrics_cache_.begin(), camera_metrics_cache_.end(),
                [&request_matches, &request](const auto& entry) {
                    return request_matches(entry.request, request);
                });
            if (cached != camera_metrics_cache_.end()) {
                const auto& cached_metrics = cached->metrics;
                const bool metrics_changed =
                    !latest_camera_metrics_ ||
                    latest_camera_metrics_->camera_id != cached_metrics.camera_id ||
                    latest_camera_metrics_->iteration != cached_metrics.iteration ||
                    latest_camera_metrics_->psnr != cached_metrics.psnr ||
                    latest_camera_metrics_->ssim != cached_metrics.ssim ||
                    latest_camera_metrics_->used_mask != cached_metrics.used_mask;
                latest_camera_metrics_ = cached->metrics;
                if (metrics_changed) {
                    cached_app_metrics = toAppCameraMetrics(cached_metrics);
                }
                last_camera_metrics_refresh_time_ = now;
                cached->request = request;
            } else {

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

                if ((immediate_refresh || (stale_iteration && refresh_interval_elapsed)) &&
                    !same_as_pending &&
                    !same_as_active) {
                    request.generation = ++camera_metrics_request_generation_;
                    pending_camera_metrics_request_ = request;
                    last_camera_metrics_refresh_time_ = now;
                    should_queue = true;
                }
            }
        }

        if (cached_app_metrics) {
            app_store().camera_metrics.set(std::move(cached_app_metrics));
            markDirty(DirtyFlag::OVERLAY);
            return;
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
                        const auto same_cached_request = [&](const auto& entry) {
                            return entry.request.trainer_manager == request.trainer_manager &&
                                   entry.request.camera_id == request.camera_id &&
                                   entry.request.iteration == request.iteration &&
                                   entry.request.settings.camera_metrics_mode == request.settings.camera_metrics_mode &&
                                   entry.request.settings.apply_appearance_correction == request.settings.apply_appearance_correction &&
                                   entry.request.settings.ppisp_mode == request.settings.ppisp_mode &&
                                   ppispOverridesEqual(entry.request.settings.ppisp_overrides,
                                                       request.settings.ppisp_overrides);
                        };
                        auto cached = std::find_if(
                            camera_metrics_cache_.begin(), camera_metrics_cache_.end(),
                            same_cached_request);
                        if (cached == camera_metrics_cache_.end()) {
                            camera_metrics_cache_.push_back(
                                {.request = request, .metrics = *metrics});
                        } else {
                            cached->metrics = *metrics;
                            cached->request = request;
                        }
                        while (camera_metrics_cache_.size() > 4) {
                            camera_metrics_cache_.pop_front();
                        }
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
        if (!split_view_service_.isActive(settings_) ||
            (splitViewUsesGTComparison(settings_.split_view_mode) &&
             gtComparisonShowsLoss(settings_.gt_comparison_mode))) {
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

        // Any GT enter/exit invalidates the published GT selection camera, whether or not the
        // viewport output is cleared: clear_viewport_output is only set for
        // enabled -> disabled (split_view_service.cpp:202), so re-entering GT would otherwise
        // expose the previous session's camera until the next GT frame is presented.
        vulkan_gt_comparison_selection_view_.reset();
        vulkan_gt_comparison_content_size_ = {0, 0};

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
                                                 const int focused_gaussian_id, const bool request_render) {
        viewport_overlay_service_.setCursorPreview(active, x, y, radius, add_mode, selection_tensor,
                                                   saturation_mode, saturation_amount, panel, focused_gaussian_id);
        if (request_render)
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

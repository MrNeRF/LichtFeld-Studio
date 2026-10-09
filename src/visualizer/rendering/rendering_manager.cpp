/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "rendering_manager.hpp"
#include "core/camera_metrics.hpp"
#include "core/events.hpp"
#include "core/logger.hpp"
#include "core/raster_arena.hpp"
#include "core/tensor_backend.hpp"
#include "operation/undo_entry.hpp"
#include "operation/undo_history.hpp"
#include "preferences.hpp"
#include "rendering/export_post_process.hpp"
#include "rendering/rendering.hpp"
#include "rendering/scene_upscaler_registry.hpp"
#include "rendering/selection_ops.hpp"
#include "scene/scene_manager.hpp"
#include "scene_renderer.hpp"
#include "scene_renderer_factory.hpp"
#include "scene_training_interop.hpp"
#include "theme/theme.hpp"
#include "viewport_reference_renderer.hpp"
#include "window/graphics_context.hpp"
#if LFS_BUILD_TRAINER
#include "training/trainer.hpp"
#endif
#include "core/training_manager.hpp"
#include "visualizer/app_store.hpp"

#include <algorithm>
#include <bit>
#include <cassert>
#include <cmath>
#include <limits>
#include <mutex>
#include <stdexcept>
#include <string>
#include <unordered_set>
#include <utility>

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
            // Fresh disabled settings still carry the legacy positive-Z sentinel.
            // Decode it into a forward depth interval before the first edit.
            const bool legacy_default = !settings.depth_filter_enabled &&
                                        settings.depth_filter_min.z == 0.0f &&
                                        settings.depth_filter_max.z == 100.0f;
            return {
                .near_plane = legacy_default ? 0.0f : -settings.depth_filter_max.z,
                .far_plane = legacy_default ? 100.0f : -settings.depth_filter_min.z,
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
                   old_settings.ppisp_overrides != new_settings.ppisp_overrides;
        }

        constexpr auto kVksplatIdleScratchReleaseDelay = std::chrono::seconds{3};

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
            lfs::training::CameraMetricsAppearanceConfig appearance{};
            appearance.enabled = settings.apply_appearance_correction;
            appearance.use_controller =
                settings.ppisp_mode == RenderSettings::PPISPMode::AUTO;
            appearance.overrides = settings.ppisp_overrides;

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

    // RenderingManager Implementation
    RenderingManager::RenderingManager(ViewSource& views) : view_source_(views) {
        screen_epoch_ = views.screenEpoch();

        frame_demand_ledger_.request(FrameRequest{.reason = FrameReason::Startup,
                                                  .scope = FrameScope::All,
                                                  .flags = DirtyFlag::ALL,
                                                  .detail = "startup"});
        frame_demand_ledger_.setWakeCallback([this] {
            std::function<void()> wake_callback;
            {
                std::scoped_lock lock(wake_callback_mutex_);
                wake_callback = wake_callback_;
            }
            if (wake_callback)
                wake_callback();
        });
        gt_comparison_image_worker_ = std::jthread([this](std::stop_token stop_token) {
            gtComparisonImageWorkerLoop(stop_token);
        });
        camera_metrics_worker_ = std::jthread([this](std::stop_token stop_token) {
            cameraMetricsWorkerLoop(stop_token);
        });
        setupEventHandlers();
    }

    RenderingManager::~RenderingManager() {
        event_handlers_ = lfs::event::ScopedHandler{};
        invalidateGTComparisonImageCache(state());
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

    void RenderingManager::clearViewportSceneImage() {
        clearViewportReference(this->state());
    }

    void RenderingManager::shutdownViewportInterop(GraphicsContext* context) {
        std::lock_guard lock(views_mutex_);
        for (auto& [id, view] : view_states_)
            shutdownViewportReference(*view, context);
        for (auto& view : retired_view_states_)
            shutdownViewportReference(*view, context);
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
    }

    ViewRenderState& RenderingManager::viewState(ViewId id) const {
        std::lock_guard lock(views_mutex_);
        auto& entry = view_states_[id];
        if (!entry) {
            entry = std::make_unique<ViewRenderState>();
            entry->id = id;
            if (const auto saved = depth_window_epochs_.find(id); saved != depth_window_epochs_.end()) {
                entry->depth_window_mode_epoch_ = saved->second.first;
                entry->depth_window_projection_generation_ = saved->second.second;
            }
            entry->last_visible = std::chrono::steady_clock::now();
        }
        return *entry;
    }

    ViewRenderState& RenderingManager::state() const {
        return viewState(view_source_.activeView());
    }

    void RenderingManager::markDirty(const DirtyMask flags, const FrameReason reason, std::string detail) {
        std::lock_guard lock(views_mutex_);
        for (auto& [id, view] : view_states_)
            view->dirty_mask_.fetch_or(flags, std::memory_order_relaxed);
        if (flags)
            frame_demand_ledger_.request({.reason = reason, .scope = FrameScope::All, .flags = flags, .detail = std::move(detail)});
    }

    ViewMask RenderingManager::viewMask(ViewId view) const {
        std::lock_guard lock(views_mutex_);
        const auto it = std::find(ledger_views_.begin(), ledger_views_.end(), view);
        return it == ledger_views_.end() ? 0 : ViewMask{1} << std::distance(ledger_views_.begin(), it);
    }

    ViewMask RenderingManager::visibleViewMask() const {
        std::lock_guard lock(views_mutex_);
        return ledger_views_.size() == kMaxViews ? ~ViewMask{0} : (ViewMask{1} << ledger_views_.size()) - 1;
    }

    void RenderingManager::markViewDirty(ViewId view, DirtyMask flags, FrameReason reason, std::string detail) {
        viewState(view).dirty_mask_.fetch_or(flags, std::memory_order_relaxed);
        if (const auto mask = viewMask(view); flags && mask)
            frame_demand_ledger_.request({.reason = reason, .scope = FrameScope::View, .views = mask, .flags = flags, .detail = std::move(detail)});
    }

    void RenderingManager::requestViewportResize(const ViewId id, const glm::ivec2 size) {
        auto& view = viewState(id);
        if (view.requested_viewport_size_ == size)
            return;
        view.requested_viewport_size_ = size;
        // Layout can finish after the command's redraw used the old extent.
        markViewDirty(id, DirtyFlag::VIEWPORT | DirtyFlag::CAMERA | DirtyFlag::OVERLAY,
                      FrameReason::ViewportResize);
    }

    std::uint64_t RenderingManager::viewInputFingerprint(const Viewport& viewport,
                                                         const SceneManager* scene_manager, const ViewId view) const {
        std::uint64_t hash = 1469598103934665603ULL;
        const auto append = [&hash](const std::uint64_t value) {
            hash ^= value;
            hash *= 1099511628211ULL;
        };
        for (int column = 0; column < 3; ++column) {
            for (int row = 0; row < 3; ++row)
                append(std::bit_cast<std::uint32_t>(viewport.getRotationMatrix()[column][row]));
        }
        const auto translation = viewport.getTranslation();
        for (int axis = 0; axis < 3; ++axis)
            append(std::bit_cast<std::uint32_t>(translation[axis]));
        append(static_cast<std::uint32_t>(viewport.windowSize.x));
        append(static_cast<std::uint32_t>(viewport.windowSize.y));
        append(static_cast<std::uint32_t>(viewport.frameBufferSize.x));
        append(static_cast<std::uint32_t>(viewport.frameBufferSize.y));
        // These generations cover inputs that do not live on Viewport:
        // loaded/model data, selection, and render settings (including
        // depth, split/GT comparison, and overlay toggles).
        const auto& store = app_store();
        append(store.scene_generation.get());
        append(store.selection_generation.get());
        if (scene_manager) {
            // Cache invalidation also advances between scheduled training previews.
            // Published scene mutations are covered by scene_generation above.
            // A cache revision alone does not require an immediate redraw.
            append(scene_manager->getScene().selectionGeneration());
            append(scene_manager->selectionState().generation());
        }
        append(static_cast<std::uint64_t>(settingsForView(view).split_view_mode));
        return hash;
    }

    void RenderingManager::markCameraPoseChanged(ViewId view) { markViewDirty(view, DirtyFlag::CAMERA, lfs::vis::FrameReason::CameraMotion); }

    void RenderingManager::markCameraCut(ViewId view) {
        viewState(view).temporal_camera_cut_generation_.fetch_add(1, std::memory_order_release);
        markCameraPoseChanged(view);
    }

    DirtyMask RenderingManager::pendingDirtyMask() const {
        std::lock_guard lock(views_mutex_);
        DirtyMask mask = 0;
        for (const auto& [id, view] : view_states_)
            mask |= view->dirty_mask_.load(std::memory_order_relaxed);
        return mask;
    }

    bool RenderingManager::pollDirtyState() {
        std::lock_guard lock(views_mutex_);
        for (auto& [id, view] : view_states_)
            if (const auto flags = view->animation_state_.pollDirtyState())
                markViewDirty(id, flags, FrameReason::Overlay);
        for (auto& [id, view] : view_states_) {
            // The idle loop also polls here. Only request a scene frame once the
            // existing cooldown expires; the Fit preview can stay cached meanwhile.
            const auto& native_state = view->gt_comparison_actual_size_state_;
            if (!native_state.error.empty()) {
                const auto now = std::chrono::steady_clock::now();
                const bool tile_retry_due = native_state.tile_failure &&
                                            now - native_state.tile_failure->time >= GT_COMPARISON_IMAGE_RETRY_COOLDOWN;
                std::lock_guard lock(gt_comparison_image_mutex_);
                const bool source_retry_due = gt_comparison_full_source_slot_ &&
                                              gt_comparison_full_source_slot_->owner == id &&
                                              gt_comparison_full_source_slot_->status == GTComparisonImageStatus::Failed &&
                                              now - gt_comparison_full_source_slot_->failure_time >= GT_COMPARISON_IMAGE_RETRY_COOLDOWN;
                if (tile_retry_due || source_retry_due) {
                    markViewDirty(id, DirtyFlag::SPLIT_VIEW, FrameReason::AsyncCompletion);
                }
            }
        }
        if (lod_controller_ && lod_controller_->hasReadyResults())
            markDirty(DirtyFlag::CAMERA, lfs::vis::FrameReason::CameraMotion);
        for (const auto id : ledger_views_)
            if (viewState(id).dirty_mask_.load(std::memory_order_relaxed) != 0)
                return true;
        return false;
    }

    bool RenderingManager::releaseViewTargets(ViewRenderState& view) {
        bool ready = true;
        for (auto* target : {&view.main_render_target_, &view.split_left_render_target_, &view.split_right_render_target_}) {
            if (!target->valid())
                continue;
            const bool splat_ready = !scene_renderer_ || scene_renderer_->releaseRenderTarget(*target);
            const bool point_ready = !point_scene_renderer_ || point_scene_renderer_->releaseRenderTarget(*target);
            if (splat_ready && point_ready) {
                render_targets_.release(*target);
                *target = {};
            } else
                ready = false;
        }
        return ready;
    }

    void RenderingManager::dropViewStates() {
        std::lock_guard lock(views_mutex_);
        for (auto& [id, view] : view_states_) {
            invalidateGTComparisonActualSizeResources(*view);
            retired_view_states_.push_back(std::move(view));
        }
        view_states_.clear();
        depth_window_epochs_.clear();
        ++view_lifetime_epoch_;
    }

    void RenderingManager::retainVisibleViews(const std::vector<ViewId>& visible) {
        const auto epoch = view_source_.screenEpoch();
        std::vector<ViewId> known_views;
        {
            std::lock_guard lock(views_mutex_);
            known_views.reserve(view_states_.size());
            for (const auto& [id, view] : view_states_)
                known_views.push_back(id);
        }
        // Query the screen outside views_mutex_: ScreenService notifies
        // markViewDirty while holding its own lock.
        std::unordered_set<ViewId> removed_views;
        for (const auto id : known_views)
            if (!view_source_.viewSettings(id))
                removed_views.insert(id);

        std::lock_guard lock(views_mutex_);
        if (screen_epoch_ != epoch) {
            dropViewStates();
            screen_epoch_ = epoch;
        }
        ledger_views_ = visible;
        frame_demand_ledger_.setVisibleViews(visibleViewMask());
        const auto now = std::chrono::steady_clock::now();
        for (auto id : visible)
            viewState(id).last_visible = now;
        for (auto it = view_states_.begin(); it != view_states_.end();) {
            if (removed_views.contains(it->first) ||
                now - it->second->last_visible > std::chrono::milliseconds(300)) {
                depth_window_epochs_[it->first] = {it->second->depth_window_mode_epoch_, it->second->depth_window_projection_generation_};
                invalidateGTComparisonActualSizeResources(*it->second);
                retired_view_states_.push_back(std::move(it->second));
                it = view_states_.erase(it);
            } else
                ++it;
        }
        std::erase_if(retired_view_states_, [&](auto& view) {
            if (!releaseViewTargets(*view))
                return false;
            shutdownViewportReference(*view, last_graphics_context_);
            return true;
        });
    }

    void RenderingManager::requestViewFollowUp(ViewRenderState& view, const DirtyMask flags) {
        markViewDirty(view.id, flags, FrameReason::AsyncCompletion);

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
        markDirty(DirtyFlag::CAMERA, lfs::vis::FrameReason::CameraMotion);
        std::function<void()> wake;
        {
            std::lock_guard lock(wake_callback_mutex_);
            wake = wake_callback_;
        }
        if (wake)
            wake();
    }

    void RenderingManager::setViewportResizeActive(bool active, ViewportResizeRenderPolicy policy) {
        std::lock_guard lock(views_mutex_);
        for (auto& [id, view] : view_states_)
            markViewDirty(id, view->frame_lifecycle_service_.setViewportResizeActive(active, policy), FrameReason::ViewportResize);
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

        if (gpu_selection_eligible && scene_renderer_) {
            const auto gpu = scene_renderer_->gpuLodSelectionStatus(this->state().main_render_target_);
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
        dropViewStates();

        invalidatePointCloudData();
        ++point_cloud_preview_selection_revision_;

        if (scene_renderer_) {
            scene_renderer_->releaseSceneResources();
        }
        if (point_scene_renderer_) {
            point_scene_renderer_->reset();
        }
    }

    void RenderingManager::clearViewportImageState(ViewRenderState& view, const glm::ivec2 size,
                                                   const bool flip_y,
                                                   const glm::ivec2 alloc_size) {
        view.vulkan_viewport_image_.reset();
        view.viewport_depth_image_.reset();
        view.viewport_environment_ = {};
        view.viewport_meshes_ = {};
        clearViewportReferenceOutput(view);
        view.vulkan_viewport_image_size_ = size;
        view.vulkan_viewport_image_alloc_size_ = alloc_size.x > 0 && alloc_size.y > 0 ? alloc_size : size;
        view.vulkan_viewport_image_flip_y_ = flip_y;
        view.vulkan_gt_comparison_content_size_ = {0, 0};
        view.vulkan_gt_comparison_selection_view_.reset();
    }

    void RenderingManager::releaseSceneRenderResources() {
        invalidateGTComparisonImageCache(state());
        dropViewStates();
        invalidatePointCloudData();
        ++point_cloud_preview_selection_revision_;
        if (scene_renderer_)
            scene_renderer_->reset();
        if (point_scene_renderer_)
            point_scene_renderer_->reset();
        if (lfs::core::gpu_backend_available(lfs::core::GpuBackend::CUDA))
            lfs::core::Tensor::trim_memory_pool();
    }

    void RenderingManager::releaseIdleVksplatScratch(const bool training_active) {
        if (!scene_renderer_) {
            vksplat_idle_since_ = {};
            return;
        }
        // A parked refresh polls for its turn on the training arena; releasing
        // here would cancel the reservation it is waiting on.
        if (hasParkedArenaRetry()) {
            return;
        }

#if LFS_BUILD_TRAINER
        const bool under_pressure = lfs::core::raster_arena_under_memory_pressure();
#else
        constexpr bool under_pressure = false;
#endif

        if (!training_active) {
            vksplat_idle_since_ = {};
            if (under_pressure) {
                rendererTrainingInterop(*scene_renderer_).releaseScratchOnIdle(true);
            }
            return;
        }

        const auto now = std::chrono::steady_clock::now();
        if (vksplat_idle_since_ == std::chrono::steady_clock::time_point{})
            vksplat_idle_since_ = now;
        const bool release_private_scratch = now - vksplat_idle_since_ >= kVksplatIdleScratchReleaseDelay;
        if (under_pressure || release_private_scratch) {
            // During training the shared arena is owned by FastGS. Only release
            // private viewer allocations here; the terminal callback below is
            // the point at which the shared import may be relinquished.
            rendererTrainingInterop(*scene_renderer_).releaseScratchOnIdle(false, release_private_scratch);
            vksplat_idle_since_ = now;
        }
    }

    void RenderingManager::setCurrentCameraId(const int cam_id) {
        const bool changed = camera_interaction_service_.currentCameraId() != cam_id;
        camera_interaction_service_.setCurrentCameraId(cam_id);
        if (changed)
            invalidateCameraMetricsRequests(true);
        std::vector<ViewId> known_views;
        {
            std::lock_guard lock(views_mutex_);
            known_views.reserve(view_states_.size());
            for (const auto& [id, view] : view_states_)
                known_views.push_back(id);
        }
        std::vector<ViewId> gt_views;
        gt_views.reserve(known_views.size());
        for (const auto id : known_views)
            if (const auto settings = view_source_.viewSettings(id);
                settings && splitViewUsesGTComparison(settings->split_view_mode))
                gt_views.push_back(id);
        {
            std::lock_guard lock(views_mutex_);
            for (const auto id : gt_views) {
                const auto found = view_states_.find(id);
                if (found == view_states_.end())
                    continue;
                auto& view = *found->second;
                if (view.gt_comparison_camera_uid_ != cam_id) {
                    view.gt_comparison_camera_uid_ = cam_id;
                    invalidateGTComparisonActualSizeResources(view);
                    clearPublishedGTComparisonActualFrame(view);
                }
            }
        }
        markDirty(DirtyFlag::SPLIT_VIEW | DirtyFlag::PPISP, FrameReason::SettingsChange);
    }

    bool RenderingManager::isGTComparisonActualSizeRequested(ViewId id) const {
        if (id == kNoView)
            id = activeViewId();
        const auto settings = view_source_.viewSettings(id);
        return settings && splitViewUsesGTComparison(settings->split_view_mode) &&
               settings->gt_comparison_actual_size &&
               gtComparisonActualSizeEligible(*settings);
    }

    bool RenderingManager::isGTComparisonActualSizeAvailable(
        const SceneManager* const scene_manager, const ViewId view) const {
        if (!scene_manager) {
            return false;
        }
        const auto settings = trySettingsForView(
            view == kNoView ? activeViewId() : view);
        if (!settings || !gtComparisonActualSizeEligible(settings->view()))
            return false;
        const auto cameras = scene_manager->getScene().getAllCamerasCached();
        std::shared_ptr<lfs::core::Camera> camera;
        const int current_camera_id = camera_interaction_service_.currentCameraId();
        for (const auto& candidate : *cameras) {
            if (candidate && candidate->uid() == current_camera_id) {
                camera = candidate;
                break;
            }
        }
        if (!camera) {
            const auto first = std::find_if(cameras->begin(), cameras->end(), [](const auto& candidate) {
                return static_cast<bool>(candidate);
            });
            if (first != cameras->end()) {
                camera = *first;
            }
        }
        return camera && detail::isGTComparisonActualSizeAvailable(
                             *camera, settings->view());
    }

    void RenderingManager::setGTComparisonCropOrigin(const glm::ivec2 origin, const ViewId id) {
        auto& view = viewState(id == kNoView ? activeViewId() : id);
        if (!view.gt_comparison_published_actual_frame_) {
            return;
        }
        const auto& published = *view.gt_comparison_published_actual_frame_;
        if (view.gt_comparison_actual_size_state_.source_key != published.source_key ||
            view.gt_comparison_actual_size_state_.source_generation !=
                published.source_generation) {
            return;
        }
        const auto crop = detail::clampGTComparisonCrop(
            published.full_extent,
            published.framebuffer_extent,
            origin);
        if (!crop.valid()) {
            return;
        }
        // A pan anchors a new center in the displayed crop. Clamping/rounding
        // during a later resize does not overwrite this intent.
        view.gt_comparison_actual_size_state_.desired_crop_center =
            glm::dvec2(crop.origin) + glm::dvec2(crop.extent) * 0.5;
        view.gt_comparison_actual_size_state_.pending_pan_camera_uid.reset();
        view.gt_comparison_actual_size_state_.pending_pan_offset = {0, 0};
        const bool crop_changed =
            crop.origin != view.gt_comparison_actual_size_state_.crop.origin;
        if (!crop_changed && crop.origin == published.crop.origin) {
            return;
        }
        if (crop_changed) {
            view.gt_comparison_actual_size_state_.crop = crop;
            invalidateGTComparisonActualSizeTile(view);
        }
        markViewDirty(view.id, DirtyFlag::SPLIT_VIEW, FrameReason::CameraMotion);
    }

    void RenderingManager::setGTComparisonCropOffsetFromCenter(
        const glm::ivec2 offset, const ViewId id) {
        const ViewId target = id == kNoView ? activeViewId() : id;
        if (!isGTComparisonActualSizeRequested(target))
            return;
        auto& view = viewState(target);
        auto& state = view.gt_comparison_actual_size_state_;
        state.pending_pan_camera_uid = camera_interaction_service_.currentCameraId();
        state.pending_pan_offset = offset;
        if (state.full_extent.x > 0 && state.full_extent.y > 0 &&
            state.framebuffer_extent.x > 0 && state.framebuffer_extent.y > 0) {
            state.desired_crop_center =
                glm::dvec2(state.full_extent) * 0.5 + glm::dvec2(offset);
            const auto crop = detail::cropGTComparisonFromCenter(
                state.full_extent, state.framebuffer_extent,
                *state.desired_crop_center);
            if (crop.valid() && crop != state.crop) {
                state.crop = crop;
                invalidateGTComparisonActualSizeTile(view);
            }
        }
        markViewDirty(view.id, DirtyFlag::SPLIT_VIEW, FrameReason::CameraMotion);
    }

    void RenderingManager::retainVksplatScratch() {
        vksplat_idle_since_ = std::chrono::steady_clock::now();
    }

    double RenderingManager::secondsUntilVksplatScratchRelease() const {
        if (!scene_renderer_ || hasParkedArenaRetry())
            return std::numeric_limits<double>::infinity();
        if (vksplat_idle_since_ == std::chrono::steady_clock::time_point{})
            return std::chrono::duration<double>(kVksplatIdleScratchReleaseDelay).count();
        const auto due = vksplat_idle_since_ + kVksplatIdleScratchReleaseDelay;
        return std::max(0.0, std::chrono::duration<double>(due - std::chrono::steady_clock::now()).count());
    }

    void RenderingManager::updateSettings(const RenderSettings& new_settings) {
        updateSettings(new_settings, DirtyFlag::ALL);
    }

    void RenderingManager::updateSettings(const RenderSettings& new_settings,
                                          const DirtyMask dirty_flags,
                                          const SceneUpscalerPresetUpdate preset_update) {
        (void)updateSettingsForView(activeViewId(), new_settings, dirty_flags, preset_update);
    }

    bool RenderingManager::updateSettingsForView(const ViewId target_view, const RenderSettings& new_settings,
                                                 const DirtyMask dirty_flags,
                                                 const SceneUpscalerPresetUpdate preset_update) {
        if (!view_source_.viewSettings(target_view))
            return false;
        auto& target_state = viewState(target_view);
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
        sanitizeGTComparisonSettings(sanitized_settings);
        bool clear_metrics = false;
        bool lod_request_changed = false;
        bool lod_enabled_turned_on = false;
        bool actual_size_setting_changed = false;
        bool gt_comparison_deactivated = false;
        bool leaving_gt_rgb = false;
        // Equal-mode writes can re-enter from latch release and take only
        // settings_mutex_. A mode change releases it before acquiring the
        // transition mutex, then rechecks the mode under both locks.
        std::unique_lock<std::mutex> transition_lock;
        for (;;) {
            std::unique_lock<std::mutex> lock(settings_mutex_);
            const auto current_view = view_source_.viewSettings(target_view);
            if (!current_view)
                return false;
            auto settings = RenderSettings(settings_, *current_view);
            const bool split_mode_changes =
                settings.split_view_mode != sanitized_settings.split_view_mode;
            if (split_mode_changes && !transition_lock.owns_lock()) {
                lock.unlock();
                transition_lock = std::unique_lock<std::mutex>(target_state.depth_window_transition_mutex_);
                continue;
            }
            const SplitViewMode previous_split_mode = settings.split_view_mode;
            if (target_state.split_view_service_.isGTComparisonActive(settings) ||
                target_state.split_view_service_.isGTComparisonActive(sanitized_settings)) {
                sanitized_settings.show_camera_frustums = false;
            }

            actual_size_setting_changed =
                settings.gt_comparison_actual_size !=
                sanitized_settings.gt_comparison_actual_size;
            gt_comparison_deactivated =
                target_state.split_view_service_.isGTComparisonActive(settings) &&
                !target_state.split_view_service_.isGTComparisonActive(sanitized_settings);
            leaving_gt_rgb = settings.gt_comparison_mode == GTComparisonMode::RGB &&
                             sanitized_settings.gt_comparison_mode != GTComparisonMode::RGB;
            const float previous_depth_filter_scale_x = settings.depth_filter_scale_x;
            const float previous_depth_filter_scale_y = settings.depth_filter_scale_y;
            const float previous_depth_filter_offset_x =
                settings.depth_filter_offset_x;
            const float previous_depth_filter_offset_y =
                settings.depth_filter_offset_y;
            const float previous_depth_filter_min_z = settings.depth_filter_min.z;
            const float previous_depth_filter_max_z = settings.depth_filter_max.z;
            lod_enabled_turned_on =
                !settings.lod_enabled && sanitized_settings.lod_enabled;
            lod_request_changed =
                settings.lod_enabled != sanitized_settings.lod_enabled ||
                settings.lod_max_splats != sanitized_settings.lod_max_splats ||
                settings.lod_render_scale != sanitized_settings.lod_render_scale ||
                settings.lod_behind_camera_penalty != sanitized_settings.lod_behind_camera_penalty ||
                settings.lod_cone_foveation != sanitized_settings.lod_cone_foveation ||
                settings.lod_cone_inner_degrees != sanitized_settings.lod_cone_inner_degrees ||
                settings.lod_cone_outer_degrees != sanitized_settings.lod_cone_outer_degrees;

            if (sanitized_settings.camera_metrics_mode == RenderSettings::CameraMetricsMode::Off) {
                clear_metrics = true;
            } else if (camera_interaction_service_.currentCameraId() >= 0 &&
                       shouldRefreshCameraMetricsForSettings(settings, sanitized_settings)) {
                clear_metrics = true;
            }

            const float previous_depth_min_z = settings.depth_filter_min.z;
            const float previous_depth_max_z = settings.depth_filter_max.z;
            const auto previous_backend = settings.raster_backend;
            const bool previous_gut = settings.gut;
            settings = sanitized_settings;
            const bool gut_toggle_only = settings.raster_backend == previous_backend &&
                                         settings.gut != previous_gut;
            settings.raster_backend = gut_toggle_only
                                          ? lfs::rendering::viewerRasterBackendForGutMode(settings.gut)
                                          : lfs::rendering::normalizeViewerRasterBackend(
                                                settings.raster_backend, settings.gut);
            settings.gut = lfs::rendering::isGutBackend(settings.raster_backend);
            sanitizeDepthViewSettings(settings);
            sanitizeGTComparisonSettings(settings);
            sanitizeSelectionWindowSettings(settings);
            settings.grid_plane = clampGridPlane(settings.grid_plane);

            const bool depth_window_projection_changed =
                previous_depth_filter_scale_x != settings.depth_filter_scale_x ||
                previous_depth_filter_scale_y != settings.depth_filter_scale_y ||
                previous_depth_filter_offset_x != settings.depth_filter_offset_x ||
                previous_depth_filter_offset_y != settings.depth_filter_offset_y ||
                previous_depth_filter_min_z != settings.depth_filter_min.z ||
                previous_depth_filter_max_z != settings.depth_filter_max.z;

            if (depth_window_projection_changed) {
                target_state.depth_window_drag_owner_ = 0;
                target_state.depth_window_drag_backup_.reset();
            }
            if (settings.depth_filter_min.z != previous_depth_min_z ||
                settings.depth_filter_max.z != previous_depth_max_z) {
                ++target_state.depth_window_projection_generation_;
            }
            if (split_mode_changes)
                applyDepthWindowModeTransitionLocked(
                    previous_split_mode,
                    settings.split_view_mode, target_view);
            const bool scene_changed = settings_ != settings.scene();
            if (!view_source_.editViewSettings(target_view, [&](ViewSettings& view) { view = settings.view(); }))
                return false;
            settings_ = settings.scene();
            if (scene_changed)
                markDirty(dirty_flags, lfs::vis::FrameReason::SceneChange);
            else
                markViewDirty(target_view, dirty_flags, FrameReason::SettingsChange);
            break;
        }

        if (lod_request_changed && lod_controller_) {
            lod_controller_->invalidatePendingWork();
        }
        if (actual_size_setting_changed || gt_comparison_deactivated || leaving_gt_rgb) {
            invalidateGTComparisonActualSizeResources(target_state,
                                                      actual_size_setting_changed && !gt_comparison_deactivated && !leaving_gt_rgb);
            if (sanitized_settings.gt_comparison_actual_size) {
                target_state.gt_comparison_actual_size_state_.requested_at = std::chrono::steady_clock::now();
            }
        }
        if (lod_enabled_turned_on) {
            lod_controller_needs_sync_traversal_ = true;
        }

        auto& render_settings_generation = app_store().render_settings_generation;
        render_settings_generation.set(render_settings_generation.get() + 1);

        if (clear_metrics) {
            invalidateCameraMetricsRequests(true);
        }
        return true;
    }

    RenderSettings RenderingManager::activeSettingsLocked() const {
        return RenderSettings(settings_, view_source_.viewSettings(view_source_.activeView()).value());
    }

    void RenderingManager::storeActiveSettingsLocked(const RenderSettings& settings) {
        settings_ = settings.scene();
        view_source_.editViewSettings(view_source_.activeView(), [&](ViewSettings& view) { view = settings.view(); });
    }

    void RenderingManager::editViewSettings(ViewId view, const std::function<void(ViewSettings&)>& edit) {
        std::lock_guard lock(settings_mutex_);
        if (view_source_.editViewSettings(view, edit))
            markViewDirty(view, DirtyFlag::ALL, lfs::vis::FrameReason::SceneChange);
    }

    RenderSettings RenderingManager::settingsForView(const ViewId view) const {
        std::lock_guard lock(settings_mutex_);
        return RenderSettings(settings_, view_source_.viewSettings(view).value());
    }

    std::optional<RenderSettings> RenderingManager::trySettingsForView(const ViewId view) const {
        std::lock_guard lock(settings_mutex_);
        const auto settings = view_source_.viewSettings(view);
        return settings ? std::optional{RenderSettings(settings_, *settings)} : std::nullopt;
    }

    RenderSettings RenderingManager::getSettings() const {
        std::lock_guard lock(settings_mutex_);
        return RenderSettings(settings_, view_source_.viewSettings(view_source_.activeView()).value());
    }

    void RenderingManager::logSceneUpscalerRequest(ViewRenderState& view, const RenderSettings& settings) {
        if (view.scene_reconstruction_request_logged_ &&
            view.last_scene_reconstruction_backend_ == settings.scene_upscaler &&
            view.last_scene_reconstruction_preset_ == settings.scene_upscaler_preset)
            return;
        if (view.scene_reconstruction_request_logged_)
            LOG_INFO("Scene reconstruction request: {}/{} -> {}/{} (input_scale={:.4f})",
                     view.last_scene_reconstruction_backend_, view.last_scene_reconstruction_preset_,
                     settings.scene_upscaler, settings.scene_upscaler_preset, settings.scene_upscaler_scale);
        else
            LOG_INFO("Scene reconstruction initial request: {}/{} (input_scale={:.4f})",
                     settings.scene_upscaler, settings.scene_upscaler_preset, settings.scene_upscaler_scale);
        view.scene_reconstruction_request_logged_ = true;
        view.last_scene_reconstruction_backend_ = settings.scene_upscaler;
        view.last_scene_reconstruction_preset_ = settings.scene_upscaler_preset;
    }

    void RenderingManager::reportSceneUpscalerRuntimeSelection(
        const ViewId id, const SceneUpscalerSelection selection) {
        bool changed = false;
        {
            std::lock_guard lock(settings_mutex_);
            changed = viewState(id).scene_upscaler_runtime_selection_ != selection;
            viewState(id).scene_upscaler_runtime_selection_ = selection;
        }
        // The renderer chooses its source resolution before the presentation pass
        // proves whether reconstruction is available. A real active/fallback
        // transition therefore needs one feedback frame: active may adopt the
        // preset scale, while fallback must replace any cached reduced source with
        // a full-resolution native frame. TEMPORAL deliberately avoids restarting
        // the convergence sequence as CAMERA would.
        if (changed) {
            LOG_INFO("Scene reconstruction effective: {} -> {} (fallback={})",
                     sceneUpscalerBackendId(selection.requested), sceneUpscalerBackendId(selection.effective),
                     sceneUpscalerFallbackId(selection.fallback));
            auto& generation = app_store().scene_upscaler_generation;
            generation.set(generation.get() + 1);
            markViewDirty(id, DirtyFlag::TEMPORAL, lfs::vis::FrameReason::SceneChange);
        }
    }

    SceneUpscalerSelection RenderingManager::sceneUpscalerRuntimeSelection(const ViewId view) const {
        std::lock_guard lock(settings_mutex_);
        return viewState(view == kNoView ? view_source_.activeView() : view).scene_upscaler_runtime_selection_;
    }

    bool RenderingManager::sceneUpscalerModeUnsupported(const ViewId view) const {
        std::lock_guard lock(settings_mutex_);
        return viewState(view).scene_upscaler_mode_unsupported_;
    }

    void RenderingManager::setOrthographic(const bool enabled, const float viewport_height, const float distance_to_pivot) {
        auto settings = getSettings();
        if (enabled && !settings.orthographic) {
            const float vfov = lfs::rendering::focalLengthToVFov(settings.focal_length_mm);
            settings.ortho_scale = viewport_height > 0.0f && distance_to_pivot > 0.01f
                                       ? std::clamp(
                                             viewport_height / (2.0f * distance_to_pivot * std::tan(glm::radians(vfov) * 0.5f)), 1.0f, 10000.0f)
                                       : 100.0f;
        }
        settings.orthographic = enabled;
        updateSettings(settings, DirtyFlag::CAMERA);
    }

    void RenderingManager::beginImportRenderCheck(const uint64_t generation) {
        import_render_check_ = true;
        import_render_generation_ = generation;
        import_render_frames_ = 0;
        import_render_result_.reset();
        markDirty(DirtyFlag::ALL, FrameReason::SceneChange);
    }

    double RenderingManager::secondsUntilTrainingRefresh() const {
        std::lock_guard lock(views_mutex_);
        double remaining = std::numeric_limits<double>::infinity();
        for (const auto& [id, view] : view_states_)
            remaining = std::min(remaining, view->frame_lifecycle_service_.secondsUntilTrainingRefresh(
                                                trainingRefreshIntervalSec(*view)));
        return remaining;
    }

    void RenderingManager::invalidatePointCloudData() {
        point_cloud_colors_cache_ = {};
        point_cloud_colors_cache_key_ = nullptr;
        point_cloud_colors_cache_size_ = 0;
        ++point_cloud_data_revision_;
    }

    float RenderingManager::getFocalLengthMm() const {
        std::lock_guard<std::mutex> lock(settings_mutex_);
        return activeSettingsLocked().focal_length_mm;
    }

    void RenderingManager::setFocalLength(const float focal_mm) {
        auto settings = getSettings();
        const float clamped = std::clamp(focal_mm,
                                         lfs::rendering::MIN_FOCAL_LENGTH_MM,
                                         lfs::rendering::MAX_FOCAL_LENGTH_MM);
        if (settings.focal_length_mm == clamped)
            return;
        settings.focal_length_mm = clamped;
        updateSettings(settings, DirtyFlag::CAMERA);
    }

    void RenderingManager::advanceSplitOffset() {
        auto settings = getSettings();
        this->state().split_view_service_.advanceSplitOffset(settings);
        updateSettings(settings, DirtyFlag::SPLIT_VIEW);
    }

    SplitViewInfo RenderingManager::getSplitViewInfo() const {
        return this->state().split_view_service_.getInfo();
    }

    std::optional<SplitViewInfo> RenderingManager::getSplitViewInfoIfChanged(
        std::uint64_t& generation) const {
        return this->state().split_view_service_.getInfoIfChanged(generation);
    }

    bool RenderingManager::isGTComparisonActive() const {
        std::lock_guard<std::mutex> lock(settings_mutex_);
        return this->state().split_view_service_.isGTComparisonActive(activeSettingsLocked());
    }

    bool RenderingManager::hasGTComparisonAvailable() const {
        const auto* const scene_manager = services().sceneOrNull();
        if (!scene_manager || !scene_manager->hasDataset()) {
            return false;
        }

        const auto cameras = scene_manager->getScene().getAllCamerasCached();
        return cameras && std::any_of(cameras->begin(), cameras->end(), [](const auto& camera) {
                   return camera && camera->has_image() && !camera->image_path().empty();
               });
    }

    bool RenderingManager::isPLYComparisonActive() const {
        std::lock_guard<std::mutex> lock(settings_mutex_);
        return splitViewUsesPLYComparison(activeSettingsLocked().split_view_mode);
    }

    void RenderingManager::beginDepthWindowDrag(ViewId view, uint64_t& out_drag_token) {
        std::lock_guard lock(settings_mutex_);
        if (!viewState(view).depth_window_drag_owner_)
            viewState(view).depth_window_drag_backup_ = depthWindowFromProjection(RenderSettings(settings_, view_source_.viewSettings(view).value()));
        out_drag_token = viewState(view).depth_window_drag_owner_ = ++viewState(view).depth_window_last_drag_token_;
    }

    void RenderingManager::endDepthWindowDrag(ViewId view, const uint64_t drag_token) {
        if (!hasViewState(view))
            return;
        std::lock_guard lock(settings_mutex_);
        if (viewState(view).depth_window_drag_owner_ == drag_token) {
            viewState(view).depth_window_drag_owner_ = 0;
            viewState(view).depth_window_drag_backup_.reset();
        }
    }

    void RenderingManager::beginDepthWindowPreview(ViewId view) {
        std::lock_guard lock(settings_mutex_);
        ++viewState(view).depth_window_preview_count_;
        markViewDirty(viewState(view).id, DirtyFlag::OVERLAY, lfs::vis::FrameReason::Overlay);
    }

    void RenderingManager::endDepthWindowPreview(ViewId view) {
        if (!hasViewState(view))
            return;
        std::lock_guard lock(settings_mutex_);
        viewState(view).depth_window_preview_count_ = std::max(0, viewState(view).depth_window_preview_count_ - 1);
        markViewDirty(viewState(view).id, DirtyFlag::OVERLAY, lfs::vis::FrameReason::Overlay);
    }

    bool RenderingManager::depthWindowDragPreview(const ViewId view) const {
        std::lock_guard lock(settings_mutex_);
        return viewState(view == kNoView ? view_source_.activeView() : view).depth_window_preview_count_ > 0;
    }

    GTComparisonMode RenderingManager::getGTComparisonMode() const {
        std::lock_guard lock(settings_mutex_);
        return activeSettingsLocked().gt_comparison_mode;
    }

    SplitViewMode RenderingManager::getSplitViewMode() const {
        std::lock_guard lock(settings_mutex_);
        return activeSettingsLocked().split_view_mode;
    }

    float RenderingManager::getSplitPosition() const {
        std::lock_guard lock(settings_mutex_);
        return activeSettingsLocked().split_position;
    }

    DepthWindowState RenderingManager::getDepthWindow() const {
        std::lock_guard lock(settings_mutex_);
        return depthWindowFromProjection(activeSettingsLocked());
    }

    void RenderingManager::setDepthWindow(const DepthWindowState& state) {
        auto clamped = state;
        clampDepthWindowState(clamped);
        std::lock_guard lock(settings_mutex_);
        applyDepthWindowProjectionLocked(this->state().id, clamped);
        this->state().depth_window_drag_owner_ = 0;
        this->state().depth_window_drag_backup_.reset();
        markViewDirty(this->state().id, DirtyFlag::ALL, lfs::vis::FrameReason::SceneChange);
    }

    bool RenderingManager::applyDepthWindowIfEpoch(ViewId view, const DepthWindowState& state,
                                                   const uint64_t expected_epoch,
                                                   const uint64_t drag_token) {
        auto clamped = state;
        clampDepthWindowState(clamped);
        std::lock_guard lock(settings_mutex_);
        if (viewState(view).depth_window_mode_epoch_ != expected_epoch || !drag_token ||
            viewState(view).depth_window_drag_owner_ != drag_token)
            return false;
        applyDepthWindowProjectionLocked(view, clamped);
        markViewDirty(viewState(view).id, DirtyFlag::ALL, lfs::vis::FrameReason::SceneChange);
        return true;
    }

    bool RenderingManager::restorePinnedDepthWindow(ViewId view, const DepthWindowState& state,
                                                    const uint64_t expected_epoch,
                                                    const uint64_t drag_token) {
        return applyDepthWindowIfEpoch(view, state, expected_epoch, drag_token);
    }

    bool RenderingManager::commitDepthWindowIfEpoch(ViewId view,
                                                    const DepthWindowState& state, const uint64_t expected_epoch,
                                                    const uint64_t drag_token, op::DepthWindowModeSnapshot& out_snapshot) {
        auto clamped = state;
        clampDepthWindowState(clamped);
        std::lock_guard lock(settings_mutex_);
        if (viewState(view).depth_window_mode_epoch_ != expected_epoch || !drag_token ||
            viewState(view).depth_window_drag_owner_ != drag_token)
            return false;
        applyDepthWindowProjectionLocked(view, clamped);
        viewState(view).depth_window_drag_owner_ = 0;
        viewState(view).depth_window_drag_backup_.reset();
        out_snapshot = depthWindowSnapshotLocked(view);
        markViewDirty(viewState(view).id, DirtyFlag::ALL, lfs::vis::FrameReason::SceneChange);
        return true;
    }

    op::DepthWindowModeSnapshot
    RenderingManager::depthWindowSnapshotLocked(ViewId view) const {
        return {.view = viewState(view).id, .screen_epoch = view_source_.screenEpoch(), .lifetime_epoch = view_lifetime_epoch_, .window = depthWindowFromProjection(RenderSettings(settings_, view_source_.viewSettings(view).value())), .mode_epoch = viewState(view).depth_window_mode_epoch_};
    }

    op::DepthWindowModeSnapshot RenderingManager::depthWindowSnapshot(ViewId view) const {
        std::lock_guard lock(settings_mutex_);
        return depthWindowSnapshotLocked(view);
    }

    op::DepthWindowModeSnapshot
    RenderingManager::depthWindowBaselineSnapshotForDrag(ViewId view,
                                                         const uint64_t drag_token) const {
        std::lock_guard lock(settings_mutex_);
        auto snapshot = depthWindowSnapshotLocked(view);
        if (drag_token && viewState(view).depth_window_drag_owner_ == drag_token &&
            viewState(view).depth_window_drag_backup_)
            snapshot.window = *viewState(view).depth_window_drag_backup_;
        return snapshot;
    }

    void RenderingManager::restoreDepthWindowStateFromProject() {
        const auto transition_lock = acquireDepthWindowTransitionLock(this->state().id);
        std::lock_guard lock(settings_mutex_);
        this->state().depth_window_drag_owner_ = 0;
        this->state().depth_window_drag_backup_.reset();
        ++this->state().depth_window_projection_generation_;
        ++this->state().depth_window_mode_epoch_;
    }

    bool RenderingManager::depthWindowSnapshotCurrent(const op::DepthWindowModeSnapshot& snapshot) const {
        const auto screen_epoch = view_source_.screenEpoch();
        const bool view_exists = view_source_.viewSettings(snapshot.view).has_value();
        std::lock_guard lock(views_mutex_);
        if (snapshot.screen_epoch != screen_epoch || snapshot.lifetime_epoch != view_lifetime_epoch_ || !view_exists)
            return false;
        if (const auto view = view_states_.find(snapshot.view); view != view_states_.end())
            return snapshot.mode_epoch == view->second->depth_window_mode_epoch_;
        const auto saved = depth_window_epochs_.find(snapshot.view);
        return saved != depth_window_epochs_.end() && snapshot.mode_epoch == saved->second.first;
    }

    bool RenderingManager::restoreDepthWindowSnapshotIfEpoch(
        const op::DepthWindowModeSnapshot& snapshot, const uint64_t expected_epoch) {
        if (!depthWindowSnapshotCurrent(snapshot))
            return false;
        auto& view = viewState(snapshot.view);
        std::lock_guard transition(view.depth_window_transition_mutex_);
        std::lock_guard lock(settings_mutex_);
        if (view.depth_window_mode_epoch_ != expected_epoch)
            return false;
        view_source_.editViewSettings(snapshot.view, [&](ViewSettings& settings) {
            RenderSettings composed(settings_, settings);
            applyDepthWindowToProjection(composed, snapshot.window);
            settings = composed.view();
        });
        ++view.depth_window_projection_generation_;
        view.depth_window_drag_owner_ = 0;
        view.depth_window_drag_backup_.reset();
        markViewDirty(snapshot.view, DirtyFlag::ALL, lfs::vis::FrameReason::SceneChange);
        return true;
    }

    uint64_t RenderingManager::depthWindowProjectionGeneration() const {
        std::lock_guard lock(settings_mutex_);
        return this->state().depth_window_projection_generation_;
    }

    void RenderingManager::applyDepthWindowProjectionLocked(ViewId view, const DepthWindowState& state) {
        auto settings = RenderSettings(settings_, view_source_.viewSettings(view).value());
        const auto previous = depthWindowFromProjection(settings);
        applyDepthWindowToProjection(settings, state);
        view_source_.editViewSettings(view, [&](ViewSettings& target) { target = settings.view(); });
        if (previous.near_plane != state.near_plane || previous.far_plane != state.far_plane)
            ++viewState(view).depth_window_projection_generation_;
    }

    void RenderingManager::applyDepthWindowModeTransitionLocked(
        const SplitViewMode previous_mode, const SplitViewMode new_mode, const ViewId target_view) {
        auto& target_state = viewState(target_view == kNoView ? activeViewId() : target_view);
        if (splitViewUsesGTComparison(previous_mode) ==
            splitViewUsesGTComparison(new_mode))
            return;
        // GT suspends selection filtering; an old drag must not overwrite its
        // successor.
        if (target_state.depth_window_drag_owner_ && target_state.depth_window_drag_backup_)
            applyDepthWindowProjectionLocked(target_state.id, *target_state.depth_window_drag_backup_);
        target_state.depth_window_drag_owner_ = 0;
        target_state.depth_window_drag_backup_.reset();
        ++target_state.depth_window_mode_epoch_;
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

    void RenderingManager::queueCameraMetricsRefreshIfStale(ViewId view, SceneManager* const scene_manager) {
        if (!scene_manager) {
            return;
        }

        auto* const trainer_mgr = scene_manager->getTrainerManager();
        if (!trainer_mgr || !trainer_mgr->getTrainer()) {
            return;
        }

        const auto settings = settingsForView(view);
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
                   lhs.settings.ppisp_overrides == rhs.settings.ppisp_overrides;
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
            markDirty(DirtyFlag::OVERLAY, lfs::vis::FrameReason::Overlay);
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
                                   entry.request.settings.ppisp_overrides ==
                                       request.settings.ppisp_overrides;
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
                markDirty(DirtyFlag::OVERLAY, lfs::vis::FrameReason::Overlay);
            }
        }
    }

    std::optional<float> RenderingManager::getSplitDividerScreenX(ViewId view, const glm::vec2& viewport_pos,
                                                                  const glm::vec2& viewport_size) const {
        const auto settings = settingsForView(view);
        if (!viewState(view).split_view_service_.isActive(settings) ||
            (splitViewUsesGTComparison(settings.split_view_mode) &&
             gtComparisonShowsLoss(settings.gt_comparison_mode))) {
            return std::nullopt;
        }

        const auto content_bounds = getContentBounds(viewState(view).id, glm::ivec2(
                                                                             std::max(static_cast<int>(viewport_size.x), 0),
                                                                             std::max(static_cast<int>(viewport_size.y), 0)));
        const int content_width = std::max(static_cast<int>(std::lround(content_bounds.width)), 0);
        if (content_width <= 0) {
            return std::nullopt;
        }

        return viewport_pos.x + content_bounds.x +
               static_cast<float>(splitViewDividerPixel(content_width, settings.split_position));
    }

    void RenderingManager::applySplitModeChange(const SplitViewService::ModeChangeResult& result) {
        if (!result.mode_changed) {
            return;
        }

        // Any GT enter/exit invalidates the published GT selection camera, whether or not the
        // viewport output is cleared: clear_viewport_output is only set for
        // enabled -> disabled (split_view_service.cpp:202), so re-entering GT would otherwise
        // expose the previous session's camera until the next GT frame is presented.
        this->state().vulkan_gt_comparison_selection_view_.reset();
        this->state().vulkan_gt_comparison_content_size_ = {0, 0};

        if (result.clear_viewport_output) {
            this->state().viewport_artifact_service_.clearViewportOutput();
            clearPublishedGTComparisonActualFrame(state());
        }
        if (splitViewUsesGTComparison(result.previous_mode) &&
            !splitViewUsesGTComparison(result.current_mode)) {
            invalidateGTComparisonActualSizeResources(state());
        }

        if (result.restore_equirectangular) {
            auto event = lfs::core::events::ui::RenderSettingsChanged{};
            event.equirectangular = *result.restore_equirectangular;
            event.emit();
        }
        if (result.render_settings_changed) {
            markDirty(DirtyFlag::OVERLAY, lfs::vis::FrameReason::Overlay);
            auto& render_settings_generation = app_store().render_settings_generation;
            render_settings_generation.set(render_settings_generation.get() + 1);
        }
    }

    void RenderingManager::setCursorPreviewState(const bool active, const float x, const float y, const float radius,
                                                 const bool add_mode, lfs::core::Tensor* selection_tensor,
                                                 const bool saturation_mode, const float saturation_amount,
                                                 const std::optional<SplitViewPanelId> panel,
                                                 const int focused_gaussian_id, const bool highlight_splats) {
        this->state().viewport_overlay_service_.setCursorPreview(active, x, y, radius, add_mode, selection_tensor,
                                                                 saturation_mode, saturation_amount, panel, focused_gaussian_id, highlight_splats);
        if (highlight_splats)
            markViewDirty(this->state().id, DirtyFlag::SELECTION, lfs::vis::FrameReason::Selection);
    }

    void RenderingManager::clearCursorPreviewState() {
        this->state().viewport_overlay_service_.clearCursorPreview();
        markViewDirty(this->state().id, DirtyFlag::SELECTION, lfs::vis::FrameReason::Selection);
    }

    void RenderingManager::setRectPreview(float x0, float y0, float x1, float y1, bool add_mode,
                                          const std::optional<SplitViewPanelId> panel,
                                          const bool track_cursor) {
        this->state().viewport_overlay_service_.setRect(x0, y0, x1, y1, add_mode, panel, track_cursor);
    }

    void RenderingManager::clearRectPreview() {
        this->state().viewport_overlay_service_.clearRect();
    }

    void RenderingManager::setPolygonPreview(const std::vector<std::pair<float, float>>& points, bool closed,
                                             bool add_mode, const std::optional<SplitViewPanelId> panel) {
        this->state().viewport_overlay_service_.setPolygon(points, closed, add_mode, panel);
    }

    void RenderingManager::setPolygonPreviewWorldSpace(const std::vector<glm::vec3>& world_points,
                                                       const bool closed, const bool add_mode,
                                                       const std::optional<SplitViewPanelId> panel) {
        this->state().viewport_overlay_service_.setPolygonWorldSpace(world_points, closed, add_mode, panel);
    }

    void RenderingManager::clearPolygonPreview() {
        this->state().viewport_overlay_service_.clearPolygon();
    }

    void RenderingManager::setLassoPreview(const std::vector<std::pair<float, float>>& points, bool add_mode,
                                           const std::optional<SplitViewPanelId> panel,
                                           const bool track_cursor) {
        this->state().viewport_overlay_service_.setLasso(points, add_mode, panel, track_cursor);
    }

    void RenderingManager::clearLassoPreview() {
        this->state().viewport_overlay_service_.clearLasso();
    }

    void RenderingManager::clearSelectionPreviews() {
        auto& overlay = this->state().viewport_overlay_service_;
        const bool had_preview = overlay.isCursorPreviewActive() || overlay.isRectPreviewActive() ||
                                 overlay.isPolygonPreviewActive() || overlay.isLassoPreviewActive() || overlay.cursorPreview().preview_selection;
        overlay.clearSelectionPreviews();
        if (had_preview)
            markViewDirty(this->state().id, DirtyFlag::SELECTION, lfs::vis::FrameReason::Selection);
    }

} // namespace lfs::vis

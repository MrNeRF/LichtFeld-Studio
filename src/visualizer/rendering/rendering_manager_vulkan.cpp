/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "core/camera.hpp"
#include "core/cuda/undistort/undistort.hpp"
#include "core/image_io.hpp"
#include "core/logger.hpp"
#include "core/memory_pressure.hpp"
#include "core/raster_arena.hpp"
#include "core/splat_data.hpp"
#include "core/tensor.hpp"
#include "core/tensor_backend.hpp"
#include "core/tensor_image.hpp"
#include "core/tensor_readback.hpp"
#include "diagnostics/vram_profiler.hpp"
#include "display_tensors.hpp"
#include "gt_comparison_cache_utils.hpp"
#include "model_renderability.hpp"
#include "output_image_pool.hpp"
#include "output_slot_ring.hpp"
#include "rendering/image_layout.hpp"
#include "rendering/passes/vulkan_scene_plugin_pipeline.hpp"
#include "rendering_manager.hpp"
#include "rendering_manager_split_view.hpp"
#include "scene/scene_manager.hpp"
#include "scene_renderer.hpp"
#include "scene_renderer_factory.hpp"
#include "scene_temporal_frame_setup.hpp"
#include "scene_training_interop.hpp"
#include "scene_upscaler_plugin.hpp"
#include "scene_upscaler_registry.hpp"
#include "split_capture_compositor.hpp"
#include "vksplat_shared_scratch_install.hpp"
#include "vulkan_scene_output.hpp"
#if LFS_BUILD_TRAINER
#include "training/trainer.hpp"
#endif
#include "core/training_manager.hpp"
#include "graphics_external_tensor.hpp"
#include "viewport_appearance_correction.hpp"
#include "viewport_interop_service.hpp"
#include "viewport_region_utils.hpp"
#include "viewport_request_builder.hpp"
#include "visualizer/scene_coordinate_utils.hpp"
#include "vulkan_view_render_state.hpp"
#include "vulkan_viewport_frame_resources.hpp"
#include "window/vulkan_graphics_context.hpp"
#include <algorithm>
#include <cassert>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <expected>
#include <filesystem>
#include <format>
#include <limits>
#include <optional>
#include <shared_mutex>
#include <stdexcept>
#include <string>
#include <string_view>
#include <tuple>
#include <utility>
#include <vector>

namespace lfs::vis {

    namespace {
        constexpr float kInteractiveResizeRenderScale = 0.33f;
        constexpr auto kTrainingOutputResizeStableDelay = std::chrono::milliseconds(500);

        void setVulkanMeshFrame(ViewRenderState& view, VulkanMeshFrame frame) {
            auto& native = vulkanViewRenderState(view);
            std::lock_guard lock(native.mesh_frame_mutex);
            native.mesh_frame = std::move(frame);
        }

        void clearVulkanMeshFrame(ViewRenderState& view) {
            auto& native = vulkanViewRenderState(view);
            std::lock_guard lock(native.mesh_frame_mutex);
            native.mesh_frame = {};
        }

        std::shared_ptr<const lfs::core::Tensor> publishVulkanFrameResources(
            ViewRenderState& state, std::shared_ptr<const lfs::core::Tensor> tensor,
            const VkImage image = VK_NULL_HANDLE,
            const VkImageView view = VK_NULL_HANDLE,
            const VkImageLayout layout = VK_IMAGE_LAYOUT_UNDEFINED,
            const std::uint64_t generation = 0,
            const VkSemaphore completion = VK_NULL_HANDLE,
            const std::uint64_t completion_value = 0,
            std::vector<ViewportInteropService::FrameCompletion> additional = {}) {
            auto resources = std::make_shared<VulkanViewportFrameResources>();
            resources->external_image = image;
            resources->external_image_view = view;
            resources->external_image_layout = layout;
            resources->external_image_generation = generation;
            resources->completion_semaphore = completion;
            resources->completion_value = completion_value;
            resources->additional_completions = std::move(additional);
            vulkanViewRenderState(state).pending_frame_resources = std::move(resources);
            return tensor;
        }

        [[nodiscard]] std::optional<LfsSceneUpscalerOptimalSettingsV1> pluginOptimalSettings(
            SceneUpscalerPlugin& plugin, const glm::ivec2 output_extent, const std::uint32_t quality) {
            if (output_extent.x <= 0 || output_extent.y <= 0)
                return std::nullopt;
            const auto optimal = plugin.optimalSettings(
                static_cast<std::uint32_t>(output_extent.x),
                static_cast<std::uint32_t>(output_extent.y),
                quality);
            if (!optimal || optimal->render_width == 0 || optimal->render_height == 0 ||
                optimal->render_width > static_cast<std::uint32_t>(output_extent.x) ||
                optimal->render_height > static_cast<std::uint32_t>(output_extent.y)) {
                return std::nullopt;
            }
            return optimal;
        }

        struct LiveModelLockBundle {
            std::shared_lock<std::shared_mutex> lock;
            const lfs::core::Scene* scene = nullptr;
            LiveModelLockBundle() = default;
            LiveModelLockBundle(std::shared_lock<std::shared_mutex>&& l, const lfs::core::Scene* s)
                : lock(std::move(l)),
                  scene(s) {
                if (scene && lock.owns_lock()) {
                    scene->noteLiveModelLockAcquired();
                }
            }
            LiveModelLockBundle(LiveModelLockBundle&& other) noexcept
                : lock(std::move(other.lock)),
                  scene(other.scene) {
                other.scene = nullptr;
            }
            LiveModelLockBundle& operator=(LiveModelLockBundle&& other) noexcept {
                if (this != &other) {
                    if (scene && lock.owns_lock()) {
                        scene->noteLiveModelLockReleased();
                    }
                    lock = std::move(other.lock);
                    scene = other.scene;
                    other.scene = nullptr;
                }
                return *this;
            }
            ~LiveModelLockBundle() {
                if (scene && lock.owns_lock()) {
                    scene->noteLiveModelLockReleased();
                }
            }
            LiveModelLockBundle(const LiveModelLockBundle&) = delete;
            LiveModelLockBundle& operator=(const LiveModelLockBundle&) = delete;
            [[nodiscard]] bool owns_lock() const { return lock.owns_lock(); }
        };

        [[nodiscard]] std::optional<LiveModelLockBundle> acquireLiveModelRenderLock(
            const SceneManager* const scene_manager,
            const bool try_lock = false) {
#if LFS_BUILD_TRAINER
            if (const auto* tm = scene_manager ? scene_manager->getTrainerManager() : nullptr) {
                if (const auto* trainer = tm->getTrainer()) {
                    const lfs::core::Scene* scene = trainer->getScene();
                    if (try_lock) {
                        std::shared_lock<std::shared_mutex> candidate(
                            trainer->getRenderMutex(), std::try_to_lock);
                        if (!candidate.owns_lock()) {
                            return std::nullopt;
                        }
                        return LiveModelLockBundle(std::move(candidate), scene);
                    }
                    return LiveModelLockBundle(
                        std::shared_lock<std::shared_mutex>(trainer->getRenderMutex()), scene);
                }
            }
#endif
            return std::nullopt;
        }

        [[nodiscard]] bool isRetryableSharedScratchUnavailable(const std::string_view error) {
            return error.find("shared scratch") != std::string_view::npos &&
                   (error.find("busy") != std::string_view::npos ||
                    error.find("capacity insufficient") != std::string_view::npos);
        }

        [[nodiscard]] DirtyMask vksplatOutputResizeRetryDirty(const DirtyMask frame_dirty) {
            return frame_dirty != 0 ? frame_dirty : DirtyFlag::VIEWPORT | DirtyFlag::CAMERA;
        }

        [[nodiscard]] DirtyMask vksplatSharedScratchRetryDirty(const DirtyMask frame_dirty) {
            return frame_dirty != 0 ? frame_dirty : DirtyFlag::SPLATS;
        }

        [[nodiscard]] std::shared_ptr<lfs::core::Tensor> makeDepthDisplayTensor(
            const lfs::core::Tensor& depth,
            const RenderSettings& settings) {
            return lfs::vis::makeDepthDisplayTensor(
                depth, settings.depth_visualization_mode, settings.background_color);
        }

        [[nodiscard]] SplitViewCpuPanelDesc cpuPanelDesc(
            const VulkanSplitViewPanel& panel) {
            return {
                .image = panel.image,
                .start_position = panel.start_position,
                .end_position = panel.end_position,
                .normalize_x_to_panel = panel.normalize_x_to_panel,
                .flip_y = panel.flip_y,
                .uv_scale = panel.uv_scale,
                .uv_clamp_max = panel.uv_clamp_max,
                .spatial_filter = panel.spatial_filter,
            };
        }

        [[nodiscard]] SplitViewCpuDesc cpuSplitViewDesc(
            const VulkanSplitViewParams& params) {
            return {
                .loss_visualization = params.loss_visualization,
                .exact_texel_sampling = params.exact_texel_sampling,
                .left = cpuPanelDesc(params.left),
                .right = cpuPanelDesc(params.right),
                .split_position = params.split_position,
                .content_rect = params.content_rect,
                .coordinate_extent = params.coordinate_extent,
                .background = params.background,
            };
        }

        [[nodiscard]] std::shared_ptr<lfs::core::Tensor> makeViewportCaptureImageHwc(
            lfs::core::Tensor image) {
            if (!image.is_valid() || image.ndim() != 3) {
                return std::make_shared<lfs::core::Tensor>(std::move(image));
            }

            const auto layout = lfs::rendering::detectImageLayout(image);
            if (layout == lfs::rendering::ImageLayout::CHW) {
                image = image.permute({1, 2, 0});
            }
            return std::make_shared<lfs::core::Tensor>(image.cpu().contiguous());
        }

        // Per-frame mesh-pass payload (without depth blit). The Vulkan splat path
        // publishes this every frame so the viewport pass sees the current camera;
        // otherwise the mesh stays anchored to whichever frame last set it.
        [[nodiscard]] VulkanMeshFrame populateMeshFrame(
            const FrameContext& frame_ctx,
            const RenderSettings& settings,
            const VulkanSplitViewParams& split_view_params) {
            auto meshes = buildViewportMeshes(frame_ctx, settings);
            VulkanMeshFrame frame;
            frame.view_projection = meshes.view_projection;
            frame.camera_position = meshes.camera_position;
            frame.items = std::move(meshes.items);
            frame.environment = buildViewportEnvironment(frame_ctx, settings, environmentBackgroundEnabled(settings));
            frame.split_view = split_view_params;

            return frame;
        }
    } // namespace

    std::shared_ptr<lfs::core::Tensor> RenderingManager::composeSplitViewCpu(
        const SplitViewCpuDesc& params, const glm::ivec2& output_size) {
        return composeSplitViewCpuImage(params, output_size);
    }

    std::expected<lfs::core::Tensor, std::string> RenderingManager::buildVksplatSelectionMask(
        SceneManager& scene_manager,
        const lfs::rendering::FrameView& frame_view,
        const bool equirectangular,
        const VksplatSelectionMaskShape shape,
        const std::vector<glm::vec4>& primitives,
        const std::vector<glm::vec2>& polygon_vertices,
        std::uint32_t* const picked_ring_id_out) {
        LOG_TIMER("RenderingManager::buildVksplatSelectionMask");
        const auto settings = getSettings();
        if (!lfs::rendering::isVkSplatBackend(settings.raster_backend)) {
            return std::unexpected("Scene selection query is available only when a Scene backend is active");
        }
        if (!last_graphics_context_) {
            return std::unexpected("Scene selection query requires an active Vulkan context");
        }
        // Point-cloud mode renders with a separate graphics pipeline, but selection
        // still needs the same projected-center mask. Let it use the GPU query
        // instead of falling back to SelectionService's CPU screen-position pass.
        const bool polygon_mode = (shape == VksplatSelectionMaskShape::Polygon);
        if (polygon_mode) {
            if (polygon_vertices.size() < 3) {
                return std::unexpected("Scene polygon selection requires at least 3 vertices");
            }
        } else if (primitives.empty()) {
            return std::unexpected("Scene selection query requires at least one primitive");
        }

        auto render_lock = acquireLiveModelRenderLock(&scene_manager);
        auto scene_state = scene_manager.buildRenderState();
        const lfs::core::SplatData* const model = scene_state.combined_model;
        if (!hasRenderableGaussians(model)) {
            return std::unexpected("Scene selection query found no renderable Gaussian model");
        }

        if (!scene_renderer_) {
            scene_renderer_ = createSceneRenderer(*last_graphics_context_);
        }

        const auto map_shape = [](const VksplatSelectionMaskShape value) {
            switch (value) {
            case VksplatSelectionMaskShape::Brush:
                return SceneRenderer::SelectionMaskShape::Brush;
            case VksplatSelectionMaskShape::Rectangle:
                return SceneRenderer::SelectionMaskShape::Rectangle;
            case VksplatSelectionMaskShape::Polygon:
                return SceneRenderer::SelectionMaskShape::Polygon;
            case VksplatSelectionMaskShape::Ring:
                return SceneRenderer::SelectionMaskShape::Ring;
            }
            return SceneRenderer::SelectionMaskShape::Brush;
        };

        SceneRenderer::SelectionMaskRequest request{
            .frame_view = frame_view,
            .scene =
                {.model_transforms = &scene_state.model_transforms,
                 .transform_indices = scene_state.transform_indices,
                 .node_visibility_mask = scene_state.node_visibility_mask},
            .shape = map_shape(shape),
            .primitives = primitives,
            .polygon_vertices = polygon_vertices,
            .gut = lfs::rendering::isGutBackend(settings.raster_backend),
            .equirectangular = equirectangular,
            .mip_filter = settings.mip_filter,
            .ring_width = settings.ring_width,
            .picked_ring_id_out = picked_ring_id_out,
        };

        const bool force_input_upload =
            (this->state().dirty_mask_.load(std::memory_order_relaxed) & DirtyFlag::SPLATS) != 0;
        return scene_renderer_->buildSelectionMask(*model, request, force_input_upload);
    }

    // Camera and edit changes retry on the next frame. A passive training refresh
    // parks instead, so the idle preview draws only once its render can proceed.
    void RenderingManager::queueSharedScratchRetry(ViewRenderState& view, const DirtyMask retry_dirty) {
        if ((retry_dirty & ~DirtyFlag::SPLATS) == 0) {
            // A parked preview is an active waiter polled every 4 ms. Preserve
            // its reservation; only camera retries drop the standing lease.
            if (retry_dirty != 0 && scene_renderer_)
                rendererTrainingInterop(*scene_renderer_).requestArenaHandoff();
            view.parked_arena_retry_ |= retry_dirty;
            return;
        }
        view.dirty_mask_.fetch_or(retry_dirty, std::memory_order_relaxed);
    }

    // The configured preview rate is an upper bound; the measured viewer turn
    // stretches the interval so training keeps kIdlePreviewTrainingShare.
    float RenderingManager::trainingRefreshIntervalSec(const ViewRenderState& view) const {
        const auto turns = lfs::core::raster_arena_turns();
        const double viewer_turn_ms = turns.viewer_turn_ms + turns.viewer_record_ms;
        // Recording holds the arena too; include it in the idle training budget.
        // The budget helper gives training's rest time. The refresh period also
        // includes the viewer turn itself.
        return static_cast<float>(std::max<double>(
            view.framerate_controller_.getSettings().training_frame_refresh_time_sec,
            idlePreviewIntervalSec(viewer_turn_ms) + viewer_turn_ms * 1e-3));
    }

    void RenderingManager::pollTrainingRefresh(const bool is_training, const int current_iteration) {
        std::lock_guard lock(views_mutex_);
        for (auto& [id, view] : view_states_) {
            const DirtyMask dirty = view->frame_lifecycle_service_.handleTrainingRefresh(
                is_training, trainingRefreshIntervalSec(*view));
            if (!is_training)
                view->has_training_preview_iteration_ = false;
            if (!dirty)
                continue;
            if (is_training && view->has_training_preview_iteration_ &&
                current_iteration <= view->last_training_preview_iteration_) {
                frame_demand_ledger_.countSkippedPreview();
                continue;
            }
            if (is_training) {
                view->last_training_preview_iteration_ = current_iteration;
                view->has_training_preview_iteration_ = true;
            }
            view->training_refresh_dirty_.fetch_or(dirty, std::memory_order_relaxed);
            markViewDirty(id, dirty, FrameReason::TrainingPreview);
        }
    }

    double RenderingManager::secondsUntilCameraSettle() const {
        std::lock_guard lock(views_mutex_);
        const auto now = std::chrono::steady_clock::now();
        double remaining = std::numeric_limits<double>::infinity();
        for (const auto& [id, view] : view_states_) {
            if (view->camera_settle_pending_)
                remaining = std::min(remaining, std::max(0.0, std::chrono::duration<double>(
                                                                  view->camera_settle_deadline_ - now)
                                                                  .count()));
        }
        return remaining;
    }

    void RenderingManager::pollParkedArenaRetry() {
        if (scene_renderer_ && scene_renderer_->takeRefinementRequest())
            markDirty(DirtyFlag::CAMERA, FrameReason::AsyncCompletion, "renderer_refinement_ready");
        if (point_scene_renderer_ && point_scene_renderer_->takeRefinementRequest())
            markDirty(DirtyFlag::CAMERA, FrameReason::AsyncCompletion, "renderer_refinement_ready");
        if (scene_renderer_ && !rendererTrainingInterop(*scene_renderer_).pollArenaHandoff())
            return;
        std::lock_guard lock(views_mutex_);
        for (auto& [id, view] : view_states_)
            view->dirty_mask_.fetch_or(std::exchange(view->parked_arena_retry_, 0), std::memory_order_relaxed);
    }

    bool RenderingManager::importUsesCombinedModel() const {
        return !splitViewUsesPLYComparison(getSettings().split_view_mode);
    }

    std::optional<std::string> RenderingManager::pollImportRenderCheck(
        const RenderContext& context, const std::function<void()>& prepare_viewport) {
        if (importUsesCombinedModel())
            return prepareImportRenderCheck(context, prepare_viewport);

        // Preserve this view's displayed output while a private renderer probes
        // the provisional comparison pair. Other editor views retain their state.
        std::lock_guard lock(views_mutex_);
        auto& live = viewState(context.view);
        auto saved_view = std::move(view_states_.at(live.id));
        const auto id = saved_view->id;
        auto saved_renderer = std::move(scene_renderer_);
        auto validation = std::make_unique<ViewRenderState>();
        validation->id = id;
        validation->last_nonzero_viewport_size_ = saved_view->last_nonzero_viewport_size_;
        view_states_[id] = std::move(validation);
        auto restore = [&](void*) {
            auto& probe = *view_states_.at(id);
            probe.viewport_artifact_service_.clearViewportOutput();
            vulkanViewRenderState(probe).viewport_interop_.shutdown(vulkanContextOrNull(context.graphics_context));
            scene_renderer_.reset();
            releaseViewTargets(probe);
            view_states_[id] = std::move(saved_view);
            scene_renderer_ = std::move(saved_renderer);
            cancelImportRenderCheck();
        };
        const std::unique_ptr<void, decltype(restore)> guard(this, restore);
        if (saved_renderer)
            saved_renderer->releaseSceneResources();
        return prepareImportRenderCheck(context, prepare_viewport);
    }

    std::optional<std::string> RenderingManager::prepareImportRenderCheck(
        const RenderContext& context, const std::function<void()>& prepare_viewport) {
        const auto generation = context.scene_manager->getScene().renderGeneration();
        beginImportRenderCheck(generation);
        import_render_preparing_ = true;
        auto viewport = context.viewport;
        // Offscreen submits use the last real viewport extent while minimized;
        // no swapchain acquisition or presentation is needed to allocate slots.
        auto size = viewState(context.view).last_nonzero_viewport_size_;
        if (size.x <= 0 || size.y <= 0)
            size = glm::max(viewport.frameBufferSize, glm::ivec2{1});
        viewport.frameBufferSize = size;
        viewport.windowSize = size;
        const RenderContext preparation{
            .view = context.view,
            .viewport = viewport,
            .settings = context.settings,
            .scene_manager = context.scene_manager,
            .graphics_context = context.graphics_context,
            .preparing_import = true,
            .provisional_import_node = context.provisional_import_node};
        try {
            for (unsigned slot = 0; slot < OutputSlotRing::kFrameRingSize && !import_render_result_; ++slot) {
                markDirty(DirtyFlag::ALL, lfs::vis::FrameReason::SceneChange);
                static_cast<void>(renderFrame(preparation));
            }
        } catch (const std::exception& error) {
            LOG_ERROR("Import render preparation failed: {}", error.what());
            import_render_result_ = error.what();
        }
        import_render_preparing_ = false;
        if (import_render_result_ && import_render_result_->empty() && prepare_viewport)
            prepare_viewport();
        return import_render_result_;
    }

    void RenderingManager::cancelImportRenderCheck() {
        if (import_render_check_)
            markDirty(DirtyFlag::ALL, lfs::vis::FrameReason::SceneChange); // Publish the prepared scene through the normal viewport pass.
        import_render_check_ = false;
        import_render_result_.reset();
    }

    void RenderingManager::noteImportRenderFrame(const uint64_t generation, std::string error) {
        if (!import_render_preparing_ || !import_render_check_ || import_render_result_ || generation != import_render_generation_)
            return;
        if (!error.empty())
            import_render_result_ = std::move(error);
        // Exercise the normal allocations in every rotating frame slot.
        else if (++import_render_frames_ == OutputSlotRing::kFrameRingSize)
            import_render_result_ = std::string{};
        else
            markDirty(DirtyFlag::ALL, lfs::vis::FrameReason::SceneChange);
    }

    ViewportFrameResult RenderingManager::renderFrame(const RenderContext& context) {
        auto* const vulkan_context = vulkanContextOrNull(context.graphics_context);
        auto& view_state = viewState(context.view);
        vulkanViewRenderState(view_state).pending_frame_resources.reset();
        if (!view_state.main_render_target_.valid())
            view_state.main_render_target_ = render_targets_.allocate();
        if (context.settings.split_view_mode != SplitViewMode::Disabled) {
            if (!view_state.split_left_render_target_.valid())
                view_state.split_left_render_target_ = render_targets_.allocate();
            if (!view_state.split_right_render_target_.valid())
                view_state.split_right_render_target_ = render_targets_.allocate();
        }
        const auto release_inactive_split_outputs = [&] {
            if (view_state.split_view_service_.isActive(context.settings))
                return;
            for (auto* target : {&view_state.split_left_render_target_, &view_state.split_right_render_target_}) {
                const bool has_splats = scene_renderer_ && scene_renderer_->hasRenderTarget(*target);
                const bool has_points = point_scene_renderer_ && point_scene_renderer_->hasRenderTarget(*target);
                if (!target->valid())
                    continue;
                const bool splats_released = !has_splats || scene_renderer_->releaseRenderTarget(*target);
                const bool points_released = !has_points || point_scene_renderer_->releaseRenderTarget(*target);
                if (splats_released && points_released) {
                    render_targets_.release(*target);
                    *target = {};
                }
            }
        };

        release_inactive_split_outputs();
        syncGTComparisonViewSettings(view_state, context.settings);
        if (!view_state.rendered_settings || view_state.rendered_settings->view() != context.settings.view() || view_state.rendered_settings->scene() != context.settings.scene()) {
            view_state.dirty_mask_.fetch_or(DirtyFlag::ALL);
            view_state.rendered_settings = context.settings;
        }

        LOG_TIMER("renderVulkanFrame");
        if (view_state.vksplat_stale_frame_guard_.takeRecoveryRequest() && scene_renderer_) {
            // clear_external_backing waits for the arena to be idle. Do this
            // before taking either trainer lock, after the failed frame's borrow
            // publisher has run, so training can finish its active frame. Detach
            // first, outside releaseScratchOnIdle's readback mutex: the arena's
            // shrink callback takes that mutex while holding the arena gate.
            rendererTrainingInterop(*scene_renderer_).cancelArenaHandoff();
#if LFS_BUILD_TRAINER
            lfs::core::clear_raster_arena_external_backing();
#endif
            rendererTrainingInterop(*scene_renderer_).releaseScratchOnIdle(true);
        }
        auto frame_settings = context.settings;
        enforceProjectionBackend(frame_settings);
        const bool frame_depth_window_drag_preview = depthWindowDragPreview(context.view);
        if (context.preparing_import && splitViewUsesPLYComparison(frame_settings.split_view_mode) && context.scene_manager) {
            const auto nodes = context.scene_manager->getScene().getVisibleSplatNodeSlots();
            for (size_t index = 0; index < nodes.size(); ++index) {
                if (nodes[index].node && nodes[index].node->uuid == context.provisional_import_node) {
                    frame_settings.split_view_offset = plyComparisonImportOffset(nodes.size(), frame_settings.split_view_offset, index);
                    LOG_DEBUG("Validating provisional comparison model '{}'", nodes[index].node->name);
                    break;
                }
            }
        }

        SceneManager* const scene_manager = context.scene_manager;
        auto* const trainer_manager = scene_manager ? scene_manager->getTrainerManager() : nullptr;
        // The live trainer path deliberately bypasses scene evaluated payloads.
        // Use it only while optimization is advancing; a paused session renders
        // through the scene so its Node Editor modifier payload is visible.
        const bool is_training = scene_manager && scene_manager->hasDataset() &&
                                 trainer_manager && trainer_manager->isRunning();
        const bool training_initializing = trainer_manager &&
                                           trainer_manager->getState() == TrainingState::Starting;
        if (!is_training && scene_renderer_) {
            // Clear a callback capturing the previous trainer before any cached or
            // minimized-frame early return can leave it reachable by an auxiliary submit.
            rendererTrainingInterop(*scene_renderer_).setLiveSubmitCallback({});
            rendererTrainingInterop(*scene_renderer_).cancelArenaHandoff();
        }
        if (context.graphics_context) {
            last_graphics_context_ = context.graphics_context;
        }
        if (!is_training && scene_renderer_ &&
            vksplat_terminal_release_pending_.exchange(false, std::memory_order_acq_rel)) {
            // TrainingCompleted is posted by the training side, but Vulkan
            // resources must be released on this render thread. The event is
            // emitted only after the trainer's B3 cleanup has detached its
            // arena backing, so this also drops the viewer's final import.
            rendererTrainingInterop(*scene_renderer_).releaseScratchOnIdle(true);
            vksplat_idle_since_ = {};
        }

        const auto framebuffer_region =
            resolveFramebufferViewportRegion(context.viewport, context.logical_screen_size, context.viewport_region);
        if (framebuffer_region.valid() && !context.preparing_import) {
            // resolveFramebufferViewportRegion reports a GL bottom-left origin; window
            // readbacks are top-left, so store the flipped form callers actually crop with.
            view_state.framebuffer_viewport_rect_ = {
                .top_left = {framebuffer_region.gl_pos.x,
                             context.viewport.frameBufferSize.y - framebuffer_region.gl_pos.y - framebuffer_region.size.y},
                .size = framebuffer_region.size};
        }
        glm::ivec2 current_size = context.viewport.frameBufferSize;
        if (context.viewport_region) {
            current_size = framebuffer_region.size;
        }
        // Minimized / zero-extent: no presentable viewport work. Never hold a
        // resize training pause, never start model reads, and never publish
        // new viewer borrows — the trainer continues headless on the existing
        // handshake fences only. Restore re-enters the normal frame path.
        if (current_size.x <= 0 || current_size.y <= 0) {
            if (scene_renderer_) {
                rendererTrainingInterop(*scene_renderer_).setLiveSubmitCallback({});
                rendererTrainingInterop(*scene_renderer_).cancelArenaHandoff();
            }
            return {.image = view_state.vulkan_viewport_image_,
                    .size = view_state.vulkan_viewport_image_size_,
                    .flip_y = view_state.vulkan_viewport_image_flip_y_};
        }
        if (!context.preparing_import)
            view_state.last_nonzero_viewport_size_ = current_size;
        initialized_ = true;

        std::shared_ptr<void> frame_tensor_scope;
        const auto cached_frame_result = [this, current_size, &view_state]() -> ViewportFrameResult {
            if (!view_state.vksplat_stale_frame_guard_.canUseCachedFrame()) {
                return {};
            }
            if (vulkanViewRenderState(view_state).external_viewport_image != VK_NULL_HANDLE) {
                return {.image = publishVulkanFrameResources(view_state, {},
                                                             vulkanViewRenderState(view_state).external_viewport_image,
                                                             vulkanViewRenderState(view_state).external_viewport_image_view,
                                                             vulkanViewRenderState(view_state).external_viewport_image_layout,
                                                             vulkanViewRenderState(view_state).external_viewport_image_generation,
                                                             VK_NULL_HANDLE, 0,
                                                             {vulkanViewRenderState(view_state).viewport_interop_.frameCompletions().begin(),
                                                              vulkanViewRenderState(view_state).viewport_interop_.frameCompletions().end()}),
                        .image_generation = view_state.vulkan_viewport_image_generation_,
                        .size = view_state.vulkan_viewport_image_size_,
                        .alloc_size = view_state.vulkan_viewport_image_alloc_size_,
                        .flip_y = view_state.vulkan_viewport_image_flip_y_,
                        .matches_viewport_extent =
                            view_state.vulkan_viewport_coordinate_size_ == current_size};
            }
            if (!view_state.vulkan_viewport_image_) {
                return {.image = publishVulkanFrameResources(view_state, {},
                                                             VK_NULL_HANDLE, VK_NULL_HANDLE, VK_IMAGE_LAYOUT_UNDEFINED, 0,
                                                             VK_NULL_HANDLE, 0,
                                                             {vulkanViewRenderState(view_state).viewport_interop_.frameCompletions().begin(),
                                                              vulkanViewRenderState(view_state).viewport_interop_.frameCompletions().end()}),
                        .image_generation = view_state.split_view_image_generation_,
                        .size = view_state.vulkan_viewport_image_size_,
                        .alloc_size = view_state.vulkan_viewport_image_alloc_size_,
                        .flip_y = view_state.vulkan_viewport_image_flip_y_,
                        .matches_viewport_extent =
                            view_state.vulkan_viewport_coordinate_size_ == current_size};
            }
            return {.image = publishVulkanFrameResources(view_state, view_state.vulkan_viewport_image_,
                                                         VK_NULL_HANDLE, VK_NULL_HANDLE, VK_IMAGE_LAYOUT_UNDEFINED, 0,
                                                         VK_NULL_HANDLE, 0,
                                                         {vulkanViewRenderState(view_state).viewport_interop_.frameCompletions().begin(),
                                                          vulkanViewRenderState(view_state).viewport_interop_.frameCompletions().end()}),
                    .image_generation = view_state.vulkan_viewport_image_generation_,
                    .size = view_state.vulkan_viewport_image_size_,
                    .alloc_size = view_state.vulkan_viewport_image_alloc_size_,
                    .flip_y = view_state.vulkan_viewport_image_flip_y_,
                    .matches_viewport_extent =
                        view_state.vulkan_viewport_coordinate_size_ == current_size};
        };
        const auto defer_shared_scratch = [this, &view_state](const std::string& reason) {
            if (reason.find("arena is busy") != std::string::npos) {
                // No standing reservation: the retry on the next frame waits
                // for its window actively (waitForArenaHandoff) and a parked
                // preview polls. Reserving here held training out for the
                // whole GUI round trip after every failed attempt.
                (void)view_state.vksplat_stale_frame_guard_.onDeferral(
                    StaleFrameGuard::DeferralKind::ArenaContention);
                return;
            }
            if (view_state.vksplat_stale_frame_guard_.onDeferral()) {
                LOG_WARN("Scene shared scratch deferred {} viewport attempts; dropping stale viewport and resetting scratch before the next attempt: {}",
                         StaleFrameGuard::kMaxCachedDeferrals, reason);
                // Reset alone still needs arena access on the next render.
                // Hide the old publication until fresh output is available;
                // the renderer retains its output allocations and GPU fences.
                clearViewportImageState(view_state);
                clearVulkanMeshFrame(view_state);
                view_state.viewport_artifact_service_.clearViewportOutput();
            }
        };
        const auto update_cached_split_position = [this, &frame_settings, &view_state](const bool require_position_change) -> bool {
            if (!view_state.split_view_service_.isActive(frame_settings)) {
                return false;
            }

            const float split_position = std::clamp(frame_settings.split_position, 0.0f, 1.0f);
            std::lock_guard lock(vulkanViewRenderState(view_state).mesh_frame_mutex);
            if (!vulkanViewRenderState(view_state).mesh_frame.split_view.enabled) {
                return false;
            }

            auto& split = vulkanViewRenderState(view_state).mesh_frame.split_view;
            // PLY panels are rendered with a 12.5%-viewport-width overlap around
            // the cached splitter. A drag inside that overlap only changes the
            // composite divider; leaving it is the point where panel renders must
            // be refreshed at the new widths/origins.
            if (splitViewUsesPLYComparison(frame_settings.split_view_mode) &&
                (split_position < split.right.start_position ||
                 split_position > split.left.end_position)) {
                return false;
            }
            const bool split_position_changed =
                split.split_position != split_position ||
                split.left.end_position != split_position ||
                split.right.start_position != split_position ||
                (vulkanViewRenderState(view_state).mesh_frame.panels.size() == 2 &&
                 (vulkanViewRenderState(view_state).mesh_frame.panels[0].end_position != split_position ||
                  vulkanViewRenderState(view_state).mesh_frame.panels[1].start_position != split_position));
            if (!split_position_changed) {
                return !require_position_change;
            }

            split.split_position = split_position;
            if (!splitViewUsesPLYComparison(frame_settings.split_view_mode)) {
                split.left.start_position = 0.0f;
                split.left.end_position = split_position;
                split.right.start_position = split_position;
                split.right.end_position = 1.0f;
            }

            if (vulkanViewRenderState(view_state).mesh_frame.panels.size() == 2 &&
                !splitViewUsesPLYComparison(frame_settings.split_view_mode)) {
                vulkanViewRenderState(view_state).mesh_frame.panels[0].start_position = 0.0f;
                vulkanViewRenderState(view_state).mesh_frame.panels[0].end_position = split_position;
                vulkanViewRenderState(view_state).mesh_frame.panels[1].start_position = split_position;
                vulkanViewRenderState(view_state).mesh_frame.panels[1].end_position = 1.0f;
            }
            view_state.viewport_artifact_service_.invalidateCapturedImage();
            return true;
        };
        const auto has_cached_split_view_output = [this, &view_state]() -> bool {
            if (view_state.vulkan_viewport_image_size_.x <= 0 ||
                view_state.vulkan_viewport_image_size_.y <= 0) {
                return false;
            }
            std::lock_guard lock(vulkanViewRenderState(view_state).mesh_frame_mutex);
            return vulkanViewRenderState(view_state).mesh_frame.split_view.enabled;
        };

        const auto ensure_auxiliary_rendering_engine =
            [this]() -> std::expected<lfs::rendering::RenderingEngine*, std::string> {
            if (!engine_) {
                engine_ = lfs::rendering::RenderingEngine::create();
            }
            if (!engine_->isInitialized()) {
                if (auto init_result = engine_->initialize(); !init_result) {
                    return std::unexpected(init_result.error());
                }
            }
            return engine_.get();
        };

        glm::vec2 screen_viewport_pos(0.0f, 0.0f);
        glm::vec2 screen_viewport_size(
            static_cast<float>(context.viewport.windowSize.x),
            static_cast<float>(context.viewport.windowSize.y));
        if (context.viewport_region) {
            screen_viewport_pos = {context.viewport_region->x, context.viewport_region->y};
            screen_viewport_size = {context.viewport_region->width, context.viewport_region->height};
        }
        const auto interaction_panels = buildSplitViewInteractionPanels(
            context.viewport, frame_settings,
            screen_viewport_pos,
            screen_viewport_size);
        if (!context.preparing_import)
            view_state.viewport_interaction_context_.updatePickContext(interaction_panels);

        const auto resize_result =
            view_state.frame_lifecycle_service_.handleViewportResize(current_size);
        if (resize_result.dirty) {
            markViewDirty(context.view, resize_result.dirty, lfs::vis::FrameReason::SceneChange);
        }
        const bool resize_deferring = !context.preparing_import && view_state.frame_lifecycle_service_.isResizeDeferring();
        const auto requested_upscaler = sceneUpscalerBackendFromId(frame_settings.scene_upscaler)
                                            .value_or(SceneUpscalerBackend::Native);
        logSceneUpscalerRequest(view_state, frame_settings);
        const auto reported_upscaler = sceneUpscalerRuntimeSelection(context.view);
        const bool reconstruction_runtime_ready =
            reported_upscaler.requested == requested_upscaler &&
            requested_upscaler != SceneUpscalerBackend::Native &&
            reported_upscaler.effective == requested_upscaler &&
            !reported_upscaler.fellBack();
        float scale = effectiveSceneRenderScale(
            frame_settings.render_scale,
            frame_settings.scene_upscaler_scale,
            reconstruction_runtime_ready);
        if (!context.preparing_import && resize_result.use_interactive_render_scale) {
            scale = std::min(scale, kInteractiveResizeRenderScale);
        }
        // Only unresolved viewer allocation failures lease a reduced preview.
        // Training allocation retries must not lower the viewer resolution.
        const bool memory_pressure_active =
            lfs::core::MemoryPressureCoordinator::instance().pressure_active();
        if (memory_pressure_active) {
            scale = std::clamp(scale * 0.5f, 0.25f, 1.0f);
        }
        glm::ivec2 render_size(
            std::max(static_cast<int>(std::lround(static_cast<float>(current_size.x) * scale)), 1),
            std::max(static_cast<int>(std::lround(static_cast<float>(current_size.y) * scale)), 1));
        const std::uint32_t vendor_quality =
            frame_settings.scene_upscaler_preset == "performance"
                ? LFS_SCENE_UPSCALER_PLUGIN_PERFORMANCE
            : frame_settings.scene_upscaler_preset == "quality"
                ? LFS_SCENE_UPSCALER_PLUGIN_QUALITY
                : LFS_SCENE_UPSCALER_PLUGIN_BALANCED;
        auto* const requested_plugin = sceneUpscalerPlugin(requested_upscaler);
        std::uint32_t plugin_jitter_phase_count = 0;
        if (requested_plugin != nullptr && reconstruction_runtime_ready &&
            !resize_result.use_interactive_render_scale && !memory_pressure_active) {
            if (const auto optimal =
                    pluginOptimalSettings(*requested_plugin, current_size, vendor_quality)) {
                render_size = {static_cast<int>(optimal->render_width),
                               static_cast<int>(optimal->render_height)};
                scale = std::min(static_cast<float>(render_size.x) /
                                     static_cast<float>(current_size.x),
                                 static_cast<float>(render_size.y) /
                                     static_cast<float>(current_size.y));
                plugin_jitter_phase_count = optimal->jitter_phase_count;
            }
        }
        // Ground-truth comparison is a stable reference readout. Its preview
        // resolution follows only the viewport size and the base render scale;
        // scene reconstruction, interactive-resize, and memory-pressure
        // reductions must not re-size it, otherwise toggling the reconstruction
        // setting re-decodes the reference image and briefly flashes a stale
        // cache entry (possibly from a different camera).
        const float gt_comparison_scale =
            effectiveSceneRenderScale(frame_settings.render_scale, 1.0f, false);
        const glm::ivec2 gt_comparison_size(
            std::max(static_cast<int>(std::lround(static_cast<float>(current_size.x) * gt_comparison_scale)), 1),
            std::max(static_cast<int>(std::lround(static_cast<float>(current_size.y) * gt_comparison_scale)), 1));
        auto& resolution_profiler = lfs::diagnostics::VramProfiler::instance();
        resolution_profiler.setGauge("viewer.resolution.scene.base_scale",
                                     effectiveSceneRenderScale(frame_settings.render_scale, 1.0f, false));
        resolution_profiler.setGauge(
            "viewer.resolution.scene.reconstruction_scale",
            !reconstruction_runtime_ready
                ? 1.0
                : frame_settings.scene_upscaler_scale);
        resolution_profiler.setGauge("viewer.resolution.scene.effective_scale", scale);
        const DirtyMask pending_dirty = view_state.dirty_mask_.load(std::memory_order_relaxed);
        const bool only_split_position_pending =
            (pending_dirty & ~DirtyFlag::SPLIT_POSITION) == 0;
        if (!context.preparing_import && (pending_dirty & DirtyFlag::SPLIT_POSITION) != 0 &&
            view_state.vulkan_viewport_image_size_ == render_size &&
            has_cached_split_view_output() &&
            update_cached_split_position(!only_split_position_pending)) {
            view_state.dirty_mask_.fetch_and(~DirtyFlag::SPLIT_POSITION, std::memory_order_relaxed);
            LOG_PERF("renderVulkanFrame: split-position early cache HIT (returning cached image)");
            return cached_frame_result();
        }

        // Window resize/minimize must never alter the training schedule. Viewer
        // work quiesces itself (cached frames while deferring; output-ring
        // recreate waits ring watermarks only). pauseTrainingTemporary is not
        // used on this path â€” other interactive wait sites keep it.

        const auto now = std::chrono::steady_clock::now();
        const bool camera_changed = !view_state.navigation_pose_valid_ ||
                                    view_state.last_navigation_rotation_ != context.viewport.camera.R ||
                                    view_state.last_navigation_translation_ != context.viewport.camera.t;
        view_state.last_navigation_rotation_ = context.viewport.camera.R;
        view_state.last_navigation_translation_ = context.viewport.camera.t;
        view_state.navigation_pose_valid_ = true;
        const double viewer_turn_ms = lfs::core::raster_arena_turns().viewer_turn_ms;
        const bool rest_only = is_training && navigationRendersOnlyAtRest(viewer_turn_ms);
        if (rest_only && camera_changed) {
            view_state.camera_settle_pending_ = true;
            view_state.camera_settle_deadline_ = now + kCameraSettle;
        }
        if (rest_only && view_state.camera_settle_pending_ && now < view_state.camera_settle_deadline_ &&
            (vulkanViewRenderState(view_state).external_viewport_image != VK_NULL_HANDLE || view_state.vulkan_viewport_image_)) {
            static bool warned = false;
            if (!warned) {
                warned = true;
                LOG_WARN("Viewer frames take {:.0f} ms on the GPU (interactive budget {:.0f} ms): "
                         "the viewport updates when the camera rests; GPU memory is probably oversubscribed",
                         viewer_turn_ms, kInteractiveViewerBudgetMs);
            }
            lfs::core::raster_arena_predict_viewer_frame(false);
            // Retain CAMERA until the settle wakeup; never reserve scratch while
            // deferring. New input, rather than this retry, extends the deadline.
            view_state.dirty_mask_.fetch_or(DirtyFlag::CAMERA, std::memory_order_relaxed);
            if (scene_renderer_)
                rendererTrainingInterop(*scene_renderer_).cancelArenaHandoff();
            return cached_frame_result();
        }
        view_state.camera_settle_pending_ = false;
        lfs::core::raster_arena_predict_viewer_frame(
            is_training && (view_state.dirty_mask_.load(std::memory_order_relaxed) & DirtyFlag::CAMERA) != 0);

        // Training previews never wait for a step-boundary read, including
        // discrete layout resizes. On contention, retain the previous matching
        // frame and retry on the next cadence tick; the GUI commits a staged
        // layout only after matches_viewport_extent reports a fresh output.
        // This also applies without a cached frame: model growth can hold the
        // exclusive lock while waiting for chunk binding on this viewer thread.
        const bool training_try_lock = is_training;
        if (is_training && scene_renderer_ &&
            (view_state.dirty_mask_.load(std::memory_order_relaxed) & ~DirtyFlag::SPLATS) != 0) {
            // Explicit edits (camera, selection, visibility, settings) actively wait
            // for a fresh frame. Only passive SPLATS refreshes park. No model
            // lock is held, so a refining trainer can take the exclusive one.
            (void)rendererTrainingInterop(*scene_renderer_).waitForArenaHandoff(kNavigationArenaWait);
        }
        auto render_lock = acquireLiveModelRenderLock(scene_manager, training_try_lock);
        const bool render_lock_contended = training_try_lock && !render_lock.has_value() &&
                                           scene_manager && scene_manager->getTrainerManager() &&
                                           scene_manager->getTrainerManager()->getTrainer();

        const lfs::core::SplatData* model = nullptr;
        SceneRenderState scene_state;
        size_t comparison_identity = 0;
        auto sample_model_under_lock = [&]() {
            scene_state = {};
            comparison_identity = 0;
            if (!scene_manager) {
                model = nullptr;
                return;
            }
            LOG_TIMER("renderVulkanFrame.buildRenderState");
            if (splitViewUsesPLYComparison(frame_settings.split_view_mode)) {
                // Comparison draws owned node models. Do not concatenate them
                // into a hidden combined copy just to fill FrameContext.model.
                scene_state = scene_manager->buildRenderState({.metadata_only = true});
                model = nullptr;
                const auto visible_nodes = scene_manager->getScene().getVisibleSplatNodeSlots();
                const auto pair =
                    plyComparisonPairForOffset(visible_nodes.size(), frame_settings.split_view_offset);
                if (pair) {
                    const auto* const left_node = visible_nodes[pair->first].node;
                    const auto* const right_node = visible_nodes[pair->second].node;
                    comparison_identity =
                        reinterpret_cast<size_t>(left_node) ^
                        (reinterpret_cast<size_t>(right_node) << 1);
                    if (left_node && left_node->model) {
                        comparison_identity ^=
                            reinterpret_cast<size_t>(left_node->model.get());
                    }
                    if (right_node && right_node->model) {
                        comparison_identity ^=
                            reinterpret_cast<size_t>(right_node->model.get());
                    }
                }
                return;
            }
            scene_state = scene_manager->buildRenderState();
            model = scene_state.combined_model;
        };
        if (!render_lock_contended) {
            sample_model_under_lock();
        }
        bool has_renderable_model = false;
        bool has_visible_gaussian_model = false;
        bool has_point_cloud = false;
        bool has_meshes = false;
        bool has_environment = false;
        bool has_render_content = false;
        auto refresh_content_flags = [&]() {
            has_renderable_model = hasRenderableGaussians(model);
            has_visible_gaussian_model =
                has_renderable_model && scene_state.visible_splat_count > 0;
            if (!has_visible_gaussian_model &&
                splitViewUsesPLYComparison(frame_settings.split_view_mode) && scene_manager) {
                const auto visible_nodes = scene_manager->getScene().getVisibleSplatNodeSlots();
                has_visible_gaussian_model = std::any_of(
                    visible_nodes.begin(),
                    visible_nodes.end(),
                    [](const auto& slot) {
                        return slot.node && hasRenderableGaussians(slot.node->model.get());
                    });
                if (!has_visible_gaussian_model) {
                    has_visible_gaussian_model = hasRenderableGaussians(
                        scene_manager->getScene().peekCombinedModel());
                }
                has_renderable_model = has_visible_gaussian_model;
            }
            has_point_cloud =
                scene_state.point_cloud != nullptr && scene_state.point_cloud->size() > 0;
            has_meshes = std::any_of(scene_state.meshes.begin(),
                                     scene_state.meshes.end(),
                                     [](const auto& mesh) { return mesh.mesh != nullptr; });
            has_environment = environmentBackgroundEnabled(frame_settings);
            has_render_content =
                has_visible_gaussian_model || has_point_cloud || has_meshes || has_environment;
        };
        refresh_content_flags();
        const size_t model_ptr = comparison_identity != 0 ? comparison_identity
                                                          : reinterpret_cast<size_t>(model);
        // Edit-mode handoff moves the same SplatData into a scene node. Its
        // address does not change, but the trainer's GPU handshake is gone.
        // Use dataset ownership, not Running/Paused, so completion alone does
        // not reset the renderer while the trainer still owns its fences.
        const auto model_source = scene_manager && scene_manager->hasDataset()
                                      ? ViewportFrameLifecycleService::ModelSource::Training
                                      : ViewportFrameLifecycleService::ModelSource::Scene;

        // Skip model-change tracking while the step-boundary lock is contended:
        // model_ptr is null only because densify holds exclusive, not because the
        // scene dropped the model. Treating it as a change would wipe the retained
        // last splat image (the whole point of the cadence path).
        if (!render_lock_contended) {
            const bool ownership_changed = renderer_model_source_ != model_source;
            renderer_model_source_ = model_source;
            if (ownership_changed && !is_training && scene_renderer_) {
                // A new view does not replace the shared model or its GPU outputs.
                // Only a trainer ownership transition tears down the handshake.
#if LFS_BUILD_TRAINER
                if (trainer_manager) {
                    if (auto* trainer = trainer_manager->getTrainer())
                        trainer->setViewerReleaseFence(nullptr, {});
                }
#endif
                frame_tensor_scope.reset();
                scene_renderer_->reset();
                invalidateGTComparisonImageCache(view_state);
                std::lock_guard lock(views_mutex_);
                for (auto& [id, view] : view_states_) {
                    view->vulkan_viewport_image_.reset();
                    vulkanViewRenderState(*view).external_viewport_image = VK_NULL_HANDLE;
                    vulkanViewRenderState(*view).external_viewport_image_view = VK_NULL_HANDLE;
                    vulkanViewRenderState(*view).external_viewport_image_layout = VK_IMAGE_LAYOUT_UNDEFINED;
                    vulkanViewRenderState(*view).external_viewport_image_generation = 0;
                    view->vulkan_viewport_image_size_ = {};
                    vulkanViewRenderState(*view).mesh_frame = {};
                    vulkanViewRenderState(*view).viewport_interop_.setSceneImage({}, {}, false, 0);
                    vulkanViewRenderState(*view).viewport_interop_.clearSplitRightImage();
                    vulkanViewRenderState(*view).viewport_interop_.clearDepthBlitImage();
                    view->viewport_artifact_service_.clearViewportOutput();
                    view->gt_async_depth_ticket_ = 0;
                    view->gt_async_depth_dest_ = {};
                    view->gt_async_ticket_intrinsics_.reset();
                    view->gt_async_ticket_view_.reset();
                    view->gt_async_held_display_.reset();
                    view->gt_async_held_view_.reset();
                    view->dirty_mask_.fetch_or(DirtyFlag::ALL);
                }
            }
            if (view_state.frame_lifecycle_service_.handleModelChange(model_ptr, view_state.viewport_artifact_service_, model_source).changed) {
                clearViewportImageState(view_state);
                view_state.last_logged_vksplat_render_error_.clear();
                view_state.viewport_artifact_service_.clearViewportOutput();
                markViewDirty(context.view, DirtyFlag::ALL, lfs::vis::FrameReason::SceneChange);
            }
        }

        const bool synchronize_vksplat_input_upload = is_training;
        // Training refresh marks the scene dirty from the main loop; its provenance decides whether a
        // temporal settle burst can produce useful history or would only race the next training tick.
        const DirtyMask training_refresh_dirty =
            view_state.training_refresh_dirty_.exchange(0, std::memory_order_relaxed);

        const bool has_cached_gpu_only_frame = [&]() {
            if (view_state.vulkan_viewport_image_size_.x <= 0 ||
                view_state.vulkan_viewport_image_size_.y <= 0) {
                return false;
            }
            std::lock_guard lock(vulkanViewRenderState(view_state).mesh_frame_mutex);
            return vulkanViewRenderState(view_state).mesh_frame.split_view.enabled ||
                   !vulkanViewRenderState(view_state).mesh_frame.items.empty() ||
                   vulkanViewRenderState(view_state).mesh_frame.environment.enabled;
        }();
        const bool has_cached_viewport_output =
            view_state.vulkan_viewport_image_ != nullptr ||
            vulkanViewRenderState(view_state).external_viewport_image != VK_NULL_HANDLE ||
            has_cached_gpu_only_frame;
        LOG_PERF("renderVulkanFrame.resize deferring={} render={} interactive_scale={} completed={} render_scale={:.2f} render_size={}x{} current_size={}x{}",
                 resize_deferring,
                 resize_result.render_resized_frame,
                 resize_result.use_interactive_render_scale,
                 resize_result.completed,
                 scale,
                 render_size.x,
                 render_size.y,
                 current_size.x,
                 current_size.y);

        if (const DirtyMask required_dirty =
                view_state.frame_lifecycle_service_.requiredDirtyMask(
                    has_cached_viewport_output,
                    has_render_content,
                    frame_settings.split_view_mode);
            required_dirty) {
            view_state.dirty_mask_.fetch_or(required_dirty, std::memory_order_relaxed);
        }

        DirtyMask frame_dirty = view_state.dirty_mask_.exchange(0);
        if (scene_renderer_) {
            scene_renderer_->setCameraNavigating(
                is_training && (frame_dirty & DirtyFlag::CAMERA) != 0);
        }
        const bool lod_results_ready = lod_controller_ && lod_controller_->hasReadyResults();
        const bool lod_transition_active = lod_controller_ && lod_controller_->transitionActive();
        const DirtyMask temporal_dirty = frame_dirty;
        if (lod_results_ready) {
            frame_dirty |= DirtyFlag::CAMERA;
        }
        if (lod_transition_active) {
            frame_dirty |= DirtyFlag::CAMERA;
        }
        const bool temporal_split_supported =
            !view_state.split_view_service_.isActive(frame_settings) ||
            splitViewUsesPLYComparison(frame_settings.split_view_mode);
        const bool temporal_backend_requested =
            requested_upscaler == SceneUpscalerBackend::Temporal || requested_plugin != nullptr;
        const bool projection_supported =
            requested_plugin == nullptr || !frame_settings.orthographic ||
            !requested_plugin->hasCapability(
                LFS_SCENE_UPSCALER_PLUGIN_CAPABILITY_REQUIRES_PERSPECTIVE);
        const bool plugin_eligible =
            requested_plugin == nullptr ||
            (sceneUpscalerPluginSupportsOutputExtent(current_size) &&
             !resize_result.use_interactive_render_scale && !memory_pressure_active);
        const auto temporal_setup = prepareSceneTemporalFrame(
            view_state.temporal_convergence_,
            {.backend_requested = temporal_backend_requested,
             .runtime_ready = reported_upscaler.requested == requested_upscaler &&
                              !reported_upscaler.fellBack(),
             .pipeline_eligible = plugin_eligible,
             .projection_supported = projection_supported,
             .equirectangular = frame_settings.equirectangular,
             .appearance_correction = frame_settings.apply_appearance_correction,
             .split_supported = temporal_split_supported,
             .raster_supported = lfs::rendering::isVkSplatBackend(frame_settings.raster_backend),
             .interactive_scale = resize_result.use_interactive_render_scale,
             .memory_pressure = memory_pressure_active,
             .lod_results_ready = lod_results_ready,
             .lod_transition_active = lod_transition_active,
             .frame_dirty = temporal_dirty,
             .training_refresh_dirty = training_refresh_dirty,
             .settle_sample_count = plugin_jitter_phase_count > 0
                                        ? plugin_jitter_phase_count
                                        : TemporalConvergenceController::SAMPLE_COUNT,
             .jitter_phase_count = plugin_jitter_phase_count});
        const bool temporal_eligible = temporal_setup.eligible;
        {
            std::lock_guard lock(settings_mutex_);
            view_state.scene_upscaler_mode_unsupported_ = temporal_setup.mode_unsupported;
        }
        const bool training_refresh_only = temporal_setup.training_refresh_only;
        const glm::vec2 applied_temporal_jitter_pixels = temporal_setup.jitter_pixels;
        const std::uint64_t temporal_camera_cut_generation =
            view_state.temporal_camera_cut_generation_.load(std::memory_order_acquire);
        const bool temporal_camera_cut =
            temporal_camera_cut_generation !=
            view_state.consumed_temporal_camera_cut_generation_;
        if ((frame_dirty & (DirtyFlag::SPLATS | DirtyFlag::MESH | DirtyFlag::BACKGROUND)) != 0) {
            if (++view_state.temporal_scene_revision_ == 0)
                ++view_state.temporal_scene_revision_;
        }
        constexpr DirtyMask projection_dirty =
            DirtyFlag::CAMERA | DirtyFlag::SPLATS | DirtyFlag::VIEWPORT |
            DirtyFlag::SPLIT_VIEW | DirtyFlag::TEMPORAL;
        if ((frame_dirty & projection_dirty) != 0) {
            ++view_state.viewport_projection_generation_;
            if (view_state.viewport_projection_generation_ == 0) {
                ++view_state.viewport_projection_generation_;
            }
        }
        LOG_PERF("renderVulkanFrame.frame_dirty=0x{:x} model={} pc={} mesh={} env={} lock_contended={}",
                 frame_dirty,
                 has_renderable_model,
                 has_point_cloud,
                 has_meshes,
                 has_environment,
                 render_lock_contended);
        // On contention, retain any previous output and re-queue
        // dirty so the next cadence tick retries after the exclusive lock drops.
        if (render_lock_contended) {
            DirtyMask retry_dirty = frame_dirty;
            if (training_refresh_only) {
                retry_dirty &= ~training_refresh_dirty;
            }
            if (retry_dirty != 0) {
                view_state.dirty_mask_.fetch_or(retry_dirty, std::memory_order_relaxed);
            }
            LOG_PERF("renderVulkanFrame: {} lock contended (deferring preview)",
                     training_initializing ? "training initialization" : "step-boundary");
            render_lock.reset();
            return cached_frame_result();
        }
        // Scene state is authoritative here (contended frames returned above): nothing
        // visible must clear the viewport even when a cached frame exists â€” a consolidated
        // multi-splat scene keeps the same combined-model pointer when every node is
        // hidden, so model-change tracking never clears the stale image.
        if (!has_render_content) {
            if (scene_manager)
                noteImportRenderFrame(scene_manager->getScene().renderGeneration());
            clearViewportImageState(view_state);
            view_state.vulkan_viewport_coordinate_size_ = current_size;
            view_state.last_logged_vksplat_render_error_.clear();
            view_state.viewport_artifact_service_.clearViewportOutput();
            clearVulkanMeshFrame(view_state);
            clearPublishedGTComparisonActualFrame(view_state);
            render_lock.reset();
            return {.matches_viewport_extent = true, .rendered = true};
        }

        const DirtyMask split_deferred_dirty = frame_dirty & ~DirtyFlag::SPLIT_POSITION;
        if (!context.preparing_import && (frame_dirty & DirtyFlag::SPLIT_POSITION) != 0 &&
            has_cached_viewport_output &&
            update_cached_split_position(split_deferred_dirty != 0)) {
            const DirtyMask deferred_dirty = split_deferred_dirty;
            if (deferred_dirty != 0) {
                view_state.dirty_mask_.fetch_or(deferred_dirty, std::memory_order_relaxed);
            }
            LOG_PERF("renderVulkanFrame: split-position cache HIT (returning cached image, deferred_dirty=0x{:x})",
                     deferred_dirty);
            render_lock.reset();
            return cached_frame_result();
        }

        if (resize_deferring &&
            has_cached_viewport_output &&
            !resize_result.render_resized_frame) {
            update_cached_split_position(false);
            constexpr DirtyMask resize_defer_consumed_dirty =
                DirtyFlag::CAMERA | DirtyFlag::VIEWPORT | DirtyFlag::OVERLAY;
            const DirtyMask deferred_dirty = frame_dirty & ~resize_defer_consumed_dirty;
            if (deferred_dirty != 0) {
                view_state.dirty_mask_.fetch_or(deferred_dirty, std::memory_order_relaxed);
            }
            LOG_PERF("renderVulkanFrame: resize defer (returning cached image)");
            render_lock.reset();
            return cached_frame_result();
        }

        const bool vksplat_viewport_resize =
            is_training &&
            vulkan_context != nullptr &&
            scene_renderer_ != nullptr &&
            scene_renderer_->nextOutputImagesNeedResize(
                render_size, view_state.main_render_target_) &&
            lfs::rendering::isVkSplatBackend(frame_settings.raster_backend);
        if (vksplat_viewport_resize) {
            // While the viewport size is moving, return cached frames without
            // starting model reads or publishing viewer borrows. Once it is
            // stable, ensureOutputImages retires prior rings through the
            // producer/consumer watermarks; the viewport-independent exportable
            // model import is retained. A normal live submit then re-arms the
            // handshake.
            if (!resize_result.require_immediate_output_resize &&
                has_cached_viewport_output &&
                view_state.frame_lifecycle_service_.resizeRecentlyChanged(kTrainingOutputResizeStableDelay)) {
                view_state.dirty_mask_.fetch_or(vksplatOutputResizeRetryDirty(frame_dirty),
                                                std::memory_order_relaxed);
                if (scene_renderer_) {
                    rendererTrainingInterop(*scene_renderer_).setLiveSubmitCallback({});
                }
                render_lock.reset();
                return cached_frame_result();
            }
            if (vulkan_context)
                vulkan_context->tensorInterop().drain(lfs::core::default_gpu_backend());
            LOG_DEBUG("Scene output resize to {}x{} (viewer-side quiesce; training continues)",
                      render_size.x,
                      render_size.y);
        }

        // Camera thumbnails are uploaded and composited by the GUI viewport
        // pass. Reuse the cached splat image for an overlay-only refresh; a
        // thumbnail must never turn a stream-in update into a full splat pass.
        if (!vksplat_viewport_resize &&
            (frame_dirty == 0 || frame_dirty == DirtyFlag::OVERLAY) &&
            has_cached_viewport_output) {
            LOG_PERF("renderVulkanFrame: cache HIT (returning cached image)");
            render_lock.reset();
            return cached_frame_result();
        }

        // Only a frame that will perform renderer work may join the trainer's
        // read epoch. Cached frames must not enqueue a legacy-stream wait or
        // publish a reader edge that serializes the next training step.
        if (is_training && vulkan_context &&
            lfs::rendering::isVkSplatBackend(frame_settings.raster_backend)) {
            if (!scene_renderer_) {
                scene_renderer_ = createSceneRenderer(*context.graphics_context);
            }
            if (const auto ok = rendererTrainingInterop(*scene_renderer_).ensureHandshakeReady(); !ok) {
                LOG_WARN("Scene handshake pre-init skipped: {}", ok.error());
            }
        }
        frame_tensor_scope.reset();
        if (vulkan_context)
            frame_tensor_scope = vulkan_context->tensorInterop().execution_scope(
                lfs::core::default_gpu_backend());
#if LFS_BUILD_TRAINER
        lfs::training::Trainer* live_trainer = nullptr;
        if (is_training && trainer_manager && scene_renderer_ &&
            rendererTrainingInterop(*scene_renderer_).hasLiveTrainerReleaseFence()) {
            // A live stream without its release fence is a partial initialization;
            // render() will fail too, so never install a handshake missing the
            // trainer's reverse dependency.
            live_trainer = trainer_manager->getTrainer();
        }
        // Held shared until all CPU/GPU frame preparation and readback paths exit.
        // Passive preview: try_to_lock so a mid-step optimizer exclusive does not stall
        // the UI; retain last splat image and retry on the next cadence tick.
        std::optional<std::shared_lock<std::shared_mutex>> model_read_lock;
        if (live_trainer) {
            std::shared_lock<std::shared_mutex> candidate(
                live_trainer->getModelAccessMutex(), std::try_to_lock);
            if (!candidate.owns_lock()) {
                if (has_cached_viewport_output) {
                    if (frame_dirty != 0) {
                        view_state.dirty_mask_.fetch_or(frame_dirty, std::memory_order_relaxed);
                    }
                    LOG_PERF("renderVulkanFrame: model-access lock contended (retaining cached splat)");
                    render_lock.reset();
                    return cached_frame_result();
                }
                // No cache yet â€” block once for the first published frame.
                candidate = std::shared_lock<std::shared_mutex>(live_trainer->getModelAccessMutex());
            }
            model_read_lock.emplace(std::move(candidate));
        }
        if (live_trainer) {
            live_trainer->setViewerReleaseFence(vulkan_context->device(),
                                                {rendererTrainingInterop(*scene_renderer_).renderCompleteTimeline(), 0, {}});
            live_trainer->beginModelRead(lfs::core::TensorExecutionTarget::current());
            lfs::training::Trainer* const trainer = live_trainer;
            rendererTrainingInterop(*scene_renderer_).setLiveSubmitCallback([trainer](const std::uint64_t value) { trainer->publishViewerBorrow(value); });
        } else if (scene_renderer_) {
            rendererTrainingInterop(*scene_renderer_).setLiveSubmitCallback({});
        }
        // Prompt callbacks publish after each submit. This scope-exit edge also
        // covers preparation errors after CUDA model reads but before a submit.
        struct ViewerBorrowPublisher {
            lfs::training::Trainer* trainer;
            SceneRenderer* renderer;
            ~ViewerBorrowPublisher() {
                if (trainer && renderer) {
                    try {
                        trainer->endModelRead(lfs::core::TensorExecutionTarget::current());
                        trainer->publishViewerBorrow(rendererTrainingInterop(*renderer).renderCompleteValue());
                    } catch (const std::exception& e) {
                        LOG_ERROR("ViewerBorrowPublisher: endModelRead/publishViewerBorrow failed "
                                  "during frame teardown: {}",
                                  e.what());
                    } catch (...) {
                        LOG_ERROR("ViewerBorrowPublisher: endModelRead/publishViewerBorrow failed "
                                  "during frame teardown with an unknown error");
                    }
                }
            }
        } viewer_borrow_publisher{live_trainer, scene_renderer_.get()};
#else
        if (scene_renderer_)
            rendererTrainingInterop(*scene_renderer_).setLiveSubmitCallback({});
#endif

        view_state.framerate_controller_.beginFrame();

        const FrameContext frame_ctx{
            .viewport = context.viewport,
            .viewport_region = context.viewport_region,
            .render_lock_held = render_lock.has_value(),
            .scene_manager = scene_manager,
            .model = model,
            .scene_state = std::move(scene_state),
            .settings = frame_settings,
            .render_size = render_size,
            .viewport_pos = {0, 0},
            .frame_dirty = frame_dirty,
            .training_active = is_training,
            .depth_window_drag_preview = frame_depth_window_drag_preview,
            .cursor_preview = view_state.viewport_overlay_service_.cursorPreview(),
            .gizmo = gizmo_state_,
            .hovered_camera_id = camera_interaction_service_.hoveredCameraId(),
            .current_camera_id = view_state.gt_comparison_camera_uid_ >= 0 ? view_state.gt_comparison_camera_uid_ : camera_interaction_service_.currentCameraId(),
            .hovered_gaussian_id = view_state.viewport_overlay_service_.hoveredGaussianId(),
            .selection_flash_intensity = view_state.animation_state_.selectionFlashIntensity(),
            .view_panels = {},
            .scene_jitter_pixels = applied_temporal_jitter_pixels,
        };

        const auto complete_temporal_convergence_frame =
            [this, temporal_camera_cut_generation, &view_state]() {
                view_state.consumed_temporal_camera_cut_generation_ = temporal_camera_cut_generation;
                if (view_state.temporal_convergence_.completeSuccessfulFrame())
                    requestViewFollowUp(view_state, DirtyFlag::TEMPORAL);
            };

        std::shared_ptr<lfs::core::Tensor> rendered_image;
        lfs::rendering::FrameMetadata rendered_metadata{};
        std::string render_error;
        bool rendered_image_contains_ground_truth = false;
        bool rendered_gt_actual_size = false;
        std::optional<GTComparisonActualFrame::Snapshot>
            rendered_gt_actual_snapshot;
        glm::ivec2 rendered_gt_content_size{0, 0};
        std::optional<GTPresentedView> rendered_gt_view;
        const SceneTemporalQuality temporal_quality =
            frame_settings.scene_upscaler_preset == "performance"
                ? SceneTemporalQuality::Performance
            : frame_settings.scene_upscaler_preset == "quality"
                ? SceneTemporalQuality::Quality
                : SceneTemporalQuality::Balanced;
        std::optional<SplitViewInfo> rendered_split_info;
        VulkanSplitViewParams pending_split_view{};

        struct RenderedPanel {
            std::shared_ptr<lfs::core::Tensor> image;
            lfs::rendering::FrameMetadata metadata;
            VkImage external_image = VK_NULL_HANDLE;
            VkImageView external_image_view = VK_NULL_HANDLE;
            VkImageLayout external_image_layout = VK_IMAGE_LAYOUT_UNDEFINED;
            std::uint64_t external_image_generation = 0;
            VkImage depth_image = VK_NULL_HANDLE;
            VkImageView depth_image_view = VK_NULL_HANDLE;
            VkImageLayout depth_image_layout = VK_IMAGE_LAYOUT_UNDEFINED;
            VkFormat depth_image_format = VK_FORMAT_UNDEFINED;
            std::uint64_t depth_image_generation = 0;
            glm::ivec2 depth_image_size{0, 0};
            glm::ivec2 depth_alloc_size{0, 0};
            std::optional<TemporalFrameInput> temporal_input;
            bool flip_y = false;
            glm::ivec2 size{0, 0};
            glm::ivec2 alloc_size{0, 0};
        };

        std::vector<ViewportInteropService::FrameCompletion> point_completions;
        const auto render_native_point_cloud =
            [&](const lfs::rendering::PointCloudRenderRequest& pc_request,
                const RenderTargetId target)
            -> lfs::Result<PointSceneRenderer::RenderResult> {
            const auto fail = [](std::string message) -> lfs::Result<PointSceneRenderer::RenderResult> {
                return lfs::make_error({
                    .code = lfs::ErrorCode::Internal,
                    .domain = lfs::ErrorDomain::Rendering,
                    .detail = std::move(message),
                    .detection = LFS_SOURCE_SITE_CURRENT(),
                });
            };
            if (!vulkan_context) {
                return fail("Point-cloud viewing requires an active Vulkan context");
            }
            if ((frame_dirty & DirtyFlag::SELECTION) != 0) {
                ++point_cloud_preview_selection_revision_;
            }
            if (!point_scene_renderer_) {
                point_scene_renderer_ = createPointSceneRenderer(*context.graphics_context);
            }

            lfs::core::Tensor splat_positions;
            const lfs::core::Tensor* positions_ptr = nullptr;
            const lfs::core::Tensor* colors_ptr = nullptr;
            if (frame_settings.point_cloud_mode && has_visible_gaussian_model) {
                constexpr float SH_C0 = 0.28209479177387814f;
                const auto& sh0 = model->sh0_raw();
                const void* sh0_key = sh0.is_valid() ? sh0.ptr<float>() : nullptr;
                const std::size_t sh0_count =
                    sh0.is_valid() ? static_cast<std::size_t>(sh0.size(0)) : 0;
                if (sh0_key != point_cloud_colors_cache_key_ ||
                    sh0_count != point_cloud_colors_cache_size_ ||
                    !point_cloud_colors_cache_.is_valid()) {
                    try {
                        point_cloud_colors_cache_ =
                            (sh0.slice(1, 0, 1).squeeze(1) * SH_C0 + 0.5f).clamp(0.0f, 1.0f);
                    } catch (const std::exception& e) {
                        LOG_ERROR("Point cloud color derivation failed: {}", e.what());
                        return fail(e.what());
                    }
                    point_cloud_colors_cache_key_ = sh0_key;
                    point_cloud_colors_cache_size_ = sh0_count;
                }
                splat_positions = model->get_means();
                positions_ptr = &splat_positions;
                colors_ptr = &point_cloud_colors_cache_;
            } else {
                positions_ptr = &frame_ctx.scene_state.point_cloud->means;
                colors_ptr = &frame_ctx.scene_state.point_cloud->colors;
            }

            auto vk_req = buildPointSceneRequest(pc_request, frame_settings);
            vk_req.positions = positions_ptr;
            vk_req.colors = colors_ptr;
            vk_req.positions_revision = point_cloud_data_revision_;
            vk_req.colors_revision = point_cloud_data_revision_;
            if (frame_settings.point_cloud_mode && has_visible_gaussian_model &&
                model->has_deleted_mask() && model->deleted_mask_matches_size()) {
                vk_req.deleted_mask = &model->deleted();
                // Content revision for the soft-delete mask itself (not the
                // point-cloud positions revision). Stale wiring here caused
                // silent cache reuse after densify/compact.
                vk_req.deleted_mask_revision = model->deleted_mask_version();
            }
            vk_req.selection_revision = point_cloud_preview_selection_revision_;
            vk_req.preview_selection_revision = point_cloud_preview_selection_revision_;
            vk_req.synchronize_output = context.preparing_import;

            LOG_TIMER("renderVulkanFrame.point_cloud_vulkan");
            auto rendered = point_scene_renderer_->render(vk_req, target);
            if (!rendered) {
                return fail(rendered.error());
            }
            const auto semaphore = vulkanSceneTimeline(rendered->completion_semaphore);
            if (semaphore != VK_NULL_HANDLE && rendered->completion_value != 0) {
                const auto existing = std::find_if(point_completions.begin(), point_completions.end(),
                                                   [&](const auto& completion) { return completion.semaphore == semaphore; });
                if (existing == point_completions.end())
                    point_completions.push_back({semaphore, rendered->completion_value});
                else
                    existing->value = std::max(existing->value, rendered->completion_value);
            }
            return *rendered;
        };
        bool gt_native_point_cloud_panel = false;

        const auto ensure_viewport_image =
            [](std::shared_ptr<lfs::core::Tensor> image,
               const std::string_view label) -> std::shared_ptr<lfs::core::Tensor> {
            if (!image || !image->is_valid()) {
                return {};
            }
            if (image->device() == lfs::core::Device::GPU) {
                return image;
            }
            auto cuda_image = image->to(lfs::core::Device::GPU);
            if (!cuda_image.is_valid() || cuda_image.device() != lfs::core::Device::GPU) {
                LOG_WARN("{} produced a non-GPU tensor; using the external image path", label);
                return {};
            }
            return std::make_shared<lfs::core::Tensor>(std::move(cuda_image));
        };

        VkSemaphore latest_scene_completion_semaphore = VK_NULL_HANDLE;
        std::uint64_t latest_scene_completion_value = 0;
        bool vksplat_inputs_forced_this_frame = false;
        const auto note_lod_page_generation = [&](const std::uint64_t generation) {
            noteLodPageGeneration(generation);
        };
        const auto note_vksplat_render_progress =
            [&](const SceneRenderer::RenderResult& result) {
                if (result.lod_streaming_active) {
                    requestViewFollowUp(view_state, DirtyFlag::CAMERA);
                }
            };
        const auto render_panel_image =
            [&](const Viewport& source_viewport,
                const glm::ivec2 panel_size,
                const std::optional<SplitViewPanelId> panel_id,
                const std::optional<std::vector<bool>>& node_visibility_override,
                const lfs::core::SplatData* model_override = nullptr,
                const std::vector<glm::mat4>* model_transforms_override = nullptr,
                const std::optional<RenderTargetId> vksplat_target = std::nullopt,
                const lfs::rendering::ViewportRenderRequest* request_override = nullptr)
            -> std::expected<RenderedPanel, std::string> {
            const lfs::core::SplatData* const panel_model = model_override ? model_override : model;
            std::uint64_t panel_scene_generation =
                static_cast<std::uint64_t>(reinterpret_cast<std::uintptr_t>(panel_model));
            if (panel_model) {
                panel_scene_generation ^= static_cast<std::uint64_t>(panel_model->size()) +
                                          0x9e3779b97f4a7c15ull +
                                          (panel_scene_generation << 6) +
                                          (panel_scene_generation >> 2);
                panel_scene_generation ^= panel_model->param_layout_generation() +
                                          0x9e3779b97f4a7c15ull +
                                          (panel_scene_generation << 6) +
                                          (panel_scene_generation >> 2);
            }
            panel_scene_generation ^= view_state.temporal_scene_revision_ + 0x9e3779b97f4a7c15ull +
                                      (panel_scene_generation << 6) +
                                      (panel_scene_generation >> 2);
            const bool panel_has_visible_gaussian_model =
                hasRenderableGaussians(panel_model) &&
                (model_override != nullptr || panel_model != model || has_visible_gaussian_model);
            if (panel_size.x <= 0 || panel_size.y <= 0) {
                return std::unexpected("Invalid split-view panel size");
            }

            if ((frame_settings.point_cloud_mode || !panel_has_visible_gaussian_model) && has_point_cloud && !panel_model) {
                const std::vector<glm::mat4> point_cloud_transforms = {frame_ctx.scene_state.point_cloud_transform};
                const auto state = buildSplitViewPointCloudPanelRenderState(frame_ctx, panel_size, &source_viewport);
                lfs::rendering::PointCloudRenderRequest request{
                    .frame_view = state.frame_view,
                    .render = state.render,
                    .scene =
                        {.model_transforms = &point_cloud_transforms,
                         .transform_indices = nullptr,
                         .node_visibility_mask = {}},
                    .filters = state.filters,
                    .overlay = state.overlay,
                    .transparent_background = environmentBackgroundUsesTransparentViewerCompositing(frame_settings)};
                auto auxiliary_engine = ensure_auxiliary_rendering_engine();
                if (!auxiliary_engine) {
                    return std::unexpected(auxiliary_engine.error());
                }
                auto result = (*auxiliary_engine)->renderPointCloudImage(*frame_ctx.scene_state.point_cloud, request);
                if (!result || !result->image) {
                    return std::unexpected(result ? "Raw point-cloud panel render returned no image"
                                                  : result.error());
                }
                const bool flip_y = result->metadata.flip_y;
                return RenderedPanel{.image = std::move(result->image),
                                     .metadata = std::move(result->metadata),
                                     .flip_y = flip_y};
            }

            if (!panel_has_visible_gaussian_model) {
                return std::unexpected("No renderable model for split-view panel");
            }

            if (frame_settings.point_cloud_mode) {
                const auto state = buildSplitViewPointCloudPanelRenderState(frame_ctx, panel_size, &source_viewport);
                std::vector<glm::mat4> transforms_storage;
                auto scene = state.scene;
                if (model_transforms_override) {
                    if (model_override && !node_visibility_override) {
                        scene = {};
                    }
                    scene.model_transforms = model_transforms_override;
                } else if (!scene.model_transforms) {
                    transforms_storage = {glm::mat4(1.0f)};
                    scene.model_transforms = &transforms_storage;
                }
                if (node_visibility_override) {
                    scene.node_visibility_mask = *node_visibility_override;
                }
                lfs::rendering::PointCloudRenderRequest request{
                    .frame_view = state.frame_view,
                    .render = state.render,
                    .scene = scene,
                    .filters = state.filters,
                    .overlay = state.overlay,
                    .transparent_background = environmentBackgroundUsesTransparentViewerCompositing(frame_settings)};
                auto auxiliary_engine = ensure_auxiliary_rendering_engine();
                if (!auxiliary_engine) {
                    return std::unexpected(auxiliary_engine.error());
                }
                auto result = (*auxiliary_engine)->renderPointCloudImage(*panel_model, request);
                if (!result || !result->image) {
                    return std::unexpected(result ? "Point-cloud panel render returned no image"
                                                  : result.error());
                }
                const bool flip_y = result->metadata.flip_y;
                return RenderedPanel{.image = std::move(result->image),
                                     .metadata = std::move(result->metadata),
                                     .flip_y = flip_y};
            }

            auto request = request_override
                               ? *request_override
                               : [&] {
                                     if (frame_settings.split_view_mode == SplitViewMode::PLYComparison &&
                                         panel_id) {
                                         // Keep the projection in full-viewport coordinates while the
                                         // raster target is clipped; VkSplat subtracts this origin from
                                         // projected pixels after applying the full camera intrinsics.
                                         return buildPlyComparisonRenderRequest(
                                             frame_ctx, panel_size, source_viewport,
                                             *panel_id, render_size);
                                     }
                                     return buildViewportRenderRequest(
                                         frame_ctx, panel_size, &source_viewport, panel_id);
                                 }();
            if (model_override && scene_manager &&
                splitViewUsesPLYComparison(frame_settings.split_view_mode)) {
                const auto visible_nodes = scene_manager->getScene().getVisibleSplatNodeSlots();
                for (const auto& slot : visible_nodes) {
                    if (slot.node && slot.node->model.get() == model_override) {
                        applyPlyComparisonNodeScope(
                            request.scene,
                            request.filters,
                            request.overlay,
                            frame_ctx,
                            *slot.node,
                            static_cast<int>(slot.slot_index));
                        break;
                    }
                }
            }
            if (node_visibility_override && scene_manager && !request.scene.transform_indices) {
                request.scene.transform_indices =
                    scene_manager->getScene().peekTransformIndices();
            }
            std::vector<std::uint32_t> lod_touched_chunks;
            const bool skip_combined_lod =
                model_override != nullptr && model_override != model;
            if (!skip_combined_lod && lod_controller_ && lod_controller_->hasTree()) {
                lod_controller_->advanceTransition();
                const bool lod_transition_active = lod_controller_->transitionActive();
                if (lod_transition_active) {
                    notifyAsyncLodResultsReady();
                }
                const auto& selected = frame_settings.lod_enabled
                                           ? lod_controller_->selectedIndices()
                                           : lod_controller_->fullQualityIndices();
                if (!selected.empty()) {
                    request.lod_indices = selected.data();
                    if (frame_settings.lod_enabled && lod_transition_active) {
                        const auto& weights = lod_controller_->selectedWeights();
                        if (weights.size() == selected.size()) {
                            request.lod_weights = weights.data();
                        }
                    }
                    if (lod_controller_->pageMappingActive()) {
                        const auto& logical = frame_settings.lod_enabled
                                                  ? lod_controller_->selectedLogicalIndices()
                                                  : lod_controller_->fullQualityLogicalIndices();
                        if (logical.size() == selected.size()) {
                            request.lod_logical_indices = logical.data();
                        }
                    }
                    if (frame_settings.lod_debug_colors) {
                        const auto& levels = frame_settings.lod_enabled
                                                 ? lod_controller_->selectedLevels()
                                                 : lod_controller_->fullQualityLevels();
                        if (levels.size() == selected.size()) {
                            request.lod_levels = levels.data();
                        }
                    }
                    request.lod_count = selected.size();
                    request.lod_selection_hash = lod_controller_->selectionHash();
                    request.lod_generation = lod_controller_->statsGeneration();
                    lod_touched_chunks = lod_controller_->touchedChunks();
                    request.lod_touched_chunks = lod_touched_chunks.data();
                    request.lod_touched_chunk_count = lod_touched_chunks.size();
                    request.lod_debug_mode = frame_settings.lod_debug_colors;
                }
            }
            std::vector<glm::mat4> transforms_storage;
            if (model_transforms_override) {
                request.scene.model_transforms = model_transforms_override;
            } else if (!request.scene.model_transforms) {
                transforms_storage = {glm::mat4(1.0f)};
                request.scene.model_transforms = &transforms_storage;
            }
            if (node_visibility_override) {
                request.scene.node_visibility_mask = *node_visibility_override;
            }
            request.raster_backend =
                lfs::rendering::normalizeViewerRasterBackend(request.raster_backend, request.gut);
            request.gut = lfs::rendering::isGutBackend(request.raster_backend);

            const bool vksplat_panel_supported =
                vksplat_target.has_value() &&
                vulkan_context != nullptr &&
                lfs::rendering::isVkSplatBackend(request.raster_backend);
            if (vksplat_panel_supported) {
                if (!scene_renderer_) {
                    scene_renderer_ = createSceneRenderer(*context.graphics_context);
                }
                const bool force_input_upload =
                    (frame_dirty & DirtyFlag::SPLATS) != 0 && !vksplat_inputs_forced_this_frame;
                LOG_TIMER("scene.split_panel.render");
                auto result = scene_renderer_->render(
                    *panel_model,
                    request,
                    force_input_upload,
                    *vksplat_target,
                    synchronize_vksplat_input_upload);
                if (result) {
                    if (force_input_upload) {
                        vksplat_inputs_forced_this_frame = true;
                    }
                    note_lod_page_generation(result->lod_page_generation);
                    note_vksplat_render_progress(*result);
                    latest_scene_completion_semaphore = vulkanSceneTimeline(result->completion_semaphore);
                    latest_scene_completion_value = result->completion_value;
                    lfs::rendering::FrameMetadata metadata{};
                    metadata.valid = true;
                    metadata.flip_y = result->flip_y;
                    metadata.viewer_backend = result->viewer_backend;
                    return RenderedPanel{.image = nullptr,
                                         .metadata = std::move(metadata),
                                         .external_image = vulkanSceneImage(result->image),
                                         .external_image_view = vulkanSceneImageView(result->image_view),
                                         .external_image_layout = vulkanSceneImageLayout(result->image_layout),
                                         .external_image_generation = result->generation,
                                         .depth_image = vulkanSceneImage(result->depth_image),
                                         .depth_image_view = vulkanSceneImageView(result->depth_image_view),
                                         .depth_image_layout = vulkanSceneImageLayout(result->depth_image_layout),
                                         .depth_image_format = VK_FORMAT_R32_SFLOAT,
                                         .depth_image_generation = result->depth_generation,
                                         .depth_image_size = result->size,
                                         .depth_alloc_size = result->alloc_size,
                                         .temporal_input = temporal_eligible
                                                               ? std::optional<TemporalFrameInput>{
                                                                     {.view = frame_ctx.makeFrameView(
                                                                          source_viewport, panel_size),
                                                                      .output_extent = panel_size,
                                                                      .jitter = request.frame_view.orthographic
                                                                                    ? glm::vec2(0.0f)
                                                                                    : temporalJitterNdc(
                                                                                          applied_temporal_jitter_pixels,
                                                                                          panel_size),
                                                                      .render_scale = scale,
                                                                      .scene_generation = panel_scene_generation,
                                                                      .backend_key =
                                                                          (static_cast<std::uint64_t>(requested_upscaler) << 32) |
                                                                          (static_cast<std::uint64_t>(temporal_quality) + 1),
                                                                      .camera_cut = temporal_camera_cut}}
                                                               : std::nullopt,
                                         .flip_y = result->flip_y,
                                         .size = result->size,
                                         .alloc_size = result->alloc_size};
                }
                return std::unexpected(result.error());
            }

            if (!vulkan_context) {
                return std::unexpected("Scene split-view panel requires an active Vulkan context");
            }
            return std::unexpected("Scene comparison panel did not receive a render target");
        };

        const auto make_split_panel =
            [&](const RenderedPanel& panel,
                const float start,
                const float end,
                const bool normalize_x_to_panel) {
                const glm::ivec2 valid = panel.size;
                // Readback tensors are tightly sized. Only the external image
                // retains the renderer's allocation padding; cached frames may
                // display the tensor through the staging-upload fallback.
                const glm::ivec2 alloc =
                    panel.external_image_view != VK_NULL_HANDLE &&
                            panel.alloc_size.x > 0 && panel.alloc_size.y > 0
                        ? panel.alloc_size
                        : valid;
                return VulkanSplitViewPanel{
                    .image = panel.image,
                    .start_position = start,
                    .end_position = end,
                    .normalize_x_to_panel = normalize_x_to_panel,
                    .flip_y = panel.flip_y,
                    .external_image = panel.external_image,
                    .external_image_view = panel.external_image_view,
                    .external_image_layout = panel.external_image_layout,
                    .external_image_generation = panel.external_image_generation,
                    .depth_image = panel.depth_image,
                    .depth_image_view = panel.depth_image_view,
                    .depth_image_layout = panel.depth_image_layout,
                    .depth_image_format = panel.depth_image_format,
                    .depth_image_generation = panel.depth_image_generation,
                    .depth_image_size = panel.depth_image_size,
                    .depth_allocation_size = panel.depth_alloc_size,
                    .image_size = valid,
                    .allocation_size = alloc,
                    .temporal_input = panel.temporal_input,
                    .temporal_settings = sceneTemporalQualitySettings(temporal_quality),
                    .temporal_quality = temporal_quality,
                    .uv_scale = outputUvScale(valid, alloc),
                    .uv_clamp_max = outputUvClampMax(valid, alloc),
                    .texcoord_scale = {1.0f, 1.0f},
                    .texcoord_offset = {0.0f, 0.0f},
                };
            };
        const auto make_placeholder_panel =
            [](const glm::ivec2 size, const glm::vec3 tint) {
                lfs::rendering::FrameMetadata metadata{};
                metadata.valid = true;
                metadata.flip_y = false;
                return RenderedPanel{
                    .image = makeGTComparePlaceholderTensor(size, tint),
                    .metadata = std::move(metadata),
                    .flip_y = false,
                };
            };

        if (splitViewUsesGTComparison(frame_settings.split_view_mode) && scene_manager &&
            (has_visible_gaussian_model || has_point_cloud)) {
            auto& scene = scene_manager->getScene();
            std::shared_ptr<lfs::core::Camera> camera;
            if (frame_ctx.current_camera_id >= 0) {
                camera = scene.getCameraByUid(frame_ctx.current_camera_id);
            }
            if (scene.cameraListGeneration() != gt_camera_index_generation_ ||
                gt_camera_index_scene_ != &scene) {
                gt_camera_index_cameras_ = scene.getAllCameras();
                gt_camera_index_by_uid_.clear();
                gt_camera_index_by_uid_.reserve(gt_camera_index_cameras_.size());
                for (std::size_t index = 0; index < gt_camera_index_cameras_.size(); ++index) {
                    if (gt_camera_index_cameras_[index]) {
                        gt_camera_index_by_uid_.emplace(
                            gt_camera_index_cameras_[index]->uid(), index);
                    }
                }
                gt_camera_index_scene_ = &scene;
                gt_camera_index_generation_ = scene.cameraListGeneration();
            }
            if (!camera && !gt_camera_index_cameras_.empty()) {
                for (const auto& candidate : gt_camera_index_cameras_) {
                    if (candidate) {
                        camera = candidate;
                        break;
                    }
                }
            }

            if (camera) {
                try {
                    const GTComparisonMode gt_mode = frame_settings.gt_comparison_mode;
                    const glm::ivec2 preview_gt_size = gtComparisonPreviewSize(*camera, gt_comparison_size);
                    const int preview_max_dimension = std::max(preview_gt_size.x, preview_gt_size.y);
                    std::shared_ptr<lfs::core::Tensor> gt_image;
                    std::string gt_error;
                    bool gt_loading = false;
                    std::optional<detail::GTComparisonPixelRegion> gt_pixel_region;
                    std::optional<glm::ivec4> gt_actual_content_rect;

                    if (gtComparisonUsesRGBReference(gt_mode)) {
                        if (camera->image_path().empty() || !camera->has_image()) {
                            gt_error = "RGB GT comparison requires a source image";
                        } else {
                            const bool undistort_requested =
                                camera->camera_model_type() != lfs::core::CameraModelType::EQUIRECTANGULAR &&
                                camera->is_undistort_precomputed();
                            const bool actual_size_requested =
                                frame_settings.gt_comparison_actual_size &&
                                detail::isGTComparisonActualSizeAvailable(
                                    *camera, frame_settings.view());
                            if (actual_size_requested) {
                                const auto actual_frame = prepareGTActualFrame(view_state,
                                                                               *camera,
                                                                               current_size,
                                                                               nullptr);
                                if (actual_frame.status == GTComparisonImageStatus::Ready &&
                                    actual_frame.tile && actual_frame.tile->is_valid()) {
                                    gt_image = actual_frame.tile;
                                    gt_pixel_region = actual_frame.pixel_region;
                                    gt_actual_content_rect = actual_frame.content_rect;
                                    rendered_gt_actual_size = true;
                                    rendered_gt_actual_snapshot =
                                        actual_frame.snapshot;
                                } else if (
                                    actual_frame.status == GTComparisonImageStatus::Loading) {
                                    gt_image = actual_frame.fallback;
                                    gt_loading = !gt_image;
                                } else {
                                    gt_image = actual_frame.fallback;
                                    gt_error = actual_frame.error.empty()
                                                   ? "RGB GT comparison could not load the full-resolution source"
                                                   : actual_frame.error;
                                }
                            } else {
                                // Fit comparison loads a viewport-scaled preview. Do not publish
                                // that transient size on the shared camera; training uses image
                                // dimensions as its raster target and may run concurrently.
                                const auto lookup = getOrQueueGTComparisonImage(
                                    {.owner = view_state.id, .camera_uid = camera->uid(), .mode = gt_mode, .image_path = camera->image_path(), .preview_max_dimension = preview_max_dimension, .image_size = preview_gt_size, .undistort_requested = undistort_requested, .undistort_params = undistort_requested ? camera->undistort_params() : lfs::core::UndistortParams{}, .depth_visualization_mode = frame_settings.depth_visualization_mode, .background_color = frame_settings.background_color, .camera = camera});
                                gt_image = lookup.image;
                                if (lookup.status == GTComparisonImageStatus::Loading) {
                                    markDirty(DirtyFlag::SPLIT_VIEW, FrameReason::AsyncCompletion);
                                    if (lookup.stale_image && !lookup.grace_elapsed) {
                                        gt_image = lookup.stale_image;
                                    } else {
                                        gt_loading = true;
                                    }
                                } else if (lookup.status == GTComparisonImageStatus::Failed) {
                                    gt_error = lookup.error.empty()
                                                   ? "RGB GT comparison could not load the source image"
                                                   : lookup.error;
                                }

                                const auto current_camera_it =
                                    gt_camera_index_by_uid_.find(camera->uid());
                                if (current_camera_it != gt_camera_index_by_uid_.end() &&
                                    !gt_camera_index_cameras_.empty()) {
                                    const auto current_index = current_camera_it->second;
                                    const auto queue_neighbor = [&](const int direction) {
                                        for (std::size_t offset = 1;
                                             offset <= gt_camera_index_cameras_.size();
                                             ++offset) {
                                            const auto delta = direction * static_cast<int>(offset);
                                            const auto index = static_cast<std::size_t>(
                                                (static_cast<int>(current_index) + delta +
                                                 static_cast<int>(gt_camera_index_cameras_.size()) * 2) %
                                                static_cast<int>(gt_camera_index_cameras_.size()));
                                            const auto& neighbor = gt_camera_index_cameras_[index];
                                            if (!neighbor || neighbor->uid() == camera->uid() ||
                                                neighbor->image_path().empty()) {
                                                continue;
                                            }
                                            const glm::ivec2 neighbor_size =
                                                gtComparisonPreviewSize(*neighbor, gt_comparison_size);
                                            const bool neighbor_undistort =
                                                neighbor->camera_model_type() !=
                                                    lfs::core::CameraModelType::EQUIRECTANGULAR &&
                                                neighbor->is_undistort_precomputed();
                                            queueGTComparisonImagePrefetch(
                                                {.owner = view_state.id, .camera_uid = neighbor->uid(), .mode = gt_mode, .image_path = neighbor->image_path(), .preview_max_dimension = std::max(neighbor_size.x, neighbor_size.y), .image_size = neighbor_size, .undistort_requested = neighbor_undistort, .undistort_params = neighbor_undistort ? neighbor->undistort_params() : lfs::core::UndistortParams{}, .depth_visualization_mode = frame_settings.depth_visualization_mode, .background_color = frame_settings.background_color, .camera = neighbor});
                                            break;
                                        }
                                    };
                                    queue_neighbor(-1);
                                    queue_neighbor(1);
                                }
                            }
                        }
                    } else if (gt_mode == GTComparisonMode::Depth) {
                        if (!camera->has_depth()) {
                            gt_error = "Depth GT comparison requires a depth map for the selected camera";
                        } else {
                            const bool undistort_requested =
                                camera->camera_model_type() != lfs::core::CameraModelType::EQUIRECTANGULAR &&
                                camera->is_undistort_precomputed() && !camera->is_undistort_prepared();
                            const auto lookup = getOrQueueGTComparisonImage({.owner = view_state.id, .camera_uid = camera->uid(), .mode = gt_mode, .image_path = camera->depth_path(), .preview_max_dimension = preview_max_dimension, .image_size = preview_gt_size, .undistort_requested = undistort_requested, .undistort_params = undistort_requested ? camera->undistort_params() : lfs::core::UndistortParams{}, .depth_visualization_mode = frame_settings.depth_visualization_mode, .background_color = frame_settings.background_color, .camera = camera});
                            gt_image = lookup.image;
                            if (lookup.status == GTComparisonImageStatus::Loading) {
                                markViewDirty(context.view, DirtyFlag::SPLIT_VIEW, lfs::vis::FrameReason::SettingsChange);
                                gt_loading = true;
                                if (lookup.stale_image && !lookup.grace_elapsed) {
                                    gt_image = lookup.stale_image;
                                    gt_loading = false;
                                }
                            } else if (lookup.status == GTComparisonImageStatus::Failed) {
                                gt_error = lookup.error.empty()
                                               ? "Depth GT comparison could not load the camera depth map"
                                               : lookup.error;
                            }
                        }
                    } else {
                        if (!camera->has_normal()) {
                            gt_error = "Normal GT comparison requires a normal map for the selected camera";
                        } else {
                            const bool undistort_requested =
                                camera->camera_model_type() != lfs::core::CameraModelType::EQUIRECTANGULAR &&
                                camera->is_undistort_precomputed() && !camera->is_undistort_prepared();
                            const auto lookup = getOrQueueGTComparisonImage({.owner = view_state.id, .camera_uid = camera->uid(), .mode = gt_mode, .image_path = camera->normal_path(), .preview_max_dimension = preview_max_dimension, .image_size = preview_gt_size, .undistort_requested = undistort_requested, .undistort_params = undistort_requested ? camera->undistort_params() : lfs::core::UndistortParams{}, .camera = camera});
                            gt_image = lookup.image;
                            if (lookup.status == GTComparisonImageStatus::Loading) {
                                markViewDirty(context.view, DirtyFlag::SPLIT_VIEW, lfs::vis::FrameReason::SettingsChange);
                                gt_loading = true;
                                if (lookup.stale_image && !lookup.grace_elapsed) {
                                    gt_image = lookup.stale_image;
                                    gt_loading = false;
                                }
                            } else if (lookup.status == GTComparisonImageStatus::Failed) {
                                gt_error = lookup.error.empty()
                                               ? "Normal GT comparison could not load the camera normal map"
                                               : lookup.error;
                            }
                        }
                    }

                    if (!gt_image || !gt_image->is_valid()) {
                        if (!gt_loading) {
                            LOG_DEBUG("{}",
                                      gt_error.empty() ? "GT comparison could not load ground-truth data" : gt_error);
                        }
                        const glm::vec3 placeholder_color =
                            gt_loading ? glm::vec3(0.09f) : glm::vec3(0.16f, 0.035f, 0.050f);
                        auto& placeholder = gt_loading ? gt_comparison_loading_placeholder_
                                                       : gt_comparison_failed_placeholder_;
                        auto& placeholder_size = gt_loading ? gt_comparison_loading_placeholder_size_
                                                            : gt_comparison_failed_placeholder_size_;
                        if (!placeholder || placeholder_size != preview_gt_size) {
                            placeholder = makeGTComparePlaceholderTensor(preview_gt_size, placeholder_color);
                            placeholder_size = preview_gt_size;
                        }
                        gt_image = placeholder;
                    }

                    if (gt_image && gt_image->is_valid()) {
                        const auto gt_layout = lfs::rendering::detectImageLayout(*gt_image);
                        if (gt_layout == lfs::rendering::ImageLayout::Unknown) {
                            render_error = "GT comparison produced an unsupported ground-truth image layout";
                        } else {
                            const glm::ivec2 gt_size{
                                lfs::rendering::imageWidth(*gt_image, gt_layout),
                                lfs::rendering::imageHeight(*gt_image, gt_layout)};
                            const glm::ivec2 render_gt_size{
                                std::max(static_cast<int>(std::lround(
                                             static_cast<float>(gt_size.x) * scale /
                                             std::max(gt_comparison_scale, 1.0e-6f))),
                                         1),
                                std::max(static_cast<int>(std::lround(
                                             static_cast<float>(gt_size.y) * scale /
                                             std::max(gt_comparison_scale, 1.0e-6f))),
                                         1)};

                            const glm::ivec2 comparison_render_size =
                                gt_pixel_region ? gt_size : render_gt_size;
                            auto request = buildViewportRenderRequest(frame_ctx, comparison_render_size);
                            const glm::mat4 scene_transform =
                                detail::currentSceneTransform(scene_manager, camera->uid());
                            const auto render_camera =
                                detail::buildGTRenderCamera(
                                    *camera, comparison_render_size, scene_transform, gt_pixel_region);
                            if (render_camera) {
                                applyGTComparisonRenderCamera(
                                    request.frame_view, request.equirectangular, *render_camera);
                            }

                            RenderedPanel compare_panel{};
                            std::string compare_error;

                            if (!gtComparisonUsesRGBReference(gt_mode)) {
                                // #1574 hold-then-swap: never host-wait the GT depth ticket on the
                                // render thread. Poll any outstanding ticket; swap into the held
                                // display when Ready; submit at most one new ticket when free. The
                                // panel keeps the previous display until the new one is ready.
                                const auto clear_gt_async_ticket = [this, &view_state]() {
                                    // Detach dest under the ring lock before freeing host
                                    // storage; markFailed keeps pins until timeline reclaim.
                                    if (view_state.gt_async_depth_ticket_ != 0 && scene_renderer_) {
                                        scene_renderer_->abandonReadbackTicket(
                                            view_state.gt_async_depth_ticket_);
                                    }
                                    view_state.gt_async_depth_ticket_ = 0;
                                    view_state.gt_async_depth_dest_ = {};
                                    view_state.gt_async_ticket_mode_ = GTComparisonMode::RGB;
                                    view_state.gt_async_ticket_intrinsics_.reset();
                                    view_state.gt_async_ticket_flip_y_ = false;
                                    view_state.gt_async_ticket_metadata_ = {};
                                    view_state.gt_async_ticket_view_.reset();
                                };
                                const auto apply_gt_async_ready =
                                    [this, &frame_settings, &clear_gt_async_ticket, &view_state]()
                                    -> std::expected<void, std::string> {
                                    if (view_state.gt_async_depth_ticket_ == 0 || !scene_renderer_) {
                                        return {};
                                    }
                                    const auto polled =
                                        scene_renderer_->pollReadbackTicket(
                                            view_state.gt_async_depth_ticket_);
                                    if (!polled) {
                                        const std::string err = polled.error();
                                        clear_gt_async_ticket();
                                        return std::unexpected(err);
                                    }
                                    if (*polled == SceneRenderer::ReadbackTicketStatus::NotReady) {
                                        return {};
                                    }
                                    if (*polled != SceneRenderer::ReadbackTicketStatus::Ready) {
                                        clear_gt_async_ticket();
                                        return std::unexpected("GT depth ticket finished without Ready delivery");
                                    }
                                    auto display =
                                        view_state.gt_async_ticket_mode_ == GTComparisonMode::Depth
                                            ? makeDepthDisplayTensor(view_state.gt_async_depth_dest_, frame_settings)
                                            : (view_state.gt_async_ticket_intrinsics_.has_value()
                                                   ? makeNormalDisplayFromDepthTensor(
                                                         view_state.gt_async_depth_dest_,
                                                         *view_state.gt_async_ticket_intrinsics_)
                                                   : std::shared_ptr<lfs::core::Tensor>{});
                                    if (!display || !display->is_valid()) {
                                        clear_gt_async_ticket();
                                        return std::unexpected(view_state.gt_async_ticket_mode_ == GTComparisonMode::Depth
                                                                   ? "Depth GT comparison could not visualize rendered depth"
                                                                   : "Normal GT comparison could not derive rendered normals");
                                    }
                                    view_state.gt_async_held_display_ = std::move(display);
                                    view_state.split_right_image_generation_ =
                                        (view_state.split_right_image_generation_ + 1) |
                                        gt_comparison_detail::SPLIT_RIGHT_GENERATION_BIT;
                                    view_state.gt_async_held_flip_y_ = view_state.gt_async_ticket_flip_y_;
                                    view_state.gt_async_held_metadata_ = view_state.gt_async_ticket_metadata_;
                                    view_state.gt_async_held_view_ = view_state.gt_async_ticket_view_;
                                    clear_gt_async_ticket();
                                    return {};
                                };

                                if (!has_visible_gaussian_model || frame_settings.point_cloud_mode) {
                                    compare_error =
                                        "Normal/depth GT comparison requires a visible Gaussian model in splat mode";
                                    if (view_state.gt_async_depth_ticket_ != 0 && scene_renderer_) {
                                        (void)scene_renderer_->waitReadbackTicket(
                                            view_state.gt_async_depth_ticket_);
                                        clear_gt_async_ticket();
                                    }
                                    view_state.gt_async_held_display_.reset();
                                    view_state.gt_async_held_view_.reset();
                                } else if (!vulkan_context) {
                                    compare_error = "Normal/depth GT comparison requires an active Vulkan context";
                                } else if (!render_camera) {
                                    compare_error = "Normal/depth GT comparison could not build the GT render camera";
                                } else if (gt_mode == GTComparisonMode::Normal &&
                                           !request.frame_view.intrinsics_override.has_value()) {
                                    compare_error = "Normal GT comparison requires pinhole camera intrinsics";
                                } else {
                                    if (!scene_renderer_) {
                                        scene_renderer_ = createSceneRenderer(*context.graphics_context);
                                    }
                                    if (const auto ready = apply_gt_async_ready(); !ready) {
                                        compare_error = ready.error();
                                    }

                                    // Size/mode change while a ticket is in flight: bounded-wait it
                                    // (wait delivers into dest), then promote to held and re-submit.
                                    if (compare_error.empty() && view_state.gt_async_depth_ticket_ != 0 &&
                                        (view_state.gt_async_ticket_mode_ != gt_mode ||
                                         !view_state.gt_async_depth_dest_.is_valid() ||
                                         static_cast<int>(view_state.gt_async_depth_dest_.size(0)) != render_gt_size.y ||
                                         static_cast<int>(view_state.gt_async_depth_dest_.size(1)) != render_gt_size.x)) {
                                        if (const auto waited =
                                                scene_renderer_->waitReadbackTicket(
                                                    view_state.gt_async_depth_ticket_);
                                            !waited) {
                                            compare_error = waited.error();
                                            clear_gt_async_ticket();
                                        } else {
                                            // waitReadbackTicket already delivered into dest.
                                            const bool dest_matches_panel =
                                                view_state.gt_async_depth_dest_.is_valid() &&
                                                static_cast<int>(view_state.gt_async_depth_dest_.size(0)) == render_gt_size.y &&
                                                static_cast<int>(view_state.gt_async_depth_dest_.size(1)) == render_gt_size.x;
                                            if (dest_matches_panel) {
                                                auto display =
                                                    view_state.gt_async_ticket_mode_ == GTComparisonMode::Depth
                                                        ? makeDepthDisplayTensor(view_state.gt_async_depth_dest_, frame_settings)
                                                        : (view_state.gt_async_ticket_intrinsics_.has_value()
                                                               ? makeNormalDisplayFromDepthTensor(
                                                                     view_state.gt_async_depth_dest_,
                                                                     *view_state.gt_async_ticket_intrinsics_)
                                                               : std::shared_ptr<lfs::core::Tensor>{});
                                                if (display && display->is_valid()) {
                                                    view_state.gt_async_held_display_ = std::move(display);
                                                    view_state.split_right_image_generation_ =
                                                        (view_state.split_right_image_generation_ + 1) |
                                                        gt_comparison_detail::SPLIT_RIGHT_GENERATION_BIT;
                                                    view_state.gt_async_held_flip_y_ = view_state.gt_async_ticket_flip_y_;
                                                    view_state.gt_async_held_metadata_ = view_state.gt_async_ticket_metadata_;
                                                    view_state.gt_async_held_view_ = view_state.gt_async_ticket_view_;
                                                }
                                            } else {
                                                view_state.gt_async_held_display_.reset();
                                                view_state.gt_async_held_view_.reset();
                                            }
                                            clear_gt_async_ticket();
                                        }
                                    }

                                    if (compare_error.empty() && view_state.gt_async_depth_ticket_ == 0) {
                                        request.depth_view = false;
                                        scene_renderer_->setDepthCaptureMode(true, true);
                                        struct DepthCaptureModeGuard {
                                            SceneRenderer* renderer = nullptr;
                                            ~DepthCaptureModeGuard() {
                                                if (renderer) {
                                                    renderer->setDepthCaptureMode(false);
                                                }
                                            }
                                        } depth_capture_guard{scene_renderer_.get()};

                                        auto rendered = render_panel_image(
                                            context.viewport,
                                            render_gt_size,
                                            std::nullopt,
                                            std::nullopt,
                                            nullptr,
                                            nullptr,
                                            preview_render_target_,
                                            &request);
                                        if (!rendered) {
                                            compare_error = rendered.error();
                                        } else {
                                            view_state.gt_async_depth_dest_ = lfs::core::Tensor::empty(
                                                {static_cast<std::size_t>(render_gt_size.y),
                                                 static_cast<std::size_t>(render_gt_size.x)},
                                                lfs::core::Device::CPU,
                                                lfs::core::DataType::Float32);
                                            if (!view_state.gt_async_depth_dest_.is_valid()) {
                                                compare_error =
                                                    "Scene GT comparison failed to allocate depth destination";
                                            } else {
                                                auto ticket =
                                                    scene_renderer_->submitReadOutputDepthImageTicket(
                                                        preview_render_target_, view_state.gt_async_depth_dest_);
                                                if (!ticket) {
                                                    compare_error = ticket.error();
                                                    view_state.gt_async_depth_dest_ = {};
                                                } else {
                                                    view_state.gt_async_depth_ticket_ = *ticket;
                                                    view_state.gt_async_ticket_mode_ = gt_mode;
                                                    view_state.gt_async_ticket_intrinsics_ =
                                                        request.frame_view.intrinsics_override;
                                                    view_state.gt_async_ticket_flip_y_ = rendered->flip_y;
                                                    view_state.gt_async_ticket_metadata_ = rendered->metadata;
                                                    view_state.gt_async_ticket_view_ = GTPresentedView{*render_camera, gt_size};
                                                }
                                            }
                                        }
                                    }

                                    if (view_state.gt_async_held_display_ &&
                                        view_state.gt_async_held_display_->is_valid()) {
                                        compare_panel.image = view_state.gt_async_held_display_;
                                        compare_panel.metadata = view_state.gt_async_held_metadata_;
                                        compare_panel.flip_y = view_state.gt_async_held_flip_y_;
                                        compare_panel.external_image_view = VK_NULL_HANDLE;
                                        compare_panel.external_image_generation = 0;
                                        compare_panel.size = render_gt_size;
                                        rendered_gt_view = view_state.gt_async_held_view_;
                                    }

                                    const auto readback_stats = scene_renderer_->readbackStats();
                                    LOG_PERF(
                                        "scene.readback.gt_compare outstanding_tickets={} "
                                        "ring_full_waits={} cell_pin_waits={} has_held={}",
                                        readback_stats.outstanding,
                                        readback_stats.ring_full_waits,
                                        readback_stats.cell_pin_waits,
                                        view_state.gt_async_held_display_ &&
                                            view_state.gt_async_held_display_->is_valid());
                                }
                            } else {
                                // Leaving depth/normal: drain any outstanding async GT ticket.
                                if (view_state.gt_async_depth_ticket_ != 0 && scene_renderer_) {
                                    (void)scene_renderer_->waitReadbackTicket(
                                        view_state.gt_async_depth_ticket_);
                                    scene_renderer_->abandonReadbackTicket(
                                        view_state.gt_async_depth_ticket_);
                                    view_state.gt_async_depth_ticket_ = 0;
                                    view_state.gt_async_depth_dest_ = {};
                                }
                                view_state.gt_async_held_display_.reset();
                                view_state.gt_async_held_view_.reset();
                                const bool use_point_cloud_compare =
                                    frame_settings.point_cloud_mode || !has_visible_gaussian_model;
                                if (use_point_cloud_compare) {
                                    const std::vector<glm::mat4> raw_transforms = {
                                        frame_ctx.scene_state.point_cloud_transform};
                                    const auto& transforms = has_visible_gaussian_model
                                                                 ? frame_ctx.scene_state.model_transforms
                                                                 : raw_transforms;
                                    auto point_request = buildPointCloudRenderRequest(
                                        frame_ctx, render_gt_size, transforms);
                                    point_request.frame_view = request.frame_view;
                                    auto rendered = render_native_point_cloud(
                                        point_request, view_state.split_right_render_target_);
                                    if (rendered) {
                                        compare_panel.metadata.valid = true;
                                        compare_panel.metadata.flip_y = rendered->flip_y;
                                        compare_panel.metadata.viewer_backend = rendered->viewer_backend;
                                        compare_panel.external_image = vulkanSceneImage(rendered->image);
                                        compare_panel.external_image_view = vulkanSceneImageView(rendered->image_view);
                                        compare_panel.external_image_layout = vulkanSceneImageLayout(rendered->image_layout);
                                        compare_panel.external_image_generation = rendered->generation;
                                        compare_panel.flip_y = rendered->flip_y;
                                        compare_panel.size = rendered->size;
                                        gt_native_point_cloud_panel = true;
                                    } else {
                                        compare_error = lfs::format_for_developer(rendered.error());
                                    }
                                } else {
                                    auto rendered = render_panel_image(
                                        context.viewport,
                                        render_gt_size,
                                        std::nullopt,
                                        std::nullopt,
                                        nullptr,
                                        nullptr, view_state.split_right_render_target_,
                                        &request);
                                    if (rendered) {
                                        compare_panel = std::move(*rendered);
                                    } else {
                                        compare_error = rendered.error();
                                    }
                                }
                            }

                            if (!compare_panel.image && compare_panel.external_image_view == VK_NULL_HANDLE &&
                                !compare_error.empty() &&
                                !(gtComparisonUsesRGBReference(gt_mode) &&
                                  isRetryableSharedScratchUnavailable(compare_error))) {
                                LOG_WARN("GT comparison using placeholder rendered panel: {}", compare_error);
                                compare_panel = make_placeholder_panel(gt_size, glm::vec3(0.035f, 0.045f, 0.070f));
                                rendered_gt_view.reset();
                            }

                            if (compare_panel.image || compare_panel.external_image_view != VK_NULL_HANDLE) {
                                if (gtComparisonUsesRGBReference(gt_mode)) {
                                    lfs::core::PpispRegion appearance_region;
                                    if (gt_pixel_region) {
                                        appearance_region = {
                                            .x_offset = gt_pixel_region->origin.x,
                                            .y_offset = gt_pixel_region->origin.y,
                                            .full_width = gt_pixel_region->full_extent.x,
                                            .full_height = gt_pixel_region->full_extent.y};
                                    }
                                    bool compare_panel_ppisp_applied = false;
                                    if (!compare_panel.image &&
                                        compare_panel.external_image_view != VK_NULL_HANDLE &&
                                        frame_settings.apply_appearance_correction &&
                                        scene_renderer_ &&
                                        vulkan_context) {
                                        auto image = scene_renderer_->readOutputImage(view_state.split_right_render_target_);
                                        if (image && *image) {
                                            compare_panel.image = applyViewportAppearanceCorrection(
                                                std::move(*image),
                                                scene_manager,
                                                frame_settings,
                                                camera->uid(), appearance_region);
                                            compare_panel.image = ensure_viewport_image(
                                                std::move(compare_panel.image),
                                                "Scene GT comparison PPISP correction");
                                            if (compare_panel.image && compare_panel.image->is_valid()) {
                                                compare_panel.flip_y = compare_panel.metadata.flip_y;
                                                compare_panel.external_image_view = VK_NULL_HANDLE;
                                                compare_panel.external_image_generation = 0;
                                                compare_panel_ppisp_applied = true;
                                            }
                                        } else {
                                            LOG_WARN("Scene GT comparison PPISP readback failed: {}",
                                                     image ? "missing image payload" : image.error());
                                        }
                                    }
                                    if (compare_panel.image && !compare_panel_ppisp_applied) {
                                        compare_panel.image = applyViewportAppearanceCorrection(
                                            std::move(compare_panel.image),
                                            scene_manager,
                                            frame_settings,
                                            camera->uid(), appearance_region);
                                    }
                                }

                                if (gt_image) {
                                    const bool gt_undistorted =
                                        camera->camera_model_type() != lfs::core::CameraModelType::EQUIRECTANGULAR &&
                                        camera->is_undistort_precomputed();
                                    if (auto cuda_gt = ensureCudaGTViewportImage(view_state,
                                                                                 gt_image,
                                                                                 camera->uid(),
                                                                                 gt_size,
                                                                                 gt_undistorted,
                                                                                 "GT comparison ground-truth display")) {
                                        gt_image = std::move(cuda_gt);
                                    }
                                }
                                if (compare_panel.image) {
                                    if (auto cuda_compare = ensure_viewport_image(
                                            compare_panel.image,
                                            "GT comparison rendered display")) {
                                        compare_panel.image = std::move(cuda_compare);
                                    }
                                }

                                SplitCompositeContentRect rect =
                                    resolveSplitCompositeContentRect(render_size, true, gt_size);
                                if (rendered_gt_actual_size) {
                                    const glm::ivec4 physical_rect =
                                        gt_actual_content_rect.value_or(
                                            detail::centeredGTComparisonContentRect(
                                                current_size, gt_size));
                                    rect = {
                                        .x = physical_rect.x,
                                        .y = physical_rect.y,
                                        .width = physical_rect.z,
                                        .height = physical_rect.w};
                                }
                                pending_split_view.enabled = true;
                                pending_split_view.loss_visualization = gtComparisonShowsLoss(gt_mode);
                                pending_split_view.exact_texel_sampling =
                                    rendered_gt_actual_size;
                                pending_split_view.split_position = frame_settings.split_position;
                                pending_split_view.background = frame_settings.background_color;
                                pending_split_view.content_rect = {rect.x, rect.y, rect.width, rect.height};
                                pending_split_view.left = {std::move(gt_image), 0.0f, frame_settings.split_position, false, !rendered_gt_actual_size};
                                pending_split_view.right =
                                    make_split_panel(compare_panel, frame_settings.split_position, 1.0f, false);
                                rendered_metadata = compare_panel.metadata;
                                rendered_image_contains_ground_truth = true;
                                rendered_gt_content_size = gt_size;
                                if (render_camera && gtComparisonUsesRGBReference(gt_mode)) {
                                    // Synchronous production only (RGB and, post-#1857, Loss â€” the
                                    // predicate is upstream's own name for the synchronous set): the
                                    // panel just rendered IS the current camera's image, so publish
                                    // that pairing. Depth/Normal go
                                    // through the hold-then-swap ticket and publish the HELD camera with
                                    // the held image above â€” assigning the current camera here would pair
                                    // panel image A with camera B while a new ticket is in flight.
                                    // Degenerate GT calibration: buildGTRenderCamera returns nullopt when render_size is
                                    // degenerate or the camera's R/T tensors have no CPU pointer
                                    // (split_view_service.cpp:56-65). We publish nothing for such a frame, so the selection
                                    // lane keeps its pre-GT behaviour. Known and accepted consequence (Alex, Phase 1): in RGB
                                    // GT the compare panel is then drawn with the interactive camera at gt_size while CUDA
                                    // classifies with that same camera at the full composited size, so the two lanes disagree
                                    // on size for that frame. Disclosed in the PR text; do not "fix" this silently.
                                    rendered_gt_view = GTPresentedView{*render_camera, gt_size};
                                }
                                rendered_split_info =
                                    makeGTSplitViewInfo(gt_mode, camera->image_name());
                            } else {
                                render_error = compare_error;
                            }
                        }
                    }
                } catch (const std::exception& e) {
                    render_error = std::format("GT comparison failed: {}", e.what());
                    LOG_WARN("{}", render_error);
                }
            }
        } else if (splitViewUsesPLYComparison(frame_settings.split_view_mode) && scene_manager && has_visible_gaussian_model) {
            const auto& scene = scene_manager->getScene();
            const auto visible_nodes = scene.getVisibleSplatNodeSlots();
            const auto pair = plyComparisonPairForOffset(visible_nodes.size(), frame_settings.split_view_offset);
            if (pair) {
                const auto& left_node = visible_nodes[pair->first];
                const auto& right_node = visible_nodes[pair->second];
                const auto* const left_owned =
                    left_node.node && hasRenderableGaussians(left_node.node->model.get())
                        ? left_node.node->model.get()
                        : nullptr;
                const auto* const right_owned =
                    right_node.node && hasRenderableGaussians(right_node.node->model.get())
                        ? right_node.node->model.get()
                        : nullptr;
                const auto* const prepared_combined = scene.peekCombinedModel();
                const bool render_owned_nodes = left_owned && right_owned;
                const size_t slot_count = std::max(frame_ctx.scene_state.model_transforms.size(),
                                                   frame_ctx.scene_state.node_visibility_mask.size());
                if (!left_node.node || !right_node.node) {
                    render_error = "PLY comparison render slots are unavailable";
                } else if (!render_owned_nodes &&
                           (!hasRenderableGaussians(prepared_combined) ||
                            left_node.slot_index >= slot_count ||
                            right_node.slot_index >= slot_count)) {
                    render_error = "PLY comparison render slots are unavailable";
                } else {
                    std::vector<bool> left_mask;
                    std::vector<bool> right_mask;
                    std::vector<glm::mat4> left_transforms;
                    std::vector<glm::mat4> right_transforms;
                    const lfs::core::SplatData* left_model = nullptr;
                    const lfs::core::SplatData* right_model = nullptr;
                    const std::vector<glm::mat4>* left_transform_override = nullptr;
                    const std::vector<glm::mat4>* right_transform_override = nullptr;
                    std::optional<std::vector<bool>> left_visibility;
                    std::optional<std::vector<bool>> right_visibility;
                    if (render_owned_nodes) {
                        left_model = left_owned;
                        right_model = right_owned;
                        left_transforms = {scene_coords::nodeVisualizerWorldTransform(
                            scene, left_node.node->id)};
                        right_transforms = {scene_coords::nodeVisualizerWorldTransform(
                            scene, right_node.node->id)};
                        left_transform_override = &left_transforms;
                        right_transform_override = &right_transforms;
                    } else {
                        left_model = prepared_combined;
                        right_model = prepared_combined;
                        left_mask.assign(slot_count, false);
                        right_mask.assign(slot_count, false);
                        left_mask[left_node.slot_index] = true;
                        right_mask[right_node.slot_index] = true;
                        left_visibility = left_mask;
                        right_visibility = right_mask;
                    }

                    const auto ply_layouts = makePlyComparisonPanelLayouts(
                        render_size.x, frame_settings.split_position);
                    const auto& left_layout = ply_layouts[0];
                    const auto& right_layout = ply_layouts[1];

                    auto left = render_panel_image(
                        context.viewport,
                        {std::max(left_layout.panel.width, 1), render_size.y},
                        SplitViewPanelId::Left,
                        left_visibility,
                        left_model,
                        left_transform_override, view_state.split_left_render_target_);
                    auto right = render_panel_image(
                        context.viewport,
                        {std::max(right_layout.panel.width, 1), render_size.y},
                        SplitViewPanelId::Right,
                        right_visibility,
                        right_model,
                        right_transform_override, view_state.split_right_render_target_);
                    if (left && right) {
                        if (left->temporal_input)
                            left->temporal_input->output_extent = current_size;
                        if (right->temporal_input)
                            right->temporal_input->output_extent = current_size;
                        pending_split_view.enabled = true;
                        pending_split_view.split_position = frame_settings.split_position;
                        pending_split_view.background = frame_settings.background_color;
                        pending_split_view.content_rect = {0, 0, render_size.x, render_size.y};
                        pending_split_view.left = make_split_panel(
                            *left, left_layout.panel.start_position, left_layout.panel.end_position, false);
                        pending_split_view.right = make_split_panel(
                            *right, right_layout.panel.start_position, right_layout.panel.end_position, false);
                        pending_split_view.left.texcoord_scale = left_layout.texcoord_scale;
                        pending_split_view.left.texcoord_offset = left_layout.texcoord_offset;
                        pending_split_view.right.texcoord_scale = right_layout.texcoord_scale;
                        pending_split_view.right.texcoord_offset = right_layout.texcoord_offset;
                        rendered_metadata = makeSplitMetadata(left->metadata, right->metadata, frame_settings.split_position);
                        rendered_split_info = SplitViewInfo{
                            .enabled = true,
                            .mode_label = "Split View",
                            .detail_label = std::format("{} | {}",
                                                        left_node.node->name,
                                                        right_node.node->name),
                            .left_name = left_node.node->name,
                            .right_name = right_node.node->name};
                    } else {
                        render_error = std::format(
                            "PLY comparison panel rendering failed (left: {}; right: {})",
                            left ? "ok" : left.error(),
                            right ? "ok" : right.error());
                    }
                }
            } else {
                render_error = "PLY comparison requires at least two visible Gaussian models";
            }
        }

        // Split-view panels borrow the trainer's shared rasterizer arena while
        // training. When the trainer holds it the panel render reports a retryable
        // "shared scratch busy" error; collapsing to the full-frame fallback would
        // drop the ground-truth panel for as long as the arena stays contended.
        // Keep the last good split frame instead, mirroring the single-view path,
        // and retry next frame once the arena frees.
        if (view_state.split_view_service_.isActive(frame_settings) && !pending_split_view.enabled &&
            synchronize_vksplat_input_upload && has_cached_viewport_output &&
            isRetryableSharedScratchUnavailable(render_error)) {
            view_state.dirty_mask_.fetch_or(frame_dirty != 0 ? frame_dirty : DirtyFlag::SPLATS,
                                            std::memory_order_relaxed);
            defer_shared_scratch(render_error);
            render_lock.reset();
            LOG_DEBUG("Split-view shared scratch unavailable ({}); returning cached split image",
                      render_error);
            return cached_frame_result();
        }

        if (pending_split_view.enabled) {
            // Ground-truth comparisons deliberately preserve the reference image.
            // Other split panels use the reconstruction filter only after the
            // presentation pass confirmed that spatial reconstruction is active.
            const bool filter_reconstructed_panels =
                reconstruction_runtime_ready && !rendered_image_contains_ground_truth;
            pending_split_view.left.spatial_filter = filter_reconstructed_panels;
            pending_split_view.right.spatial_filter = filter_reconstructed_panels;
            pending_split_view.coordinate_extent =
                rendered_gt_actual_size ? current_size : render_size;
        }

        const bool render_point_cloud = frame_settings.point_cloud_mode || !has_visible_gaussian_model;

        if (rendered_image || pending_split_view.enabled) {
            // Split-view paths populate pending_split_view directly; skip the
            // full-viewport fallback that would set rendered_image to a wrong-
            // sized tensor and squash the left panel through the scene interop.
        } else if (render_point_cloud &&
                   (!has_visible_gaussian_model || hasRenderableGaussians(model)) &&
                   ((frame_settings.point_cloud_mode && has_visible_gaussian_model) || has_point_cloud)) {
            // Brush edits mutate sh0 in place â€” same tensor pointer but new
            // contents. Invalidate the derived-colors cache so the next frame
            // re-derives + re-uploads.
            if ((frame_dirty & DirtyFlag::SPLATS) != 0) {
                invalidatePointCloudData();
            }
            std::vector<glm::mat4> point_cloud_transforms_storage;
            const std::vector<glm::mat4>* transforms_for_request = nullptr;
            if (frame_settings.point_cloud_mode && has_visible_gaussian_model) {
                transforms_for_request = &frame_ctx.scene_state.model_transforms;
            } else {
                point_cloud_transforms_storage = {frame_ctx.scene_state.point_cloud_transform};
                transforms_for_request = &point_cloud_transforms_storage;
            }
            auto pc_request = buildPointCloudRenderRequest(
                frame_ctx, render_size, *transforms_for_request);
            // Vulkan-native path: skip CUDA staging, drive an external VkImage
            // straight through the same plumbing the VkSplat backend uses.
            auto try_vulkan = [&]() -> std::optional<ViewportFrameResult> {
                if (!vulkan_context) {
                    return std::nullopt;
                }
                auto render_result = render_native_point_cloud(
                    pc_request, view_state.main_render_target_);
                if (!render_result) {
                    LOG_ERROR("Point cloud Vulkan render failed: {}", lfs::format_for_developer(render_result.error()));
                    return std::nullopt;
                }

                render_lock.reset();
                view_state.vksplat_stale_frame_guard_.onSuccess();
                clearViewportImageState(view_state, render_result->size, render_result->flip_y);
                vulkanViewRenderState(view_state).external_viewport_image = vulkanSceneImage(render_result->image);
                vulkanViewRenderState(view_state).external_viewport_image_view = vulkanSceneImageView(render_result->image_view);
                vulkanViewRenderState(view_state).external_viewport_image_layout = vulkanSceneImageLayout(render_result->image_layout);
                vulkanViewRenderState(view_state).external_viewport_image_generation = render_result->generation;

                lfs::rendering::FrameMetadata metadata{};
                metadata.valid = true;
                metadata.flip_y = render_result->flip_y;
                metadata.viewer_backend = render_result->viewer_backend;
                view_state.viewport_artifact_service_.clearViewportOutput();
                view_state.viewport_artifact_service_.setLazyCapture(
                    [this, &view_state]() -> std::shared_ptr<lfs::core::Tensor> {
                        if (!point_scene_renderer_ || !last_graphics_context_) {
                            return {};
                        }
                        auto image = point_scene_renderer_->readOutputImage(view_state.main_render_target_);
                        if (!image) {
                            LOG_ERROR("Failed to capture point-cloud Vulkan viewport image: {}",
                                      image.error());
                            return {};
                        }
                        return std::move(*image);
                    },
                    metadata,
                    render_result->size);

                if (resize_result.completed) {
                    lfs::core::Tensor::trim_memory_pool();
                }
                queueCameraMetricsRefreshIfStale(context.view, scene_manager);
                view_state.viewport_interaction_context_.scene_manager = scene_manager;
                view_state.split_view_service_.updateInfo(FrameResources{});

                if (!frame_ctx.scene_state.meshes.empty() ||
                    environmentBackgroundEnabled(frame_settings)) {
                    auto mesh_frame = populateMeshFrame(frame_ctx, frame_settings, pending_split_view);
                    if (vulkanSceneImageView(render_result->depth_image_view) != VK_NULL_HANDLE) {
                        // The Vulkan point renderer exposes its hardware depth attachment
                        // (NDC z); the Metal one writes linear view depth, cleared to -1.
                        mesh_frame.depth_blit.external_image = vulkanSceneImage(render_result->depth_image);
                        mesh_frame.depth_blit.external_image_view = vulkanSceneImageView(render_result->depth_image_view);
                        mesh_frame.depth_blit.external_image_layout = vulkanSceneImageLayout(render_result->depth_image_layout);
                        mesh_frame.depth_blit.external_image_format = VK_FORMAT_R32_SFLOAT;
                        mesh_frame.depth_blit.external_image_generation = render_result->depth_generation;
                        mesh_frame.depth_blit.external_image_size = render_result->size;
                        mesh_frame.depth_blit.external_image_allocation_size =
                            render_result->size;
                        mesh_frame.depth_blit.depth_is_ndc =
                            render_result->viewer_backend != lfs::rendering::ViewerBackend::Metal;
                        if (mesh_frame.depth_blit.depth_is_ndc) {
                            const auto& view = pc_request.frame_view;
                            const auto projection = lfs::rendering::createProjectionMatrix(
                                view.size, lfs::rendering::focalLengthToVFov(view.focal_length_mm),
                                view.orthographic, view.ortho_scale, view.near_plane, view.far_plane);
                            mesh_frame.depth_blit.ndc_to_view_coeffs = {
                                projection[2][2], projection[3][2],
                                projection[2][3], projection[3][3]};
                        }
                        mesh_frame.depth_blit.flip_y = render_result->flip_y;
                        mesh_frame.depth_blit.near_plane = pc_request.frame_view.near_plane > 0.0f
                                                               ? pc_request.frame_view.near_plane
                                                               : 0.1f;
                        mesh_frame.depth_blit.far_plane = pc_request.frame_view.far_plane > 0.0f
                                                              ? pc_request.frame_view.far_plane
                                                              : 1000.0f;
                    }
                    setVulkanMeshFrame(view_state, std::move(mesh_frame));
                } else {
                    clearVulkanMeshFrame(view_state);
                }

                view_state.vulkan_viewport_coordinate_size_ = current_size;
                return ViewportFrameResult{
                    .image = publishVulkanFrameResources(view_state, {},
                                                         vulkanViewRenderState(view_state).external_viewport_image,
                                                         vulkanViewRenderState(view_state).external_viewport_image_view,
                                                         vulkanViewRenderState(view_state).external_viewport_image_layout,
                                                         vulkanViewRenderState(view_state).external_viewport_image_generation,
                                                         vulkanSceneTimeline(render_result->completion_semaphore),
                                                         render_result->completion_value),
                    .size = view_state.vulkan_viewport_image_size_,
                    .flip_y = view_state.vulkan_viewport_image_flip_y_,
                    .matches_viewport_extent = true,
                    .rendered = true};
            };
            if (auto vk_result = try_vulkan(); vk_result) {
                if (scene_manager && vk_result->matches_viewport_extent)
                    noteImportRenderFrame(scene_manager->getScene().renderGeneration(), render_error);
                return *vk_result;
            }

            if (!vulkan_context) {
                render_error = "Point-cloud viewer rendering requires an active Vulkan context";
            } else {
                render_error = "Point-cloud Vulkan render failed";
            }
        } else if (has_visible_gaussian_model && hasRenderableGaussians(model)) {
            auto request = buildViewportRenderRequest(frame_ctx, render_size, nullptr);
            request.raster_backend =
                lfs::rendering::normalizeViewerRasterBackend(request.raster_backend, request.gut);
            request.gut = lfs::rendering::isGutBackend(request.raster_backend);
            std::vector<std::uint32_t> lod_touched_chunks;
            if (lfs::rendering::isVkSplatBackend(request.raster_backend) &&
                vulkan_context &&
                !scene_renderer_) {
                scene_renderer_ = createSceneRenderer(*context.graphics_context);
            }

            prepareLodRequest(frame_settings, model, request, lod_touched_chunks);

            if (lfs::rendering::isVkSplatBackend(request.raster_backend)) {
                if (!vulkan_context) {
                    render_error = "Scene backend requires an active Vulkan context";
                } else {
                    if (!scene_renderer_) {
                        scene_renderer_ = createSceneRenderer(*context.graphics_context);
                    }
                    const auto publish_vksplat_result = [&](const SceneRenderer::RenderResult& render_result) -> ViewportFrameResult {
                        // Passive previews already advanced the refresh clock
                        // when requested. Resetting it again after a parked
                        // retry adds that delay to every subsequent interval.
                        if (is_training && (frame_dirty & ~DirtyFlag::SPLATS) != 0)
                            view_state.frame_lifecycle_service_.restartTrainingRefresh();
                        view_state.vksplat_stale_frame_guard_.onSuccess();
                        render_lock.reset();
                        note_lod_page_generation(render_result.lod_page_generation);
                        note_vksplat_render_progress(render_result);
                        const bool transparent_viewer_compositing =
                            environmentBackgroundUsesTransparentViewerCompositing(frame_settings);
                        lfs::rendering::FrameMetadata metadata{};
                        metadata.valid = true;
                        metadata.flip_y = render_result.flip_y;
                        metadata.viewer_backend = render_result.viewer_backend;
                        bool temporal_frame_published =
                            pending_split_view.enabled &&
                            pending_split_view.left.temporal_input.has_value() &&
                            pending_split_view.right.temporal_input.has_value();

                        const auto publish_mesh_frame_for_vksplat = [&]() {
                            // VkSplat returns before the shared mesh-frame setup below.
                            // Republish here so overlays and mesh/environment passes see
                            // the current camera and splat depth view_state.
                            const bool publish_mesh_frame =
                                !frame_ctx.scene_state.meshes.empty() ||
                                environmentBackgroundEnabled(frame_settings) ||
                                vulkanSceneImageView(render_result.depth_image_view) != VK_NULL_HANDLE ||
                                pending_split_view.enabled;
                            if (publish_mesh_frame) {
                                auto mesh_frame = populateMeshFrame(frame_ctx, frame_settings, pending_split_view);
                                if (vulkanSceneImageView(render_result.depth_image_view) != VK_NULL_HANDLE) {
                                    mesh_frame.depth_blit.external_image = vulkanSceneImage(render_result.depth_image);
                                    mesh_frame.depth_blit.external_image_view = vulkanSceneImageView(render_result.depth_image_view);
                                    mesh_frame.depth_blit.external_image_layout = vulkanSceneImageLayout(render_result.depth_image_layout);
                                    mesh_frame.depth_blit.external_image_format = VK_FORMAT_R32_SFLOAT;
                                    mesh_frame.depth_blit.external_image_generation = render_result.depth_generation;
                                    mesh_frame.depth_blit.external_image_size = render_result.size;
                                    mesh_frame.depth_blit.external_image_allocation_size =
                                        render_result.alloc_size;
                                    mesh_frame.depth_blit.depth_is_ndc = false;
                                    mesh_frame.depth_blit.flip_y = render_result.flip_y;
                                    mesh_frame.depth_blit.near_plane = request.frame_view.near_plane > 0.0f
                                                                           ? request.frame_view.near_plane
                                                                           : 0.1f;
                                    mesh_frame.depth_blit.far_plane = request.frame_view.far_plane > 0.0f
                                                                          ? request.frame_view.far_plane
                                                                          : 1000.0f;
                                    const glm::ivec2 depth_valid = render_result.size;
                                    const glm::ivec2 depth_alloc =
                                        render_result.alloc_size.x > 0 && render_result.alloc_size.y > 0
                                            ? render_result.alloc_size
                                            : depth_valid;
                                    mesh_frame.depth_blit.uv_scale = outputUvScale(depth_valid, depth_alloc);
                                    mesh_frame.depth_blit.uv_clamp_max =
                                        outputUvClampMax(depth_valid, depth_alloc);
                                    if (temporal_eligible) {
                                        const SceneTemporalQuality quality = temporal_quality;
                                        std::uint64_t scene_generation =
                                            static_cast<std::uint64_t>(reinterpret_cast<std::uintptr_t>(model));
                                        scene_generation ^= static_cast<std::uint64_t>(model->size()) +
                                                            0x9e3779b97f4a7c15ull +
                                                            (scene_generation << 6) +
                                                            (scene_generation >> 2);
                                        scene_generation ^= model->param_layout_generation() +
                                                            0x9e3779b97f4a7c15ull +
                                                            (scene_generation << 6) +
                                                            (scene_generation >> 2);
                                        scene_generation ^=
                                            view_state.temporal_scene_revision_ +
                                            0x9e3779b97f4a7c15ull +
                                            (scene_generation << 6) +
                                            (scene_generation >> 2);
                                        mesh_frame.temporal = VulkanMeshFrame::TemporalFrame{
                                            .input = {
                                                .view = frame_ctx.makeFrameView(),
                                                .output_extent = current_size,
                                                .jitter = request.frame_view.orthographic
                                                              ? glm::vec2(0.0f)
                                                              : temporalJitterNdc(
                                                                    applied_temporal_jitter_pixels, render_size),
                                                .render_scale = scale,
                                                .scene_generation = scene_generation,
                                                .backend_key =
                                                    (static_cast<std::uint64_t>(requested_upscaler) << 32) |
                                                    (static_cast<std::uint64_t>(quality) + 1),
                                                .camera_cut = temporal_camera_cut,
                                            },
                                            .resolve_settings = sceneTemporalQualitySettings(quality),
                                            .quality = quality,
                                        };
                                        temporal_frame_published = true;
                                    }
                                }
                                setVulkanMeshFrame(view_state, std::move(mesh_frame));
                            } else {
                                clearVulkanMeshFrame(view_state);
                            }
                        };

                        const lfs::rendering::FrameView capture_frame_view = request.frame_view;
                        const auto make_capture_composite_request =
                            [&]() -> lfs::rendering::VideoCompositeFrameRequest {
                            return {
                                .viewport =
                                    {.rotation = capture_frame_view.rotation,
                                     .translation = capture_frame_view.translation,
                                     .size = capture_frame_view.size,
                                     .focal_length_mm = capture_frame_view.focal_length_mm,
                                     .orthographic = capture_frame_view.orthographic,
                                     .ortho_scale = capture_frame_view.ortho_scale},
                                .frame_view = capture_frame_view,
                                .background_color = frame_settings.background_color,
                                .environment =
                                    {.enabled = transparent_viewer_compositing,
                                     .map_path = transparent_viewer_compositing
                                                     ? std::filesystem::path(frame_settings.environment_map_path)
                                                     : std::filesystem::path{},
                                     .exposure = frame_settings.environment_exposure,
                                     .rotation_degrees = frame_settings.environment_rotation_degrees,
                                     .equirectangular = frame_settings.equirectangular},
                            };
                        };

                        if (frame_settings.apply_appearance_correction) {
                            auto image = transparent_viewer_compositing
                                             ? scene_renderer_->readOutputImageRgba(view_state.main_render_target_)
                                             : scene_renderer_->readOutputImage(view_state.main_render_target_);
                            if (image && *image) {
                                auto corrected_image = applyViewportAppearanceCorrection(
                                    std::move(*image),
                                    scene_manager,
                                    frame_settings,
                                    frame_ctx.current_camera_id);
                                corrected_image = ensure_viewport_image(
                                    std::move(corrected_image),
                                    "Scene viewport PPISP correction");
                                if (corrected_image && corrected_image->is_valid()) {
                                    view_state.vulkan_viewport_image_ = corrected_image;
                                    ++view_state.vulkan_viewport_image_generation_;
                                    if (view_state.vulkan_viewport_image_generation_ == 0) {
                                        ++view_state.vulkan_viewport_image_generation_;
                                    }
                                    vulkanViewRenderState(view_state).external_viewport_image = VK_NULL_HANDLE;
                                    vulkanViewRenderState(view_state).external_viewport_image_view = VK_NULL_HANDLE;
                                    vulkanViewRenderState(view_state).external_viewport_image_layout = VK_IMAGE_LAYOUT_UNDEFINED;
                                    vulkanViewRenderState(view_state).external_viewport_image_generation = 0;
                                    view_state.vulkan_viewport_image_size_ = render_result.size;
                                    view_state.vulkan_viewport_image_alloc_size_ = render_result.size;
                                    view_state.vulkan_viewport_image_flip_y_ = render_result.flip_y;
                                    view_state.vulkan_gt_comparison_content_size_ = {0, 0};
                                    view_state.vulkan_gt_comparison_selection_view_.reset();
                                    view_state.viewport_artifact_service_.updateFromImageOutput(
                                        corrected_image, metadata, render_result.size, true);
                                    if (transparent_viewer_compositing) {
                                        const auto capture_composite_request = make_capture_composite_request();
                                        view_state.viewport_artifact_service_.setLazyCaptureForCurrentOutput(
                                            [this,
                                             capture_composite_request,
                                             metadata,
                                             render_size = render_result.size,
                                             corrected_capture_image = corrected_image]()
                                                -> std::shared_ptr<lfs::core::Tensor> {
                                                auto* const engine = getRenderingEngine();
                                                if (!engine) {
                                                    LOG_ERROR("Failed to composite Scene viewport capture: rendering engine unavailable");
                                                    return {};
                                                }
                                                auto materialized = engine->materializeGpuFrame(
                                                    corrected_capture_image,
                                                    metadata,
                                                    render_size);
                                                if (!materialized || !materialized->valid()) {
                                                    LOG_ERROR("Failed to materialize corrected Scene viewport capture for environment composite: {}",
                                                              materialized ? "invalid frame" : materialized.error());
                                                    return corrected_capture_image;
                                                }
                                                auto composited = engine->renderVideoCompositeFrame(
                                                    *materialized,
                                                    capture_composite_request);
                                                if (!composited) {
                                                    LOG_ERROR("Failed to composite environment into corrected Scene viewport capture: {}",
                                                              composited.error());
                                                    return corrected_capture_image;
                                                }
                                                return makeViewportCaptureImageHwc(std::move(*composited));
                                            },
                                            metadata,
                                            render_result.size);
                                    }

                                    if (resize_result.completed) {
                                        lfs::core::Tensor::trim_memory_pool();
                                    }
                                    queueCameraMetricsRefreshIfStale(context.view, scene_manager);
                                    view_state.viewport_interaction_context_.scene_manager = scene_manager;
                                    view_state.split_view_service_.updateInfo(FrameResources{});
                                    publish_mesh_frame_for_vksplat();
                                    release_inactive_split_outputs();
                                    if (temporal_frame_published)
                                        complete_temporal_convergence_frame();
                                    else if (view_state.temporal_convergence_.enabled())
                                        view_state.temporal_convergence_.cancelSettle();

                                    view_state.vulkan_viewport_coordinate_size_ = current_size;
                                    return {.image = view_state.vulkan_viewport_image_,
                                            .image_generation = view_state.vulkan_viewport_image_generation_,
                                            .size = view_state.vulkan_viewport_image_size_,
                                            .flip_y = view_state.vulkan_viewport_image_flip_y_,
                                            .matches_viewport_extent = true,
                                            .rendered = true};
                                }
                                LOG_WARN("Scene PPISP correction produced no valid viewport image; falling back to uncorrected external image");
                            } else {
                                LOG_WARN("Scene PPISP readback failed: {}",
                                         image ? "missing image payload" : image.error());
                            }
                        }

                        clearViewportImageState(view_state, render_result.size,
                                                render_result.flip_y,
                                                render_result.alloc_size);
                        vulkanViewRenderState(view_state).external_viewport_image = vulkanSceneImage(render_result.image);
                        vulkanViewRenderState(view_state).external_viewport_image_view = vulkanSceneImageView(render_result.image_view);
                        vulkanViewRenderState(view_state).external_viewport_image_layout = vulkanSceneImageLayout(render_result.image_layout);
                        vulkanViewRenderState(view_state).external_viewport_image_generation = render_result.generation;
                        const auto capture_composite_request = make_capture_composite_request();
                        view_state.viewport_artifact_service_.setLazyCapture(
                            [this, transparent_viewer_compositing, capture_composite_request, metadata, render_size = render_result.size, &view_state]()
                                -> std::shared_ptr<lfs::core::Tensor> {
                                if (!scene_renderer_ || !last_graphics_context_) {
                                    return {};
                                }
                                auto image = transparent_viewer_compositing
                                                 ? scene_renderer_->readOutputImageRgba(view_state.main_render_target_)
                                                 : scene_renderer_->readOutputImage(view_state.main_render_target_);
                                if (!image) {
                                    LOG_ERROR("Failed to capture Scene viewport image: {}", image.error());
                                    return {};
                                }
                                if (transparent_viewer_compositing) {
                                    auto* const engine = getRenderingEngine();
                                    if (!engine) {
                                        LOG_ERROR("Failed to composite Scene viewport capture: rendering engine unavailable");
                                        return {};
                                    }
                                    auto frame_image = std::move(*image);
                                    auto materialized = engine->materializeGpuFrame(
                                        frame_image,
                                        metadata,
                                        render_size);
                                    if (!materialized || !materialized->valid()) {
                                        LOG_ERROR("Failed to materialize Scene viewport capture for environment composite: {}",
                                                  materialized ? "invalid frame" : materialized.error());
                                        return frame_image;
                                    }
                                    auto composited = engine->renderVideoCompositeFrame(
                                        *materialized,
                                        capture_composite_request);
                                    if (!composited) {
                                        LOG_ERROR("Failed to composite environment into Scene viewport capture: {}",
                                                  composited.error());
                                        return frame_image;
                                    }
                                    return makeViewportCaptureImageHwc(std::move(*composited));
                                }
                                return std::move(*image);
                            },
                            metadata,
                            render_result.size);

                        if (resize_result.completed) {
                            lfs::core::Tensor::trim_memory_pool();
                        }
                        queueCameraMetricsRefreshIfStale(context.view, scene_manager);
                        view_state.viewport_interaction_context_.scene_manager = scene_manager;
                        view_state.split_view_service_.updateInfo(FrameResources{});

                        publish_mesh_frame_for_vksplat();
                        release_inactive_split_outputs();
                        if (temporal_frame_published)
                            complete_temporal_convergence_frame();
                        else if (view_state.temporal_convergence_.enabled())
                            view_state.temporal_convergence_.cancelSettle();

                        view_state.vulkan_viewport_coordinate_size_ = current_size;
                        return {.image = publishVulkanFrameResources(view_state, {},
                                                                     vulkanViewRenderState(view_state).external_viewport_image,
                                                                     vulkanViewRenderState(view_state).external_viewport_image_view,
                                                                     vulkanViewRenderState(view_state).external_viewport_image_layout,
                                                                     vulkanViewRenderState(view_state).external_viewport_image_generation,
                                                                     vulkanSceneTimeline(render_result.completion_semaphore),
                                                                     render_result.completion_value),
                                .size = view_state.vulkan_viewport_image_size_,
                                .alloc_size = view_state.vulkan_viewport_image_alloc_size_,
                                .flip_y = view_state.vulkan_viewport_image_flip_y_,
                                .matches_viewport_extent = true,
                                .rendered = true};
                    };

                    const DirtyMask non_overlay_dirty = frame_dirty & ~DirtyFlag::SELECTION;
                    // During a depth-window drag the per-move changes are CONTAINMENT-side
                    // (overlay flags from the projection pass), which this fast path cannot
                    // refresh â€” it re-rasters from cached overlay_flags. Force the full
                    // render while the drag is live so the reveal tracks the box.
                    const bool can_rerender_selection_overlay =
                        !frame_depth_window_drag_preview &&
                        (frame_dirty & DirtyFlag::SELECTION) != 0 &&
                        (frame_dirty == DirtyFlag::SELECTION ||
                         (is_training && (non_overlay_dirty & ~DirtyFlag::SPLATS) == 0)) &&
                        vulkanViewRenderState(view_state).external_viewport_image != VK_NULL_HANDLE &&
                        view_state.vulkan_viewport_image_size_ == render_size &&
                        !view_state.split_view_service_.isActive(frame_settings);
                    if (can_rerender_selection_overlay) {
                        LOG_TIMER("scene.selection_overlay");
                        std::expected<SceneRenderer::RenderResult, std::string> overlay_result =
                            std::unexpected("Scene selection overlay was not executed");
                        try {
                            overlay_result = scene_renderer_->rerenderSelectionOverlay(
                                *model,
                                request,
                                view_state.main_render_target_,
                                synchronize_vksplat_input_upload);
                        } catch (const std::exception& e) {
                            overlay_result = std::unexpected(
                                std::format("Scene selection overlay threw: {}", e.what()));
                            lfs::core::Tensor::trim_memory_pool();
                        }
                        if (overlay_result) {
                            if (is_training && non_overlay_dirty != 0) {
                                view_state.dirty_mask_.fetch_or(non_overlay_dirty, std::memory_order_relaxed);
                            }
                            return publish_vksplat_result(*overlay_result);
                        }
                        LOG_DEBUG("Scene selection overlay fast path unavailable: {}",
                                  overlay_result.error());
                    }

                    const bool force_input_upload = (frame_dirty & DirtyFlag::SPLATS) != 0;
                    LOG_TIMER("scene.render");
                    std::expected<SceneRenderer::RenderResult, std::string> render_result =
                        std::unexpected("Scene render was not executed");
                    try {
                        render_result = scene_renderer_->render(
                            *model,
                            request,
                            force_input_upload,
                            view_state.main_render_target_,
                            synchronize_vksplat_input_upload);
                    } catch (const std::exception& e) {
                        render_result = std::unexpected(std::format("Scene render threw: {}", e.what()));
                        lfs::core::Tensor::trim_memory_pool();
                    }
                    if (render_result) {
                        noteImportRenderFrame(scene_manager->getScene().renderGeneration(), render_error);
                        view_state.last_logged_vksplat_render_error_.clear();
                        return publish_vksplat_result(*render_result);
                    }
                    const bool shared_scratch_retryable =
                        isRetryableSharedScratchUnavailable(render_result.error());
                    if (synchronize_vksplat_input_upload &&
                        has_cached_viewport_output &&
                        shared_scratch_retryable) {
                        const DirtyMask retry_dirty = vksplatSharedScratchRetryDirty(frame_dirty);
                        queueSharedScratchRetry(view_state, retry_dirty);
                        const bool cached_size_matches =
                            view_state.vulkan_viewport_image_size_ == render_size;
                        if (vksplat_viewport_resize || !cached_size_matches) {
                            LOG_DEBUG("{} ({}); skipping cached viewport image, retry_dirty=0x{:x}, scene_resize={}, cached_size={}x{}, render_size={}x{}",
                                      "Scene shared scratch unavailable",
                                      render_result.error(),
                                      retry_dirty,
                                      vksplat_viewport_resize,
                                      view_state.vulkan_viewport_image_size_.x, view_state.vulkan_viewport_image_size_.y,
                                      render_size.x,
                                      render_size.y);
                            render_lock.reset();
                            return {};
                        }
                        LOG_DEBUG("{} ({}); returning cached viewport image, retry_dirty=0x{:x}",
                                  "Scene shared scratch unavailable",
                                  render_result.error(),
                                  retry_dirty);
                        defer_shared_scratch(render_result.error());
                        render_lock.reset();
                        return cached_frame_result();
                    }
                    render_error = render_result.error();
                }
            } else {
                render_error = "Gaussian viewer rendering requires a Scene backend";
            }
        } else if (has_visible_gaussian_model && render_error.empty()) {
            // PLY comparison intentionally leaves FrameContext.model empty and
            // renders the two owned node models into dedicated output slots.
            // If panel preparation fails, never fall through to the full-frame
            // renderer and dereference that absent aggregate model.
            render_error = "No full-frame Gaussian model is available for rendering";
        }

        if (rendered_image && !rendered_image_contains_ground_truth) {
            rendered_image = applyViewportAppearanceCorrection(
                std::move(rendered_image),
                scene_manager,
                frame_settings,
                frame_ctx.current_camera_id);
        }

        if (frame_ctx.scene_state.meshes.empty()) {
            clearVulkanMeshFrame(view_state);
        }

        if ((rendered_image || render_error.empty() || pending_split_view.enabled) &&
            (environmentBackgroundEnabled(frame_settings) || !frame_ctx.scene_state.meshes.empty() ||
             pending_split_view.enabled)) {
            VulkanMeshFrame gpu_mesh_frame = populateMeshFrame(frame_ctx, frame_settings, pending_split_view);

            // Splat depth -> mesh-pass z-test source. Only meaningful when the
            // active render path produced a tensor-backed depth output.
            if (rendered_image && rendered_metadata.depth_panel_count > 0 &&
                rendered_metadata.depth_panels[0].depth &&
                rendered_metadata.depth_panels[0].depth->is_valid()) {
                gpu_mesh_frame.depth_blit.depth = rendered_metadata.depth_panels[0].depth;
                gpu_mesh_frame.depth_blit.depth_is_ndc = rendered_metadata.depth_is_ndc;
                // Depth and color tensors share storage orientation; the viewport
                // pass already flips the screen quad's UVs for the color image,
                // so the depth-blit pass inherits that flip and needs no extra one.
                gpu_mesh_frame.depth_blit.flip_y = false;
                gpu_mesh_frame.depth_blit.near_plane = rendered_metadata.near_plane > 0.0f
                                                           ? rendered_metadata.near_plane
                                                           : 0.1f;
                gpu_mesh_frame.depth_blit.far_plane = rendered_metadata.far_plane > 0.0f
                                                          ? rendered_metadata.far_plane
                                                          : 1000.0f;
            }

            setVulkanMeshFrame(view_state, std::move(gpu_mesh_frame));
            if (pending_split_view.enabled &&
                pending_split_view.left.temporal_input.has_value() &&
                pending_split_view.right.temporal_input.has_value()) {
                complete_temporal_convergence_frame();
            } else if (view_state.temporal_convergence_.enabled()) {
                view_state.temporal_convergence_.cancelSettle();
            }
        }
        render_lock.reset();

        const bool has_gpu_only_pass =
            !frame_ctx.scene_state.meshes.empty() ||
            environmentBackgroundEnabled(frame_settings) ||
            pending_split_view.enabled;

        if (!rendered_image && has_gpu_only_pass) {
            if (pending_split_view.enabled || render_error.empty()) {
                view_state.vksplat_stale_frame_guard_.onSuccess();
            }
            clearViewportImageState(view_state, render_size, false);
            view_state.vulkan_gt_comparison_content_size_ =
                rendered_image_contains_ground_truth ? rendered_gt_content_size : glm::ivec2{0, 0};
            if (rendered_image_contains_ground_truth) {
                view_state.vulkan_gt_comparison_selection_view_ = rendered_gt_view;
            } else {
                view_state.vulkan_gt_comparison_selection_view_.reset();
            }
            FrameResources split_info_resources;
            if (rendered_split_info) {
                split_info_resources.split_view_executed = true;
                split_info_resources.split_info = *rendered_split_info;
            }
            view_state.split_view_service_.updateInfo(split_info_resources);

            if (pending_split_view.enabled) {
                view_state.viewport_artifact_service_.setLazyCapture(
                    [this, params = pending_split_view, current_size, gt_native_point_cloud_panel, &view_state]()
                        -> std::shared_ptr<lfs::core::Tensor> {
                        VulkanSplitViewParams capture_params = params;
                        {
                            std::lock_guard lock(vulkanViewRenderState(view_state).mesh_frame_mutex);
                            if (vulkanViewRenderState(view_state).mesh_frame.split_view.enabled) {
                                capture_params = vulkanViewRenderState(view_state).mesh_frame.split_view;
                            }
                        }
                        const auto read_panel = [this](const RenderTargetId target)
                            -> std::shared_ptr<lfs::core::Tensor> {
                            if (!scene_renderer_ || !last_graphics_context_) {
                                return {};
                            }
                            auto image = scene_renderer_->readOutputImage(
                                target);
                            if (!image) {
                                LOG_ERROR("Failed to capture Scene split-view panel: {}", image.error());
                                return {};
                            }
                            return std::move(*image);
                        };
                        if (!capture_params.left.image &&
                            capture_params.left.external_image_view != VK_NULL_HANDLE) {
                            capture_params.left.image =
                                read_panel(view_state.split_left_render_target_);
                        }
                        if (!capture_params.right.image &&
                            capture_params.right.external_image_view != VK_NULL_HANDLE) {
                            if (gt_native_point_cloud_panel && point_scene_renderer_ && last_graphics_context_) {
                                auto image = point_scene_renderer_->readOutputImage(view_state.split_right_render_target_);
                                if (image) {
                                    capture_params.right.image = std::move(*image);
                                }
                            } else {
                                capture_params.right.image =
                                    read_panel(view_state.split_right_render_target_);
                            }
                        }
                        if (!capture_params.left.image || !capture_params.right.image) {
                            return {};
                        }
                        // content_rect is authored in coordinate_extent (render_size)
                        // space; the capture tensor is current_size. Match
                        // recordScenePass: per-edge rounding into output pixels.
                        // Capture has no on-screen viewport offset.
                        if (capture_params.coordinate_extent.x > 0 &&
                            capture_params.coordinate_extent.y > 0 &&
                            capture_params.coordinate_extent != current_size) {
                            const float scale_x =
                                static_cast<float>(current_size.x) /
                                static_cast<float>(capture_params.coordinate_extent.x);
                            const float scale_y =
                                static_cast<float>(current_size.y) /
                                static_cast<float>(capture_params.coordinate_extent.y);
                            const int left = static_cast<int>(
                                std::lround(capture_params.content_rect.x * scale_x));
                            const int top = static_cast<int>(
                                std::lround(capture_params.content_rect.y * scale_y));
                            const int right = static_cast<int>(std::lround(
                                (capture_params.content_rect.x + capture_params.content_rect.z) *
                                scale_x));
                            const int bottom = static_cast<int>(std::lround(
                                (capture_params.content_rect.y + capture_params.content_rect.w) *
                                scale_y));
                            capture_params.content_rect = {
                                left, top, std::max(right - left, 1), std::max(bottom - top, 1)};
                            capture_params.coordinate_extent = current_size;
                        }
                        return composeSplitViewCpuImage(cpuSplitViewDesc(capture_params), current_size);
                    },
                    rendered_metadata,
                    current_size);
            } else {
                view_state.viewport_artifact_service_.clearViewportOutput();
            }

            // Split-view: hand both panels to gui_manager so the existing scene
            // image interop covers the left panel and a parallel slot covers the right.
            // Both arrive in the viewport pass as external Vulkan image views; the
            // CPU staging path stays as a fallback for any frame where interop fails.
            ViewportFrameResult result{};
            result.image_generation = ++view_state.split_view_image_generation_;
            if (result.image_generation == 0) {
                result.image_generation = ++view_state.split_view_image_generation_;
            }
            const bool split_uses_external_image =
                pending_split_view.enabled &&
                (pending_split_view.left.external_image_view != VK_NULL_HANDLE ||
                 pending_split_view.right.external_image_view != VK_NULL_HANDLE);
            if (split_uses_external_image) {
                result.image = publishVulkanFrameResources(view_state, result.image,
                                                           VK_NULL_HANDLE, VK_NULL_HANDLE, VK_IMAGE_LAYOUT_UNDEFINED, 0,
                                                           latest_scene_completion_semaphore, latest_scene_completion_value,
                                                           point_completions);
            }
            result.size = view_state.vulkan_viewport_image_size_;
            result.flip_y = view_state.vulkan_viewport_image_flip_y_;
            result.matches_viewport_extent = true;
            view_state.vulkan_viewport_coordinate_size_ = current_size;

            const auto tensor_size = [](const lfs::core::Tensor& t) -> glm::ivec2 {
                const auto layout = lfs::rendering::detectImageLayout(t);
                if (layout == lfs::rendering::ImageLayout::Unknown) {
                    return {0, 0};
                }
                return {lfs::rendering::imageWidth(t, layout),
                        lfs::rendering::imageHeight(t, layout)};
            };

            if (pending_split_view.enabled) {
                if (auto left_tensor = pending_split_view.left.image) {
                    const auto sz = tensor_size(*left_tensor);
                    result.split_left_image_generation =
                        gt_comparison_cuda_image_ && gt_comparison_cuda_image_.get() == left_tensor.get()
                            ? view_state.split_left_image_generation_
                            : result.image_generation;
                    result.image = std::move(left_tensor);
                    result.size = sz;
                    result.flip_y = pending_split_view.left.flip_y;
                }
                if (auto right_tensor = pending_split_view.right.image) {
                    const auto sz = tensor_size(*right_tensor);
                    gt_comparison_detail::updateSplitImageGeneration(
                        right_tensor.get(),
                        sz, view_state.gt_async_held_display_.get(),
                        view_state.split_right_source_size_,
                        result.image_generation,
                        view_state.split_right_image_generation_);
                    view_state.split_right_source_size_ = sz;
                    result.split_right_image = std::move(right_tensor);
                    result.split_right_image_generation = view_state.split_right_image_generation_;
                    result.split_right_size = sz;
                    result.split_right_flip_y = pending_split_view.right.flip_y;
                } else {
                    gt_comparison_detail::updateSplitImageGeneration(
                        nullptr,
                        {0, 0}, view_state.gt_async_held_display_.get(),
                        view_state.split_right_source_size_,
                        result.image_generation,
                        view_state.split_right_image_generation_);
                    view_state.split_right_source_size_ = {0, 0};
                    result.split_right_image_generation = view_state.split_right_image_generation_;
                }
            }
            if (!pending_split_view.enabled) {
                release_inactive_split_outputs();
            }
            if (pending_split_view.enabled && rendered_gt_actual_size &&
                rendered_gt_actual_snapshot) {
                publishGTComparisonActualFrame(view_state, *rendered_gt_actual_snapshot);
            } else {
                clearPublishedGTComparisonActualFrame(view_state);
            }
            if (scene_manager)
                noteImportRenderFrame(scene_manager->getScene().renderGeneration(), render_error);
            result.rendered = true;
            return result;
        }

        if (!rendered_image) {
            const bool shared_scratch_retryable =
                synchronize_vksplat_input_upload &&
                isRetryableSharedScratchUnavailable(render_error);
            if (shared_scratch_retryable) {
                const DirtyMask retry_dirty = vksplatSharedScratchRetryDirty(frame_dirty);
                queueSharedScratchRetry(view_state, retry_dirty);
                render_lock.reset();
                const bool cached_size_matches =
                    view_state.vulkan_viewport_image_size_ == render_size;
                if (has_cached_viewport_output && !vksplat_viewport_resize && cached_size_matches) {
                    LOG_DEBUG("{} ({}); returning cached viewport image, retry_dirty=0x{:x}",
                              "Scene shared scratch unavailable",
                              render_error,
                              retry_dirty);
                    defer_shared_scratch(render_error);
                    return cached_frame_result();
                }
                if (render_error.find("arena is busy") != std::string::npos &&
                    scene_renderer_) {
                    // Cold start and resize have no matching publication to
                    // return, but still need the same bounded priority on their
                    // next attempt.
                    rendererTrainingInterop(*scene_renderer_).requestArenaHandoff();
                }

                LOG_DEBUG("{} ({}); skipping viewport frame, retry_dirty=0x{:x}, cached_output={}, scene_resize={}, cached_size={}x{}, render_size={}x{}",
                          "Scene shared scratch unavailable",
                          render_error,
                          retry_dirty,
                          has_cached_viewport_output,
                          vksplat_viewport_resize,
                          view_state.vulkan_viewport_image_size_.x, view_state.vulkan_viewport_image_size_.y,
                          render_size.x,
                          render_size.y);
                return {};
            }

            if (has_visible_gaussian_model &&
                lfs::rendering::isVkSplatBackend(frame_settings.raster_backend)) {
                const std::string degraded_error =
                    render_error.empty() ? "missing image payload" : render_error;
                noteImportRenderFrame(scene_manager->getScene().renderGeneration(), degraded_error);
                if (view_state.last_logged_vksplat_render_error_ != degraded_error) {
                    view_state.last_logged_vksplat_render_error_ = degraded_error;
                    LOG_ERROR("Scene entered degraded mode; retaining the last good viewport image: {}",
                              degraded_error);
                }

                // A failed attempt never publishes its candidate completion
                // value. Retry next frame, but keep presenting the previous
                // image when its dimensions still match.
                const DirtyMask retry_dirty = frame_dirty != 0 ? frame_dirty : DirtyFlag::SPLATS;
                view_state.dirty_mask_.fetch_or(retry_dirty, std::memory_order_relaxed);
                render_lock.reset();
                const bool cached_size_matches =
                    view_state.vulkan_viewport_image_size_ == render_size;
                if (has_cached_viewport_output && !vksplat_viewport_resize && cached_size_matches) {
                    return cached_frame_result();
                }
                return {};
            }

            if (scene_manager)
                noteImportRenderFrame(scene_manager->getScene().renderGeneration(),
                                      render_error.empty() ? "missing image payload" : render_error);
            LOG_ERROR("Failed to render Vulkan viewport image: {}",
                      render_error.empty() ? "missing image payload" : render_error);
            clearViewportImageState(view_state);
            clearPublishedGTComparisonActualFrame(view_state);
            return {};
        }

        auto viewport_image = std::move(rendered_image);
        view_state.vksplat_stale_frame_guard_.onSuccess();
        view_state.vulkan_viewport_image_ = viewport_image;
        ++view_state.vulkan_viewport_image_generation_;
        if (view_state.vulkan_viewport_image_generation_ == 0)
            ++view_state.vulkan_viewport_image_generation_;
        vulkanViewRenderState(view_state).external_viewport_image = VK_NULL_HANDLE;
        vulkanViewRenderState(view_state).external_viewport_image_view = VK_NULL_HANDLE;
        vulkanViewRenderState(view_state).external_viewport_image_layout = VK_IMAGE_LAYOUT_UNDEFINED;
        vulkanViewRenderState(view_state).external_viewport_image_generation = 0;
        view_state.vulkan_viewport_image_size_ = render_size;
        view_state.vulkan_viewport_image_alloc_size_ = render_size;
        view_state.vulkan_viewport_coordinate_size_ = current_size;
        view_state.vulkan_viewport_image_flip_y_ = !rendered_metadata.flip_y;
        view_state.vulkan_gt_comparison_content_size_ =
            rendered_image_contains_ground_truth ? rendered_gt_content_size : glm::ivec2{0, 0};
        if (rendered_image_contains_ground_truth) {
            view_state.vulkan_gt_comparison_selection_view_ = rendered_gt_view;
        } else {
            view_state.vulkan_gt_comparison_selection_view_.reset();
        }
        view_state.viewport_artifact_service_.updateFromImageOutput(
            std::move(viewport_image), rendered_metadata, render_size, true);
        release_inactive_split_outputs();

        if (resize_result.completed) {
            lfs::core::Tensor::trim_memory_pool();
        }

        if (scene_manager)
            noteImportRenderFrame(scene_manager->getScene().renderGeneration(), render_error);
        queueCameraMetricsRefreshIfStale(context.view, scene_manager);
        view_state.viewport_interaction_context_.scene_manager = scene_manager;
        FrameResources split_info_resources;
        if (rendered_split_info) {
            split_info_resources.split_view_executed = true;
            split_info_resources.split_info = std::move(*rendered_split_info);
        }
        view_state.split_view_service_.updateInfo(split_info_resources);

        if (pending_split_view.enabled && rendered_gt_actual_size &&
            rendered_gt_actual_snapshot) {
            publishGTComparisonActualFrame(view_state, *rendered_gt_actual_snapshot);
        } else {
            clearPublishedGTComparisonActualFrame(view_state);
        }

        return {.image = view_state.vulkan_viewport_image_,
                .image_generation = view_state.vulkan_viewport_image_generation_,
                .size = view_state.vulkan_viewport_image_size_,
                .flip_y = view_state.vulkan_viewport_image_flip_y_,
                .matches_viewport_extent = true,
                .rendered = true};
    }

    void RenderingManager::publishFrameToInterop(const ViewId view_id,
                                                 const ViewportFrameResult& frame) {
        auto& view = viewState(view_id);
        auto& interop = vulkanViewRenderState(view).viewport_interop_;
        const auto resources = std::exchange(vulkanViewRenderState(view).pending_frame_resources, {});
        const VkSemaphore completion = resources ? resources->completion_semaphore : VK_NULL_HANDLE;
        const std::uint64_t completion_value = resources ? resources->completion_value : 0;
        if (resources && resources->external_image != VK_NULL_HANDLE) {
            interop.setExternalSceneImage(
                resources->external_image, resources->external_image_view,
                resources->external_image_layout, frame.size, frame.flip_y,
                resources->external_image_generation, completion, completion_value,
                frame.alloc_size);
        } else {
            interop.setSceneImage(
                frame.image, frame.size, frame.flip_y,
                frame.split_left_image_generation != 0
                    ? frame.split_left_image_generation
                    : frame.image_generation,
                completion, completion_value);
        }
        if (resources)
            interop.addFrameCompletions(resources->additional_completions);
        if (frame.split_right_image) {
            interop.setSplitRightImage(frame.split_right_image, frame.split_right_size,
                                       frame.split_right_flip_y,
                                       frame.split_right_image_generation);
        } else {
            interop.clearSplitRightImage();
        }

        const auto mesh_frame = vulkanViewRenderState(view).mesh_frame;
        if (mesh_frame.depth_blit.depth && mesh_frame.depth_blit.depth->is_valid() &&
            mesh_frame.depth_blit.depth->ndim() == 3 &&
            mesh_frame.depth_blit.depth->size(0) == 1) {
            const auto& depth = *mesh_frame.depth_blit.depth;
            interop.setDepthBlitImage(mesh_frame.depth_blit.depth,
                                      {static_cast<int>(depth.size(2)),
                                       static_cast<int>(depth.size(1))},
                                      frame.image_generation);
        } else {
            interop.clearDepthBlitImage();
        }
    }

    std::expected<void, std::string> RenderingManager::ensureVksplatTrainingSharedScratchReady(
        GraphicsContext& context,
        const lfs::core::SplatData& model,
        glm::ivec2 viewport_size) {
        const RenderSettings settings = getSettings();
        if (!lfs::rendering::isVkSplatBackend(settings.raster_backend) || model.size() <= 0) {
            return {};
        }
        if (viewport_size.x <= 0 || viewport_size.y <= 0) {
            viewport_size = getRenderedSize();
        }
        if (viewport_size.x <= 0 || viewport_size.y <= 0) {
            viewport_size = {1280, 720};
        }
        if (!scene_renderer_) {
            scene_renderer_ = createSceneRenderer(context);
        }
        if (auto ok = rendererTrainingInterop(*scene_renderer_).ensureHandshakeReady(); !ok) {
            return std::unexpected(ok.error());
        }
        return rendererTrainingInterop(*scene_renderer_).ensureTrainingSharedScratchReady(static_cast<std::size_t>(model.size()), viewport_size);
    }

    lfs::io::SplatTensorAllocator RenderingManager::makeSplatTensorAllocator() const {
        auto* const context = vulkanContextOrNull(last_graphics_context_);
        return context ? context->tensorInterop().splat_allocator(true)
                       : lfs::io::SplatTensorAllocator{};
    }

} // namespace lfs::vis

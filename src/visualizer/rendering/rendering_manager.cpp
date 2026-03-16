/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "rendering_manager.hpp"
#include "core/camera.hpp"
#include "core/cuda_debug.hpp"
#include "core/logger.hpp"
#include "core/mesh_data.hpp"
#include "core/splat_data.hpp"
#include "geometry/euclidean_transform.hpp"
#include "passes/mesh_pass.hpp"
#include "passes/overlay_pass.hpp"
#include "passes/point_cloud_pass.hpp"
#include "passes/present_pass.hpp"
#include "passes/splat_raster_pass.hpp"
#include "passes/split_view_pass.hpp"
#include "render_pass.hpp"
#include "rendering/rasterizer/rasterization/include/rasterization_api_tensor.h"
#include "rendering/rasterizer/rasterization/include/rasterization_config.h"
#include "rendering/rendering.hpp"
#include "rendering/rendering_pipeline.hpp"
#include "scene/scene_manager.hpp"
#include "scene/scene_render_state.hpp"
#include "theme/theme.hpp"
#include "training/components/ppisp.hpp"
#include "training/components/ppisp_controller.hpp"
#include "training/trainer.hpp"
#include "training/training_manager.hpp"
#include <algorithm>
#include <cuda_runtime.h>
#include <glad/glad.h>
#include <limits>
#include <shared_mutex>
#include <stdexcept>
#include <string_view>

namespace lfs::vis {

    namespace {

        void executeTimedPass(RenderPass& pass,
                              lfs::rendering::RenderingEngine& engine,
                              const FrameContext& frame_ctx,
                              FrameResources& resources,
                              const std::string_view phase = {}) {
            std::string timer_name = "RenderPass::";
            timer_name += pass.name();
            if (!phase.empty()) {
                timer_name += "[";
                timer_name += phase;
                timer_name += "]";
            }

            lfs::core::ScopedTimer timer(std::move(timer_name));
            pass.execute(engine, frame_ctx, resources);
        }

        [[nodiscard]] float linearizeDepthSample(const float depth_sample,
                                                 const float near_plane,
                                                 const float far_plane,
                                                 const bool orthographic,
                                                 const bool depth_is_ndc) {
            if (!depth_is_ndc) {
                return depth_sample < 1e9f ? depth_sample : -1.0f;
            }

            constexpr float DEPTH_BG_THRESHOLD = 0.9999f;
            if (depth_sample >= DEPTH_BG_THRESHOLD) {
                return -1.0f;
            }

            if (orthographic) {
                return near_plane + depth_sample * (far_plane - near_plane);
            }

            const float z_ndc = depth_sample * 2.0f - 1.0f;
            const float A = (far_plane + near_plane) / (far_plane - near_plane);
            const float B = (2.0f * far_plane * near_plane) / (far_plane - near_plane);
            return B / (A - z_ndc);
        }

    } // namespace

    // RenderingManager Implementation
    RenderingManager::RenderingManager() {
        passes_.push_back(std::make_unique<SplitViewPass>());
        passes_.push_back(std::make_unique<SplatRasterPass>());
        splat_raster_pass_ = static_cast<SplatRasterPass*>(passes_.back().get());
        passes_.push_back(std::make_unique<PointCloudPass>());
        point_cloud_pass_ = static_cast<PointCloudPass*>(passes_.back().get());
        passes_.push_back(std::make_unique<PresentPass>());
        passes_.push_back(std::make_unique<MeshPass>());
        passes_.push_back(std::make_unique<OverlayPass>());
        setupEventHandlers();
    }

    RenderingManager::~RenderingManager() {
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
        if (const DirtyMask dirty = frame_lifecycle_state_.setViewportResizeActive(active); dirty) {
            markDirty(dirty);
        }
    }

    void RenderingManager::updateSettings(const RenderSettings& new_settings) {
        std::lock_guard<std::mutex> lock(settings_mutex_);

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

        settings_ = new_settings;
        markDirty();
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

    void RenderingManager::advanceSplitOffset() {
        std::lock_guard<std::mutex> lock(settings_mutex_);
        split_view_state_.advanceSplitOffset(settings_);
        markDirty(DirtyFlag::SPLIT_VIEW | DirtyFlag::SPLATS);
    }

    SplitViewInfo RenderingManager::getSplitViewInfo() const {
        return split_view_state_.getInfo();
    }

    RenderingManager::ContentBounds RenderingManager::getContentBounds(const glm::ivec2& viewport_size) const {
        ContentBounds bounds{0.0f, 0.0f, static_cast<float>(viewport_size.x), static_cast<float>(viewport_size.y), false};

        if (settings_.split_view_mode == SplitViewMode::GTComparison) {
            const auto content_dims = split_view_state_.gtContentDimensions();
            if (!content_dims) {
                return bounds;
            }

            const float content_aspect = static_cast<float>(content_dims->x) / content_dims->y;
            const float viewport_aspect = static_cast<float>(viewport_size.x) / viewport_size.y;

            if (content_aspect > viewport_aspect) {
                bounds.width = static_cast<float>(viewport_size.x);
                bounds.height = viewport_size.x / content_aspect;
                bounds.x = 0.0f;
                bounds.y = (viewport_size.y - bounds.height) / 2.0f;
            } else {
                bounds.height = static_cast<float>(viewport_size.y);
                bounds.width = viewport_size.y * content_aspect;
                bounds.x = (viewport_size.x - bounds.width) / 2.0f;
                bounds.y = 0.0f;
            }
            bounds.letterboxed = true;
        }
        return bounds;
    }

    lfs::rendering::RenderingEngine* RenderingManager::getRenderingEngine() {
        if (!initialized_) {
            initialize();
        }
        return engine_.get();
    }

    std::shared_ptr<lfs::core::Tensor> RenderingManager::getViewportImageIfAvailable() const {
        return viewport_artifacts_.getCapturedImageIfCurrent();
    }

    std::shared_ptr<lfs::core::Tensor> RenderingManager::captureViewportImage() {
        if (auto image = getViewportImageIfAvailable()) {
            return image;
        }

        if (!engine_ || !viewport_artifacts_.hasGpuFrame()) {
            return {};
        }

        std::optional<std::shared_lock<std::shared_mutex>> render_lock;
        if (const auto* tm = viewport_interaction_context_.scene_manager
                                 ? viewport_interaction_context_.scene_manager->getTrainerManager()
                                 : nullptr) {
            if (const auto* trainer = tm->getTrainer()) {
                render_lock.emplace(trainer->getRenderMutex());
            }
        }

        auto readback_result = engine_->readbackGpuFrameColor(*viewport_artifacts_.gpu_frame);
        if (!readback_result) {
            LOG_ERROR("Failed to capture viewport image from GPU frame: {}", readback_result.error());
            return {};
        }

        viewport_artifacts_.storeCapturedImage(*readback_result);
        return viewport_artifacts_.captured_image;
    }

    int RenderingManager::pickCameraFrustum(const glm::vec2& mouse_pos) {
        if (!settings_.show_camera_frustums)
            return -1;

        const auto now = std::chrono::steady_clock::now();
        if (camera_interaction_state_.shouldThrottlePick(now))
            return camera_interaction_state_.hovered_camera_id;
        camera_interaction_state_.notePick(now);

        if (!engine_ || !viewport_interaction_context_.scene_manager ||
            !viewport_interaction_context_.pick_context_valid)
            return camera_interaction_state_.hovered_camera_id;

        auto cameras = viewport_interaction_context_.scene_manager->getScene().getVisibleCameras();
        if (cameras.empty())
            return -1;

        glm::mat4 scene_transform(1.0f);
        auto transforms = viewport_interaction_context_.scene_manager->getScene().getVisibleNodeTransforms();
        if (!transforms.empty())
            scene_transform = transforms[0];

        auto pick_result = engine_->pickCameraFrustum(
            cameras, mouse_pos,
            glm::vec2(viewport_interaction_context_.viewport_region.x,
                      viewport_interaction_context_.viewport_region.y),
            glm::vec2(viewport_interaction_context_.viewport_region.width,
                      viewport_interaction_context_.viewport_region.height),
            viewport_interaction_context_.viewport_data,
            settings_.camera_frustum_scale,
            scene_transform);

        int cam_id = -1;
        if (pick_result)
            cam_id = *pick_result;

        const int previous_hovered_camera = camera_interaction_state_.hovered_camera_id;
        if (camera_interaction_state_.updateHoveredCamera(cam_id)) {
            LOG_DEBUG("Camera hover changed: {} -> {}", previous_hovered_camera, cam_id);
            markDirty(DirtyFlag::OVERLAY);
        }

        return camera_interaction_state_.hovered_camera_id;
    }

    bool RenderingManager::renderPreviewFrame(SceneManager* const scene_manager,
                                              const glm::mat3& rotation,
                                              const glm::vec3& position,
                                              const float focal_length_mm,
                                              const unsigned int fbo,
                                              [[maybe_unused]] const unsigned int texture,
                                              const int width, const int height) {
        if (!initialized_ || !engine_)
            return false;

        const auto* const model = scene_manager ? scene_manager->getModelForRendering() : nullptr;
        if (!model || model->size() == 0)
            return false;

        glBindFramebuffer(GL_FRAMEBUFFER, fbo);
        glViewport(0, 0, width, height);
        const auto& bg = settings_.background_color;
        glClearColor(bg.r, bg.g, bg.b, 1.0f);
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

        const lfs::rendering::ViewportRenderRequest request{
            .frame_view =
                {.rotation = rotation,
                 .translation = position,
                 .size = {width, height},
                 .focal_length_mm = focal_length_mm,
                 .background_color = bg},
            .scaling_modifier = settings_.scaling_modifier,
            .antialiasing = false,
            .sh_degree = 0,
            .point_cloud_mode = settings_.point_cloud_mode,
            .voxel_size = settings_.voxel_size,
            .gut = settings_.gut,
            .equirectangular = settings_.equirectangular,
            .show_rings = false,
            .ring_width = 0.0f,
            .show_center_markers = false,
            .transform_indices = nullptr,
            .selection_mask = nullptr,
            .brush = {},
            .crop_box = std::nullopt,
            .ellipsoid = std::nullopt,
            .depth_filter = std::nullopt,
            .selected_node_mask = {},
            .node_visibility_mask = {}};

        if (const auto result = engine_->renderGaussiansViewportFrame(*model, request)) {
            engine_->presentGpuFrame(result->frame, {0, 0}, {width, height});
            glBindFramebuffer(GL_FRAMEBUFFER, 0);
            return true;
        }

        glBindFramebuffer(GL_FRAMEBUFFER, 0);
        return false;
    }

    void RenderingManager::renderFrame(const RenderContext& context, SceneManager* scene_manager) {
        if (!initialized_) {
            initialize();
        }

        if (scene_manager && (dirty_mask_.load(std::memory_order_relaxed) & DirtyFlag::SELECTION)) {
            for (const auto& group : scene_manager->getScene().getSelectionGroups()) {
                lfs::rendering::config::setSelectionGroupColor(
                    group.id, make_float3(group.color.x, group.color.y, group.color.z));
            }
        }

        // Calculate current render size
        glm::ivec2 current_size = context.viewport.windowSize;
        if (context.viewport_region) {
            current_size = glm::ivec2(
                static_cast<int>(context.viewport_region->width),
                static_cast<int>(context.viewport_region->height));
        }

        // SAFETY CHECK: Don't render with invalid viewport dimensions
        if (current_size.x <= 0 || current_size.y <= 0) {
            LOG_TRACE("Skipping render - invalid viewport size: {}x{}", current_size.x, current_size.y);
            const auto& shell_bg = theme().menu_background();
            glClearColor(shell_bg.x, shell_bg.y, shell_bg.z, 1.0f);
            glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
            return;
        }

        const auto resize_result = frame_lifecycle_state_.handleViewportResize(current_size);
        if (resize_result.dirty) {
            markDirty(resize_result.dirty);
        }
        const bool resize_completed = resize_result.completed;

        const lfs::core::SplatData* const model = scene_manager ? scene_manager->getModelForRendering() : nullptr;
        const auto* const visible_point_cloud =
            (scene_manager && !model) ? scene_manager->getScene().getVisiblePointCloud() : nullptr;
        const bool has_visible_point_cloud = visible_point_cloud && visible_point_cloud->size() > 0;
        const size_t model_ptr = reinterpret_cast<size_t>(model);

        if (const auto model_change = frame_lifecycle_state_.handleModelChange(model_ptr, viewport_artifacts_);
            model_change.changed) {
            LOG_DEBUG("Model ptr changed: {} -> {}, size={}",
                      model_change.previous_model_ptr, model_ptr, model ? model->size() : 0);
            markDirty(DirtyFlag::ALL);
        }

        const bool is_training = scene_manager && scene_manager->hasDataset() &&
                                 scene_manager->getTrainerManager() &&
                                 scene_manager->getTrainerManager()->isRunning();

        if (const DirtyMask training_dirty = frame_lifecycle_state_.handleTrainingRefresh(
                is_training,
                framerate_controller_.getSettings().training_frame_refresh_time_sec);
            training_dirty) {
            markDirty(training_dirty);
        }

        bool request_gt_prerender = false;
        split_view_state_.prepareGTComparisonContext(
            scene_manager,
            settings_,
            camera_interaction_state_.current_camera_id,
            model || has_visible_point_cloud,
            viewport_artifacts_.hasGpuFrame(),
            gt_texture_cache_,
            request_gt_prerender);
        if (request_gt_prerender) {
            dirty_mask_.fetch_or(DirtyFlag::SPLATS, std::memory_order_relaxed);
        }

        if (const DirtyMask required_dirty = frame_lifecycle_state_.requiredDirtyMask(
                viewport_artifacts_.hasViewportOutput(),
                model || has_visible_point_cloud,
                settings_.split_view_mode);
            required_dirty) {
            dirty_mask_.fetch_or(required_dirty, std::memory_order_relaxed);
        }

        glViewport(0, 0, context.viewport.frameBufferSize.x, context.viewport.frameBufferSize.y);

        const auto& shell_bg = theme().menu_background();
        glClearColor(shell_bg.x, shell_bg.y, shell_bg.z, 1.0f);
        glClear(GL_COLOR_BUFFER_BIT);

        if (context.viewport_region) {
            const GLint x = static_cast<GLint>(context.viewport_region->x);
            const GLint y = context.viewport.frameBufferSize.y - static_cast<GLint>(context.viewport_region->y + context.viewport_region->height);
            const GLsizei w = static_cast<GLsizei>(context.viewport_region->width);
            const GLsizei h = static_cast<GLsizei>(context.viewport_region->height);
            glViewport(x, y, w, h);
            glScissor(x, y, w, h);
            glEnable(GL_SCISSOR_TEST);
        }

        doFullRender(context, scene_manager, model);

        if (resize_completed) {
            frame_lifecycle_state_.noteResizeCompleted();
            lfs::core::Tensor::trim_memory_pool();
        }

        if (context.viewport_region) {
            glDisable(GL_SCISSOR_TEST);
        }

        viewport_interaction_context_.scene_manager = scene_manager;
        viewport_interaction_context_.updatePickContext(context.viewport_region,
                                                       viewport_interaction_context_.viewport_data);
    }

    void RenderingManager::doFullRender(const RenderContext& context, SceneManager* scene_manager,
                                        const lfs::core::SplatData* model) {
        LOG_TIMER_TRACE("RenderingManager::doFullRender");

        render_count_++;
        LOG_TRACE("Render #{}", render_count_);

        glm::ivec2 render_size = context.viewport.windowSize;
        glm::ivec2 viewport_pos(0, 0);
        if (context.viewport_region) {
            render_size = glm::ivec2(
                static_cast<int>(context.viewport_region->width),
                static_cast<int>(context.viewport_region->height));
            const int gl_y = context.viewport.frameBufferSize.y -
                             static_cast<int>(context.viewport_region->y) -
                             static_cast<int>(context.viewport_region->height);
            viewport_pos = glm::ivec2(static_cast<int>(context.viewport_region->x), gl_y);
        }

        glClearColor(settings_.background_color.r, settings_.background_color.g,
                     settings_.background_color.b, 1.0f);
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

        const DirtyMask frame_dirty = dirty_mask_.exchange(0);
        const bool count_frame = frame_dirty != 0;
        if (count_frame) {
            framerate_controller_.beginFrame();
        }

        SceneRenderState scene_state;
        if (scene_manager) {
            scene_state = scene_manager->buildRenderState();
        }

        const bool has_splats = model && model->size() > 0;
        const bool has_point_cloud = scene_state.point_cloud && scene_state.point_cloud->size() > 0;
        if (!has_point_cloud && point_cloud_pass_) {
            point_cloud_pass_->resetCache();
        }

        viewport_interaction_context_.viewport_data = lfs::rendering::ViewportData{
            .rotation = context.viewport.getRotationMatrix(),
            .translation = context.viewport.getTranslation(),
            .size = render_size,
            .focal_length_mm = settings_.focal_length_mm,
            .orthographic = settings_.orthographic,
            .ortho_scale = settings_.ortho_scale};

        const FrameContext frame_ctx{
            .viewport = context.viewport,
            .viewport_region = context.viewport_region,
            .scene_manager = scene_manager,
            .model = model,
            .scene_state = std::move(scene_state),
            .settings = settings_,
            .render_size = render_size,
            .viewport_pos = viewport_pos,
            .frame_dirty = frame_dirty,
            .brush = interaction_state_.brush,
            .gizmo = gizmo_state_.makeFrameState(),
            .hovered_camera_id = camera_interaction_state_.hovered_camera_id,
            .current_camera_id = camera_interaction_state_.current_camera_id,
            .hovered_gaussian_id = interaction_state_.hovered_gaussian_id,
            .selection_flash_intensity = getSelectionFlashIntensity()};

        FrameResources resources{
            .cached_metadata = viewport_artifacts_.metadata,
            .cached_gpu_frame = viewport_artifacts_.gpu_frame,
            .cached_result_size = viewport_artifacts_.rendered_size,
            .gt_context = split_view_state_.gt_context,
            .hovered_gaussian_id = interaction_state_.hovered_gaussian_id,
            .split_info = {},
            .additional_dirty = 0,
            .pivot_animation_end = std::nullopt};

        if (!has_splats && !has_point_cloud) {
            const bool had_cached_output = viewport_artifacts_.hasOutputArtifacts();
            if (had_cached_output) {
                resources.cached_metadata = {};
                resources.cached_gpu_frame.reset();
                resources.cached_result_size = {0, 0};
                resources.hovered_gaussian_id = -1;
                lfs::core::Tensor::trim_memory_pool();
            }
        }

        if (frame_ctx.settings.split_view_mode == SplitViewMode::GTComparison &&
            resources.gt_context && resources.gt_context->valid()) {
            const bool needs_gt_pre_render =
                !(resources.cached_gpu_frame && resources.cached_gpu_frame->valid()) ||
                (has_splats && (frame_dirty & splat_raster_pass_->sensitivity())) ||
                (has_point_cloud && point_cloud_pass_ && (frame_dirty & point_cloud_pass_->sensitivity()));

            if (needs_gt_pre_render) {
                if (has_splats) {
                    executeTimedPass(*splat_raster_pass_, *engine_, frame_ctx, resources, "gt_pre");
                    resources.splat_pre_rendered = true;
                } else if (has_point_cloud && point_cloud_pass_) {
                    executeTimedPass(*point_cloud_pass_, *engine_, frame_ctx, resources, "gt_pre");
                    resources.splat_pre_rendered = true;
                }
            }
        }

        for (auto& pass : passes_) {
            if (pass->shouldExecute(frame_dirty, frame_ctx)) {
                executeTimedPass(*pass, *engine_, frame_ctx, resources);
            }
        }

        // Apply pass-produced side effects
        if (resources.additional_dirty)
            markDirty(resources.additional_dirty);
        if (resources.pivot_animation_end)
            setPivotAnimationEndTime(*resources.pivot_animation_end);

        // Write-back from FrameResources to manager state
        const bool viewport_output_updated =
            (frame_dirty & (DirtyFlag::SPLATS | DirtyFlag::CAMERA | DirtyFlag::VIEWPORT |
                            DirtyFlag::SELECTION | DirtyFlag::BACKGROUND | DirtyFlag::PPISP |
                            DirtyFlag::SPLIT_VIEW)) != 0;
        viewport_artifacts_.updateFromFrameResources(resources, viewport_output_updated);
        interaction_state_.hovered_gaussian_id = resources.hovered_gaussian_id;

        split_view_state_.updateInfo(resources);

        if (count_frame) {
            framerate_controller_.endFrame();
        }
    }

    float RenderingManager::getDepthAtPixel(int x, int y) const {
        const auto& cached_metadata = viewport_artifacts_.metadata;
        const auto& cached_gpu_frame = viewport_artifacts_.gpu_frame;

        int viewport_width = viewport_artifacts_.rendered_size.x;
        int viewport_height = viewport_artifacts_.rendered_size.y;
        if (viewport_width <= 0 || viewport_height <= 0) {
            viewport_width = frame_lifecycle_state_.last_viewport_size.x;
            viewport_height = frame_lifecycle_state_.last_viewport_size.y;
            if (viewport_width <= 0 || viewport_height <= 0)
                return -1.0f;
        }

        float splat_depth = -1.0f;

        const float active_near_plane =
            (cached_gpu_frame && cached_gpu_frame->valid()) ? cached_gpu_frame->near_plane
                                                            : (cached_metadata.valid ? cached_metadata.near_plane
                                                                                     : lfs::rendering::DEFAULT_NEAR_PLANE);
        const float active_far_plane =
            (cached_gpu_frame && cached_gpu_frame->valid()) ? cached_gpu_frame->far_plane
                                                            : (cached_metadata.valid ? cached_metadata.far_plane
                                                                                     : lfs::rendering::DEFAULT_FAR_PLANE);
        const bool active_orthographic =
            (cached_gpu_frame && cached_gpu_frame->valid()) ? cached_gpu_frame->orthographic
                                                            : cached_metadata.orthographic;

        if (cached_metadata.valid) {
            const lfs::core::Tensor* depth_ptr = nullptr;

            if (cached_metadata.split_position > 0.0f &&
                cached_metadata.depth && cached_metadata.depth->is_valid()) {
                const float normalized_x = static_cast<float>(x) / static_cast<float>(viewport_width);

                if (normalized_x >= cached_metadata.split_position &&
                    cached_metadata.depth_right && cached_metadata.depth_right->is_valid()) {
                    depth_ptr = cached_metadata.depth_right.get();
                } else {
                    depth_ptr = cached_metadata.depth.get();
                }
            } else if (cached_metadata.depth && cached_metadata.depth->is_valid()) {
                depth_ptr = cached_metadata.depth.get();
            }

            if (depth_ptr && depth_ptr->ndim() == 3) {
                const int depth_height = static_cast<int>(depth_ptr->size(1));
                const int depth_width = static_cast<int>(depth_ptr->size(2));

                int scaled_x = x;
                int scaled_y = y;
                if (depth_width != viewport_width || depth_height != viewport_height) {
                    scaled_x = static_cast<int>(static_cast<float>(x) * depth_width / viewport_width);
                    scaled_y = static_cast<int>(static_cast<float>(y) * depth_height / viewport_height);
                }

                if (scaled_x >= 0 && scaled_x < depth_width && scaled_y >= 0 && scaled_y < depth_height) {
                    float d;
                    const float* gpu_ptr = depth_ptr->ptr<float>() + scaled_y * depth_width + scaled_x;
                    CHECK_CUDA(cudaMemcpy(&d, gpu_ptr, sizeof(float), cudaMemcpyDeviceToHost));
                    splat_depth = linearizeDepthSample(
                        d, active_near_plane, active_far_plane, active_orthographic, cached_metadata.depth_is_ndc);
                }
            }
        }

        if (splat_depth <= 0.0f && cached_gpu_frame && cached_gpu_frame->valid() && cached_gpu_frame->depth.valid()) {
            splat_depth = gpu_depth_readback_state_.readLinearDepth(
                *cached_gpu_frame, x, y, viewport_height);
        }

        float mesh_depth = -1.0f;
        if (engine_ && engine_->hasMeshRender()) {
            const GLuint mesh_fbo = engine_->getMeshFramebuffer();
            if (mesh_fbo != 0 && x >= 0 && x < viewport_width && y >= 0 && y < viewport_height) {
                float ndc_depth = 1.0f;
                glBindFramebuffer(GL_READ_FRAMEBUFFER, mesh_fbo);
                glReadPixels(x, viewport_height - 1 - y, 1, 1,
                             GL_DEPTH_COMPONENT, GL_FLOAT, &ndc_depth);
                glBindFramebuffer(GL_READ_FRAMEBUFFER, 0);

                constexpr float DEPTH_BG_THRESHOLD = 0.9999f;
                if (ndc_depth < DEPTH_BG_THRESHOLD) {
                    mesh_depth = linearizeDepthSample(
                        ndc_depth, active_near_plane, active_far_plane, active_orthographic, true);
                }
            }
        }

        if (splat_depth > 0.0f && mesh_depth > 0.0f) {
            return std::min(splat_depth, mesh_depth);
        }
        if (splat_depth > 0.0f) {
            return splat_depth;
        }
        if (mesh_depth > 0.0f) {
            return mesh_depth;
        }
        return -1.0f;
    }

    void RenderingManager::setBrushState(const bool active, const float x, const float y, const float radius,
                                         const bool add_mode, lfs::core::Tensor* selection_tensor,
                                         const bool saturation_mode, const float saturation_amount) {
        interaction_state_.setBrush(active, x, y, radius, add_mode, selection_tensor,
                                    saturation_mode, saturation_amount);
        markDirty(DirtyFlag::SELECTION);
    }

    void RenderingManager::clearBrushState() {
        interaction_state_.clearBrush();
        markDirty(DirtyFlag::SELECTION);
    }

    void RenderingManager::setRectPreview(float x0, float y0, float x1, float y1, bool add_mode) {
        interaction_state_.setRect(x0, y0, x1, y1, add_mode);
    }

    void RenderingManager::clearRectPreview() {
        interaction_state_.clearRect();
    }

    void RenderingManager::setPolygonPreview(const std::vector<std::pair<float, float>>& points, bool closed, bool add_mode) {
        interaction_state_.setPolygon(points, closed, add_mode);
    }

    void RenderingManager::setPolygonPreviewWorldSpace(const std::vector<glm::vec3>& world_points,
                                                       const bool closed, const bool add_mode) {
        interaction_state_.setPolygonWorldSpace(world_points, closed, add_mode);
    }

    void RenderingManager::clearPolygonPreview() {
        interaction_state_.clearPolygon();
    }

    void RenderingManager::setLassoPreview(const std::vector<std::pair<float, float>>& points, bool add_mode) {
        interaction_state_.setLasso(points, add_mode);
    }

    void RenderingManager::clearLassoPreview() {
        interaction_state_.clearLasso();
    }

    void RenderingManager::clearSelectionPreviews() {
        interaction_state_.clearSelectionPreviews();
        markDirty(DirtyFlag::SELECTION);
    }

} // namespace lfs::vis

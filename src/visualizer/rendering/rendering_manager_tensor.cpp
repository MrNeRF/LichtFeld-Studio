/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "rendering_manager.hpp"

#include "core/logger.hpp"
#include "core/training_manager.hpp"
#include "rendering/model_renderability.hpp"
#include "scene/scene_manager.hpp"
#include "scene_renderer_factory.hpp"
#include "viewport_request_builder.hpp"
#include "window/graphics_context.hpp"

#include <limits>

namespace lfs::vis {
    namespace {
        // Same parameters as the Vulkan manager's mesh frame.
        ViewportEnvironment environmentFor(const FrameContext& frame_ctx, const RenderSettings& settings) {
            ViewportEnvironment environment;
            environment.enabled = environmentBackgroundUsesTransparentViewerCompositing(settings);
            if (!environment.enabled)
                return environment;
            const auto vp_data = frame_ctx.makeViewportData();
            const auto frame_view = frame_ctx.makeFrameView();
            environment.map_path = settings.environment_map_path;
            environment.camera_to_world = vp_data.rotation;
            environment.viewport_size = glm::vec2(static_cast<float>(frame_view.size.x),
                                                  static_cast<float>(frame_view.size.y));
            if (frame_view.intrinsics_override.has_value() && !frame_view.orthographic) {
                const auto& intr = *frame_view.intrinsics_override;
                environment.intrinsics = glm::vec4(intr.focal_x, intr.focal_y, intr.center_x, intr.center_y);
            } else {
                const auto [fx, fy] =
                    lfs::rendering::computePixelFocalLengths(frame_view.size, frame_view.focal_length_mm);
                environment.intrinsics = glm::vec4(fx, fy, frame_view.size.x * 0.5f, frame_view.size.y * 0.5f);
            }
            environment.exposure = settings.environment_exposure;
            environment.rotation_radians = glm::radians(settings.environment_rotation_degrees);
            environment.equirectangular_view = settings.equirectangular;
            return environment;
        }
        // Same payload as the Vulkan manager's populateMeshFrame.
        ViewportMeshPassDesc meshesFor(const FrameContext& frame_ctx, const RenderSettings& settings) {
            ViewportMeshPassDesc frame;
            const auto& meshes = frame_ctx.scene_state.meshes;
            if (meshes.empty())
                return frame;
            const auto vp_data = frame_ctx.makeViewportData();
            frame.view_projection = vp_data.getProjectionMatrix() * vp_data.getViewMatrix();
            frame.camera_position = vp_data.translation;
            const bool any_selected_mesh = std::any_of(meshes.begin(), meshes.end(),
                                                       [](const auto& mesh) { return mesh.is_selected; });
            const auto& selected_nodes = frame_ctx.scene_state.selected_node_mask;
            const bool any_selected_node = std::any_of(selected_nodes.begin(), selected_nodes.end(),
                                                       [](const bool selected) { return selected; });
            const bool dim_non_emphasized = settings.desaturate_unselected && (any_selected_mesh || any_selected_node);
            const glm::vec3 headlight_dir = glm::length(vp_data.translation) > 1e-6f
                                                ? glm::normalize(vp_data.translation)
                                                : settings.mesh_light_dir;
            frame.items.reserve(meshes.size());
            for (const auto& mesh : meshes) {
                if (!mesh.mesh)
                    continue;
                frame.items.push_back({
                    .mesh = mesh.mesh,
                    .model = mesh.transform,
                    .light_dir = headlight_dir,
                    .light_intensity = settings.mesh_light_intensity,
                    .ambient = settings.mesh_ambient,
                    .backface_culling = settings.mesh_backface_culling,
                    .is_emphasized = mesh.is_selected,
                    .dim_non_emphasized = dim_non_emphasized,
                    .flash_intensity = frame_ctx.selection_flash_intensity,
                    .wireframe_overlay = settings.mesh_wireframe,
                    .wireframe_color = settings.mesh_wireframe_color,
                    .wireframe_width = settings.mesh_wireframe_width,
                    .shadow_enabled = settings.mesh_shadow_enabled,
                    .shadow_map_resolution = settings.mesh_shadow_resolution,
                });
            }
            return frame;
        }
    } // namespace

    std::shared_ptr<lfs::core::Tensor> RenderingManager::composeSplitViewCpu(
        const SplitViewCpuDesc&, const glm::ivec2&) {
        // Split view is an explicit Phase 3 capability on Metal. Keeping this
        // empty prevents capture from manufacturing a view the user did not see.
        return {};
    }

    void RenderingManager::queueSharedScratchRetry(ViewRenderState& view,
                                                   const DirtyMask retry_dirty) {
        view.dirty_mask_.fetch_or(retry_dirty, std::memory_order_relaxed);
    }

    float RenderingManager::trainingRefreshIntervalSec(const ViewRenderState& view) const {
        return view.framerate_controller_.getSettings().training_frame_refresh_time_sec;
    }

    void RenderingManager::pollTrainingRefresh(const bool is_training,
                                               const int current_iteration) {
        std::lock_guard lock(views_mutex_);
        for (auto& [id, view] : view_states_) {
            const auto dirty = view->frame_lifecycle_service_.handleTrainingRefresh(
                is_training, trainingRefreshIntervalSec(*view));
            if (!dirty)
                continue;
            view->last_training_preview_iteration_ = current_iteration;
            view->has_training_preview_iteration_ = is_training;
            view->training_refresh_dirty_.fetch_or(dirty, std::memory_order_relaxed);
            view->dirty_mask_.fetch_or(dirty, std::memory_order_relaxed);
        }
    }

    double RenderingManager::secondsUntilTrainingRefresh() const {
        std::lock_guard lock(views_mutex_);
        double remaining = std::numeric_limits<double>::infinity();
        for (const auto& [id, view] : view_states_)
            remaining = std::min(
                remaining, view->frame_lifecycle_service_.secondsUntilTrainingRefresh(
                               trainingRefreshIntervalSec(*view)));
        return remaining;
    }

    double RenderingManager::secondsUntilCameraSettle() const {
        return std::numeric_limits<double>::infinity();
    }

    void RenderingManager::pollParkedArenaRetry() {
        if (scene_renderer_ && scene_renderer_->takeRefinementRequest())
            markDirty(DirtyFlag::CAMERA, FrameReason::AsyncCompletion,
                      "tensor_renderer_refinement_ready");
        if (point_scene_renderer_ && point_scene_renderer_->takeRefinementRequest())
            markDirty(DirtyFlag::CAMERA, FrameReason::AsyncCompletion,
                      "tensor_renderer_refinement_ready");
    }

    void RenderingManager::beginImportRenderCheck(const uint64_t generation) {
        import_render_check_ = true;
        import_render_generation_ = generation;
        import_render_frames_ = 0;
        import_render_result_.reset();
        markDirty(DirtyFlag::ALL, FrameReason::SceneChange);
    }

    bool RenderingManager::importUsesCombinedModel() const { return true; }

    std::optional<std::string> RenderingManager::prepareImportRenderCheck(
        const RenderContext& context, const std::function<void()>& prepare_viewport) {
        beginImportRenderCheck(context.scene_manager->getScene().renderGeneration());
        import_render_preparing_ = true;
        auto result = renderFrame(RenderContext{
            .view = context.view,
            .viewport = context.viewport,
            .settings = context.settings,
            .logical_screen_size = context.logical_screen_size,
            .viewport_region = context.viewport_region,
            .scene_manager = context.scene_manager,
            .graphics_context = context.graphics_context,
            .preparing_import = true,
            .provisional_import_node = context.provisional_import_node,
        });
        import_render_preparing_ = false;
        import_render_result_ = result.rendered ? std::optional<std::string>{std::string{}}
                                                : std::optional<std::string>{"Metal tensor import validation did not render"};
        if (import_render_result_->empty() && prepare_viewport)
            prepare_viewport();
        return import_render_result_;
    }

    std::optional<std::string> RenderingManager::pollImportRenderCheck(
        const RenderContext& context, const std::function<void()>& prepare_viewport) {
        return prepareImportRenderCheck(context, prepare_viewport);
    }

    void RenderingManager::cancelImportRenderCheck() {
        import_render_check_ = false;
        import_render_preparing_ = false;
        import_render_result_.reset();
    }

    void RenderingManager::noteImportRenderFrame(const uint64_t generation, std::string error) {
        if (!import_render_check_ || generation != import_render_generation_)
            return;
        import_render_result_ = std::move(error);
    }

    ViewportFrameResult RenderingManager::renderFrame(const RenderContext& context) {
        auto& view = viewState(context.view);
        if (context.graphics_context)
            last_graphics_context_ = context.graphics_context;

        glm::ivec2 size = context.viewport.frameBufferSize;
        glm::ivec2 top_left{0, 0};
        if (context.viewport_region) {
            size = {static_cast<int>(context.viewport_region->width),
                    static_cast<int>(context.viewport_region->height)};
            top_left = {static_cast<int>(context.viewport_region->x),
                        static_cast<int>(context.viewport_region->y)};
        }
        if (size.x <= 0 || size.y <= 0)
            return {.image = view.vulkan_viewport_image_,
                    .image_generation = view.vulkan_viewport_image_generation_,
                    .size = view.vulkan_viewport_image_size_};

        view.last_nonzero_viewport_size_ = size;
        view.framebuffer_viewport_rect_ = {.top_left = top_left, .size = size};
        // Same settings bookkeeping as the Vulkan manager: a settings change
        // re-renders the view, and the stale-view check compares against it.
        if (!view.rendered_settings || view.rendered_settings->view() != context.settings.view() ||
            view.rendered_settings->scene() != context.settings.scene()) {
            view.dirty_mask_.fetch_or(DirtyFlag::ALL);
            view.rendered_settings = context.settings;
        }
        // Render-target identities are backend-neutral ownership tokens. Keep
        // their lazy/per-view lifetime contract even when an empty Metal frame
        // has no scene tensor to allocate yet.
        if (!view.main_render_target_.valid())
            view.main_render_target_ = render_targets_.allocate();
        if (!context.scene_manager) {
            view.dirty_mask_.exchange(0, std::memory_order_acq_rel);
            clearViewportImageState(view, size);
            view.viewport_artifact_service_.clearViewportOutput();
            return {.size = size, .matches_viewport_extent = true, .rendered = true};
        }

        if (context.settings.split_view_mode != SplitViewMode::Disabled) {
            LOG_WARN("Split view is unavailable on the Metal tensor compositor");
        }

        auto frame_settings = context.settings;
        enforceProjectionBackend(frame_settings);
        SceneRenderState scene_state = context.scene_manager->buildRenderState();
        const auto* model = scene_state.combined_model;
        const bool has_model = hasRenderableGaussians(model);
        const bool has_points = scene_state.point_cloud && scene_state.point_cloud->size() > 0;
        if (!has_model && !has_points && scene_state.meshes.empty()) {
            // An empty scene is still a completed render. Leaving its invalidation
            // pending makes the frame-demand ledger repaint forever at display rate.
            view.dirty_mask_.exchange(0, std::memory_order_acq_rel);
            clearViewportImageState(view, size);
            view.viewport_artifact_service_.clearViewportOutput();
            return {.size = size, .matches_viewport_extent = true, .rendered = true};
        }

        const FrameContext frame_context{
            .viewport = context.viewport,
            .viewport_region = context.viewport_region,
            .scene_manager = context.scene_manager,
            .model = model,
            .scene_state = scene_state,
            .settings = frame_settings,
            .render_size = size,
            .viewport_pos = top_left,
            .frame_dirty = view.dirty_mask_.exchange(0, std::memory_order_acq_rel),
            .training_active = context.scene_manager->getTrainerManager() &&
                               context.scene_manager->getTrainerManager()->isRunning(),
            .depth_window_drag_preview = depthWindowDragPreview(context.view),
            .cursor_preview = view.viewport_overlay_service_.cursorPreview(),
            .gizmo = gizmo_state_,
            .hovered_camera_id = camera_interaction_service_.hoveredCameraId(),
            .current_camera_id = camera_interaction_service_.currentCameraId(),
            .hovered_gaussian_id = view.viewport_overlay_service_.hoveredGaussianId(),
            .selection_flash_intensity = view.animation_state_.selectionFlashIntensity(),
        };

        if (!has_model && !has_points) {
            // Meshes only: the compositor draws them over the background.
            view.dirty_mask_.exchange(0, std::memory_order_acq_rel);
            clearViewportImageState(view, size);
            view.viewport_artifact_service_.clearViewportOutput();
            view.viewport_environment_ = environmentFor(frame_context, frame_settings);
            view.viewport_meshes_ = meshesFor(frame_context, frame_settings);
            return {.size = size, .matches_viewport_extent = true, .rendered = true};
        }

        std::shared_ptr<lfs::core::Tensor> image;
        std::shared_ptr<lfs::core::Tensor> depth;
        if (!context.graphics_context)
            return {.size = size};
        const auto keep_previous = [&](const std::string& error) -> ViewportFrameResult {
            LOG_ERROR("Metal scene render failed: {}", error);
            view.dirty_mask_.fetch_or(DirtyFlag::ALL, std::memory_order_relaxed);
            return {.image = view.vulkan_viewport_image_,
                    .image_generation = view.vulkan_viewport_image_generation_,
                    .size = view.vulkan_viewport_image_size_};
        };
        const bool render_points = frame_settings.point_cloud_mode || !has_model;
        if (!render_points) {
            if (!scene_renderer_)
                scene_renderer_ = createSceneRenderer(*context.graphics_context);
            auto request = buildViewportRenderRequest(frame_context, size);
            request.raster_backend =
                lfs::rendering::normalizeViewerRasterBackend(request.raster_backend, request.gut);
            request.gut = lfs::rendering::isGutBackend(request.raster_backend);
            std::vector<std::uint32_t> lod_touched_chunks;
            prepareLodRequest(frame_settings, model, request, lod_touched_chunks);
            auto rendered = scene_renderer_->render(*model, request, true,
                                                    view.main_render_target_);
            if (!rendered)
                return keep_previous(rendered.error());
            noteLodPageGeneration(rendered->lod_page_generation);
            if (rendered->lod_streaming_active)
                requestViewFollowUp(view, DirtyFlag::CAMERA);
            auto outputs = scene_renderer_->readOutputTensors(view.main_render_target_);
            if (!outputs)
                return keep_previous(lfs::format_for_developer(outputs.error()));
            image = std::move(outputs->color);
            depth = std::move(outputs->depth);
        } else {
            if ((frame_context.frame_dirty & DirtyFlag::SPLATS) != 0) {
                point_cloud_colors_cache_key_ = nullptr;
                point_cloud_colors_cache_size_ = 0;
                point_cloud_colors_cache_ = lfs::core::Tensor{};
                ++point_cloud_data_revision_;
            }
            if ((frame_context.frame_dirty & DirtyFlag::SELECTION) != 0)
                ++point_cloud_preview_selection_revision_;
            const bool splats_as_points = frame_settings.point_cloud_mode && has_model;
            const std::vector<glm::mat4> cloud_transforms{scene_state.point_cloud_transform};
            const auto pc_request = buildPointCloudRenderRequest(
                frame_context, size, splats_as_points ? scene_state.model_transforms : cloud_transforms);
            lfs::core::Tensor splat_positions;
            const lfs::core::Tensor* positions = nullptr;
            const lfs::core::Tensor* colors = nullptr;
            if (splats_as_points) {
                constexpr float SH_C0 = 0.28209479177387814f;
                const auto& sh0 = model->sh0_raw();
                const void* sh0_key = sh0.is_valid() ? sh0.data_ptr() : nullptr;
                const std::size_t sh0_count = sh0.is_valid() ? static_cast<std::size_t>(sh0.size(0)) : 0;
                if (sh0_key != point_cloud_colors_cache_key_ || sh0_count != point_cloud_colors_cache_size_ ||
                    !point_cloud_colors_cache_.is_valid()) {
                    point_cloud_colors_cache_ = (sh0.slice(1, 0, 1).squeeze(1) * SH_C0 + 0.5f).clamp(0.0f, 1.0f);
                    point_cloud_colors_cache_key_ = sh0_key;
                    point_cloud_colors_cache_size_ = sh0_count;
                }
                splat_positions = model->get_means();
                positions = &splat_positions;
                colors = &point_cloud_colors_cache_;
            } else {
                positions = &scene_state.point_cloud->means;
                colors = &scene_state.point_cloud->colors;
            }
            const auto& frame_view = pc_request.frame_view;
            const glm::mat4 view_matrix = frame_view.getViewMatrix();
            const glm::mat4 projection = lfs::rendering::createProjectionMatrix(
                frame_view.size, lfs::rendering::focalLengthToVFov(frame_view.focal_length_mm),
                frame_view.orthographic, frame_view.ortho_scale, frame_view.near_plane, frame_view.far_plane);
            // Same clip-space flip as the Vulkan compositor: images are top-left origin.
            glm::mat4 clip_y_flip(1.0f);
            clip_y_flip[1][1] = -1.0f;
            PointSceneRenderer::RenderRequest request{};
            request.positions = positions;
            request.colors = colors;
            request.positions_revision = point_cloud_data_revision_;
            request.colors_revision = point_cloud_data_revision_;
            request.model_transforms = pc_request.scene.model_transforms;
            request.transform_indices = pc_request.scene.transform_indices.get();
            request.node_visibility_mask = &pc_request.scene.node_visibility_mask;
            if (splats_as_points && model->has_deleted_mask() && model->deleted_mask_matches_size()) {
                request.deleted_mask = &model->deleted();
                request.deleted_mask_revision = model->deleted_mask_version();
            }
            request.selection_mask = pc_request.overlay.selection_mask.get();
            request.preview_selection_mask = pc_request.overlay.transient_mask.mask;
            request.selection_colors = &pc_request.overlay.selection_colors;
            request.preview_selection_additive = pc_request.overlay.transient_mask.additive;
            request.selection_revision = point_cloud_preview_selection_revision_;
            request.preview_selection_revision = point_cloud_preview_selection_revision_;
            if (pc_request.filters.crop_box) {
                request.crop = PointSceneRenderer::CropBox{
                    .to_local = pc_request.filters.crop_box->transform,
                    .min = pc_request.filters.crop_box->min,
                    .max = pc_request.filters.crop_box->max,
                    .inverse = pc_request.filters.crop_inverse,
                    .desaturate = pc_request.filters.crop_desaturate};
            } else if (pc_request.filters.crop_ellipsoid) {
                request.crop_ellipsoid = PointSceneRenderer::CropEllipsoid{
                    .to_local = pc_request.filters.crop_ellipsoid->transform,
                    .radii = pc_request.filters.crop_ellipsoid->radii,
                    .inverse = pc_request.filters.crop_inverse,
                    .desaturate = pc_request.filters.crop_desaturate};
            }
            request.view = view_matrix;
            request.view_projection = clip_y_flip * projection * view_matrix;
            request.size = frame_view.size;
            request.background_color = frame_view.background_color;
            request.transparent_background = pc_request.transparent_background;
            request.synchronize_output = context.preparing_import;
            request.orthographic = frame_view.orthographic;
            request.ortho_scale = frame_view.ortho_scale;
            request.focal_y = lfs::core::fov2focal(
                lfs::rendering::focalLengthToVFovRad(frame_view.focal_length_mm), frame_view.size.y);
            request.voxel_size = pc_request.render.voxel_size;
            request.scaling_modifier = pc_request.render.scaling_modifier;
            request.depth_view = frame_settings.depth_view;
            request.depth_view_min = frame_settings.depth_view_min;
            request.depth_view_max = frame_settings.depth_view_max;
            request.depth_visualization_mode = frame_settings.depth_visualization_mode;
            if (!point_scene_renderer_)
                point_scene_renderer_ = createPointSceneRenderer(*context.graphics_context);
            auto rendered = point_scene_renderer_->render(request, view.main_render_target_);
            if (!rendered)
                return keep_previous(rendered.error());
            auto outputs = point_scene_renderer_->readOutputTensors(view.main_render_target_);
            if (!outputs)
                return keep_previous(lfs::format_for_developer(outputs.error()));
            image = std::move(outputs->color);
            depth = std::move(outputs->depth);
        }

        if (!image || !image->is_valid()) {
            view.dirty_mask_.fetch_or(DirtyFlag::ALL, std::memory_order_relaxed);
            return {.image = view.vulkan_viewport_image_,
                    .image_generation = view.vulkan_viewport_image_generation_,
                    .size = view.vulkan_viewport_image_size_};
        }

        ++view.vulkan_viewport_image_generation_;
        view.vulkan_viewport_image_ = image;
        view.viewport_depth_image_ = depth;
        view.viewport_environment_ = environmentFor(frame_context, frame_settings);
        view.viewport_meshes_ = meshesFor(frame_context, frame_settings);
        view.vulkan_viewport_image_size_ = size;
        view.vulkan_viewport_image_alloc_size_ = size;
        view.vulkan_viewport_coordinate_size_ = size;
        view.vulkan_viewport_image_flip_y_ = false;
        lfs::rendering::FrameMetadata metadata{
            .viewer_backend = lfs::rendering::ViewerBackend::Metal,
            .depth_panels = {lfs::rendering::FramePanelMetadata{
                .depth = depth, .start_position = 0.0f, .end_position = 1.0f}},
            .depth_panel_count = depth ? 1u : 0u,
            .valid = true,
            .far_plane = context.settings.depth_clip_enabled
                             ? context.settings.depth_clip_far
                             : lfs::rendering::DEFAULT_FAR_PLANE,
            .orthographic = context.settings.orthographic,
        };
        view.viewport_artifact_service_.updateFromImageOutput(image, metadata, size, true);
        initialized_ = true;
        return {.image = std::move(image),
                .image_generation = view.vulkan_viewport_image_generation_,
                .size = size,
                .alloc_size = size,
                .matches_viewport_extent = true,
                .rendered = true};
    }

    void RenderingManager::publishFrameToInterop(ViewId, const ViewportFrameResult&) {
        // Tensor frames are already the compositor input. No native ownership
        // transfer or external semaphore publication is required.
    }

    std::expected<void, std::string> RenderingManager::ensureVksplatTrainingSharedScratchReady(
        GraphicsContext&, const lfs::core::SplatData&, glm::ivec2) {
        return {};
    }

    std::expected<lfs::core::Tensor, std::string> RenderingManager::buildVksplatSelectionMask(
        SceneManager& scene_manager, const lfs::rendering::FrameView& frame_view,
        bool equirectangular, VksplatSelectionMaskShape shape,
        const std::vector<glm::vec4>& primitives,
        const std::vector<glm::vec2>& polygon_vertices,
        std::uint32_t* picked_ring_id_out) {
        if (!last_graphics_context_)
            return std::unexpected("Tensor selection requires an active graphics context");
        auto scene_state = scene_manager.buildRenderState();
        if (!hasRenderableGaussians(scene_state.combined_model))
            return std::unexpected("Tensor selection found no renderable scene");
        if (!scene_renderer_)
            scene_renderer_ = createSceneRenderer(*last_graphics_context_);
        const auto mapped = static_cast<SceneRenderer::SelectionMaskShape>(shape);
        return scene_renderer_->buildSelectionMask(
            *scene_state.combined_model,
            {.frame_view = frame_view,
             .scene = {.model_transforms = &scene_state.model_transforms,
                       .transform_indices = scene_state.transform_indices,
                       .node_visibility_mask = scene_state.node_visibility_mask},
             .shape = mapped,
             .primitives = primitives,
             .polygon_vertices = polygon_vertices,
             .equirectangular = equirectangular,
             .picked_ring_id_out = picked_ring_id_out},
            false);
    }

    lfs::io::SplatTensorAllocator RenderingManager::makeSplatTensorAllocator() const {
        return last_graphics_context_ ? last_graphics_context_->splatTensorAllocator(true)
                                      : lfs::io::SplatTensorAllocator{};
    }
} // namespace lfs::vis

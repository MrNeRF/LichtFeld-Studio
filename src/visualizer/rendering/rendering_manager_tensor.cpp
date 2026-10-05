/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "rendering_manager.hpp"
#include "rendering_manager_split_view.hpp"
#include "scene_temporal_frame_setup.hpp"
#include "tensor_scene_temporal_pipeline.hpp"

#include "core/camera.hpp"
#include "core/gpu_backend_fwd.hpp"
#include "core/logger.hpp"
#include "core/training_manager.hpp"
#include "display_tensors.hpp"
#include "rendering/image_layout.hpp"
#include "rendering/model_renderability.hpp"
#include "scene/scene_manager.hpp"
#include "scene_renderer_factory.hpp"
#include "viewport_appearance_correction.hpp"
#include "viewport_request_builder.hpp"
#include "visualizer/scene_coordinate_utils.hpp"
#include "window/graphics_context.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <format>
#include <limits>
#include <utility>

namespace lfs::vis {
    namespace {
        glm::ivec2 tensorImageSize(const lfs::core::Tensor& image) {
            const auto layout = lfs::rendering::detectImageLayout(image);
            return layout == lfs::rendering::ImageLayout::Unknown
                       ? glm::ivec2{0, 0}
                       : glm::ivec2{lfs::rendering::imageWidth(image, layout),
                                    lfs::rendering::imageHeight(image, layout)};
        }

        std::shared_ptr<lfs::core::Tensor> ensureGpuDisplayTensor(
            std::shared_ptr<lfs::core::Tensor> image) {
            if (!image || !image->is_valid())
                return {};
            if (image->device() == lfs::core::Device::GPU)
                return image;
            auto gpu = image->to(lfs::core::Device::GPU);
            return gpu.is_valid() ? std::make_shared<lfs::core::Tensor>(std::move(gpu)) : nullptr;
        }

        ViewportSplitPanel makeTensorSplitPanel(
            std::shared_ptr<lfs::core::Tensor> image,
            std::shared_ptr<lfs::core::Tensor> depth,
            const RenderTargetId target,
            const float start,
            const float end,
            const bool flip_y = false) {
            const glm::ivec2 size = image ? tensorImageSize(*image) : glm::ivec2{0, 0};
            const glm::vec2 clamp{
                size.x > 0 ? (static_cast<float>(size.x) - 0.5f) / static_cast<float>(size.x) : 1.0f,
                size.y > 0 ? (static_cast<float>(size.y) - 0.5f) / static_cast<float>(size.y) : 1.0f};
            return {.target = target,
                    .image = std::move(image),
                    .depth = std::move(depth),
                    .start_position = start,
                    .end_position = end,
                    .flip_y = flip_y,
                    .image_size = size,
                    .allocation_size = size,
                    .uv_scale = {1.0f, 1.0f},
                    .uv_clamp_max = clamp};
        }

        SplitViewCpuPanelDesc cpuPanelDesc(const ViewportSplitPanel& panel) {
            return {.image = panel.image,
                    .start_position = panel.start_position,
                    .end_position = panel.end_position,
                    .normalize_x_to_panel = panel.normalize_x_to_panel,
                    .flip_y = panel.flip_y,
                    .uv_scale = panel.uv_scale,
                    .uv_clamp_max = panel.uv_clamp_max,
                    .texcoord_scale = panel.texcoord_scale,
                    .texcoord_offset = panel.texcoord_offset,
                    .spatial_filter = panel.spatial_filter};
        }

        SplitViewCpuDesc cpuSplitViewDesc(const ViewportSplitView& split) {
            return {.loss_visualization = split.loss_visualization,
                    .left = cpuPanelDesc(split.left),
                    .right = cpuPanelDesc(split.right),
                    .split_position = split.split_position,
                    .content_rect = split.content_rect,
                    .coordinate_extent = split.coordinate_extent,
                    .background = split.background};
        }
    } // namespace

    std::shared_ptr<lfs::core::Tensor> RenderingManager::composeSplitViewCpu(
        const SplitViewCpuDesc& params, const glm::ivec2& output_size) {
        return composeSplitViewCpuImage(params, output_size);
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
        if (!splitViewEnabled(context.settings.split_view_mode))
            view.split_view_ = {};
        if (!context.scene_manager) {
            view.dirty_mask_.exchange(0, std::memory_order_acq_rel);
            clearViewportImageState(view, size);
            view.viewport_artifact_service_.clearViewportOutput();
            return {.size = size, .matches_viewport_extent = true, .rendered = true};
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

        const auto requested_upscaler = sceneUpscalerBackendFromId(frame_settings.scene_upscaler)
                                            .value_or(SceneUpscalerBackend::Native);
        const auto reported_upscaler = sceneUpscalerRuntimeSelection(context.view);
        const bool reconstruction_runtime_ready =
            reported_upscaler.requested == requested_upscaler &&
            requested_upscaler != SceneUpscalerBackend::Native &&
            reported_upscaler.effective == requested_upscaler && !reported_upscaler.fellBack();
        const bool split_active = splitViewEnabled(frame_settings.split_view_mode);
        const bool temporal_requested = requested_upscaler == SceneUpscalerBackend::Temporal;
        const bool temporal_mode_supported = !frame_settings.equirectangular &&
                                             !frame_settings.apply_appearance_correction &&
                                             !split_active && has_model &&
                                             !frame_settings.point_cloud_mode;
        const bool built_in_reconstruction =
            requested_upscaler == SceneUpscalerBackend::Spatial ||
            (temporal_requested && temporal_mode_supported);
        const float scale = effectiveSceneRenderScale(
            frame_settings.render_scale, frame_settings.scene_upscaler_scale,
            built_in_reconstruction && reconstruction_runtime_ready);
        const glm::ivec2 render_size{
            std::max(static_cast<int>(std::lround(static_cast<float>(size.x) * scale)), 1),
            std::max(static_cast<int>(std::lround(static_cast<float>(size.y) * scale)), 1)};
        const DirtyMask frame_dirty = view.dirty_mask_.exchange(0, std::memory_order_acq_rel);
        const DirtyMask training_refresh_dirty =
            view.training_refresh_dirty_.exchange(0, std::memory_order_relaxed);
        const auto temporal_setup = prepareSceneTemporalFrame(
            view.temporal_convergence_,
            {.backend_requested = temporal_requested,
             .runtime_ready = reconstruction_runtime_ready,
             .projection_supported = true,
             .equirectangular = frame_settings.equirectangular,
             .appearance_correction = frame_settings.apply_appearance_correction,
             .split_supported = !split_active,
             .raster_supported = has_model && !frame_settings.point_cloud_mode,
             .lod_results_ready = lod_controller_ && lod_controller_->hasReadyResults(),
             .lod_transition_active = lod_controller_ && lod_controller_->transitionActive(),
             .frame_dirty = frame_dirty,
             .training_refresh_dirty = training_refresh_dirty});
        {
            std::lock_guard lock(settings_mutex_);
            view.scene_upscaler_mode_unsupported_ =
                temporal_setup.mode_unsupported ||
                (temporal_requested && !temporal_mode_supported);
        }
        const std::uint64_t temporal_camera_cut_generation =
            view.temporal_camera_cut_generation_.load(std::memory_order_acquire);
        const bool temporal_camera_cut = temporal_camera_cut_generation !=
                                         view.consumed_temporal_camera_cut_generation_;
        if ((frame_dirty & (DirtyFlag::SPLATS | DirtyFlag::MESH | DirtyFlag::BACKGROUND)) != 0) {
            if (++view.temporal_scene_revision_ == 0)
                ++view.temporal_scene_revision_;
        }

        const FrameContext frame_context{
            .viewport = context.viewport,
            .viewport_region = context.viewport_region,
            .scene_manager = context.scene_manager,
            .model = model,
            .scene_state = scene_state,
            .settings = frame_settings,
            .render_size = render_size,
            .viewport_pos = top_left,
            .frame_dirty = frame_dirty,
            .training_active = context.scene_manager->getTrainerManager() &&
                               context.scene_manager->getTrainerManager()->isRunning(),
            .depth_window_drag_preview = depthWindowDragPreview(context.view),
            .cursor_preview = view.viewport_overlay_service_.cursorPreview(),
            .gizmo = gizmo_state_,
            .hovered_camera_id = camera_interaction_service_.hoveredCameraId(),
            .current_camera_id = camera_interaction_service_.currentCameraId(),
            .hovered_gaussian_id = view.viewport_overlay_service_.hoveredGaussianId(),
            .selection_flash_intensity = view.animation_state_.selectionFlashIntensity(),
            .scene_jitter_pixels = temporal_setup.jitter_pixels,
        };
        if (!context.preparing_import) {
            const glm::vec2 screen_position = context.viewport_region
                                                  ? glm::vec2{context.viewport_region->x, context.viewport_region->y}
                                                  : glm::vec2{0.0f};
            const glm::vec2 screen_size = context.viewport_region
                                              ? glm::vec2{context.viewport_region->width, context.viewport_region->height}
                                              : glm::vec2{context.viewport.windowSize};
            const auto panels = buildSplitViewInteractionPanels(
                context.viewport, frame_settings, screen_position, screen_size);
            view.viewport_interaction_context_.updatePickContext(panels);
        }

        if (!has_model && !has_points) {
            // Meshes only: the compositor draws them over the background.
            view.dirty_mask_.exchange(0, std::memory_order_acq_rel);
            clearViewportImageState(view, size);
            view.viewport_artifact_service_.clearViewportOutput();
            view.viewport_environment_ = buildViewportEnvironment(
                frame_context, frame_settings, environmentBackgroundUsesTransparentViewerCompositing(frame_settings));
            view.viewport_meshes_ = buildViewportMeshes(frame_context, frame_settings);
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

        if (splitViewEnabled(frame_settings.split_view_mode)) {
            if (!view.split_left_render_target_.valid())
                view.split_left_render_target_ = render_targets_.allocate();
            if (!view.split_right_render_target_.valid())
                view.split_right_render_target_ = render_targets_.allocate();

            struct PanelOutput {
                std::shared_ptr<lfs::core::Tensor> image;
                std::shared_ptr<lfs::core::Tensor> depth;
                lfs::rendering::FrameMetadata metadata;
            };
            const auto render_panel = [&](const lfs::core::SplatData& panel_model,
                                          SceneRenderState panel_state,
                                          const glm::ivec2 panel_size,
                                          const RenderTargetId target,
                                          const std::optional<GTRenderCamera>& camera = std::nullopt,
                                          const std::optional<SplitViewPanelId> panel_id = std::nullopt,
                                          const glm::ivec2 subregion_origin = {0, 0},
                                          const glm::ivec2 subregion_full_size = {0, 0})
                -> std::expected<PanelOutput, std::string> {
                FrameContext panel_context = frame_context;
                panel_context.model = &panel_model;
                panel_context.scene_state = std::move(panel_state);
                panel_context.render_size = panel_size;
                auto request = panel_id && subregion_full_size.x > 0 && subregion_full_size.y > 0
                                   ? buildPlyComparisonRenderRequest(
                                         panel_context, panel_size, context.viewport,
                                         *panel_id, subregion_full_size)
                                   : buildViewportRenderRequest(
                                         panel_context, panel_size, nullptr, panel_id,
                                         subregion_origin, subregion_full_size);
                if (camera)
                    applyGTComparisonRenderCamera(request.frame_view, request.equirectangular, *camera);
                request.raster_backend =
                    lfs::rendering::normalizeViewerRasterBackend(request.raster_backend, request.gut);
                request.gut = lfs::rendering::isGutBackend(request.raster_backend);
                std::vector<std::uint32_t> lod_touched_chunks;
                prepareLodRequest(frame_settings, &panel_model, request, lod_touched_chunks);
                if (!scene_renderer_)
                    scene_renderer_ = createSceneRenderer(*context.graphics_context);
                auto rendered = scene_renderer_->render(panel_model, request, true, target);
                if (!rendered)
                    return std::unexpected(rendered.error());
                noteLodPageGeneration(rendered->lod_page_generation);
                if (rendered->lod_streaming_active)
                    requestViewFollowUp(view, DirtyFlag::CAMERA);
                auto outputs = scene_renderer_->readOutputTensors(target);
                if (!outputs)
                    return std::unexpected(lfs::format_for_developer(outputs.error()));
                lfs::rendering::FrameMetadata metadata{
                    .viewer_backend = lfs::rendering::ViewerBackend::Metal,
                    .depth_panels = {lfs::rendering::FramePanelMetadata{
                        .depth = outputs->depth, .start_position = 0.0f, .end_position = 1.0f}},
                    .depth_panel_count = outputs->depth ? 1u : 0u,
                    .valid = true,
                    .far_plane = frame_settings.depth_clip_enabled
                                     ? frame_settings.depth_clip_far
                                     : lfs::rendering::DEFAULT_FAR_PLANE,
                    .orthographic = camera ? false : frame_settings.orthographic};
                return PanelOutput{std::move(outputs->color), std::move(outputs->depth), std::move(metadata)};
            };
            const auto publish_split = [&](ViewportSplitView split,
                                           lfs::rendering::FrameMetadata metadata,
                                           SplitViewInfo info,
                                           std::optional<GTPresentedView> gt_view = std::nullopt) {
                view.split_view_ = std::move(split);
                ++view.split_view_image_generation_;
                view.vulkan_viewport_image_ = view.split_view_.left.image;
                view.viewport_depth_image_ = view.split_view_.left.depth;
                view.vulkan_viewport_image_size_ = size;
                view.vulkan_viewport_image_alloc_size_ = size;
                view.vulkan_viewport_coordinate_size_ = size;
                view.vulkan_viewport_image_flip_y_ = false;
                view.vulkan_gt_comparison_content_size_ =
                    splitViewUsesGTComparison(frame_settings.split_view_mode)
                        ? glm::ivec2{view.split_view_.content_rect.z, view.split_view_.content_rect.w}
                        : glm::ivec2{0, 0};
                view.vulkan_gt_comparison_selection_view_ = std::move(gt_view);
                FrameResources resources{.split_view_executed = true, .split_info = std::move(info)};
                view.split_view_service_.updateInfo(resources);
                const auto capture = view.split_view_;
                view.viewport_artifact_service_.setLazyCapture(
                    [capture, size] { return composeSplitViewCpuImage(cpuSplitViewDesc(capture), size); },
                    metadata, size);
                view.viewport_environment_ = {};
                view.viewport_meshes_ = {};
                initialized_ = true;
                return ViewportFrameResult{
                    .image = view.split_view_.left.image,
                    .image_generation = view.split_view_image_generation_,
                    .split_left_image_generation = view.split_view_image_generation_,
                    .size = view.split_view_.left.image_size,
                    .alloc_size = view.split_view_.left.allocation_size,
                    .matches_viewport_extent = true,
                    .rendered = true,
                    .split_right_image = view.split_view_.right.image,
                    .split_right_image_generation = view.split_view_image_generation_,
                    .split_right_size = view.split_view_.right.image_size};
            };

            if (splitViewUsesPLYComparison(frame_settings.split_view_mode)) {
                const auto& scene = context.scene_manager->getScene();
                const auto visible_nodes = scene.getVisibleSplatNodeSlots();
                const auto pair = plyComparisonPairForOffset(visible_nodes.size(), frame_settings.split_view_offset);
                if (!pair)
                    return keep_previous("PLY comparison requires at least two visible Gaussian models");
                const auto& left_slot = visible_nodes[pair->first];
                const auto& right_slot = visible_nodes[pair->second];
                if (!left_slot.node || !right_slot.node ||
                    !hasRenderableGaussians(left_slot.node->model.get()) ||
                    !hasRenderableGaussians(right_slot.node->model.get())) {
                    return keep_previous("PLY comparison render slots are unavailable");
                }
                const auto layouts = makePlyComparisonPanelLayouts(render_size.x, frame_settings.split_position);
                SceneRenderState left_state = scene_state;
                SceneRenderState right_state = scene_state;
                scopeSceneRenderStateToVisibleSplatNode(
                    left_state, scene, *left_slot.node, static_cast<int>(left_slot.slot_index),
                    scene_coords::nodeVisualizerWorldTransform(scene, left_slot.node->id));
                scopeSceneRenderStateToVisibleSplatNode(
                    right_state, scene, *right_slot.node, static_cast<int>(right_slot.slot_index),
                    scene_coords::nodeVisualizerWorldTransform(scene, right_slot.node->id));
                auto left = render_panel(*left_slot.node->model, std::move(left_state),
                                         {std::max(layouts[0].panel.width, 1), render_size.y},
                                         view.split_left_render_target_, std::nullopt,
                                         SplitViewPanelId::Left,
                                         {layouts[0].panel.x, 0}, size);
                auto right = render_panel(*right_slot.node->model, std::move(right_state),
                                          {std::max(layouts[1].panel.width, 1), render_size.y},
                                          view.split_right_render_target_, std::nullopt,
                                          SplitViewPanelId::Right,
                                          {layouts[1].panel.x, 0}, size);
                if (!left || !right) {
                    return keep_previous(std::format(
                        "PLY comparison panel rendering failed (left: {}; right: {})",
                        left ? "ok" : left.error(), right ? "ok" : right.error()));
                }
                ViewportSplitView split{
                    .enabled = true,
                    .left = makeTensorSplitPanel(std::move(left->image), std::move(left->depth),
                                                 view.split_left_render_target_,
                                                 layouts[0].panel.start_position,
                                                 layouts[0].panel.end_position),
                    .right = makeTensorSplitPanel(std::move(right->image), std::move(right->depth),
                                                  view.split_right_render_target_,
                                                  layouts[1].panel.start_position,
                                                  layouts[1].panel.end_position),
                    .split_position = frame_settings.split_position,
                    .content_rect = {0, 0, render_size.x, render_size.y},
                    .coordinate_extent = render_size,
                    .background = frame_settings.background_color};
                const bool spatial_filter = requested_upscaler == SceneUpscalerBackend::Spatial &&
                                            reconstruction_runtime_ready;
                split.left.spatial_filter = spatial_filter;
                split.right.spatial_filter = spatial_filter;
                split.left.texcoord_scale = layouts[0].texcoord_scale;
                split.left.texcoord_offset = layouts[0].texcoord_offset;
                split.right.texcoord_scale = layouts[1].texcoord_scale;
                split.right.texcoord_offset = layouts[1].texcoord_offset;
                return publish_split(
                    std::move(split), makeSplitMetadata(left->metadata, right->metadata,
                                                       frame_settings.split_position),
                    {.enabled = true,
                     .mode_label = "Split View",
                     .detail_label = std::format("{} | {}", left_slot.node->name, right_slot.node->name),
                     .left_name = left_slot.node->name,
                     .right_name = right_slot.node->name});
            }

            auto& scene = context.scene_manager->getScene();
            std::shared_ptr<lfs::core::Camera> camera;
            if (frame_context.current_camera_id >= 0)
                camera = scene.getCameraByUid(frame_context.current_camera_id);
            if (!camera) {
                for (const auto& candidate : scene.getAllCameras()) {
                    if (candidate) {
                        camera = candidate;
                        break;
                    }
                }
            }
            if (!camera || !has_model)
                return keep_previous("GT comparison requires a dataset camera and visible Gaussian model");

            const GTComparisonMode gt_mode = frame_settings.gt_comparison_mode;
            const glm::ivec2 gt_size = gtComparisonPreviewSize(*camera, size);
            const bool rgb_reference = gtComparisonUsesRGBReference(gt_mode);
            const std::filesystem::path reference_path = rgb_reference
                                                             ? camera->image_path()
                                                         : gt_mode == GTComparisonMode::Depth
                                                             ? camera->depth_path()
                                                             : camera->normal_path();
            const bool has_reference = rgb_reference ? camera->has_image()
                                       : gt_mode == GTComparisonMode::Depth ? camera->has_depth()
                                                                            : camera->has_normal();
            GTComparisonImageLookup lookup;
            if (has_reference && !reference_path.empty()) {
                const bool undistort =
                    camera->camera_model_type() != lfs::core::CameraModelType::EQUIRECTANGULAR &&
                    camera->is_undistort_precomputed() &&
                    (rgb_reference || !camera->is_undistort_prepared());
                lookup = getOrQueueGTComparisonImage({
                    .camera_uid = camera->uid(),
                    .mode = gt_mode,
                    .image_path = reference_path,
                    .preview_max_dimension = std::max(gt_size.x, gt_size.y),
                    .image_size = gt_size,
                    .undistort_requested = undistort,
                    .undistort_params = undistort ? camera->undistort_params() : lfs::core::UndistortParams{},
                    .depth_visualization_mode = frame_settings.depth_visualization_mode,
                    .background_color = frame_settings.background_color,
                    .camera = camera});
                if (lookup.status == GTComparisonImageStatus::Loading)
                    markViewDirty(context.view, DirtyFlag::SPLIT_VIEW, FrameReason::SettingsChange);
            } else {
                lookup.status = GTComparisonImageStatus::Failed;
            }
            std::shared_ptr<lfs::core::Tensor> reference = lookup.image;
            if ((!reference || !reference->is_valid()) && lookup.stale_image && !lookup.grace_elapsed)
                reference = lookup.stale_image;
            if (!reference || !reference->is_valid()) {
                const bool loading = lookup.status == GTComparisonImageStatus::Loading;
                reference = makeGTComparePlaceholderTensor(
                    gt_size, loading ? glm::vec3(0.09f) : glm::vec3(0.16f, 0.035f, 0.050f));
            }
            const auto reference_layout = lfs::rendering::detectImageLayout(*reference);
            if (reference_layout == lfs::rendering::ImageLayout::Unknown)
                return keep_previous("GT comparison produced an unsupported ground-truth image layout");
            const glm::ivec2 reference_size{lfs::rendering::imageWidth(*reference, reference_layout),
                                            lfs::rendering::imageHeight(*reference, reference_layout)};
            const auto render_camera = detail::buildGTRenderCamera(
                *camera, reference_size, detail::currentSceneTransform(context.scene_manager, camera->uid()));
            if (!render_camera)
                return keep_previous("GT comparison could not build the dataset render camera");
            auto rendered = render_panel(*model, scene_state, reference_size,
                                         view.split_right_render_target_, render_camera);
            if (!rendered)
                return keep_previous(std::format("GT comparison rendered panel failed: {}", rendered.error()));

            std::shared_ptr<lfs::core::Tensor> compare = rendered->image;
            if (gt_mode == GTComparisonMode::Depth) {
                compare = makeDepthDisplayTensor(*rendered->depth,
                                                 frame_settings.depth_visualization_mode,
                                                 frame_settings.background_color);
            } else if (gt_mode == GTComparisonMode::Normal) {
                if (!render_camera->intrinsics)
                    return keep_previous("Normal GT comparison requires pinhole camera intrinsics");
                compare = makeNormalDisplayFromDepthTensor(*rendered->depth, *render_camera->intrinsics);
            } else {
                compare = applyViewportAppearanceCorrection(
                    std::move(compare), context.scene_manager, frame_settings, camera->uid());
            }
            if (!compare || !compare->is_valid())
                return keep_previous("GT comparison could not prepare the rendered display panel");

            if (reference->device() == lfs::core::Device::CPU &&
                gt_comparison_cuda_source_ == reference.get() && gt_comparison_cuda_image_) {
                reference = gt_comparison_cuda_image_;
            } else {
                gt_comparison_cuda_source_ = reference.get();
                reference = ensureGpuDisplayTensor(std::move(reference));
                gt_comparison_cuda_image_ = reference;
            }
            compare = ensureGpuDisplayTensor(std::move(compare));
            if (!reference || !compare)
                return keep_previous("GT comparison failed to upload display panels to the GPU");

            const SplitCompositeContentRect rect =
                resolveSplitCompositeContentRect(size, true, reference_size);
            ViewportSplitView split{
                .enabled = true,
                .loss_visualization = gtComparisonShowsLoss(gt_mode),
                .left = makeTensorSplitPanel(std::move(reference), {}, view.split_left_render_target_,
                                             0.0f, frame_settings.split_position, true),
                .right = makeTensorSplitPanel(std::move(compare), std::move(rendered->depth),
                                              view.split_right_render_target_,
                                              frame_settings.split_position, 1.0f),
                .split_position = frame_settings.split_position,
                .content_rect = {rect.x, rect.y, rect.width, rect.height},
                .coordinate_extent = size,
                .background = frame_settings.background_color};
            const auto presented = GTPresentedView{*render_camera, reference_size};
            return publish_split(std::move(split), rendered->metadata,
                                 makeGTSplitViewInfo(gt_mode, camera->image_name()), presented);
        }

        view.split_view_ = {};
        view.vulkan_gt_comparison_content_size_ = {0, 0};
        view.vulkan_gt_comparison_selection_view_.reset();
        view.split_view_service_.updateInfo({});
        const bool render_points = frame_settings.point_cloud_mode || !has_model;
        if (!render_points) {
            if (!scene_renderer_)
                scene_renderer_ = createSceneRenderer(*context.graphics_context);
            auto request = buildViewportRenderRequest(frame_context, render_size);
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
                invalidatePointCloudData();
            }
            if ((frame_context.frame_dirty & DirtyFlag::SELECTION) != 0)
                ++point_cloud_preview_selection_revision_;
            const bool splats_as_points = frame_settings.point_cloud_mode && has_model;
            const std::vector<glm::mat4> cloud_transforms{scene_state.point_cloud_transform};
            const auto pc_request = buildPointCloudRenderRequest(
                frame_context, render_size,
                splats_as_points ? scene_state.model_transforms : cloud_transforms);
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
            auto request = buildPointSceneRequest(pc_request, frame_settings);
            request.positions = positions;
            request.colors = colors;
            request.positions_revision = point_cloud_data_revision_;
            request.colors_revision = point_cloud_data_revision_;
            if (splats_as_points && model->has_deleted_mask() && model->deleted_mask_matches_size()) {
                request.deleted_mask = &model->deleted();
                request.deleted_mask_revision = model->deleted_mask_version();
            }
            request.selection_revision = point_cloud_preview_selection_revision_;
            request.preview_selection_revision = point_cloud_preview_selection_revision_;
            request.synchronize_output = context.preparing_import;
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

        glm::ivec2 published_size = render_size;
        if (temporal_setup.eligible) {
            if (!view.tensor_temporal_pipeline_) {
                view.tensor_temporal_pipeline_ = std::make_shared<TensorSceneTemporalPipeline>(
                    lfs::core::GpuBackend::Metal);
            }
            const SceneTemporalQuality quality =
                frame_settings.scene_upscaler_preset == "performance"
                    ? SceneTemporalQuality::Performance
                : frame_settings.scene_upscaler_preset == "quality"
                    ? SceneTemporalQuality::Quality
                    : SceneTemporalQuality::Balanced;
            std::uint64_t scene_generation =
                static_cast<std::uint64_t>(reinterpret_cast<std::uintptr_t>(model));
            scene_generation ^= static_cast<std::uint64_t>(model->size()) +
                                0x9e3779b97f4a7c15ull + (scene_generation << 6) +
                                (scene_generation >> 2);
            scene_generation ^= model->param_layout_generation() +
                                0x9e3779b97f4a7c15ull + (scene_generation << 6) +
                                (scene_generation >> 2);
            scene_generation ^= view.temporal_scene_revision_ +
                                0x9e3779b97f4a7c15ull + (scene_generation << 6) +
                                (scene_generation >> 2);
            auto resolved = view.tensor_temporal_pipeline_->resolve({
                .color = image,
                .depth = depth,
                .frame = {
                    .view = frame_context.makeFrameView(),
                    .output_extent = size,
                    .jitter = frame_settings.orthographic
                                  ? glm::vec2(0.0f)
                                  : temporalJitterNdc(temporal_setup.jitter_pixels, render_size),
                    .render_scale = scale,
                    .scene_generation = scene_generation,
                    .backend_key = (static_cast<std::uint64_t>(requested_upscaler) << 32) |
                                   (static_cast<std::uint64_t>(quality) + 1),
                    .camera_cut = temporal_camera_cut,
                },
                .render_extent = render_size,
                .output_extent = size,
                .settings = sceneTemporalQualitySettings(quality),
            });
            if (!resolved)
                return keep_previous(lfs::format_for_developer(resolved.error()));
            image = std::move(resolved->color);
            published_size = size;
            view.consumed_temporal_camera_cut_generation_ = temporal_camera_cut_generation;
            if (view.temporal_convergence_.completeSuccessfulFrame())
                requestViewFollowUp(view, DirtyFlag::TEMPORAL);
        } else if (requested_upscaler == SceneUpscalerBackend::Spatial &&
                   reconstruction_runtime_ready) {
            if (!view.tensor_temporal_pipeline_) {
                view.tensor_temporal_pipeline_ = std::make_shared<TensorSceneTemporalPipeline>(
                    lfs::core::GpuBackend::Metal);
            }
            auto resolved = view.tensor_temporal_pipeline_->spatial(image, render_size, size);
            if (!resolved)
                return keep_previous(lfs::format_for_developer(resolved.error()));
            image = std::move(*resolved);
            published_size = size;
            view.tensor_temporal_pipeline_->resetAll();
            if (view.temporal_convergence_.completeSuccessfulFrame())
                requestViewFollowUp(view, DirtyFlag::TEMPORAL);
        } else {
            if (view.tensor_temporal_pipeline_)
                view.tensor_temporal_pipeline_->resetAll();
            if (view.temporal_convergence_.completeSuccessfulFrame())
                requestViewFollowUp(view, DirtyFlag::TEMPORAL);
        }

        ++view.vulkan_viewport_image_generation_;
        view.vulkan_viewport_image_ = image;
        view.viewport_depth_image_ = depth;
        view.viewport_environment_ = buildViewportEnvironment(
            frame_context, frame_settings, environmentBackgroundUsesTransparentViewerCompositing(frame_settings));
        view.viewport_meshes_ = buildViewportMeshes(frame_context, frame_settings);
        view.vulkan_viewport_image_size_ = published_size;
        view.vulkan_viewport_image_alloc_size_ = published_size;
        view.vulkan_viewport_coordinate_size_ = size;
        view.vulkan_viewport_image_flip_y_ = false;
        lfs::rendering::FrameMetadata metadata{
            .viewer_backend = lfs::rendering::ViewerBackend::Metal,
            .depth_panels = {lfs::rendering::FramePanelMetadata{
                .depth = depth,
                .start_position = 0.0f,
                .end_position = 1.0f}},
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
                .size = published_size,
                .alloc_size = published_size,
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

/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#include "viewport_reference_renderer.hpp"
#include "core/logger.hpp"
#include "passes/vulkan_viewport_pass.hpp"
#include "scene_upscaler_plugin.hpp"
#include "vulkan_view_render_state.hpp"
#include "window/vulkan_graphics_context.hpp"
namespace lfs::vis {
    std::optional<std::uint64_t> referenceSceneOutputGeneration(const ViewRenderState& view) {
        const auto* state = vulkanViewRenderStateOrNull(view);
        return state ? state->sceneOutputGeneration() : std::nullopt;
    }
    void clearViewportReferenceOutput(ViewRenderState& view) {
        if (view.reference_state_) view.reference_state_->clearSceneOutput();
    }
    struct ViewportReferenceResources::Impl {
        std::shared_ptr<SharedViewportGpuAssets> assets = std::make_shared<SharedViewportGpuAssets>();
    };
    ViewportReferenceResources::ViewportReferenceResources() : impl_(std::make_unique<Impl>()) {}
    ViewportReferenceResources::~ViewportReferenceResources() = default;
    struct ViewportReferenceRenderer::Impl {
        explicit Impl(std::shared_ptr<SharedViewportGpuAssets> assets) : pass(std::move(assets)) {}
        VulkanViewportPass pass;
        VulkanViewportPassParams params;
        GraphicsContext* graphics = nullptr;
    };
    namespace {
        VulkanViewportPassParams referenceParams(const ViewportFrameDesc& desc, ViewRenderState& view_state) {
            VulkanViewportPassParams params;
            params.frame_slot = desc.frame_slot;
            params.viewport_pos = desc.viewport_pos;
            params.viewport_size = desc.viewport_size;
            params.framebuffer_scale = desc.framebuffer_scale;
            params.background_color = desc.background_color;
            params.scene_image = desc.scene_image;
            params.scene_image_size = desc.scene_image_size;
            params.scene_image_alloc_size = desc.scene_image_alloc_size;
            params.scene_image_flip_y = desc.scene_image_flip_y;
            params.preserve_scene_image_binding = desc.preserve_scene_image_binding;
            params.scene_upscaler = desc.scene_upscaler;
            params.scene_upscaler_mode_unsupported = desc.scene_upscaler_mode_unsupported;
            params.grid_enabled = desc.grid_enabled;
            params.grid_view = desc.grid_view;
            params.grid_projection = desc.grid_projection;
            params.grid_view_projection = desc.grid_view_projection;
            params.grid_view_position = desc.grid_view_position;
            params.grid_plane = desc.grid_plane;
            params.grid_opacity = desc.grid_opacity;
            params.grid_orthographic = desc.grid_orthographic;
            params.grid_overlays = desc.grid_overlays;
            params.vignette_enabled = desc.vignette_enabled;
            params.vignette_intensity = desc.vignette_intensity;
            params.vignette_radius = desc.vignette_radius;
            params.vignette_softness = desc.vignette_softness;
            params.overlay_triangles = desc.overlay_triangles;
            params.shape_overlay_triangles = desc.shape_overlay_triangles;
            params.ui_shape_overlay_triangles = desc.ui_shape_overlay_triangles;
            params.post_ui_overlay_vertex_count = desc.post_ui_overlay_vertex_count;
            params.pivot_overlays = desc.pivot_overlays;
            params.textured_overlays = desc.textured_overlays;
            params.ui_textured_overlays = desc.ui_textured_overlays;
            params.frustum_overlay_data = desc.frustum_overlay_data;
            params.frustum_instances = desc.frustum_instances;
            params.frustum_batches = desc.frustum_batches;
            params.mesh_view_projection = desc.mesh_view_projection;
            params.mesh_camera_position = desc.mesh_camera_position;
            params.mesh_items = desc.mesh_items;
            params.mesh_panels = desc.mesh_panels;
            params.environment = desc.environment;
            const auto frame_slot = desc.frame_slot;
            const auto export_locked = desc.export_locked;
            // Pull GPU mesh / environment frame populated by renderVulkanFrame.
            // vulkan_viewport_pass rasterizes these on the GPU.
            auto mesh_frame = [&] {
                auto& native = vulkanViewRenderState(view_state);
                std::lock_guard lock(native.mesh_frame_mutex);
                return native.mesh_frame;
            }();
            auto temporal_frame = std::move(mesh_frame.temporal);
            params.mesh_view_projection = mesh_frame.view_projection;
            params.mesh_camera_position = mesh_frame.camera_position;
            params.mesh_items = std::move(mesh_frame.items);
            params.mesh_panels = std::move(mesh_frame.panels);
            params.environment = std::move(mesh_frame.environment);
            params.depth_blit = std::move(mesh_frame.depth_blit);
            params.split_view = std::move(mesh_frame.split_view);
            // Late bind: interop-owned scene / depth-blit / split-view fields. Must
            // run after params.split_view is populated (split stitching is gated on
            // params.split_view.enabled).
            vulkanViewRenderState(view_state).viewport_interop_.bindViewportParams(params, frame_slot, export_locked, view_state.frame_lifecycle_service_.isResizeDeferring());

            const glm::ivec2 output_extent = temporal_frame
                                                 ? temporal_frame->input.output_extent
                                                 : glm::ivec2(0);
            const bool temporal_inputs_match =
                temporal_frame.has_value() && !params.split_view.enabled &&
                params.external_scene_image_view != VK_NULL_HANDLE &&
                params.depth_blit.external_image_view != VK_NULL_HANDLE &&
                params.scene_image_size.x > 0 && params.scene_image_size.y > 0 &&
                params.depth_blit.external_image_size == params.scene_image_size &&
                output_extent.x > 0 && output_extent.y > 0 &&
                temporal_frame->input.view.size == params.scene_image_size;
            if (temporal_inputs_match) {
                const bool jitter_enabled = !temporal_frame->input.view.orthographic;
                const glm::ivec2 allocation_extent =
                    params.scene_image_alloc_size.x >= params.scene_image_size.x &&
                            params.scene_image_alloc_size.y >= params.scene_image_size.y
                        ? params.scene_image_alloc_size
                        : params.scene_image_size;
                const glm::ivec2 depth_allocation_extent =
                    params.depth_blit.external_image_allocation_size.x >=
                                params.depth_blit.external_image_size.x &&
                            params.depth_blit.external_image_allocation_size.y >=
                                params.depth_blit.external_image_size.y
                        ? params.depth_blit.external_image_allocation_size
                        : params.depth_blit.external_image_size;
                const SceneDepthContract depth = makeSceneDepthContract(
                    true,
                    SceneDepthStorage::VulkanImage,
                    params.depth_blit.depth_is_ndc ? SceneDepthEncoding::VulkanNdc
                                                   : SceneDepthEncoding::LinearView,
                    params.scene_image_size,
                    params.depth_blit.near_plane,
                    params.depth_blit.far_plane,
                    temporal_frame->input.view.orthographic,
                    params.depth_blit.flip_y);
                params.temporal = VulkanSceneTemporalPipelineRequest{
                    .temporal = {
                        .view = TemporalViewId::Main,
                        .requirements = {
                            .depth = true,
                            .motion = true,
                            .jitter = jitter_enabled,
                            .history_color = true,
                            .history_depth = true,
                        },
                        .frame = temporal_frame->input,
                        .render_extent = params.scene_image_size,
                        .output_extent = output_extent,
                    },
                    .motion = {
                        .enabled = true,
                        .depth_view = params.depth_blit.external_image_view,
                        .current_depth_layout = params.depth_blit.external_image_layout,
                        .depth_generation = params.depth_blit.external_image_generation,
                        .depth = depth,
                        .render_extent = params.scene_image_size,
                        .flip_y = params.scene_image_flip_y,
                    },
                    .resolve = {
                        .enabled = true,
                        .view = TemporalViewId::Main,
                        .current_color_view = params.external_scene_image_view,
                        .current_color_layout = params.external_scene_image_layout,
                        .render_extent = params.scene_image_size,
                        .output_extent = output_extent,
                        .current_allocation_extent = allocation_extent,
                        .history_weight = temporal_frame->resolve_settings.history_weight,
                        .motion_rejection_pixels = temporal_frame->resolve_settings.motion_rejection_pixels,
                        .motion_confidence_pixels = temporal_frame->resolve_settings.motion_confidence_pixels,
                        .current_sharpness = temporal_frame->resolve_settings.current_sharpness,
                        .current_depth = {
                            .enabled = true,
                            .view = TemporalViewId::Main,
                            .current_depth_view = params.depth_blit.external_image_view,
                            .current_depth_layout = params.depth_blit.external_image_layout,
                            .depth = depth,
                            .allocation_extent = depth_allocation_extent,
                        },
                        .depth_relative_threshold = temporal_frame->resolve_settings.depth_relative_threshold,
                        .depth_absolute_threshold = temporal_frame->resolve_settings.depth_absolute_threshold,
                    },
                    .frame_slot = frame_slot,
                };
                if (sceneUpscalerPlugin(params.scene_upscaler) != nullptr &&
                    params.external_scene_image != VK_NULL_HANDLE &&
                    params.depth_blit.external_image != VK_NULL_HANDLE) {
                    VulkanScenePluginPipelineRequest plugin_request{
                        .temporal = *params.temporal,
                        .color_image = params.external_scene_image,
                        .color_format = VK_FORMAT_R8G8B8A8_UNORM,
                        .color_generation = params.external_scene_image_generation,
                        .depth_image = params.depth_blit.external_image,
                        .depth_format = params.depth_blit.external_image_format,
                        .depth_generation = params.depth_blit.external_image_generation,
                        .quality = temporal_frame->quality,
                    };
                    if (validVulkanScenePluginPipelineRequest(plugin_request))
                        params.plugin = plugin_request;
                }
            }

            if (params.split_view.enabled) {
                const auto make_split_temporal_request =
                    [frame_slot](const VulkanSplitViewPanel& panel,
                                 const TemporalViewId view)
                    -> std::optional<VulkanSceneTemporalPipelineRequest> {
                    if (!panel.temporal_input ||
                        panel.external_image_view == VK_NULL_HANDLE ||
                        panel.depth_image_view == VK_NULL_HANDLE ||
                        panel.image_size.x <= 0 || panel.image_size.y <= 0 ||
                        panel.depth_image_size != panel.image_size ||
                        panel.temporal_input->view.size != panel.image_size) {
                        return std::nullopt;
                    }
                    const glm::ivec2 output_extent = panel.temporal_input->output_extent;
                    if (output_extent.x <= 0 || output_extent.y <= 0) {
                        return std::nullopt;
                    }
                    const glm::ivec2 allocation_extent =
                        panel.allocation_size.x >= panel.image_size.x &&
                                panel.allocation_size.y >= panel.image_size.y
                            ? panel.allocation_size
                            : panel.image_size;
                    const glm::ivec2 depth_allocation_extent =
                        panel.depth_allocation_size.x >= panel.depth_image_size.x &&
                                panel.depth_allocation_size.y >= panel.depth_image_size.y
                            ? panel.depth_allocation_size
                            : panel.depth_image_size;
                    const bool jitter_enabled = !panel.temporal_input->view.orthographic;
                    const SceneDepthContract depth = makeSceneDepthContract(
                        true,
                        SceneDepthStorage::VulkanImage,
                        SceneDepthEncoding::LinearView,
                        panel.image_size,
                        panel.temporal_input->view.near_plane,
                        panel.temporal_input->view.far_plane,
                        panel.temporal_input->view.orthographic,
                        panel.flip_y);
                    return VulkanSceneTemporalPipelineRequest{
                        .temporal = {
                            .view = view,
                            .requirements = {
                                .depth = true,
                                .motion = true,
                                .jitter = jitter_enabled,
                                .history_color = true,
                                .history_depth = true,
                            },
                            .frame = *panel.temporal_input,
                            .render_extent = panel.image_size,
                            .output_extent = output_extent,
                        },
                        .motion = {
                            .enabled = true,
                            .depth_view = panel.depth_image_view,
                            .current_depth_layout = panel.depth_image_layout,
                            .depth_generation = panel.depth_image_generation,
                            .depth = depth,
                            .render_extent = panel.image_size,
                            .flip_y = panel.flip_y,
                        },
                        .resolve = {
                            .enabled = true,
                            .view = view,
                            .current_color_view = panel.external_image_view,
                            .current_color_layout = panel.external_image_layout,
                            .render_extent = panel.image_size,
                            .output_extent = output_extent,
                            .current_allocation_extent = allocation_extent,
                            .history_weight = panel.temporal_settings.history_weight,
                            .motion_rejection_pixels = panel.temporal_settings.motion_rejection_pixels,
                            .motion_confidence_pixels = panel.temporal_settings.motion_confidence_pixels,
                            .current_sharpness = panel.temporal_settings.current_sharpness,
                            .current_depth = {
                                .enabled = true,
                                .view = view,
                                .current_depth_view = panel.depth_image_view,
                                .current_depth_layout = panel.depth_image_layout,
                                .depth = depth,
                                .allocation_extent = depth_allocation_extent,
                            },
                            .depth_relative_threshold = panel.temporal_settings.depth_relative_threshold,
                            .depth_absolute_threshold = panel.temporal_settings.depth_absolute_threshold,
                        },
                        .frame_slot = frame_slot,
                    };
                };
                params.split_temporal[0] = make_split_temporal_request(
                    params.split_view.left, TemporalViewId::SplitLeft);
                params.split_temporal[1] = make_split_temporal_request(
                    params.split_view.right, TemporalViewId::SplitRight);
                if (sceneUpscalerPlugin(params.scene_upscaler) != nullptr) {
                    const auto make_split_plugin_request =
                        [](const VulkanSplitViewPanel& panel,
                           const std::optional<VulkanSceneTemporalPipelineRequest>& temporal)
                        -> std::optional<VulkanScenePluginPipelineRequest> {
                        if (!temporal ||
                            panel.external_image == VK_NULL_HANDLE ||
                            panel.depth_image == VK_NULL_HANDLE) {
                            return std::nullopt;
                        }
                        VulkanScenePluginPipelineRequest request{
                            .temporal = *temporal,
                            .color_image = panel.external_image,
                            .color_format = VK_FORMAT_R8G8B8A8_UNORM,
                            .color_generation = panel.external_image_generation,
                            .depth_image = panel.depth_image,
                            .depth_format = panel.depth_image_format,
                            .depth_generation = panel.depth_image_generation,
                            .quality = panel.temporal_quality,
                        };
                        if (!validVulkanScenePluginPipelineRequest(request))
                            return std::nullopt;
                        return request;
                    };
                    params.split_plugin[0] = make_split_plugin_request(
                        params.split_view.left, params.split_temporal[0]);
                    params.split_plugin[1] = make_split_plugin_request(
                        params.split_view.right, params.split_temporal[1]);
                }
            }
            return params;
        }
    } // namespace
    void snapshotViewportReference(const ViewRenderState& view, ViewportFrameDesc& desc) {
        desc.scene_image = view.vulkan_viewport_image_;
        desc.scene_image_size = view.vulkan_viewport_image_size_;
        desc.scene_image_alloc_size = view.vulkan_viewport_image_alloc_size_;
        desc.scene_image_flip_y = view.vulkan_viewport_image_flip_y_;
        desc.scene_outputs = {{.target = view.main_render_target_, .color = desc.scene_image, .size = desc.scene_image_size, .allocation_size = desc.scene_image_alloc_size, .generation = view.presentedImageGeneration(), .flip_y = desc.scene_image_flip_y}};
        const auto* state = vulkanViewRenderStateOrNull(view);
        if (!state)
            return;
        std::lock_guard lock(state->mesh_frame_mutex);
        const auto& frame = state->mesh_frame;
        desc.mesh_view_projection = frame.view_projection;
        desc.mesh_camera_position = frame.camera_position;
        desc.mesh_items = frame.items;
        desc.mesh_panels = frame.panels;
        desc.environment = frame.environment;
        desc.depth_blit = {.depth = frame.depth_blit.depth, .depth_is_ndc = frame.depth_blit.depth_is_ndc, .flip_y = frame.depth_blit.flip_y, .near_plane = frame.depth_blit.near_plane, .far_plane = frame.depth_blit.far_plane, .uv_scale = frame.depth_blit.uv_scale, .uv_clamp_max = frame.depth_blit.uv_clamp_max};
        desc.scene_outputs.front().depth = frame.depth_blit.depth;
        if (frame.temporal) {
            desc.temporal = ViewportTemporalRequest{view.main_render_target_, frame.temporal->input,
                                                    frame.temporal->resolve_settings, frame.temporal->quality};
            if (sceneUpscalerPlugin(desc.scene_upscaler))
                desc.plugin = ViewportPluginRequest{*desc.temporal, desc.scene_upscaler};
        }
        const auto& split = frame.split_view;
        desc.split_view.enabled = split.enabled;
        desc.split_view.loss_visualization = split.loss_visualization;
        desc.split_view.split_position = split.split_position;
        desc.split_view.content_rect = split.content_rect;
        desc.split_view.coordinate_extent = split.coordinate_extent;
        desc.split_view.background = split.background;
        const auto panel = [&](const VulkanSplitViewPanel& source, ViewportSplitPanel& dest, RenderTargetId target, size_t index) {
            dest = {.target = target, .image = source.image, .start_position = source.start_position, .end_position = source.end_position, .normalize_x_to_panel = source.normalize_x_to_panel, .flip_y = source.flip_y, .spatial_filter = source.spatial_filter, .image_size = source.image_size, .allocation_size = source.allocation_size, .uv_scale = source.uv_scale, .uv_clamp_max = source.uv_clamp_max, .texcoord_scale = source.texcoord_scale, .texcoord_offset = source.texcoord_offset};
            desc.scene_outputs.push_back({.target = target, .color = source.image, .size = source.image_size, .allocation_size = source.allocation_size, .generation = source.external_image_generation, .flip_y = source.flip_y});
            if (source.temporal_input) {
                desc.split_temporal[index] = ViewportTemporalRequest{target, *source.temporal_input, source.temporal_settings, source.temporal_quality};
                if (sceneUpscalerPlugin(desc.scene_upscaler))
                    desc.split_plugin[index] = ViewportPluginRequest{*desc.split_temporal[index], desc.scene_upscaler};
            }
        };
        if (split.enabled) {
            panel(split.left, desc.split_view.left, view.split_left_render_target_, 0);
            panel(split.right, desc.split_view.right, view.split_right_render_target_, 1);
        }
    }
    ViewportReferenceRenderer::ViewportReferenceRenderer(std::shared_ptr<ViewportReferenceResources> shared) {
        if (!shared)
            shared = std::make_shared<ViewportReferenceResources>();
        impl_ = std::make_unique<Impl>(shared->impl_->assets);
    }
    ViewportReferenceRenderer::~ViewportReferenceRenderer() = default;
    bool ViewportReferenceRenderer::initialize(GraphicsContext& graphics) {
        impl_->graphics = &graphics;
        auto* context = vulkanContextOrNull(&graphics);
        return context && impl_->pass.init(*context);
    }
    void ViewportReferenceRenderer::prepare(GraphicsContext& graphics, const ViewportFrameDesc& desc, ViewRenderState& view) {
        impl_->params = referenceParams(desc, view);
        if (auto* context = vulkanContextOrNull(&graphics))
            impl_->pass.prepare(*context, impl_->params);
    }
    bool ViewportReferenceRenderer::hasPreRenderWork(const ViewportFrameDesc&) const { return impl_->pass.hasPreRenderWork(impl_->params); }
    bool ViewportReferenceRenderer::recordPreRenderWork(const GraphicsFrame& frame, const ViewportFrameDesc&) {
        const auto* native = vulkanFrameOrNull(impl_->graphics, frame);
        return native && impl_->pass.recordPreRenderWork(native->command_buffer, impl_->params);
    }
    void ViewportReferenceRenderer::record(const GraphicsFrame& frame, const ViewportFrameDesc&) {
        if (const auto* native = vulkanFrameOrNull(impl_->graphics, frame))
            impl_->pass.record(native->command_buffer, native->extent, impl_->params);
    }
    void ViewportReferenceRenderer::recordFrame(const GraphicsFrame& frame, const ViewportFrameDesc& desc, ViewRenderState& view) {
        const auto* native = vulkanFrameOrNull(impl_->graphics, frame);
        auto* context = vulkanContextOrNull(impl_->graphics);
        if (!native || !context)
            return;
        const bool temporal = hasPreRenderWork(desc);
        const bool closed = context->finishActiveRendering(native->command_buffer);
        bool ready = closed;
        if (!closed)
            LOG_ERROR("Unable to close dynamic rendering before viewport pre-render work: {}", context->lastError());
        if (ready) {
            auto& interop = vulkanViewRenderState(view).viewport_interop_;
            interop.recordFrameBarriers(native->command_buffer, *context);
            for (const auto completion : interop.frameCompletions()) {
                const VkPipelineStageFlags stage = temporal ? VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT : VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT;
                if (!context->addFrameTimelineWait(completion.semaphore, completion.value, stage)) {
                    LOG_ERROR("Unable to wait on scene frame completion timeline: {}", context->lastError());
                    ready = false;
                }
            }
        }
        if (ready && temporal)
            static_cast<void>(recordPreRenderWork(frame, desc));
        if (closed) {
            const bool restarted = context->restartActiveRendering(native->command_buffer, *native);
            ready = ready && restarted;
            if (!restarted)
                LOG_ERROR("Unable to restart dynamic rendering after viewport pre-render work: {}", context->lastError());
        }
        if (ready)
            record(frame, desc);
    }

    void prepareViewportReference(GraphicsContext& graphics, ViewRenderState& view, bool export_locked, bool resize_deferring) {
        if (auto* context = vulkanContextOrNull(&graphics)) {
            auto& interop = vulkanViewRenderState(view).viewport_interop_;
            if (export_locked)
                interop.syncUnsubmittedLayoutCommits(*context);
            else
                interop.prepareFrame(*context, resize_deferring);
        }
    }
    size_t viewportReferenceFramesInFlight(GraphicsContext& graphics) {
        auto* context = vulkanContextOrNull(&graphics);
        return context ? context->framesInFlight() : 0;
    }
    void clearViewportReference(ViewRenderState& view) {
        vulkanViewRenderState(view).viewport_interop_.setSceneImage(nullptr, {}, false, 0);
    }
    void shutdownViewportReference(ViewRenderState& view, GraphicsContext* graphics) {
        vulkanViewRenderState(view).viewport_interop_.shutdown(vulkanContextOrNull(graphics));
    }
    size_t viewportReferenceMemoryBytes(GraphicsContext& graphics, size_t blocks, size_t allocations) {
        auto* context = vulkanContextOrNull(&graphics);
        return context ? context->queryVmaUsedBytes(blocks, allocations) : 0;
    }
    void ViewportReferenceRenderer::prepareImport(GraphicsContext& graphics, const ViewportFrameDesc& desc,
                                                  ViewRenderState& view, ViewportReferenceRenderer* resident) {
        impl_->params = referenceParams(desc, view);
        if (auto* context = vulkanContextOrNull(&graphics))
            impl_->pass.prepareImport(*context, impl_->params, resident ? &resident->impl_->pass : nullptr);
    }
    void ViewportReferenceRenderer::discardImportMesh(uint64_t mesh_id) { impl_->pass.discardImportMesh(mesh_id); }
    SceneUpscalerSelection ViewportReferenceRenderer::sceneUpscalerSelection() const { return impl_->pass.sceneUpscalerSelection(); }
    std::unique_ptr<ViewportReferenceRenderer> createViewportReferenceRenderer(
        GraphicsContext&, std::shared_ptr<ViewportReferenceResources>& resources) {
        if (!resources)
            resources = std::make_shared<ViewportReferenceResources>();
        return std::make_unique<ViewportReferenceRenderer>(resources);
    }
} // namespace lfs::vis

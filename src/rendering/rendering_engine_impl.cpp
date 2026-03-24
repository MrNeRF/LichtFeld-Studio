/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "rendering_engine_impl.hpp"
#include "core/logger.hpp"
#include "core/mesh_data.hpp"
#include "core/point_cloud.hpp"
#include "framebuffer_factory.hpp"
#include "geometry/bounding_box.hpp"
#include "gl_state_guard.hpp"
#include "rendering/render_constants.hpp"
#include <cuda_runtime.h>
#include <limits>
#include <vector>

namespace lfs::rendering {

    namespace {
        struct GaussianRasterResources {
            std::unique_ptr<lfs::geometry::BoundingBox> temp_crop_box;
            Tensor crop_box_transform_tensor;
            Tensor crop_box_min_tensor;
            Tensor crop_box_max_tensor;
            Tensor ellipsoid_transform_tensor;
            Tensor ellipsoid_radii_tensor;
            Tensor view_volume_transform_tensor;
            Tensor view_volume_min_tensor;
            Tensor view_volume_max_tensor;
        };

        void applyCropBoxToPipeline(RenderingPipeline::RasterRequest& pipeline_req,
                                    const std::optional<GaussianScopedBoxFilter>& crop_region,
                                    GaussianRasterResources& resources) {
            if (!crop_region.has_value()) {
                return;
            }

            resources.temp_crop_box = std::make_unique<lfs::geometry::BoundingBox>();
            resources.temp_crop_box->setBounds(crop_region->bounds.min, crop_region->bounds.max);

            lfs::geometry::EuclideanTransform transform(crop_region->bounds.transform);
            resources.temp_crop_box->setworld2BBox(transform);

            pipeline_req.crop_box = resources.temp_crop_box.get();

            const glm::mat4& w2b = crop_region->bounds.transform;
            std::vector<float> transform_data(16);
            for (int row = 0; row < 4; ++row) {
                for (int col = 0; col < 4; ++col) {
                    transform_data[row * 4 + col] = w2b[col][row];
                }
            }
            resources.crop_box_transform_tensor =
                Tensor::from_vector(transform_data, {4, 4}, lfs::core::Device::CPU).cuda();

            std::vector<float> min_data = {
                crop_region->bounds.min.x,
                crop_region->bounds.min.y,
                crop_region->bounds.min.z};
            resources.crop_box_min_tensor =
                Tensor::from_vector(min_data, {3}, lfs::core::Device::CPU).cuda();

            std::vector<float> max_data = {
                crop_region->bounds.max.x,
                crop_region->bounds.max.y,
                crop_region->bounds.max.z};
            resources.crop_box_max_tensor =
                Tensor::from_vector(max_data, {3}, lfs::core::Device::CPU).cuda();

            pipeline_req.crop_box_transform = &resources.crop_box_transform_tensor;
            pipeline_req.crop_box_min = &resources.crop_box_min_tensor;
            pipeline_req.crop_box_max = &resources.crop_box_max_tensor;
            pipeline_req.crop_inverse = crop_region->inverse;
            pipeline_req.crop_desaturate = crop_region->desaturate;
            pipeline_req.crop_parent_node_index = crop_region->parent_node_index;
        }

        void applyEllipsoidToPipeline(RenderingPipeline::RasterRequest& pipeline_req,
                                      const std::optional<GaussianScopedEllipsoidFilter>& ellipsoid_region,
                                      GaussianRasterResources& resources) {
            if (!ellipsoid_region.has_value()) {
                return;
            }

            const glm::mat4& w2e = ellipsoid_region->bounds.transform;
            std::vector<float> transform_data(16);
            for (int row = 0; row < 4; ++row) {
                for (int col = 0; col < 4; ++col) {
                    transform_data[row * 4 + col] = w2e[col][row];
                }
            }
            resources.ellipsoid_transform_tensor =
                Tensor::from_vector(transform_data, {4, 4}, lfs::core::Device::CPU).cuda();

            std::vector<float> radii_data = {
                ellipsoid_region->bounds.radii.x,
                ellipsoid_region->bounds.radii.y,
                ellipsoid_region->bounds.radii.z};
            resources.ellipsoid_radii_tensor =
                Tensor::from_vector(radii_data, {3}, lfs::core::Device::CPU).cuda();

            pipeline_req.ellipsoid_transform = &resources.ellipsoid_transform_tensor;
            pipeline_req.ellipsoid_radii = &resources.ellipsoid_radii_tensor;
            pipeline_req.ellipsoid_inverse = ellipsoid_region->inverse;
            pipeline_req.ellipsoid_desaturate = ellipsoid_region->desaturate;
            pipeline_req.ellipsoid_parent_node_index = ellipsoid_region->parent_node_index;
        }

        void applyViewVolumeToPipeline(RenderingPipeline::RasterRequest& pipeline_req,
                                       const std::optional<BoundingBox>& view_volume,
                                       const bool view_volume_cull,
                                       GaussianRasterResources& resources) {
            if (!view_volume.has_value()) {
                return;
            }

            const glm::mat4& w2b = view_volume->transform;
            std::vector<float> transform_data(16);
            for (int row = 0; row < 4; ++row) {
                for (int col = 0; col < 4; ++col) {
                    transform_data[row * 4 + col] = w2b[col][row];
                }
            }
            resources.view_volume_transform_tensor =
                Tensor::from_vector(transform_data, {4, 4}, lfs::core::Device::CPU).cuda();

            std::vector<float> min_data = {view_volume->min.x, view_volume->min.y, view_volume->min.z};
            resources.view_volume_min_tensor =
                Tensor::from_vector(min_data, {3}, lfs::core::Device::CPU).cuda();

            std::vector<float> max_data = {view_volume->max.x, view_volume->max.y, view_volume->max.z};
            resources.view_volume_max_tensor =
                Tensor::from_vector(max_data, {3}, lfs::core::Device::CPU).cuda();

            pipeline_req.view_volume_transform = &resources.view_volume_transform_tensor;
            pipeline_req.view_volume_min = &resources.view_volume_min_tensor;
            pipeline_req.view_volume_max = &resources.view_volume_max_tensor;
            pipeline_req.view_volume_cull = view_volume_cull;
        }

        [[nodiscard]] RenderingPipeline::RasterRequest makeGaussianPipelineRequest(
            const ViewportRenderRequest& request) {
            return RenderingPipeline::RasterRequest{
                .view_rotation = request.frame_view.rotation,
                .view_translation = request.frame_view.translation,
                .viewport_size = request.frame_view.size,
                .focal_length_mm = request.frame_view.focal_length_mm,
                .scaling_modifier = request.scaling_modifier,
                .antialiasing = request.antialiasing,
                .mip_filter = request.mip_filter,
                .sh_degree = request.sh_degree,
                .render_mode = RenderMode::RGB,
                .crop_box = nullptr,
                .background_color = request.frame_view.background_color,
                .voxel_size = 0.01f,
                .gut = request.gut,
                .equirectangular = request.equirectangular,
                .show_rings = request.overlay.markers.show_rings,
                .ring_width = request.overlay.markers.ring_width,
                .show_center_markers = request.overlay.markers.show_center_markers,
                .model_transforms = request.scene.model_transforms ? *request.scene.model_transforms
                                                                   : std::vector<glm::mat4>{},
                .transform_indices = request.scene.transform_indices,
                .selection_mask = request.overlay.emphasis.mask,
                .cursor_active = request.overlay.cursor.enabled,
                .cursor_x = request.overlay.cursor.cursor.x,
                .cursor_y = request.overlay.cursor.cursor.y,
                .cursor_radius = request.overlay.cursor.radius,
                .preview_selection_add_mode = request.overlay.emphasis.transient_mask.additive,
                .preview_selection_tensor = request.overlay.emphasis.transient_mask.mask,
                .cursor_saturation_preview = request.overlay.cursor.saturation_preview,
                .cursor_saturation_amount = request.overlay.cursor.saturation_amount,
                .hovered_depth_id = nullptr,
                .focused_gaussian_id = request.overlay.emphasis.focused_gaussian_id,
                .far_plane = request.frame_view.far_plane,
                .emphasized_node_mask = request.overlay.emphasis.emphasized_node_mask,
                .node_visibility_mask = request.scene.node_visibility_mask,
                .dim_non_emphasized = request.overlay.emphasis.dim_non_emphasized,
                .emphasis_flash_intensity = request.overlay.emphasis.flash_intensity,
                .orthographic = request.frame_view.orthographic,
                .ortho_scale = request.frame_view.ortho_scale};
        }

        [[nodiscard]] RenderingPipeline::RasterRequest makeHoveredGaussianQueryPipelineRequest(
            const HoveredGaussianQueryRequest& request,
            unsigned long long* const hovered_depth_id) {
            return RenderingPipeline::RasterRequest{
                .view_rotation = request.frame_view.rotation,
                .view_translation = request.frame_view.translation,
                .viewport_size = request.frame_view.size,
                .focal_length_mm = request.frame_view.focal_length_mm,
                .scaling_modifier = request.scaling_modifier,
                .antialiasing = false,
                .mip_filter = request.mip_filter,
                .sh_degree = request.sh_degree,
                .render_mode = RenderMode::RGB,
                .crop_box = nullptr,
                .background_color = request.frame_view.background_color,
                .voxel_size = 0.01f,
                .gut = request.gut,
                .equirectangular = request.equirectangular,
                .show_rings = false,
                .ring_width = 0.0f,
                .show_center_markers = false,
                .model_transforms = request.scene.model_transforms ? *request.scene.model_transforms
                                                                   : std::vector<glm::mat4>{},
                .transform_indices = request.scene.transform_indices,
                .selection_mask = nullptr,
                .cursor_active = true,
                .cursor_x = request.cursor.x,
                .cursor_y = request.cursor.y,
                .cursor_radius = 0.0f,
                .preview_selection_add_mode = true,
                .preview_selection_tensor = nullptr,
                .cursor_saturation_preview = false,
                .cursor_saturation_amount = 0.0f,
                .hovered_depth_id = hovered_depth_id,
                .focused_gaussian_id = -1,
                .far_plane = request.frame_view.far_plane,
                .emphasized_node_mask = {},
                .node_visibility_mask = request.scene.node_visibility_mask,
                .dim_non_emphasized = false,
                .emphasis_flash_intensity = 0.0f,
                .orthographic = request.frame_view.orthographic,
                .ortho_scale = request.frame_view.ortho_scale};
        }

        [[nodiscard]] PointCloudCropParams makePointCloudCropParams(const PointCloudRenderRequest& request) {
            PointCloudCropParams crop_params;
            if (request.filters.crop_box.has_value()) {
                crop_params.enabled = true;
                crop_params.transform = request.filters.crop_box->transform;
                crop_params.min = request.filters.crop_box->min;
                crop_params.max = request.filters.crop_box->max;
                crop_params.inverse = request.filters.crop_inverse;
                crop_params.desaturate = request.filters.crop_desaturate;
            }
            return crop_params;
        }

        [[nodiscard]] RenderingPipeline::RasterRequest makePointCloudPipelineRequest(
            const PointCloudRenderRequest& request) {
            return RenderingPipeline::RasterRequest{
                .view_rotation = request.frame_view.rotation,
                .view_translation = request.frame_view.translation,
                .viewport_size = request.frame_view.size,
                .focal_length_mm = request.frame_view.focal_length_mm,
                .scaling_modifier = request.render.scaling_modifier,
                .antialiasing = false,
                .mip_filter = false,
                .sh_degree = 0,
                .render_mode = RenderMode::RGB,
                .crop_box = nullptr,
                .background_color = request.frame_view.background_color,
                .voxel_size = request.render.voxel_size,
                .gut = false,
                .equirectangular = request.render.equirectangular,
                .show_rings = false,
                .ring_width = 0.0f,
                .show_center_markers = false,
                .model_transforms = request.scene.model_transforms ? *request.scene.model_transforms
                                                                   : std::vector<glm::mat4>{},
                .transform_indices = request.scene.transform_indices,
                .selection_mask = nullptr,
                .cursor_active = false,
                .cursor_x = 0.0f,
                .cursor_y = 0.0f,
                .cursor_radius = 0.0f,
                .preview_selection_add_mode = true,
                .preview_selection_tensor = nullptr,
                .cursor_saturation_preview = false,
                .cursor_saturation_amount = 0.0f,
                .hovered_depth_id = nullptr,
                .focused_gaussian_id = -1,
                .far_plane = request.frame_view.far_plane,
                .emphasized_node_mask = {},
                .orthographic = request.frame_view.orthographic,
                .ortho_scale = request.frame_view.ortho_scale,
                .point_cloud_crop_params = makePointCloudCropParams(request)};
        }

    } // namespace

    RenderingEngineImpl::RenderingEngineImpl() {
        LOG_DEBUG("Initializing RenderingEngineImpl");
    };

    RenderingEngineImpl::~RenderingEngineImpl() {
        shutdown();
    }

    Result<void> RenderingEngineImpl::initialize() {
        LOG_TIMER("RenderingEngine::initialize");

        if (quad_shader_.valid()) {
            LOG_TRACE("RenderingEngine already initialized, skipping");
            return {};
        }

        LOG_INFO("Initializing rendering engine...");

        pipeline_.setRenderTargetPool(&render_target_pool_);
        mesh_renderer_.setRenderTargetPool(&render_target_pool_);
        screen_renderer_ = std::make_shared<ScreenQuadRenderer>(getPreferredFrameBufferMode());

        split_view_renderer_ = std::make_unique<SplitViewRenderer>();
        if (auto result = split_view_renderer_->initialize(); !result) {
            LOG_ERROR("Failed to initialize split view renderer: {}", result.error());
            shutdown();
            return std::unexpected(result.error());
        }
        LOG_DEBUG("Split view renderer initialized");

        if (auto result = grid_renderer_.init(); !result) {
            LOG_ERROR("Failed to initialize grid renderer: {}", result.error());
            shutdown();
            return std::unexpected(result.error());
        }
        LOG_DEBUG("Grid renderer initialized");

        if (auto result = bbox_renderer_.init(); !result) {
            LOG_ERROR("Failed to initialize bounding box renderer: {}", result.error());
            shutdown();
            return std::unexpected(result.error());
        }
        LOG_DEBUG("Bounding box renderer initialized");

        if (auto result = ellipsoid_renderer_.init(); !result) {
            LOG_ERROR("Failed to initialize ellipsoid renderer: {}", result.error());
            shutdown();
            return std::unexpected(result.error());
        }
        LOG_DEBUG("Ellipsoid renderer initialized");

        if (auto result = axes_renderer_.init(); !result) {
            LOG_ERROR("Failed to initialize axes renderer: {}", result.error());
            shutdown();
            return std::unexpected(result.error());
        }
        LOG_DEBUG("Axes renderer initialized");

        if (auto result = pivot_renderer_.init(); !result) {
            LOG_ERROR("Failed to initialize pivot renderer: {}", result.error());
            shutdown();
            return std::unexpected(result.error());
        }
        LOG_DEBUG("Pivot renderer initialized");

        if (auto result = viewport_gizmo_.initialize(); !result) {
            LOG_ERROR("Failed to initialize viewport gizmo: {}", result.error());
            shutdown();
            return std::unexpected(result.error());
        }
        LOG_DEBUG("Viewport gizmo initialized");

        if (auto result = camera_frustum_renderer_.init(); !result) {
            LOG_ERROR("Failed to initialize camera frustum renderer: {}", result.error());
        } else {
            LOG_DEBUG("Camera frustum renderer initialized");
        }

        if (auto result = mesh_renderer_.initialize(); !result) {
            LOG_WARN("Failed to initialize mesh renderer: {}", result.error());
        } else {
            LOG_DEBUG("Mesh renderer initialized");
        }

        if (auto result = depth_compositor_.initialize(); !result) {
            LOG_WARN("Failed to initialize depth compositor: {}", result.error());
        } else {
            LOG_DEBUG("Depth compositor initialized");
        }

        auto shader_result = initializeShaders();
        if (!shader_result) {
            LOG_ERROR("Failed to initialize shaders: {}", shader_result.error());
            shutdown();
            return std::unexpected(shader_result.error());
        }

        LOG_INFO("Rendering engine initialized successfully");
        return {};
    }

    void RenderingEngineImpl::shutdown() {
        LOG_DEBUG("Shutting down rendering engine");
        quad_shader_ = ManagedShader();
        invalidatePresentUploadCache();
        if (hovered_depth_id_device_) {
            cudaFree(hovered_depth_id_device_);
            hovered_depth_id_device_ = nullptr;
        }
        if (hovered_depth_id_host_) {
            cudaFreeHost(hovered_depth_id_host_);
            hovered_depth_id_host_ = nullptr;
        }
#ifdef CUDA_GL_INTEROP_ENABLED
        gpu_frame_readback_interop_.reset();
        gpu_frame_readback_source_ = 0;
        gpu_frame_readback_size_ = {0, 0};
#endif
        gpu_frame_readback_fbo_ = FBO();
        pipeline_.resetResources();
        render_target_pool_.clear();
        screen_renderer_.reset();
        split_view_renderer_.reset();
        viewport_gizmo_.shutdown();
    }

    bool RenderingEngineImpl::isInitialized() const {
        return quad_shader_.valid() && screen_renderer_;
    }

    Result<void> RenderingEngineImpl::initializeShaders() {
        LOG_TIMER_TRACE("RenderingEngineImpl::initializeShaders");

        auto result = load_shader("screen_quad", "screen_quad.vert", "screen_quad.frag", true);
        if (!result) {
            LOG_ERROR("Failed to create screen quad shader: {}", result.error().what());
            return std::unexpected(std::string("Failed to create shaders: ") + result.error().what());
        }
        quad_shader_ = std::move(*result);
        LOG_DEBUG("Screen quad shader loaded successfully");
        return {};
    }

    Result<RenderingPipeline::ImageRenderResult> RenderingEngineImpl::renderGaussiansRasterResult(
        const lfs::core::SplatData& splat_data,
        const ViewportRenderRequest& request) {

        if (!isInitialized()) {
            LOG_ERROR("Rendering engine not initialized");
            return std::unexpected("Rendering engine not initialized");
        }

        if (request.frame_view.size.x <= 0 || request.frame_view.size.y <= 0 ||
            request.frame_view.size.x > MAX_VIEWPORT_SIZE || request.frame_view.size.y > MAX_VIEWPORT_SIZE) {
            LOG_ERROR("Invalid viewport dimensions: {}x{}", request.frame_view.size.x, request.frame_view.size.y);
            return std::unexpected("Invalid viewport dimensions");
        }

        LOG_TRACE("Rendering gaussians with viewport {}x{}",
                  request.frame_view.size.x, request.frame_view.size.y);

        auto pipeline_req = makeGaussianPipelineRequest(request);
        GaussianRasterResources raster_resources;
        applyCropBoxToPipeline(pipeline_req, request.filters.crop_region, raster_resources);
        applyEllipsoidToPipeline(pipeline_req, request.filters.ellipsoid_region, raster_resources);
        applyViewVolumeToPipeline(
            pipeline_req, request.filters.view_volume, request.filters.cull_outside_view_volume, raster_resources);

        auto pipeline_result = pipeline_.renderGaussianImage(splat_data, pipeline_req);

        if (!pipeline_result) {
            LOG_ERROR("Pipeline render failed: {}", pipeline_result.error());
            return std::unexpected(pipeline_result.error());
        }

        return *pipeline_result;
    }

    FrameMetadata RenderingEngineImpl::makeFrameMetadata(const RenderingPipeline::ImageRenderResult& result) {
        return FrameMetadata{
            .depth = result.depth.is_valid() ? std::make_shared<Tensor>(result.depth) : nullptr,
            .valid = result.valid,
            .depth_is_ndc = result.depth_is_ndc,
            .external_depth_texture = result.external_depth_texture,
            .near_plane = result.near_plane,
            .far_plane = result.far_plane,
            .orthographic = result.orthographic};
    }

    Result<GpuFrame> RenderingEngineImpl::uploadRenderResultToGpuFrame(
        const RenderingPipeline::ImageRenderResult& result,
        const glm::ivec2& viewport_size) {
        if (auto upload_result = RenderingPipeline::uploadToScreen(result, *screen_renderer_, viewport_size);
            !upload_result) {
            invalidatePresentUploadCache();
            return std::unexpected(upload_result.error());
        }

        invalidatePresentUploadCache();

        const glm::vec2 texcoord_scale = screen_renderer_->getTexcoordScale();
        const GLuint uploaded_depth_texture =
            result.external_depth_texture != 0
                ? result.external_depth_texture
                : (result.depth.is_valid() ? screen_renderer_->getUploadedDepthTexture() : 0);

        return GpuFrame{
            .color = {.id = screen_renderer_->getUploadedColorTexture(),
                      .size = viewport_size,
                      .texcoord_scale = texcoord_scale},
            .depth = {.id = uploaded_depth_texture,
                      .size = viewport_size,
                      .texcoord_scale = texcoord_scale},
            .depth_is_ndc = result.depth_is_ndc,
            .near_plane = result.near_plane,
            .far_plane = result.far_plane,
            .orthographic = result.orthographic};
    }

    Result<GaussianGpuFrameResult> RenderingEngineImpl::renderGaussiansGpuFrame(
        const lfs::core::SplatData& splat_data,
        const ViewportRenderRequest& request) {
        auto raster_result = renderGaussiansRasterResult(splat_data, request);
        if (!raster_result) {
            return std::unexpected(raster_result.error());
        }

        auto gpu_frame = uploadRenderResultToGpuFrame(*raster_result, request.frame_view.size);
        if (!gpu_frame) {
            return std::unexpected(gpu_frame.error());
        }

        return GaussianGpuFrameResult{
            .frame = *gpu_frame,
            .metadata = makeFrameMetadata(*raster_result)};
    }

    Result<GaussianImageResult> RenderingEngineImpl::renderGaussiansImage(
        const lfs::core::SplatData& splat_data,
        const ViewportRenderRequest& request) {

        auto raster_result = renderGaussiansRasterResult(splat_data, request);
        if (!raster_result) {
            return std::unexpected(raster_result.error());
        }

        auto image = std::make_shared<Tensor>(std::move(raster_result->image));
        return GaussianImageResult{
            .image = std::move(image),
            .metadata = makeFrameMetadata(*raster_result)};
    }

    Result<GpuFrame> RenderingEngineImpl::renderPointCloudGpuFrame(
        const lfs::core::SplatData& splat_data,
        const PointCloudRenderRequest& request) {

        if (!isInitialized()) {
            LOG_ERROR("Rendering engine not initialized");
            return std::unexpected("Rendering engine not initialized");
        }

        if (request.frame_view.size.x <= 0 || request.frame_view.size.y <= 0 ||
            request.frame_view.size.x > MAX_VIEWPORT_SIZE || request.frame_view.size.y > MAX_VIEWPORT_SIZE) {
            LOG_ERROR("Invalid viewport dimensions: {}x{}", request.frame_view.size.x, request.frame_view.size.y);
            return std::unexpected("Invalid viewport dimensions");
        }

        LOG_TRACE("Rendering splat-backed point cloud GPU frame with viewport {}x{}",
                  request.frame_view.size.x, request.frame_view.size.y);

        auto pipeline_req = makePointCloudPipelineRequest(request);
        auto pipeline_result = pipeline_.renderPointCloudGpuFrame(splat_data, pipeline_req);
        if (!pipeline_result) {
            LOG_ERROR("Splat-backed point cloud GPU-frame render failed: {}", pipeline_result.error());
            return std::unexpected(pipeline_result.error());
        }

        return *pipeline_result;
    }

    Result<PointCloudImageResult> RenderingEngineImpl::renderPointCloudImage(
        const lfs::core::SplatData& splat_data,
        const PointCloudRenderRequest& request) {

        if (!isInitialized()) {
            LOG_ERROR("Rendering engine not initialized");
            return std::unexpected("Rendering engine not initialized");
        }

        if (request.frame_view.size.x <= 0 || request.frame_view.size.y <= 0 ||
            request.frame_view.size.x > MAX_VIEWPORT_SIZE || request.frame_view.size.y > MAX_VIEWPORT_SIZE) {
            LOG_ERROR("Invalid viewport dimensions: {}x{}", request.frame_view.size.x, request.frame_view.size.y);
            return std::unexpected("Invalid viewport dimensions");
        }

        LOG_TRACE("Rendering splat-backed point cloud image with viewport {}x{}",
                  request.frame_view.size.x, request.frame_view.size.y);

        auto pipeline_req = makePointCloudPipelineRequest(request);
        auto pipeline_result = pipeline_.renderPointCloudImage(splat_data, pipeline_req);
        if (!pipeline_result) {
            LOG_ERROR("Splat-backed point cloud image render failed: {}", pipeline_result.error());
            return std::unexpected(pipeline_result.error());
        }

        auto image = std::make_shared<Tensor>(std::move(pipeline_result->image));
        return PointCloudImageResult{
            .image = std::move(image),
            .metadata = makeFrameMetadata(*pipeline_result)};
    }

    bool RenderingEngineImpl::ensureHoveredDepthQueryBuffersAllocated() {
        if (!hovered_depth_id_device_) {
            if (cudaMalloc(&hovered_depth_id_device_, sizeof(unsigned long long)) != cudaSuccess) {
                LOG_WARN("Failed to allocate hovered-depth device buffer");
                hovered_depth_id_device_ = nullptr;
                return false;
            }
        }

        if (!hovered_depth_id_host_) {
            if (cudaMallocHost(&hovered_depth_id_host_, sizeof(unsigned long long)) != cudaSuccess) {
                LOG_WARN("Failed to allocate hovered-depth host buffer");
                if (hovered_depth_id_device_) {
                    cudaFree(hovered_depth_id_device_);
                    hovered_depth_id_device_ = nullptr;
                }
                hovered_depth_id_host_ = nullptr;
                return false;
            }
        }

        return true;
    }

    Result<std::optional<int>> RenderingEngineImpl::queryHoveredGaussianId(
        const lfs::core::SplatData& splat_data,
        const HoveredGaussianQueryRequest& request) {

        if (!isInitialized()) {
            LOG_ERROR("Rendering engine not initialized");
            return std::unexpected("Rendering engine not initialized");
        }

        if (request.frame_view.size.x <= 0 || request.frame_view.size.y <= 0 ||
            request.frame_view.size.x > MAX_VIEWPORT_SIZE || request.frame_view.size.y > MAX_VIEWPORT_SIZE) {
            LOG_ERROR("Invalid viewport dimensions: {}x{}", request.frame_view.size.x, request.frame_view.size.y);
            return std::unexpected("Invalid viewport dimensions");
        }

        if (!ensureHoveredDepthQueryBuffersAllocated()) {
            return std::unexpected("Failed to allocate hovered-depth query buffers");
        }

        constexpr auto NO_HOVERED_RESULT = std::numeric_limits<unsigned long long>::max();
        if (cudaMemset(hovered_depth_id_device_, 0xFF, sizeof(unsigned long long)) != cudaSuccess) {
            return std::unexpected("Failed to reset hovered-depth query buffer");
        }

        auto pipeline_req = makeHoveredGaussianQueryPipelineRequest(request, hovered_depth_id_device_);
        GaussianRasterResources raster_resources;
        applyCropBoxToPipeline(pipeline_req, request.filters.crop_region, raster_resources);
        applyEllipsoidToPipeline(pipeline_req, request.filters.ellipsoid_region, raster_resources);
        applyViewVolumeToPipeline(
            pipeline_req, request.filters.view_volume, request.filters.cull_outside_view_volume, raster_resources);

        auto pipeline_result = pipeline_.renderGaussianImage(splat_data, pipeline_req);
        if (!pipeline_result) {
            return std::unexpected(pipeline_result.error());
        }

        if (cudaMemcpy(hovered_depth_id_host_, hovered_depth_id_device_,
                       sizeof(unsigned long long), cudaMemcpyDeviceToHost) != cudaSuccess) {
            return std::unexpected("Failed to read back hovered-depth query result");
        }

        const unsigned long long packed = *hovered_depth_id_host_;
        if (packed == NO_HOVERED_RESULT) {
            return std::optional<int>{};
        }

        return std::optional<int>{static_cast<int>(packed & 0xFFFFFFFFu)};
    }

    Result<std::shared_ptr<lfs::core::Tensor>> RenderingEngineImpl::renderGaussianScreenPositions(
        const lfs::core::SplatData& splat_data,
        const ScreenPositionRenderRequest& request) {

        if (!isInitialized()) {
            LOG_ERROR("Rendering engine not initialized");
            return std::unexpected("Rendering engine not initialized");
        }

        if (request.frame_view.size.x <= 0 || request.frame_view.size.y <= 0 ||
            request.frame_view.size.x > MAX_VIEWPORT_SIZE || request.frame_view.size.y > MAX_VIEWPORT_SIZE) {
            LOG_ERROR("Invalid viewport dimensions: {}x{}", request.frame_view.size.x, request.frame_view.size.y);
            return std::unexpected("Invalid viewport dimensions");
        }

        RenderingPipeline::RasterRequest pipeline_req{
            .view_rotation = request.frame_view.rotation,
            .view_translation = request.frame_view.translation,
            .viewport_size = request.frame_view.size,
            .focal_length_mm = request.frame_view.focal_length_mm,
            .scaling_modifier = 1.0f,
            .antialiasing = false,
            .mip_filter = false,
            .sh_degree = 0,
            .render_mode = RenderMode::RGB,
            .crop_box = nullptr,
            .background_color = request.frame_view.background_color,
            .voxel_size = 0.01f,
            .gut = false,
            .equirectangular = request.equirectangular,
            .show_rings = false,
            .ring_width = 0.0f,
            .show_center_markers = false,
            .model_transforms = request.scene.model_transforms ? *request.scene.model_transforms
                                                               : std::vector<glm::mat4>{},
            .transform_indices = request.scene.transform_indices,
            .selection_mask = nullptr,
            .cursor_active = false,
            .cursor_x = 0.0f,
            .cursor_y = 0.0f,
            .cursor_radius = 0.0f,
            .preview_selection_add_mode = true,
            .preview_selection_tensor = nullptr,
            .cursor_saturation_preview = false,
            .cursor_saturation_amount = 0.0f,
            .hovered_depth_id = nullptr,
            .focused_gaussian_id = -1,
            .far_plane = request.frame_view.far_plane,
            .emphasized_node_mask = {},
            .node_visibility_mask = request.scene.node_visibility_mask,
            .orthographic = request.frame_view.orthographic,
            .ortho_scale = request.frame_view.ortho_scale};

        auto screen_positions = pipeline_.renderScreenPositions(splat_data, pipeline_req);
        if (!screen_positions) {
            LOG_ERROR("Screen-position render failed: {}", screen_positions.error());
            return std::unexpected(screen_positions.error());
        }

        return std::make_shared<Tensor>(std::move(*screen_positions));
    }

    Result<GpuFrame> RenderingEngineImpl::renderPointCloudGpuFrame(
        const lfs::core::PointCloud& point_cloud,
        const PointCloudRenderRequest& request) {

        if (!isInitialized()) {
            LOG_ERROR("Rendering engine not initialized");
            return std::unexpected("Rendering engine not initialized");
        }

        if (request.frame_view.size.x <= 0 || request.frame_view.size.y <= 0 ||
            request.frame_view.size.x > MAX_VIEWPORT_SIZE || request.frame_view.size.y > MAX_VIEWPORT_SIZE) {
            LOG_ERROR("Invalid viewport dimensions: {}x{}", request.frame_view.size.x, request.frame_view.size.y);
            return std::unexpected("Invalid viewport dimensions");
        }

        LOG_TRACE("Rendering point cloud GPU frame with viewport {}x{}",
                  request.frame_view.size.x, request.frame_view.size.y);

        auto pipeline_req = makePointCloudPipelineRequest(request);
        auto pipeline_result = pipeline_.renderRawPointCloudGpuFrame(point_cloud, pipeline_req);
        if (!pipeline_result) {
            LOG_ERROR("Point cloud GPU-frame render failed: {}", pipeline_result.error());
            return std::unexpected(pipeline_result.error());
        }

        return *pipeline_result;
    }

    Result<SplitViewFrameResult> RenderingEngineImpl::renderSplitViewGpuFrame(
        const SplitViewRequest& request) {

        if (!isInitialized()) {
            LOG_ERROR("Rendering engine not initialized");
            return std::unexpected("Rendering engine not initialized");
        }

        if (!split_view_renderer_) {
            LOG_ERROR("Split view renderer not initialized");
            return std::unexpected("Split view renderer not initialized");
        }

        LOG_TRACE("Rendering split view GPU frame with {} panels", request.panels.size());

        return split_view_renderer_->renderGpuFrame(request, render_target_pool_, *this);
    }

    Result<GpuFrame> RenderingEngineImpl::materializeGpuFrame(
        const std::shared_ptr<Tensor>& image,
        const FrameMetadata& metadata,
        const glm::ivec2& viewport_size) {
        LOG_TIMER_TRACE("RenderingEngineImpl::materializeGpuFrame");

        if (!isInitialized()) {
            LOG_ERROR("Rendering engine not initialized");
            return std::unexpected("Rendering engine not initialized");
        }

        if (!image) {
            LOG_ERROR("Invalid frame payload - image is null");
            return std::unexpected("Invalid frame payload");
        }

        if (auto upload_result = ensureRenderResultUploaded(image, metadata, viewport_size); !upload_result) {
            LOG_ERROR("Failed to materialize GPU frame: {}", upload_result.error());
            return std::unexpected(upload_result.error());
        }

        const glm::vec2 texcoord_scale = screen_renderer_->getTexcoordScale();
        const GLuint uploaded_depth_texture =
            metadata.external_depth_texture != 0
                ? metadata.external_depth_texture
                : (metadata.depth ? screen_renderer_->getUploadedDepthTexture() : 0);

        return GpuFrame{
            .color = {.id = screen_renderer_->getUploadedColorTexture(),
                      .size = viewport_size,
                      .texcoord_scale = texcoord_scale},
            .depth = {.id = uploaded_depth_texture,
                      .size = viewport_size,
                      .texcoord_scale = texcoord_scale},
            .depth_is_ndc = metadata.depth_is_ndc,
            .near_plane = metadata.near_plane,
            .far_plane = metadata.far_plane,
            .orthographic = metadata.orthographic};
    }

    Result<std::shared_ptr<Tensor>> RenderingEngineImpl::readbackGpuFrameColor(
        const GpuFrame& frame) {
        LOG_TIMER_TRACE("RenderingEngineImpl::readbackGpuFrameColor");

        if (!isInitialized()) {
            LOG_ERROR("Rendering engine not initialized");
            return std::unexpected("Rendering engine not initialized");
        }

        if (!frame.valid()) {
            LOG_ERROR("Invalid GPU frame");
            return std::unexpected("Invalid GPU frame");
        }

        const int width = frame.color.size.x;
        const int height = frame.color.size.y;
        if (width <= 0 || height <= 0) {
            LOG_ERROR("Invalid GPU frame size: {}x{}", width, height);
            return std::unexpected("Invalid GPU frame size");
        }

#ifdef CUDA_GL_INTEROP_ENABLED
        if (!gpu_frame_readback_interop_) {
            gpu_frame_readback_interop_ = std::make_unique<CudaGLInteropTexture>();
        }

        if (gpu_frame_readback_source_ != frame.color.id || gpu_frame_readback_size_ != frame.color.size) {
            if (auto init_result = gpu_frame_readback_interop_->initForReading(frame.color.id, width, height); init_result) {
                gpu_frame_readback_source_ = frame.color.id;
                gpu_frame_readback_size_ = frame.color.size;
            } else {
                LOG_WARN("Failed to initialize CUDA-GL viewport readback interop: {}", init_result.error());
                gpu_frame_readback_interop_.reset();
                gpu_frame_readback_source_ = 0;
                gpu_frame_readback_size_ = {0, 0};
            }
        }

        if (gpu_frame_readback_interop_ &&
            gpu_frame_readback_source_ == frame.color.id &&
            gpu_frame_readback_size_ == frame.color.size) {
            Tensor image_hwc;
            if (auto read_result = gpu_frame_readback_interop_->readToTensor(image_hwc, width, height); read_result) {
                return std::make_shared<Tensor>(image_hwc.permute({2, 0, 1}).contiguous());
            }

            LOG_WARN("CUDA-GL viewport readback failed, falling back to glReadPixels");
            gpu_frame_readback_interop_.reset();
            gpu_frame_readback_source_ = 0;
            gpu_frame_readback_size_ = {0, 0};
        }
#endif

        if (!gpu_frame_readback_fbo_) {
            GLuint fbo_id = 0;
            glGenFramebuffers(1, &fbo_id);
            gpu_frame_readback_fbo_ = FBO(fbo_id);
        }
        if (!gpu_frame_readback_fbo_) {
            LOG_ERROR("Failed to allocate viewport readback framebuffer");
            return std::unexpected("Failed to allocate viewport readback framebuffer");
        }

        GLFramebufferGuard framebuffer_guard;

        glBindFramebuffer(GL_FRAMEBUFFER, gpu_frame_readback_fbo_.get());
        glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, frame.color.id, 0);
        glReadBuffer(GL_COLOR_ATTACHMENT0);

        if (glCheckFramebufferStatus(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE) {
            LOG_ERROR("Viewport readback framebuffer incomplete");
            return std::unexpected("Viewport readback framebuffer incomplete");
        }

        std::vector<float> pixels(static_cast<size_t>(width) * static_cast<size_t>(height) * 3);
        glReadPixels(0, 0, width, height, GL_RGB, GL_FLOAT, pixels.data());
        const GLenum readback_error = glGetError();

        if (readback_error != GL_NO_ERROR) {
            LOG_ERROR("Viewport readback failed with GL error {}", static_cast<unsigned int>(readback_error));
            return std::unexpected("Viewport readback failed");
        }

        auto image_cpu = Tensor::from_vector(
            pixels,
            {static_cast<size_t>(height), static_cast<size_t>(width), 3},
            lfs::core::Device::CPU);

        return std::make_shared<Tensor>(image_cpu.permute({2, 0, 1}).cuda());
    }

    void RenderingEngineImpl::invalidatePresentUploadCache() {
        last_presented_image_.reset();
        last_presented_depth_.reset();
        last_presented_external_depth_texture_ = 0;
        last_presented_depth_is_ndc_ = false;
        last_presented_near_plane_ = 0.0f;
        last_presented_far_plane_ = 0.0f;
        last_presented_orthographic_ = false;
        has_present_upload_cache_ = false;
    }

    Result<void> RenderingEngineImpl::presentGpuFrame(
        const GpuFrame& frame,
        const glm::ivec2& viewport_pos,
        const glm::ivec2& viewport_size) {
        LOG_TIMER_TRACE("RenderingEngineImpl::presentGpuFrame");

        if (!isInitialized()) {
            LOG_ERROR("Rendering engine not initialized");
            return std::unexpected("Rendering engine not initialized");
        }

        if (!frame.valid()) {
            LOG_ERROR("Invalid GPU frame");
            return std::unexpected("Invalid GPU frame");
        }

        LOG_TRACE("Presenting GPU frame at ({}, {}) size {}x{}",
                  viewport_pos.x, viewport_pos.y, viewport_size.x, viewport_size.y);

        DepthParams params = screen_renderer_->getDepthParams();
        params.near_plane = frame.near_plane;
        params.far_plane = frame.far_plane;
        params.orthographic = frame.orthographic;
        params.has_depth = frame.depth.valid();
        params.depth_is_ndc = frame.depth_is_ndc;
        params.external_depth_texture = frame.depth.valid() ? frame.depth.id : 0;

        return screen_renderer_->renderTexture(
            quad_shader_,
            frame.color.id,
            params,
            frame.color.texcoord_scale,
            frame.depth.valid() ? frame.depth.id : 0);
    }

    Result<void> RenderingEngineImpl::ensureRenderResultUploaded(
        const std::shared_ptr<const Tensor>& image,
        const FrameMetadata& metadata,
        const glm::ivec2& viewport_size) {
        // Pointer-identity cache: explicit tensor-producing gaussian entry points create
        // a new shared_ptr per render, so distinct renders always have distinct pointers.
        // Same pointer == same content.
        const bool same_image_ptr = (last_presented_image_.get() == image.get());
        const bool same_depth_ptr = (!metadata.depth && !last_presented_depth_) ||
                                    (metadata.depth && last_presented_depth_.get() == metadata.depth.get());
        const bool same_depth_tex = (last_presented_external_depth_texture_ == metadata.external_depth_texture);
        const bool same_depth_mode = (last_presented_depth_is_ndc_ == metadata.depth_is_ndc);
        const bool same_near = (last_presented_near_plane_ == metadata.near_plane);
        const bool same_far = (last_presented_far_plane_ == metadata.far_plane);
        const bool same_projection = (last_presented_orthographic_ == metadata.orthographic);

        const bool needs_upload = !has_present_upload_cache_ ||
                                  !same_image_ptr ||
                                  !same_depth_ptr ||
                                  !same_depth_tex ||
                                  !same_depth_mode ||
                                  !same_near ||
                                  !same_far ||
                                  !same_projection;

        if (!needs_upload) {
            LOG_TRACE("Skipping screen upload (unchanged frame payload)");
            return {};
        }

        RenderingPipeline::ImageRenderResult internal_result;
        internal_result.image = *image;
        internal_result.depth = metadata.depth ? *metadata.depth : Tensor();
        internal_result.valid = true;
        internal_result.depth_is_ndc = metadata.depth_is_ndc;
        internal_result.external_depth_texture = metadata.external_depth_texture;
        internal_result.near_plane = metadata.near_plane;
        internal_result.far_plane = metadata.far_plane;
        internal_result.orthographic = metadata.orthographic;

        if (auto upload_result = RenderingPipeline::uploadToScreen(internal_result, *screen_renderer_, viewport_size);
            !upload_result) {
            invalidatePresentUploadCache();
            return upload_result;
        }

        last_presented_image_ = image;
        last_presented_depth_ = metadata.depth;
        last_presented_external_depth_texture_ = metadata.external_depth_texture;
        last_presented_depth_is_ndc_ = metadata.depth_is_ndc;
        last_presented_near_plane_ = metadata.near_plane;
        last_presented_far_plane_ = metadata.far_plane;
        last_presented_orthographic_ = metadata.orthographic;
        has_present_upload_cache_ = true;
        return {};
    }

    Result<void> RenderingEngineImpl::renderGrid(
        const ViewportData& viewport,
        GridPlane plane,
        float opacity) {

        if (!isInitialized() || !grid_renderer_.isInitialized()) {
            LOG_ERROR("Grid renderer not initialized");
            return std::unexpected("Grid renderer not initialized");
        }

        auto view = createViewMatrix(viewport);
        auto proj = createProjectionMatrix(viewport);

        grid_renderer_.setPlane(static_cast<RenderInfiniteGrid::GridPlane>(plane));
        grid_renderer_.setOpacity(opacity);

        return grid_renderer_.render(view, proj, viewport.orthographic);
    }

    Result<void> RenderingEngineImpl::renderBoundingBox(
        const BoundingBox& box,
        const ViewportData& viewport,
        const glm::vec3& color,
        float line_width) {

        if (!isInitialized() || !bbox_renderer_.isInitialized()) {
            LOG_ERROR("Bounding box renderer not initialized");
            return std::unexpected("Bounding box renderer not initialized");
        }

        bbox_renderer_.setBounds(box.min, box.max);
        bbox_renderer_.setColor(color);
        bbox_renderer_.setLineWidth(line_width);

        bbox_renderer_.setWorld2BBoxMat4(box.transform);

        auto view = createViewMatrix(viewport);
        auto proj = createProjectionMatrix(viewport);

        return bbox_renderer_.render(view, proj);
    }

    Result<void> RenderingEngineImpl::renderEllipsoid(
        const Ellipsoid& ellipsoid,
        const ViewportData& viewport,
        const glm::vec3& color,
        float line_width) {

        if (!isInitialized() || !ellipsoid_renderer_.isInitialized()) {
            LOG_ERROR("Ellipsoid renderer not initialized");
            return std::unexpected("Ellipsoid renderer not initialized");
        }

        ellipsoid_renderer_.setRadii(ellipsoid.radii);
        ellipsoid_renderer_.setTransform(ellipsoid.transform);
        ellipsoid_renderer_.setColor(color);
        ellipsoid_renderer_.setLineWidth(line_width);

        auto view = createViewMatrix(viewport);
        auto proj = createProjectionMatrix(viewport);

        return ellipsoid_renderer_.render(view, proj);
    }

    Result<void> RenderingEngineImpl::renderCoordinateAxes(
        const ViewportData& viewport,
        float size,
        const std::array<bool, 3>& visible,
        bool equirectangular) {

        if (!isInitialized() || !axes_renderer_.isInitialized()) {
            LOG_ERROR("Axes renderer not initialized");
            return std::unexpected("Axes renderer not initialized");
        }

        axes_renderer_.setSize(size);
        for (int i = 0; i < 3; ++i) {
            axes_renderer_.setAxisVisible(i, visible[i]);
        }

        auto view = createViewMatrix(viewport);
        auto proj = createProjectionMatrix(viewport);

        return axes_renderer_.render(view, proj, equirectangular);
    }

    Result<void> RenderingEngineImpl::renderPivot(
        const ViewportData& viewport,
        const glm::vec3& pivot_position,
        float size,
        float opacity) {

        if (!isInitialized() || !pivot_renderer_.isInitialized()) {
            return std::unexpected("Pivot renderer not initialized");
        }

        pivot_renderer_.setPosition(pivot_position);
        pivot_renderer_.setSize(size);
        pivot_renderer_.setOpacity(opacity);

        auto view = createViewMatrix(viewport);
        auto proj = createProjectionMatrix(viewport);

        return pivot_renderer_.render(view, proj);
    }

    Result<void> RenderingEngineImpl::renderViewportGizmo(
        const glm::mat3& camera_rotation,
        const glm::vec2& viewport_pos,
        const glm::vec2& viewport_size) {

        if (!isInitialized()) {
            LOG_ERROR("Viewport gizmo not initialized");
            return std::unexpected("Viewport gizmo not initialized");
        }

        return viewport_gizmo_.render(camera_rotation, viewport_pos, viewport_size);
    }

    int RenderingEngineImpl::hitTestViewportGizmo(
        const glm::vec2& click_pos,
        const glm::vec2& viewport_pos,
        const glm::vec2& viewport_size) const {
        if (const auto hit = viewport_gizmo_.hitTest(click_pos, viewport_pos, viewport_size)) {
            return static_cast<int>(hit->axis) + (hit->negative ? 3 : 0);
        }
        return -1;
    }

    void RenderingEngineImpl::setViewportGizmoHover(const int axis) {
        if (axis >= 0 && axis <= 2) {
            viewport_gizmo_.setHoveredAxis(static_cast<GizmoAxis>(axis), false);
        } else if (axis >= 3 && axis <= 5) {
            viewport_gizmo_.setHoveredAxis(static_cast<GizmoAxis>(axis - 3), true);
        } else {
            viewport_gizmo_.setHoveredAxis(std::nullopt);
        }
    }

    Result<void> RenderingEngineImpl::renderCameraFrustums(
        const std::vector<std::shared_ptr<const lfs::core::Camera>>& cameras,
        const CameraFrustumRenderRequest& request) {

        if (!camera_frustum_renderer_.isInitialized()) {
            return {};
        }

        camera_frustum_renderer_.setFocusedCamera(request.focused_index);

        auto view = createViewMatrix(request.viewport);
        auto proj = createProjectionMatrix(request.viewport);

        return camera_frustum_renderer_.render(
            cameras, view, proj, request.scale, request.train_color, request.eval_color,
            request.scene_transform, request.equirectangular_view,
            request.disabled_uids, request.emphasized_uids);
    }

    Result<int> RenderingEngineImpl::pickCameraFrustum(
        const std::vector<std::shared_ptr<const lfs::core::Camera>>& cameras,
        const CameraFrustumPickRequest& request) {

        if (!camera_frustum_renderer_.isInitialized()) {
            return -1;
        }

        auto view = createViewMatrix(request.viewport);
        auto proj = createProjectionMatrix(request.viewport);

        return camera_frustum_renderer_.pickCamera(
            cameras, request.mouse_pos, request.viewport_pos, request.viewport_size, view, proj,
            request.scale, request.scene_transform);
    }

    void RenderingEngineImpl::clearFrustumCache() {
        camera_frustum_renderer_.clearThumbnailCache();
    }

    void RenderingEngineImpl::setFrustumImageLoader(std::shared_ptr<lfs::io::PipelinedImageLoader> loader) {
        camera_frustum_renderer_.setImageLoader(std::move(loader));
    }

    glm::mat4 RenderingEngineImpl::createViewMatrix(const ViewportData& viewport) const {
        glm::mat3 flip_yz = glm::mat3(1, 0, 0, 0, -1, 0, 0, 0, -1);
        glm::mat3 R_inv = glm::transpose(viewport.rotation);
        glm::vec3 t_inv = -R_inv * viewport.translation;

        R_inv = flip_yz * R_inv;
        t_inv = flip_yz * t_inv;

        glm::mat4 view(1.0f);
        for (int i = 0; i < 3; ++i) {
            for (int j = 0; j < 3; ++j) {
                view[i][j] = R_inv[i][j];
            }
        }
        view[3][0] = t_inv.x;
        view[3][1] = t_inv.y;
        view[3][2] = t_inv.z;

        return view;
    }

    glm::mat4 RenderingEngineImpl::createProjectionMatrix(const ViewportData& viewport) const {
        return viewport.getProjectionMatrix();
    }

    Result<void> RenderingEngineImpl::renderMesh(
        const lfs::core::MeshData& mesh,
        const ViewportData& viewport,
        const glm::mat4& model_transform,
        const MeshRenderOptions& options,
        bool use_fbo) {

        if (!mesh_renderer_.isInitialized())
            return std::unexpected("Mesh renderer not initialized");

        mesh_renderer_.resize(viewport.size.x, viewport.size.y);

        const glm::mat4 view = createViewMatrix(viewport);
        const glm::mat4 projection = createProjectionMatrix(viewport);
        const glm::vec3 camera_pos = -glm::transpose(glm::mat3(view)) * glm::vec3(view[3]);

        const bool clear_fbo = !mesh_rendered_this_frame_;
        auto result = mesh_renderer_.render(mesh, model_transform, view, projection, camera_pos, options, true, clear_fbo);
        if (result) {
            mesh_rendered_this_frame_ = true;
        }
        return result;
    }

    unsigned int RenderingEngineImpl::getMeshColorTexture() const {
        return mesh_renderer_.getColorTexture();
    }

    unsigned int RenderingEngineImpl::getMeshDepthTexture() const {
        return mesh_renderer_.getDepthTexture();
    }

    unsigned int RenderingEngineImpl::getMeshFramebuffer() const {
        return mesh_renderer_.getFramebuffer();
    }

    bool RenderingEngineImpl::hasMeshRender() const {
        return mesh_rendered_this_frame_ && mesh_renderer_.isInitialized();
    }

    Result<void> RenderingEngineImpl::compositeMeshAndGpuFrame(
        const GpuFrame& splat_frame,
        const glm::ivec2& viewport_size) {

        if (!depth_compositor_.isInitialized())
            return std::unexpected("Depth compositor not initialized");

        if (!mesh_rendered_this_frame_)
            return {};

        if (!splat_frame.valid())
            return std::unexpected("Invalid GPU frame");

        if (!splat_frame.depth.valid())
            return std::unexpected("GPU frame missing depth texture");

        return depth_compositor_.composite(
            splat_frame.color.id,
            splat_frame.depth.id,
            mesh_renderer_.getColorTexture(),
            mesh_renderer_.getDepthTexture(),
            splat_frame.near_plane,
            splat_frame.far_plane,
            true,
            splat_frame.color.texcoord_scale,
            splat_frame.depth_is_ndc);
    }

    Result<void> RenderingEngineImpl::presentMeshOnly() {
        if (!depth_compositor_.isInitialized())
            return std::unexpected("Depth compositor not initialized");

        if (!mesh_rendered_this_frame_)
            return {};

        return depth_compositor_.presentMeshOnly(
            mesh_renderer_.getColorTexture(),
            mesh_renderer_.getDepthTexture());
    }

} // namespace lfs::rendering

/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "rendering_engine_impl.hpp"
#include "core/logger.hpp"
#include "core/point_cloud.hpp"
#include "framebuffer_factory.hpp"
#include "geometry/bounding_box.hpp"
#include "rendering/render_constants.hpp"

namespace lfs::rendering {

    RenderingEngineImpl::RenderingEngineImpl() {
        LOG_DEBUG("Initializing RenderingEngineImpl");
    };

    RenderingEngineImpl::~RenderingEngineImpl() {
        shutdown();
    }

    Result<void> RenderingEngineImpl::initialize() {
        LOG_TIMER("RenderingEngine::initialize");

        // Check if already initialized by checking if key components exist
        if (quad_shader_.valid()) {
            LOG_TRACE("RenderingEngine already initialized, skipping");
            return {};
        }

        LOG_INFO("Initializing rendering engine...");

        // Create screen renderer with preferred mode
        screen_renderer_ = std::make_shared<ScreenQuadRenderer>(getPreferredFrameBufferMode());

        // Initialize split view renderer
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

        // Initialize camera frustum renderer
        if (auto result = camera_frustum_renderer_.init(); !result) {
            LOG_ERROR("Failed to initialize camera frustum renderer: {}", result.error());
            // Non-critical, continue without it
        } else {
            LOG_DEBUG("Camera frustum renderer initialized");
        }

        auto shader_result = initializeShaders();
        if (!shader_result) {
            LOG_ERROR("Failed to initialize shaders: {}", shader_result.error());
            shutdown(); // Clean up partial initialization
            return std::unexpected(shader_result.error());
        }

        LOG_INFO("Rendering engine initialized successfully");
        return {};
    }

    void RenderingEngineImpl::shutdown() {
        LOG_DEBUG("Shutting down rendering engine");
        // Just reset/clean up - safe to call multiple times
        quad_shader_ = ManagedShader();
        screen_renderer_.reset();
        split_view_renderer_.reset();
        viewport_gizmo_.shutdown();
        // Other components clean up in their destructors
    }

    bool RenderingEngineImpl::isInitialized() const {
        // Check if key components exist
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

    Result<RenderResult> RenderingEngineImpl::renderGaussians(
        const lfs::core::SplatData& splat_data,
        const RenderRequest& request) {

        if (!isInitialized()) {
            LOG_ERROR("Rendering engine not initialized");
            return std::unexpected("Rendering engine not initialized");
        }

        // Validate request
        if (request.viewport.size.x <= 0 || request.viewport.size.y <= 0 ||
            request.viewport.size.x > 16384 || request.viewport.size.y > 16384) {
            LOG_ERROR("Invalid viewport dimensions: {}x{}", request.viewport.size.x, request.viewport.size.y);
            return std::unexpected("Invalid viewport dimensions");
        }

        LOG_TRACE("Rendering gaussians with viewport {}x{}", request.viewport.size.x, request.viewport.size.y);

        // Convert to internal pipeline request using designated initializers
        RenderingPipeline::RenderRequest pipeline_req{
            .view_rotation = request.viewport.rotation,
            .view_translation = request.viewport.translation,
            .viewport_size = request.viewport.size,
            .fov = request.viewport.fov,
            .scaling_modifier = request.scaling_modifier,
            .antialiasing = request.antialiasing,
            .sh_degree = request.sh_degree,
            .render_mode = RenderMode::RGB,
            .crop_box = nullptr,
            .background_color = request.background_color,
            .point_cloud_mode = request.point_cloud_mode,
            .voxel_size = request.voxel_size,
            .gut = request.gut,
            .show_rings = request.show_rings,
            .ring_width = request.ring_width,
            .show_center_markers = request.show_center_markers,
            .model_transforms = request.model_transforms,
            .transform_indices = request.transform_indices,
            .selection_mask = request.selection_mask,
            .output_screen_positions = request.output_screen_positions,
            .brush_active = request.brush_active,
            .brush_x = request.brush_x,
            .brush_y = request.brush_y,
            .brush_radius = request.brush_radius,
            .brush_add_mode = request.brush_add_mode,
            .brush_selection_tensor = request.brush_selection_tensor,
            .brush_saturation_mode = request.brush_saturation_mode,
            .brush_saturation_amount = request.brush_saturation_amount,
            .selection_mode_rings = request.selection_mode_rings,
            .hovered_depth_id = request.hovered_depth_id,
            .highlight_gaussian_id = request.highlight_gaussian_id,
            .far_plane = request.far_plane,
            .selected_node_mask = request.selected_node_mask,
            .desaturate_unselected = request.desaturate_unselected,
            .selection_flash_intensity = request.selection_flash_intensity,
            .orthographic = request.orthographic,
            .ortho_scale = request.ortho_scale};

        // Convert crop box if present
        std::unique_ptr<lfs::geometry::BoundingBox> temp_crop_box;
        Tensor crop_box_transform_tensor, crop_box_min_tensor, crop_box_max_tensor;
        if (request.crop_box.has_value()) {
            temp_crop_box = std::make_unique<lfs::geometry::BoundingBox>();
            temp_crop_box->setBounds(request.crop_box->min, request.crop_box->max);

            // Convert the transform matrix to EuclideanTransform
            lfs::geometry::EuclideanTransform transform(request.crop_box->transform);
            temp_crop_box->setworld2BBox(transform);

            pipeline_req.crop_box = temp_crop_box.get();

            // Prepare crop box tensors for GPU visualization
            // The transform is world-to-box (inverse of box-to-world)
            const glm::mat4& w2b = request.crop_box->transform;
            std::vector<float> transform_data(16);
            for (int row = 0; row < 4; ++row) {
                for (int col = 0; col < 4; ++col) {
                    transform_data[row * 4 + col] = w2b[col][row]; // Transpose to row-major
                }
            }
            crop_box_transform_tensor = Tensor::from_vector(transform_data, {4, 4}, lfs::core::Device::CPU).cuda();

            std::vector<float> min_data = {request.crop_box->min.x, request.crop_box->min.y, request.crop_box->min.z};
            crop_box_min_tensor = Tensor::from_vector(min_data, {3}, lfs::core::Device::CPU).cuda();

            std::vector<float> max_data = {request.crop_box->max.x, request.crop_box->max.y, request.crop_box->max.z};
            crop_box_max_tensor = Tensor::from_vector(max_data, {3}, lfs::core::Device::CPU).cuda();

            pipeline_req.crop_box_transform = &crop_box_transform_tensor;
            pipeline_req.crop_box_min = &crop_box_min_tensor;
            pipeline_req.crop_box_max = &crop_box_max_tensor;
            pipeline_req.crop_inverse = request.crop_inverse;
            pipeline_req.crop_desaturate = request.crop_desaturate;
        }

        // Convert depth filter if present (Selection tool - separate from crop box)
        Tensor depth_filter_transform_tensor, depth_filter_min_tensor, depth_filter_max_tensor;
        if (request.depth_filter.has_value()) {
            // Prepare depth filter tensors for GPU desaturation
            const glm::mat4& w2b = request.depth_filter->transform;
            std::vector<float> transform_data(16);
            for (int row = 0; row < 4; ++row) {
                for (int col = 0; col < 4; ++col) {
                    transform_data[row * 4 + col] = w2b[col][row]; // Transpose to row-major
                }
            }
            depth_filter_transform_tensor = Tensor::from_vector(transform_data, {4, 4}, lfs::core::Device::CPU).cuda();

            std::vector<float> min_data = {request.depth_filter->min.x, request.depth_filter->min.y, request.depth_filter->min.z};
            depth_filter_min_tensor = Tensor::from_vector(min_data, {3}, lfs::core::Device::CPU).cuda();

            std::vector<float> max_data = {request.depth_filter->max.x, request.depth_filter->max.y, request.depth_filter->max.z};
            depth_filter_max_tensor = Tensor::from_vector(max_data, {3}, lfs::core::Device::CPU).cuda();

            pipeline_req.depth_filter_transform = &depth_filter_transform_tensor;
            pipeline_req.depth_filter_min = &depth_filter_min_tensor;
            pipeline_req.depth_filter_max = &depth_filter_max_tensor;
        }

        auto pipeline_result = pipeline_.render(splat_data, pipeline_req);

        if (!pipeline_result) {
            LOG_ERROR("Pipeline render failed: {}", pipeline_result.error());
            return std::unexpected(pipeline_result.error());
        }

        // Convert result
        RenderResult result{
            .image = std::make_shared<Tensor>(pipeline_result->image),
            .depth = std::make_shared<Tensor>(pipeline_result->depth),
            .screen_positions = pipeline_result->screen_positions.is_valid()
                                    ? std::make_shared<Tensor>(pipeline_result->screen_positions)
                                    : nullptr,
            .valid = true,
            .depth_is_ndc = pipeline_result->depth_is_ndc,
            .external_depth_texture = pipeline_result->external_depth_texture,
            .near_plane = pipeline_result->near_plane,
            .far_plane = pipeline_result->far_plane,
            .orthographic = pipeline_result->orthographic};

        return result;
    }

    Result<RenderResult> RenderingEngineImpl::renderPointCloud(
        const lfs::core::PointCloud& point_cloud,
        const RenderRequest& request) {

        if (!isInitialized()) {
            LOG_ERROR("Rendering engine not initialized");
            return std::unexpected("Rendering engine not initialized");
        }

        // Validate request
        if (request.viewport.size.x <= 0 || request.viewport.size.y <= 0 ||
            request.viewport.size.x > 16384 || request.viewport.size.y > 16384) {
            LOG_ERROR("Invalid viewport dimensions: {}x{}", request.viewport.size.x, request.viewport.size.y);
            return std::unexpected("Invalid viewport dimensions");
        }

        LOG_TRACE("Rendering point cloud with viewport {}x{}", request.viewport.size.x, request.viewport.size.y);

        // Convert to internal pipeline request (simplified - no crop box for point clouds)
        RenderingPipeline::RenderRequest pipeline_req{
            .view_rotation = request.viewport.rotation,
            .view_translation = request.viewport.translation,
            .viewport_size = request.viewport.size,
            .fov = request.viewport.fov,
            .scaling_modifier = request.scaling_modifier,
            .antialiasing = false,
            .sh_degree = 0,
            .render_mode = RenderMode::RGB,
            .crop_box = nullptr,
            .background_color = request.background_color,
            .point_cloud_mode = true,
            .voxel_size = request.voxel_size,
            .gut = false,
            .show_rings = false,
            .ring_width = 0.0f,
            .show_center_markers = false,
            .model_transforms = request.model_transforms,
            .transform_indices = nullptr,
            .selection_mask = nullptr,
            .output_screen_positions = false,
            .brush_active = false,
            .brush_x = 0.0f,
            .brush_y = 0.0f,
            .brush_radius = 0.0f,
            .brush_add_mode = true,
            .brush_selection_tensor = nullptr,
            .brush_saturation_mode = false,
            .brush_saturation_amount = 0.0f,
            .selection_mode_rings = false,
            .hovered_depth_id = nullptr,
            .highlight_gaussian_id = -1,
            .far_plane = DEFAULT_FAR_PLANE,
            .selected_node_mask = {}};

        auto pipeline_result = pipeline_.renderRawPointCloud(point_cloud, pipeline_req);

        if (!pipeline_result) {
            LOG_ERROR("Pipeline render failed: {}", pipeline_result.error());
            return std::unexpected(pipeline_result.error());
        }

        // Convert result
        RenderResult result{
            .image = std::make_shared<Tensor>(pipeline_result->image),
            .depth = std::make_shared<Tensor>(pipeline_result->depth),
            .screen_positions = nullptr,
            .valid = true,
            .depth_is_ndc = pipeline_result->depth_is_ndc,
            .external_depth_texture = pipeline_result->external_depth_texture,
            .near_plane = pipeline_result->near_plane,
            .far_plane = pipeline_result->far_plane,
            .orthographic = pipeline_result->orthographic};

        return result;
    }

    Result<RenderResult> RenderingEngineImpl::renderSplitView(
        const SplitViewRequest& request) {

        if (!isInitialized()) {
            LOG_ERROR("Rendering engine not initialized");
            return std::unexpected("Rendering engine not initialized");
        }

        if (!split_view_renderer_) {
            LOG_ERROR("Split view renderer not initialized");
            return std::unexpected("Split view renderer not initialized");
        }

        LOG_TRACE("Rendering split view with {} panels", request.panels.size());

        return split_view_renderer_->render(request, pipeline_, *screen_renderer_, quad_shader_);
    }

    Result<void> RenderingEngineImpl::presentToScreen(
        const RenderResult& result,
        const glm::ivec2& viewport_pos,
        const glm::ivec2& viewport_size) {
        LOG_TIMER_TRACE("RenderingEngineImpl::presentToScreen");

        if (!isInitialized()) {
            LOG_ERROR("Rendering engine not initialized");
            return std::unexpected("Rendering engine not initialized");
        }

        if (!result.image) {
            LOG_ERROR("Invalid render result - image is null");
            return std::unexpected("Invalid render result");
        }

        LOG_TRACE("Presenting to screen at ({}, {}) size {}x{}",
                  viewport_pos.x, viewport_pos.y, viewport_size.x, viewport_size.y);

        // Convert back to internal result type
        RenderingPipeline::RenderResult internal_result;
        internal_result.image = *result.image;
        internal_result.depth = result.depth ? *result.depth : Tensor();
        internal_result.valid = true;
        internal_result.depth_is_ndc = result.depth_is_ndc;
        internal_result.external_depth_texture = result.external_depth_texture;
        internal_result.near_plane = result.near_plane;
        internal_result.far_plane = result.far_plane;
        internal_result.orthographic = result.orthographic;

        if (auto upload_result = RenderingPipeline::uploadToScreen(internal_result, *screen_renderer_, viewport_size);
            !upload_result) {
            LOG_ERROR("Failed to upload to screen: {}", upload_result.error());
            return upload_result;
        }

        glViewport(viewport_pos.x, viewport_pos.y, viewport_size.x, viewport_size.y);

        return screen_renderer_->render(quad_shader_);
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

    Result<void> RenderingEngineImpl::renderCoordinateAxes(
        const ViewportData& viewport,
        float size,
        const std::array<bool, 3>& visible) {

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

        return axes_renderer_.render(view, proj);
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

    Result<void> RenderingEngineImpl::renderTranslationGizmo(
        [[maybe_unused]] const glm::vec3& position,
        [[maybe_unused]] const ViewportData& viewport,
        [[maybe_unused]] float scale) {
        // Deprecated - translation gizmo removed, now using ImGuizmo
        return {};
    }

    std::shared_ptr<GizmoInteraction> RenderingEngineImpl::getGizmoInteraction() {
        // Deprecated - return nullptr since gizmo is removed
        return nullptr;
    }

    Result<void> RenderingEngineImpl::renderCameraFrustums(
        const std::vector<std::shared_ptr<const lfs::core::Camera>>& cameras,
        const ViewportData& viewport,
        float scale,
        const glm::vec3& train_color,
        const glm::vec3& eval_color,
        const glm::mat4& scene_transform) {

        if (!camera_frustum_renderer_.isInitialized()) {
            return {}; // Silent fail if not initialized
        }

        auto view = createViewMatrix(viewport);
        auto proj = createProjectionMatrix(viewport);

        return camera_frustum_renderer_.render(cameras, view, proj, scale, train_color, eval_color, scene_transform);
    }

    Result<void> RenderingEngineImpl::renderCameraFrustumsWithHighlight(
        const std::vector<std::shared_ptr<const lfs::core::Camera>>& cameras,
        const ViewportData& viewport,
        float scale,
        const glm::vec3& train_color,
        const glm::vec3& eval_color,
        int highlight_index,
        const glm::mat4& scene_transform) {

        if (!camera_frustum_renderer_.isInitialized()) {
            return {}; // Silent fail if not initialized
        }

        // Set the highlight before rendering
        camera_frustum_renderer_.setHighlightedCamera(highlight_index);

        auto view = createViewMatrix(viewport);
        auto proj = createProjectionMatrix(viewport);

        return camera_frustum_renderer_.render(cameras, view, proj, scale, train_color, eval_color, scene_transform);
    }

    Result<int> RenderingEngineImpl::pickCameraFrustum(
        const std::vector<std::shared_ptr<const lfs::core::Camera>>& cameras,
        const glm::vec2& mouse_pos,
        const glm::vec2& viewport_pos,
        const glm::vec2& viewport_size,
        const ViewportData& viewport,
        float scale,
        const glm::mat4& scene_transform) {

        if (!camera_frustum_renderer_.isInitialized()) {
            return -1;
        }

        auto view = createViewMatrix(viewport);
        auto proj = createProjectionMatrix(viewport);

        return camera_frustum_renderer_.pickCamera(
            cameras, mouse_pos, viewport_pos, viewport_size, view, proj, scale, scene_transform);
    }

    void RenderingEngineImpl::clearFrustumCache() {
        camera_frustum_renderer_.clearThumbnailCache();
    }

    RenderingPipelineResult RenderingEngineImpl::renderWithPipeline(
        const lfs::core::SplatData& model,
        const RenderingPipelineRequest& request) {

        LOG_TRACE("Rendering with pipeline");

        // Convert from public types to internal types using designated initializers
        RenderingPipeline::RenderRequest internal_request{
            .view_rotation = request.view_rotation,
            .view_translation = request.view_translation,
            .viewport_size = request.viewport_size,
            .fov = request.fov,
            .scaling_modifier = request.scaling_modifier,
            .antialiasing = request.antialiasing,
            .render_mode = request.render_mode,
            .crop_box = static_cast<const lfs::geometry::BoundingBox*>(request.crop_box),
            .background_color = request.background_color,
            .point_cloud_mode = request.point_cloud_mode,
            .voxel_size = request.voxel_size,
            .gut = request.gut,
            .show_rings = request.show_rings,
            .ring_width = request.ring_width};

        auto result = pipeline_.render(model, internal_request);

        // Convert back to public types
        RenderingPipelineResult public_result;

        if (!result) {
            public_result.valid = false;
            // Log error but don't expose internal error details
            LOG_ERROR("Pipeline render error: {}", result.error());
        } else {
            public_result.valid = result->valid;
            if (result->valid) {
                public_result.image = result->image;
                public_result.depth = result->depth;
            }
        }

        return public_result;
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

    Result<std::shared_ptr<IBoundingBox>> RenderingEngineImpl::createBoundingBox() {
        // Make sure we're initialized first
        if (!isInitialized()) {
            LOG_ERROR("RenderingEngine must be initialized before creating bounding boxes");
            return std::unexpected("RenderingEngine must be initialized before creating bounding boxes");
        }

        auto bbox = std::make_shared<RenderBoundingBox>();
        if (auto result = bbox->init(); !result) {
            LOG_ERROR("Failed to initialize bounding box: {}", result.error());
            return std::unexpected(result.error());
        }
        LOG_DEBUG("Created bounding box renderer");
        return bbox;
    }

    Result<std::shared_ptr<ICoordinateAxes>> RenderingEngineImpl::createCoordinateAxes() {
        // Make sure we're initialized first
        if (!isInitialized()) {
            LOG_ERROR("RenderingEngine must be initialized before creating coordinate axes");
            return std::unexpected("RenderingEngine must be initialized before creating coordinate axes");
        }

        auto axes = std::make_shared<RenderCoordinateAxes>();
        if (auto result = axes->init(); !result) {
            LOG_ERROR("Failed to initialize coordinate axes: {}", result.error());
            return std::unexpected(result.error());
        }
        LOG_DEBUG("Created coordinate axes renderer");
        return axes;
    }

} // namespace lfs::rendering
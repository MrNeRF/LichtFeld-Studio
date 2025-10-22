/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "rendering_engine_impl.hpp"
#include "core/logger.hpp"
#include "framebuffer_factory.hpp"
#include "geometry/bounding_box.hpp"

namespace gs::rendering {

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

        if (auto result = viewport_gizmo_.initialize(); !result) {
            LOG_ERROR("Failed to initialize viewport gizmo: {}", result.error());
            shutdown();
            return std::unexpected(result.error());
        }
        LOG_DEBUG("Viewport gizmo initialized");

        // Initialize translation gizmo
        if (auto result = translation_gizmo_.initialize(); !result) {
            LOG_ERROR("Failed to initialize translation gizmo: {}", result.error());
            shutdown();
            return std::unexpected(result.error());
        }
        LOG_DEBUG("Translation gizmo initialized");

        // Initialize camera frustum renderer
        if (auto result = camera_frustum_renderer_.init(); !result) {
            LOG_ERROR("Failed to initialize camera frustum renderer: {}", result.error());
            // Non-critical, continue without it
        } else {
            LOG_DEBUG("Camera frustum renderer initialized");
        }

        // Create gizmo interaction adapter
        gizmo_interaction_ = std::make_shared<GizmoInteractionAdapter>(&translation_gizmo_);

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
        translation_gizmo_.shutdown();
        viewport_gizmo_.shutdown();
        gizmo_interaction_.reset();
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
        const SplatData& splat_data,
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
            .render_mode = RenderMode::RGB,
            .crop_box = nullptr,
            .background_color = request.background_color,
            .point_cloud_mode = request.point_cloud_mode,
            .voxel_size = request.voxel_size,
            .gut = request.gut,
            .equirectangular = request.equirectangular,
            .sh_degree = request.sh_degree};

        // Convert crop box if present
        std::unique_ptr<gs::geometry::BoundingBox> temp_crop_box;
        if (request.crop_box.has_value()) {
            temp_crop_box = std::make_unique<gs::geometry::BoundingBox>();
            temp_crop_box->setBounds(request.crop_box->min, request.crop_box->max);

            // Convert the transform matrix to EuclideanTransform
            geometry::EuclideanTransform transform(request.crop_box->transform);
            temp_crop_box->setworld2BBox(transform);

            pipeline_req.crop_box = temp_crop_box.get();
        }

        auto pipeline_result = pipeline_.render(splat_data, pipeline_req);

        if (!pipeline_result) {
            LOG_ERROR("Pipeline render failed: {}", pipeline_result.error());
            return std::unexpected(pipeline_result.error());
        }

        // Convert result
        RenderResult result{
            .image = std::make_shared<torch::Tensor>(pipeline_result->image),
            .depth = std::make_shared<torch::Tensor>(pipeline_result->depth)};

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

        if (!isInitialized()) {
            LOG_ERROR("Rendering engine not initialized");
            return std::unexpected("Rendering engine not initialized");
        }

        if (!result.image) {
            LOG_ERROR("Invalid render result - image is null");
            return std::unexpected("Invalid render result");
        }

        LOG_TRACE("Presenting to screen at position ({}, {}) with size {}x{}",
                  viewport_pos.x, viewport_pos.y, viewport_size.x, viewport_size.y);

        // Convert back to internal result type
        RenderingPipeline::RenderResult internal_result;
        internal_result.image = *result.image;
        internal_result.depth = result.depth ? *result.depth : torch::Tensor();
        internal_result.valid = true;

        if (auto upload_result = RenderingPipeline::uploadToScreen(internal_result, *screen_renderer_, viewport_size);
            !upload_result) {
            LOG_ERROR("Failed to upload to screen: {}", upload_result.error());
            return upload_result;
        }

        // Set viewport for rendering
        glViewport(viewport_pos.x, viewport_pos.y, viewport_size.x, viewport_size.y);

        // Use the quad shader directly
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

        return grid_renderer_.render(view, proj);
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

        // Set the transform from the box
        geometry::EuclideanTransform transform(box.transform);
        bbox_renderer_.setworld2BBox(transform);

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

    Result<void> RenderingEngineImpl::renderTranslationGizmo(
        const glm::vec3& position,
        const ViewportData& viewport,
        float scale) {

        if (!isInitialized()) {
            LOG_ERROR("Rendering engine not initialized");
            return std::unexpected("Rendering engine not initialized");
        }

        auto view = createViewMatrix(viewport);
        auto proj = createProjectionMatrix(viewport);

        return translation_gizmo_.render(view, proj, position, scale);
    }

    Result<void> RenderingEngineImpl::renderCameraFrustums(
        const std::vector<std::shared_ptr<const Camera>>& cameras,
        const ViewportData& viewport,
        float scale,
        const glm::vec3& train_color,
        const glm::vec3& eval_color,
        const glm::mat4& world_transform) {

        if (!camera_frustum_renderer_.isInitialized()) {
            return {}; // Silent fail if not initialized
        }

        auto view = createViewMatrix(viewport);
        auto proj = createProjectionMatrix(viewport);

        return camera_frustum_renderer_.render(cameras, view, proj, scale, train_color, eval_color, world_transform);
    }

    Result<void> RenderingEngineImpl::renderCameraFrustumsWithHighlight(
        const std::vector<std::shared_ptr<const Camera>>& cameras,
        const ViewportData& viewport,
        float scale,
        const glm::vec3& train_color,
        const glm::vec3& eval_color,
        int highlight_index,
        const glm::mat4& world_transform) {

        if (!camera_frustum_renderer_.isInitialized()) {
            return {}; // Silent fail if not initialized
        }

        // Set the highlight before rendering
        camera_frustum_renderer_.setHighlightedCamera(highlight_index);

        auto view = createViewMatrix(viewport);
        auto proj = createProjectionMatrix(viewport);

        return camera_frustum_renderer_.render(cameras, view, proj, scale, train_color, eval_color, world_transform);
    }

    Result<int> RenderingEngineImpl::pickCameraFrustum(
        const std::vector<std::shared_ptr<const Camera>>& cameras,
        const glm::vec2& mouse_pos,
        const glm::vec2& viewport_pos,
        const glm::vec2& viewport_size,
        const ViewportData& viewport,
        float scale,
        const glm::mat4& world_transform) {

        if (!camera_frustum_renderer_.isInitialized()) {
            return -1;
        }

        auto view = createViewMatrix(viewport);
        auto proj = createProjectionMatrix(viewport);

        return camera_frustum_renderer_.pickCamera(
            cameras, mouse_pos, viewport_pos, viewport_size, view, proj, scale, world_transform);
    }

    std::shared_ptr<GizmoInteraction> RenderingEngineImpl::getGizmoInteraction() {
        return gizmo_interaction_;
    }

    RenderingPipelineResult RenderingEngineImpl::renderWithPipeline(
        const SplatData& model,
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
            .crop_box = static_cast<const geometry::BoundingBox*>(request.crop_box),
            .background_color = request.background_color,
            .point_cloud_mode = request.point_cloud_mode,
            .voxel_size = request.voxel_size,
            .gut = request.gut,
            .sh_degree = request.sh_degree};

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
        float aspect = static_cast<float>(viewport.size.x) / viewport.size.y;
        float fov_rad = glm::radians(viewport.fov);
        return glm::perspective(fov_rad, aspect, 0.1f, 1000.0f);
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

} // namespace gs::rendering
/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "rendering_engine_impl.hpp"
#include "core/logger.hpp"
#include "core/mesh_data.hpp"
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

        if (quad_shader_.valid()) {
            LOG_TRACE("RenderingEngine already initialized, skipping");
            return {};
        }

        LOG_INFO("Initializing rendering engine...");

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
        last_presented_image_.reset();
        last_presented_depth_.reset();
        last_presented_external_depth_texture_ = 0;
        last_presented_depth_is_ndc_ = false;
        last_presented_near_plane_ = 0.0f;
        last_presented_far_plane_ = 0.0f;
        last_presented_orthographic_ = false;
        has_present_upload_cache_ = false;
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

    Result<RenderResult> RenderingEngineImpl::renderGaussians(
        const lfs::core::SplatData& splat_data,
        const RenderRequest& request) {

        if (!isInitialized()) {
            LOG_ERROR("Rendering engine not initialized");
            return std::unexpected("Rendering engine not initialized");
        }

        if (request.viewport.size.x <= 0 || request.viewport.size.y <= 0 ||
            request.viewport.size.x > MAX_VIEWPORT_SIZE || request.viewport.size.y > MAX_VIEWPORT_SIZE) {
            LOG_ERROR("Invalid viewport dimensions: {}x{}", request.viewport.size.x, request.viewport.size.y);
            return std::unexpected("Invalid viewport dimensions");
        }

        LOG_TRACE("Rendering gaussians with viewport {}x{}", request.viewport.size.x, request.viewport.size.y);

        RenderingPipeline::RenderRequest pipeline_req{
            .view_rotation = request.viewport.rotation,
            .view_translation = request.viewport.translation,
            .viewport_size = request.viewport.size,
            .focal_length_mm = request.viewport.focal_length_mm,
            .scaling_modifier = request.scaling_modifier,
            .antialiasing = request.antialiasing,
            .mip_filter = request.mip_filter,
            .sh_degree = request.sh_degree,
            .render_mode = RenderMode::RGB,
            .crop_box = nullptr,
            .background_color = request.background_color,
            .point_cloud_mode = request.point_cloud_mode,
            .voxel_size = request.voxel_size,
            .gut = request.gut,
            .equirectangular = request.equirectangular,
            .show_rings = request.show_rings,
            .ring_width = request.ring_width,
            .show_center_markers = request.show_center_markers,
            .model_transforms = request.model_transforms ? *request.model_transforms : std::vector<glm::mat4>{},
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
            .node_visibility_mask = request.node_visibility_mask,
            .desaturate_unselected = request.desaturate_unselected,
            .selection_flash_intensity = request.selection_flash_intensity,
            .orthographic = request.orthographic,
            .ortho_scale = request.ortho_scale};

        std::unique_ptr<lfs::geometry::BoundingBox> temp_crop_box;
        Tensor crop_box_transform_tensor, crop_box_min_tensor, crop_box_max_tensor;
        if (request.crop_box.has_value()) {
            temp_crop_box = std::make_unique<lfs::geometry::BoundingBox>();
            temp_crop_box->setBounds(request.crop_box->min, request.crop_box->max);

            lfs::geometry::EuclideanTransform transform(request.crop_box->transform);
            temp_crop_box->setworld2BBox(transform);

            pipeline_req.crop_box = temp_crop_box.get();

            const glm::mat4& w2b = request.crop_box->transform;
            std::vector<float> transform_data(16);
            for (int row = 0; row < 4; ++row) {
                for (int col = 0; col < 4; ++col) {
                    transform_data[row * 4 + col] = w2b[col][row];
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
            pipeline_req.crop_parent_node_index = request.crop_parent_node_index;
        }

        Tensor ellipsoid_transform_tensor, ellipsoid_radii_tensor;
        if (request.ellipsoid.has_value()) {
            const glm::mat4& w2e = request.ellipsoid->transform;
            std::vector<float> transform_data(16);
            for (int row = 0; row < 4; ++row) {
                for (int col = 0; col < 4; ++col) {
                    transform_data[row * 4 + col] = w2e[col][row];
                }
            }
            ellipsoid_transform_tensor = Tensor::from_vector(transform_data, {4, 4}, lfs::core::Device::CPU).cuda();

            std::vector<float> radii_data = {request.ellipsoid->radii.x, request.ellipsoid->radii.y, request.ellipsoid->radii.z};
            ellipsoid_radii_tensor = Tensor::from_vector(radii_data, {3}, lfs::core::Device::CPU).cuda();

            pipeline_req.ellipsoid_transform = &ellipsoid_transform_tensor;
            pipeline_req.ellipsoid_radii = &ellipsoid_radii_tensor;
            pipeline_req.ellipsoid_inverse = request.ellipsoid_inverse;
            pipeline_req.ellipsoid_desaturate = request.ellipsoid_desaturate;
            pipeline_req.ellipsoid_parent_node_index = request.ellipsoid_parent_node_index;
        }

        Tensor depth_filter_transform_tensor, depth_filter_min_tensor, depth_filter_max_tensor;
        if (request.depth_filter.has_value()) {
            const glm::mat4& w2b = request.depth_filter->transform;
            std::vector<float> transform_data(16);
            for (int row = 0; row < 4; ++row) {
                for (int col = 0; col < 4; ++col) {
                    transform_data[row * 4 + col] = w2b[col][row];
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

        if (request.viewport.size.x <= 0 || request.viewport.size.y <= 0 ||
            request.viewport.size.x > MAX_VIEWPORT_SIZE || request.viewport.size.y > MAX_VIEWPORT_SIZE) {
            LOG_ERROR("Invalid viewport dimensions: {}x{}", request.viewport.size.x, request.viewport.size.y);
            return std::unexpected("Invalid viewport dimensions");
        }

        LOG_TRACE("Rendering point cloud with viewport {}x{}", request.viewport.size.x, request.viewport.size.y);

        PointCloudCropParams crop_params;
        if (request.crop_box.has_value()) {
            crop_params.enabled = true;
            crop_params.transform = request.crop_box->transform;
            crop_params.min = request.crop_box->min;
            crop_params.max = request.crop_box->max;
            crop_params.inverse = request.crop_inverse;
            crop_params.desaturate = request.crop_desaturate;
        }

        RenderingPipeline::RenderRequest pipeline_req{
            .view_rotation = request.viewport.rotation,
            .view_translation = request.viewport.translation,
            .viewport_size = request.viewport.size,
            .focal_length_mm = request.viewport.focal_length_mm,
            .scaling_modifier = request.scaling_modifier,
            .antialiasing = false,
            .mip_filter = false,
            .sh_degree = 0,
            .render_mode = RenderMode::RGB,
            .crop_box = nullptr,
            .background_color = request.background_color,
            .point_cloud_mode = true,
            .voxel_size = request.voxel_size,
            .gut = false,
            .equirectangular = request.equirectangular,
            .show_rings = false,
            .ring_width = 0.0f,
            .show_center_markers = false,
            .model_transforms = request.model_transforms ? *request.model_transforms : std::vector<glm::mat4>{},
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
            .selected_node_mask = {},
            .point_cloud_crop_params = crop_params};

        auto pipeline_result = pipeline_.renderRawPointCloud(point_cloud, pipeline_req);

        if (!pipeline_result) {
            LOG_ERROR("Pipeline render failed: {}", pipeline_result.error());
            return std::unexpected(pipeline_result.error());
        }

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

        // Pointer-identity cache: renderGaussians() creates a new shared_ptr per frame,
        // so distinct renders always have distinct pointers. Same pointer == same content.
        const bool same_image_ptr = (last_presented_image_.get() == result.image.get());
        const bool same_depth_ptr = (!result.depth && !last_presented_depth_) ||
                                    (result.depth && last_presented_depth_.get() == result.depth.get());
        const bool same_depth_tex = (last_presented_external_depth_texture_ == result.external_depth_texture);
        const bool same_depth_mode = (last_presented_depth_is_ndc_ == result.depth_is_ndc);
        const bool same_near = (last_presented_near_plane_ == result.near_plane);
        const bool same_far = (last_presented_far_plane_ == result.far_plane);
        const bool same_projection = (last_presented_orthographic_ == result.orthographic);

        const bool needs_upload = !has_present_upload_cache_ ||
                                  !same_image_ptr ||
                                  !same_depth_ptr ||
                                  !same_depth_tex ||
                                  !same_depth_mode ||
                                  !same_near ||
                                  !same_far ||
                                  !same_projection;

        if (needs_upload) {
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
                has_present_upload_cache_ = false;
                LOG_ERROR("Failed to upload to screen: {}", upload_result.error());
                return upload_result;
            }

            last_presented_image_ = result.image;
            last_presented_depth_ = result.depth;
            last_presented_external_depth_texture_ = result.external_depth_texture;
            last_presented_depth_is_ndc_ = result.depth_is_ndc;
            last_presented_near_plane_ = result.near_plane;
            last_presented_far_plane_ = result.far_plane;
            last_presented_orthographic_ = result.orthographic;
            has_present_upload_cache_ = true;
        } else {
            LOG_TRACE("Skipping screen upload (unchanged frame payload)");
        }

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

    Result<void> RenderingEngineImpl::renderCameraFrustumsWithHighlight(
        const std::vector<std::shared_ptr<const lfs::core::Camera>>& cameras,
        const ViewportData& viewport,
        float scale,
        const glm::vec3& train_color,
        const glm::vec3& eval_color,
        int highlight_index,
        const glm::mat4& scene_transform,
        bool equirectangular_view,
        const std::unordered_set<int>& disabled_uids,
        const std::unordered_set<int>& selected_uids) {

        if (!camera_frustum_renderer_.isInitialized()) {
            return {};
        }

        camera_frustum_renderer_.setHighlightedCamera(highlight_index);

        auto view = createViewMatrix(viewport);
        auto proj = createProjectionMatrix(viewport);

        return camera_frustum_renderer_.render(cameras, view, proj, scale, train_color, eval_color, scene_transform, equirectangular_view, disabled_uids, selected_uids);
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

    Result<void> RenderingEngineImpl::compositeMeshAndSplat(
        const RenderResult& splat_result,
        const glm::ivec2& viewport_size) {

        if (!depth_compositor_.isInitialized())
            return std::unexpected("Depth compositor not initialized");

        if (!mesh_rendered_this_frame_)
            return {};

        const GLuint splat_color = screen_renderer_->getUploadedColorTexture();
        const GLuint splat_depth = screen_renderer_->getUploadedDepthTexture();

        if (splat_color == 0 || splat_depth == 0)
            return {};

        const glm::vec2 splat_tc_scale = screen_renderer_->getTexcoordScale();

        return depth_compositor_.composite(
            splat_color, splat_depth,
            mesh_renderer_.getColorTexture(),
            mesh_renderer_.getDepthTexture(),
            splat_result.near_plane,
            splat_result.far_plane,
            true,
            splat_tc_scale,
            splat_result.depth_is_ndc);
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

/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "rendering_pipeline.hpp"
#include "core/camera.hpp"
#include "core/point_cloud.hpp"
#include "core/splat_data.hpp"
#include "gs_rasterizer_tensor.hpp"

#include <cstring>
#include <print>

namespace lfs::rendering {

    namespace {
        constexpr int GPU_ALIGNMENT = 16; // 16-pixel alignment for GPU texture efficiency
    }

    RenderingPipeline::RenderingPipeline()
        : background_(Tensor::zeros({3}, lfs::core::Device::CUDA, lfs::core::DataType::Float32)) {
        point_cloud_renderer_ = std::make_unique<PointCloudRenderer>();
        LOG_DEBUG("RenderingPipeline initialized");
    }

    RenderingPipeline::~RenderingPipeline() {
        cleanupFBO();
        cleanupPBO();
    }

    Result<RenderingPipeline::RenderResult> RenderingPipeline::render(
        const lfs::core::SplatData& model,
        const RenderRequest& request) {

        LOG_TIMER_TRACE("RenderingPipeline::render");

        // Validate dimensions
        if (request.viewport_size.x <= 0 || request.viewport_size.y <= 0 ||
            request.viewport_size.x > 16384 || request.viewport_size.y > 16384) {
            LOG_ERROR("Invalid viewport dimensions: {}x{}", request.viewport_size.x, request.viewport_size.y);
            return std::unexpected("Invalid viewport dimensions");
        }

        // Point cloud rendering mode
        if (request.point_cloud_mode) {
            LOG_TRACE("Using point cloud rendering mode");
            return renderPointCloud(model, request);
        }

        // Regular gaussian splatting rendering
        LOG_TRACE("Using gaussian splatting rendering mode");

        // Update background tensor in-place to avoid allocation
        // Access the tensor data directly
        auto bg_data = background_.ptr<float>();
        if (bg_data && background_.device() == lfs::core::Device::CUDA) {
            float bg_values[3] = {
                request.background_color.r,
                request.background_color.g,
                request.background_color.b};
            cudaMemcpy(bg_data, bg_values, 3 * sizeof(float), cudaMemcpyHostToDevice);
        }

        // Create camera for this frame
        auto cam_result = createCamera(request);
        if (!cam_result) {
            return std::unexpected(cam_result.error());
        }
        lfs::core::Camera cam = std::move(*cam_result);

        // Handle crop box conversion
        const lfs::geometry::BoundingBox* geom_bbox = nullptr;
        std::unique_ptr<lfs::geometry::BoundingBox> temp_bbox;

        if (request.crop_box) {
            // Create a temporary lfs::geometry::BoundingBox with the full transform
            temp_bbox = std::make_unique<lfs::geometry::BoundingBox>();
            temp_bbox->setBounds(request.crop_box->getMinBounds(), request.crop_box->getMaxBounds());
            temp_bbox->setworld2BBox(request.crop_box->getworld2BBox());
            geom_bbox = temp_bbox.get();
            LOG_TRACE("Using crop box for rendering");
        }

        // Create model transforms tensor if provided
        std::unique_ptr<Tensor> model_transforms_tensor;
        if (!request.model_transforms.empty()) {
            // Convert vector of glm::mat4 to row-major float array for CUDA kernel
            // GLM is column-major, kernel expects row-major
            std::vector<float> transform_data(request.model_transforms.size() * 16);
            for (size_t i = 0; i < request.model_transforms.size(); ++i) {
                const auto& mat = request.model_transforms[i];
                for (int row = 0; row < 4; ++row) {
                    for (int col = 0; col < 4; ++col) {
                        transform_data[i * 16 + row * 4 + col] = mat[col][row]; // Transpose column to row major
                    }
                }
            }
            model_transforms_tensor = std::make_unique<Tensor>(
                Tensor::from_vector(transform_data,
                                    {request.model_transforms.size(), 4, 4},
                                    lfs::core::Device::CPU)
                    .cuda());
        }

        // Get transform indices pointer (already a tensor, just need to ensure it's on CUDA)
        std::unique_ptr<Tensor> transform_indices_cuda;
        Tensor* transform_indices_ptr = nullptr;
        if (request.transform_indices && request.transform_indices->is_valid()) {
            if (request.transform_indices->device() == lfs::core::Device::CUDA) {
                transform_indices_ptr = request.transform_indices.get();
            } else {
                transform_indices_cuda = std::make_unique<Tensor>(request.transform_indices->cuda());
                transform_indices_ptr = transform_indices_cuda.get();
            }
        }

        // Get selection mask pointer (already a tensor, just need to ensure it's on CUDA)
        std::unique_ptr<Tensor> selection_mask_cuda;
        Tensor* selection_mask_ptr = nullptr;
        if (request.selection_mask && request.selection_mask->is_valid()) {
            if (request.selection_mask->device() == lfs::core::Device::CUDA) {
                selection_mask_ptr = request.selection_mask.get();
            } else {
                selection_mask_cuda = std::make_unique<Tensor>(request.selection_mask->cuda());
                selection_mask_ptr = selection_mask_cuda.get();
            }
        }

        try {
            if (request.sh_degree != model.get_active_sh_degree()) {
                // Temporarily set sh_degree for rendering, then immediately restore
                int original_sh_degree = model.get_active_sh_degree();
                const_cast<lfs::core::SplatData&>(model).set_active_sh_degree(request.sh_degree);

                RenderResult result;

                if (request.gut || request.equirectangular) {
                    const auto camera_model = request.equirectangular
                                                  ? GutCameraModel::EQUIRECTANGULAR
                                                  : GutCameraModel::PINHOLE;
                    LOG_TRACE("GUT rasterizer: sh_degree {} -> {}", original_sh_degree, request.sh_degree);
                    auto render_output = gut_rasterize_tensor(
                        cam, const_cast<lfs::core::SplatData&>(model), background_,
                        request.scaling_modifier, camera_model);
                    result.image = std::move(render_output.image);
                    result.depth = std::move(render_output.depth);
                } else {
                    LOG_TRACE("Using TENSOR_NATIVE backend (sh_degree temporarily changed from {} to {})",
                              original_sh_degree, request.sh_degree);
                    Tensor screen_positions;
                    auto [image, depth] = rasterize_tensor(cam, const_cast<lfs::core::SplatData&>(model), background_,
                                                           request.show_rings, request.ring_width,
                                                           model_transforms_tensor.get(), transform_indices_ptr,
                                                           selection_mask_ptr,
                                                           request.output_screen_positions ? &screen_positions : nullptr,
                                                           request.brush_active, request.brush_x, request.brush_y, request.brush_radius,
                                                           request.brush_add_mode, request.brush_selection_tensor,
                                                           request.brush_saturation_mode, request.brush_saturation_amount,
                                                           request.selection_mode_rings,
                                                           request.show_center_markers,
                                                           request.crop_box_transform, request.crop_box_min, request.crop_box_max,
                                                           request.crop_inverse, request.crop_desaturate,
                                                           request.depth_filter_transform, request.depth_filter_min, request.depth_filter_max,
                                                           request.deleted_mask,
                                                           request.hovered_depth_id,
                                                           request.highlight_gaussian_id,
                                                           request.far_plane,
                                                           request.selected_node_mask,
                                                           request.desaturate_unselected,
                                                           request.selection_flash_intensity,
                                                           request.orthographic,
                                                           request.ortho_scale,
                                                           request.mip_filter);
                    result.image = std::move(image);
                    result.depth = std::move(depth);
                    if (request.output_screen_positions) {
                        result.screen_positions = std::move(screen_positions);
                    }
                }

                // IMMEDIATELY restore original sh_degree
                const_cast<lfs::core::SplatData&>(model).set_active_sh_degree(original_sh_degree);

                result.valid = true;
                result.orthographic = request.orthographic;
                result.far_plane = request.far_plane;
                LOG_TRACE("Rasterization completed successfully (sh_degree restored to {})", original_sh_degree);
                return result;
            }

            // No sh_degree change needed - safe to use model as-is
            lfs::core::SplatData& mutable_model = const_cast<lfs::core::SplatData&>(model);
            RenderResult result;

            if (request.gut || request.equirectangular) {
                const auto camera_model = request.equirectangular
                                              ? GutCameraModel::EQUIRECTANGULAR
                                              : GutCameraModel::PINHOLE;
                auto render_output = gut_rasterize_tensor(
                    cam, mutable_model, background_, request.scaling_modifier, camera_model);
                return RenderResult{
                    .image = std::move(render_output.image),
                    .depth = std::move(render_output.depth),
                    .valid = true,
                    .far_plane = request.far_plane,
                    .orthographic = request.orthographic};
            }

            // Use libtorch-free tensor-based rasterizer
            LOG_TRACE("Using TENSOR_NATIVE backend (libtorch-free rasterizer)");
            Tensor screen_positions;
            auto [image, depth] = rasterize_tensor(cam, mutable_model, background_,
                                                   request.show_rings, request.ring_width,
                                                   model_transforms_tensor.get(), transform_indices_ptr,
                                                   selection_mask_ptr,
                                                   request.output_screen_positions ? &screen_positions : nullptr,
                                                   request.brush_active, request.brush_x, request.brush_y, request.brush_radius,
                                                   request.brush_add_mode, request.brush_selection_tensor,
                                                   request.brush_saturation_mode, request.brush_saturation_amount,
                                                   request.selection_mode_rings,
                                                   request.show_center_markers,
                                                   request.crop_box_transform, request.crop_box_min, request.crop_box_max,
                                                   request.crop_inverse, request.crop_desaturate,
                                                   request.depth_filter_transform, request.depth_filter_min, request.depth_filter_max,
                                                   request.deleted_mask,
                                                   request.hovered_depth_id,
                                                   request.highlight_gaussian_id,
                                                   request.far_plane,
                                                   request.selected_node_mask,
                                                   request.desaturate_unselected,
                                                   request.selection_flash_intensity,
                                                   request.orthographic,
                                                   request.ortho_scale,
                                                   request.mip_filter);
            result.image = std::move(image);
            result.depth = std::move(depth);
            if (request.output_screen_positions) {
                result.screen_positions = std::move(screen_positions);
            }
            result.valid = true;
            result.orthographic = request.orthographic;
            result.far_plane = request.far_plane;

            LOG_TRACE("Rasterization completed successfully");
            return result;

        } catch (const std::exception& e) {
            LOG_ERROR("Rasterization failed: {}", e.what());
            return std::unexpected(std::format("Rasterization failed: {}", e.what()));
        }
    }

    Result<RenderingPipeline::RenderResult> RenderingPipeline::renderPointCloud(
        const lfs::core::SplatData& model,
        const RenderRequest& request) {

        LOG_TIMER_TRACE("RenderingPipeline::renderPointCloud");

        // Initialize point cloud renderer if needed
        if (!point_cloud_renderer_->isInitialized()) {
            LOG_DEBUG("Initializing point cloud renderer");
            if (auto result = point_cloud_renderer_->initialize(); !result) {
                LOG_ERROR("Failed to initialize point cloud renderer: {}", result.error());
                return std::unexpected(std::format("Failed to initialize point cloud renderer: {}",
                                                   result.error()));
            }
        }

        // Save GL state for FBO rendering
        GLint saved_viewport[4];
        GLint saved_fbo;
        glGetIntegerv(GL_VIEWPORT, saved_viewport);
        glGetIntegerv(GL_FRAMEBUFFER_BINDING, &saved_fbo);
        const GLboolean saved_scissor = glIsEnabled(GL_SCISSOR_TEST);
        if (saved_scissor)
            glDisable(GL_SCISSOR_TEST);

        // RAII restore
        const struct StateGuard {
            const GLint* vp;
            const GLint fbo;
            const GLboolean scissor;
            ~StateGuard() {
                glBindFramebuffer(GL_FRAMEBUFFER, fbo);
                glViewport(vp[0], vp[1], vp[2], vp[3]);
                if (scissor)
                    glEnable(GL_SCISSOR_TEST);
            }
        } guard{saved_viewport, saved_fbo, saved_scissor};

        // Create view matrix using the same convention as Viewport::getViewMatrix()
        glm::mat3 flip_yz = glm::mat3(
            1, 0, 0,
            0, -1, 0,
            0, 0, -1);

        // Convert from camera space (what we get in request) to view space
        glm::mat3 R_inv = glm::transpose(request.view_rotation); // Inverse of rotation matrix
        glm::vec3 t_inv = -R_inv * request.view_translation;     // Inverse translation

        // Apply flip
        R_inv = flip_yz * R_inv;
        t_inv = flip_yz * t_inv;

        // Build view matrix
        glm::mat4 view(1.0f);
        for (int i = 0; i < 3; ++i) {
            for (int j = 0; j < 3; ++j) {
                view[i][j] = R_inv[i][j];
            }
        }
        view[3][0] = t_inv.x;
        view[3][1] = t_inv.y;
        view[3][2] = t_inv.z;
        view[3][3] = 1.0f;

        // Create projection matrix (Y-flipped for OpenGL bottom-left origin)
        glm::mat4 projection = request.getProjectionMatrix();
        projection[1][1] *= -1.0f;

        // OPTIMIZATION: Use persistent FBO (avoids expensive glGenFramebuffers/glDeleteFramebuffers)
        // This saves ~3-5ms per frame by reusing the same FBO across renders
        ensureFBOSize(request.viewport_size.x, request.viewport_size.y);
        if (persistent_fbo_ == 0) {
            LOG_ERROR("Failed to setup persistent framebuffer");
            return std::unexpected("Failed to setup persistent framebuffer");
        }

        // Set viewport to match the request size
        glViewport(0, 0, request.viewport_size.x, request.viewport_size.y);

        // Render point cloud to framebuffer
        {
            LOG_TIMER_TRACE("point_cloud_renderer_->render");
            if (auto result = point_cloud_renderer_->render(model, view, projection,
                                                            request.voxel_size, request.background_color,
                                                            request.model_transforms, request.transform_indices,
                                                            request.equirectangular);
                !result) {
                LOG_ERROR("Point cloud rendering failed: {}", result.error());
                return std::unexpected(std::format("Point cloud rendering failed: {}", result.error()));
            }
        }

        const int width = request.viewport_size.x;
        const int height = request.viewport_size.y;
        RenderResult result;

#ifdef CUDA_GL_INTEROP_ENABLED
        if (use_fbo_interop_) {
            LOG_TIMER_TRACE("CUDA-GL FBO interop readback");

            const bool fbo_changed = fbo_interop_last_width_ != persistent_fbo_width_ ||
                                     fbo_interop_last_height_ != persistent_fbo_height_;
            const bool dims_mismatch = fbo_interop_texture_ &&
                                       (fbo_interop_texture_->getWidth() != width ||
                                        fbo_interop_texture_->getHeight() != height);
            const bool should_init = persistent_color_texture_ != 0 &&
                                     (!fbo_interop_texture_ || fbo_changed || dims_mismatch);

            if (should_init) {
                fbo_interop_texture_.reset();
                fbo_interop_texture_.emplace();
                if (auto init_result = fbo_interop_texture_->initForReading(
                        persistent_color_texture_, width, height);
                    !init_result) {
                    LOG_TRACE("FBO interop init failed: {}", init_result.error());
                    fbo_interop_texture_.reset();
                }
                fbo_interop_last_width_ = persistent_fbo_width_;
                fbo_interop_last_height_ = persistent_fbo_height_;
            }

            if (use_fbo_interop_ && fbo_interop_texture_) {
                Tensor image_hwc;
                if (auto read_result = fbo_interop_texture_->readToTensor(image_hwc); read_result) {
                    result.image = image_hwc.permute({2, 0, 1}).contiguous();
                    result.valid = true;
                    result.external_depth_texture = persistent_depth_texture_;
                    result.depth_is_ndc = true;
                } else {
                    LOG_TRACE("FBO interop read failed: {}", read_result.error());
                    fbo_interop_texture_.reset();
                    result.valid = false;
                }
            }
        }

        if (!result.valid)
#endif
        {
            LOG_TIMER_TRACE("PBO fallback readback");

            ensurePBOSize(width, height);

            // Ping-pong between two PBOs for double-buffering
            int current_pbo = pbo_index_;
            int next_pbo = 1 - pbo_index_;

            std::vector<float> pixels(width * height * 3);
            {
                // Start async readback into current PBO
                glBindBuffer(GL_PIXEL_PACK_BUFFER, pbo_[current_pbo]);
                glReadPixels(0, 0, width, height, GL_RGB, GL_FLOAT, nullptr);

                // Map the PBO to read data (may wait if transfer not complete)
                void* mapped_data = glMapBuffer(GL_PIXEL_PACK_BUFFER, GL_READ_ONLY);
                if (mapped_data) {
                    // Copy data from mapped PBO to our vector
                    std::memcpy(pixels.data(), mapped_data, width * height * 3 * sizeof(float));
                    glUnmapBuffer(GL_PIXEL_PACK_BUFFER);
                } else {
                    LOG_ERROR("Failed to map PBO for readback");
                    glBindBuffer(GL_PIXEL_PACK_BUFFER, 0);
                    return std::unexpected("Failed to map PBO for readback");
                }

                glBindBuffer(GL_PIXEL_PACK_BUFFER, 0);
            }

            // Swap PBO index for next frame
            pbo_index_ = next_pbo;

            const auto image_cpu = Tensor::from_vector(pixels, {static_cast<size_t>(height), static_cast<size_t>(width), 3},
                                                       lfs::core::Device::CPU);
            {
                LOG_TIMER_TRACE("permute and cuda upload");
                result.image = image_cpu.permute({2, 0, 1}).cuda();
            }
            result.external_depth_texture = persistent_depth_texture_;
            result.depth_is_ndc = true;
            result.valid = true;
        }

        result.orthographic = request.orthographic;
        result.far_plane = request.far_plane;

        LOG_TRACE("Point cloud rendering completed");
        return result;
    }

    Result<RenderingPipeline::RenderResult> RenderingPipeline::renderRawPointCloud(
        const lfs::core::PointCloud& point_cloud,
        const RenderRequest& request) {

        LOG_TIMER_TRACE("RenderingPipeline::renderRawPointCloud");

        // Initialize point cloud renderer if needed
        if (!point_cloud_renderer_->isInitialized()) {
            LOG_DEBUG("Initializing point cloud renderer");
            if (auto result = point_cloud_renderer_->initialize(); !result) {
                LOG_ERROR("Failed to initialize point cloud renderer: {}", result.error());
                return std::unexpected(std::format("Failed to initialize point cloud renderer: {}",
                                                   result.error()));
            }
        }

        // Save GL state for FBO rendering
        GLint saved_viewport[4];
        GLint saved_fbo;
        glGetIntegerv(GL_VIEWPORT, saved_viewport);
        glGetIntegerv(GL_FRAMEBUFFER_BINDING, &saved_fbo);
        const GLboolean saved_scissor = glIsEnabled(GL_SCISSOR_TEST);
        if (saved_scissor)
            glDisable(GL_SCISSOR_TEST);

        // RAII restore
        const struct StateGuard {
            const GLint* vp;
            const GLint fbo;
            const GLboolean scissor;
            ~StateGuard() {
                glBindFramebuffer(GL_FRAMEBUFFER, fbo);
                glViewport(vp[0], vp[1], vp[2], vp[3]);
                if (scissor)
                    glEnable(GL_SCISSOR_TEST);
            }
        } guard{saved_viewport, saved_fbo, saved_scissor};

        // Create view matrix using the same convention as Viewport::getViewMatrix()
        glm::mat3 flip_yz = glm::mat3(
            1, 0, 0,
            0, -1, 0,
            0, 0, -1);

        // Convert from camera space (what we get in request) to view space
        glm::mat3 R_inv = glm::transpose(request.view_rotation); // Inverse of rotation matrix
        glm::vec3 t_inv = -R_inv * request.view_translation;     // Inverse translation

        // Apply flip
        R_inv = flip_yz * R_inv;
        t_inv = flip_yz * t_inv;

        // Build view matrix
        glm::mat4 view(1.0f);
        for (int i = 0; i < 3; ++i) {
            for (int j = 0; j < 3; ++j) {
                view[i][j] = R_inv[i][j];
            }
        }
        view[3][0] = t_inv.x;
        view[3][1] = t_inv.y;
        view[3][2] = t_inv.z;
        view[3][3] = 1.0f;

        // Apply model transform if provided (for point cloud node transforms)
        if (!request.model_transforms.empty()) {
            // model_view = view * model_transform
            // This transforms points from model space -> world space -> view space
            view = view * request.model_transforms[0];
        }

        // Create projection matrix (Y-flipped for OpenGL bottom-left origin)
        glm::mat4 projection = request.getProjectionMatrix();
        projection[1][1] *= -1.0f;

        // OPTIMIZATION: Use persistent FBO
        ensureFBOSize(request.viewport_size.x, request.viewport_size.y);
        if (persistent_fbo_ == 0) {
            LOG_ERROR("Failed to setup persistent framebuffer");
            return std::unexpected("Failed to setup persistent framebuffer");
        }

        // Set viewport to match the request size
        glViewport(0, 0, request.viewport_size.x, request.viewport_size.y);

        // Raw point clouds: transform already baked into view matrix
        {
            LOG_TIMER_TRACE("point_cloud_renderer_->render(PointCloud)");
            if (auto result = point_cloud_renderer_->render(point_cloud, view, projection,
                                                            request.voxel_size, request.background_color,
                                                            {}, nullptr, request.equirectangular);
                !result) {
                LOG_ERROR("Raw point cloud rendering failed: {}", result.error());
                return std::unexpected(std::format("Raw point cloud rendering failed: {}", result.error()));
            }
        }

        const int width = request.viewport_size.x;
        const int height = request.viewport_size.y;
        RenderResult result;

#ifdef CUDA_GL_INTEROP_ENABLED
        if (use_fbo_interop_) {
            LOG_TIMER_TRACE("CUDA-GL FBO interop readback");

            const bool fbo_changed = fbo_interop_last_width_ != persistent_fbo_width_ ||
                                     fbo_interop_last_height_ != persistent_fbo_height_;
            const bool dims_mismatch = fbo_interop_texture_ &&
                                       (fbo_interop_texture_->getWidth() != width ||
                                        fbo_interop_texture_->getHeight() != height);
            const bool should_init = persistent_color_texture_ != 0 &&
                                     (!fbo_interop_texture_ || fbo_changed || dims_mismatch);

            if (should_init) {
                fbo_interop_texture_.reset();
                fbo_interop_texture_.emplace();
                if (auto init_result = fbo_interop_texture_->initForReading(
                        persistent_color_texture_, width, height);
                    !init_result) {
                    LOG_TRACE("FBO interop init failed: {}", init_result.error());
                    fbo_interop_texture_.reset();
                }
                fbo_interop_last_width_ = persistent_fbo_width_;
                fbo_interop_last_height_ = persistent_fbo_height_;
            }

            if (use_fbo_interop_ && fbo_interop_texture_) {
                Tensor image_hwc;
                if (auto read_result = fbo_interop_texture_->readToTensor(image_hwc); read_result) {
                    result.image = image_hwc.permute({2, 0, 1}).contiguous();
                    result.valid = true;
                    result.external_depth_texture = persistent_depth_texture_;
                    result.depth_is_ndc = true;
                } else {
                    LOG_TRACE("FBO interop read failed: {}", read_result.error());
                    fbo_interop_texture_.reset();
                    result.valid = false;
                }
            }
        }

        // Fallback to PBO path if interop failed
        if (!result.valid)
#endif
        {
            LOG_TIMER_TRACE("PBO fallback readback");

            ensurePBOSize(width, height);

            int current_pbo = pbo_index_;
            int next_pbo = 1 - pbo_index_;

            std::vector<float> pixels(width * height * 3);
            {
                glBindBuffer(GL_PIXEL_PACK_BUFFER, pbo_[current_pbo]);
                glReadPixels(0, 0, width, height, GL_RGB, GL_FLOAT, nullptr);

                void* mapped_data = glMapBuffer(GL_PIXEL_PACK_BUFFER, GL_READ_ONLY);
                if (mapped_data) {
                    std::memcpy(pixels.data(), mapped_data, width * height * 3 * sizeof(float));
                    glUnmapBuffer(GL_PIXEL_PACK_BUFFER);
                } else {
                    LOG_ERROR("Failed to map PBO for readback");
                    glBindBuffer(GL_PIXEL_PACK_BUFFER, 0);
                    return std::unexpected("Failed to map PBO for readback");
                }

                glBindBuffer(GL_PIXEL_PACK_BUFFER, 0);
            }

            pbo_index_ = next_pbo;

            const auto image_cpu = Tensor::from_vector(pixels, {static_cast<size_t>(height), static_cast<size_t>(width), 3},
                                                       lfs::core::Device::CPU);
            {
                LOG_TIMER_TRACE("permute and cuda upload");
                result.image = image_cpu.permute({2, 0, 1}).cuda();
            }
            result.external_depth_texture = persistent_depth_texture_;
            result.depth_is_ndc = true;
            result.valid = true;
        }

        result.orthographic = request.orthographic;
        result.far_plane = request.far_plane;
        LOG_TRACE("Raw point cloud rendering completed");
        return result;
    }

    void RenderingPipeline::applyDepthParams(
        const RenderResult& result,
        ScreenQuadRenderer& renderer,
        const glm::ivec2& viewport_size) {

        DepthParams params = renderer.getDepthParams();
        params.near_plane = result.near_plane;
        params.far_plane = result.far_plane;
        params.orthographic = result.orthographic;

        if (result.external_depth_texture != 0) {
            // Use external OpenGL texture directly (zero-copy)
            params.has_depth = true;
            params.depth_is_ndc = result.depth_is_ndc;
            params.external_depth_texture = result.external_depth_texture;
        } else if (result.depth.is_valid()) {
            // Upload depth from CUDA
            if (renderer.uploadDepthFromCUDA(result.depth, viewport_size.x, viewport_size.y)) {
                params.has_depth = true;
                params.depth_is_ndc = result.depth_is_ndc;
                params.external_depth_texture = 0;
            }
        }

        renderer.setDepthParams(params);
    }

    Result<void> RenderingPipeline::uploadToScreen(
        const RenderResult& result,
        ScreenQuadRenderer& renderer,
        const glm::ivec2& viewport_size) {
        LOG_TIMER_TRACE("RenderingPipeline::uploadToScreen");

        if (!result.valid || !result.image.is_valid()) {
            LOG_ERROR("Invalid render result for upload");
            return std::unexpected("Invalid render result");
        }

        if (renderer.isInteropEnabled() && result.image.device() == lfs::core::Device::CUDA) {
            const auto image_hwc = result.image.permute({1, 2, 0}).contiguous();

            if (image_hwc.size(0) == static_cast<size_t>(viewport_size.y) &&
                image_hwc.size(1) == static_cast<size_t>(viewport_size.x)) {
                if (auto upload_result = renderer.uploadFromCUDA(image_hwc, viewport_size.x, viewport_size.y);
                    !upload_result) {
                    return upload_result;
                }
                applyDepthParams(result, renderer, viewport_size);
                return {};
            }
            LOG_WARN("Dimension mismatch: image {}x{}, expected {}x{}",
                     image_hwc.size(1), image_hwc.size(0), viewport_size.x, viewport_size.y);
        }

        // CPU fallback
        const auto image = (result.image * 255.0f)
                               .cpu()
                               .to(lfs::core::DataType::UInt8)
                               .permute({1, 2, 0})
                               .contiguous();

        if (image.size(0) != static_cast<size_t>(viewport_size.y) ||
            image.size(1) != static_cast<size_t>(viewport_size.x) ||
            !image.ptr<unsigned char>()) {
            LOG_ERROR("CPU upload dimension mismatch: {}x{} vs {}x{}",
                      image.size(1), image.size(0), viewport_size.x, viewport_size.y);
            return std::unexpected("Image dimensions mismatch or invalid data");
        }

        if (auto upload_result = renderer.uploadData(image.ptr<unsigned char>(),
                                                     viewport_size.x, viewport_size.y);
            !upload_result) {
            return upload_result;
        }

        applyDepthParams(result, renderer, viewport_size);
        return {};
    }

    Result<lfs::core::Camera> RenderingPipeline::createCamera(const RenderRequest& request) {
        LOG_TIMER_TRACE("RenderingPipeline::createCamera");

        // Convert view matrix to camera matrix
        std::vector<float> R_data = {
            request.view_rotation[0][0], request.view_rotation[1][0], request.view_rotation[2][0],
            request.view_rotation[0][1], request.view_rotation[1][1], request.view_rotation[2][1],
            request.view_rotation[0][2], request.view_rotation[1][2], request.view_rotation[2][2]};

        auto R_tensor = Tensor::from_vector(R_data, {3, 3}, lfs::core::Device::CPU);

        std::vector<float> t_data = {
            request.view_translation[0],
            request.view_translation[1],
            request.view_translation[2]};

        auto t_tensor = Tensor::from_vector(t_data, {3, 1}, lfs::core::Device::CPU);

        // Convert from view to camera space
        R_tensor = R_tensor.transpose(0, 1);
        t_tensor = (-R_tensor.mm(t_tensor)).squeeze();

        // Compute field of view
        glm::vec2 fov = computeFov(request.fov,
                                   request.viewport_size.x,
                                   request.viewport_size.y);

        try {
            return lfs::core::Camera(
                R_tensor,
                t_tensor,
                lfs::core::fov2focal(fov.x, request.viewport_size.x),
                lfs::core::fov2focal(fov.y, request.viewport_size.y),
                request.viewport_size.x / 2.0f,
                request.viewport_size.y / 2.0f,
                Tensor::empty({0}, lfs::core::Device::CPU, lfs::core::DataType::Float32),
                Tensor::empty({0}, lfs::core::Device::CPU, lfs::core::DataType::Float32),
                lfs::core::CameraModelType::PINHOLE,
                "render_camera",
                "none",
                std::filesystem::path{}, // No mask path for render camera
                request.viewport_size.x,
                request.viewport_size.y,
                -1);
        } catch (const std::exception& e) {
            LOG_ERROR("Failed to create camera: {}", e.what());
            return std::unexpected(std::format("Failed to create camera: {}", e.what()));
        }
    }

    glm::vec2 RenderingPipeline::computeFov(float fov_degrees, int width, int height) {
        float fov_rad = glm::radians(fov_degrees);
        float aspect = static_cast<float>(width) / height;

        return glm::vec2(
            atan(tan(fov_rad / 2.0f) * aspect) * 2.0f,
            fov_rad);
    }

    void RenderingPipeline::ensureFBOSize(int width, int height) {
        const int alloc_width = ((width + GPU_ALIGNMENT - 1) / GPU_ALIGNMENT) * GPU_ALIGNMENT;
        const int alloc_height = ((height + GPU_ALIGNMENT - 1) / GPU_ALIGNMENT) * GPU_ALIGNMENT;

        if (persistent_fbo_ != 0 && alloc_width == persistent_fbo_width_ && alloc_height == persistent_fbo_height_) {
            glBindFramebuffer(GL_FRAMEBUFFER, persistent_fbo_);
            return;
        }

        LOG_DEBUG("FBO resize: {}x{} -> {}x{}", persistent_fbo_width_, persistent_fbo_height_, alloc_width, alloc_height);

        if (persistent_fbo_ != 0) {
#ifdef CUDA_GL_INTEROP_ENABLED
            fbo_interop_texture_.reset();
#endif
            cleanupFBO();
        }

        // Create new FBO
        glGenFramebuffers(1, &persistent_fbo_);
        if (persistent_fbo_ == 0) {
            LOG_ERROR("Failed to create persistent framebuffer");
            return;
        }
        glBindFramebuffer(GL_FRAMEBUFFER, persistent_fbo_);

        glGenTextures(1, &persistent_color_texture_);
        glBindTexture(GL_TEXTURE_2D, persistent_color_texture_);
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB32F, alloc_width, alloc_height, 0, GL_RGB, GL_FLOAT, nullptr);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
        glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D,
                               persistent_color_texture_, 0);

        glGenTextures(1, &persistent_depth_texture_);
        glBindTexture(GL_TEXTURE_2D, persistent_depth_texture_);
        glTexImage2D(GL_TEXTURE_2D, 0, GL_DEPTH_COMPONENT32F, alloc_width, alloc_height,
                     0, GL_DEPTH_COMPONENT, GL_FLOAT, nullptr);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
        glFramebufferTexture2D(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_TEXTURE_2D,
                               persistent_depth_texture_, 0);

        const GLenum status = glCheckFramebufferStatus(GL_FRAMEBUFFER);
        if (status != GL_FRAMEBUFFER_COMPLETE) {
            LOG_ERROR("FBO incomplete: 0x{:x}", status);
            cleanupFBO();
            return;
        }

        glBindTexture(GL_TEXTURE_2D, 0);
        glFinish();

        persistent_fbo_width_ = alloc_width;
        persistent_fbo_height_ = alloc_height;

        LOG_DEBUG("FBO allocated: {}x{}", alloc_width, alloc_height);
    }

    void RenderingPipeline::cleanupFBO() {
        if (persistent_fbo_ != 0) {
            glDeleteFramebuffers(1, &persistent_fbo_);
            persistent_fbo_ = 0;
        }
        if (persistent_color_texture_ != 0) {
            glDeleteTextures(1, &persistent_color_texture_);
            persistent_color_texture_ = 0;
        }
        if (persistent_depth_texture_ != 0) {
            glDeleteTextures(1, &persistent_depth_texture_);
            persistent_depth_texture_ = 0;
        }
        persistent_fbo_width_ = 0;
        persistent_fbo_height_ = 0;
    }

    void RenderingPipeline::ensurePBOSize(int width, int height) {
        const int alloc_width = ((width + GPU_ALIGNMENT - 1) / GPU_ALIGNMENT) * GPU_ALIGNMENT;
        const int alloc_height = ((height + GPU_ALIGNMENT - 1) / GPU_ALIGNMENT) * GPU_ALIGNMENT;

        if (pbo_[0] != 0 && alloc_width == allocated_pbo_width_ && alloc_height == allocated_pbo_height_) {
            pbo_width_ = width;
            pbo_height_ = height;
            return;
        }

        LOG_DEBUG("PBO resize: {}x{} -> {}x{}", allocated_pbo_width_, allocated_pbo_height_, alloc_width, alloc_height);

        if (pbo_[0] != 0) {
            cleanupPBO();
        }

        const size_t buffer_size = static_cast<size_t>(alloc_width) * alloc_height * 3 * sizeof(float);

        glGenBuffers(2, pbo_);
        for (int i = 0; i < 2; i++) {
            glBindBuffer(GL_PIXEL_PACK_BUFFER, pbo_[i]);
            glBufferData(GL_PIXEL_PACK_BUFFER, static_cast<GLsizeiptr>(buffer_size), nullptr, GL_STREAM_READ);
        }
        glBindBuffer(GL_PIXEL_PACK_BUFFER, 0);

        allocated_pbo_width_ = alloc_width;
        allocated_pbo_height_ = alloc_height;
        pbo_width_ = width;
        pbo_height_ = height;
        pbo_index_ = 0;
    }

    void RenderingPipeline::cleanupPBO() {
        if (pbo_[0] != 0 || pbo_[1] != 0) {
            glDeleteBuffers(2, pbo_);
            pbo_[0] = 0;
            pbo_[1] = 0;
        }
        pbo_width_ = 0;
        pbo_height_ = 0;
        allocated_pbo_width_ = 0;
        allocated_pbo_height_ = 0;
        pbo_index_ = 0;
    }

} // namespace lfs::rendering
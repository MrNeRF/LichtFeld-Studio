/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "rendering_pipeline.hpp"
#include "gs_rasterizer.hpp"
#include "training/rasterization/rasterizer.hpp"

#include <print>

namespace gs::rendering {

    RenderingPipeline::RenderingPipeline()
        : background_(torch::zeros({3}, torch::kFloat32).to(torch::kCUDA)) {
        point_cloud_renderer_ = std::make_unique<PointCloudRenderer>();
        LOG_DEBUG("RenderingPipeline initialized");
    }

    Result<RenderingPipeline::RenderResult> RenderingPipeline::render(
        const SplatData& model,
        const RenderRequest& request) {

        LOG_TIMER_TRACE("RenderingPipeline::render");

        // Validate dimensions
        if (request.viewport_size.x <= 0 || request.viewport_size.y <= 0 ||
            request.viewport_size.x > 16384 || request.viewport_size.y > 16384) {
            LOG_ERROR("Invalid viewport dimensions: {}x{}", request.viewport_size.x, request.viewport_size.y);
            return std::unexpected("Invalid viewport dimensions");
        }

        // Check if we should use point cloud rendering
        if (request.point_cloud_mode) {
            LOG_TRACE("Using point cloud rendering mode");
            return renderPointCloud(model, request);
        }

        // Regular gaussian splatting rendering
        LOG_TRACE("Using gaussian splatting rendering mode");

        // Update background tensor in-place to avoid allocation
        background_[0] = request.background_color.r;
        background_[1] = request.background_color.g;
        background_[2] = request.background_color.b;

        // Create camera for this frame
        auto cam_result = createCamera(request);
        if (!cam_result) {
            return std::unexpected(cam_result.error());
        }
        Camera cam = std::move(*cam_result);

        // Handle crop box conversion
        const geometry::BoundingBox* geom_bbox = nullptr;
        std::unique_ptr<geometry::BoundingBox> temp_bbox;

        if (request.crop_box) {
            // Create a temporary geometry::BoundingBox with the full transform
            temp_bbox = std::make_unique<geometry::BoundingBox>();
            temp_bbox->setBounds(request.crop_box->getMinBounds(), request.crop_box->getMaxBounds());
            temp_bbox->setworld2BBox(request.crop_box->getworld2BBox());
            geom_bbox = temp_bbox.get();
            LOG_TRACE("Using crop box for rendering");
        }

        try {
            // Perform rendering with fast_rasterize

            SplatData cropped_model;
            if (request.crop_box) {
                cropped_model = model.crop_by_cropbox(*request.crop_box);
            }

            SplatData& mutable_model = request.crop_box ? const_cast<SplatData&>(cropped_model) : const_cast<SplatData&>(model);

            mutable_model.set_active_sh_degree(request.sh_degree);

            RenderResult result;
            if (request.gut || request.equirectangular) {
                auto render_result = gs::training::rasterize(
                    cam, mutable_model, background_, request.scaling_modifier, false, request.antialiasing, static_cast<training::RenderMode>(request.render_mode), nullptr);
                result.image = render_result.image;
                result.depth = render_result.depth;
            } else {
                result.image = rasterize(cam, mutable_model, background_);
                result.depth = torch::empty({0}, torch::kFloat32); // No depth support in fast_rasterize; set empty
            }
            result.valid = true;

            LOG_TRACE("Rasterization completed successfully");
            return result;

        } catch (const std::exception& e) {
            LOG_ERROR("Rasterization failed: {}", e.what());
            return std::unexpected(std::format("Rasterization failed: {}", e.what()));
        }
    }

    Result<RenderingPipeline::RenderResult> RenderingPipeline::renderPointCloud(
        const SplatData& model,
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

        // Save current OpenGL state
        GLint current_viewport[4];
        GLint current_fbo;
        glGetIntegerv(GL_VIEWPORT, current_viewport);
        glGetIntegerv(GL_FRAMEBUFFER_BINDING, &current_fbo);

        // RAII guard to restore state
        struct StateGuard {
            GLint viewport[4];
            GLint fbo;
            ~StateGuard() {
                glBindFramebuffer(GL_FRAMEBUFFER, fbo);
                glViewport(viewport[0], viewport[1], viewport[2], viewport[3]);
            }
        } state_guard{
            {current_viewport[0], current_viewport[1], current_viewport[2], current_viewport[3]},
            current_fbo};

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

        // Create projection matrix
        float aspect = static_cast<float>(request.viewport_size.x) / request.viewport_size.y;
        float fov_rad = glm::radians(request.fov);
        glm::mat4 projection = glm::perspective(fov_rad, aspect, 0.1f, 1000.0f);

        // Create framebuffer for offscreen rendering using RAII
        auto fbo_result = create_vao(); // Using create_vao as proxy for FBO creation
        if (!fbo_result) {
            LOG_ERROR("Failed to create framebuffer");
            return std::unexpected("Failed to create framebuffer");
        }

        GLuint fbo, color_texture, depth_texture;
        glGenFramebuffers(1, &fbo);
        if (fbo == 0) {
            LOG_ERROR("Failed to create framebuffer - glGenFramebuffers returned 0");
            return std::unexpected("Failed to create framebuffer");
        }

        // RAII cleanup for OpenGL resources
        struct FBOGuard {
            GLuint fbo, color_tex, depth_tex;
            ~FBOGuard() {
                if (fbo)
                    glDeleteFramebuffers(1, &fbo);
                if (color_tex)
                    glDeleteTextures(1, &color_tex);
                if (depth_tex)
                    glDeleteTextures(1, &depth_tex);
            }
        } fbo_guard{fbo, 0, 0};

        glBindFramebuffer(GL_FRAMEBUFFER, fbo);

        // Color texture
        glGenTextures(1, &color_texture);
        fbo_guard.color_tex = color_texture;
        glBindTexture(GL_TEXTURE_2D, color_texture);
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB32F, request.viewport_size.x, request.viewport_size.y,
                     0, GL_RGB, GL_FLOAT, nullptr);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
        glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, color_texture, 0);

        // Depth texture
        glGenTextures(1, &depth_texture);
        fbo_guard.depth_tex = depth_texture;
        glBindTexture(GL_TEXTURE_2D, depth_texture);
        glTexImage2D(GL_TEXTURE_2D, 0, GL_DEPTH_COMPONENT32F, request.viewport_size.x, request.viewport_size.y,
                     0, GL_DEPTH_COMPONENT, GL_FLOAT, nullptr);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
        glFramebufferTexture2D(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_TEXTURE_2D, depth_texture, 0);

        GLenum fb_status = glCheckFramebufferStatus(GL_FRAMEBUFFER);
        if (fb_status != GL_FRAMEBUFFER_COMPLETE) {
            LOG_ERROR("Framebuffer not complete: 0x{:x}", fb_status);
            return std::unexpected(std::format("Framebuffer not complete: 0x{:x}", fb_status));
        }

        // Set viewport to match the request size
        glViewport(0, 0, request.viewport_size.x, request.viewport_size.y);

        // Render point cloud to framebuffer
        if (auto result = point_cloud_renderer_->render(model, view, projection,
                                                        request.voxel_size, request.background_color);
            !result) {
            LOG_ERROR("Point cloud rendering failed: {}", result.error());
            return std::unexpected(std::format("Point cloud rendering failed: {}", result.error()));
        }

        // Read back the rendered image
        std::vector<float> pixels(request.viewport_size.x * request.viewport_size.y * 3);
        glReadPixels(0, 0, request.viewport_size.x, request.viewport_size.y, GL_RGB, GL_FLOAT, pixels.data());

        // Convert to torch tensor
        auto options = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCPU);
        torch::Tensor image_cpu = torch::from_blob(pixels.data(),
                                                   {request.viewport_size.y, request.viewport_size.x, 3},
                                                   options)
                                      .clone();

        // Flip vertically (OpenGL has origin at bottom-left)
        image_cpu = torch::flip(image_cpu, {0});

        // Convert to CHW format and move to CUDA
        RenderResult result;
        result.image = image_cpu.permute({2, 0, 1}).to(torch::kCUDA);
        result.valid = true;

        LOG_TRACE("Point cloud rendering completed");
        return result;
    }

    Result<void> RenderingPipeline::uploadToScreen(
        const RenderResult& result,
        ScreenQuadRenderer& renderer,
        const glm::ivec2& viewport_size) {

        if (!result.valid || !result.image.defined()) {
            LOG_ERROR("Invalid render result for upload");
            return std::unexpected("Invalid render result");
        }

        // Try direct CUDA upload if available
        if (renderer.isInteropEnabled() && result.image.is_cuda()) {
            LOG_TRACE("Using CUDA interop for screen upload");
            // Keep data on GPU - convert [C, H, W] to [H, W, C] format
            auto image_hwc = result.image.permute({1, 2, 0}).contiguous();

            if (image_hwc.size(0) == viewport_size.y && image_hwc.size(1) == viewport_size.x) {
                return renderer.uploadFromCUDA(image_hwc, viewport_size.x, viewport_size.y);
            }
        }

        // Fallback to CPU copy
        LOG_TRACE("Using CPU copy for screen upload");
        auto image = (result.image * 255)
                         .to(torch::kCPU)
                         .to(torch::kU8)
                         .permute({1, 2, 0})
                         .contiguous();

        if (image.size(0) != viewport_size.y ||
            image.size(1) != viewport_size.x ||
            !image.data_ptr<unsigned char>()) {
            LOG_ERROR("Image dimensions mismatch or invalid data");
            return std::unexpected("Image dimensions mismatch or invalid data");
        }

        return renderer.uploadData(image.data_ptr<unsigned char>(),
                                   viewport_size.x, viewport_size.y);
    }

    Result<Camera> RenderingPipeline::createCamera(const RenderRequest& request) {
        LOG_TIMER_TRACE("RenderingPipeline::createCamera");

        // Convert view matrix to camera matrix
        torch::Tensor R_tensor = torch::tensor({request.view_rotation[0][0], request.view_rotation[1][0], request.view_rotation[2][0],
                                                request.view_rotation[0][1], request.view_rotation[1][1], request.view_rotation[2][1],
                                                request.view_rotation[0][2], request.view_rotation[1][2], request.view_rotation[2][2]},
                                               torch::TensorOptions().dtype(torch::kFloat32))
                                     .reshape({3, 3});

        torch::Tensor t_tensor = torch::tensor({request.view_translation[0],
                                                request.view_translation[1],
                                                request.view_translation[2]},
                                               torch::TensorOptions().dtype(torch::kFloat32))
                                     .reshape({3, 1});

        // Convert from view to camera space
        R_tensor = R_tensor.transpose(0, 1);
        t_tensor = -R_tensor.mm(t_tensor).squeeze();

        // Compute field of view
        glm::vec2 fov = computeFov(request.fov,
                                   request.viewport_size.x,
                                   request.viewport_size.y);

        try {
            return Camera(
                R_tensor,
                t_tensor,
                fov2focal(fov.x, request.viewport_size.x),
                fov2focal(fov.y, request.viewport_size.y),
                request.viewport_size.x / 2.0f,
                request.viewport_size.y / 2.0f,
                torch::empty({0}, torch::kFloat32),
                torch::empty({0}, torch::kFloat32),
                request.equirectangular ? gsplat::CameraModelType::EQUIRECTANGULAR : gsplat::CameraModelType::PINHOLE,
                "render_camera",
                "none",
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

} // namespace gs::rendering
/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "rendering_pipeline.hpp"
#include "core/camera.hpp"
#include "core/point_cloud.hpp"
#include "core/splat_data.hpp"
#include "gl_state_guard.hpp"
#include "gs_rasterizer_tensor.hpp"
#include "image_layout.hpp"

#include <cstring>
#include <print>

namespace lfs::rendering {

    namespace {
        constexpr int GPU_ALIGNMENT = 16; // 16-pixel alignment for GPU texture efficiency

        [[nodiscard]] GpuFrame buildPersistentGpuFrame(const GLuint color_texture,
                                                       const GLuint depth_texture,
                                                       const glm::ivec2 alloc_size,
                                                       const glm::ivec2 render_size,
                                                       const float far_plane,
                                                       const bool orthographic) {
            return {
                .color = {.id = color_texture,
                          .size = alloc_size,
                          .texcoord_scale = {
                              alloc_size.x > 0 ? static_cast<float>(render_size.x) / static_cast<float>(alloc_size.x) : 1.0f,
                              alloc_size.y > 0 ? static_cast<float>(render_size.y) / static_cast<float>(alloc_size.y) : 1.0f}},
                .depth = {.id = depth_texture, .size = alloc_size, .texcoord_scale = {alloc_size.x > 0 ? static_cast<float>(render_size.x) / static_cast<float>(alloc_size.x) : 1.0f, alloc_size.y > 0 ? static_cast<float>(render_size.y) / static_cast<float>(alloc_size.y) : 1.0f}},
                .depth_is_ndc = true,
                .near_plane = DEFAULT_NEAR_PLANE,
                .far_plane = far_plane,
                .orthographic = orthographic};
        }

        [[nodiscard]] glm::mat4 buildPointCloudViewMatrix(
            const RenderingPipeline::RasterRequest& request,
            const bool apply_first_model_transform = false) {
            glm::mat3 flip_yz = glm::mat3(
                1, 0, 0,
                0, -1, 0,
                0, 0, -1);

            glm::mat3 rotation_inv = glm::transpose(request.view_rotation);
            glm::vec3 translation_inv = -rotation_inv * request.view_translation;

            rotation_inv = flip_yz * rotation_inv;
            translation_inv = flip_yz * translation_inv;

            glm::mat4 view(1.0f);
            for (int i = 0; i < 3; ++i) {
                for (int j = 0; j < 3; ++j) {
                    view[i][j] = rotation_inv[i][j];
                }
            }
            view[3][0] = translation_inv.x;
            view[3][1] = translation_inv.y;
            view[3][2] = translation_inv.z;
            view[3][3] = 1.0f;

            if (apply_first_model_transform && !request.model_transforms.empty()) {
                view = view * request.model_transforms[0];
            }

            return view;
        }

        [[nodiscard]] glm::mat4 buildPointCloudProjectionMatrix(
            const RenderingPipeline::RasterRequest& request) {
            glm::mat4 projection = request.getProjectionMatrix();
            projection[1][1] *= -1.0f;
            return projection;
        }

        [[nodiscard]] bool tensorMatchesGaussianCount(const Tensor* const tensor,
                                                      const size_t gaussian_count) {
            return tensor == nullptr || !tensor->is_valid() || tensor->numel() == gaussian_count;
        }
    } // namespace

    RenderingPipeline::RenderingPipeline()
        : background_(Tensor::zeros({3}, lfs::core::Device::CUDA, lfs::core::DataType::Float32)) {
        point_cloud_renderer_ = std::make_unique<PointCloudRenderer>();
        LOG_DEBUG("RenderingPipeline initialized");
    }

    RenderingPipeline::~RenderingPipeline() {
        cleanupFBO();
        cleanupPBO();
    }

    void RenderingPipeline::resetResources() {
#ifdef CUDA_GL_INTEROP_ENABLED
        fbo_interop_texture_.reset();
        fbo_interop_last_width_ = 0;
        fbo_interop_last_height_ = 0;
#endif
        cleanupFBO();
        cleanupPBO();
    }

    Result<RenderingPipeline::ImageRenderResult> RenderingPipeline::renderGaussianImage(
        const lfs::core::SplatData& model,
        const RasterRequest& request) {

        LOG_TIMER_TRACE("RenderingPipeline::renderGaussianImage");

        // Validate dimensions
        if (request.viewport_size.x <= 0 || request.viewport_size.y <= 0 ||
            request.viewport_size.x > 16384 || request.viewport_size.y > 16384) {
            LOG_ERROR("Invalid viewport dimensions: {}x{}", request.viewport_size.x, request.viewport_size.y);
            return std::unexpected("Invalid viewport dimensions");
        }

        return renderGaussianImageResult(model, request, nullptr);
    }

    Result<RenderingPipeline::ImageRenderResult> RenderingPipeline::renderPointCloudImage(
        const lfs::core::SplatData& model,
        const RasterRequest& request) {

        LOG_TIMER_TRACE("RenderingPipeline::renderPointCloudImage");

        if (request.viewport_size.x <= 0 || request.viewport_size.y <= 0 ||
            request.viewport_size.x > 16384 || request.viewport_size.y > 16384) {
            LOG_ERROR("Invalid viewport dimensions: {}x{}", request.viewport_size.x, request.viewport_size.y);
            return std::unexpected("Invalid viewport dimensions");
        }

        return renderPointCloudImageResult(model, request);
    }

    Result<Tensor> RenderingPipeline::renderScreenPositions(
        const lfs::core::SplatData& model,
        const RasterRequest& request) {

        LOG_TIMER_TRACE("RenderingPipeline::renderScreenPositions");

        Tensor screen_positions;
        auto render_result = renderGaussianImageResult(model, request, &screen_positions);
        if (!render_result) {
            return std::unexpected(render_result.error());
        }
        if (!screen_positions.is_valid()) {
            return std::unexpected("Screen-position render returned no screen positions");
        }
        return screen_positions;
    }

    Result<RenderingPipeline::ImageRenderResult> RenderingPipeline::renderGaussianImageResult(
        const lfs::core::SplatData& model,
        const RasterRequest& request,
        Tensor* const screen_positions_out) {

        // Regular gaussian splatting rendering
        LOG_TRACE("Using gaussian splatting rendering mode");
        const size_t gaussian_count = static_cast<size_t>(model.size());

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
        if (!tensorMatchesGaussianCount(transform_indices_ptr, gaussian_count)) {
            LOG_WARN("Ignoring transform_indices with stale size: model has {}, tensor has {}",
                     gaussian_count, transform_indices_ptr->numel());
            transform_indices_ptr = nullptr;
            transform_indices_cuda.reset();
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
        if (!tensorMatchesGaussianCount(selection_mask_ptr, gaussian_count)) {
            LOG_WARN("Ignoring selection_mask with stale size: model has {}, tensor has {}",
                     gaussian_count, selection_mask_ptr->numel());
            selection_mask_ptr = nullptr;
            selection_mask_cuda.reset();
        }

        Tensor* preview_selection_ptr = request.preview_selection_tensor;
        if (!tensorMatchesGaussianCount(preview_selection_ptr, gaussian_count)) {
            LOG_WARN("Ignoring preview_selection_tensor with stale size: model has {}, tensor has {}",
                     gaussian_count, preview_selection_ptr->numel());
            preview_selection_ptr = nullptr;
        }

        const Tensor* deleted_mask_ptr = request.deleted_mask;
        if (!tensorMatchesGaussianCount(deleted_mask_ptr, gaussian_count)) {
            LOG_WARN("Ignoring deleted_mask with stale size: model has {}, tensor has {}",
                     gaussian_count, deleted_mask_ptr->numel());
            deleted_mask_ptr = nullptr;
        }

        try {
            const int effective_sh_degree = std::clamp(request.sh_degree, 0, model.get_max_sh_degree());
            if (effective_sh_degree != request.sh_degree) {
                LOG_TRACE("Clamped requested SH degree {} to model max {}", request.sh_degree, effective_sh_degree);
            }
            ImageRenderResult result;

            if (request.gut || request.equirectangular) {
                const auto camera_model = request.equirectangular
                                              ? GutCameraModel::EQUIRECTANGULAR
                                              : GutCameraModel::PINHOLE;
                auto render_output = gut_rasterize_tensor(
                    cam, model, background_, effective_sh_degree, request.scaling_modifier, camera_model,
                    model_transforms_tensor.get(), transform_indices_ptr, request.node_visibility_mask);
                return ImageRenderResult{
                    .image = std::move(render_output.image),
                    .depth = std::move(render_output.depth),
                    .valid = true,
                    .far_plane = request.far_plane,
                    .orthographic = request.orthographic};
            }

            // Use libtorch-free tensor-based rasterizer
            auto [image, depth] = rasterize_tensor(cam, model, background_,
                                                   effective_sh_degree,
                                                   request.show_rings, request.ring_width,
                                                   model_transforms_tensor.get(), transform_indices_ptr,
                                                   selection_mask_ptr,
                                                   screen_positions_out,
                                                   request.cursor_active, request.cursor_x, request.cursor_y, request.cursor_radius,
                                                   request.preview_selection_add_mode, preview_selection_ptr,
                                                   request.cursor_saturation_preview, request.cursor_saturation_amount,
                                                   request.show_center_markers,
                                                   request.crop_box_transform, request.crop_box_min, request.crop_box_max,
                                                   request.crop_inverse, request.crop_desaturate, request.crop_parent_node_index,
                                                   request.ellipsoid_transform, request.ellipsoid_radii,
                                                   request.ellipsoid_inverse, request.ellipsoid_desaturate, request.ellipsoid_parent_node_index,
                                                   request.view_volume_transform, request.view_volume_min, request.view_volume_max,
                                                   deleted_mask_ptr,
                                                   request.hovered_depth_id,
                                                   request.focused_gaussian_id,
                                                   request.far_plane,
                                                   request.emphasized_node_mask,
                                                   request.dim_non_emphasized,
                                                   request.node_visibility_mask,
                                                   request.emphasis_flash_intensity,
                                                   request.orthographic,
                                                   request.ortho_scale,
                                                   request.mip_filter);
            result.image = std::move(image);
            result.depth = std::move(depth);
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

    Result<RenderingPipeline::ImageRenderResult> RenderingPipeline::renderPointCloudImageResult(
        const lfs::core::SplatData& model,
        const RasterRequest& request) {

        LOG_TIMER_TRACE("RenderingPipeline::renderPointCloud");

        if (auto init_result = ensurePointCloudRendererInitialized(); !init_result) {
            return std::unexpected(init_result.error());
        }

        GLFramebufferGuard framebuffer_guard;
        GLViewportGuard viewport_guard;
        GLScissorEnableGuard scissor_guard;
        glDisable(GL_SCISSOR_TEST);

        if (auto target_result = preparePointCloudRenderTarget(request); !target_result) {
            return std::unexpected(target_result.error());
        }

        const glm::mat4 view = buildPointCloudViewMatrix(request);
        const glm::mat4 projection = buildPointCloudProjectionMatrix(request);

        {
            LOG_TIMER_TRACE("point_cloud_renderer_->render(SplatData)");
            if (auto result = point_cloud_renderer_->render(model, view, projection,
                                                            request.voxel_size, request.background_color,
                                                            request.model_transforms, request.transform_indices,
                                                            request.equirectangular, request.point_cloud_crop_params);
                !result) {
                LOG_ERROR("Point cloud rendering failed: {}", result.error());
                return std::unexpected(std::format("Point cloud rendering failed: {}", result.error()));
            }
        }

        auto image_result = readPersistentPointCloudImage(request);
        if (!image_result) {
            return std::unexpected(image_result.error());
        }

        LOG_TRACE("Point cloud image rendering completed");
        return *image_result;
    }

    Result<GpuFrame> RenderingPipeline::renderPointCloudGpuFrame(
        const lfs::core::SplatData& model,
        const RasterRequest& request) {

        LOG_TIMER_TRACE("RenderingPipeline::renderPointCloudGpuFrame");

        if (auto init_result = ensurePointCloudRendererInitialized(); !init_result) {
            return std::unexpected(init_result.error());
        }

        GLFramebufferGuard framebuffer_guard;
        GLViewportGuard viewport_guard;
        GLScissorEnableGuard scissor_guard;
        glDisable(GL_SCISSOR_TEST);

        if (auto target_result = preparePointCloudRenderTarget(request); !target_result) {
            return std::unexpected(target_result.error());
        }

        const glm::mat4 view = buildPointCloudViewMatrix(request);
        const glm::mat4 projection = buildPointCloudProjectionMatrix(request);

        {
            LOG_TIMER_TRACE("point_cloud_renderer_->render(SplatData)");
            if (auto result = point_cloud_renderer_->render(model, view, projection,
                                                            request.voxel_size, request.background_color,
                                                            request.model_transforms, request.transform_indices,
                                                            request.equirectangular, request.point_cloud_crop_params);
                !result) {
                LOG_ERROR("Point cloud GPU-frame render failed: {}", result.error());
                return std::unexpected(std::format("Point cloud GPU-frame render failed: {}", result.error()));
            }
        }

        return buildPersistentGpuFrame(
            persistent_render_target_->colorTexture(),
            persistent_render_target_->depthTexture(),
            {persistent_fbo_width_, persistent_fbo_height_},
            request.viewport_size,
            request.far_plane,
            request.orthographic);
    }

    Result<GpuFrame> RenderingPipeline::renderRawPointCloudGpuFrame(
        const lfs::core::PointCloud& point_cloud,
        const RasterRequest& request) {

        LOG_TIMER_TRACE("RenderingPipeline::renderRawPointCloudGpuFrame");

        if (auto init_result = ensurePointCloudRendererInitialized(); !init_result) {
            return std::unexpected(init_result.error());
        }

        GLFramebufferGuard framebuffer_guard;
        GLViewportGuard viewport_guard;
        GLScissorEnableGuard scissor_guard;
        glDisable(GL_SCISSOR_TEST);

        if (auto target_result = preparePointCloudRenderTarget(request); !target_result) {
            return std::unexpected(target_result.error());
        }

        const glm::mat4 view = buildPointCloudViewMatrix(request, true);
        const glm::mat4 projection = buildPointCloudProjectionMatrix(request);

        {
            LOG_TIMER_TRACE("point_cloud_renderer_->render(PointCloud)");
            if (auto result = point_cloud_renderer_->render(point_cloud, view, projection,
                                                            request.voxel_size, request.background_color,
                                                            {}, nullptr, request.equirectangular, request.point_cloud_crop_params);
                !result) {
                LOG_ERROR("Raw point cloud rendering failed: {}", result.error());
                return std::unexpected(std::format("Raw point cloud rendering failed: {}", result.error()));
            }
        }

        return buildPersistentGpuFrame(
            persistent_render_target_->colorTexture(),
            persistent_render_target_->depthTexture(),
            {persistent_fbo_width_, persistent_fbo_height_},
            request.viewport_size,
            request.far_plane,
            request.orthographic);
    }

    Result<void> RenderingPipeline::ensurePointCloudRendererInitialized() {
        if (point_cloud_renderer_->isInitialized()) {
            return {};
        }

        LOG_DEBUG("Initializing point cloud renderer");
        if (auto result = point_cloud_renderer_->initialize(); !result) {
            LOG_ERROR("Failed to initialize point cloud renderer: {}", result.error());
            return std::unexpected(std::format("Failed to initialize point cloud renderer: {}", result.error()));
        }

        return {};
    }

    Result<void> RenderingPipeline::preparePointCloudRenderTarget(const RasterRequest& request) {
        ensureFBOSize(request.viewport_size.x, request.viewport_size.y);
        if (!persistent_render_target_) {
            LOG_ERROR("Failed to setup persistent framebuffer");
            return std::unexpected("Failed to setup persistent framebuffer");
        }

        glViewport(0, 0, request.viewport_size.x, request.viewport_size.y);
        return {};
    }

    Result<RenderingPipeline::ImageRenderResult> RenderingPipeline::readPersistentPointCloudImage(
        const RasterRequest& request) {
        const int width = request.viewport_size.x;
        const int height = request.viewport_size.y;
        ImageRenderResult result;

#ifdef CUDA_GL_INTEROP_ENABLED
        if (use_fbo_interop_) {
            LOG_TIMER_TRACE("CUDA-GL FBO interop readback");

            const bool fbo_changed = fbo_interop_last_width_ != persistent_fbo_width_ ||
                                     fbo_interop_last_height_ != persistent_fbo_height_;
            const bool dims_mismatch = fbo_interop_texture_ &&
                                       (fbo_interop_texture_->getWidth() != persistent_fbo_width_ ||
                                        fbo_interop_texture_->getHeight() != persistent_fbo_height_);
            const bool should_init = persistent_render_target_ &&
                                     (!fbo_interop_texture_ || fbo_changed || dims_mismatch);

            if (should_init) {
                fbo_interop_texture_.reset();
                fbo_interop_texture_.emplace();
                if (auto init_result = fbo_interop_texture_->initForReading(
                        persistent_render_target_->colorTexture(), persistent_fbo_width_, persistent_fbo_height_);
                    !init_result) {
                    LOG_TRACE("FBO interop init failed: {}", init_result.error());
                    fbo_interop_texture_.reset();
                }
                fbo_interop_last_width_ = persistent_fbo_width_;
                fbo_interop_last_height_ = persistent_fbo_height_;
            }

            if (use_fbo_interop_ && fbo_interop_texture_) {
                Tensor image_hwc;
                if (auto read_result = fbo_interop_texture_->readToTensor(image_hwc, width, height); read_result) {
                    result.image = image_hwc.permute({2, 0, 1}).contiguous();
                    result.valid = true;
                    result.external_depth_texture = persistent_render_target_->depthTexture();
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

            const int current_pbo = pbo_index_;
            const int next_pbo = 1 - pbo_index_;

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
            result.external_depth_texture = persistent_render_target_->depthTexture();
            result.depth_is_ndc = true;
            result.valid = true;
        }

        result.orthographic = request.orthographic;
        result.far_plane = request.far_plane;
        return result;
    }

    void RenderingPipeline::applyDepthParams(
        const ImageRenderResult& result,
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
        const ImageRenderResult& result,
        ScreenQuadRenderer& renderer,
        const glm::ivec2& viewport_size) {
        LOG_TIMER_TRACE("RenderingPipeline::uploadToScreen");

        if (!result.valid || !result.image.is_valid()) {
            LOG_ERROR("Invalid render result for upload");
            return std::unexpected("Invalid render result");
        }

        if (renderer.isInteropEnabled() && result.image.device() == lfs::core::Device::CUDA) {
            if (result.image.ndim() != 3) {
                LOG_WARN("Unsupported CUDA image rank for interop upload: {}", result.image.ndim());
            } else {
                const auto layout = detectImageLayout(result.image);
                size_t image_h = 0;
                size_t image_w = 0;
                if (layout != ImageLayout::Unknown) {
                    image_h = imageHeight(result.image, layout);
                    image_w = imageWidth(result.image, layout);
                } else {
                    LOG_WARN("Unsupported CUDA image layout for interop upload: shape=[{}, {}, {}]",
                             result.image.size(0), result.image.size(1), result.image.size(2));
                }

                if (image_h == static_cast<size_t>(viewport_size.y) &&
                    image_w == static_cast<size_t>(viewport_size.x)) {
                    if (auto upload_result = renderer.uploadFromCUDA(result.image, viewport_size.x, viewport_size.y);
                        !upload_result) {
                        return upload_result;
                    }
                    applyDepthParams(result, renderer, viewport_size);
                    return {};
                }

                if (image_h > 0 && image_w > 0) {
                    LOG_WARN("Dimension mismatch: image {}x{}, expected {}x{}",
                             image_w, image_h, viewport_size.x, viewport_size.y);
                }
            }
        }

        // CPU fallback
        if (result.image.ndim() != 3) {
            LOG_ERROR("CPU upload requires rank-3 image tensor, got {}", result.image.ndim());
            return std::unexpected("CPU upload requires rank-3 image tensor");
        }

        const auto cpu_layout = detectImageLayout(result.image);
        if (cpu_layout == ImageLayout::Unknown) {
            LOG_ERROR("CPU upload unsupported image layout: shape=[{}, {}, {}]",
                      result.image.size(0), result.image.size(1), result.image.size(2));
            return std::unexpected("CPU upload unsupported image layout");
        }

        Tensor image_hwc = (cpu_layout == ImageLayout::HWC)
                               ? result.image
                               : result.image.permute({1, 2, 0}).contiguous();
        if (image_hwc.size(2) == 4) {
            image_hwc = image_hwc.slice(2, 0, 3).contiguous();
        }
        if (image_hwc.dtype() != lfs::core::DataType::UInt8) {
            image_hwc = (image_hwc.clamp(0.0f, 1.0f) * 255.0f).to(lfs::core::DataType::UInt8);
        }
        const auto image = image_hwc.cpu().contiguous();

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

    Result<lfs::core::Camera> RenderingPipeline::createCamera(const RasterRequest& request) {
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

        // Compute field of view from focal length (single conversion point)
        const float vfov_rad = focalLengthToVFovRad(request.focal_length_mm);
        glm::vec2 fov = computeFov(vfov_rad,
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

    glm::vec2 RenderingPipeline::computeFov(float vfov_rad, int width, int height) {
        float aspect = static_cast<float>(width) / height;
        return glm::vec2(
            std::atan(std::tan(vfov_rad * 0.5f) * aspect) * 2.0f,
            vfov_rad);
    }

    void RenderingPipeline::ensureFBOSize(int width, int height) {
        const int alloc_width = ((width + GPU_ALIGNMENT - 1) / GPU_ALIGNMENT) * GPU_ALIGNMENT;
        const int alloc_height = ((height + GPU_ALIGNMENT - 1) / GPU_ALIGNMENT) * GPU_ALIGNMENT;

        if (persistent_render_target_ &&
            alloc_width == persistent_fbo_width_ && alloc_height == persistent_fbo_height_) {
            glBindFramebuffer(GL_FRAMEBUFFER, persistent_render_target_->framebuffer());
            return;
        }

        LOG_DEBUG("FBO resize: {}x{} -> {}x{}", persistent_fbo_width_, persistent_fbo_height_, alloc_width, alloc_height);

        if (render_target_pool_) {
            auto target = render_target_pool_->acquireHighPrecision(
                "rendering_pipeline.point_cloud", {alloc_width, alloc_height});
            if (!target) {
                LOG_ERROR("Failed to acquire high-precision render target: {}", target.error());
                cleanupFBO();
                return;
            }
            persistent_render_target_ = *target;
        } else {
            if (!persistent_render_target_) {
                persistent_render_target_ = std::make_shared<HighPrecisionRenderTarget>();
            }
            if (auto result = persistent_render_target_->ensureSize({alloc_width, alloc_height}); !result) {
                LOG_ERROR("Failed to resize persistent render target: {}", result.error());
                cleanupFBO();
                return;
            }
        }

#ifdef CUDA_GL_INTEROP_ENABLED
        fbo_interop_texture_.reset();
#endif
        glBindFramebuffer(GL_FRAMEBUFFER, persistent_render_target_->framebuffer());

        persistent_fbo_width_ = alloc_width;
        persistent_fbo_height_ = alloc_height;

        LOG_DEBUG("FBO allocated: {}x{}", alloc_width, alloc_height);
    }

    void RenderingPipeline::cleanupFBO() {
        persistent_render_target_.reset();
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

/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include "config.h"
#include "core/camera.hpp"
#include "core/splat_data.hpp"
#include "core/tensor.hpp"
#include "geometry/bounding_box.hpp"
#include "gl_resources.hpp"
#include "point_cloud_renderer.hpp"
#include "render_target_pool.hpp"
#include "rendering/render_constants.hpp"
#include "rendering/rendering.hpp"
#include "screen_renderer.hpp"
#include <glm/glm.hpp>
#include <optional>

#ifdef CUDA_GL_INTEROP_ENABLED
#include "cuda_gl_interop.hpp"
#include <optional>
#endif

namespace lfs::rendering {

    // Import Tensor from lfs::core
    using lfs::core::Tensor;

    class RenderingPipeline {
    public:
        struct RasterRequest {
            glm::mat3 view_rotation;
            glm::vec3 view_translation;
            glm::ivec2 viewport_size;
            float focal_length_mm = DEFAULT_FOCAL_LENGTH_MM;
            float scaling_modifier = 1.0f;
            bool antialiasing = false;
            bool mip_filter = false;
            int sh_degree = 3;
            RenderMode render_mode = RenderMode::RGB;
            const lfs::geometry::BoundingBox* crop_box = nullptr;
            glm::vec3 background_color = glm::vec3(0.0f, 0.0f, 0.0f);
            float voxel_size = 0.01f;
            bool gut = false;
            bool equirectangular = false;
            bool show_rings = false;
            float ring_width = 0.01f;
            bool show_center_markers = false;
            // Per-node transforms: array of 4x4 matrices and per-Gaussian indices
            std::vector<glm::mat4> model_transforms;              // Array of transforms, one per node
            std::shared_ptr<lfs::core::Tensor> transform_indices; // Per-Gaussian index [N], nullable
            // Selection mask for highlighting selected Gaussians
            std::shared_ptr<lfs::core::Tensor> selection_mask;
            bool cursor_active = false;
            float cursor_x = 0.0f;
            float cursor_y = 0.0f;
            float cursor_radius = 0.0f;
            bool preview_selection_add_mode = true;
            lfs::core::Tensor* preview_selection_tensor = nullptr;
            bool cursor_saturation_preview = false;
            float cursor_saturation_amount = 0.0f;
            // Crop box filtering (scoped to parent node if >= 0)
            const Tensor* crop_box_transform = nullptr;
            const Tensor* crop_box_min = nullptr;
            const Tensor* crop_box_max = nullptr;
            bool crop_inverse = false;
            bool crop_desaturate = false;
            int crop_parent_node_index = -1;
            // Ellipsoid filtering (scoped to parent node if >= 0)
            const Tensor* ellipsoid_transform = nullptr;
            const Tensor* ellipsoid_radii = nullptr;
            bool ellipsoid_inverse = false;
            bool ellipsoid_desaturate = false;
            int ellipsoid_parent_node_index = -1;
            // View-volume filter used by the selection depth box.
            const Tensor* view_volume_transform = nullptr;
            const Tensor* view_volume_min = nullptr;
            const Tensor* view_volume_max = nullptr;
            bool view_volume_cull = false;
            const Tensor* deleted_mask = nullptr; // Soft deletion mask [N], true = skip
            // Hover query output
            unsigned long long* hovered_depth_id = nullptr;
            int focused_gaussian_id = -1;
            float far_plane = DEFAULT_FAR_PLANE;
            std::vector<bool> emphasized_node_mask;
            std::vector<bool> node_visibility_mask; // Per-node visibility for culling (consolidated models)
            bool dim_non_emphasized = false;
            float emphasis_flash_intensity = 0.0f;
            bool orthographic = false;
            float ortho_scale = DEFAULT_ORTHO_SCALE;
            PointCloudCropParams point_cloud_crop_params;

            [[nodiscard]] glm::mat4 getProjectionMatrix(const float near_plane = DEFAULT_NEAR_PLANE,
                                                        const float far_plane = DEFAULT_FAR_PLANE) const {
                const float vfov = focalLengthToVFov(focal_length_mm);
                return createProjectionMatrix(viewport_size, vfov, orthographic, ortho_scale, near_plane, far_plane);
            }
        };

        struct ImageRenderResult {
            Tensor image;
            Tensor depth;
            bool valid = false;
            bool depth_is_ndc = false;         // True if depth is already NDC (0-1), e.g., from OpenGL
            GLuint external_depth_texture = 0; // If set, use this OpenGL texture directly (zero-copy)
            // Depth conversion parameters (needed for view-space to NDC conversion)
            float near_plane = DEFAULT_NEAR_PLANE;
            float far_plane = DEFAULT_FAR_PLANE;
            bool orthographic = false;
        };

        struct DualImageRenderResult {
            std::array<ImageRenderResult, 2> views;
        };

        RenderingPipeline();
        ~RenderingPipeline();

        Result<ImageRenderResult> renderGaussianImage(const lfs::core::SplatData& model, const RasterRequest& request);
        Result<DualImageRenderResult> renderGaussianImagePair(const lfs::core::SplatData& model,
                                                              const std::array<RasterRequest, 2>& requests);
        Result<ImageRenderResult> renderPointCloudImage(const lfs::core::SplatData& model, const RasterRequest& request);
        Result<Tensor> renderScreenPositions(const lfs::core::SplatData& model, const RasterRequest& request);
        void setRenderTargetPool(RenderTargetPool* pool) { render_target_pool_ = pool; }
        void resetResources();

        // Static upload function for image-backed raster results
        static Result<void> uploadToScreen(const ImageRenderResult& result,
                                           ScreenQuadRenderer& renderer,
                                           const glm::ivec2& viewport_size);

        Result<GpuFrame> renderPointCloudGpuFrame(const lfs::core::SplatData& model, const RasterRequest& request);
        Result<GpuFrame> renderRawPointCloudGpuFrame(const lfs::core::PointCloud& point_cloud, const RasterRequest& request);

    private:
        // Apply depth params from image-backed raster results to ScreenQuadRenderer
        static void applyDepthParams(const ImageRenderResult& result,
                                     ScreenQuadRenderer& renderer,
                                     const glm::ivec2& viewport_size);
        Result<lfs::core::Camera> createCamera(const RasterRequest& request);
        Result<ImageRenderResult> renderGaussianImageResult(const lfs::core::SplatData& model,
                                                            const RasterRequest& request,
                                                            Tensor* screen_positions_out);
        glm::vec2 computeFov(float vfov_rad, int width, int height);
        Result<ImageRenderResult> renderPointCloudImageResult(const lfs::core::SplatData& model,
                                                              const RasterRequest& request);
        Result<void> ensurePointCloudRendererInitialized();
        Result<void> preparePointCloudRenderTarget(const RasterRequest& request);
        Result<ImageRenderResult> readPersistentPointCloudImage(const RasterRequest& request);

        // Ensure persistent FBO is sized correctly (avoids recreation every frame)
        void ensureFBOSize(int width, int height);
        void cleanupFBO();

        // Ensure PBOs are sized correctly (avoids recreation every frame)
        void ensurePBOSize(int width, int height);
        void cleanupPBO();

        Tensor background_;
        std::unique_ptr<PointCloudRenderer> point_cloud_renderer_;

        // Persistent framebuffer objects (reused across frames)
        // Avoids expensive glGenFramebuffers/glDeleteFramebuffers every render
        std::shared_ptr<HighPrecisionRenderTarget> persistent_render_target_;
        int persistent_fbo_width_ = 0;
        int persistent_fbo_height_ = 0;

        // Pixel Buffer Objects for async GPU→CPU readback
        // Uses double-buffering to overlap memory transfer with rendering
        GLuint pbo_[2] = {0, 0};
        int pbo_index_ = 0;
        int pbo_width_ = 0;
        int pbo_height_ = 0;
        int allocated_pbo_width_ = 0;
        int allocated_pbo_height_ = 0;

#ifdef CUDA_GL_INTEROP_ENABLED
        // CUDA-GL interop for direct FBO→CUDA texture readback (eliminates CPU round-trip)
        std::optional<CudaGLInteropTexture> fbo_interop_texture_;
        bool use_fbo_interop_ = true;
        int fbo_interop_last_width_ = 0;  // Track FBO size when interop was initialized
        int fbo_interop_last_height_ = 0; // to detect when we need to reinitialize
#endif
        RenderTargetPool* render_target_pool_ = nullptr;
    };

} // namespace lfs::rendering

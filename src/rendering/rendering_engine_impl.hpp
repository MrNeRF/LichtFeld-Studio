/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include "axes_renderer.hpp"
#include "bbox_renderer.hpp"
#include "camera_frustum_renderer.hpp"
#include "cuda_gl_interop.hpp"
#include "depth_compositor.hpp"
#include "ellipsoid_renderer.hpp"
#include "grid_renderer.hpp"
#include "mesh_renderer.hpp"
#include "pivot_renderer.hpp"
#include "render_target_pool.hpp"
#include "rendering/rendering.hpp"
#include "rendering_pipeline.hpp"
#include "screen_renderer.hpp"
#include "shader_manager.hpp"
#include "split_view_renderer.hpp"
#include "viewport_gizmo.hpp"

namespace lfs::rendering {

    class RenderingEngineImpl : public RenderingEngine {
    public:
        RenderingEngineImpl();
        ~RenderingEngineImpl() override;

        Result<void> initialize() override;
        void shutdown() override;
        bool isInitialized() const override;

        Result<GaussianGpuFrameResult> renderGaussiansGpuFrame(
            const lfs::core::SplatData& splat_data,
            const ViewportRenderRequest& request) override;

        Result<GaussianImageResult> renderGaussiansImage(
            const lfs::core::SplatData& splat_data,
            const ViewportRenderRequest& request) override;

        Result<DualGaussianImageResult> renderGaussiansImagePair(
            const lfs::core::SplatData& splat_data,
            const std::array<ViewportRenderRequest, 2>& requests) override;

        Result<std::optional<int>> queryHoveredGaussianId(
            const lfs::core::SplatData& splat_data,
            const HoveredGaussianQueryRequest& request) override;

        Result<std::shared_ptr<lfs::core::Tensor>> renderGaussianScreenPositions(
            const lfs::core::SplatData& splat_data,
            const ScreenPositionRenderRequest& request) override;

        Result<GpuFrame> renderPointCloudGpuFrame(
            const lfs::core::SplatData& splat_data,
            const PointCloudRenderRequest& request) override;

        Result<PointCloudImageResult> renderPointCloudImage(
            const lfs::core::SplatData& splat_data,
            const PointCloudRenderRequest& request) override;

        Result<GpuFrame> renderPointCloudGpuFrame(
            const lfs::core::PointCloud& point_cloud,
            const PointCloudRenderRequest& request) override;

        Result<SplitViewFrameResult> renderSplitViewGpuFrame(
            const SplitViewRequest& request) override;

        Result<void> renderMesh(
            const lfs::core::MeshData& mesh,
            const ViewportData& viewport,
            const glm::mat4& model_transform = glm::mat4(1.0f),
            const MeshRenderOptions& options = {},
            bool use_fbo = false) override;

        unsigned int getMeshColorTexture() const override;
        unsigned int getMeshDepthTexture() const override;
        unsigned int getMeshFramebuffer() const override;
        bool hasMeshRender() const override;
        void resetMeshFrameState() override { mesh_rendered_this_frame_ = false; }

        Result<GpuFrame> materializeGpuFrame(
            const std::shared_ptr<lfs::core::Tensor>& image,
            const FrameMetadata& metadata,
            const glm::ivec2& viewport_size) override;

        Result<std::shared_ptr<lfs::core::Tensor>> readbackGpuFrameColor(
            const GpuFrame& frame) override;

        Result<void> compositeMeshAndGpuFrame(
            const GpuFrame& splat_frame,
            const glm::ivec2& viewport_size) override;

        Result<void> presentMeshOnly() override;

        Result<void> presentGpuFrame(
            const GpuFrame& frame,
            const glm::ivec2& viewport_pos,
            const glm::ivec2& viewport_size) override;

        Result<void> renderGrid(
            const ViewportData& viewport,
            GridPlane plane,
            float opacity) override;

        Result<void> renderBoundingBox(
            const BoundingBox& box,
            const ViewportData& viewport,
            const glm::vec3& color,
            float line_width) override;

        Result<void> renderEllipsoid(
            const Ellipsoid& ellipsoid,
            const ViewportData& viewport,
            const glm::vec3& color,
            float line_width) override;

        Result<void> renderCoordinateAxes(
            const ViewportData& viewport,
            float size,
            const std::array<bool, 3>& visible,
            bool equirectangular = false) override;

        Result<void> renderPivot(
            const ViewportData& viewport,
            const glm::vec3& pivot_position,
            float size = 50.0f,
            float opacity = 1.0f) override;

        Result<void> renderViewportGizmo(
            const glm::mat3& camera_rotation,
            const glm::vec2& viewport_pos,
            const glm::vec2& viewport_size) override;

        int hitTestViewportGizmo(
            const glm::vec2& click_pos,
            const glm::vec2& viewport_pos,
            const glm::vec2& viewport_size) const override;

        void setViewportGizmoHover(int axis) override;

        Result<void> renderCameraFrustums(
            const std::vector<std::shared_ptr<const lfs::core::Camera>>& cameras,
            const CameraFrustumRenderRequest& request) override;

        Result<int> pickCameraFrustum(
            const std::vector<std::shared_ptr<const lfs::core::Camera>>& cameras,
            const CameraFrustumPickRequest& request) override;

        void clearFrustumCache() override;
        void setFrustumImageLoader(std::shared_ptr<lfs::io::PipelinedImageLoader> loader) override;

    private:
        Result<RenderingPipeline::ImageRenderResult> renderGaussiansRasterResult(
            const lfs::core::SplatData& splat_data,
            const ViewportRenderRequest& request);
        Result<RenderingPipeline::DualImageRenderResult> renderGaussiansRasterResultPair(
            const lfs::core::SplatData& splat_data,
            const std::array<ViewportRenderRequest, 2>& requests);
        [[nodiscard]] static FrameMetadata makeFrameMetadata(const RenderingPipeline::ImageRenderResult& result);
        Result<GpuFrame> uploadRenderResultToGpuFrame(
            const RenderingPipeline::ImageRenderResult& result,
            const glm::ivec2& viewport_size);
        void invalidatePresentUploadCache();
        [[nodiscard]] bool ensureHoveredDepthQueryBuffersAllocated();
        Result<void> initializeShaders();
        Result<void> ensureRenderResultUploaded(
            const std::shared_ptr<const Tensor>& image,
            const FrameMetadata& metadata,
            const glm::ivec2& viewport_size);
        glm::mat4 createProjectionMatrix(const ViewportData& viewport) const;
        glm::mat4 createViewMatrix(const ViewportData& viewport) const;

        RenderingPipeline pipeline_;
        RenderTargetPool render_target_pool_;
        std::shared_ptr<ScreenQuadRenderer> screen_renderer_;
        std::unique_ptr<SplitViewRenderer> split_view_renderer_;

        RenderInfiniteGrid grid_renderer_;
        RenderBoundingBox bbox_renderer_;
        EllipsoidRenderer ellipsoid_renderer_;
        RenderCoordinateAxes axes_renderer_;
        ViewportGizmo viewport_gizmo_;
        CameraFrustumRenderer camera_frustum_renderer_;
        RenderPivotPoint pivot_renderer_;

        MeshRenderer mesh_renderer_;
        DepthCompositor depth_compositor_;
        bool mesh_rendered_this_frame_ = false;

        ManagedShader quad_shader_;

        // Cache the last uploaded frame payload to avoid redundant CUDA->GL uploads
        // when presenting the exact same render result repeatedly (idle cached frames).
        std::shared_ptr<const Tensor> last_presented_image_;
        std::shared_ptr<const Tensor> last_presented_depth_;
        unsigned int last_presented_external_depth_texture_ = 0;
        bool last_presented_depth_is_ndc_ = false;
        float last_presented_near_plane_ = 0.0f;
        float last_presented_far_plane_ = 0.0f;
        bool last_presented_orthographic_ = false;
        bool has_present_upload_cache_ = false;
        unsigned long long* hovered_depth_id_device_ = nullptr;
        unsigned long long* hovered_depth_id_host_ = nullptr;

#ifdef CUDA_GL_INTEROP_ENABLED
        std::unique_ptr<CudaGLInteropTexture> gpu_frame_readback_interop_;
        unsigned int gpu_frame_readback_source_ = 0;
        glm::ivec2 gpu_frame_readback_size_{0, 0};
#endif
        FBO gpu_frame_readback_fbo_;
    };

} // namespace lfs::rendering

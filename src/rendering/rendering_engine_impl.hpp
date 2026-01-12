/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include "axes_renderer.hpp"
#include "bbox_renderer.hpp"
#include "camera_frustum_renderer.hpp"
#include "grid_renderer.hpp"
#include "pivot_renderer.hpp"
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

        Result<RenderResult> renderGaussians(
            const lfs::core::SplatData& splat_data,
            const RenderRequest& request) override;

        Result<RenderResult> renderPointCloud(
            const lfs::core::PointCloud& point_cloud,
            const RenderRequest& request) override;

        Result<RenderResult> renderSplitView(
            const SplitViewRequest& request) override;

        Result<void> presentToScreen(
            const RenderResult& result,
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

        Result<void> renderTranslationGizmo(
            const glm::vec3& position,
            const ViewportData& viewport,
            float scale) override;

        std::shared_ptr<GizmoInteraction> getGizmoInteraction() override;

        Result<void> renderCameraFrustums(
            const std::vector<std::shared_ptr<const lfs::core::Camera>>& cameras,
            const ViewportData& viewport,
            float scale,
            const glm::vec3& train_color,
            const glm::vec3& eval_color,
            const glm::mat4& scene_transform = glm::mat4(1.0f),
            bool equirectangular_view = false) override;

        Result<void> renderCameraFrustumsWithHighlight(
            const std::vector<std::shared_ptr<const lfs::core::Camera>>& cameras,
            const ViewportData& viewport,
            float scale,
            const glm::vec3& train_color,
            const glm::vec3& eval_color,
            int highlight_index,
            const glm::mat4& scene_transform = glm::mat4(1.0f),
            bool equirectangular_view = false) override;

        Result<int> pickCameraFrustum(
            const std::vector<std::shared_ptr<const lfs::core::Camera>>& cameras,
            const glm::vec2& mouse_pos,
            const glm::vec2& viewport_pos,
            const glm::vec2& viewport_size,
            const ViewportData& viewport,
            float scale,
            const glm::mat4& scene_transform = glm::mat4(1.0f)) override;

        void clearFrustumCache() override;

        // Pipeline compatibility
        RenderingPipelineResult renderWithPipeline(
            const lfs::core::SplatData& model,
            const RenderingPipelineRequest& request) override;

        // Factory methods
        Result<std::shared_ptr<IBoundingBox>> createBoundingBox() override;
        Result<std::shared_ptr<ICoordinateAxes>> createCoordinateAxes() override;

    private:
        Result<void> initializeShaders();
        glm::mat4 createProjectionMatrix(const ViewportData& viewport) const;
        glm::mat4 createViewMatrix(const ViewportData& viewport) const;

        // Core components
        RenderingPipeline pipeline_;
        std::shared_ptr<ScreenQuadRenderer> screen_renderer_;

        // Split view renderer
        std::unique_ptr<SplitViewRenderer> split_view_renderer_;

        // Overlay renderers
        RenderInfiniteGrid grid_renderer_;
        RenderBoundingBox bbox_renderer_;
        RenderCoordinateAxes axes_renderer_;
        ViewportGizmo viewport_gizmo_;
        CameraFrustumRenderer camera_frustum_renderer_;
        RenderPivotPoint pivot_renderer_;

        // Shaders
        ManagedShader quad_shader_;
    };

} // namespace lfs::rendering
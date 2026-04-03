/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include "core/tensor.hpp"
#include "frame_contract.hpp"
#include "geometry/euclidean_transform.hpp"
#include "render_constants.hpp"
#include <array>
#include <expected>
#include <glm/glm.hpp>
#include <memory>
#include <optional>
#include <string>
#include <unordered_set>
#include <vector>

namespace lfs::core {
    class SplatData;
    struct PointCloud;
    struct MeshData;
    class Camera;
    class Tensor;
} // namespace lfs::core

namespace lfs::io {
    class PipelinedImageLoader;
}

namespace lfs::rendering {

    // Import Tensor into this namespace for convenience
    using lfs::core::Tensor;

    // Error handling with std::expected (C++23)
    template <typename T>
    using Result = std::expected<T, std::string>;

    // Public renderer-facing boundary.
    // Keep editor workflow semantics constrained to the explicit renderer
    // request types below and prefer frame_contract.hpp for new abstractions.

    // Public types
    struct ViewportData {
        glm::mat3 rotation;
        glm::vec3 translation;
        glm::ivec2 size;
        float focal_length_mm = DEFAULT_FOCAL_LENGTH_MM;
        bool orthographic = false;
        float ortho_scale = DEFAULT_ORTHO_SCALE;

        [[nodiscard]] glm::mat4 getViewMatrix() const {
            return makeViewMatrix(rotation, translation);
        }

        [[nodiscard]] glm::mat4 getProjectionMatrix(const float near_plane = DEFAULT_NEAR_PLANE,
                                                    const float far_plane = DEFAULT_FAR_PLANE) const {
            const float vfov = focalLengthToVFov(focal_length_mm);
            return createProjectionMatrix(size, vfov, orthographic, ortho_scale, near_plane, far_plane);
        }
    };

    struct BoundingBox {
        glm::vec3 min;
        glm::vec3 max;
        glm::mat4 transform{1.0f};
    };

    struct Ellipsoid {
        glm::vec3 radii{1.0f, 1.0f, 1.0f};
        glm::mat4 transform{1.0f};
    };

    struct GaussianSceneState {
        const std::vector<glm::mat4>* model_transforms = nullptr;
        std::shared_ptr<lfs::core::Tensor> transform_indices;
        std::vector<bool> node_visibility_mask;
    };

    struct GaussianScopedBoxFilter {
        BoundingBox bounds;
        bool inverse = false;
        bool desaturate = false;
        int parent_node_index = -1;
    };

    struct GaussianScopedEllipsoidFilter {
        Ellipsoid bounds;
        bool inverse = false;
        bool desaturate = false;
        int parent_node_index = -1;
    };

    struct GaussianFilterState {
        std::optional<GaussianScopedBoxFilter> crop_region;
        std::optional<GaussianScopedEllipsoidFilter> ellipsoid_region;
        std::optional<BoundingBox> view_volume;
        bool cull_outside_view_volume = false;
    };

    struct GaussianMarkerOverlayState {
        bool show_rings = false;
        float ring_width = 0.002f;
        bool show_center_markers = false;
    };

    struct GaussianTransientMaskOverlayState {
        lfs::core::Tensor* mask = nullptr;
        bool additive = true;
    };

    struct GaussianCursorOverlayState {
        bool enabled = false;
        glm::vec2 cursor{0.0f, 0.0f};
        float radius = 0.0f;
        bool saturation_preview = false;
        float saturation_amount = 0.0f;
    };

    struct GaussianEmphasisOverlayState {
        std::shared_ptr<lfs::core::Tensor> mask;
        GaussianTransientMaskOverlayState transient_mask;
        std::vector<bool> emphasized_node_mask;
        bool dim_non_emphasized = false;
        float flash_intensity = 0.0f;
        int focused_gaussian_id = -1;
    };

    struct GaussianOverlayState {
        GaussianMarkerOverlayState markers;
        GaussianCursorOverlayState cursor;
        GaussianEmphasisOverlayState emphasis;
    };

    struct ViewportRenderRequest {
        FrameView frame_view;
        float scaling_modifier = 1.0f;
        bool antialiasing = false;
        bool mip_filter = false;
        int sh_degree = 3;
        bool gut = false;
        bool equirectangular = false;
        GaussianSceneState scene;
        GaussianFilterState filters;
        GaussianOverlayState overlay;
    };

    struct HoveredGaussianQueryRequest {
        FrameView frame_view;
        float scaling_modifier = 1.0f;
        bool mip_filter = false;
        int sh_degree = 3;
        bool gut = false;
        bool equirectangular = false;
        GaussianSceneState scene;
        GaussianFilterState filters;
        glm::vec2 cursor{0.0f, 0.0f};
    };

    struct ScreenPositionRenderRequest {
        FrameView frame_view;
        bool equirectangular = false;
        GaussianSceneState scene;
    };

    struct PointCloudSceneState {
        const std::vector<glm::mat4>* model_transforms = nullptr;
        std::shared_ptr<lfs::core::Tensor> transform_indices;
    };

    struct PointCloudFilterState {
        std::optional<BoundingBox> crop_box;
        bool crop_inverse = false;
        bool crop_desaturate = false;
    };

    struct PointCloudRenderState {
        float scaling_modifier = 1.0f;
        float voxel_size = 0.01f;
        bool equirectangular = false;
    };

    struct PointCloudRenderRequest {
        FrameView frame_view;
        PointCloudRenderState render;
        PointCloudSceneState scene;
        PointCloudFilterState filters;
    };

    struct FramePanelMetadata {
        std::shared_ptr<lfs::core::Tensor> depth;
        float start_position = 0.0f;
        float end_position = 1.0f;

        [[nodiscard]] bool valid() const {
            return end_position > start_position;
        }
    };

    struct FrameMetadata {
        std::array<FramePanelMetadata, 2> depth_panels{};
        size_t depth_panel_count = 0;
        bool valid = false;
        // Depth conversion parameters (needed for proper depth buffer writing)
        bool depth_is_ndc = false;               // True if depth is already NDC (0-1), e.g., from OpenGL
        unsigned int external_depth_texture = 0; // If set, use this OpenGL texture directly (zero-copy)
        glm::vec2 depth_texcoord_scale{1.0f, 1.0f};
        float near_plane = DEFAULT_NEAR_PLANE;
        float far_plane = DEFAULT_FAR_PLANE;
        bool orthographic = false;

        [[nodiscard]] const std::shared_ptr<lfs::core::Tensor>& primaryDepth() const {
            return depth_panels[0].depth;
        }
    };

    struct GaussianGpuFrameResult {
        GpuFrame frame;
        FrameMetadata metadata;
    };

    struct GaussianImageResult {
        std::shared_ptr<lfs::core::Tensor> image;
        FrameMetadata metadata;
    };

    using DualGaussianImageResult = std::array<GaussianImageResult, 2>;

    struct PointCloudImageResult {
        std::shared_ptr<lfs::core::Tensor> image;
        FrameMetadata metadata;
    };

    struct SplitViewFrameResult {
        GpuFrame frame;
        FrameMetadata metadata;
    };

    // Split view support
    enum class PanelContentType {
        Model3D,     // Regular 3D model rendering
        Image2D,     // GT image display
        CachedRender // Previously rendered frame
    };

    struct SplitViewGaussianPanelRenderState {
        FrameView frame_view;
        float scaling_modifier = 1.0f;
        bool antialiasing = false;
        bool mip_filter = false;
        int sh_degree = 3;
        bool gut = false;
        bool equirectangular = false;
        GaussianSceneState scene;
        GaussianFilterState filters;
        GaussianOverlayState overlay;
    };

    struct SplitViewPointCloudPanelRenderState {
        FrameView frame_view;
        PointCloudRenderState render;
        PointCloudSceneState scene;
        PointCloudFilterState filters;
    };

    struct SplitViewPanelContent {
        PanelContentType type = PanelContentType::Model3D;
        const lfs::core::SplatData* model = nullptr;
        glm::mat4 model_transform{1.0f};
        std::optional<SplitViewGaussianPanelRenderState> gaussian_render;
        std::optional<SplitViewPointCloudPanelRenderState> point_cloud_render;
        unsigned int texture_id = 0;
    };

    struct SplitViewPanelPresentation {
        float start_position = 0.0f;
        float end_position = 1.0f;
        glm::vec2 texcoord_scale{1.0f, 1.0f};
        std::optional<bool> flip_y;
        bool normalize_x_to_panel = false;
    };

    struct SplitViewPanel {
        SplitViewPanelContent content;
        SplitViewPanelPresentation presentation;
    };

    struct SplitViewCompositeState {
        glm::ivec2 output_size{0, 0};
        glm::vec3 background_color{0.0f, 0.0f, 0.0f};
    };

    struct SplitViewPresentationState {
        glm::vec4 divider_color{0.29f, 0.33f, 0.42f, 1.0f};
        bool letterbox = false;
        glm::ivec2 content_size{0, 0};
    };

    struct SplitViewRequest {
        std::array<SplitViewPanel, 2> panels;
        SplitViewCompositeState composite;
        SplitViewPresentationState presentation;
        bool prefer_batched_gaussian_render = false;
    };

    enum class GridPlane {
        YZ = 0, // X plane
        XZ = 1, // Y plane
        XY = 2  // Z plane
    };

    // Render modes
    enum class RenderMode {
        RGB = 0,
        D = 1,
        ED = 2,
        RGB_D = 3,
        RGB_ED = 4
    };

    // Interface for bounding box manipulation (for visualizer)
    class IBoundingBox {
    public:
        virtual ~IBoundingBox() = default;

        virtual void setBounds(const glm::vec3& min, const glm::vec3& max) = 0;
        virtual glm::vec3 getMinBounds() const = 0;
        virtual glm::vec3 getMaxBounds() const = 0;
        virtual glm::vec3 getCenter() const = 0;
        virtual glm::vec3 getSize() const = 0;
        virtual glm::vec3 getLocalCenter() const = 0;

        virtual void setColor(const glm::vec3& color) = 0;
        virtual void setLineWidth(float width) = 0;
        virtual bool isInitialized() const = 0;

        virtual void setworld2BBox(const lfs::geometry::EuclideanTransform& transform) = 0;
        virtual lfs::geometry::EuclideanTransform getworld2BBox() const = 0;

        virtual glm::vec3 getColor() const = 0;
        virtual float getLineWidth() const = 0;
    };

    // Interface for coordinate axes (for visualizer)
    class ICoordinateAxes {
    public:
        virtual ~ICoordinateAxes() = default;

        virtual void setSize(float size) = 0;
        virtual void setAxisVisible(int axis, bool visible) = 0;
        virtual bool isAxisVisible(int axis) const = 0;
    };

    struct MeshRenderOptions {
        bool wireframe_overlay = false;
        glm::vec3 wireframe_color{0.2f};
        float wireframe_width = 1.0f;
        glm::vec3 light_dir{0.3f, 1.0f, 0.5f};
        float light_intensity = 0.7f;
        float ambient = 0.4f;
        bool backface_culling = true;
        bool shadow_enabled = false;
        int shadow_map_resolution = 2048;
        bool is_emphasized = false;
        bool dim_non_emphasized = false;
        float flash_intensity = 0.0f;
        glm::vec3 background_color{0.0f};
    };

    struct CameraFrustumRenderRequest {
        ViewportData viewport;
        float scale = 0.1f;
        glm::vec3 train_color{0.0f, 1.0f, 0.0f};
        glm::vec3 eval_color{1.0f, 0.0f, 0.0f};
        std::vector<glm::vec3> per_camera_colors;
        int focused_index = -1;
        glm::mat4 scene_transform{1.0f};
        std::vector<glm::mat4> scene_transforms;
        bool equirectangular_view = false;
        std::unordered_set<int> disabled_uids;
        std::unordered_set<int> emphasized_uids;
    };

    struct CameraFrustumPickRequest {
        glm::vec2 mouse_pos{0.0f, 0.0f};
        glm::vec2 viewport_pos{0.0f, 0.0f};
        glm::vec2 viewport_size{0.0f, 0.0f};
        ViewportData viewport;
        float scale = 0.1f;
        glm::mat4 scene_transform{1.0f};
        std::vector<glm::mat4> scene_transforms;
    };

    // Main rendering engine
    class RenderingEngine {
    public:
        static std::unique_ptr<RenderingEngine> create();

        virtual ~RenderingEngine() = default;

        // Lifecycle
        virtual Result<void> initialize() = 0;
        virtual void shutdown() = 0;
        virtual bool isInitialized() const = 0;

        virtual Result<GaussianGpuFrameResult> renderGaussiansGpuFrame(
            const lfs::core::SplatData& splat_data,
            const ViewportRenderRequest& request) = 0;

        virtual Result<GaussianImageResult> renderGaussiansImage(
            const lfs::core::SplatData& splat_data,
            const ViewportRenderRequest& request) = 0;

        virtual Result<DualGaussianImageResult> renderGaussiansImagePair(
            const lfs::core::SplatData& splat_data,
            const std::array<ViewportRenderRequest, 2>& requests) = 0;

        virtual Result<std::optional<int>> queryHoveredGaussianId(
            const lfs::core::SplatData& splat_data,
            const HoveredGaussianQueryRequest& request) = 0;

        virtual Result<std::shared_ptr<lfs::core::Tensor>> renderGaussianScreenPositions(
            const lfs::core::SplatData& splat_data,
            const ScreenPositionRenderRequest& request) = 0;

        virtual Result<GpuFrame> renderPointCloudGpuFrame(
            const lfs::core::SplatData& splat_data,
            const PointCloudRenderRequest& request) = 0;

        virtual Result<PointCloudImageResult> renderPointCloudImage(
            const lfs::core::SplatData& splat_data,
            const PointCloudRenderRequest& request) = 0;

        virtual Result<GpuFrame> renderPointCloudGpuFrame(
            const lfs::core::PointCloud& point_cloud,
            const PointCloudRenderRequest& request) = 0;

        virtual Result<SplitViewFrameResult> renderSplitViewGpuFrame(
            const SplitViewRequest& request) = 0;

        virtual Result<void> renderMesh(
            const lfs::core::MeshData& mesh,
            const ViewportData& viewport,
            const glm::mat4& model_transform = glm::mat4(1.0f),
            const MeshRenderOptions& options = {},
            bool use_fbo = false) = 0;

        virtual unsigned int getMeshColorTexture() const = 0;
        virtual unsigned int getMeshDepthTexture() const = 0;
        virtual unsigned int getMeshFramebuffer() const = 0;
        virtual bool hasMeshRender() const = 0;
        virtual void resetMeshFrameState() = 0;

        virtual Result<GpuFrame> materializeGpuFrame(
            const std::shared_ptr<lfs::core::Tensor>& image,
            const FrameMetadata& metadata,
            const glm::ivec2& viewport_size) = 0;

        virtual Result<std::shared_ptr<lfs::core::Tensor>> readbackGpuFrameColor(
            const GpuFrame& frame) = 0;

        virtual Result<void> compositeMeshAndGpuFrame(
            const GpuFrame& splat_frame,
            const glm::ivec2& viewport_size) = 0;

        virtual Result<void> presentMeshOnly() = 0;

        virtual Result<void> presentGpuFrame(
            const GpuFrame& frame,
            const glm::ivec2& viewport_pos,
            const glm::ivec2& viewport_size) = 0;

        virtual Result<void> renderScreenSpaceVignette(
            const glm::ivec2& viewport_size,
            ScreenSpaceVignette vignette) = 0;

        // Overlay rendering - now returns Result for consistency
        virtual Result<void> renderGrid(
            const ViewportData& viewport,
            GridPlane plane = GridPlane::XZ,
            float opacity = 0.5f) = 0;

        virtual Result<void> renderBoundingBox(
            const BoundingBox& box,
            const ViewportData& viewport,
            const glm::vec3& color = glm::vec3(1.0f, 1.0f, 0.0f),
            float line_width = 2.0f) = 0;

        virtual Result<void> renderEllipsoid(
            const Ellipsoid& ellipsoid,
            const ViewportData& viewport,
            const glm::vec3& color = glm::vec3(0.3f, 0.8f, 1.0f),
            float line_width = 2.0f) = 0;

        virtual Result<void> renderCoordinateAxes(
            const ViewportData& viewport,
            float size = 2.0f,
            const std::array<bool, 3>& visible = {true, true, true},
            bool equirectangular = false) = 0;

        virtual Result<void> renderPivot(
            const ViewportData& viewport,
            const glm::vec3& pivot_position,
            float size = 50.0f,
            float opacity = 1.0f) = 0;

        // Viewport gizmo rendering
        virtual Result<void> renderViewportGizmo(
            const glm::mat3& camera_rotation,
            const glm::vec2& viewport_pos,
            const glm::vec2& viewport_size) = 0;

        // Hit-test viewport gizmo (returns 0-2=+X/Y/Z, 3-5=-X/Y/Z, or -1 for none)
        virtual int hitTestViewportGizmo(
            const glm::vec2& click_pos,
            const glm::vec2& viewport_pos,
            const glm::vec2& viewport_size) const = 0;

        // Set hovered axis for highlighting (0-2=+X/Y/Z, 3-5=-X/Y/Z, -1 for none)
        virtual void setViewportGizmoHover(int axis) = 0;

        // Get camera rotation matrix to view along axis
        [[nodiscard]] static glm::mat3 getAxisViewRotation(int axis, bool negative = false);

        // Camera frustum rendering with focus/emphasis state
        virtual Result<void> renderCameraFrustums(
            const std::vector<std::shared_ptr<const lfs::core::Camera>>& cameras,
            const CameraFrustumRenderRequest& request) = 0;

        // Camera frustum picking
        virtual Result<int> pickCameraFrustum(
            const std::vector<std::shared_ptr<const lfs::core::Camera>>& cameras,
            const CameraFrustumPickRequest& request) = 0;

        virtual void clearFrustumCache() = 0;
        virtual void setFrustumImageLoader(std::shared_ptr<lfs::io::PipelinedImageLoader> loader) = 0;
    };

} // namespace lfs::rendering

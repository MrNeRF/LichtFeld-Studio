/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include "core/tensor.hpp"
#include "frame_contract.hpp"
#include "render_constants.hpp"
#include <array>
#include <cstdint>
#include <expected>
#include <filesystem>
#include <glm/glm.hpp>
#include <memory>
#include <optional>
#include <string>
#include <vector>

namespace lfs::core {
    class SplatData;
    struct PointCloud;
    struct MeshData;
    class Camera;
    class Tensor;
} // namespace lfs::core

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
        GaussianRasterBackend raster_backend = GaussianRasterBackend::FastGs;
        bool gut = false;
        bool equirectangular = false;
        GaussianSceneState scene;
        GaussianFilterState filters;
        GaussianOverlayState overlay;
        bool transparent_background = false;
    };

    struct HoveredGaussianQueryRequest {
        FrameView frame_view;
        float scaling_modifier = 1.0f;
        bool mip_filter = false;
        int sh_degree = 3;
        GaussianRasterBackend raster_backend = GaussianRasterBackend::FastGs;
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
        std::vector<bool> node_visibility_mask;
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
        bool transparent_background = false;
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
        bool depth_is_ndc = false; // True if depth is already normalized device depth (0-1).
        glm::vec2 depth_texcoord_scale{1.0f, 1.0f};
        // Presentation orientation for the screen quad.
        bool flip_y = false;
        float near_plane = DEFAULT_NEAR_PLANE;
        float far_plane = DEFAULT_FAR_PLANE;
        bool orthographic = false;
        bool color_has_alpha = false;

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
        GaussianRasterBackend raster_backend = GaussianRasterBackend::FastGs;
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
        uint64_t image_handle = 0;
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

    // Render modes
    enum class RenderMode {
        RGB = 0,
        D = 1,
        ED = 2,
        RGB_D = 3,
        RGB_ED = 4
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
        bool transparent_background = false;
    };

    struct EnvironmentRenderOptions {
        bool enabled = false;
        std::filesystem::path map_path;
        float exposure = 0.0f;
        float rotation_degrees = 0.0f;
        bool equirectangular = false;
    };

    struct MeshFrameItem {
        const lfs::core::MeshData* mesh = nullptr;
        glm::mat4 transform{1.0f};
        MeshRenderOptions options{};
    };

    struct VideoCompositeFrameRequest {
        ViewportData viewport;
        FrameView frame_view;
        glm::vec3 background_color{0.0f};
        EnvironmentRenderOptions environment;
        std::vector<MeshFrameItem> meshes;
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
        static std::unique_ptr<RenderingEngine> createRasterOnly();

        virtual ~RenderingEngine() = default;

        // Lifecycle
        virtual Result<void> initialize() = 0;
        virtual Result<void> initializeRasterOnly() = 0;
        virtual void shutdown() = 0;
        virtual bool isInitialized() const = 0;
        virtual bool isRasterInitialized() const = 0;

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

        virtual Result<PointCloudImageResult> renderPointCloudImage(
            const lfs::core::PointCloud& point_cloud,
            const PointCloudRenderRequest& request) = 0;

        virtual Result<GpuFrame> renderPointCloudGpuFrame(
            const lfs::core::PointCloud& point_cloud,
            const PointCloudRenderRequest& request) = 0;

        virtual Result<GpuFrame> materializeGpuFrame(
            const std::shared_ptr<lfs::core::Tensor>& image,
            const FrameMetadata& metadata,
            const glm::ivec2& viewport_size) = 0;

        virtual Result<std::shared_ptr<lfs::core::Tensor>> readbackGpuFrameColor(
            const GpuFrame& frame) = 0;

        virtual Result<lfs::core::Tensor> renderVideoCompositeFrame(
            const std::optional<GpuFrame>& primary_frame,
            const VideoCompositeFrameRequest& request) = 0;

        // Camera frustum picking
        virtual Result<int> pickCameraFrustum(
            const std::vector<std::shared_ptr<const lfs::core::Camera>>& cameras,
            const CameraFrustumPickRequest& request) = 0;

        virtual class ScreenOverlayRenderer* getScreenOverlayRenderer() = 0;
    };

} // namespace lfs::rendering

/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include "core/tensor.hpp"
#include "geometry/euclidean_transform.hpp"
#include "render_constants.hpp"
#include <array>
#include <expected>
#include <glm/glm.hpp>
#include <memory>
#include <optional>
#include <string>
#include <vector>

namespace lfs::core {
    class SplatData;
    struct PointCloud;
    class Camera;
    class Tensor;
} // namespace lfs::core

namespace lfs::rendering {

    // Import Tensor into this namespace for convenience
    using lfs::core::Tensor;

    // Error handling with std::expected (C++23)
    template <typename T>
    using Result = std::expected<T, std::string>;

    enum class SelectionMode {
        Centers,
        Rectangle,
        Polygon,
        Lasso,
        Rings
    };

    // Public types
    struct ViewportData {
        glm::mat3 rotation;
        glm::vec3 translation;
        glm::ivec2 size;
        float fov = DEFAULT_FOV;
        bool orthographic = false;
        float ortho_scale = DEFAULT_ORTHO_SCALE;

        [[nodiscard]] glm::mat4 getProjectionMatrix(const float near_plane = DEFAULT_NEAR_PLANE,
                                                    const float far_plane = DEFAULT_FAR_PLANE) const {
            return createProjectionMatrix(size, fov, orthographic, ortho_scale, near_plane, far_plane);
        }
    };

    struct BoundingBox {
        glm::vec3 min;
        glm::vec3 max;
        glm::mat4 transform{1.0f};
    };

    struct RenderRequest {
        ViewportData viewport;
        float scaling_modifier = 1.0f;
        bool antialiasing = false;
        int sh_degree = 3;
        glm::vec3 background_color{0.0f, 0.0f, 0.0f};
        std::optional<BoundingBox> crop_box;
        bool point_cloud_mode = false;
        float voxel_size = 0.01f;
        bool gut = false;
        bool show_rings = false;
        float ring_width = 0.002f;
        bool show_center_markers = false;
        // Per-node transforms: array of 4x4 matrices and per-Gaussian indices
        std::vector<glm::mat4> model_transforms;              // Array of transforms, one per node
        std::shared_ptr<lfs::core::Tensor> transform_indices; // Per-Gaussian index [N], nullable
        // Selection mask for highlighting selected Gaussians
        std::shared_ptr<lfs::core::Tensor> selection_mask; // Per-Gaussian uint8 [N], nullable (1 = selected, 0 = not)
        // Request screen positions output for brush tool
        bool output_screen_positions = false;
        // Brush selection (computed in preprocess for coordinate consistency)
        bool brush_active = false;  // Whether brush selection is active this frame
        float brush_x = 0.0f;       // Brush center X in screen coords
        float brush_y = 0.0f;       // Brush center Y in screen coords
        float brush_radius = 0.0f;  // Brush radius in pixels
        bool brush_add_mode = true; // true = add to selection, false = remove from selection
        lfs::core::Tensor* brush_selection_tensor = nullptr;
        bool brush_saturation_mode = false;
        float brush_saturation_amount = 0.0f;
        bool selection_mode_rings = false;
        bool crop_inverse = false;
        bool crop_desaturate = false;
        // Depth filter for selection tool (separate from crop box, always desaturates outside)
        std::optional<BoundingBox> depth_filter;
        // Per-node selection mask: true = selected. Empty = no selection effects.
        std::vector<bool> selected_node_mask;
        bool desaturate_unselected = false;
        float selection_flash_intensity = 0.0f;
        unsigned long long* hovered_depth_id = nullptr;
        int highlight_gaussian_id = -1;
        float far_plane = DEFAULT_FAR_PLANE;
        bool orthographic = false;  // Use orthographic projection instead of perspective
        float ortho_scale = 100.0f; // Pixels per world unit for orthographic projection
    };

    struct RenderResult {
        std::shared_ptr<lfs::core::Tensor> image;
        std::shared_ptr<lfs::core::Tensor> depth;
        std::shared_ptr<lfs::core::Tensor> screen_positions; // Optional: screen positions [N, 2] for brush tool
        bool valid = false;
        // Depth conversion parameters (needed for proper depth buffer writing)
        bool depth_is_ndc = false;       // True if depth is already NDC (0-1), e.g., from OpenGL
        unsigned int external_depth_texture = 0;  // If set, use this OpenGL texture directly (zero-copy)
        float near_plane = DEFAULT_NEAR_PLANE;
        float far_plane = DEFAULT_FAR_PLANE;
        bool orthographic = false;
    };

    // Split view support
    enum class PanelContentType {
        Model3D,     // Regular 3D model rendering
        Image2D,     // GT image display
        CachedRender // Previously rendered frame
    };

    struct SplitViewPanel {
        PanelContentType content_type = PanelContentType::Model3D;

        // For Model3D
        const lfs::core::SplatData* model = nullptr;

        // For Image2D or CachedRender
        unsigned int texture_id = 0;

        // Common fields
        std::string label;
        float start_position; // 0.0 to 1.0
        float end_position;   // 0.0 to 1.0
    };

    struct SplitViewRequest {
        std::vector<SplitViewPanel> panels;
        ViewportData viewport;

        // Common render settings
        float scaling_modifier = 1.0f;
        bool antialiasing = false;
        int sh_degree = 3;
        glm::vec3 background_color{0.0f, 0.0f, 0.0f};
        std::optional<BoundingBox> crop_box;
        bool point_cloud_mode = false;
        float voxel_size = 0.01f;
        bool gut = false;
        bool show_rings = false;
        float ring_width = 0.002f;

        // UI settings
        bool show_dividers = true;
        glm::vec4 divider_color{1.0f, 0.85f, 0.0f, 1.0f};
        bool show_labels = true;
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

    // Translation Gizmo types
    enum class GizmoElement {
        None,
        XAxis,
        YAxis,
        ZAxis,
        XYPlane,
        XZPlane,
        YZPlane
    };

    // Abstract interface for gizmo interaction
    class GizmoInteraction {
    public:
        virtual ~GizmoInteraction() = default;

        virtual GizmoElement pick(const glm::vec2& mouse_pos, const glm::mat4& view,
                                  const glm::mat4& projection, const glm::vec3& position) = 0;

        virtual glm::vec3 startDrag(GizmoElement element, const glm::vec2& mouse_pos,
                                    const glm::mat4& view, const glm::mat4& projection,
                                    const glm::vec3& position) = 0;

        virtual glm::vec3 updateDrag(const glm::vec2& mouse_pos, const glm::mat4& view,
                                     const glm::mat4& projection) = 0;

        virtual void endDrag() = 0;
        virtual bool isDragging() const = 0;

        virtual void setHovered(GizmoElement element) = 0;
        virtual GizmoElement getHovered() const = 0;
    };

    // Rendering pipeline types (for compatibility)
    struct RenderingPipelineRequest {
        glm::mat3 view_rotation;
        glm::vec3 view_translation;
        glm::ivec2 viewport_size;
        float fov = 60.0f;
        float scaling_modifier = 1.0f;
        bool antialiasing = false;
        RenderMode render_mode = RenderMode::RGB;
        const void* crop_box = nullptr; // Actually lfs::geometry::BoundingBox*
        glm::vec3 background_color = glm::vec3(0.0f, 0.0f, 0.0f);
        bool point_cloud_mode = false;
        float voxel_size = 0.01f;
        bool gut = false;
        bool show_rings = false;
        float ring_width = 0.002f;
    };

    struct RenderingPipelineResult {
        Tensor image;
        Tensor depth;
        bool valid = false;
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

    // Main rendering engine
    class RenderingEngine {
    public:
        static std::unique_ptr<RenderingEngine> create();

        virtual ~RenderingEngine() = default;

        // Lifecycle
        virtual Result<void> initialize() = 0;
        virtual void shutdown() = 0;
        virtual bool isInitialized() const = 0;

        // Core rendering with error handling
        virtual Result<RenderResult> renderGaussians(
            const lfs::core::SplatData& splat_data,
            const RenderRequest& request) = 0;

        // Point cloud rendering (for pre-training visualization)
        virtual Result<RenderResult> renderPointCloud(
            const lfs::core::PointCloud& point_cloud,
            const RenderRequest& request) = 0;

        // Split view rendering
        virtual Result<RenderResult> renderSplitView(
            const SplitViewRequest& request) = 0;

        // Present to screen
        virtual Result<void> presentToScreen(
            const RenderResult& result,
            const glm::ivec2& viewport_pos,
            const glm::ivec2& viewport_size) = 0;

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

        virtual Result<void> renderCoordinateAxes(
            const ViewportData& viewport,
            float size = 2.0f,
            const std::array<bool, 3>& visible = {true, true, true}) = 0;

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

        // Translation gizmo rendering
        virtual Result<void> renderTranslationGizmo(
            const glm::vec3& position,
            const ViewportData& viewport,
            float scale = 1.0f) = 0;

        // Camera frustum rendering
        virtual Result<void> renderCameraFrustums(
            const std::vector<std::shared_ptr<const lfs::core::Camera>>& cameras,
            const ViewportData& viewport,
            float scale = 0.1f,
            const glm::vec3& train_color = glm::vec3(0.0f, 1.0f, 0.0f),
            const glm::vec3& eval_color = glm::vec3(1.0f, 0.0f, 0.0f),
            const glm::mat4& scene_transform = glm::mat4(1.0f)) = 0;

        // Camera frustum rendering with highlighting
        virtual Result<void> renderCameraFrustumsWithHighlight(
            const std::vector<std::shared_ptr<const lfs::core::Camera>>& cameras,
            const ViewportData& viewport,
            float scale = 0.1f,
            const glm::vec3& train_color = glm::vec3(0.0f, 1.0f, 0.0f),
            const glm::vec3& eval_color = glm::vec3(1.0f, 0.0f, 0.0f),
            int highlight_index = -1,
            const glm::mat4& scene_transform = glm::mat4(1.0f)) = 0;

        // Camera frustum picking
        virtual Result<int> pickCameraFrustum(
            const std::vector<std::shared_ptr<const lfs::core::Camera>>& cameras,
            const glm::vec2& mouse_pos,
            const glm::vec2& viewport_pos,
            const glm::vec2& viewport_size,
            const ViewportData& viewport,
            float scale = 0.1f,
            const glm::mat4& scene_transform = glm::mat4(1.0f)) = 0;

        virtual void clearFrustumCache() = 0;

        // Get gizmo interaction interface
        virtual std::shared_ptr<GizmoInteraction> getGizmoInteraction() = 0;

        // Pipeline rendering (for visualizer compatibility)
        virtual RenderingPipelineResult renderWithPipeline(
            const lfs::core::SplatData& model,
            const RenderingPipelineRequest& request) = 0;

        // Factory methods - now return Result
        virtual Result<std::shared_ptr<IBoundingBox>> createBoundingBox() = 0;
        virtual Result<std::shared_ptr<ICoordinateAxes>> createCoordinateAxes() = 0;
    };

} // namespace lfs::rendering
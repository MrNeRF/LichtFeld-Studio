/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include "geometry/euclidean_transform.hpp"
#include <array>
#include <expected>
#include <glm/glm.hpp>
#include <memory>
#include <optional>
#include <string>
#include <torch/types.h>
#include <vector>

namespace gs {
    class SplatData;
    class Camera;
} // namespace gs

namespace gs::rendering {

    // Error handling with std::expected (C++23)
    template <typename T>
    using Result = std::expected<T, std::string>;

    // Public types
    struct ViewportData {
        glm::mat3 rotation;
        glm::vec3 translation;
        glm::ivec2 size;
        float fov = 60.0f;
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
        glm::vec3 background_color{0.0f, 0.0f, 0.0f};
        std::optional<BoundingBox> crop_box;
        bool point_cloud_mode = false;
        float voxel_size = 0.01f;
        bool gut = false;
        bool equirectangular = false;
        int sh_degree = 0;
    };

    struct RenderResult {
        std::shared_ptr<torch::Tensor> image;
        std::shared_ptr<torch::Tensor> depth;
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
        const SplatData* model = nullptr;

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
        glm::vec3 background_color{0.0f, 0.0f, 0.0f};
        std::optional<BoundingBox> crop_box;
        bool point_cloud_mode = false;
        float voxel_size = 0.01f;
        bool gut = false;

        // UI settings
        bool show_dividers = true;
        glm::vec4 divider_color{1.0f, 0.85f, 0.0f, 1.0f};
        bool show_labels = true;
        int sh_degree = 0;
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
        const void* crop_box = nullptr; // Actually geometry::BoundingBox*
        glm::vec3 background_color = glm::vec3(0.0f, 0.0f, 0.0f);
        bool point_cloud_mode = false;
        float voxel_size = 0.01f;
        bool gut = false;
        int sh_degree = 0;
    };

    struct RenderingPipelineResult {
        torch::Tensor image;
        torch::Tensor depth;
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

        virtual void setworld2BBox(const geometry::EuclideanTransform& transform) = 0;
        virtual geometry::EuclideanTransform getworld2BBox() const = 0;

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
            const SplatData& splat_data,
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

        // Viewport gizmo rendering
        virtual Result<void> renderViewportGizmo(
            const glm::mat3& camera_rotation,
            const glm::vec2& viewport_pos,
            const glm::vec2& viewport_size) = 0;

        // Translation gizmo rendering
        virtual Result<void> renderTranslationGizmo(
            const glm::vec3& position,
            const ViewportData& viewport,
            float scale = 1.0f) = 0;

        // Camera frustum rendering
        virtual Result<void> renderCameraFrustums(
            const std::vector<std::shared_ptr<const Camera>>& cameras,
            const ViewportData& viewport,
            float scale = 0.1f,
            const glm::vec3& train_color = glm::vec3(0.0f, 1.0f, 0.0f),
            const glm::vec3& eval_color = glm::vec3(1.0f, 0.0f, 0.0f),
            const glm::mat4& world_transform = glm::mat4(1.0f)) = 0;

        // Camera frustum rendering with highlighting
        virtual Result<void> renderCameraFrustumsWithHighlight(
            const std::vector<std::shared_ptr<const Camera>>& cameras,
            const ViewportData& viewport,
            float scale = 0.1f,
            const glm::vec3& train_color = glm::vec3(0.0f, 1.0f, 0.0f),
            const glm::vec3& eval_color = glm::vec3(1.0f, 0.0f, 0.0f),
            int highlight_index = -1,
            const glm::mat4& world_transform = glm::mat4(1.0f)) = 0;

        // Camera frustum picking
        virtual Result<int> pickCameraFrustum(
            const std::vector<std::shared_ptr<const Camera>>& cameras,
            const glm::vec2& mouse_pos,
            const glm::vec2& viewport_pos,
            const glm::vec2& viewport_size,
            const ViewportData& viewport,
            float scale = 0.1f,
            const glm::mat4& world_transform = glm::mat4(1.0f)) = 0;

        // Get gizmo interaction interface
        virtual std::shared_ptr<GizmoInteraction> getGizmoInteraction() = 0;

        // Pipeline rendering (for visualizer compatibility)
        virtual RenderingPipelineResult renderWithPipeline(
            const SplatData& model,
            const RenderingPipelineRequest& request) = 0;

        // Factory methods - now return Result
        virtual Result<std::shared_ptr<IBoundingBox>> createBoundingBox() = 0;
        virtual Result<std::shared_ptr<ICoordinateAxes>> createCoordinateAxes() = 0;
    };

} // namespace gs::rendering
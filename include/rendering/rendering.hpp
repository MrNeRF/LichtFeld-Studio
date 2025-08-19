#pragma once

#include "geometry/euclidean_transform.hpp"
#include <array>
#include <expected>
#include <glm/glm.hpp>
#include <memory>
#include <optional>
#include <string>
#include <torch/types.h>

namespace gs {
    class SplatData;
}

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
    };

    struct RenderResult {
        std::shared_ptr<torch::Tensor> image;
        std::shared_ptr<torch::Tensor> depth;
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

        // Add missing methods
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

        // Pipeline rendering (for visualizer compatibility)
        virtual RenderingPipelineResult renderWithPipeline(
            const SplatData& model,
            const RenderingPipelineRequest& request) = 0;

        // Factory methods - now return Result
        virtual Result<std::shared_ptr<IBoundingBox>> createBoundingBox() = 0;
        virtual Result<std::shared_ptr<ICoordinateAxes>> createCoordinateAxes() = 0;
    };

} // namespace gs::rendering
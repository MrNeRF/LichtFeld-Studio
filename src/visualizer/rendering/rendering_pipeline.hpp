#pragma once

#include "core/camera.hpp"
#include "core/rasterizer.hpp"
#include "core/splat_data.hpp"
#include "rendering/point_cloud_renderer.hpp"
#include "rendering/render_bounding_box.hpp"
#include "rendering/renderer.hpp"
#include <glm/glm.hpp>
#include <torch/torch.h>

namespace gs {

    class RenderingPipeline {
    public:
        struct RenderRequest {
            glm::mat3 view_rotation;
            glm::vec3 view_translation;
            glm::ivec2 viewport_size;
            float fov = 60.0f;
            float scaling_modifier = 1.0f;
            bool antialiasing = false;
            RenderMode render_mode = RenderMode::RGB;
            const geometry::BoundingBox* crop_box = nullptr; // crop box
            glm::vec3 background_color = glm::vec3(0.0f, 0.0f, 0.0f);
            bool point_cloud_mode = false;
            float voxel_size = 0.01f;
        };

        struct RenderResult {
            torch::Tensor image = torch::Tensor();
            torch::Tensor depth = torch::Tensor();
            bool valid = false;

            RenderResult(bool v = false) : image(),
                                           depth(),
                                           valid(v) {}
        };

        RenderingPipeline();

        // Main render function
        RenderResult render(const SplatData& model, const RenderRequest& request);

        // Static upload function
        static void uploadToScreen(const RenderResult& result,
                                   ScreenQuadRenderer& renderer,
                                   const glm::ivec2& viewport_size);

    private:
        Camera createCamera(const RenderRequest& request);
        glm::vec2 computeFov(float fov_degrees, int width, int height);
        RenderResult renderPointCloud(const SplatData& model, const RenderRequest& request);

        torch::Tensor background_;
        std::unique_ptr<PointCloudRenderer> point_cloud_renderer_;
    };

} // namespace gs

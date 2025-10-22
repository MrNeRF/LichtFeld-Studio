/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include "core/camera.hpp"
#include "core/splat_data.hpp"
#include "geometry/bounding_box.hpp"
#include "point_cloud_renderer.hpp"
#include "rendering/rendering.hpp"
#include "screen_renderer.hpp"
#include <glm/glm.hpp>
#include <torch/torch.h>

namespace gs::rendering {

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
            const geometry::BoundingBox* crop_box = nullptr;
            glm::vec3 background_color = glm::vec3(0.0f, 0.0f, 0.0f);
            bool point_cloud_mode = false;
            float voxel_size = 0.01f;
            bool gut = false;
            bool equirectangular = false;
            int sh_degree = 0;
        };

        struct RenderResult {
            torch::Tensor image;
            torch::Tensor depth;
            bool valid = false;
        };

        RenderingPipeline();

        // Main render function - now returns Result
        Result<RenderResult> render(const SplatData& model, const RenderRequest& request);

        // Static upload function - now returns Result
        static Result<void> uploadToScreen(const RenderResult& result,
                                           ScreenQuadRenderer& renderer,
                                           const glm::ivec2& viewport_size);

    private:
        Result<Camera> createCamera(const RenderRequest& request);
        glm::vec2 computeFov(float fov_degrees, int width, int height);
        Result<RenderResult> renderPointCloud(const SplatData& model, const RenderRequest& request);

        torch::Tensor background_;
        std::unique_ptr<PointCloudRenderer> point_cloud_renderer_;
    };

} // namespace gs::rendering
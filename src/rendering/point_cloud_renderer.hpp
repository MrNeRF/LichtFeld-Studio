/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include "config.h"
#include "core/point_cloud.hpp"
#include "core/splat_data.hpp"
#include "core/tensor.hpp"
#include "gl_resources.hpp"
#include "shader_manager.hpp"
#include <glm/glm.hpp>
#include <memory>
#include <span>
#include <vector>

#ifdef CUDA_GL_INTEROP_ENABLED
#include "cuda_gl_interop.hpp"
#include <optional>
#endif

namespace lfs::rendering {

    class PointCloudRenderer {
    public:
        PointCloudRenderer() = default;
        ~PointCloudRenderer() = default;

        Result<void> initialize();

        Result<void> render(const lfs::core::SplatData& splat_data,
                            const glm::mat4& view,
                            const glm::mat4& projection,
                            float voxel_size,
                            const glm::vec3& background_color,
                            const std::vector<glm::mat4>& model_transforms = {},
                            const std::shared_ptr<lfs::core::Tensor>& transform_indices = nullptr,
                            bool equirectangular = false);

        Result<void> render(const lfs::core::PointCloud& point_cloud,
                            const glm::mat4& view,
                            const glm::mat4& projection,
                            float voxel_size,
                            const glm::vec3& background_color,
                            const std::vector<glm::mat4>& model_transforms = {},
                            const std::shared_ptr<lfs::core::Tensor>& transform_indices = nullptr,
                            bool equirectangular = false);

        // Check if initialized
        bool isInitialized() const { return initialized_; }

    private:
        Result<void> createCubeGeometry();
        static Tensor extractRGBFromSH(const Tensor& shs);

        // Core rendering implementation (shared by both overloads)
        Result<void> renderInternal(const Tensor& positions,
                                    const Tensor& colors,
                                    const glm::mat4& view,
                                    const glm::mat4& projection,
                                    float voxel_size,
                                    const glm::vec3& background_color,
                                    const std::vector<glm::mat4>& model_transforms,
                                    const std::shared_ptr<lfs::core::Tensor>& transform_indices,
                                    bool equirectangular);

        // OpenGL resources using RAII
        VAO cube_vao_;
        VBO cube_vbo_;
        EBO cube_ebo_;
        VBO instance_vbo_; // For positions and colors

        // Framebuffer resources using RAII
        FBO fbo_;
        Texture color_texture_;
        Texture depth_texture_;
        int fbo_width_ = 0;
        int fbo_height_ = 0;

        // Shaders
        ManagedShader shader_;

        // State
        bool initialized_ = false;
        size_t current_point_count_ = 0;

        // Cached buffer to avoid per-frame allocation
        Tensor interleaved_cache_;

#ifdef CUDA_GL_INTEROP_ENABLED
        std::optional<CudaGLInteropBuffer> interop_buffer_;
        size_t interop_buffer_size_ = 0;
        bool use_interop_ = true;
#endif

        // Cube vertices
        static constexpr float cube_vertices_[] = {
            // Front face
            -0.5f, -0.5f, 0.5f,
            0.5f, -0.5f, 0.5f,
            0.5f, 0.5f, 0.5f,
            -0.5f, 0.5f, 0.5f,
            // Back face
            -0.5f, -0.5f, -0.5f,
            0.5f, -0.5f, -0.5f,
            0.5f, 0.5f, -0.5f,
            -0.5f, 0.5f, -0.5f};

        // Triangle indices for solid cube (6 faces, 12 triangles, 36 indices)
        static constexpr unsigned int cube_indices_[] = {
            // Front face
            0, 1, 2, 2, 3, 0,
            // Back face
            4, 6, 5, 6, 4, 7,
            // Left face
            0, 3, 7, 7, 4, 0,
            // Right face
            1, 5, 6, 6, 2, 1,
            // Top face
            3, 2, 6, 6, 7, 3,
            // Bottom face
            0, 4, 5, 5, 1, 0};
    };

} // namespace lfs::rendering
